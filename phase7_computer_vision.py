"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM
 PHASE 7 — Computer Vision / Live Movement Analysis
=============================================================================
 Pipeline:
   Webcam → OpenCV → MediaPipe Pose → Body Landmarks
        → Joint Angles → Movement Features
        → Feature Mapping → Trained ML Model → Movement Risk

 Feature schema is read from the Phase 4 preprocessor object so that
 the mapping layer always stays in sync with the trained model.
=============================================================================
"""

import os
import sys
import time
import math
import warnings
import collections
import traceback

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS & CONFIG
# =============================================================================
BASE_DIR   = r'c:\studies\sports monitoring system'
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MOVEMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'movement_model.pkl')
MOVEMENT_META_PATH  = os.path.join(MODELS_DIR, 'movement_preprocessor.pkl')

# ── Risk thresholds (application-level — NOT clinically validated) ────────────
LOW_THRESHOLD      = 0.30   # probability < 30 % → LOW
MODERATE_THRESHOLD = 0.60   # 30 % ≤ probability < 60 % → MODERATE
                             # probability ≥ 60 % → HIGH

# ── Temporal buffer ───────────────────────────────────────────────────────────
BUFFER_SIZE = 30            # frames kept for velocity / ROM / stability

# ── Display ───────────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLUE   = (255, 160, 0)
COLOR_DARK   = (30, 30, 30)

# ── Estimated body-weight for GRF proxy (configurable) ────────────────────────
ATHLETE_BODY_WEIGHT_KG = 70.0   # used only when no force plate is available


# =============================================================================
# LOAD TRAINED MODEL & PREPROCESSOR
# =============================================================================
def load_movement_model():
    """Load the Phase 4 trained pipeline and preprocessor metadata."""
    if not os.path.exists(MOVEMENT_MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MOVEMENT_MODEL_PATH}')
    if not os.path.exists(MOVEMENT_META_PATH):
        raise FileNotFoundError(f'Preprocessor not found: {MOVEMENT_META_PATH}')

    model = joblib.load(MOVEMENT_MODEL_PATH)
    meta  = joblib.load(MOVEMENT_META_PATH)
    print(f'  [Phase7] Model loaded    : {meta["model_name"]}')
    print(f'  [Phase7] Feature columns : {meta["feature_columns"]}')
    return model, meta


# =============================================================================
# JOINT ANGLE CALCULATION
# =============================================================================
def calculate_angle(point_a: np.ndarray,
                    point_b: np.ndarray,
                    point_c: np.ndarray) -> float:
    """
    Calculate the angle (degrees) at joint B formed by vectors B→A and B→C.

    Parameters
    ----------
    point_a, point_b, point_c : array-like of shape (2,) or (3,)
        2-D or 3-D landmark coordinates. Only x/y are used.

    Returns
    -------
    float : angle in degrees [0, 180]
    """
    a = np.array(point_a[:2], dtype=float)
    b = np.array(point_b[:2], dtype=float)
    c = np.array(point_c[:2], dtype=float)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division with zero-guard."""
    return a / b if abs(b) > 1e-9 else default


# =============================================================================
# LANDMARK EXTRACTOR
# =============================================================================
class LandmarkExtractor:
    """
    Wraps MediaPipe Pose and extracts normalised (x, y) coordinates
    for the joints we need.  All coordinates are in [0, 1] relative
    to the frame dimensions.
    """

    # MediaPipe landmark indices
    LM = {
        'left_shoulder':  11,
        'right_shoulder': 12,
        'left_hip':       23,
        'right_hip':      24,
        'left_knee':      25,
        'right_knee':     26,
        'left_ankle':     27,
        'right_ankle':    28,
        'left_heel':      29,
        'right_heel':     30,
        'left_foot':      31,
        'right_foot':     32,
    }

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence:  float = 0.5):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def process(self, bgr_frame: np.ndarray):
        """
        Run pose estimation on one BGR frame.

        Returns
        -------
        results : MediaPipe results object (or None on failure)
        landmarks : dict {name: np.array([x, y])} in normalised coords
        annotated : BGR frame with skeleton drawn
        """
        rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True
        annotated = bgr_frame.copy()

        landmarks = {}
        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2),
            )
            lm_list = results.pose_landmarks.landmark
            for name, idx in self.LM.items():
                lm = lm_list[idx]
                landmarks[name] = np.array([lm.x, lm.y])

        return results, landmarks, annotated

    def close(self):
        self.pose.close()


# =============================================================================
# TEMPORAL BUFFER — stores per-frame measurements for velocity / ROM / stability
# =============================================================================
class TemporalBuffer:
    """
    Rolling window of fixed length.
    Stores dicts of {feature_name: value} with timestamps.
    """

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self.angles    = collections.deque(maxlen=maxlen)   # joint angles
        self.positions = collections.deque(maxlen=maxlen)   # hip centre (x,y)
        self.timestamps= collections.deque(maxlen=maxlen)   # wall-clock seconds

    def push(self, angles: dict, hip_centre: np.ndarray, ts: float):
        self.angles.append(angles)
        self.positions.append(hip_centre.copy())
        self.timestamps.append(ts)

    def __len__(self):
        return len(self.timestamps)

    # ── Velocity ─────────────────────────────────────────────────────────────
    def angular_velocity(self, key: str) -> float:
        """
        Mean absolute angular velocity (°/s) for joint `key`
        over the last two frames.
        """
        if len(self) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]
        da = abs(self.angles[-1].get(key, 0.0) -
                 self.angles[-2].get(key, 0.0))
        return safe_div(da, dt)

    def mean_angular_velocity(self) -> float:
        """Mean velocity across all tracked joints (last 2 frames)."""
        if len(self) < 2:
            return 0.0
        keys = list(self.angles[-1].keys())
        vels = [self.angular_velocity(k) for k in keys]
        return float(np.mean(vels)) if vels else 0.0

    # ── Acceleration ──────────────────────────────────────────────────────────
    def linear_acceleration(self) -> float:
        """
        Magnitude of hip-centre acceleration (normalised units / s²).
        acceleration = Δvelocity / Δtime  using 3 consecutive frames.
        """
        if len(self) < 3:
            return 0.0
        p0, p1, p2 = (np.array(self.positions[-3]),
                      np.array(self.positions[-2]),
                      np.array(self.positions[-1]))
        t0, t1, t2 = (self.timestamps[-3],
                      self.timestamps[-2],
                      self.timestamps[-1])
        dt1 = max(t1 - t0, 1e-6)
        dt2 = max(t2 - t1, 1e-6)
        v1  = (p1 - p0) / dt1
        v2  = (p2 - p1) / dt2
        return float(np.linalg.norm((v2 - v1) / ((dt1 + dt2) / 2)))

    # ── Range of Motion ───────────────────────────────────────────────────────
    def range_of_motion(self, key: str) -> float:
        """ROM = max − min of a joint angle over the full buffer."""
        if len(self) < 2:
            return 0.0
        vals = [a.get(key, 0.0) for a in self.angles]
        return float(max(vals) - min(vals))

    # ── Postural Stability ────────────────────────────────────────────────────
    def postural_stability(self) -> float:
        """
        Postural Instability Index:
          Standard deviation of hip-centre displacement over the buffer,
          normalised to [0, 1] via a physiological reference scale.

        Higher value = more instability.
        Uses rolling window — not a single-frame metric.

        Reference scale: 0.05 normalised units ≈ high instability for
        standing / slow movement. This is an engineering approximation,
        not a clinically validated measure.
        """
        if len(self) < 3:
            return 0.5   # neutral default until buffer fills
        pts = np.array(list(self.positions))
        # displacement from mean position
        centre = pts.mean(axis=0)
        dists  = np.linalg.norm(pts - centre, axis=1)
        std_disp = float(np.std(dists))
        # normalise: clamp to [0, 1] with 0.05 as the upper reference
        return float(np.clip(std_disp / 0.05, 0.0, 1.0))


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================
class MovementFeatureExtractor:
    """
    Converts MediaPipe landmarks + temporal buffer into the exact
    13-column feature vector expected by the Phase 4 trained model.

    Training schema (from movement_preprocessor.pkl):
      hip_flexion_angle, knee_flexion_angle, ankle_rotation_angle,
      angular_velocity, linear_acceleration, ground_reaction_force,
      postural_instability_index, biomechanical_deviation_score,
      fatigue_level, movement_intensity, biomechanical_risk_index,
      movement_asymmetry, fatigue_grf_interaction
    """

    # Normative reference angles for biomechanical deviation scoring
    # (typical comfortable standing/walking values — engineering approximation)
    NORMATIVE = {
        'knee':  170.0,   # near-full extension
        'hip':   175.0,
        'ankle': 90.0,
    }

    def __init__(self, buffer: TemporalBuffer,
                 body_weight_kg: float = ATHLETE_BODY_WEIGHT_KG):
        self.buffer        = buffer
        self.body_weight_kg = body_weight_kg
        self._frame_count  = 0          # proxy for fatigue estimation

    def extract(self, landmarks: dict, ts: float) -> dict | None:
        """
        Extract all features from current landmarks + buffer history.

        Returns None if landmarks are incomplete (person not fully visible).
        """
        required = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
                    'left_ankle', 'right_ankle',
                    'left_shoulder', 'right_shoulder']
        if not all(k in landmarks for k in required):
            return None

        self._frame_count += 1

        # ── Raw joint angles ─────────────────────────────────────────────────
        # Left side
        l_knee_ang  = calculate_angle(landmarks['left_hip'],
                                      landmarks['left_knee'],
                                      landmarks['left_ankle'])
        l_hip_ang   = calculate_angle(landmarks['left_shoulder'],
                                      landmarks['left_hip'],
                                      landmarks['left_knee'])
        l_ankle_ang = calculate_angle(landmarks['left_knee'],
                                      landmarks['left_ankle'],
                                      landmarks.get('left_foot',
                                          landmarks['left_ankle'] + np.array([0, 0.05])))

        # Right side
        r_knee_ang  = calculate_angle(landmarks['right_hip'],
                                      landmarks['right_knee'],
                                      landmarks['right_ankle'])
        r_hip_ang   = calculate_angle(landmarks['right_shoulder'],
                                      landmarks['right_hip'],
                                      landmarks['right_knee'])
        r_ankle_ang = calculate_angle(landmarks['right_knee'],
                                      landmarks['right_ankle'],
                                      landmarks.get('right_foot',
                                          landmarks['right_ankle'] + np.array([0, 0.05])))

        # Representative angles (average left/right)
        knee_angle  = (l_knee_ang  + r_knee_ang)  / 2.0
        hip_angle   = (l_hip_ang   + r_hip_ang)   / 2.0
        ankle_angle = (l_ankle_ang + r_ankle_ang) / 2.0

        # ── Hip centre (proxy for CoM) ────────────────────────────────────────
        hip_centre = (landmarks['left_hip'] + landmarks['right_hip']) / 2.0

        # Push to buffer BEFORE computing temporal features
        current_angles = {
            'knee': knee_angle, 'hip': hip_angle, 'ankle': ankle_angle,
            'l_knee': l_knee_ang, 'r_knee': r_knee_ang,
            'l_hip': l_hip_ang,  'r_hip': r_hip_ang,
        }
        self.buffer.push(current_angles, hip_centre, ts)

        # ── Temporal features ─────────────────────────────────────────────────
        ang_vel   = self.buffer.mean_angular_velocity()           # °/s
        lin_accel = self.buffer.linear_acceleration()             # norm-units/s²
        knee_rom  = self.buffer.range_of_motion('knee')           # °

        # ── Ground Reaction Force proxy ───────────────────────────────────────
        # GRF ≈ body_weight × (1 + vertical_acceleration_normalised × g)
        # lin_accel is in normalised screen units; scale to approximate m/s²
        # (1 normalised unit ≈ frame_height pixels; assume 720px ≈ 1.8 m)
        # This is a coarse proxy — a force plate gives true GRF.
        SCREEN_TO_METRES = 1.8
        accel_ms2 = lin_accel * SCREEN_TO_METRES * 9.81
        grf = self.body_weight_kg * (9.81 + accel_ms2)           # Newtons

        # ── Postural instability ──────────────────────────────────────────────
        postural_instability = self.buffer.postural_stability()   # [0, 1]

        # ── Biomechanical deviation score ─────────────────────────────────────
        # Mean normalised deviation from normative angles
        knee_dev  = abs(knee_angle  - self.NORMATIVE['knee'])  / 180.0
        hip_dev   = abs(hip_angle   - self.NORMATIVE['hip'])   / 180.0
        ankle_dev = abs(ankle_angle - self.NORMATIVE['ankle']) / 180.0
        bio_dev   = float(np.clip((knee_dev + hip_dev + ankle_dev) / 3.0, 0, 1))

        # ── Fatigue level proxy ───────────────────────────────────────────────
        # Modelled as a slow ramp that increases with session duration
        # and with postural instability.  Resets when the session restarts.
        # This is a simplified proxy — true fatigue requires EMG / HRV.
        session_minutes = self._frame_count / (30.0 * 60.0)   # assume 30 fps
        fatigue = float(np.clip(
            session_minutes / 30.0 +          # ramp: full fatigue at 30 min
            postural_instability * 0.3,       # instability contribution
            0.0, 1.0
        ))

        # ── Engineered features (must match Phase 3 derivations exactly) ──────
        # movement_intensity = angular_velocity × linear_acceleration
        movement_intensity = ang_vel * lin_accel

        # biomechanical_risk_index = 0.5×postural_instability + 0.5×bio_dev
        biomechanical_risk_index = 0.5 * postural_instability + 0.5 * bio_dev

        # movement_asymmetry = |hip_L - hip_R| (using knee as primary measure)
        movement_asymmetry = abs(l_knee_ang - r_knee_ang)

        # fatigue_grf_interaction = fatigue_level × ground_reaction_force
        fatigue_grf_interaction = fatigue * grf

        # ── Map to training schema ────────────────────────────────────────────
        features = {
            'hip_flexion_angle':            hip_angle,
            'knee_flexion_angle':           knee_angle,
            'ankle_rotation_angle':         ankle_angle - 90.0,  # offset to match training range
            'angular_velocity':             ang_vel,
            'linear_acceleration':          lin_accel * 10.0,    # scale to training units (~m/s²)
            'ground_reaction_force':        grf,
            'postural_instability_index':   postural_instability,
            'biomechanical_deviation_score': bio_dev,
            'fatigue_level':                fatigue,
            'movement_intensity':           movement_intensity,
            'biomechanical_risk_index':     biomechanical_risk_index,
            'movement_asymmetry':           movement_asymmetry,
            'fatigue_grf_interaction':      fatigue_grf_interaction,
        }

        return features, {
            'l_knee': l_knee_ang, 'r_knee': r_knee_ang,
            'l_hip':  l_hip_ang,  'r_hip':  r_hip_ang,
            'l_ankle': l_ankle_ang,'r_ankle': r_ankle_ang,
        }


# =============================================================================
# RISK PREDICTOR
# =============================================================================
class MovementRiskPredictor:
    """Wraps the Phase 4 sklearn pipeline for single-sample inference."""

    def __init__(self, model, meta: dict):
        self.model       = model
        self.feat_cols   = meta['feature_columns']
        self.low_thr     = meta.get('low_threshold',  LOW_THRESHOLD)
        self.mod_thr     = meta.get('mod_threshold',  MODERATE_THRESHOLD)
        self.model_name  = meta.get('model_name', 'Unknown')

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single feature dict.

        Returns
        -------
        dict:
            risk_probability : float [0, 1]
            risk_pct         : float [0, 100]
            risk_category    : 'LOW' | 'MODERATE' | 'HIGH'
        """
        row = {col: features.get(col, 0.0) for col in self.feat_cols}
        df  = pd.DataFrame([row])[self.feat_cols]

        prob = float(self.model.predict_proba(df)[0, 1])

        if prob < self.low_thr:
            cat = 'LOW'
        elif prob < self.mod_thr:
            cat = 'MODERATE'
        else:
            cat = 'HIGH'

        return {
            'risk_probability': prob,
            'risk_pct':         round(prob * 100, 1),
            'risk_category':    cat,
        }


# =============================================================================
# HUD OVERLAY DRAWING
# =============================================================================
def risk_color(category: str):
    return {'LOW': COLOR_GREEN, 'MODERATE': COLOR_YELLOW, 'HIGH': COLOR_RED}.get(
        category, COLOR_WHITE
    )

def draw_hud(frame: np.ndarray,
             angles: dict,
             features: dict,
             prediction: dict,
             fps: float) -> np.ndarray:
    """Draw semi-transparent HUD panel with risk info and joint angles."""
    h, w = frame.shape[:2]

    # ── Left panel background ─────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Risk score ────────────────────────────────────────────────────────────
    cat   = prediction['risk_category']
    pct   = prediction['risk_pct']
    color = risk_color(cat)

    cv2.putText(frame, 'MOVEMENT RISK', (10, 30),
                FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f'{pct:.1f}%', (10, 70),
                FONT, 1.6, color, 3, cv2.LINE_AA)
    cv2.putText(frame, cat, (10, 105),
                FONT, 0.85, color, 2, cv2.LINE_AA)

    # Risk bar
    bar_w = int(pct / 100 * 250)
    cv2.rectangle(frame, (10, 115), (260, 130), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 115), (10 + bar_w, 130), color, -1)

    # ── Joint angles ──────────────────────────────────────────────────────────
    y = 155
    cv2.putText(frame, 'JOINT ANGLES', (10, y),
                FONT, 0.50, COLOR_BLUE, 1, cv2.LINE_AA)
    angle_items = [
        ('L Knee',  angles.get('l_knee',  0)),
        ('R Knee',  angles.get('r_knee',  0)),
        ('L Hip',   angles.get('l_hip',   0)),
        ('R Hip',   angles.get('r_hip',   0)),
        ('L Ankle', angles.get('l_ankle', 0)),
        ('R Ankle', angles.get('r_ankle', 0)),
    ]
    for label, val in angle_items:
        y += 22
        cv2.putText(frame, f'{label}: {val:.1f}\xb0', (10, y),
                    FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── Key features ─────────────────────────────────────────────────────────
    y += 30
    cv2.putText(frame, 'FEATURES', (10, y),
                FONT, 0.50, COLOR_BLUE, 1, cv2.LINE_AA)
    feat_display = [
        ('AngVel',   features.get('angular_velocity',         0), '°/s'),
        ('Accel',    features.get('linear_acceleration',      0), 'u/s²'),
        ('GRF',      features.get('ground_reaction_force',    0), 'N'),
        ('Fatigue',  features.get('fatigue_level',            0) * 100, '%'),
        ('Asymm',    features.get('movement_asymmetry',       0), '°'),
        ('PostInst', features.get('postural_instability_index', 0) * 100, '%'),
        ('BioRisk',  features.get('biomechanical_risk_index', 0) * 100, '%'),
    ]
    for label, val, unit in feat_display:
        y += 20
        cv2.putText(frame, f'{label}: {val:.1f}{unit}', (10, y),
                    FONT, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, h - 15),
                FONT, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    cv2.putText(frame,
                'Thresholds: engineering only - not clinically validated',
                (5, h - 5), FONT, 0.28, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def draw_no_detection(frame: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.putText(frame, 'No pose detected — ensure full body is visible',
                (w // 2 - 250, h // 2), FONT, 0.6, COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
                FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame


# =============================================================================
# MAIN VISION LOOP
# =============================================================================
class PhaseSevenPipeline:
    """
    Full Phase 7 computer-vision pipeline.

    Usage:
        pipeline = PhaseSevenPipeline()
        pipeline.run()                    # live webcam
        pipeline.run(source='video.mp4')  # video file
    """

    def __init__(self):
        print('\n[Phase7] Loading model...')
        self.model, self.meta = load_movement_model()
        self.predictor   = MovementRiskPredictor(self.model, self.meta)
        self.extractor   = LandmarkExtractor()
        self.buffer      = TemporalBuffer(maxlen=BUFFER_SIZE)
        self.feat_engine = MovementFeatureExtractor(self.buffer)

        # Latest prediction (shared with Phase 8)
        self.latest_prediction: dict = {
            'risk_probability': 0.0,
            'risk_pct':         0.0,
            'risk_category':    'LOW',
        }
        self.latest_features: dict = {}

    def run(self, source=0, save_output: str = None):
        """
        Run the live pipeline.

        Parameters
        ----------
        source : int | str
            0 = default webcam, or a video file path.
        save_output : str | None
            Path to save annotated video (.mp4). None = don't save.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f'[Phase7] ERROR: Cannot open video source: {source}')
            return

        fps_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        fps_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, src_fps, (fps_w, fps_h))

        print(f'[Phase7] Pipeline running  (press Q to quit)')
        print(f'[Phase7] Source : {source}  |  Resolution: {fps_w}×{fps_h}')
        print(f'[Phase7] Thresholds: LOW<{LOW_THRESHOLD*100:.0f}%'
              f'  MODERATE<{MODERATE_THRESHOLD*100:.0f}%  HIGH≥{MODERATE_THRESHOLD*100:.0f}%')
        print(f'[Phase7] Note: thresholds are application-level, NOT clinically validated\n')

        t_prev   = time.perf_counter()
        fps_disp = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_now  = time.perf_counter()
                dt     = max(t_now - t_prev, 1e-6)
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt)   # smoothed FPS
                t_prev = t_now

                # ── Pose estimation ───────────────────────────────────────────
                results, landmarks, annotated = self.extractor.process(frame)

                if landmarks:
                    # ── Feature extraction ────────────────────────────────────
                    result = self.feat_engine.extract(landmarks, t_now)

                    if result is not None:
                        features, angles = result
                        self.latest_features = features

                        # ── Prediction ────────────────────────────────────────
                        prediction = self.predictor.predict(features)
                        self.latest_prediction = prediction

                        # ── Draw HUD ──────────────────────────────────────────
                        annotated = draw_hud(annotated, angles,
                                             features, prediction, fps_disp)
                    else:
                        annotated = draw_no_detection(annotated, fps_disp)
                else:
                    annotated = draw_no_detection(annotated, fps_disp)

                cv2.imshow('Phase 7 — Movement Risk Analysis', annotated)

                if writer:
                    writer.write(annotated)

                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print('\n[Phase7] Pipeline stopped.')

    def get_latest_risk(self) -> dict:
        """
        Return the most recent risk output.
        Called by Phase 8 fusion engine for integration.
        """
        return self.latest_prediction.copy()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pipeline = PhaseSevenPipeline()
    pipeline.run(source=source)
