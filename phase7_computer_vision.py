"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM
 PHASE 7 — Computer Vision / Live Movement Analysis
=============================================================================
 Uses: mediapipe.tasks (Python 3.13 compatible — new API)
       mediapipe >= 0.10.x  (solutions API dropped in 1.0+)

 Pipeline:
   Webcam → OpenCV → MediaPipe PoseLandmarker (Tasks API)
        → Body Landmarks → Joint Angles → Movement Features
        → Feature Mapping → Phase 4 ML Model → Movement Risk
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
import joblib

# ── MediaPipe Tasks API (Python 3.13 compatible) ──────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR   = r'c:\studies\sports monitoring system'
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MOVEMENT_MODEL_PATH  = os.path.join(MODELS_DIR, 'movement_model.pkl')
MOVEMENT_META_PATH   = os.path.join(MODELS_DIR, 'movement_preprocessor.pkl')
POSE_MODEL_PATH      = os.path.join(MODELS_DIR, 'pose_landmarker_lite.task')

# ── Risk thresholds (application-level — NOT clinically validated) ────────────
LOW_THRESHOLD      = 0.30
MODERATE_THRESHOLD = 0.60

# ── Temporal buffer ───────────────────────────────────────────────────────────
BUFFER_SIZE = 30

# ── Body weight for GRF proxy ─────────────────────────────────────────────────
ATHLETE_BODY_WEIGHT_KG = 70.0

# ── Display ───────────────────────────────────────────────────────────────────
FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLUE   = (255, 160, 0)
COLOR_DARK   = (30, 30, 30)

# ── MediaPipe PoseLandmark indices ─────────────────────────────────────────────
class LM:
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    LEFT_HEEL      = 29
    RIGHT_HEEL     = 30
    LEFT_FOOT      = 31
    RIGHT_FOOT     = 32


# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================
def load_movement_model():
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
# GEOMETRY HELPERS
# =============================================================================
def calculate_angle(point_a: np.ndarray,
                    point_b: np.ndarray,
                    point_c: np.ndarray) -> float:
    """
    Angle at joint B (degrees) formed by vectors B→A and B→C.
    Accepts 2-D or 3-D arrays; only x/y are used.
    """
    a = np.array(point_a[:2], dtype=float)
    b = np.array(point_b[:2], dtype=float)
    c = np.array(point_c[:2], dtype=float)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-9 else default


def lm_to_arr(lm: NormalizedLandmark) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmark to a numpy [x, y] array."""
    return np.array([lm.x, lm.y], dtype=float)


# =============================================================================
# TEMPORAL BUFFER
# =============================================================================
class TemporalBuffer:
    """
    Rolling window storing joint angles, hip-centre position, and timestamps.
    Used to compute velocity, acceleration, ROM, and postural stability.
    """
    def __init__(self, maxlen: int = BUFFER_SIZE):
        self.angles     = collections.deque(maxlen=maxlen)
        self.positions  = collections.deque(maxlen=maxlen)
        self.timestamps = collections.deque(maxlen=maxlen)

    def push(self, angles: dict, hip_centre: np.ndarray, ts: float):
        self.angles.append(angles)
        self.positions.append(hip_centre.copy())
        self.timestamps.append(ts)

    def __len__(self):
        return len(self.timestamps)

    def angular_velocity(self, key: str) -> float:
        if len(self) < 2:
            return 0.0
        dt = max(self.timestamps[-1] - self.timestamps[-2], 1e-6)
        da = abs(self.angles[-1].get(key, 0.0) - self.angles[-2].get(key, 0.0))
        return safe_div(da, dt)

    def mean_angular_velocity(self) -> float:
        if len(self) < 2:
            return 0.0
        keys = list(self.angles[-1].keys())
        return float(np.mean([self.angular_velocity(k) for k in keys]))

    def linear_acceleration(self) -> float:
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

    def range_of_motion(self, key: str) -> float:
        if len(self) < 2:
            return 0.0
        vals = [a.get(key, 0.0) for a in self.angles]
        return float(max(vals) - min(vals))

    def postural_stability(self) -> float:
        """
        Postural Instability Index:
        Std-dev of hip displacement over buffer, normalised to [0, 1].
        Higher = more instability. Engineering approximation, not clinical.
        """
        if len(self) < 3:
            return 0.5
        pts    = np.array(list(self.positions))
        centre = pts.mean(axis=0)
        dists  = np.linalg.norm(pts - centre, axis=1)
        return float(np.clip(np.std(dists) / 0.05, 0.0, 1.0))


# =============================================================================
# FEATURE EXTRACTOR  (maps MediaPipe landmarks → Phase 4 training schema)
# =============================================================================
class MovementFeatureExtractor:
    """
    Converts PoseLandmarker results + temporal buffer into the 13-column
    feature vector required by the Phase 4 Decision Tree pipeline.

    Training schema (movement_preprocessor.pkl):
      hip_flexion_angle, knee_flexion_angle, ankle_rotation_angle,
      angular_velocity, linear_acceleration, ground_reaction_force,
      postural_instability_index, biomechanical_deviation_score,
      fatigue_level, movement_intensity, biomechanical_risk_index,
      movement_asymmetry, fatigue_grf_interaction
    """
    NORMATIVE = {'knee': 170.0, 'hip': 175.0, 'ankle': 90.0}

    def __init__(self, buffer: TemporalBuffer,
                 body_weight_kg: float = ATHLETE_BODY_WEIGHT_KG):
        self.buffer         = buffer
        self.body_weight_kg = body_weight_kg
        self._frame_count   = 0

    def extract(self, landmarks: list) -> tuple | None:
        """
        Parameters
        ----------
        landmarks : list of NormalizedLandmark (33 points from PoseLandmarker)

        Returns
        -------
        (features_dict, angles_dict) or None if body not fully visible
        """
        try:
            # ── Landmark extraction ───────────────────────────────────────────
            ls  = lm_to_arr(landmarks[LM.LEFT_SHOULDER])
            rs  = lm_to_arr(landmarks[LM.RIGHT_SHOULDER])
            lh  = lm_to_arr(landmarks[LM.LEFT_HIP])
            rh  = lm_to_arr(landmarks[LM.RIGHT_HIP])
            lk  = lm_to_arr(landmarks[LM.LEFT_KNEE])
            rk  = lm_to_arr(landmarks[LM.RIGHT_KNEE])
            la  = lm_to_arr(landmarks[LM.LEFT_ANKLE])
            ra  = lm_to_arr(landmarks[LM.RIGHT_ANKLE])

            # Optional foot points (fallback if not visible)
            lf = lm_to_arr(landmarks[LM.LEFT_FOOT])  if len(landmarks) > LM.LEFT_FOOT  else la + np.array([0, 0.05])
            rf = lm_to_arr(landmarks[LM.RIGHT_FOOT]) if len(landmarks) > LM.RIGHT_FOOT else ra + np.array([0, 0.05])

        except (IndexError, AttributeError):
            return None

        self._frame_count += 1

        # ── Joint angles ─────────────────────────────────────────────────────
        l_knee_ang  = calculate_angle(lh, lk, la)
        r_knee_ang  = calculate_angle(rh, rk, ra)
        l_hip_ang   = calculate_angle(ls, lh, lk)
        r_hip_ang   = calculate_angle(rs, rh, rk)
        l_ankle_ang = calculate_angle(lk, la, lf)
        r_ankle_ang = calculate_angle(rk, ra, rf)

        knee_angle  = (l_knee_ang  + r_knee_ang)  / 2.0
        hip_angle   = (l_hip_ang   + r_hip_ang)   / 2.0
        ankle_angle = (l_ankle_ang + r_ankle_ang) / 2.0

        # ── Hip centre ────────────────────────────────────────────────────────
        hip_centre = (lh + rh) / 2.0

        # Push to buffer
        self.buffer.push(
            {'knee': knee_angle, 'hip': hip_angle, 'ankle': ankle_angle,
             'l_knee': l_knee_ang, 'r_knee': r_knee_ang,
             'l_hip':  l_hip_ang,  'r_hip':  r_hip_ang},
            hip_centre, time.perf_counter()
        )

        # ── Temporal features ─────────────────────────────────────────────────
        ang_vel   = self.buffer.mean_angular_velocity()
        lin_accel = self.buffer.linear_acceleration()

        # GRF proxy: body_weight × (g + vertical_accel_scaled)
        accel_ms2 = lin_accel * 1.8 * 9.81
        grf       = self.body_weight_kg * (9.81 + accel_ms2)

        # ── Postural instability ──────────────────────────────────────────────
        postural_instability = self.buffer.postural_stability()

        # ── Biomechanical deviation score ─────────────────────────────────────
        knee_dev  = abs(knee_angle  - self.NORMATIVE['knee'])  / 180.0
        hip_dev   = abs(hip_angle   - self.NORMATIVE['hip'])   / 180.0
        ankle_dev = abs(ankle_angle - self.NORMATIVE['ankle']) / 180.0
        bio_dev   = float(np.clip((knee_dev + hip_dev + ankle_dev) / 3.0, 0, 1))

        # ── Fatigue proxy (session ramp + instability) ───────────────────────
        session_minutes = self._frame_count / (30.0 * 60.0)
        fatigue = float(np.clip(
            session_minutes / 30.0 + postural_instability * 0.3, 0.0, 1.0))

        # ── Engineered features (match Phase 3 derivations exactly) ──────────
        movement_intensity   = ang_vel * lin_accel
        bio_risk_index       = 0.5 * postural_instability + 0.5 * bio_dev
        movement_asymmetry   = abs(l_knee_ang - r_knee_ang)
        fatigue_grf          = fatigue * grf

        features = {
            'hip_flexion_angle':            hip_angle,
            'knee_flexion_angle':           knee_angle,
            'ankle_rotation_angle':         ankle_angle - 90.0,
            'angular_velocity':             ang_vel,
            'linear_acceleration':          lin_accel * 10.0,
            'ground_reaction_force':        grf,
            'postural_instability_index':   postural_instability,
            'biomechanical_deviation_score': bio_dev,
            'fatigue_level':                fatigue,
            'movement_intensity':           movement_intensity,
            'biomechanical_risk_index':     bio_risk_index,
            'movement_asymmetry':           movement_asymmetry,
            'fatigue_grf_interaction':      fatigue_grf,
        }

        angles = {
            'l_knee': l_knee_ang, 'r_knee': r_knee_ang,
            'l_hip':  l_hip_ang,  'r_hip':  r_hip_ang,
            'l_ankle': l_ankle_ang, 'r_ankle': r_ankle_ang,
        }

        return features, angles


# =============================================================================
# RISK PREDICTOR
# =============================================================================
class MovementRiskPredictor:
    def __init__(self, model, meta: dict):
        self.model      = model
        self.feat_cols  = meta['feature_columns']
        self.low_thr    = meta.get('low_threshold',  LOW_THRESHOLD)
        self.mod_thr    = meta.get('mod_threshold',  MODERATE_THRESHOLD)
        self.model_name = meta.get('model_name', 'Unknown')

    def predict(self, features: dict) -> dict:
        row  = {col: features.get(col, 0.0) for col in self.feat_cols}
        df   = pd.DataFrame([row])[self.feat_cols]
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
# SKELETON DRAWING  (manual — replaces mp.solutions.drawing_utils)
# =============================================================================
# Pose connections (subset of MediaPipe's 33-point skeleton)
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (23, 25), (25, 27), (27, 29), (27, 31),    # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),    # right leg
    (11, 13), (13, 15), (12, 14), (14, 16),    # arms
]

def draw_skeleton(frame: np.ndarray,
                  landmarks: list,
                  w: int, h: int) -> np.ndarray:
    """Draw pose skeleton manually using pixel coordinates."""
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = (int(lm.x * w), int(lm.y * h))

    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)

    for i, (x, y) in pts.items():
        color = (0, 255, 0) if i in {23, 24, 25, 26, 27, 28} else (100, 200, 255)
        cv2.circle(frame, (x, y), 4, color, -1)

    # Highlight key joints
    for idx, label in [(25, 'LK'), (26, 'RK'), (23, 'LH'), (24, 'RH')]:
        if idx in pts:
            cv2.circle(frame, pts[idx], 7, (0, 255, 255), 2)

    return frame


# =============================================================================
# HUD OVERLAY
# =============================================================================
def risk_color(cat: str):
    return {'LOW': COLOR_GREEN, 'MODERATE': COLOR_YELLOW,
            'HIGH': COLOR_RED}.get(cat, COLOR_WHITE)

def draw_hud(frame: np.ndarray, angles: dict,
             features: dict, prediction: dict, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]

    # Semi-transparent left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (285, h), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cat   = prediction['risk_category']
    pct   = prediction['risk_pct']
    color = risk_color(cat)

    # Risk score
    cv2.putText(frame, 'MOVEMENT RISK', (10, 30),
                FONT, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f'{pct:.1f}%', (10, 70),
                FONT, 1.6, color, 3, cv2.LINE_AA)
    cv2.putText(frame, cat, (10, 105),
                FONT, 0.80, color, 2, cv2.LINE_AA)

    bar_w = int(min(pct, 100) / 100 * 255)
    cv2.rectangle(frame, (10, 115), (265, 130), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 115), (10 + bar_w, 130), color, -1)

    # Joint angles
    y = 155
    cv2.putText(frame, 'JOINT ANGLES', (10, y),
                FONT, 0.45, COLOR_BLUE, 1, cv2.LINE_AA)
    for label, val in [
        ('L Knee',  angles.get('l_knee',  0)),
        ('R Knee',  angles.get('r_knee',  0)),
        ('L Hip',   angles.get('l_hip',   0)),
        ('R Hip',   angles.get('r_hip',   0)),
        ('L Ankle', angles.get('l_ankle', 0)),
        ('R Ankle', angles.get('r_ankle', 0)),
    ]:
        y += 22
        cv2.putText(frame, f'{label}: {val:.1f}\xb0',
                    (10, y), FONT, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)

    # Features
    y += 30
    cv2.putText(frame, 'FEATURES', (10, y),
                FONT, 0.45, COLOR_BLUE, 1, cv2.LINE_AA)
    for label, val, unit in [
        ('AngVel',   features.get('angular_velocity',          0), '/s'),
        ('GRF',      features.get('ground_reaction_force',     0), 'N'),
        ('Fatigue',  features.get('fatigue_level',             0) * 100, '%'),
        ('Asymmetry',features.get('movement_asymmetry',        0), 'deg'),
        ('PostInst', features.get('postural_instability_index',0) * 100, '%'),
        ('BioRisk',  features.get('biomechanical_risk_index',  0) * 100, '%'),
    ]:
        y += 20
        cv2.putText(frame, f'{label}: {val:.1f}{unit}',
                    (10, y), FONT, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f'FPS: {fps:.1f}', (10, h - 28),
                FONT, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame,
                'Thresholds: application-level, not clinical',
                (5, h - 10), FONT, 0.26, (100, 100, 100), 1, cv2.LINE_AA)
    return frame


def draw_no_detection(frame: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.putText(frame, 'No pose detected — ensure full body is visible',
                (w // 2 - 280, h // 2), FONT, 0.6, COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
                FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class PhaseSevenPipeline:
    """
    Full Phase 7 real-time computer-vision pipeline.

    Usage:
        pipeline = PhaseSevenPipeline()
        pipeline.run()                     # webcam (default)
        pipeline.run(source='video.mp4')   # video file
        pipeline.run(source=1)             # second camera
    """

    def __init__(self):
        print('\n[Phase7] Loading model...')
        self.model, self.meta = load_movement_model()
        self.predictor   = MovementRiskPredictor(self.model, self.meta)
        self.buffer      = TemporalBuffer(maxlen=BUFFER_SIZE)
        self.feat_engine = MovementFeatureExtractor(self.buffer)

        # Verify pose model exists
        if not os.path.exists(POSE_MODEL_PATH):
            raise FileNotFoundError(
                f'Pose model not found: {POSE_MODEL_PATH}\n'
                f'Download from: https://storage.googleapis.com/mediapipe-models/'
                f'pose_landmarker/pose_landmarker_lite/float16/latest/'
                f'pose_landmarker_lite.task'
            )

        # ── Build PoseLandmarker (Tasks API) ──────────────────────────────────
        base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=mp_vision.RunningMode.IMAGE,  # per-frame synchronous
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        print('[Phase7] PoseLandmarker loaded (Tasks API)')

        # Latest results (shared with Phase 8)
        self.latest_prediction: dict = {
            'risk_probability': 0.0, 'risk_pct': 0.0, 'risk_category': 'LOW',
        }
        self.latest_features: dict = {}

    def _process_frame(self, bgr_frame: np.ndarray):
        """Run pose estimation on one BGR frame using Tasks API."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.landmarker.detect(mp_image)
        return result

    def run(self, source=0, save_output: str = None):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f'[Phase7] ERROR: Cannot open: {source}')
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, src_fps, (fw, fh))

        print(f'[Phase7] Pipeline running  (press Q or ESC to quit)')
        print(f'[Phase7] Source: {source}  |  {fw}×{fh}  |  {src_fps:.0f} fps')
        print(f'[Phase7] LOW < {LOW_THRESHOLD*100:.0f}%  '
              f'MODERATE < {MODERATE_THRESHOLD*100:.0f}%  '
              f'HIGH ≥ {MODERATE_THRESHOLD*100:.0f}%')
        print(f'[Phase7] Thresholds are application-level, NOT clinically validated\n')

        t_prev   = time.perf_counter()
        fps_disp = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_now    = time.perf_counter()
                dt       = max(t_now - t_prev, 1e-6)
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt)
                t_prev   = t_now

                annotated = frame.copy()

                # ── Pose detection ────────────────────────────────────────────
                result = self._process_frame(frame)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]  # first person

                    # Draw skeleton
                    annotated = draw_skeleton(annotated, landmarks, fw, fh)

                    # Extract features
                    out = self.feat_engine.extract(landmarks)
                    if out is not None:
                        features, angles = out
                        self.latest_features = features

                        # Predict
                        prediction = self.predictor.predict(features)
                        self.latest_prediction = prediction

                        # Draw HUD
                        annotated = draw_hud(annotated, angles,
                                             features, prediction, fps_disp)
                    else:
                        annotated = draw_no_detection(annotated, fps_disp)
                else:
                    annotated = draw_no_detection(annotated, fps_disp)

                cv2.imshow('Phase 7 — Movement Risk Analysis', annotated)

                if writer:
                    writer.write(annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.landmarker.close()
            print('\n[Phase7] Pipeline stopped.')

    def get_latest_risk(self) -> dict:
        """Called by Phase 8 fusion engine to get the current vision risk."""
        return self.latest_prediction.copy()


# =============================================================================
# PREDICTION FUNCTION  (importable)
# =============================================================================
def predict_movement_risk(
    input_data: dict,
    model_path: str = MOVEMENT_MODEL_PATH,
    meta_path:  str = MOVEMENT_META_PATH,
    low_thr:    float = LOW_THRESHOLD,
    mod_thr:    float = MODERATE_THRESHOLD,
) -> dict:
    """
    Single-sample inference from a feature dict.
    Probability comes from model.predict_proba — not hardcoded.

    Thresholds are application-level — NOT clinically validated.
    """
    model     = joblib.load(model_path)
    meta      = joblib.load(meta_path)
    feat_cols = meta['feature_columns']

    df_input = pd.DataFrame([input_data])[feat_cols]
    raw_prob = float(model.predict_proba(df_input)[0, 1])

    if raw_prob < low_thr:
        category = 'LOW'
    elif raw_prob < mod_thr:
        category = 'MODERATE'
    else:
        category = 'HIGH'

    return {
        'risk_probability_pct': round(raw_prob * 100, 2),
        'risk_category':        category,
        'raw_probability':      round(raw_prob, 6),
        'model_name':           meta['model_name'],
        'threshold_note':       (
            f'LOW < {low_thr*100:.0f}% | '
            f'MODERATE {low_thr*100:.0f}–{mod_thr*100:.0f}% | '
            f'HIGH ≥ {mod_thr*100:.0f}%  '
            '[APPLICATION-LEVEL THRESHOLDS — NOT clinically validated]'
        ),
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pipeline = PhaseSevenPipeline()
    pipeline.run(source=source)
