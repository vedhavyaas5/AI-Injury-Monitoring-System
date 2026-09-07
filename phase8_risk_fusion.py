"""
=============================================================================
 REAL-TIME SPORTS INJURY MONITORING SYSTEM
 PHASE 8 — Multimodal Risk Fusion Engine
=============================================================================
 Combines outputs from:
   Phase 4 — Physical Movement ML Model
   Phase 5 — Sensor-Based ML Model
   Phase 7 — Computer Vision Movement Analysis

 Fusion strategy: configurable weighted average (default).
 Meta-learner stacking available as an optional upgrade path.

 DISCLAIMER:
   Fusion weights are initial engineering values derived from
   relative model confidence on the validation set.
   They are NOT medically validated and MUST be reviewed by
   domain experts before any clinical or semi-clinical use.
=============================================================================
"""

import os
import time
import warnings
import json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — all values are configurable
# =============================================================================
BASE_DIR   = r'c:\studies\sports monitoring system'
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Fusion weights (must sum to 1.0) ─────────────────────────────────────────
# Engineering defaults — NOT medically validated.
# Adjust based on validation-set performance of each sub-model.
MOVEMENT_WEIGHT = 0.35
SENSOR_WEIGHT   = 0.30
VISION_WEIGHT   = 0.35

assert abs(MOVEMENT_WEIGHT + SENSOR_WEIGHT + VISION_WEIGHT - 1.0) < 1e-6, \
    'Fusion weights must sum to 1.0'

# ── Risk category thresholds (application-level, NOT clinically validated) ────
RISK_LOW_MAX      = 0.40   # score < 40 % → LOW
RISK_MODERATE_MAX = 0.70   # 40 % ≤ score < 70 % → MODERATE
                            # score ≥ 70 % → HIGH

# ── Alert escalation rule ─────────────────────────────────────────────────────
# Trigger HIGH alert if ANY single model exceeds this threshold,
# regardless of the fused score.
SINGLE_MODEL_HIGH_TRIGGER = 0.80


# =============================================================================
# HELPERS
# =============================================================================
def clamp01(x: float) -> float:
    """Clamp a probability to [0, 1]."""
    return float(np.clip(x, 0.0, 1.0))


def risk_category(score: float) -> str:
    """
    Convert a fused score in [0, 1] to a risk label.

    Thresholds are configurable. NOT clinically validated.
    """
    if score < RISK_LOW_MAX:
        return 'LOW'
    elif score < RISK_MODERATE_MAX:
        return 'MODERATE'
    else:
        return 'HIGH'


def format_bar(prob: float, width: int = 30) -> str:
    """ASCII progress bar for console display."""
    filled = int(prob * width)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {prob*100:5.1f}%'


# =============================================================================
# INDIVIDUAL MODEL INTERFACES
# =============================================================================
class MovementModelInterface:
    """
    Wraps the Phase 4 Physical Movement sklearn pipeline.
    Input  : dict of biomechanical features (13 columns)
    Output : float probability [0, 1]
    """
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'movement_model.pkl'))
        self.meta  = joblib.load(os.path.join(MODELS_DIR, 'movement_preprocessor.pkl'))
        self.feat_cols = self.meta['feature_columns']
        print(f'  [Fusion] Movement model loaded  : {self.meta["model_name"]}')

    def predict_proba(self, features: dict) -> float:
        """Return injury-risk probability for one feature dict."""
        row = {c: features.get(c, 0.0) for c in self.feat_cols}
        df  = pd.DataFrame([row])[self.feat_cols]
        return clamp01(float(self.model.predict_proba(df)[0, 1]))


class SensorModelInterface:
    """
    Wraps the Phase 5 Sensor-Based XGBoost pipeline.
    Input  : dict of sensor features (12 columns)
    Output : float probability [0, 1]
    """
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'sensor_model.pkl'))
        self.meta  = joblib.load(os.path.join(MODELS_DIR, 'sensor_preprocessor.pkl'))
        self.feat_cols = self.meta['feature_columns']
        print(f'  [Fusion] Sensor model loaded    : {self.meta["model_name"]}')

    def predict_proba(self, features: dict) -> float:
        row = {c: features.get(c, 0.0) for c in self.feat_cols}
        df  = pd.DataFrame([row])[self.feat_cols]
        return clamp01(float(self.model.predict_proba(df)[0, 1]))


class VisionModelInterface:
    """
    Receives real-time probability from Phase 7 computer-vision pipeline.

    In live mode : Phase7Pipeline.get_latest_risk() is called each cycle.
    In demo mode : a probability value is passed directly (for testing
                   the fusion engine without a camera).
    """
    def __init__(self, live_pipeline=None):
        """
        Parameters
        ----------
        live_pipeline : PhaseSevenPipeline | None
            If provided, the latest risk is pulled from the live pipeline.
            If None, predictions must be supplied manually via set_prob().
        """
        self._pipeline    = live_pipeline
        self._manual_prob = 0.0
        print(f'  [Fusion] Vision interface ready : '
              f'{"live pipeline" if live_pipeline else "manual / demo mode"}')

    def set_prob(self, prob: float):
        """Manually set a probability (used when pipeline is not running)."""
        self._manual_prob = clamp01(prob)

    def predict_proba(self) -> float:
        if self._pipeline is not None:
            return clamp01(
                self._pipeline.get_latest_risk().get('risk_probability', 0.0)
            )
        return self._manual_prob


# =============================================================================
# FUSION ENGINE
# =============================================================================
class RiskFusionEngine:
    """
    Multimodal risk fusion combining movement, sensor, and vision signals.

    Fusion formula (weighted average):
        fused = w_m × movement_prob
              + w_s × sensor_prob
              + w_v × vision_prob

    All weights are configurable and must sum to 1.0.

    Optional escalation rule:
        If any single model exceeds SINGLE_MODEL_HIGH_TRIGGER,
        the alert is elevated to HIGH regardless of the fused score.

    !! DISCLAIMER !!
        Weights and thresholds are engineering defaults only.
        They have NOT been validated against clinical outcomes.
    """

    def __init__(self,
                 movement_interface: MovementModelInterface,
                 sensor_interface:   SensorModelInterface,
                 vision_interface:   VisionModelInterface,
                 movement_weight: float = MOVEMENT_WEIGHT,
                 sensor_weight:   float = SENSOR_WEIGHT,
                 vision_weight:   float = VISION_WEIGHT):

        self.movement = movement_interface
        self.sensor   = sensor_interface
        self.vision   = vision_interface

        total = movement_weight + sensor_weight + vision_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Weights must sum to 1.0, got {total:.4f}')

        self.w_m = movement_weight
        self.w_s = sensor_weight
        self.w_v = vision_weight

        self._history = []   # list of FusionResult dicts for trend analysis

    def fuse(self,
             movement_features: dict,
             sensor_features:   dict,
             vision_prob:       float | None = None) -> dict:
        """
        Run all three models and fuse their outputs.

        Parameters
        ----------
        movement_features : dict  — biomechanical feature dict
        sensor_features   : dict  — wearable sensor feature dict
        vision_prob       : float | None
            Override vision probability (for demo / testing).
            If None, pulled from the live pipeline or manual setting.

        Returns
        -------
        FusionResult dict:
            movement_prob   : float [0, 1]
            sensor_prob     : float [0, 1]
            vision_prob     : float [0, 1]
            fused_score     : float [0, 1]
            fused_pct       : float [0, 100]
            risk_level      : 'LOW' | 'MODERATE' | 'HIGH'
            alert_triggered : bool
            alert_reason    : str
            weights         : dict
            timestamp       : float (Unix time)
        """
        # ── Individual model outputs ──────────────────────────────────────────
        m_prob = self.movement.predict_proba(movement_features)
        s_prob = self.sensor.predict_proba(sensor_features)

        if vision_prob is not None:
            v_prob = clamp01(vision_prob)
            self.vision.set_prob(v_prob)
        else:
            v_prob = self.vision.predict_proba()

        # ── Weighted average fusion ───────────────────────────────────────────
        fused = (self.w_m * m_prob +
                 self.w_s * s_prob +
                 self.w_v * v_prob)
        fused = clamp01(fused)

        # ── Risk classification ───────────────────────────────────────────────
        level = risk_category(fused)

        # ── Escalation rule ───────────────────────────────────────────────────
        alert     = False
        alert_why = ''
        if level == 'HIGH':
            alert     = True
            alert_why = f'Fused score {fused*100:.1f}% ≥ HIGH threshold'
        elif max(m_prob, s_prob, v_prob) >= SINGLE_MODEL_HIGH_TRIGGER:
            alert     = True
            level     = 'HIGH'
            alert_why = (
                f'Single-model escalation: '
                f'Movement={m_prob*100:.1f}%  '
                f'Sensor={s_prob*100:.1f}%  '
                f'Vision={v_prob*100:.1f}%'
            )

        result = {
            'movement_prob':   round(m_prob, 6),
            'sensor_prob':     round(s_prob, 6),
            'vision_prob':     round(v_prob, 6),
            'fused_score':     round(fused,  6),
            'fused_pct':       round(fused * 100, 2),
            'risk_level':      level,
            'alert_triggered': alert,
            'alert_reason':    alert_why,
            'weights': {
                'movement': self.w_m,
                'sensor':   self.w_s,
                'vision':   self.w_v,
            },
            'timestamp': time.time(),
        }

        self._history.append(result)
        return result

    def trend(self, last_n: int = 10) -> dict:
        """
        Summarise trend over the last N fusion cycles.

        Returns mean / max / direction (rising / stable / falling).
        """
        if not self._history:
            return {}
        recent = self._history[-last_n:]
        scores = [r['fused_score'] for r in recent]
        direction = 'STABLE'
        if len(scores) >= 3:
            slope = np.polyfit(range(len(scores)), scores, 1)[0]
            if slope > 0.005:
                direction = 'RISING ↑'
            elif slope < -0.005:
                direction = 'FALLING ↓'
        return {
            'mean_risk':  round(float(np.mean(scores)) * 100, 1),
            'max_risk':   round(float(np.max(scores))  * 100, 1),
            'direction':  direction,
            'n_samples':  len(recent),
        }

    def print_result(self, result: dict):
        """Pretty-print one fusion result to the console."""
        cat_color = {'LOW': '\033[92m', 'MODERATE': '\033[93m', 'HIGH': '\033[91m'}
        reset     = '\033[0m'
        c         = cat_color.get(result['risk_level'], '')

        print(f'\n  {"─"*58}')
        print(f'  MULTIMODAL RISK FUSION RESULT')
        print(f'  {"─"*58}')
        print(f'  Movement Risk  {format_bar(result["movement_prob"])}')
        print(f'  Sensor Risk    {format_bar(result["sensor_prob"])}')
        print(f'  Vision Risk    {format_bar(result["vision_prob"])}')
        print(f'  {"─"*58}')
        print(f'  Overall Risk   {format_bar(result["fused_score"])}')
        print(f'  Risk Level     : {c}{result["risk_level"]}{reset}')
        if result['alert_triggered']:
            print(f'  ⚠  ALERT       : {result["alert_reason"]}')
        print(f'  Weights        : Movement={result["weights"]["movement"]}'
              f'  Sensor={result["weights"]["sensor"]}'
              f'  Vision={result["weights"]["vision"]}')
        print(f'  Note: weights are engineering defaults — NOT medically validated')
        print(f'  {"─"*58}')


# =============================================================================
# LIVE FUSION LOOP (integrates with Phase 7 when camera is active)
# =============================================================================
class LiveFusionSystem:
    """
    Orchestrates all three models in real time.

    In full deployment (Phase 9):
        - Phase 7 pipeline feeds vision_prob via get_latest_risk()
        - Sensor data arrives from wearable API / BLE stream
        - This loop fuses at a configurable rate (default 1 Hz)

    In demo / testing mode:
        - Simulated sensor and vision inputs are used
        - The fusion logic and output format are identical
    """

    def __init__(self, vision_pipeline=None):
        print('\n[Phase8] Initialising Risk Fusion Engine...')
        self.movement_iface = MovementModelInterface()
        self.sensor_iface   = SensorModelInterface()
        self.vision_iface   = VisionModelInterface(live_pipeline=vision_pipeline)
        self.engine         = RiskFusionEngine(
            self.movement_iface,
            self.sensor_iface,
            self.vision_iface,
        )
        print('[Phase8] Ready.\n')

    def run_single(self,
                   movement_features: dict,
                   sensor_features:   dict,
                   vision_prob:       float | None = None) -> dict:
        """Run one fusion cycle and return + print the result."""
        result = self.engine.fuse(movement_features, sensor_features, vision_prob)
        self.engine.print_result(result)
        return result

    def run_demo(self, n_cycles: int = 5, interval_s: float = 1.0):
        """
        Run N demo fusion cycles with synthetic inputs to validate
        the full pipeline end-to-end without requiring live hardware.
        """
        print(f'[Phase8] Running {n_cycles} demo fusion cycles...')
        print('[Phase8] ⚠ Demo uses synthetic inputs — outputs are illustrative only\n')

        # Representative synthetic feature values drawn from training distributions
        DEMO_MOVEMENT_BASE = {
            'hip_flexion_angle':             70.0,
            'knee_flexion_angle':            110.0,
            'ankle_rotation_angle':          -10.0,
            'angular_velocity':               3.5,
            'linear_acceleration':            8.0,
            'ground_reaction_force':       2100.0,
            'postural_instability_index':     0.45,
            'biomechanical_deviation_score':  0.40,
            'fatigue_level':                  0.55,
            'movement_intensity':            28.0,
            'biomechanical_risk_index':       0.425,
            'movement_asymmetry':            12.0,
            'fatigue_grf_interaction':     1155.0,
        }
        DEMO_SENSOR_BASE = {
            'heart_rate_bpm':              145.0,
            'oxygen_saturation_spo2':       96.5,
            'skin_temperature_celsius':     35.8,
            'muscle_activation_emg':         0.62,
            'training_load':               310.0,
            'hydration_level':               0.55,
            'stress_index':                  0.48,
            'sensor_fatigue_index':          0.52,
            'fatigue_adjusted_load':       470.0,
            'physiological_stress_score':    0.50,
            'recovery_index':                0.26,
            'cardiovascular_load':        44950.0,
        }

        results = []
        for i in range(n_cycles):
            print(f'  Cycle {i+1}/{n_cycles}', end=' ')

            # Add slight variation across cycles to simulate changing conditions
            rng = np.random.default_rng(seed=42 + i)
            mv  = {k: v * (1 + rng.uniform(-0.08, 0.08))
                   for k, v in DEMO_MOVEMENT_BASE.items()}
            sv  = {k: v * (1 + rng.uniform(-0.05, 0.05))
                   for k, v in DEMO_SENSOR_BASE.items()}
            vp  = float(np.clip(0.45 + rng.uniform(-0.15, 0.30), 0, 1))

            result = self.engine.fuse(mv, sv, vision_prob=vp)
            self.engine.print_result(result)
            results.append(result)

            if i < n_cycles - 1:
                time.sleep(interval_s)

        # ── Trend summary ──────────────────────────────────────────────────────
        trend = self.engine.trend(last_n=n_cycles)
        print(f'\n  {"─"*58}')
        print(f'  TREND SUMMARY (last {n_cycles} cycles)')
        print(f'  {"─"*58}')
        print(f'  Mean Risk  : {trend["mean_risk"]:.1f}%')
        print(f'  Peak Risk  : {trend["max_risk"]:.1f}%')
        print(f'  Direction  : {trend["direction"]}')
        print(f'  {"─"*58}\n')

        return results

    def run_live(self,
                 get_movement_features,
                 get_sensor_features,
                 fusion_rate_hz: float = 1.0,
                 max_cycles:     int   = 0):
        """
        Continuous fusion loop for deployment.

        Parameters
        ----------
        get_movement_features : callable() → dict
            Function that returns the latest movement feature dict.
            In Phase 9 this comes from the Phase 7 pipeline buffer.

        get_sensor_features   : callable() → dict
            Function that returns the latest sensor reading.
            In Phase 9 this comes from the wearable SDK / BLE stream.

        fusion_rate_hz : float
            How many times per second to run the fusion cycle.
            Default 1 Hz (once per second).

        max_cycles : int
            Stop after N cycles. 0 = run indefinitely.
        """
        interval = 1.0 / max(fusion_rate_hz, 0.1)
        print(f'[Phase8] Live fusion running at {fusion_rate_hz} Hz  '
              f'(Ctrl-C to stop)\n')
        cycle = 0
        try:
            while True:
                t0 = time.perf_counter()
                mv = get_movement_features()
                sv = get_sensor_features()
                result = self.engine.fuse(mv, sv)
                self.engine.print_result(result)
                cycle += 1
                if max_cycles and cycle >= max_cycles:
                    break
                elapsed = time.perf_counter() - t0
                sleep_t = max(0.0, interval - elapsed)
                time.sleep(sleep_t)
        except KeyboardInterrupt:
            print('\n[Phase8] Live fusion stopped.')

        trend = self.engine.trend()
        if trend:
            print(f'\n  Trend: mean={trend["mean_risk"]}%  '
                  f'peak={trend["max_risk"]}%  '
                  f'direction={trend["direction"]}')


# =============================================================================
# STANDALONE PREDICTION FUNCTIONS (importable by Phase 9 / dashboard)
# =============================================================================
_movement_iface = None
_sensor_iface   = None

def _ensure_loaded():
    global _movement_iface, _sensor_iface
    if _movement_iface is None:
        _movement_iface = MovementModelInterface()
    if _sensor_iface is None:
        _sensor_iface   = SensorModelInterface()


def predict_fused_risk(movement_features: dict,
                       sensor_features:   dict,
                       vision_prob:       float = 0.0,
                       movement_weight:   float = MOVEMENT_WEIGHT,
                       sensor_weight:     float = SENSOR_WEIGHT,
                       vision_weight:     float = VISION_WEIGHT) -> dict:
    """
    Stateless single-call fusion function for use in Phase 9 / API endpoints.

    Parameters
    ----------
    movement_features : dict   — biomechanical feature dict (13 keys)
    sensor_features   : dict   — wearable sensor feature dict (12 keys)
    vision_prob       : float  — movement risk probability from Phase 7 [0, 1]
    *_weight          : float  — fusion weights (must sum to 1.0)

    Returns
    -------
    dict with keys:
        overall_risk_pct  : float   overall fused risk 0–100
        risk_level        : str     LOW / MODERATE / HIGH
        movement_risk_pct : float
        sensor_risk_pct   : float
        vision_risk_pct   : float
        alert_triggered   : bool
        disclaimer        : str
    """
    _ensure_loaded()

    m_prob = _movement_iface.predict_proba(movement_features)
    s_prob = _sensor_iface.predict_proba(sensor_features)
    v_prob = clamp01(vision_prob)

    total_w = movement_weight + sensor_weight + vision_weight
    if abs(total_w - 1.0) > 1e-4:
        # auto-normalise if caller passed non-summing weights
        movement_weight /= total_w
        sensor_weight   /= total_w
        vision_weight   /= total_w

    fused = clamp01(movement_weight * m_prob +
                    sensor_weight   * s_prob +
                    vision_weight   * v_prob)

    level = risk_category(fused)
    alert = (level == 'HIGH' or
             max(m_prob, s_prob, v_prob) >= SINGLE_MODEL_HIGH_TRIGGER)

    return {
        'overall_risk_pct':  round(fused   * 100, 2),
        'risk_level':        level,
        'movement_risk_pct': round(m_prob  * 100, 2),
        'sensor_risk_pct':   round(s_prob  * 100, 2),
        'vision_risk_pct':   round(v_prob  * 100, 2),
        'alert_triggered':   alert,
        'disclaimer': (
            'Fusion weights (M={:.2f} S={:.2f} V={:.2f}) are engineering '
            'defaults. Risk thresholds (LOW<{:.0f}% MODERATE<{:.0f}% HIGH≥{:.0f}%) '
            'are application-level only. NOT clinically validated.'.format(
                movement_weight, sensor_weight, vision_weight,
                RISK_LOW_MAX * 100, RISK_MODERATE_MAX * 100, RISK_MODERATE_MAX * 100
            )
        ),
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    system = LiveFusionSystem(vision_pipeline=None)
    system.run_demo(n_cycles=5, interval_s=0.5)
