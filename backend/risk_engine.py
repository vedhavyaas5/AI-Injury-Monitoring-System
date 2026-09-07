"""
Risk Engine — wraps Phase 4/5 models and Phase 8 fusion.
Applies EMA temporal smoothing and alert thresholding.
"""
import time
import logging
import numpy as np
import pandas as pd
from collections import deque
from backend.config import (
    MOVEMENT_WEIGHT, SENSOR_WEIGHT, VISION_WEIGHT,
    RISK_LOW_MAX, RISK_MODERATE_MAX,
    EMA_ALPHA, RISK_HISTORY_WINDOW, CONSECUTIVE_HIGH,
    SINGLE_MODEL_HIGH_TRIGGER, DISCLAIMER
)
from backend.models_loader import get_movement_model, get_sensor_model

logger = logging.getLogger(__name__)

MOVEMENT_COLS = [
    'hip_flexion_angle', 'knee_flexion_angle', 'ankle_rotation_angle',
    'angular_velocity', 'linear_acceleration', 'ground_reaction_force',
    'postural_instability_index', 'biomechanical_deviation_score',
    'fatigue_level', 'movement_intensity', 'biomechanical_risk_index',
    'movement_asymmetry', 'fatigue_grf_interaction'
]
SENSOR_COLS = [
    'heart_rate_bpm', 'oxygen_saturation_spo2', 'skin_temperature_celsius',
    'muscle_activation_emg', 'training_load', 'hydration_level',
    'stress_index', 'sensor_fatigue_index', 'fatigue_adjusted_load',
    'physiological_stress_score', 'recovery_index', 'cardiovascular_load'
]


class RiskEngine:
    def __init__(self):
        self._ema_risk   = 0.0
        self._history    = deque(maxlen=RISK_HISTORY_WINDOW)
        self._consec_high = 0

    def _clamp(self, v): return float(np.clip(v, 0.0, 1.0))

    def _model_prob(self, model, meta, data: dict, cols) -> float:
        if model is None:
            return 0.0
        try:
            row = {c: data.get(c, 0.0) for c in cols}
            df  = pd.DataFrame([row])[cols]
            return self._clamp(float(model.predict_proba(df)[0, 1]))
        except Exception as e:
            logger.warning("Prediction error: %s", e)
            return 0.0

    def predict(self, data: dict, vision_prob: float = 0.0) -> dict:
        t0 = time.perf_counter()

        mv_model, mv_meta = get_movement_model()
        sn_model, sn_meta = get_sensor_model()

        t1 = time.perf_counter()
        m_prob = self._model_prob(mv_model, mv_meta, data, MOVEMENT_COLS)
        t2 = time.perf_counter()
        s_prob = self._model_prob(sn_model, sn_meta, data, SENSOR_COLS)
        t3 = time.perf_counter()
        v_prob = self._clamp(vision_prob)

        fused = self._clamp(
            MOVEMENT_WEIGHT * m_prob +
            SENSOR_WEIGHT   * s_prob +
            VISION_WEIGHT   * v_prob
        )

        # EMA smoothing
        self._ema_risk = EMA_ALPHA * fused + (1 - EMA_ALPHA) * self._ema_risk
        smooth = self._clamp(self._ema_risk)
        self._history.append(smooth)

        # Risk category
        pct = smooth * 100
        if pct < RISK_LOW_MAX:
            level = "LOW"
        elif pct < RISK_MODERATE_MAX:
            level = "MODERATE"
        else:
            level = "HIGH"

        # Consecutive-high counter
        if level == "HIGH":
            self._consec_high += 1
        else:
            self._consec_high = 0

        # Alert escalation
        alert_triggered = (
            self._consec_high >= CONSECUTIVE_HIGH or
            max(m_prob, s_prob, v_prob) >= SINGLE_MODEL_HIGH_TRIGGER
        )

        t4 = time.perf_counter()

        trend = "STABLE"
        if len(self._history) >= 3:
            h = list(self._history)
            slope = np.polyfit(range(len(h)), h, 1)[0]
            if slope > 0.005:
                trend = "RISING"
            elif slope < -0.005:
                trend = "FALLING"

        return {
            "movement_risk":    round(m_prob,  4),
            "sensor_risk":      round(s_prob,  4),
            "vision_risk":      round(v_prob,  4),
            "fused_risk":       round(fused,   4),
            "smoothed_risk":    round(smooth,  4),
            "overall_risk_pct": round(pct,     1),
            "risk_level":       level,
            "alert_triggered":  alert_triggered,
            "consecutive_high": self._consec_high,
            "trend":            trend,
            "weights":          {"movement": MOVEMENT_WEIGHT,
                                 "sensor": SENSOR_WEIGHT,
                                 "vision": VISION_WEIGHT},
            "latency_ms": {
                "model_load_ms":  round((t1 - t0) * 1000, 2),
                "movement_ms":    round((t2 - t1) * 1000, 2),
                "sensor_ms":      round((t3 - t2) * 1000, 2),
                "fusion_ms":      round((t4 - t3) * 1000, 2),
                "total_ml_ms":    round((t4 - t0) * 1000, 2),
            },
            "disclaimer": DISCLAIMER,
        }
