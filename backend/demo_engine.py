"""
Demo / Simulation Engine
Generates realistic, slowly-varying synthetic data for all channels.
Clearly separate from any real sensor or model output.
"""
import math
import time
import random
import numpy as np
from collections import deque

class DemoEngine:
    """
    Generates synthetic but physiologically plausible data streams.
    Values drift slowly over time using sine waves + random noise.

    ALL outputs from this engine are marked with  "demo": True
    so the frontend can display the DEMO MODE banner.
    """

    def __init__(self, seed: int = 42):
        self._rng   = np.random.default_rng(seed)
        self._t0    = time.time()
        self._phase = self._rng.uniform(0, 2 * math.pi, size=20)
        self._risk_history = deque(maxlen=60)
        self._consecutive_high = 0

    def _wave(self, idx: int, period: float, amplitude: float,
              centre: float, noise: float = 0.02) -> float:
        t   = time.time() - self._t0
        val = centre + amplitude * math.sin(2 * math.pi * t / period + self._phase[idx])
        val += self._rng.normal(0, noise)
        return float(val)

    def generate(self, athlete_id: str = "07",
                 session_id: str = "demo") -> dict:
        t = time.time()

        # ── Physiological signals ─────────────────────────────────────────────
        hr          = self._wave(0, 60, 25, 145, 3.0)
        spo2        = self._wave(1, 120, 2, 96.5, 0.3)
        skin_temp   = self._wave(2, 90, 1.2, 35.8, 0.1)
        emg         = self._wave(3, 45, 0.25, 0.62, 0.02)
        training_ld = self._wave(4, 180, 80, 310, 5.0)
        hydration   = self._wave(5, 300, 0.15, 0.55, 0.01)
        stress      = self._wave(6, 75, 0.20, 0.48, 0.02)
        fatigue_idx = self._wave(7, 200, 0.18, 0.52, 0.02)

        # ── Joint angles ──────────────────────────────────────────────────────
        l_knee  = self._wave(8,  4, 18, 148, 1.5)
        r_knee  = self._wave(9,  4, 16, 140, 1.5)
        l_hip   = self._wave(10, 5, 12, 165, 1.0)
        r_hip   = self._wave(11, 5, 10, 160, 1.0)
        l_ankle = self._wave(12, 3,  8, 105, 1.0)
        r_ankle = self._wave(13, 3,  7, 108, 1.0)

        # ── Derived movement features ─────────────────────────────────────────
        ang_vel    = self._wave(14, 3, 1.5, 3.2, 0.2)
        lin_accel  = self._wave(15, 3, 3.0, 7.5, 0.5)
        grf        = self._wave(16, 2, 500, 2100, 30)
        post_inst  = self._wave(17, 8, 0.2, 0.4, 0.02)
        bio_dev    = self._wave(18, 8, 0.2, 0.38, 0.02)
        fatigue_lv = self._wave(19, 200, 0.25, 0.5, 0.02)

        # ── Engineered features (match Phase 3 schema exactly) ────────────────
        movement_intensity    = ang_vel * lin_accel
        bio_risk_index        = 0.5 * post_inst + 0.5 * bio_dev
        movement_asymmetry    = abs(l_knee - r_knee)
        fatigue_grf           = fatigue_lv * grf
        fatigue_adj_load      = training_ld * (1 + fatigue_idx)
        hr_norm = (hr - 55) / (190 - 55 + 1e-9)
        spo2_inv = 1 - (spo2 - 90) / (100 - 90 + 1e-9)
        physio_stress         = 0.4 * hr_norm + 0.3 * spo2_inv + 0.3 * stress
        recovery_idx          = hydration * (1 - stress)
        cardio_load           = hr * training_ld

        # ── Risk probabilities (simple heuristic for demo, not from ML) ──────
        movement_risk = float(np.clip(
            0.3 + 0.4 * (fatigue_lv - 0.3) + 0.3 * (post_inst - 0.2), 0, 1))
        sensor_risk   = float(np.clip(
            0.2 + 0.5 * (hr - 100) / 90 + 0.3 * fatigue_idx, 0, 1))
        vision_risk   = float(np.clip(
            0.25 + 0.5 * (movement_asymmetry / 30) + 0.25 * bio_dev, 0, 1))
        fused         = float(np.clip(
            0.35 * movement_risk + 0.30 * sensor_risk + 0.35 * vision_risk, 0, 1))

        self._risk_history.append(fused)
        avg_recent = float(np.mean(list(self._risk_history)[-5:])) if self._risk_history else fused

        # Risk category
        if avg_recent * 100 < 40:
            level = "LOW"
        elif avg_recent * 100 < 70:
            level = "MODERATE"
        else:
            level = "HIGH"

        return {
            "demo": True,
            "timestamp": t,
            "athlete_id": athlete_id,
            "session_id": session_id,
            # Sensor values
            "heart_rate_bpm":           round(np.clip(hr, 50, 200), 1),
            "oxygen_saturation_spo2":   round(np.clip(spo2, 88, 100), 1),
            "skin_temperature_celsius": round(np.clip(skin_temp, 32, 40), 1),
            "muscle_activation_emg":    round(np.clip(emg, 0, 1), 3),
            "training_load":            round(np.clip(training_ld, 50, 500), 1),
            "hydration_level":          round(np.clip(hydration, 0, 1), 3),
            "stress_index":             round(np.clip(stress, 0, 1), 3),
            "sensor_fatigue_index":     round(np.clip(fatigue_idx, 0, 1), 3),
            # Joint angles
            "l_knee_angle":  round(np.clip(l_knee,  60, 180), 1),
            "r_knee_angle":  round(np.clip(r_knee,  60, 180), 1),
            "l_hip_angle":   round(np.clip(l_hip,  120, 180), 1),
            "r_hip_angle":   round(np.clip(r_hip,  120, 180), 1),
            "l_ankle_angle": round(np.clip(l_ankle, 70, 130), 1),
            "r_ankle_angle": round(np.clip(r_ankle, 70, 130), 1),
            # Movement features
            "angular_velocity":               round(np.clip(ang_vel,   0, 6), 3),
            "linear_acceleration":            round(np.clip(lin_accel, 0, 15), 3),
            "ground_reaction_force":          round(np.clip(grf, 500, 3500), 1),
            "postural_instability_index":     round(np.clip(post_inst, 0, 1), 4),
            "biomechanical_deviation_score":  round(np.clip(bio_dev,   0, 1), 4),
            "fatigue_level":                  round(np.clip(fatigue_lv,0, 1), 4),
            "movement_intensity":             round(np.clip(movement_intensity, 0, 50), 3),
            "biomechanical_risk_index":       round(np.clip(bio_risk_index,     0, 1), 4),
            "movement_asymmetry":             round(np.clip(movement_asymmetry, 0, 60), 2),
            "fatigue_grf_interaction":        round(np.clip(fatigue_grf, 0, 2000), 1),
            # Engineered sensor features
            "fatigue_adjusted_load":          round(np.clip(fatigue_adj_load, 0, 1000), 1),
            "physiological_stress_score":     round(np.clip(physio_stress, 0, 1), 4),
            "recovery_index":                 round(np.clip(recovery_idx, 0, 1), 4),
            "cardiovascular_load":            round(np.clip(cardio_load, 0, 100000), 1),
            # Risk outputs
            "movement_risk": round(movement_risk, 4),
            "sensor_risk":   round(sensor_risk,   4),
            "vision_risk":   round(vision_risk,   4),
            "overall_risk":  round(avg_recent,    4),
            "overall_risk_pct": round(avg_recent * 100, 1),
            "risk_level":    level,
            "smoothed_risk": round(avg_recent, 4),
            # Performance (demo values)
            "fps":              round(self._wave(0, 10, 2, 28, 0.5), 1),
            "pose_latency_ms":  round(abs(self._wave(1, 5, 5, 18, 1)), 1),
            "ml_latency_ms":    round(abs(self._wave(2, 4, 1, 4, 0.3)), 1),
            "fusion_latency_ms":round(abs(self._wave(3, 3, 0.5, 1, 0.1)), 1),
            "e2e_latency_ms":   round(abs(self._wave(4, 6, 8, 34, 2)), 1),
        }
