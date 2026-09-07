"""
=============================================================================
 PHASE 10 — Backend Configuration
 All thresholds and weights are centralised here.
 Change values here only — never hardcode in route files.
=============================================================================
"""
import os

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR   = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Risk thresholds (application-level — NOT clinically validated) ────────────
RISK_LOW_MAX      = 40   # overall_risk_pct < 40  → LOW
RISK_MODERATE_MAX = 70   # 40 ≤ overall_risk_pct < 70 → MODERATE
                          # ≥ 70 → HIGH

# ── Fusion weights ────────────────────────────────────────────────────────────
MOVEMENT_WEIGHT = 0.35
SENSOR_WEIGHT   = 0.30
VISION_WEIGHT   = 0.35

# ── Temporal smoothing ────────────────────────────────────────────────────────
EMA_ALPHA           = 0.30   # exponential moving average smoothing factor
RISK_HISTORY_WINDOW = 10     # frames kept for trend
CONSECUTIVE_HIGH    = 3      # sustained high-risk frames before alert fires

# ── Single-model escalation ───────────────────────────────────────────────────
SINGLE_MODEL_HIGH_TRIGGER = 0.80

# ── Demo mode update interval (seconds) ──────────────────────────────────────
DEMO_UPDATE_INTERVAL = 1.0

# ── WebSocket heartbeat (seconds) ─────────────────────────────────────────────
WS_HEARTBEAT = 2.0

DISCLAIMER = (
    "This system provides AI-based risk indicators for monitoring and "
    "decision support. It does not provide a medical diagnosis or replace "
    "professional assessment."
)
