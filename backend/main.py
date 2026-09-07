"""
=============================================================================
 PHASE 10 — FastAPI Backend
 Serves REST endpoints and WebSocket live-data stream.
=============================================================================
"""
import asyncio
import json
import logging
import os
import time
import psutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import DEMO_UPDATE_INTERVAL, WS_HEARTBEAT, DISCLAIMER
from backend.models_loader import load_all_models, models_status
from backend.demo_engine import DemoEngine
from backend.risk_engine import RiskEngine
from backend import data_store as store
from backend.evaluation import evaluate_all, get_roc_data

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_demo_engine  = DemoEngine()
_risk_engine  = RiskEngine()
_active_ws:   set = set()
_demo_mode    = True          # default to demo mode; switch via /api/system/mode
_vision_prob  = 0.0           # updated from Phase 7 if camera is active
_perf_samples = []            # latency measurements


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    load_all_models()
    # Seed demo sessions
    for aid in ["07", "12", "19"]:
        s = store.create_session(aid, "Training")
        logger.info("Demo session %s created for athlete %s", s["id"], aid)
    logger.info("Backend ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Sports Injury Monitor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST — Athletes
# =============================================================================
@app.get("/api/athletes")
def get_athletes():
    return store.get_all_athletes()

@app.get("/api/athletes/{athlete_id}")
def get_athlete(athlete_id: str):
    a = store.get_athlete(athlete_id)
    if not a:
        raise HTTPException(404, f"Athlete {athlete_id} not found")
    return a


# =============================================================================
# REST — Sessions
# =============================================================================
@app.get("/api/sessions")
def get_sessions(athlete_id: Optional[str] = Query(None)):
    return store.get_sessions(athlete_id)

@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = store.get_session(session_id)
    if not s:
        raise HTTPException(404, f"Session {session_id} not found")
    return s

@app.get("/api/sessions/{session_id}/measurements")
def get_measurements(session_id: str, last_n: int = Query(100)):
    return store.get_measurements(session_id, last_n)


# =============================================================================
# REST — Risk
# =============================================================================
@app.get("/api/risk/current")
def get_current_risk():
    return store.get_risk_history(last_n=1)

@app.get("/api/risk/history")
def get_risk_history(last_n: int = Query(60)):
    return store.get_risk_history(last_n)


# =============================================================================
# REST — Alerts
# =============================================================================
@app.get("/api/alerts")
def get_alerts(athlete_id: Optional[str] = Query(None),
               level: Optional[str] = Query(None),
               last_n: int = Query(50)):
    return store.get_alerts(athlete_id, level, last_n)

class AlertAck(BaseModel):
    pass

@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    ok = store.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, "Alert not found")
    return {"status": "acknowledged"}


# =============================================================================
# REST — Analytics
# =============================================================================
@app.get("/api/analytics/risk")
def analytics_risk(athlete_id: Optional[str] = Query(None),
                   last_n: int = Query(60)):
    return store.get_risk_history(last_n)

@app.get("/api/analytics/performance")
def analytics_performance():
    recent = _perf_samples[-100:] if _perf_samples else []
    if not recent:
        return {"message": "No performance data yet"}
    import statistics
    latencies = [p["e2e_ms"] for p in recent if "e2e_ms" in p]
    fps_vals  = [p["fps"]    for p in recent if "fps"    in p]
    return {
        "avg_e2e_ms":  round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_e2e_ms":  round(sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) >= 20 else max(latencies or [0])), 
        "max_e2e_ms":  round(max(latencies or [0]), 2),
        "min_e2e_ms":  round(min(latencies or [0]), 2),
        "avg_fps":     round(statistics.mean(fps_vals), 1) if fps_vals else 0,
        "samples":     len(recent),
    }


# =============================================================================
# REST — System
# =============================================================================
@app.get("/api/system/status")
def system_status():
    ms = models_status()
    proc = psutil.Process()
    mem  = proc.memory_info().rss / (1024**3)
    return {
        "camera_connected":    False,   # updated by Phase 7 integration
        "sensors_connected":   False,
        "movement_model":      ms["movement_model"],
        "sensor_model":        ms["sensor_model"],
        "movement_model_name": ms["movement_model_name"],
        "sensor_model_name":   ms["sensor_model_name"],
        "vision_model":        True,    # Phase 7 pipeline
        "risk_fusion":         True,
        "websocket_clients":   len(_active_ws),
        "demo_mode":           _demo_mode,
        "cpu_pct":             round(psutil.cpu_percent(interval=None), 1),
        "ram_gb":              round(mem, 2),
        "uptime_s":            round(time.time()),
    }

@app.post("/api/system/mode")
def set_mode(demo: bool = Query(True)):
    global _demo_mode
    _demo_mode = demo
    return {"demo_mode": _demo_mode}

@app.post("/api/system/vision_prob")
def set_vision_prob(prob: float = Query(..., ge=0.0, le=1.0)):
    global _vision_prob
    _vision_prob = prob
    return {"vision_prob": _vision_prob}


# =============================================================================
# REST — Evaluation (Phase 11)
# =============================================================================
@app.get("/api/evaluation/models")
def evaluation():
    return evaluate_all()

@app.get("/api/evaluation/roc/{prefix}")
def roc_curve_data(prefix: str):
    if prefix not in ("movement", "sensor"):
        raise HTTPException(400, "prefix must be 'movement' or 'sensor'")
    return get_roc_data(prefix)


# =============================================================================
# REST — Reports
# =============================================================================
@app.get("/api/reports/{session_id}")
def generate_report(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    athlete = store.get_athlete(session["athlete_id"])
    measurements = store.get_measurements(session_id)
    alerts = store.get_alerts()
    alerts = [a for a in alerts if a.get("session_id") == session_id]
    evals  = evaluate_all()
    return {
        "session":      session,
        "athlete":      athlete,
        "measurements": measurements[-200:],
        "alerts":       alerts,
        "model_eval":   evals,
        "disclaimer":   DISCLAIMER,
        "generated_at": time.time(),
    }


# =============================================================================
# WebSocket — Live Data Stream
# =============================================================================
@app.websocket("/ws/monitor/{athlete_id}")
async def ws_monitor(websocket: WebSocket, athlete_id: str):
    await websocket.accept()
    _active_ws.add(websocket)
    logger.info("WS connected: athlete=%s  total=%d", athlete_id, len(_active_ws))

    # Find or create active session
    sessions = store.get_sessions(athlete_id)
    active   = next((s for s in sessions if s["status"] == "active"), None)
    if not active:
        active = store.create_session(athlete_id)

    session_id = active["id"]
    consec_high = 0
    last_alert_time = 0.0

    try:
        while True:
            t_frame = time.perf_counter()

            # ── Generate data ─────────────────────────────────────────────────
            if _demo_mode:
                data = _demo_engine.generate(athlete_id, session_id)
                risk_result = {
                    "movement_risk":    data["movement_risk"],
                    "sensor_risk":      data["sensor_risk"],
                    "vision_risk":      data["vision_risk"],
                    "fused_risk":       data["overall_risk"],
                    "smoothed_risk":    data["smoothed_risk"],
                    "overall_risk_pct": data["overall_risk_pct"],
                    "risk_level":       data["risk_level"],
                    "alert_triggered":  data["overall_risk_pct"] >= 70,
                    "trend":            "STABLE",
                    "weights":          {"movement": 0.35, "sensor": 0.30, "vision": 0.35},
                }
            else:
                data = {}  # Real sensor data would be injected here
                risk_result = _risk_engine.predict(data, _vision_prob)

            # ── Store measurement ─────────────────────────────────────────────
            measurement = {**data, **risk_result, "timestamp": time.time()}
            store.add_measurement(session_id, measurement)
            store.push_risk_history({
                "timestamp":    measurement["timestamp"],
                "athlete_id":   athlete_id,
                "overall_risk": risk_result["smoothed_risk"],
                "movement_risk": risk_result["movement_risk"],
                "sensor_risk":   risk_result["sensor_risk"],
                "vision_risk":   risk_result["vision_risk"],
                "level":         risk_result["risk_level"],
            })
            store.update_athlete_risk(
                athlete_id,
                risk_result["smoothed_risk"],
                data.get("fatigue_level", 0),
                data.get("training_load", 0) / 500.0
            )

            # ── Alert logic ───────────────────────────────────────────────────
            if risk_result.get("alert_triggered") and (time.time() - last_alert_time) > 30:
                from backend.config import CONSECUTIVE_HIGH
                alert = store.add_alert({
                    "athlete_id":  athlete_id,
                    "session_id":  session_id,
                    "level":       risk_result["risk_level"],
                    "risk_score":  risk_result["overall_risk_pct"],
                    "reasons":     _build_alert_reasons(data, risk_result),
                    "demo":        _demo_mode,
                })
                last_alert_time = time.time()
                logger.info("Alert fired: %s  risk=%.1f%%", alert["level"],
                            risk_result["overall_risk_pct"])

            # ── Build WebSocket payload ───────────────────────────────────────
            t_e2e = (time.perf_counter() - t_frame) * 1000
            _perf_samples.append({
                "e2e_ms": round(t_e2e, 2),
                "fps":    data.get("fps", 0),
            })
            if len(_perf_samples) > 500:
                _perf_samples.pop(0)

            payload = {
                "type":       "risk_update",
                "demo":       _demo_mode,
                "athlete_id": athlete_id,
                "session_id": session_id,
                "timestamp":  time.time(),
                "risk":       risk_result,
                "sensors": {
                    "heart_rate_bpm":           data.get("heart_rate_bpm"),
                    "oxygen_saturation_spo2":   data.get("oxygen_saturation_spo2"),
                    "skin_temperature_celsius": data.get("skin_temperature_celsius"),
                    "muscle_activation_emg":    data.get("muscle_activation_emg"),
                    "training_load":            data.get("training_load"),
                    "hydration_level":          data.get("hydration_level"),
                    "stress_index":             data.get("stress_index"),
                    "sensor_fatigue_index":     data.get("sensor_fatigue_index"),
                    "fatigue_level":            data.get("fatigue_level"),
                },
                "joints": {
                    "l_knee":  data.get("l_knee_angle"),
                    "r_knee":  data.get("r_knee_angle"),
                    "l_hip":   data.get("l_hip_angle"),
                    "r_hip":   data.get("r_hip_angle"),
                    "l_ankle": data.get("l_ankle_angle"),
                    "r_ankle": data.get("r_ankle_angle"),
                    "movement_asymmetry":         data.get("movement_asymmetry"),
                    "postural_instability_index": data.get("postural_instability_index"),
                    "angular_velocity":           data.get("angular_velocity"),
                    "ground_reaction_force":      data.get("ground_reaction_force"),
                },
                "performance": {
                    "fps":               data.get("fps"),
                    "pose_latency_ms":   data.get("pose_latency_ms"),
                    "ml_latency_ms":     data.get("ml_latency_ms"),
                    "fusion_latency_ms": data.get("fusion_latency_ms"),
                    "e2e_latency_ms":    round(t_e2e, 1),
                },
            }

            await websocket.send_json(payload)
            await asyncio.sleep(DEMO_UPDATE_INTERVAL)

    except WebSocketDisconnect:
        logger.info("WS disconnected: athlete=%s", athlete_id)
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        _active_ws.discard(websocket)


def _build_alert_reasons(data: dict, risk: dict) -> list:
    reasons = []
    if data.get("fatigue_level", 0) > 0.7:
        reasons.append("High fatigue level")
    if data.get("training_load", 0) > 380:
        reasons.append("Elevated training load")
    if data.get("movement_asymmetry", 0) > 20:
        reasons.append("Movement asymmetry detected")
    if data.get("heart_rate_bpm", 0) > 170:
        reasons.append("Elevated heart rate")
    if risk.get("movement_risk", 0) > 0.65:
        reasons.append("High-risk movement pattern")
    if risk.get("sensor_risk", 0) > 0.65:
        reasons.append("Elevated physiological stress indicators")
    if not reasons:
        reasons.append("Multiple elevated risk indicators")
    return reasons
