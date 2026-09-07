"""
In-memory data store for athletes, sessions, measurements, predictions and alerts.
Designed to be replaced with SQLite/PostgreSQL in production.
Thread-safe with asyncio locks.
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional
from collections import deque

# ── In-memory stores ──────────────────────────────────────────────────────────
_athletes:    Dict[str, dict] = {}
_sessions:    Dict[str, dict] = {}
_measurements: Dict[str, List[dict]] = {}   # session_id → list of measurements
_predictions:  Dict[str, List[dict]] = {}   # session_id → list
_alerts:       List[dict]            = []
_risk_history: deque                 = deque(maxlen=200)

_lock = asyncio.Lock()

# ── Seed demo athletes ─────────────────────────────────────────────────────────
def _seed():
    for i, (num, status) in enumerate([
        ("07", "active"), ("12", "active"), ("19", "resting"),
        ("03", "active"), ("21", "resting"),
    ]):
        _athletes[num] = {
            "id": num,
            "name": f"Athlete {num}",
            "position": ["Forward", "Midfielder", "Defender", "Goalkeeper", "Winger"][i],
            "age": [24, 22, 27, 25, 23][i],
            "status": status,
            "current_risk": 0.0,
            "current_fatigue": 0.0,
            "current_load": 0.0,
            "sessions": [],
        }

_seed()

# ── Athletes ───────────────────────────────────────────────────────────────────
def get_all_athletes() -> List[dict]:
    return list(_athletes.values())

def get_athlete(athlete_id: str) -> Optional[dict]:
    return _athletes.get(athlete_id)

def update_athlete_risk(athlete_id: str, risk: float, fatigue: float, load: float):
    if athlete_id in _athletes:
        _athletes[athlete_id].update({
            "current_risk": round(risk, 4),
            "current_fatigue": round(fatigue, 4),
            "current_load": round(load, 4),
        })

# ── Sessions ───────────────────────────────────────────────────────────────────
def create_session(athlete_id: str, session_type: str = "Training") -> dict:
    sid = str(uuid.uuid4())[:8]
    session = {
        "id": sid,
        "athlete_id": athlete_id,
        "type": session_type,
        "start_time": time.time(),
        "end_time": None,
        "duration_min": 0,
        "peak_risk": 0.0,
        "avg_risk": 0.0,
        "peak_hr": 0.0,
        "alerts_count": 0,
        "status": "active",
    }
    _sessions[sid] = session
    _measurements[sid] = []
    _predictions[sid]  = []
    if athlete_id in _athletes:
        _athletes[athlete_id]["sessions"].append(sid)
    return session

def get_sessions(athlete_id: Optional[str] = None) -> List[dict]:
    sessions = list(_sessions.values())
    if athlete_id:
        sessions = [s for s in sessions if s["athlete_id"] == athlete_id]
    return sorted(sessions, key=lambda s: s["start_time"], reverse=True)

def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)

def add_measurement(session_id: str, measurement: dict):
    if session_id in _measurements:
        _measurements[session_id].append(measurement)
        # Update session summary
        s = _sessions.get(session_id)
        if s:
            s["duration_min"] = round((time.time() - s["start_time"]) / 60, 1)
            risks = [m.get("overall_risk", 0) for m in _measurements[session_id]]
            hrs   = [m.get("heart_rate_bpm", 0) for m in _measurements[session_id]]
            s["peak_risk"] = round(max(risks), 4)
            s["avg_risk"]  = round(sum(risks) / len(risks), 4) if risks else 0
            s["peak_hr"]   = round(max(hrs), 1) if hrs else 0

def get_measurements(session_id: str, last_n: int = 100) -> List[dict]:
    return _measurements.get(session_id, [])[-last_n:]

# ── Alerts ─────────────────────────────────────────────────────────────────────
def add_alert(alert: dict) -> dict:
    alert["id"] = str(uuid.uuid4())[:8]
    alert["timestamp"] = time.time()
    alert["acknowledged"] = False
    _alerts.append(alert)
    # Update session alert count
    sid = alert.get("session_id")
    if sid and sid in _sessions:
        _sessions[sid]["alerts_count"] = _sessions[sid].get("alerts_count", 0) + 1
    return alert

def get_alerts(athlete_id: Optional[str] = None,
               level: Optional[str] = None,
               last_n: int = 50) -> List[dict]:
    alerts = list(reversed(_alerts))
    if athlete_id:
        alerts = [a for a in alerts if a.get("athlete_id") == athlete_id]
    if level and level.upper() != "ALL":
        alerts = [a for a in alerts if a.get("level", "").upper() == level.upper()]
    return alerts[:last_n]

def acknowledge_alert(alert_id: str) -> bool:
    for a in _alerts:
        if a["id"] == alert_id:
            a["acknowledged"] = True
            return True
    return False

# ── Risk history (global rolling window for charts) ───────────────────────────
def push_risk_history(entry: dict):
    _risk_history.append(entry)

def get_risk_history(last_n: int = 60) -> List[dict]:
    items = list(_risk_history)
    return items[-last_n:]
