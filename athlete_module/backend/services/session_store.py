"""
In-memory session store for athlete assessments.
Stores only operational data — no unnecessary personal information.
Replace with SQLite/PostgreSQL for production persistence.
"""
import time
import uuid
from typing import List, Optional, Dict

_sessions: Dict[str, dict] = {}


def save_session(record: dict) -> str:
    """Store an assessment record and return its session_id."""
    sid = str(uuid.uuid4())[:10]
    record['session_id'] = sid
    record['stored_at']  = time.time()
    _sessions[sid] = record
    return sid


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def get_athlete_sessions(athlete_id: str) -> List[dict]:
    return [
        s for s in _sessions.values()
        if s.get('athlete_id') == athlete_id
    ]


def list_sessions(last_n: int = 20) -> List[dict]:
    sorted_sessions = sorted(
        _sessions.values(),
        key=lambda s: s.get('stored_at', 0),
        reverse=True,
    )
    return sorted_sessions[:last_n]
