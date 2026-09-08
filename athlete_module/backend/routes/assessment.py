"""
POST /api/self-assessment
"""
import time
from fastapi import APIRouter, HTTPException
from ..schemas import (
    SelfAssessmentInput, AssessmentResponse,
    MRIResult
)
from ..services.assessment_service import (
    preprocess_self_assessment,
    extract_symptom_indicators,
    extract_keywords_from_text,
    risk_status_label,
)
from ..services.session_store import save_session

router = APIRouter(prefix="/api", tags=["Self Assessment"])

DISCLAIMER = (
    "This system provides AI-based risk indicators for monitoring and "
    "decision support. It does not provide a medical diagnosis or replace "
    "professional assessment."
)


@router.post("/self-assessment", response_model=AssessmentResponse)
def submit_self_assessment(data: SelfAssessmentInput):
    """
    Accept athlete self-assessment, preprocess into features,
    extract symptom indicators, and return a standardised payload
    ready for the risk-fusion engine.
    """
    try:
        features   = preprocess_self_assessment(data)
        indicators = extract_symptom_indicators(data, features)
        keywords   = extract_keywords_from_text(data.description or "")
        status     = risk_status_label(features.overall_risk_score)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # Standardised risk-fusion payload
    fusion_payload = {
        "athlete_id": data.athlete_id,
        "self_assessment": {
            "pain_level":           features.pain_level,
            "fatigue_level":        features.fatigue_level,
            "stiffness":            bool(features.stiffness),
            "swelling":             bool(features.swelling),
            "weakness":             bool(features.weakness),
            "numbness":             bool(features.numbness),
            "reduced_rom":          bool(features.reduced_rom),
            "movement_difficulty":  bool(features.movement_difficulty),
            "training_intensity":   features.training_intensity,
            "overall_risk_score":   features.overall_risk_score,
        },
        "symptom_indicators": {
            "pain_detected":               indicators.pain_detected,
            "fatigue_detected":            indicators.fatigue_detected,
            "stiffness_detected":          indicators.stiffness_detected,
            "swelling_detected":           indicators.swelling_detected,
            "weakness_detected":           indicators.weakness_detected,
            "numbness_detected":           indicators.numbness_detected,
            "reduced_rom_detected":        indicators.reduced_rom_detected,
            "movement_difficulty_detected":indicators.movement_difficulty_detected,
            "previous_injury_detected":    indicators.previous_injury_detected,
        },
        "text_keywords": keywords,
        "mri": {"available": False},
    }

    session_record = {
        "athlete_id":       data.athlete_id,
        "session_type":     data.session_type.value,
        "timestamp":        time.time(),
        "pain_level":       data.pain_level,
        "pain_locations":   [p.value for p in data.pain_locations],
        "fatigue_level":    data.fatigue_level,
        "symptoms":         [s.value for s in data.symptoms],
        "training_intensity": data.training_intensity.value,
        "training_duration_min": data.training_duration_min,
        "previous_injury":  data.previous_injury,
        "features":         features.model_dump(),
        "indicators":       indicators.model_dump(),
        "fusion_payload":   fusion_payload,
        "status_label":     status,
        "mri_available":    False,
    }
    session_id = save_session(session_record)

    return AssessmentResponse(
        athlete_id          = data.athlete_id,
        session_id          = session_id,
        timestamp           = session_record["timestamp"],
        features            = features,
        indicators          = indicators,
        mri                 = MRIResult(available=False),
        risk_fusion_payload = fusion_payload,
        status_label        = status,
        disclaimer          = DISCLAIMER,
    )


@router.get("/sessions")
def list_sessions():
    from ..services.session_store import list_sessions as _list
    return _list()


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    from ..services.session_store import get_session as _get
    s = _get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@router.get("/athletes/{athlete_id}/sessions")
def athlete_sessions(athlete_id: str):
    from ..services.session_store import get_athlete_sessions
    return get_athlete_sessions(athlete_id)
