"""
Self-assessment preprocessing service.

Converts raw SelfAssessmentInput into structured numerical features
and symptom indicators suitable for the risk-fusion engine.

IMPORTANT: These features are NOT fed into the existing movement or
sensor ML models (those models expect different feature schemas).
They produce a separate symptom-indicator signal for the fusion layer.
"""
import re
from ..schemas import (
    SelfAssessmentInput, SelfAssessmentFeatures,
    SymptomIndicators, Symptom
)

# ── Symptom keyword lists for text extraction ──────────────────────────────────
_PAIN_WORDS      = ['pain', 'hurt', 'ache', 'sore', 'tender', 'burning',
                    'sharp', 'dull', 'throbbing', 'painful']
_LOCATION_WORDS  = ['knee', 'ankle', 'hip', 'shoulder', 'elbow', 'back',
                    'hamstring', 'quad', 'quadricep', 'thigh', 'calf',
                    'groin', 'wrist', 'neck']
_STIFFNESS_WORDS = ['stiff', 'stiffness', 'tight', 'tightness', 'rigid',
                    'hard to bend', 'hard to move', 'locked']
_FATIGUE_WORDS   = ['tired', 'fatigue', 'exhausted', 'heavy', 'weak',
                    'drained', 'sluggish', 'slow']


def _flag(symptom: Symptom, symptoms_list: list) -> int:
    return 1 if symptom in symptoms_list else 0


def preprocess_self_assessment(data: SelfAssessmentInput) -> SelfAssessmentFeatures:
    """
    Transform raw self-assessment input into numerical features.

    Feature design rationale:
    - pain_level / fatigue_level kept on 0–10 scale (ordinal numeric)
    - symptoms binarised (0 = not reported, 1 = reported)
    - training_duration normalised to [0, 1] over a 120-min reference
    - overall_risk_score is a weighted composite (engineering heuristic,
      not a validated clinical score)
    """
    s = data.symptoms

    stiffness   = _flag(Symptom.STIFFNESS,       s)
    swelling    = _flag(Symptom.SWELLING,         s)
    weakness    = _flag(Symptom.WEAKNESS,         s)
    numbness    = _flag(Symptom.NUMBNESS,         s)
    reduced_rom = _flag(Symptom.REDUCED_ROM,      s)
    mov_diff    = int(
        _flag(Symptom.DIFF_WALKING,   s) or
        _flag(Symptom.DIFF_RUNNING,   s) or
        _flag(Symptom.DIFF_DIRECTION, s)
    )
    previous_inj = 1 if data.previous_injury else 0
    n_symptoms   = len(s)
    n_locations  = len(data.pain_locations)

    # Normalised training duration (0 = 0 min, 1 = 120+ min)
    duration_norm = min(data.training_duration_min / 120.0, 1.0)

    # Composite risk score (engineering weights — not clinically validated)
    # Higher pain + fatigue + symptoms → higher score
    raw = (
        (data.pain_level    / 10.0) * 0.30 +
        (data.fatigue_level / 10.0) * 0.25 +
        (n_symptoms / 9.0)          * 0.20 +
        stiffness                   * 0.05 +
        swelling                    * 0.05 +
        mov_diff                    * 0.08 +
        previous_inj                * 0.04 +
        (data.training_intensity.value / 5.0) * 0.03
    )
    overall_risk = min(raw, 1.0)

    return SelfAssessmentFeatures(
        pain_level            = float(data.pain_level),
        fatigue_level         = float(data.fatigue_level),
        stiffness             = stiffness,
        swelling              = swelling,
        weakness              = weakness,
        numbness              = numbness,
        reduced_rom           = reduced_rom,
        movement_difficulty   = mov_diff,
        previous_injury       = previous_inj,
        training_intensity    = data.training_intensity.value,
        training_duration_norm= round(duration_norm, 4),
        n_symptoms            = n_symptoms,
        n_pain_locations      = n_locations,
        overall_risk_score    = round(overall_risk, 4),
    )


def extract_symptom_indicators(data: SelfAssessmentInput,
                                features: SelfAssessmentFeatures) -> SymptomIndicators:
    """
    Build a boolean indicator vector combining:
    - Checkbox/slider responses
    - Free-text keyword extraction

    Text extraction identifies symptom keywords only.
    It does NOT diagnose conditions or diseases.
    """
    text = (data.description or "").lower()

    # Text flags (symptom keyword scanning — not diagnosis)
    text_pain       = any(w in text for w in _PAIN_WORDS)
    text_location   = any(w in text for w in _LOCATION_WORDS)
    text_stiffness  = any(w in text for w in _STIFFNESS_WORDS)
    text_fatigue    = any(w in text for w in _FATIGUE_WORDS)

    return SymptomIndicators(
        pain_detected                = data.pain_level > 0,
        fatigue_detected             = data.fatigue_level > 3,
        stiffness_detected           = bool(features.stiffness) or text_stiffness,
        swelling_detected            = bool(features.swelling),
        weakness_detected            = bool(features.weakness),
        numbness_detected            = bool(features.numbness),
        reduced_rom_detected         = bool(features.reduced_rom),
        movement_difficulty_detected = bool(features.movement_difficulty),
        previous_injury_detected     = bool(features.previous_injury),
        text_pain_detected           = text_pain,
        text_location_mentioned      = text_location,
        text_stiffness_detected      = text_stiffness,
        text_fatigue_detected        = text_fatigue,
    )


def extract_keywords_from_text(text: str) -> dict:
    """
    Scan free-text description for symptom keywords.
    Returns a report dict suitable for display.

    This is keyword matching only — NOT NLP diagnosis.
    """
    if not text:
        return {}

    t = text.lower()
    report = {}

    if any(w in t for w in _PAIN_WORDS):
        report['Pain'] = 'Detected'

    locations_found = [w.capitalize() for w in _LOCATION_WORDS if w in t]
    if locations_found:
        report['Location'] = ', '.join(locations_found)

    if any(w in t for w in _STIFFNESS_WORDS):
        report['Stiffness'] = 'Detected'

    if any(w in t for w in _FATIGUE_WORDS):
        report['Fatigue'] = 'Detected'

    if 'training' in t or 'workout' in t or 'session' in t or 'match' in t:
        report['Training Reference'] = 'Detected'

    if 'swelling' in t or 'swollen' in t:
        report['Swelling'] = 'Detected'

    if 'numb' in t or 'tingling' in t:
        report['Numbness / Tingling'] = 'Detected'

    return report


def risk_status_label(score: float) -> str:
    """Convert composite risk score to a display label."""
    if score >= 0.60:
        return "Elevated Risk Indicators"
    elif score >= 0.35:
        return "Moderate Risk Indicators"
    else:
        return "Low Risk Indicators"
