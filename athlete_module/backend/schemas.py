"""
Pydantic schemas for all API endpoints.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────
class SessionType(str, Enum):
    TRAINING  = "Training"
    MATCH     = "Match"
    RECOVERY  = "Recovery"
    WARMUP    = "Warm-Up"
    OTHER     = "Other"

class Intensity(int, Enum):
    VERY_LOW  = 1
    LOW       = 2
    MODERATE  = 3
    HIGH      = 4
    VERY_HIGH = 5

class PainLocation(str, Enum):
    KNEE       = "Knee"
    ANKLE      = "Ankle"
    HIP        = "Hip"
    SHOULDER   = "Shoulder"
    ELBOW      = "Elbow"
    BACK       = "Back"
    HAMSTRING  = "Hamstring"
    QUADRICEPS = "Quadriceps"
    OTHER      = "Other"

class Symptom(str, Enum):
    SORENESS           = "Muscle soreness"
    STIFFNESS          = "Joint stiffness"
    SWELLING           = "Swelling"
    WEAKNESS           = "Weakness"
    NUMBNESS           = "Numbness / tingling"
    REDUCED_ROM        = "Reduced range of motion"
    DIFF_WALKING       = "Difficulty walking"
    DIFF_RUNNING       = "Difficulty running"
    DIFF_DIRECTION     = "Difficulty changing direction"

class RecoveryStatus(str, Enum):
    FULLY_RECOVERED = "Fully recovered"
    PARTIALLY       = "Partially recovered"
    ONGOING         = "Ongoing"


# ── Self-Assessment input ──────────────────────────────────────────────────────
class SelfAssessmentInput(BaseModel):
    # Athlete info
    athlete_id:          str              = Field(..., min_length=1, max_length=20)
    session_type:        SessionType      = SessionType.TRAINING
    training_duration_min: int            = Field(..., ge=0, le=480,
                                            description="Session duration in minutes")
    training_intensity:  Intensity        = Intensity.MODERATE

    # Pain
    pain_level:          int              = Field(..., ge=0, le=10)
    pain_locations:      List[PainLocation] = Field(default_factory=list)

    # Symptoms
    symptoms:            List[Symptom]    = Field(default_factory=list)

    # Fatigue
    fatigue_level:       int              = Field(..., ge=0, le=10)

    # Previous injury
    previous_injury:     bool             = False
    previous_injury_location: Optional[PainLocation] = None
    recovery_status:     Optional[RecoveryStatus]    = None

    # Free-text description
    description:         Optional[str]   = Field(None, max_length=1000)

    @field_validator('pain_locations')
    @classmethod
    def validate_locations(cls, v):
        return v or []

    @field_validator('symptoms')
    @classmethod
    def validate_symptoms(cls, v):
        return v or []


# ── Preprocessed output (what gets sent to risk fusion) ───────────────────────
class SymptomIndicators(BaseModel):
    pain_detected:               bool
    fatigue_detected:            bool
    stiffness_detected:          bool
    swelling_detected:           bool
    weakness_detected:           bool
    numbness_detected:           bool
    reduced_rom_detected:        bool
    movement_difficulty_detected:bool
    previous_injury_detected:    bool
    text_pain_detected:          bool
    text_location_mentioned:     bool
    text_stiffness_detected:     bool
    text_fatigue_detected:       bool


class SelfAssessmentFeatures(BaseModel):
    pain_level:          float   # 0–10
    fatigue_level:       float   # 0–10
    stiffness:           int     # 0/1
    swelling:            int
    weakness:            int
    numbness:            int
    reduced_rom:         int
    movement_difficulty: int
    previous_injury:     int
    training_intensity:  int     # 1–5
    training_duration_norm: float  # 0–1  (normalised)
    n_symptoms:          int     # total symptom count
    n_pain_locations:    int
    overall_risk_score:  float   # composite 0–1


class MRIResult(BaseModel):
    available:   bool
    prediction:  Optional[str]  = None
    confidence:  Optional[float]= None
    disclaimer:  str = (
        "This is an AI model classification output, not a medical diagnosis. "
        "It must not be used as the sole basis for any clinical decision. "
        "Always consult a qualified medical professional."
    )


class AssessmentResponse(BaseModel):
    athlete_id:          str
    session_id:          str
    timestamp:           float
    features:            SelfAssessmentFeatures
    indicators:          SymptomIndicators
    mri:                 MRIResult
    risk_fusion_payload: dict   # standardised payload for Phase 8 fusion
    status_label:        str    # "Elevated Risk Indicators" / "Moderate" / "Low"
    disclaimer:          str
