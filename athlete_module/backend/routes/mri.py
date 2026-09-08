"""
POST /api/mri/analyze
"""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from ..services.mri_service import validate_image_bytes, predict_mri
from ..services.session_store import get_session, save_session
from ..schemas import MRIResult

router = APIRouter(prefix="/api/mri", tags=["MRI Analysis"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
}

@router.post("/analyze")
async def analyze_mri(
    file: UploadFile = File(...),
    athlete_id: str  = Form(default="unknown"),
    session_id: str  = Form(default=""),
):
    """
    Accept an MRI image upload, validate it, run inference,
    and return the model prediction with confidence.

    The result is an AI classification output — NOT a medical diagnosis.
    """
    # ── Content-type guard ────────────────────────────────────────────────────
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Accepted: JPG, JPEG, PNG, WEBP, BMP, TIFF"
        )

    content = await file.read()

    # ── Image validation ──────────────────────────────────────────────────────
    try:
        validate_image_bytes(content, file.filename or "upload.jpg")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Model inference ───────────────────────────────────────────────────────
    try:
        result = predict_mri(content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MRI inference failed: {str(e)}"
        )

    response = {
        "athlete_id":     athlete_id,
        "filename":       file.filename,
        "timestamp":      time.time(),
        "prediction":     result["prediction"],
        "confidence":     result["confidence"],
        "confidence_pct": result["confidence_pct"],
        "top_2":          result["top_2"],
        "model_name":     result["model_name"],
        "disclaimer":     result["disclaimer"],
        "demo_warning":   result.get("demo_warning"),
        "mri_fusion_payload": {
            "available":   True,
            "prediction":  result["prediction"],
            "confidence":  result["confidence"],
        },
    }

    # ── Attach to existing session if session_id provided ────────────────────
    if session_id:
        existing = get_session(session_id)
        if existing:
            existing["mri_available"]    = True
            existing["mri_prediction"]   = result["prediction"]
            existing["mri_confidence"]   = result["confidence"]
            existing["mri_model_name"]   = result["model_name"]
            # Update fusion payload
            if "fusion_payload" in existing:
                existing["fusion_payload"]["mri"] = {
                    "available":  True,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                }
    else:
        # Store standalone MRI session
        save_session({
            "athlete_id":      athlete_id,
            "timestamp":       time.time(),
            "mri_available":   True,
            "mri_prediction":  result["prediction"],
            "mri_confidence":  result["confidence"],
            "mri_model_name":  result["model_name"],
            "fusion_payload": {
                "athlete_id": athlete_id,
                "self_assessment": None,
                "symptom_indicators": None,
                "mri": {
                    "available":  True,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                },
            },
        })

    return response
