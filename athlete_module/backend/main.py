"""
=============================================================================
 SPORTS INJURY MONITOR — Athlete Input & MRI Module
 FastAPI Backend  ·  Port 8001
=============================================================================
 Runs alongside the Phase 10 dashboard backend (port 8000).
 Start: python -m uvicorn athlete_module.backend.main:app --port 8001 --reload
=============================================================================
"""
import os
import sys

# Ensure project root is on path so Phase 6 model builders are importable
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes.assessment import router as assessment_router
from .routes.mri        import router as mri_router
from .services.mri_service import _get_model   # warm up on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load MRI model so first request is fast
    try:
        _get_model()
        print("[AthleteMod] MRI model loaded.")
    except Exception as e:
        print(f"[AthleteMod] MRI model not loaded: {e}")
    yield
    print("[AthleteMod] Shutdown.")


app = FastAPI(
    title="Sports Injury Monitor — Athlete Module",
    description=(
        "Self-assessment intake and MRI image analysis endpoints. "
        "Outputs are AI-based risk indicators, NOT medical diagnoses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment_router)
app.include_router(mri_router)


@app.get("/")
def root():
    return {
        "service":    "Sports Injury Monitor — Athlete Module",
        "port":       8001,
        "endpoints": [
            "POST /api/self-assessment",
            "POST /api/mri/analyze",
            "GET  /api/sessions",
            "GET  /api/sessions/{session_id}",
            "GET  /api/athletes/{athlete_id}/sessions",
            "GET  /docs  (Swagger UI)",
        ],
        "disclaimer": (
            "This service provides AI-based risk indicators for decision support. "
            "It does not provide medical diagnoses."
        ),
    }


@app.get("/health")
def health():
    from .services.mri_service import _model, _meta
    return {
        "status":          "ok",
        "mri_model_ready": _model is not None,
        "mri_model_name":  _meta.get("best_model") if _meta else None,
    }
