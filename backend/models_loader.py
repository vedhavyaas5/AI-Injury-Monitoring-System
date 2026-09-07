"""
Models are loaded ONCE at startup and reused for every prediction.
Never reload inside a request handler.
"""
import os
import joblib
import logging
from backend.config import MODELS_DIR

logger = logging.getLogger(__name__)

_movement_model = None
_movement_meta  = None
_sensor_model   = None
_sensor_meta    = None

def load_all_models():
    global _movement_model, _movement_meta, _sensor_model, _sensor_meta
    try:
        _movement_model = joblib.load(os.path.join(MODELS_DIR, 'movement_model.pkl'))
        _movement_meta  = joblib.load(os.path.join(MODELS_DIR, 'movement_preprocessor.pkl'))
        logger.info("Movement model loaded: %s", _movement_meta.get('model_name'))
    except Exception as e:
        logger.error("Failed to load movement model: %s", e)

    try:
        _sensor_model = joblib.load(os.path.join(MODELS_DIR, 'sensor_model.pkl'))
        _sensor_meta  = joblib.load(os.path.join(MODELS_DIR, 'sensor_preprocessor.pkl'))
        logger.info("Sensor model loaded: %s", _sensor_meta.get('model_name'))
    except Exception as e:
        logger.error("Failed to load sensor model: %s", e)

def get_movement_model():
    return _movement_model, _movement_meta

def get_sensor_model():
    return _sensor_model, _sensor_meta

def models_status() -> dict:
    return {
        "movement_model": _movement_model is not None,
        "sensor_model":   _sensor_model   is not None,
        "movement_model_name": _movement_meta.get('model_name') if _movement_meta else None,
        "sensor_model_name":   _sensor_meta.get('model_name')   if _sensor_meta   else None,
    }
