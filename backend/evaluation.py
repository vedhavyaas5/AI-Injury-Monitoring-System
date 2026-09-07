"""
Phase 11 — Model Evaluation
Computes real metrics from saved test splits. No fabricated numbers.
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from backend.config import BASE_DIR

PROCESSED = os.path.join(BASE_DIR, 'processed')
MODELS    = os.path.join(BASE_DIR, 'models')


def _load_split(prefix: str):
    X_test = pd.read_csv(os.path.join(PROCESSED, f'X_test_{prefix}.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED, f'y_test_{prefix}.csv')).squeeze()
    return X_test, y_test


def _eval_model(model, X_test, y_test, model_name: str) -> dict:
    t0    = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    lat   = (time.perf_counter() - t0) * 1000 / max(len(X_test), 1)

    cm = confusion_matrix(y_test, preds).tolist()
    try:
        auc = round(float(roc_auc_score(y_test, proba)), 4)
    except ValueError:
        auc = None

    # High-risk class (class 1) specific metrics
    prec_hr  = round(float(precision_score(y_test, preds, pos_label=1, zero_division=0)), 4)
    rec_hr   = round(float(recall_score(y_test,    preds, pos_label=1, zero_division=0)), 4)
    f1_hr    = round(float(f1_score(y_test,        preds, pos_label=1, zero_division=0)), 4)

    return {
        "model_name":  model_name,
        "accuracy":    round(float(accuracy_score(y_test, preds)), 4),
        "precision":   round(float(precision_score(y_test, preds, zero_division=0, average='weighted')), 4),
        "recall":      round(float(recall_score(y_test, preds, zero_division=0, average='weighted')), 4),
        "f1":          round(float(f1_score(y_test, preds, zero_division=0, average='weighted')), 4),
        "roc_auc":     auc,
        "confusion_matrix": cm,
        "high_risk_class": {
            "precision": prec_hr,
            "recall":    rec_hr,
            "f1":        f1_hr,
            "note": "Recall = fraction of actual high-risk cases detected. "
                    "Low recall = dangerous false negatives."
        },
        "inference_latency_ms_per_sample": round(lat, 4),
        "test_samples": len(y_test),
        "positive_samples": int(y_test.sum()),
    }


def evaluate_all() -> dict:
    """
    Evaluate both saved models on their test splits.
    Returns real metrics — no fabricated numbers.
    """
    results = {}

    # Movement model
    try:
        mv_model = joblib.load(os.path.join(MODELS, 'movement_model.pkl'))
        mv_meta  = joblib.load(os.path.join(MODELS, 'movement_preprocessor.pkl'))
        X_mv, y_mv = _load_split('movement')
        results['movement'] = _eval_model(
            mv_model, X_mv, y_mv, mv_meta.get('model_name', 'Movement Model')
        )
    except Exception as e:
        results['movement'] = {"error": str(e)}

    # Sensor model
    try:
        sn_model = joblib.load(os.path.join(MODELS, 'sensor_model.pkl'))
        sn_meta  = joblib.load(os.path.join(MODELS, 'sensor_preprocessor.pkl'))
        X_sn, y_sn = _load_split('sensor')
        results['sensor'] = _eval_model(
            sn_model, X_sn, y_sn, sn_meta.get('model_name', 'Sensor Model')
        )
    except Exception as e:
        results['sensor'] = {"error": str(e)}

    return results


def get_roc_data(prefix: str) -> dict:
    """Return ROC curve points for the specified model."""
    from sklearn.metrics import roc_curve
    try:
        model = joblib.load(os.path.join(MODELS, f'{prefix}_model.pkl'))
        X, y  = _load_split(prefix)
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc = float(roc_auc_score(y, proba))
        return {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
            "auc": round(auc, 4),
        }
    except Exception as e:
        return {"error": str(e)}
