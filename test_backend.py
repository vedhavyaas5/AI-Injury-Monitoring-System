import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backend.config import MOVEMENT_WEIGHT, SENSOR_WEIGHT, VISION_WEIGHT
from backend.models_loader import load_all_models, models_status
from backend.demo_engine import DemoEngine
from backend.risk_engine import RiskEngine
from backend.evaluation import evaluate_all

print("=== CONFIG ===")
print("Weights:", MOVEMENT_WEIGHT, SENSOR_WEIGHT, VISION_WEIGHT)

print("\n=== MODELS ===")
load_all_models()
ms = models_status()
print("Movement model:", ms["movement_model"], "-", ms["movement_model_name"])
print("Sensor model  :", ms["sensor_model"],   "-", ms["sensor_model_name"])

print("\n=== DEMO ENGINE ===")
de = DemoEngine()
frame = de.generate("07", "test")
print("overall_risk_pct:", frame["overall_risk_pct"])
print("heart_rate_bpm  :", frame["heart_rate_bpm"])
print("l_knee_angle    :", frame["l_knee_angle"])

print("\n=== RISK ENGINE ===")
re = RiskEngine()
res = re.predict(frame)
print("risk_level   :", res["risk_level"])
print("overall_pct  :", res["overall_risk_pct"])
print("smoothed_risk:", res["smoothed_risk"])
print("latency_ms   :", res["latency_ms"])

print("\n=== EVALUATION ===")
ev = evaluate_all()
for k, v in ev.items():
    if "error" not in v:
        acc = round(v["accuracy"] * 100, 2)
        rec = round(v["recall"]   * 100, 2)
        auc = v["roc_auc"]
        print(f"{k}: accuracy={acc}%  recall={rec}%  auc={auc}")
    else:
        print(f"{k}: ERROR - {v['error']}")

print("\nAll backend modules: PASSED")
