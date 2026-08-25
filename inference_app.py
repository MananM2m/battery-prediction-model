from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from typing import Optional


MODEL_CANDIDATES = ["linear", "rf", "xgb"]


def find_model(prefix, target):
    out = os.path.join("models", f"{prefix}_{target}.joblib")
    if os.path.exists(out):
        return out
    return None


class Telemetry(BaseModel):
    voltage: float
    current: float
    temperature: float
    load_state: Optional[str] = "idle"


app = FastAPI(title="Telemetry Inference")

# load first available models
POWER_MODEL = None
MIN_MODEL = None
for prefix in MODEL_CANDIDATES:
    p = find_model(prefix, "power")
    m = find_model(prefix, "minutes")
    if p and m:
        POWER_MODEL = joblib.load(p)
        MIN_MODEL = joblib.load(m)
        break


@app.get("/")
def root():
    return {
        "message": "Telemetry inference service. POST to /predict with voltage,current,temperature,load_state"
    }


@app.post("/predict")
def predict(t: Telemetry):
    if POWER_MODEL is None or MIN_MODEL is None:
        return {"error": "models not found - run training first"}
    X = [{
        "voltage": t.voltage,
        "current": t.current,
        "temperature": t.temperature,
        "load_state": t.load_state,
    }]
    p = POWER_MODEL.predict(X)[0]
    m = MIN_MODEL.predict(X)[0]
    return {"power": float(p), "minutes_left": float(m)}
