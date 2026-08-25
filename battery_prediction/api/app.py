from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
from typing import Optional


class Telemetry(BaseModel):
    voltage: float = Field(..., ge=2.5, le=5.0)
    current: float = Field(..., ge=0.0, le=10.0)
    temperature: float = Field(..., ge=-40.0, le=85.0)
    load_state: Optional[str] = Field("idle")


app = FastAPI(title="Battery Prediction API")


def load_models(models_dir="models"):
    # load first pair found
    candidates = ["rf", "xgb", "linear"]
    for c in candidates:
        p = os.path.join(models_dir, f"{c}_power.joblib")
        m = os.path.join(models_dir, f"{c}_minutes.joblib")
        if os.path.exists(p) and os.path.exists(m):
            return joblib.load(p), joblib.load(m), c
    return None, None, None


POWER_MODEL, MIN_MODEL, MODEL_NAME = load_models()


@app.get("/")
def root():
    return {"service": "Battery prediction", "models_available": MODEL_NAME is not None}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    if MODEL_NAME is None:
        return {"error": "no models loaded"}
    return {"model": MODEL_NAME, "targets": ["power", "minutes_left"]}


@app.post("/predict")
def predict(t: Telemetry):
    if POWER_MODEL is None or MIN_MODEL is None:
        return {"error": "models not available"}
    X = [{
        "voltage": t.voltage,
        "current": t.current,
        "temperature": t.temperature,
        "load_state": t.load_state,
    }]
    p = POWER_MODEL.predict(X)[0]
    m = MIN_MODEL.predict(X)[0]
    return {"predicted_power": float(p), "predicted_remaining_minutes": float(m), "model": MODEL_NAME}
