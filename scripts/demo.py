"""End-to-end demo: generate -> train -> evaluate -> sample prediction."""
import os
import pandas as pd
from battery_prediction.data.generator import generate
from battery_prediction.models.train import train_and_save
from battery_prediction.evaluate.evaluate import evaluate
import joblib


def run_demo():
    print("Generating data...")
    generate(n=1000, out="data/telemetry_demo.csv", seed=123)
    print("Training models...")
    train_and_save("data/telemetry_demo.csv", out_dir="models_demo", seed=42)
    print("Evaluating...")
    res = evaluate(data_path="data/telemetry_demo.csv", models_dir="models_demo", seed=42)
    print(res)
    # sample prediction
    df = pd.read_csv("data/telemetry_demo.csv")
    sample = df.iloc[0]
    # load rf models if present
    ppath = os.path.join("models_demo", "rf_power.joblib")
    mpath = os.path.join("models_demo", "rf_minutes.joblib")
    if os.path.exists(ppath) and os.path.exists(mpath):
        pmodel = joblib.load(ppath)
        mmodel = joblib.load(mpath)
        X = sample[["voltage", "current", "temperature", "load_state"]].to_frame().T
        print("Sample telemetry:\n", X.to_dict(orient="records")[0])
        print("Predicted power:", pmodel.predict(X)[0])
        print("Predicted minutes:", mmodel.predict(X)[0])


if __name__ == "__main__":
    run_demo()
