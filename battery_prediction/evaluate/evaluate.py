"""Evaluate saved models on a hold-out test set and produce metrics."""
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate(data_path="data/telemetry.csv", models_dir="models", seed=42):
    df = pd.read_csv(data_path)
    X = df[["voltage", "current", "temperature", "load_state"]]
    y_power = df["power"]
    y_min = df["minutes_left"]
    X_train, X_test, yp_train, yp_test, ym_train, ym_test = train_test_split(
        X, y_power, y_min, test_size=0.2, random_state=seed)

    results = []
    for model_name in ["linear", "rf", "xgb"]:
        ppath = os.path.join(models_dir, f"{model_name}_power.joblib")
        mpath = os.path.join(models_dir, f"{model_name}_minutes.joblib")
        if os.path.exists(ppath) and os.path.exists(mpath):
            pmodel = joblib.load(ppath)
            mmodel = joblib.load(mpath)
            ppred = pmodel.predict(X_test)
            mpred = mmodel.predict(X_test)
            results.append({
                "model": model_name,
                "target": "power",
                "mae": float(mean_absolute_error(yp_test, ppred)),
                "rmse": float(mean_squared_error(yp_test, ppred) ** 0.5),
                "r2": float(r2_score(yp_test, ppred)),
            })
            results.append({
                "model": model_name,
                "target": "minutes",
                "mae": float(mean_absolute_error(ym_test, mpred)),
                "rmse": float(mean_squared_error(ym_test, mpred) ** 0.5),
                "r2": float(r2_score(ym_test, mpred)),
            })
    return results


def cli_main(args):
    res = evaluate(args.data, args.models)
    for r in res:
        print(r)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/telemetry.csv")
    p.add_argument("--models", default="models")
    args = p.parse_args()
    cli_main(args)
