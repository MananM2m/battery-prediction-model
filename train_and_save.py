"""Train models to predict power and remaining battery minutes and save them."""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_preprocessor():
    num_feats = ["voltage", "current", "temperature"]
    cat_feats = ["load_state"]
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)])
    return pre


def train_models(df, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    X = df[["voltage", "current", "temperature", "load_state"]]
    y_power = df["power"]
    y_minutes = df["minutes_left"]

    pre = build_preprocessor()

    candidates = []

    # Linear
    lr = Pipeline([("pre", pre), ("est", LinearRegression())])
    candidates.append(("linear_power", lr, { }))

    # Random Forest
    rf = Pipeline([("pre", pre), ("est", RandomForestRegressor(random_state=42))])
    candidates.append(("rf_power", rf, {"est__n_estimators": [50, 100]}))

    # XGBoost (if available)
    if XGBRegressor is not None:
        xgb = Pipeline([("pre", pre), ("est", XGBRegressor(objective="reg:squarederror", random_state=42))])
        candidates.append(("xgb_power", xgb, {"est__n_estimators": [50, 100], "est__max_depth": [3, 5]}))

    # Train for both targets and save best models
    for name, pipe, grid in candidates:
        # power
        gs = GridSearchCV(pipe, grid or {}, cv=3, n_jobs=1)
        gs.fit(X, y_power)
        joblib.dump(gs.best_estimator_, os.path.join(out_dir, f"{name}_power.joblib"))
        print(f"Saved {name}_power.joblib (score {gs.best_score_:.3f})")

        gs2 = GridSearchCV(pipe, grid or {}, cv=3, n_jobs=1)
        gs2.fit(X, y_minutes)
        joblib.dump(gs2.best_estimator_, os.path.join(out_dir, f"{name}_minutes.joblib"))
        print(f"Saved {name}_minutes.joblib (score {gs2.best_score_:.3f})")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/telemetry.csv")
    p.add_argument("--out", default="models")
    args = p.parse_args()
    df = load_data(args.data)
    train_models(df, args.out)


if __name__ == "__main__":
    cli()
