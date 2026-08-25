import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "data", "telemetry.csv"))
X = df[["voltage", "current", "temperature", "load_state"]]
y_power = df["power"]
y_min = df["minutes_left"]

X_train, X_test, yp_train, yp_test, ym_train, ym_test = train_test_split(
    X, y_power, y_min, test_size=0.2, random_state=42)

def eval_model(path, X_test, y_test):
    m = joblib.load(path)
    preds = m.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return r2, rmse

power_model = os.path.join(BASE, "models", "rf_power_power.joblib")
min_model = os.path.join(BASE, "models", "rf_power_minutes.joblib")

rp, rmp = eval_model(power_model, X_test, yp_test)
rm, rmm = eval_model(min_model, X_test, ym_test)

print(f"Power (RF) R2={rp:.4f}, RMSE={rmp:.4f}")
print(f"Minutes (RF) R2={rm:.4f}, RMSE={rmm:.4f}")
