"""Generate synthetic telemetry data for power and battery life demo."""
import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def make_sample(t):
    # simulate voltage between 3.3 and 4.2V, current 0-2A, temp 20-50C
    voltage = np.clip(3.3 + 0.9 * np.sin(t / 50.0) + np.random.normal(0, 0.02), 3.0, 4.3)
    current = np.clip(np.abs(np.random.normal(0.6 + 0.4 * np.cos(t / 30.0), 0.2)), 0.0, 3.0)
    temp = np.clip(25 + 5 * np.sin(t / 200.0) + np.random.normal(0, 1.5), -10, 80)
    load_state = random.choice(["idle", "light", "medium", "heavy"])
    power = voltage * current + np.random.normal(0, 0.05)
    # remaining minutes: inversely proportional to current, with noise
    battery_capacity_mah = 3000
    # approximate hours left = capacity / (current*1000) * some factor
    hours_left = battery_capacity_mah / (max(current, 0.01) * 1000) * (0.8 + 0.4 * (1 if load_state=="idle" else 0))
    minutes_left = max(1.0, hours_left * 60 + np.random.normal(0, 5))
    return {
        "voltage": float(voltage),
        "current": float(current),
        "temperature": float(temp),
        "load_state": load_state,
        "power": float(power),
        "minutes_left": float(minutes_left),
    }


def generate(n=2000, out="data/telemetry.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for t in range(n):
        rows.append(make_sample(t))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.date_range(datetime.now() - timedelta(minutes=n), periods=n, freq="T")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/telemetry.csv")
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    generate(args.n, args.out)


if __name__ == "__main__":
    cli()
