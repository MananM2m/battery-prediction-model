"""Create derived, non-leaky features from raw telemetry."""
from typing import List
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add defensible derived features.

    Features added:
    - `power_inst`: voltage * current (instantaneous power)
    - `v_x_i`: voltage * current interaction (same as power_inst but kept as named feature)
    - `temp_current`: temperature * current interaction
    - `current_squared`: non-linear current term

    This function avoids using future information and only derives from existing columns.
    """
    df = df.copy()
    df["power_inst"] = df["voltage"] * df["current"]
    df["v_x_i"] = df["voltage"] * df["current"]
    df["temp_current"] = df["temperature"] * df["current"]
    df["current_squared"] = df["current"] ** 2
    return df


def feature_columns() -> List[str]:
    return [
        "voltage",
        "current",
        "temperature",
        "load_state",
        "power_inst",
        "v_x_i",
        "temp_current",
        "current_squared",
    ]
