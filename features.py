"""
Shared feature-engineering logic for the Predictive Maintenance project.

This is the single source of truth for how raw sensor readings become model
features - used by the training notebook, the Streamlit app, and the test
suite, so training and serving can never quietly drift apart.
"""

import numpy as np
import pandas as pd

TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_diff [K]",
    "Power [W]",
    "Wear_Torque_interaction",
    "Type_encoded",
]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a dataframe of raw AI4I sensor readings.
    Returns a new dataframe (no in-place mutation)."""
    data = data.copy()

    data["Temp_diff [K]"] = data["Process temperature [K]"] - data["Air temperature [K]"]

    angular_speed = data["Rotational speed [rpm]"] * (2 * np.pi / 60)
    data["Power [W]"] = data["Torque [Nm]"] * angular_speed

    data["Wear_Torque_interaction"] = data["Tool wear [min]"] * data["Torque [Nm]"]

    data["Type_encoded"] = data["Type"].map(TYPE_ENCODING)

    if data["Type_encoded"].isna().any():
        bad_values = sorted(set(data.loc[data["Type_encoded"].isna(), "Type"]))
        raise ValueError(
            f"Unrecognized machine Type value(s) {bad_values} - expected one of {list(TYPE_ENCODING)}"
        )

    return data


def clean_for_modeling(data: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that shouldn't be used as model features."""
    data = data.copy()
    id_cols = ["UDI", "Product ID"]
    return data.drop(columns=[c for c in id_cols if c in data.columns])


def build_feature_row(air_temp, process_temp, rpm, torque, tool_wear, machine_type) -> pd.DataFrame:
    """Build a single model-ready feature row from raw sensor readings -
    what the Streamlit app calls for one machine at a time. Routes through
    the exact same add_features() used in training, just on a 1-row frame."""
    raw = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type,
    }])
    engineered = add_features(raw)
    return engineered[FEATURE_COLS]


def get_feature_columns(df: pd.DataFrame) -> list:
    """The leakage-safe feature list: everything except identifiers, the
    failure-mode flags, the target, and the raw 'Type' column."""
    drop_cols = FAILURE_MODES + [TARGET, "Type"]
    return [c for c in df.columns if c not in drop_cols]
