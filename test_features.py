"""
Tests for src/features.py - the data-prep and feature-engineering logic
shared by the notebook and the Streamlit app.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS, TARGET, FAILURE_MODES,
    add_features, clean_for_modeling, build_feature_row, get_feature_columns,
)


def test_add_features_creates_expected_columns():
    raw = pd.DataFrame([{
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
        "Type": "M",
    }])
    out = add_features(raw)
    for col in ["Temp_diff [K]", "Power [W]", "Wear_Torque_interaction", "Type_encoded"]:
        assert col in out.columns


def test_temp_diff_is_correct():
    raw = pd.DataFrame([{
        "Air temperature [K]": 298.0, "Process temperature [K]": 308.5,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0,
        "Tool wear [min]": 0, "Type": "L",
    }])
    out = add_features(raw)
    assert out["Temp_diff [K]"].iloc[0] == pytest.approx(10.5)


def test_power_matches_physics_formula():
    # P = Torque * angular_speed, angular_speed = rpm * 2*pi/60
    raw = pd.DataFrame([{
        "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0,
        "Tool wear [min]": 0, "Type": "L",
    }])
    out = add_features(raw)
    expected_power = 40.0 * (1500 * 2 * np.pi / 60)
    assert out["Power [W]"].iloc[0] == pytest.approx(expected_power)


def test_type_encoding_is_ordinal_and_correct():
    raw = pd.DataFrame({
        "Air temperature [K]": [298.0, 298.0, 298.0],
        "Process temperature [K]": [308.0, 308.0, 308.0],
        "Rotational speed [rpm]": [1500, 1500, 1500],
        "Torque [Nm]": [40.0, 40.0, 40.0],
        "Tool wear [min]": [0, 0, 0],
        "Type": ["L", "M", "H"],
    })
    out = add_features(raw)
    assert list(out["Type_encoded"]) == [0, 1, 2]


def test_clean_for_modeling_drops_identifiers():
    raw = pd.DataFrame([{"UDI": 1, "Product ID": "M14860", "Type": "M",
                          "Air temperature [K]": 298.0}])
    out = clean_for_modeling(raw)
    assert "UDI" not in out.columns
    assert "Product ID" not in out.columns


def test_build_feature_row_matches_feature_cols_order():
    row = build_feature_row(298.0, 308.0, 1500, 40.0, 100, "M")
    assert list(row.columns) == FEATURE_COLS
    assert len(row) == 1


def test_build_feature_row_rejects_unknown_type():
    with pytest.raises((KeyError, ValueError)):
        build_feature_row(298.0, 308.0, 1500, 40.0, 100, "X")


def test_get_feature_columns_excludes_leakage_and_identifiers(features_df):
    cols = get_feature_columns(features_df)
    for leaky_col in FAILURE_MODES + [TARGET, "Type"]:
        assert leaky_col not in cols, f"{leaky_col} must never be a model feature"


def test_feature_columns_are_all_numeric(features_df):
    cols = get_feature_columns(features_df)
    for col in cols:
        assert pd.api.types.is_numeric_dtype(features_df[col]), f"{col} is not numeric"
