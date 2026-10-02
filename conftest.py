"""Shared pytest fixtures for the Predictive Maintenance test suite."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_CSV = "ai4i2020.csv"
FEATURES_CSV = "ai4i2020_features.csv"
MODEL_PATH = "models/best_model.pkl"
META_PATH = "models/best_model_meta.json"


@pytest.fixture(scope="session")
def raw_df():
    if not os.path.exists(RAW_CSV):
        pytest.skip(f"{RAW_CSV} not found - run this from the project root")
    return pd.read_csv(RAW_CSV)


@pytest.fixture(scope="session")
def features_df():
    if not os.path.exists(FEATURES_CSV):
        pytest.skip(f"{FEATURES_CSV} not found - run Part 1 of the notebook first")
    return pd.read_csv(FEATURES_CSV)
