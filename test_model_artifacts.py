"""
Tests for the trained model artifacts themselves - do they exist, load
correctly, and behave sanely on data the notebook produced?

These need models/best_model.pkl + models/best_model_meta.json to exist,
i.e. Part 3 of the notebook must have been run at least once. They're
skipped (not failed) if the artifacts aren't there, so this suite still
runs cleanly on a fresh checkout before training.
"""

import json
import os

import joblib
import pandas as pd
import pytest

from src.features import FEATURE_COLS, TARGET, build_feature_row

MODEL_PATH = "models/best_model.pkl"
META_PATH = "models/best_model_meta.json"


@pytest.fixture(scope="module")
def model_and_meta():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        pytest.skip("models/best_model.pkl not found - run Part 3 of the notebook first")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def test_model_metadata_features_match_shared_feature_list(model_and_meta):
    _, meta = model_and_meta
    assert meta["features"] == FEATURE_COLS


def test_model_predicts_probabilities_in_valid_range(model_and_meta):
    model, meta = model_and_meta
    row = build_feature_row(298.0, 308.0, 1500, 40.0, 100, "M")
    proba = model.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_model_flags_a_known_failure_case(model_and_meta, features_df):
    """Sanity check with a real labeled failure row from the dataset -
    not a guarantee of catching every failure, but a model that scores this
    kind of row low would indicate something has gone very wrong."""
    model, meta = model_and_meta
    threshold = meta["metrics"]["Threshold"]

    failure_rows = features_df[features_df[TARGET] == 1]
    if len(failure_rows) == 0:
        pytest.skip("no failure rows available in features_df")

    X = failure_rows[FEATURE_COLS].head(20)
    probs = model.predict_proba(X)[:, 1]
    caught = (probs >= threshold).mean()

    # at the tuned (high-recall) threshold, most known failures should be flagged
    assert caught >= 0.5, f"only caught {caught:.0%} of a sample of known failures"


def test_model_metadata_threshold_is_not_the_naive_default(model_and_meta):
    """Guards against the bug this project actually hit once: deploying the
    champion model at the untuned 0.5 threshold instead of its business-tuned one."""
    _, meta = model_and_meta
    assert meta["metrics"]["Threshold"] != 0.5
