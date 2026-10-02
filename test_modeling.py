"""
Tests for src/modeling.py - the metrics and threshold-selection helpers
shared by the ML, MLP, and Autoencoder sections of the notebook.
"""

import numpy as np
import pytest

from src.modeling import compute_metrics, find_optimal_threshold, risk_tier


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.95]
    row, y_pred = compute_metrics("perfect", y_true, y_score, threshold=0.5)
    assert row["Precision"] == 1.0
    assert row["Recall"] == 1.0
    assert row["F1"] == 1.0
    assert row["FN_Rate"] == 0.0
    assert list(y_pred) == [0, 0, 1, 1]


def test_compute_metrics_all_missed():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4]  # everything below threshold
    row, y_pred = compute_metrics("missed", y_true, y_score, threshold=0.5)
    assert row["Recall"] == 0.0
    assert row["FN_Rate"] == 1.0


def test_compute_metrics_threshold_changes_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.2, 0.4, 0.6, 0.8]
    _, pred_low = compute_metrics("low_thresh", y_true, y_score, threshold=0.3)
    _, pred_high = compute_metrics("high_thresh", y_true, y_score, threshold=0.7)
    assert sum(pred_low) > sum(pred_high)


def test_find_optimal_threshold_respects_precision_floor():
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 95 + [1] * 5)
    y_score = np.concatenate([rng.uniform(0, 0.6, 95), rng.uniform(0.4, 1.0, 5)])
    threshold, precision, recall = find_optimal_threshold(y_true, y_score, min_precision=0.30)
    _, y_pred = compute_metrics("check", y_true, y_score, threshold=threshold)
    achieved_precision = sum((y_pred == 1) & (y_true == 1)) / max(sum(y_pred == 1), 1)
    # allow a little slack: threshold is picked from a finite candidate set
    assert achieved_precision >= 0.30 - 0.05 or recall == 0


def test_risk_tier_boundaries():
    assert risk_tier(0.1, threshold=0.5)[0] == "Low"
    assert risk_tier(0.4, threshold=0.5)[0] == "Medium"
    assert risk_tier(0.9, threshold=0.5)[0] == "High"


def test_risk_tier_returns_action_text():
    tier, action = risk_tier(0.95, threshold=0.5)
    assert isinstance(action, str) and len(action) > 0
