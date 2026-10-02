"""
Shared model-evaluation helpers for the Predictive Maintenance project.

Used by the training notebook (ML, MLP, and Autoencoder sections all score
themselves the same way) and by the test suite, so "the metrics" always mean
the same computation everywhere.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def compute_metrics(name, y_true, y_score, threshold=0.5):
    """
    Turn a set of predicted scores (probabilities, or an anomaly score -
    anything where 'higher = more likely to be a failure') into one
    comparable metrics row.
    """
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    fn_rate = ((np.asarray(y_true) == 1) & (y_pred == 0)).sum() / max(np.asarray(y_true).sum(), 1)

    row = dict(
        Model=name,
        Threshold=round(float(threshold), 4),
        Precision=round(float(precision), 4),
        Recall=round(float(recall), 4),
        F1=round(float(f1), 4),
        PR_AUC=round(float(pr_auc), 4),
        ROC_AUC=round(float(roc_auc), 4),
        FN_Rate=round(float(fn_rate), 4),
    )
    return row, y_pred


def find_optimal_threshold(y_true, y_score, min_precision=0.30):
    """
    Find the score threshold that maximises recall subject to
    precision >= min_precision. Falls back to the F1-optimal threshold
    if that constraint can't be met.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)

    mask = precisions[:-1] >= min_precision
    if mask.any():
        candidate_recalls = np.where(mask, recalls[:-1], -1)
        best_idx = np.argmax(candidate_recalls)
    else:
        best_idx = np.argmax(f1s)

    return thresholds[best_idx], precisions[best_idx], recalls[best_idx]


def risk_tier(probability, threshold):
    """Low / Medium / High risk banding - shared by the app and reporting code."""
    if probability < 0.30:
        return "Low", "Continue normal operation"
    elif probability < threshold:
        return "Medium", "Schedule an inspection soon"
    else:
        return "High", "Immediate inspection required"
