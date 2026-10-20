"""Evaluation helpers: threshold tuning and a consistent metric set."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def threshold_for_recall(y_true, proba, min_recall: float = 0.85) -> float:
    """Highest-precision threshold that still catches at least `min_recall` of escalations.

    Business reason: a missed escalation (false negative) leaves a customer issue unsolved,
    while a false alarm only costs an L2 engineer a few minutes.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    best_t, best_p = float(thresholds[0]), -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall and p > best_p:
            best_t, best_p = float(t), p
    return best_t


def evaluate(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
        "pr_auc": round(average_precision_score(y_true, proba), 3),
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
        "brier": round(brier_score_loss(y_true, proba), 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
