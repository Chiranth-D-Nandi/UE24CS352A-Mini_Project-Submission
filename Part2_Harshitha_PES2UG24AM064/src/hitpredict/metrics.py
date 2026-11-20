"""Metrics fixed before model selection, thresholding, and uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    both_classes = np.unique(y_true).size == 2
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, probability)) if both_classes else None,
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_prevalence": float(y_true.mean()),
        "predicted_positive_rate": float(prediction.mean()),
    }


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def balanced_evaluation_subset(
    frame: pd.DataFrame,
    target: str,
    negatives_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    positive = frame[frame[target].eq(1)]
    negative = frame[frame[target].eq(0)]
    requested = min(len(negative), len(positive) * negatives_per_positive)
    negative_sample = negative.sample(n=requested, random_state=seed) if requested else negative.iloc[0:0]
    return pd.concat([positive, negative_sample], ignore_index=False).sample(frac=1, random_state=seed)


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile intervals; used for uncertainty, never model selection."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    if not len(positive_indices) or not len(negative_indices) or iterations <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"average_precision": [], "roc_auc": [], "f1": []}
    for _ in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            ]
        )
        y_sample = y_true[indices]
        p_sample = probability[indices]
        values["average_precision"].append(float(average_precision_score(y_sample, p_sample)))
        values["roc_auc"].append(float(roc_auc_score(y_sample, p_sample)))
        values["f1"].append(float(f1_score(y_sample, p_sample >= threshold, zero_division=0)))
    return {
        metric: {
            "mean": float(np.mean(samples)),
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
            "iterations": int(iterations),
        }
        for metric, samples in values.items()
    }
