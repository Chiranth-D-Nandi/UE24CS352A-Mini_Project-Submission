"""Plots and post-selection error/segment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from hitpredict.constants import TARGET_COLUMN
from hitpredict.metrics import classification_metrics


def save_diagnostic_plots(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction = (probability >= threshold).astype(int)

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, prediction, labels=[0, 1], cmap="Blues", ax=axis)
    axis.set_title(f"Untouched temporal test set (threshold={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, probability, ax=axis, name="Final model")
    axis.axhline(float(np.mean(y_true)), color="grey", linestyle="--", label="Prevalence")
    axis.legend()
    axis.set_title("Precision–recall curve: temporal test")
    fig.tight_layout()
    fig.savefig(output / "test_precision_recall_curve.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    RocCurveDisplay.from_predictions(y_true, probability, ax=axis, name="Final model")
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--")
    axis.set_title("ROC curve: temporal test")
    fig.tight_layout()
    fig.savefig(output / "test_roc_curve.png", dpi=180)
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot(mean_predicted, fraction_positive, marker="o", label="Final model")
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed hit rate", title="Calibration: temporal test")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "test_calibration_curve.png", dpi=180)
    plt.close(fig)


def create_error_analysis(
    test_frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = test_frame.copy()
    result["predicted_probability"] = probability
    result["prediction"] = (probability >= threshold).astype(int)
    result["error_type"] = "correct"
    result.loc[(result[TARGET_COLUMN] == 0) & (result["prediction"] == 1), "error_type"] = "false_positive"
    result.loc[(result[TARGET_COLUMN] == 1) & (result["prediction"] == 0), "error_type"] = "false_negative"
    result.to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")

    columns = [
        "id",
        "name",
        "artists",
        "year",
        TARGET_COLUMN,
        "predicted_probability",
        "prediction",
        "prior_artist_hits",
    ]
    false_positive = result[result["error_type"].eq("false_positive")].sort_values(
        "predicted_probability", ascending=False
    )
    false_negative = result[result["error_type"].eq("false_negative")].sort_values(
        "predicted_probability", ascending=True
    )
    false_positive[columns].head(200).to_csv(output / "false_positives_top200.csv", index=False)
    false_negative[columns].head(200).to_csv(output / "false_negatives_top200.csv", index=False)

    segment_rows: list[dict[str, Any]] = []
    for year, segment in result.groupby("year"):
        if segment[TARGET_COLUMN].nunique() < 2:
            continue
        metrics = classification_metrics(
            segment[TARGET_COLUMN].to_numpy(), segment["predicted_probability"].to_numpy(), threshold
        )
        segment_rows.append({"segment": "release_year", "value": int(year), "rows": len(segment), **metrics})

    history_segment = np.where(result["prior_artist_hits"].fillna(0).gt(0), "prior_hit", "no_prior_hit")
    for label in ["no_prior_hit", "prior_hit"]:
        segment = result[history_segment == label]
        if segment.empty or segment[TARGET_COLUMN].nunique() < 2:
            continue
        metrics = classification_metrics(
            segment[TARGET_COLUMN].to_numpy(), segment["predicted_probability"].to_numpy(), threshold
        )
        segment_rows.append({"segment": "artist_history", "value": label, "rows": len(segment), **metrics})

    segments = pd.DataFrame(segment_rows)
    segments.to_csv(output / "segment_metrics.csv", index=False)
    return {
        "false_positives": int(len(false_positive)),
        "false_negatives": int(len(false_negative)),
        "segment_rows": int(len(segments)),
    }
