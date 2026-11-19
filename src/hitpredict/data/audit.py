"""Generate a reproducible pre-modelling data audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hitpredict.constants import BASE_AUDIO_FEATURES, TARGET_COLUMN
from hitpredict.data.validation import validate_modeling_data
from hitpredict.utils import utc_now_iso, write_json


FEATURE_AVAILABILITY = [
    ("explicit", "Yes", "Spotify track metadata", "Low", "Keep"),
    ("danceability", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("energy", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("key", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep; categorical"),
    ("loudness", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("mode", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep; categorical"),
    ("speechiness", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("acousticness", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("instrumentalness", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("liveness", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("valence", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("tempo", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep"),
    ("duration_ms", "Yes", "Spotify track metadata", "Low", "Keep"),
    ("time_signature", "Yes", "Spotify audio analysis/static dataset", "Low", "Keep; categorical"),
    ("prior_artist_hits", "Yes, at year precision", "Earlier Billboard charts", "Medium", "Ablation only; strict prior years"),
    ("year", "Yes", "Release metadata", "Temporal shortcut", "Split/analysis only; exclude from model"),
    ("popularity", "No for pre-success prediction", "Post-release Spotify plays", "Critical", "Exclude"),
    ("peak_position", "No", "Billboard outcome", "Direct target leakage", "Exclude"),
    ("weeks_on_chart", "No", "Billboard outcome", "Direct target leakage", "Exclude"),
    ("chart_score", "No", "Billboard outcome", "Direct target leakage", "Exclude"),
]


def _safe_describe(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    existing = [column for column in columns if column in frame]
    if not existing:
        return {}
    description = frame[existing].describe(percentiles=[0.01, 0.5, 0.99]).transpose()
    return {
        str(index): {str(key): (None if pd.isna(value) else float(value)) for key, value in row.items()}
        for index, row in description.iterrows()
    }


def run_data_audit(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["data"]["modeling_dataset"])
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}. Run `hitpredict prepare` first.")
    frame = pd.read_csv(path, low_memory=False)
    validation = validate_modeling_data(frame, config)

    numeric_features = BASE_AUDIO_FEATURES + ["prior_artist_hits"]
    missing = {column: int(frame[column].isna().sum()) for column in frame.columns}
    target_correlations: dict[str, float | None] = {}
    for feature in numeric_features:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        if numeric.nunique(dropna=True) < 2:
            target_correlations[feature] = None
            continue
        correlation = numeric.corr(frame[TARGET_COLUMN])
        target_correlations[feature] = None if pd.isna(correlation) else float(correlation)

    split_summary = {}
    for split, partition in frame.groupby("split", observed=True):
        split_summary[str(split)] = {
            "rows": int(len(partition)),
            "year_min": int(partition["year"].min()),
            "year_max": int(partition["year"].max()),
            "positive_count": int(partition[TARGET_COLUMN].sum()),
            "positive_prevalence": float(partition[TARGET_COLUMN].mean()),
        }

    report = {
        "created_at_utc": utc_now_iso(),
        "validation_gate": validation,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "target_distribution": {
            str(key): int(value) for key, value in frame[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "positive_prevalence": float(frame[TARGET_COLUMN].mean()),
        "split_summary": split_summary,
        "duplicate_track_ids": int(frame["id"].duplicated().sum()),
        "duplicate_song_groups": int(frame["song_group"].duplicated().sum()),
        "missing_values": missing,
        "numeric_summary": _safe_describe(frame, numeric_features),
        "feature_target_pearson_for_screening_only": target_correlations,
        "notes": [
            "Correlation is descriptive, not evidence of causality or permission to select on test data.",
            "Missing-value imputation and scaling are fitted only inside the training pipeline.",
            "The negative class means not matched to this Billboard history, not proven commercial failure.",
        ],
    }

    root = Path(config["_meta"]["project_root"])
    write_json(root / "reports/audits/data_audit.json", report)
    availability = pd.DataFrame(
        FEATURE_AVAILABILITY,
        columns=["feature", "available_before_prediction", "source", "leakage_risk", "decision"],
    )
    availability.to_csv(root / "reports/audits/feature_availability.csv", index=False)
    return report
