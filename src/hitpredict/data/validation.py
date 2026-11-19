"""Hard validation gates that run before any model sees the data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hitpredict.constants import FORBIDDEN_MODEL_FEATURES, TARGET_COLUMN, VALID_RANGES


class DataValidationError(ValueError):
    pass


def validate_feature_sets(config: dict[str, Any], columns: list[str]) -> None:
    for name, features in config["features"]["sets"].items():
        missing = sorted(set(features) - set(columns))
        forbidden = sorted(set(features) & FORBIDDEN_MODEL_FEATURES)
        if missing:
            raise DataValidationError(f"Feature set {name!r} is missing columns: {missing}")
        if forbidden:
            raise DataValidationError(f"Feature set {name!r} contains leakage-risk fields: {forbidden}")


def validate_modeling_data(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if frame.empty:
        errors.append("modelling dataset is empty")
    if TARGET_COLUMN not in frame:
        errors.append(f"missing target column {TARGET_COLUMN}")
    elif not set(frame[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        errors.append("target must contain only 0/1")

    if "song_group" not in frame:
        errors.append("missing independent entity key song_group")
    elif frame["song_group"].duplicated().any():
        errors.append("duplicate song_group values remain; entity leakage is possible")

    expected_splits = {"train", "validation", "calibration", "test"}
    actual_splits = set(frame.get("split", pd.Series(dtype=str)).dropna().unique())
    if actual_splits != expected_splits:
        errors.append(f"expected splits {sorted(expected_splits)}, found {sorted(actual_splits)}")

    data = config["data"]
    year_rules = {
        "train": (int(data["min_release_year"]), int(data["train_end_year"])),
        "validation": (int(data["train_end_year"]) + 1, int(data["selection_end_year"])),
        "calibration": (int(data["calibration_year"]), int(data["calibration_year"])),
        "test": (int(data["calibration_year"]) + 1, int(data["test_end_year"])),
    }
    if "year" in frame and "split" in frame:
        for split, (minimum, maximum) in year_rules.items():
            values = frame.loc[frame["split"].eq(split), "year"]
            if not values.empty and not values.between(minimum, maximum).all():
                errors.append(f"{split} contains years outside [{minimum}, {maximum}]")

    range_violations: dict[str, int] = {}
    for feature, (lower_bound, upper_bound) in VALID_RANGES.items():
        if feature not in frame:
            continue
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        count = int((numeric.notna() & ~numeric.between(lower_bound, upper_bound)).sum())
        if count:
            range_violations[feature] = count

    try:
        validate_feature_sets(config, list(frame.columns))
    except DataValidationError as exc:
        errors.append(str(exc))

    # Each evaluation partition needs both classes for ROC/PR metrics.
    class_counts: dict[str, dict[str, int]] = {}
    if {"split", TARGET_COLUMN}.issubset(frame.columns):
        for split, partition in frame.groupby("split"):
            counts = partition[TARGET_COLUMN].value_counts().to_dict()
            class_counts[str(split)] = {str(key): int(value) for key, value in counts.items()}
            if set(counts) != {0, 1}:
                errors.append(f"split {split!r} does not contain both classes")

    if errors:
        raise DataValidationError("; ".join(errors))
    return {
        "rows": int(len(frame)),
        "unique_song_groups": int(frame["song_group"].nunique()),
        "class_counts": class_counts,
        "range_violations_for_audit": range_violations,
    }
