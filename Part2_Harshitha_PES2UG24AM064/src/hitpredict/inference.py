"""Single inference contract shared by tests and the reviewer-facing app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hitpredict.constants import VALID_RANGES


class InputValidationError(ValueError):
    pass


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"model", "threshold", "feature_names", "reference_values", "schema_version"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing fields: {sorted(missing)}")
    if artifact["schema_version"] != 1:
        raise ValueError(f"Unsupported artifact schema: {artifact['schema_version']}")
    return artifact


def validate_record(record: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    missing = [feature for feature in feature_names if feature not in record]
    if missing:
        raise InputValidationError(f"Missing required features: {missing}")
    cleaned: dict[str, float] = {}
    for feature in feature_names:
        try:
            value = float(record[feature])
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{feature} must be numeric") from exc
        if not np.isfinite(value):
            raise InputValidationError(f"{feature} must be finite")
        if feature in VALID_RANGES:
            minimum, maximum = VALID_RANGES[feature]
            if not minimum <= value <= maximum:
                raise InputValidationError(
                    f"{feature}={value} is outside the supported range [{minimum}, {maximum}]"
                )
        cleaned[feature] = value
    return cleaned


def predict_record(artifact: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    features = list(artifact["feature_names"])
    cleaned = validate_record(record, features)
    frame = pd.DataFrame([cleaned], columns=features)
    model = artifact["model"]
    positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    probability = float(model.predict_proba(frame)[0, positive_index])
    threshold = float(artifact["threshold"])
    return {
        "probability": probability,
        "prediction": int(probability >= threshold),
        "threshold": threshold,
    }


def local_sensitivity(artifact: dict[str, Any], record: dict[str, Any]) -> pd.DataFrame:
    """One-at-a-time reference replacement; an association, not a causal explanation."""
    features = list(artifact["feature_names"])
    cleaned = validate_record(record, features)
    base = predict_record(artifact, cleaned)["probability"]
    rows = []
    for feature in features:
        counterfactual = dict(cleaned)
        counterfactual[feature] = float(artifact["reference_values"][feature])
        probability = predict_record(artifact, counterfactual)["probability"]
        rows.append(
            {
                "feature": feature,
                "input_value": cleaned[feature],
                "reference_value": counterfactual[feature],
                "probability_change_from_reference": base - probability,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "probability_change_from_reference", key=lambda series: series.abs(), ascending=False
    )
