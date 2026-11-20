"""End-to-end baseline comparison, selection, calibration, and final evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance

from hitpredict.config import ensure_output_directories
from hitpredict.constants import CATEGORICAL_FEATURES, TARGET_COLUMN
from hitpredict.data.validation import validate_modeling_data
from hitpredict.metrics import (
    balanced_evaluation_subset,
    choose_f1_threshold,
    classification_metrics,
    stratified_bootstrap_intervals,
)
from hitpredict.models import CandidateSpec, build_candidate, candidate_specs
from hitpredict.reporting import create_error_analysis, save_diagnostic_plots
from hitpredict.utils import seed_everything, utc_now_iso, write_json


def sample_training_negatives(
    frame: pd.DataFrame,
    negatives_per_positive: int | None,
    seed: int,
) -> pd.DataFrame:
    """Year-match negatives for compute; later calibration restores real prevalence."""
    if negatives_per_positive is None:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for year, group in frame.groupby("year", sort=True):
        positive = group[group[TARGET_COLUMN].eq(1)]
        negative = group[group[TARGET_COLUMN].eq(0)]
        if positive.empty:
            continue
        count = min(len(negative), len(positive) * int(negatives_per_positive))
        sampled_negative = negative.sample(n=count, random_state=seed + int(year))
        pieces.extend([positive, sampled_negative])
    if not pieces:
        raise ValueError("Training sampler produced no rows; check match coverage and split years")
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _positive_probability(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(frame)
    classes = np.asarray(estimator.classes_)
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(probabilities[:, positive_index], dtype=float)


def _reference_values(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    references: dict[str, float] = {}
    for feature in features:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        if feature in CATEGORICAL_FEATURES:
            mode = numeric.mode(dropna=True)
            references[feature] = float(mode.iloc[0]) if not mode.empty else 0.0
        else:
            references[feature] = float(numeric.median())
    return references


def _select_record(records: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    viable = [record for record in records if record.get("average_precision") is not None]
    if not viable:
        raise RuntimeError("No candidate produced a valid average-precision score")
    best_score = max(float(record["average_precision"]) for record in viable)
    near_best = [record for record in viable if best_score - float(record["average_precision"]) <= tolerance]
    # Scientific parsimony: within a pre-declared tolerance, choose lower model
    # complexity and fewer inputs rather than rewarding noise in one validation period.
    return min(
        near_best,
        key=lambda record: (
            int(record["complexity"]),
            int(record["feature_count"]),
            float(record["log_loss"]),
        ),
    )


def _run_permutation_importance(
    model: Any,
    frame: pd.DataFrame,
    features: list[str],
    repeats: int,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) > max_rows:
        # Preserve the observed class ratio; this is post-selection analysis only.
        sample = frame.groupby(TARGET_COLUMN, group_keys=False).apply(
            lambda group: group.sample(
                n=max(1, round(max_rows * len(group) / len(frame))),
                random_state=seed + int(group.name),
            ),
            include_groups=False,
        )
        # include_groups=False removes the target in recent pandas, restore by index.
        sample = frame.loc[sample.index]
    else:
        sample = frame
    importance = permutation_importance(
        model,
        sample[features],
        sample[TARGET_COLUMN],
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
            "analysis_set": "post-selection temporal test sample; not used for model choice",
        }
    ).sort_values("importance_mean", ascending=False)


def train_and_evaluate(config: dict[str, Any], profile: str = "quick") -> dict[str, Any]:
    ensure_output_directories(config)
    seed = int(config["project"]["random_seed"])
    seed_everything(seed)
    root = Path(config["_meta"]["project_root"])
    data_path = Path(config["data"]["modeling_dataset"])
    if not data_path.exists():
        raise FileNotFoundError(f"Prepared data missing: {data_path}. Run `hitpredict prepare` first.")

    frame = pd.read_csv(data_path, low_memory=False)
    validate_modeling_data(frame, config)
    partitions = {name: part.reset_index(drop=True) for name, part in frame.groupby("split")}
    train_raw = partitions["train"]
    validation = partitions["validation"]
    calibration = partitions["calibration"]
    test = partitions["test"]
    ratio = config["data"].get("train_negatives_per_positive")
    train_sample = sample_training_negatives(train_raw, ratio, seed)

    specs = candidate_specs(profile)
    records: list[dict[str, Any]] = []
    record_specs: dict[tuple[str, str], CandidateSpec] = {}
    experiment_index = 1
    for feature_set, features in config["features"]["sets"].items():
        features = list(features)
        for spec in specs:
            experiment_id = f"E{experiment_index:02d}"
            experiment_index += 1
            model = build_candidate(spec, features, seed)
            started = time.perf_counter()
            model.fit(train_sample[features], train_sample[TARGET_COLUMN])
            elapsed = time.perf_counter() - started
            probability = _positive_probability(model, validation[features])
            metrics = classification_metrics(validation[TARGET_COLUMN].to_numpy(), probability, 0.5)
            record = {
                "experiment_id": experiment_id,
                "date_utc": utc_now_iso(),
                "question": f"Can {spec.family} rank future Billboard hits using {feature_set}?",
                "hypothesis": spec.hypothesis,
                "dataset_split": "fit=train through 2014; evaluate=2015-2016 validation",
                "feature_set": feature_set,
                "feature_count": len(features),
                "candidate": spec.name,
                "model_family": spec.family,
                "parameters": json.dumps(spec.parameters, sort_keys=True),
                "complexity": spec.complexity,
                "train_rows": int(len(train_sample)),
                "validation_rows": int(len(validation)),
                "fit_seconds": float(elapsed),
                **metrics,
                "interpretation": "Pending comparative selection; test set remains untouched.",
            }
            records.append(record)
            record_specs[(feature_set, spec.name)] = spec

    comparison = pd.DataFrame(records)
    comparison.to_csv(root / "reports/metrics/model_comparison_validation.csv", index=False)
    comparison.to_csv(root / "reports/experiments/experiment_ledger.csv", index=False)

    tolerance = float(config["training"].get("selection_tolerance", 0.005))
    selected = _select_record(records, tolerance)
    selected_features = list(config["features"]["sets"][selected["feature_set"]])
    selected_spec = record_specs[(selected["feature_set"], selected["candidate"])]

    # Refit the selected design on all pre-2017 development years. Sampling is
    # repeated deterministically, then probabilities are calibrated on the full,
    # naturally imbalanced 2017 catalog. The 2018-2020 test is still untouched.
    development = pd.concat([train_raw, validation], ignore_index=True)
    development_sample = sample_training_negatives(development, ratio, seed)
    base_model = build_candidate(selected_spec, selected_features, seed)
    base_model.fit(development_sample[selected_features], development_sample[TARGET_COLUMN])

    calibration_method = str(config["training"].get("calibration_method", "sigmoid"))
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model), method=calibration_method, n_jobs=-1
    )
    calibrated_model.fit(calibration[selected_features], calibration[TARGET_COLUMN])
    calibration_probability = _positive_probability(calibrated_model, calibration[selected_features])
    threshold, calibration_best_f1 = choose_f1_threshold(
        calibration[TARGET_COLUMN].to_numpy(), calibration_probability
    )
    calibration_metrics = classification_metrics(
        calibration[TARGET_COLUMN].to_numpy(), calibration_probability, threshold
    )

    # Single final test reveal.
    test_probability = _positive_probability(calibrated_model, test[selected_features])
    test_metrics = classification_metrics(test[TARGET_COLUMN].to_numpy(), test_probability, threshold)
    test_with_probability = test.copy()
    test_with_probability["_probability"] = test_probability
    balanced_test = balanced_evaluation_subset(
        test_with_probability,
        TARGET_COLUMN,
        int(config["data"].get("balanced_report_negatives_per_positive", 1)),
        seed,
    )
    balanced_metrics = classification_metrics(
        balanced_test[TARGET_COLUMN].to_numpy(), balanced_test["_probability"].to_numpy(), threshold
    )
    intervals = stratified_bootstrap_intervals(
        test[TARGET_COLUMN].to_numpy(),
        test_probability,
        threshold,
        int(config["training"].get("bootstrap_iterations", 100)),
        seed,
    )

    references = _reference_values(development_sample, selected_features)
    artifact = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "model": calibrated_model,
        "threshold": float(threshold),
        "feature_set": selected["feature_set"],
        "feature_names": selected_features,
        "reference_values": references,
        "candidate": selected["candidate"],
        "model_family": selected["model_family"],
        "parameters": selected_spec.parameters,
        "data_periods": {
            "base_fit": "1990-2016 (year-matched negative sampling)",
            "calibration_and_threshold": "2017 full catalog",
            "untouched_test": "2018-2020 full catalog",
        },
        "test_metrics": test_metrics,
        "limitations": [
            "Labels depend on exact title/artist linkage and contain residual false negatives.",
            "Probabilities are calibrated to the 2017 catalog prevalence and may drift.",
            "Spotify audio-feature API access is restricted for new apps; demo inputs are manual.",
            "The model estimates association, not causal musical quality or guaranteed success.",
        ],
    }
    artifact_path = root / "artifacts/final_model.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    importance = _run_permutation_importance(
        calibrated_model,
        test,
        selected_features,
        int(config["training"].get("permutation_repeats", 5)),
        int(config["training"].get("permutation_max_rows", 10_000)),
        seed,
    )
    importance.to_csv(root / "reports/metrics/permutation_importance.csv", index=False)
    error_summary = create_error_analysis(test, test_probability, threshold, root / "reports/metrics")
    save_diagnostic_plots(
        test[TARGET_COLUMN].to_numpy(), test_probability, threshold, root / "reports/figures"
    )

    final_report = {
        "created_at_utc": utc_now_iso(),
        "selection_rule": (
            "Highest validation average precision; within tolerance choose lower complexity and fewer features."
        ),
        "selected_validation_record": selected,
        "calibration_best_f1": calibration_best_f1,
        "calibration_metrics": calibration_metrics,
        "test_metrics_realistic_catalog_prevalence": test_metrics,
        "test_metrics_balanced_for_hitpredict_comparison_only": balanced_metrics,
        "test_bootstrap_intervals": intervals,
        "error_analysis": error_summary,
        "artifact": str(artifact_path),
        "test_set_reveal_count": 1,
    }
    write_json(root / "reports/metrics/final_evaluation.json", final_report)
    write_json(
        root / "reports/metrics/model_selection.json",
        {
            "selected": selected,
            "alternatives": sorted(records, key=lambda row: row["average_precision"], reverse=True),
            "reason": "Pre-declared metric and parsimony tolerance; no test metrics used in selection.",
        },
    )
    return final_report
