"""Candidate model tree and leakage-safe sklearn pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from hitpredict.constants import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    parameters: dict[str, Any]
    complexity: int
    hypothesis: str


def candidate_specs(profile: str) -> list[CandidateSpec]:
    quick = [
        CandidateSpec("dummy_prior", "dummy", {}, 0, "Audio features must beat the class-prior baseline."),
        CandidateSpec("logreg_c1", "logistic", {"C": 1.0}, 1, "A regularized linear boundary may generalize across years."),
        CandidateSpec("tree_depth6", "tree", {"max_depth": 6}, 2, "A shallow tree may capture a few nonlinear thresholds without memorizing."),
        CandidateSpec(
            "rf_250",
            "random_forest",
            {"n_estimators": 250, "max_depth": 14, "min_samples_leaf": 5},
            4,
            "Bagging may capture interactions while controlling single-tree variance.",
        ),
        CandidateSpec(
            "histgb_250",
            "hist_gradient_boosting",
            {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
            3,
            "Boosted shallow trees may fit nonlinear tabular effects efficiently.",
        ),
        CandidateSpec(
            "mlp_32_16",
            "mlp",
            {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
            5,
            "A small MLP may learn smooth feature interactions, but may add variance without more signal.",
        ),
    ]
    if profile == "quick":
        return quick
    if profile != "full":
        raise ValueError(f"Unknown training profile: {profile}")

    return quick + [
        CandidateSpec("logreg_c01", "logistic", {"C": 0.1}, 1, "Stronger L2 regularization may improve temporal generalization."),
        CandidateSpec("logreg_c10", "logistic", {"C": 10.0}, 1, "Weaker L2 regularization tests whether the linear baseline is underfit."),
        CandidateSpec("tree_depth4", "tree", {"max_depth": 4}, 2, "A smaller tree tests the bias-variance tradeoff."),
        CandidateSpec(
            "rf_wide",
            "random_forest",
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 10, "max_features": 0.7},
            4,
            "More randomized trees with larger leaves may reduce variance without a hard depth cap.",
        ),
        CandidateSpec(
            "histgb_deeper",
            "hist_gradient_boosting",
            {"max_iter": 350, "learning_rate": 0.04, "max_leaf_nodes": 31, "l2_regularization": 2.0},
            3,
            "A slightly richer boosted model may improve ranking while L2 controls complexity.",
        ),
        CandidateSpec(
            "mlp_64_32",
            "mlp",
            {"hidden_layer_sizes": (64, 32), "alpha": 0.003},
            5,
            "A wider MLP tests whether the smaller network underfits interactions.",
        ),
    ]


def make_preprocessor(features: list[str], scale_continuous: bool = True) -> ColumnTransformer:
    continuous = [feature for feature in features if feature in CONTINUOUS_FEATURES]
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    unknown = sorted(set(features) - set(continuous) - set(categorical))
    if unknown:
        raise ValueError(f"Features lack preprocessing classification: {unknown}")

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_continuous:
        numeric_steps.append(("scaler", RobustScaler()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", Pipeline(numeric_steps), continuous),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidate(spec: CandidateSpec, features: list[str], seed: int) -> Pipeline:
    scale = spec.family in {"logistic", "mlp", "dummy"}
    preprocessor = make_preprocessor(features, scale_continuous=scale)
    parameters = dict(spec.parameters)

    if spec.family == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif spec.family == "logistic":
        estimator = LogisticRegression(
            C=float(parameters.get("C", 1.0)),
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=seed,
        )
    elif spec.family == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=int(parameters.get("max_depth", 6)),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 20)),
            random_state=seed,
        )
    elif spec.family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(parameters.get("n_estimators", 250)),
            max_depth=parameters.get("max_depth", 14),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 5)),
            max_features=parameters.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=seed,
        )
    elif spec.family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=int(parameters.get("max_iter", 250)),
            learning_rate=float(parameters.get("learning_rate", 0.05)),
            max_leaf_nodes=int(parameters.get("max_leaf_nodes", 15)),
            l2_regularization=float(parameters.get("l2_regularization", 1.0)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    elif spec.family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(parameters.get("hidden_layer_sizes", (32, 16))),
            activation="relu",
            solver="adam",
            alpha=float(parameters.get("alpha", 0.001)),
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
