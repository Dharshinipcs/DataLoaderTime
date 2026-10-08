"""Persistent model artifacts for inference (Task 04).

The serving artifact is BUILT EXACTLY like the Task 03 evaluation artifact:
same chronological training window (chronological_holdout_split over the
training-eligible frame, earliest 262 rows), same candidate configuration
(AD-15: ExtraTrees, log1p), same encoding path via modeling.fit_candidate_pipeline.
No retraining decision is made here; the configuration is the approved candidate.

Persistence: joblib under backend/artifacts/. Loading is integrity-checked:
version/status/feature order/fitted state must match expectations exactly,
otherwise ArtifactError (fail closed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from dltime.config import PROJECT_ROOT
from dltime.modeling import (
    EXTRATREES_CANDIDATE,
    add_chronological_rank,
    build_preprocessor,
    chronological_holdout_split,
    fit_candidate_pipeline,
    matrix_encoding_columns,
)
from dltime.feature_assembly import TrainingFrame, TargetTransformer

ARTIFACT_DIR: Path = PROJECT_ROOT / "backend" / "artifacts"
# AD-32: the direct total is the REFERENCE/ROLLBACK model. The authoritative
# served total is the stage-aware total (stage_aware_serving.py); these
# constants describe the historical direct-total artifact only.
SERVING_MODEL_VERSION = "total-et-v3-candidate"
SERVING_MODEL_TYPE = "extra_trees_candidate"
SERVING_TARGET_MODE = "log1p"
SERVING_STATUS = "CANDIDATE_EVALUATED"  # lifecycle: RETIRED (reference) in the registry


class ArtifactError(RuntimeError):
    """Raised when artifact persistence/loading fails integrity checks."""


def serving_model_spec() -> dict:
    """Model spec for the approved candidate (same estimator config as Task 03)."""
    return {
        "estimator": lambda: ExtraTreesRegressor(**EXTRATREES_CANDIDATE),
        "needs_encoding": True,
        "needs_scaling": False,
    }


def _eligible_ordered(tf: TrainingFrame) -> pd.DataFrame:
    full = tf.frame
    eligible = full[full["is_training_eligible"].fillna(False)]
    return add_chronological_rank(eligible)


def build_serving_artifact(tf: TrainingFrame) -> dict:
    """Fit the approved candidate on the recorded chronological training window.

    Training window = chronological_holdout_split train side over the
    training-eligible rows of the AD-31 corrected source (341 eligible,
    256 train / 85 holdout). Same candidate configuration (AD-15: ExtraTrees,
    log1p), same encoding path via modeling.fit_candidate_pipeline.
    No retraining decision is made here; the configuration is the approved candidate.
    """
    from dltime.modeling import validate_features_for_model

    ordered = _eligible_ordered(tf)
    train, _test, split_info = chronological_holdout_split(ordered)
    validate_features_for_model(tf, tf.feature_names)

    feature_names = list(tf.feature_names)
    X_train = tf.X.loc[train.index, feature_names]
    y_train = tf.y_raw.loc[train.index].to_numpy()

    spec = serving_model_spec()
    est, pre, feature_out = fit_candidate_pipeline(
        spec, X_train, y_train, SERVING_TARGET_MODE, feature_names
    )
    if pre is None:  # the candidate has categorical features, so this never happens
        raise ArtifactError("Serving candidate requires a preprocessor; got none.")
    # Serving determinism: n_jobs is a runtime knob, not a modeling
    # hyperparameter; parallel float accumulation made repeated predictions
    # differ in the last ulps. n_jobs=1 keeps identical trees, identical
    # predictions bit-for-bit (hyperparameters recorded in metadata unchanged).
    est.n_jobs = 1

    categorical, numeric = matrix_encoding_columns(tf.X[feature_names])
    transformer = TargetTransformer(mode=SERVING_TARGET_MODE)

    # ---- metadata: identical schema to the Task 03 build_model_artifact ----
    spec_est = spec["estimator"]()
    metadata = {
        "model_version": SERVING_MODEL_VERSION,
        "status": SERVING_STATUS,  # not production/final by construction
        "model_type": SERVING_MODEL_TYPE,
        "hyperparameters": spec_est.get_params(),
        "target_transformation": SERVING_TARGET_MODE,
        "feature_names_in_order": feature_names,
        "categorical_features": list(categorical),
        "numeric_features": list(numeric),
        "encoding": "one_hot(handle_unknown=ignore) for categoricals; median impute for numerics",
        "training_row_count": int(split_info["train_rows"]),
        "training_period": {
            "first_start": split_info["train_first_start"],
            "last_start": split_info["train_last_start"],
        },
        "evaluation": {
            # AD-31 corrected-source regime (EXP-028 re-evaluation on the
            # corrected Loader_Executions.xlsx): computed chronological
            # evidence, not carried-over constants.
            "split_method": "expanding-window chronological CV + final chronological holdout",
            "config": {"holdout_fraction": 0.25, "min_holdout": 25, "n_folds": 5},
            "pooled_oof": {
                "mae": 469.2,
                "median_ae": 3.19,
                "rmse": 2146.0,
                "within_30pct": 25.15,
                "within_2x": 50.88,
            },
            "holdout": {
                "mae": 729.52,
                "median_ae": 17.87,
                "within_30pct": 7.06,
            },
        },
        "evaluation_run": "EXP-028 (AD-31 corrected-source re-evaluation, chronological)",
        # ---- fitted fields (additive; inference uses these) ----
        "fitted": {
            "feature_out_count": len(feature_out),
            "sklearn_version": __import__("sklearn").__version__,
            "n_training_rows": int(len(train)),
        },
    }

    return {
        "metadata": metadata,
        "estimator": est,
        "preprocessor": pre,
        "transformer": transformer,
    }


def save_serving_artifact(artifact: dict, path: Path | None = None) -> Path:
    """Persist the fitted serving artifact (pickle; objects are sklearn/pandas)."""
    path = path or ARTIFACT_DIR / f"{SERVING_MODEL_VERSION}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib  # sklearn ships joblib

        joblib.dump(artifact, path)
    except Exception as exc:  # pragma: no cover - persistence failure
        raise ArtifactError(f"Failed to save artifact to {path}: {exc}") from exc
    return path


def load_serving_artifact(path: Path | None = None) -> dict:
    """Load a serving artifact with strict integrity checks (fail closed)."""
    path = path or ARTIFACT_DIR / f"{SERVING_MODEL_VERSION}.pkl"
    if not path.exists():
        raise ArtifactError(f"Serving artifact not found: {path}")
    try:
        import joblib

        artifact = joblib.load(path)
    except Exception as exc:
        raise ArtifactError(f"Failed to load artifact from {path}: {exc}") from exc

    meta = artifact.get("metadata", {})
    for key, expected in (
        ("model_version", SERVING_MODEL_VERSION),
        ("status", SERVING_STATUS),
        ("model_type", SERVING_MODEL_TYPE),
        ("target_transformation", SERVING_TARGET_MODE),
    ):
        if meta.get(key) != expected:
            raise ArtifactError(
                f"Artifact mismatch on '{key}': got {meta.get(key)!r}, expected {expected!r}"
            )
    if meta.get("status") == "PRODUCTION":
        raise ArtifactError("Refusing to load a PRODUCTION-status artifact (not approved).")
    fitted = meta.get("fitted")
    if not isinstance(fitted, dict) or not fitted.get("feature_out_count"):
        raise ArtifactError("Artifact metadata lacks fitted-state fields.")
    for key in ("estimator", "preprocessor", "transformer"):
        if key not in artifact:
            raise ArtifactError(f"Artifact missing fitted component '{key}'.")
    # Feature-order authority: the fitted preprocessor must carry the exact
    # input feature order recorded in metadata (the estimator is fitted on
    # the transformed array, so it has no feature_names_in_ by design).
    pre = artifact["preprocessor"]
    names_in = getattr(pre, "feature_names_in_", None)
    pre_names = [str(x) for x in names_in] if names_in is not None else []
    if pre_names != meta.get("feature_names_in_order", []):
        raise ArtifactError(
            "Fitted preprocessor feature order does not match metadata "
            "feature_names_in_order; refusing to serve."
        )
    est = artifact["estimator"]
    # Required API surface (additive checks; no version drift allowed silently):
    for method in ("predict", "get_params"):
        if not callable(getattr(est, method, None)):
            raise ArtifactError(f"Estimator lacks required method '{method}'.")
    return artifact


def predict_with_artifact(artifact: dict, X: pd.DataFrame) -> float:
    """One-row preprocessed predict + expm1 back-transform; returns raw seconds."""
    meta = artifact["metadata"]
    feature_names = meta["feature_names_in_order"]
    if list(X.columns) != feature_names:
        raise ArtifactError(
            "Feature order mismatch: artifact expects "
            f"{feature_names}, got {list(X.columns)}"
        )
    if len(X) != 1:
        raise ArtifactError("predict_with_artifact expects exactly one row.")
    pre = artifact["preprocessor"]
    est = artifact["estimator"]
    # Same sentinel treatment the preprocessor was fitted with (AD-14).
    from dltime.modeling import transform_with_preprocessor

    Xp = transform_with_preprocessor(
        pre, X, meta["categorical_features"], meta["numeric_features"]
    )
    pred_t = float(est.predict(Xp)[0])
    return float(artifact["transformer"].inverse_transform([pred_t])[0])
