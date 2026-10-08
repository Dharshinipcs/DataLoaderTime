"""Stage-model serving helpers (AD-31; amended AD-31a).

Builds the five ML stage predictors for the serving path from the SAME
chronological stage frames used in the stage-model evaluations. Selected kinds
come from the Model Laboratory stage-selection records
(EXP-STAGE-<STAGE>-SELECTION series): every operational stage serves a real
fitted ML model; pre_load is EXPERIMENTAL_SMALL_SAMPLE (n=19 — real fitted
model, disclosed small-sample limitation); the historical median baseline is
a registered non-ML reference only and is never the served stage model.

AD-32: the fitted stage predictors are PERSISTED as one artifact
(artifacts/stage-models-ad34.pkl) by the serving build script and loaded
verbatim at serving time when present + version-matched; fitting from source
is the documented fallback (fresh checkout). No lifecycle action is implied:
these are EXPERIMENTAL candidate models used for the Seven-Stage Execution
View and as stage inputs to the authoritative stage-aware total model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_SERVING_VERSION = "stage-serving-v2"
STAGE_MODELS_ARTIFACT_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "artifacts" / "stage-models-ad34.pkl"
)


class _StagePredictor:
    """One stage's fitted predictor exposing predict_seconds(context rows)."""

    def __init__(self, stage: str, kind: str, model, feature_names: list[str], n_train: int, medians: pd.Series):
        self.stage = stage
        self.kind = kind
        self.model = model
        self.feature_names = feature_names
        self.n_train = n_train
        self._medians = medians  # training medians for numeric NA filling

    def predict_seconds(self, ctx_row: pd.DataFrame) -> float:
        from dltime.feature_assembly import TargetTransformer

        X = ctx_row[self.feature_names].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self._medians.get(col, 0.0))
        Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
        Xenc = Xenc.reindex(columns=self._enc_columns, fill_value=0)
        if self.kind == "historical_median_baseline":
            pred_log = np.full(len(Xenc), float(self._baseline_log_median), dtype=float)
        else:
            pred_log = np.asarray(self.model.predict(Xenc), dtype=float)
        return float(TargetTransformer(mode="log1p").inverse_transform(pred_log)[0])


def build_stage_serving_models(bundle=None) -> dict[str, object]:
    """Fit the implemented stage models; returns {stage: predictor}.

    2026-09-29 source replacement: a stage with ZERO eligible observations
    (staging on the replaced file — the export no longer carries staging
    fields) is SKIPPED, never fabricated. Consumers already treat a missing
    predictor as AvailabilityState.ML_UNAVAILABLE and the deterministic
    sum refuses the total while any operational stage is unavailable
    (fail-closed, AD-31 §15 / sum_total_serving).
    """
    from dltime.stage_models import (
        build_stage_frame,
        _make_estimator,
        STAGE_MODEL_SERVING_KINDS,
    )
    from dltime.sources import load_all_sources

    bundle = bundle or load_all_sources()
    models: dict[str, object] = {}
    for stage, serving in STAGE_MODEL_SERVING_KINDS.items():
        kind = serving["model_kind"]
        if kind is None:
            continue  # NOT_IMPLEMENTED (documented in stage_intel.STAGE_MODEL_SERVING)
        frame, feats, diag = build_stage_frame(stage=stage, bundle=bundle)
        eligible = frame[frame["stage_eligible"].astype(bool)]
        if len(eligible) == 0:
            # Honest skip: no eligible observations on this source revision.
            # The stage is left ABSENT from the dict -> ML_UNAVAILABLE downstream.
            continue
        X = eligible[feats].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        medians = X.median(numeric_only=True)
        Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
        from dltime.feature_assembly import TargetTransformer

        y_log = TargetTransformer(mode="log1p").transform(
            pd.to_numeric(eligible[f"stage_{stage}_sec"], errors="coerce")
        )
        if kind == "historical_median_baseline":
            model = None  # baseline predicts the training median directly
        else:
            model = _make_estimator(kind)
            model.fit(Xenc, y_log)
        predictor = _StagePredictor(stage, kind, model, feats, diag["stage_eligible_rows"], medians)
        predictor._enc_columns = Xenc.columns
        predictor._baseline_log_median = float(np.median(y_log))
        models[stage] = predictor
    return models


def save_stage_serving_models(models: dict[str, object], path=None) -> "Path":
    """Persist the fitted stage predictors (one artifact, version-stamped)."""
    from pathlib import Path

    import joblib

    path = path or STAGE_MODELS_ARTIFACT_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "serving_version": STAGE_SERVING_VERSION,
            "stages": list(models.keys()),
            "models": models,
        },
        path,
    )
    return path


def load_stage_serving_models(bundle=None) -> dict[str, object]:
    """Serving entry point: load the persisted stage-model artifact when it
    exists and is version-matched; otherwise fit from source (fresh-checkout
    fallback, documented) so the serving path never fabricates availability.
    """
    if STAGE_MODELS_ARTIFACT_PATH.exists():
        import joblib

        try:
            blob = joblib.load(STAGE_MODELS_ARTIFACT_PATH)
            if blob.get("serving_version") == STAGE_SERVING_VERSION and blob.get("models"):
                return blob["models"]
        except Exception:
            pass  # fall through to deterministic refit
    return build_stage_serving_models(bundle=bundle)
