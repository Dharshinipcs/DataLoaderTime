"""Conformal calibration for the SUM-OF-FIVE-STAGES total policy.

The Task 07 uncertainty layer needs a residual-quantile table for the ETA
actually served. Under the final product decision the served ETA is the
deterministic sum of the five operational stage predictions, so the
calibration residuals must be the errors OF THAT SUM — not of the retired
stage-aware total model (whose dedicated artifact
unc-log1p-conformal-v1-stage-aware-ad34.json stays on disk for traceability
but no longer describes the served number).

Protocol (leakage-safe, identical spirit to the Task 07 / AD-32 calibrations):
  1. chronological TRAIN/HOLDOUT split of the stage-frame total population
     (EXACTLY the EXP-036 clean-protocol split: same population, same
     boundary);
  2. the five operational stage models are fit ONLY on TRAIN-period eligible
     rows, then frozen;
  3. frozen stage models predict the HOLDOUT rows (pre-execution context +
     strictly-prior stage-specific history only — as-of semantics);
  4. the calibration residuals are |log1p(actual_total) - log1p(sum)|
     on that holdout, against the SAME authoritative actual total the
     EXP-036 evidence used (total_timetaken_in_min * 60; legacy
     timetaken_in_sec is corrupted and never used);
  5. empirical quantiles (80/90, method='higher', with the Task 07
     MIN_QUANTILE_FLOOR) are persisted as a versioned JSON artifact.

Honesty rules carried over from Task 07 unchanged: interval levels are
conformal TARGETS, not verified probabilities; no probability label
remains the deterministic evidence-based assessment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dltime.config import PROJECT_ROOT

SUM_TOTAL_CALIBRATION_VERSION = "unc-log1p-conformal-v1-sum-of-five-stages"
SUM_TOTAL_CALIBRATION_PATH = (
    PROJECT_ROOT / "backend" / "artifacts" / f"{SUM_TOTAL_CALIBRATION_VERSION}.json"
)

# The calibration is bound to the deterministic sum policy, not to a fitted
# artifact; the stage-serving version is recorded for traceability.
BOUND_STAGE_SERVING_VERSION = "stage-serving-v2"


class SumTotalCalibrationError(RuntimeError):
    """Raised when the sum-total calibration cannot be built/loaded safely."""


def _encode_fit(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)


def build_sum_total_calibration(*, bundle=None) -> dict[str, Any]:
    """Build the residual-quantile calibration for the sum-of-five-stages ETA.

    Reuses the EXP-036 clean protocol machinery (same population builder, same
    chronological split, same frozen-stage-model discipline). Pure evaluation:
    nothing here retrains or replaces any serving model; the serving stage
    models are untouched (this calibration only measures THEIR sum's errors).
    """
    import importlib.util

    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import chronological_holdout_split
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import ML_STAGE_NAMES
    from dltime.sources import load_all_sources
    from dltime.uncertainty import (
        CALIBRATION_METHOD,
        INTERVAL_LEVELS,
        MIN_QUANTILE_FLOOR,
    )

    # EXP-036 population + split (identical boundary by construction).
    root = PROJECT_ROOT
    spec = importlib.util.spec_from_file_location(
        "exp036_clean_holdout", root / "backend" / "scripts" / "exp036_clean_holdout.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging guard
        raise SumTotalCalibrationError("EXP-036 protocol module not found.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bundle = bundle or load_all_sources()
    frame, y_raw, _total_feats = mod.build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)

    tt = TargetTransformer(mode="log1p")
    stage_predictors: dict[str, _StagePredictor] = {}
    stage_frame_by_id: dict[str, pd.DataFrame] = {}
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, _sdiag = build_stage_frame(stage=stage, bundle=bundle)
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        train_ids = set(train["LDR_Execution_Id"])
        elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
        X = elig_train[sfeats].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        medians = X.median(numeric_only=True)
        Xenc = _encode_fit(X)
        y_log = tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce"))
        kind = {
            "staging": "extra_trees",
            "prevalidation": "extra_trees",
            "transformation": "random_forest",
            "pre_load": "extra_trees",
            "data_loading": "ridge",
        }[stage]
        model = _make_estimator(kind)
        model.fit(Xenc, y_log)
        pred = _StagePredictor(stage, kind, model, sfeats, len(elig_train), medians)
        pred._enc_columns = Xenc.columns
        stage_predictors[stage] = pred
        stage_frame_by_id[stage] = sframe.set_index("LDR_Execution_Id")

    # Frozen stage models predict the holdout rows (as-of, strictly prior).
    sums: list[float] = []
    for eid in test["LDR_Execution_Id"]:
        total = 0.0
        ok = True
        for stage in ML_STAGE_NAMES:
            try:
                total += float(stage_predictors[stage].predict_seconds(stage_frame_by_id[stage].loc[[eid]]))
            except Exception:
                ok = False
                break
        sums.append(total if ok else np.nan)
    sum_pred = np.asarray(sums, dtype=float)

    y = y_raw.loc[test.index].to_numpy(dtype=float)
    positive = (y > 0) & np.isfinite(sum_pred)
    if positive.sum() == 0:
        raise SumTotalCalibrationError("No valid calibration residuals for the sum policy.")
    abs_res_log = np.abs(np.log1p(y[positive]) - np.log1p(sum_pred[positive]))

    quantiles: dict[str, float] = {}
    for level in INTERVAL_LEVELS:
        q = float(np.quantile(abs_res_log, level, method="higher"))
        quantiles[f"{int(round(level * 100))}"] = max(q, MIN_QUANTILE_FLOOR)

    meta = {
        "calibration_version": SUM_TOTAL_CALIBRATION_VERSION,
        "method": CALIBRATION_METHOD,
        "interval_levels": list(INTERVAL_LEVELS),
        "quantiles_abs_log_residual": quantiles,
        "n_calibration_rows": int(positive.sum()),
        "n_positive_target_rows": int(positive.sum()),
        "calibration_window": {
            "first_start": str(split_info["test_first_start"]),
            "last_start": str(split_info["test_last_start"]),
            "scope": (
                "final chronological holdout (EXP-036 clean protocol split); "
                "stage models fit ONLY on the earlier train period, frozen, then "
                "applied to the holdout strictly pre-execution — out-of-sample "
                "for the sum-of-five-stages ETA"
            ),
        },
        "residual_stats": {
            "median_abs_log_residual": float(np.median(abs_res_log)),
            "mean_abs_log_residual": float(abs_res_log.mean()),
            "max_abs_log_residual": float(abs_res_log.max()),
            "share_under_predictions": float(
                (np.log1p(y[positive]) < np.log1p(sum_pred[positive])).mean()
            ),
        },
        "model_version": "sum-of-five-stages-v1",
        "model_feature_count": len(ML_STAGE_NAMES),
        "binding": {
            "policy": "TOTAL = sum of the five operational stage predictions",
            "stage_serving_version": BOUND_STAGE_SERVING_VERSION,
            "protocol": "EXP-036 clean leakage-safe holdout machinery (identical split/population)",
        },
        "diagnostics": {
            "quantile_floor": MIN_QUANTILE_FLOOR,
            "quantile_interpolation": "higher",
            "note": (
                "Levels are conformal TARGETS (share of calibration residuals "
                "within q), not verified probabilities. EXP-036/EXP-038 evidence."
            ),
        },
    }
    SUM_TOTAL_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUM_TOTAL_CALIBRATION_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_sum_total_calibration() -> dict[str, Any]:
    """Load + integrity-check the sum-total calibration (fail closed)."""
    from dltime.uncertainty import CALIBRATION_METHOD, INTERVAL_LEVELS

    if not SUM_TOTAL_CALIBRATION_PATH.exists():
        raise SumTotalCalibrationError(
            f"Sum-total calibration not found: {SUM_TOTAL_CALIBRATION_PATH}"
        )
    meta = json.loads(SUM_TOTAL_CALIBRATION_PATH.read_text(encoding="utf-8"))
    if meta.get("calibration_version") != SUM_TOTAL_CALIBRATION_VERSION:
        raise SumTotalCalibrationError(
            f"Sum-total calibration version mismatch: {meta.get('calibration_version')!r}"
        )
    if meta.get("method") != CALIBRATION_METHOD:
        raise SumTotalCalibrationError("Unknown sum-total calibration method.")
    if meta.get("model_version") != "sum-of-five-stages-v1":
        raise SumTotalCalibrationError(
            f"Sum-total calibration bound to {meta.get('model_version')!r}"
        )
    for lvl in INTERVAL_LEVELS:
        if str(int(round(lvl * 100))) not in meta.get("quantiles_abs_log_residual", {}):
            raise SumTotalCalibrationError(
                f"Sum-total calibration missing quantile for {lvl}."
            )
    return meta
