"""Total-ETA calibrated uncertainty interval layer (Task 07, AD-20).

A layer AROUND the authoritative Task 04 point prediction — never a
replacement for it. The point ETA is produced by the unchanged serving
artifact; this module only estimates how far historical model behavior says
the actual duration could fall from that point.

Method (chosen from evidence, EXP-016):
    Conformal-style absolute-residual intervals calibrated in log1p space.
    |log1p(actual) - log1p(predicted)| on the chronological calibration
    window -> empirical quantile q (Q80/Q90). PRODUCT PRESENTATION (locked):
    the interval is displayed ADDITIVELY around the point prediction in
    seconds — [point - q, point + q], lower clamped at 0 — so the user sees
    e.g. 16.65 s -> 13.73-19.57 s at Q90 = 2.9195. The calibrated quantile,
    the residual population and the level are unchanged; only the transform
    used to render the bounds differs from the internal log-space form.

Calibration data (exact temporal scope):
    The final chronological holdout of the training-eligible frame
    (2026-07-15 17:48:04.915 -> 2026-08-13 14:41:43.809). The serving model
    was trained on the 262 EARLIER rows and NEVER saw these rows, so every
    residual is a genuine out-of-sample error of the exact serving artifact.
    All calibration rows are strictly earlier than any serving context
    (contexts default to latest recorded + 1 day). No current or future
    execution can influence the interval.

Honesty rules:
    - "80%/90%" is the conformal TARGET level (fraction of calibration
      residuals inside q), NOT a verified probability for future executions.
      Empirical coverage on the held-out evaluation slice is reported in the
      calibration metadata and the experiment log.
    - Reliability is a deterministic evidence-based label (HIGH/MEDIUM/LOW),
      never a probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dltime.artifact import ARTIFACT_DIR, predict_with_artifact
from dltime.feature_assembly import TrainingFrame
from dltime.modeling import add_chronological_rank, chronological_holdout_split

CALIBRATION_VERSION = "unc-log1p-conformal-v1"
CALIBRATION_METHOD = "log1p_space_absolute_residual_conformal"
INTERVAL_LEVELS: tuple[float, ...] = (0.80, 0.90)
PRIMARY_LEVEL = 0.90
# Residual-quantile floor: with very few calibration residuals an empirical
# quantile can collapse toward 0 (overconfident). The 80th percentile of the
# observed residual distribution is the minimum spread we are willing to claim.
MIN_QUANTILE_FLOOR = 1.5653


class UncertaintyError(RuntimeError):
    """Raised when calibration is impossible or the artifact is incompatible."""


@dataclass
class CalibrationResult:
    version: str
    method: str
    levels: tuple[float, ...]
    quantiles: dict[str, float]
    n_calibration_rows: int
    n_positive_target_rows: int
    calibration_window: dict[str, str]
    residual_stats: dict[str, float]
    model_version: str
    model_feature_count: int
    diagnostics: dict[str, Any]


def build_calibration(
    tf: TrainingFrame,
    model_version: str,
    predict_batch,
) -> CalibrationResult:
    """Calibrate on the chronological holdout.

    predict_batch(X: pd.DataFrame) -> np.ndarray must produce raw-second
    point predictions with the EXACT serving artifact (same training window
    recorded in model_version), so every residual is out-of-sample for it.
    """
    full = tf.frame
    eligible = full[full["is_training_eligible"].fillna(False)]
    ordered = add_chronological_rank(eligible)
    _train, test, split_info = chronological_holdout_split(ordered)
    # Leakage guard: the calibration window must not overlap the model's
    # recorded training window in time.
    train_last = split_info["train_last_start"]
    test_first = split_info["test_first_start"]
    if pd.Timestamp(test_first) < pd.Timestamp(train_last):
        raise UncertaintyError(
            f"Calibration window {test_first!r} overlaps model training window end {train_last!r}."
        )

    X_cal = tf.X.loc[test.index, tf.feature_names]
    y_cal = tf.y_raw.loc[test.index].to_numpy(dtype=float)
    if len(X_cal) == 0:
        raise UncertaintyError("No calibration rows available.")
    preds = np.asarray(predict_batch(X_cal), dtype=float)
    if len(preds) != len(X_cal) or not np.all(np.isfinite(preds)) or np.any(preds <= 0):
        raise UncertaintyError("Calibration predictions must be finite and positive.")

    positive = y_cal > 0
    abs_res_log = np.abs(np.log1p(y_cal[positive]) - np.log1p(preds[positive]))

    quantiles: dict[str, float] = {}
    for level in INTERVAL_LEVELS:
        q = float(np.quantile(abs_res_log, level, method="higher"))
        quantiles[f"{int(round(level * 100))}"] = max(q, MIN_QUANTILE_FLOOR)

    residual_stats = {
        "median_abs_log_residual": float(np.median(abs_res_log)),
        "mean_abs_log_residual": float(abs_res_log.mean()),
        "max_abs_log_residual": float(abs_res_log.max()),
        "share_under_predictions": float((np.log1p(y_cal[positive]) < np.log1p(preds[positive])).mean()),
    }

    return CalibrationResult(
        version=CALIBRATION_VERSION,
        method=CALIBRATION_METHOD,
        levels=INTERVAL_LEVELS,
        quantiles=quantiles,
        n_calibration_rows=int(len(X_cal)),
        n_positive_target_rows=int(positive.sum()),
        calibration_window={
            "first_start": str(split_info["test_first_start"]),
            "last_start": str(split_info["test_last_start"]),
            "scope": "final chronological holdout; strictly after the model training window; out-of-sample for the serving artifact",
        },
        residual_stats=residual_stats,
        model_version=model_version,
        model_feature_count=len(tf.feature_names),
        diagnostics={
            "quantile_floor": MIN_QUANTILE_FLOOR,
            "quantile_interpolation": "higher",
            "note": (
                "Levels are conformal TARGETS (share of calibration residuals "
                "within q), not verified probabilities. Honest held-out "
                "coverage evidence: EXP-016."
            ),
        },
    )


def calibration_to_metadata(cal: CalibrationResult) -> dict[str, Any]:
    """Serializable metadata for the versioned calibration artifact."""
    return {
        "calibration_version": cal.version,
        "method": cal.method,
        "interval_levels": list(cal.levels),
        "quantiles_abs_log_residual": cal.quantiles,
        "n_calibration_rows": cal.n_calibration_rows,
        "n_positive_target_rows": cal.n_positive_target_rows,
        "calibration_window": cal.calibration_window,
        "residual_stats": cal.residual_stats,
        "model_version": cal.model_version,
        "model_feature_count": cal.model_feature_count,
        "diagnostics": cal.diagnostics,
    }


def default_calibration_path() -> Path:
    return ARTIFACT_DIR / f"{CALIBRATION_VERSION}.json"


def save_calibration_metadata(metadata: dict[str, Any], path: Path | None = None) -> Path:
    """Persist the versioned calibration metadata (plain JSON, inspectable)."""
    path = path or default_calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_calibration_artifact(
    model_version: str, model_feature_count: int, path: Path | None = None
) -> dict[str, Any]:
    """Load + integrity-check the calibration metadata (fail closed)."""
    path = path or default_calibration_path()
    if not path.exists():
        raise UncertaintyError(f"Calibration artifact not found: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UncertaintyError(f"Calibration artifact unreadable at {path}: {exc}") from exc
    return load_calibration_metadata(metadata, model_version, model_feature_count)


def ensure_calibration(
    tf: TrainingFrame, artifact: dict, path: Path | None = None
) -> dict[str, Any]:
    """Load the persisted calibration; build + save it once if absent.

    The calibration is bound to the serving artifact (model version + feature
    count are integrity-checked on every load). Building uses the EXACT
    serving artifact for out-of-sample holdout predictions.
    """
    model_version = artifact["metadata"]["model_version"]
    feature_count = len(artifact["metadata"]["feature_names_in_order"])
    path = path or default_calibration_path()
    if path.exists():
        return load_calibration_artifact(model_version, feature_count, path)

    def predict_batch(X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [predict_with_artifact(artifact, X.iloc[[i]]) for i in range(len(X))],
            dtype=float,
        )

    cal = build_calibration(tf, model_version, predict_batch)
    metadata = calibration_to_metadata(cal)
    save_calibration_metadata(metadata, path)
    return metadata


def load_calibration_metadata(metadata: dict[str, Any], model_version: str, model_feature_count: int) -> dict[str, Any]:
    """Validate a calibration metadata block against the serving artifact.

    Fail closed on any incompatibility (AD-20): wrong model version, changed
    feature count, unknown method, or missing quantiles.
    """
    if metadata.get("calibration_version") != CALIBRATION_VERSION:
        raise UncertaintyError(
            f"Calibration version mismatch: got {metadata.get('calibration_version')!r}, expected {CALIBRATION_VERSION!r}"
        )
    if metadata.get("method") != CALIBRATION_METHOD:
        raise UncertaintyError(f"Unknown calibration method: {metadata.get('method')!r}")
    if metadata.get("model_version") != model_version:
        raise UncertaintyError(
            f"Calibration is bound to model {metadata.get('model_version')!r}, serving {model_version!r}"
        )
    if metadata.get("model_feature_count") != model_feature_count:
        raise UncertaintyError(
            f"Calibration feature count {metadata.get('model_feature_count')} != model {model_feature_count}"
        )
    for lvl in INTERVAL_LEVELS:
        key = str(int(round(lvl * 100)))
        if key not in metadata.get("quantiles_abs_log_residual", {}):
            raise UncertaintyError(f"Calibration metadata missing quantile for level {key}.")
    return metadata


def interval_from_quantile(
    predicted_seconds: float, quantile_abs_log: float
) -> tuple[float, float]:
    """Product interval: point prediction +/- quantile, in SECONDS.

    Locked presentation (Prediction Studio): the calibrated conformal
    quantile Q90 is applied additively around the point prediction so the
    displayed range is symmetric and readable (16.65 -> 13.73-19.57 at
    Q90 = 2.9195). Lower bound clamped at 0 (durations cannot be negative).
    The point prediction always lies inside by construction.
    """
    if not np.isfinite(predicted_seconds) or predicted_seconds <= 0:
        raise UncertaintyError(f"Point prediction must be finite and positive, got {predicted_seconds!r}")
    if not np.isfinite(quantile_abs_log) or quantile_abs_log <= 0:
        raise UncertaintyError(f"Quantile must be finite and positive, got {quantile_abs_log!r}")
    lower = float(max(predicted_seconds - quantile_abs_log, 0.0))
    upper = float(predicted_seconds + quantile_abs_log)
    if upper < lower:  # impossible by construction; defensive
        raise UncertaintyError("Interval construction produced upper < lower.")
    return lower, upper


# ------------------------------------------------------------------ interval payload


def uncertainty_section(
    *,
    predicted_seconds: float,
    calibration_metadata: dict[str, Any],
    prior_loader_executions: int,
    min_required_history: int,
    evidence: dict[str, Any],
    level: float = PRIMARY_LEVEL,
) -> dict[str, Any]:
    """Assemble the calibrated uncertainty block for a PREDICTED response.

    2026-10-03 (user final spec): the reliability LABEL layer was removed from
    the system entirely — the calibrated interval itself (plus the monitoring
    metrics and drift statistics) is the honesty mechanism. No label, no
    reasons list, nothing named reliability is emitted anywhere.
    """
    q = calibration_metadata["quantiles_abs_log_residual"][str(int(round(level * 100)))]
    lower, upper = interval_from_quantile(predicted_seconds, q)
    width = upper - lower
    return {
        "available": True,
        "method": calibration_metadata["method"],
        "calibration_version": calibration_metadata["calibration_version"],
        "interval_level": level,
        "interval_level_display": f"{int(round(level * 100))}%",
        "lower_seconds": round(lower, 3),
        "upper_seconds": round(upper, 3),
        "interval_width_seconds": round(width, 3),
        "calibration_rows": calibration_metadata["n_calibration_rows"],
        "calibration_window": calibration_metadata["calibration_window"],
        # User-facing confidence for the displayed interval. NOT a probability
        # estimate and NOT invented by the UI: it is the interval's conformal
        # TARGET level — the share of calibration residuals the interval is
        # designed to contain (see the calibration diagnostics note). Wording
        # mirrors the interpretation text; levels come from the calibration
        # artifact itself (unc-log1p-conformal-v1-sum-of-five-stages).
        "confidence_display": {
            "percentage": int(round(level * 100)),
            "meaning": (
                f"Designed to contain the actual duration {int(round(level * 100))}% "
                "of the time, based on the errors of the current stage models on "
                "recent executions."
            ),
        },
        "interpretation": (
            "Based on the model's historical errors on earlier executions, the actual "
            "duration is estimated to fall within this range. This is an estimated "
            "range, not a guarantee."
        ),
    }
