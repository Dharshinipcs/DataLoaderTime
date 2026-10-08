"""Task 03: chronological model training and evaluation.

All evaluation is strictly chronological (AD-11 chronology):
  - final holdout: earliest rows train, latest rows evaluate (no shuffle)
  - expanding-window CV: training window grows forward in time
  - every model (including baselines) sees only earlier-than-validation rows

Every candidate pipeline runs the authoritative leakage validator on its
feature list before fitting. No random split is used anywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dltime.feature_assembly import TrainingFrame, TargetTransformer, build_training_frame

RANDOM_STATE = 42

# The previously validated ExtraTrees benchmark candidate (Part D).
EXTRATREES_CANDIDATE: dict = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Practical candidate set for a small dataset (Part C). No massive search.
CANDIDATE_MODELS: dict[str, dict] = {
    "ridge": {
        "estimator": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "needs_encoding": True,
        "needs_scaling": True,
    },
    "random_forest": {
        "estimator": lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "needs_encoding": True,
        "needs_scaling": False,
    },
    "extra_trees_candidate": {
        "estimator": lambda: ExtraTreesRegressor(**EXTRATREES_CANDIDATE),
        "needs_encoding": True,
        "needs_scaling": False,
    },
    "extra_trees_default": {
        "estimator": lambda: ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "needs_encoding": True,
        "needs_scaling": False,
    },
    "hist_gradient_boosting": {
        "estimator": lambda: HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
        "needs_encoding": True,
        "needs_scaling": False,
    },
}

BASELINE_MODELS: dict[str, dict] = {
    "baseline_global_median": {
        "estimator": lambda: DummyRegressor(strategy="median"),
        "needs_encoding": True,
        "needs_scaling": False,
    },
    "baseline_global_mean": {
        "estimator": lambda: DummyRegressor(strategy="mean"),
        "needs_encoding": True,
        "needs_scaling": False,
    },
}


def _try_optional_models() -> dict[str, dict]:
    """Include XGBoost/LightGBM/CatBoost only if already installed."""
    optional: dict[str, dict] = {}
    try:  # pragma: no cover - depends on environment
        import xgboost  # noqa: F401

        optional["xgboost"] = {
            "estimator": lambda: xgboost.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            "needs_encoding": True,
            "needs_scaling": False,
        }
    except ImportError:
        pass
    try:  # pragma: no cover - depends on environment
        import lightgbm  # noqa: F401

        optional["lightgbm"] = {
            "estimator": lambda: lightgbm.LGBMRegressor(
                n_estimators=300,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "needs_encoding": True,
            "needs_scaling": False,
        }
    except ImportError:
        pass
    try:  # pragma: no cover - depends on environment
        import catboost  # noqa: F401

        optional["catboost"] = {
            "estimator": lambda: catboost.CatBoostRegressor(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
            "needs_encoding": False,  # handles categoricals natively
            "needs_scaling": False,
        }
    except ImportError:
        pass
    return optional


def available_candidates() -> dict[str, dict]:
    candidates = dict(BASELINE_MODELS)
    candidates.update(CANDIDATE_MODELS)
    candidates.update(_try_optional_models())
    return candidates


# ------------------------- chronology utilities -------------------------

def add_chronological_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic chronological order (documented tie-break).

    Order key: (Start_Time, LDR_Execution_Id). Ties on Start_Time are broken
    by LDR_Execution_Id; no random ordering is used anywhere (Part A).
    """
    ordered = frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").copy()
    ordered["chrono_rank"] = np.arange(len(ordered))
    return ordered


def chronological_holdout_split(
    frame: pd.DataFrame, holdout_fraction: float = 0.25, min_holdout: int = 25
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Final chronological split: earliest rows train, latest rows test.

    The split point is a single timestamp boundary: every row whose Start_Time
    is strictly before the boundary belongs to train; everything else is test.
    With timestamp ties at the boundary, rows sharing the boundary timestamp
    are kept together (all in test) so no identical timestamp appears in both
    sides. Returns (train, test, info).
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1).")
    ordered = add_chronological_rank(frame)
    n_test = max(int(round(len(ordered) * holdout_fraction)), min_holdout)
    if n_test >= len(ordered):
        raise ValueError("holdout too large for frame")
    # First candidate cut index (row count based), then snap to a timestamp
    # boundary so identical timestamps never straddle the split.
    cut = len(ordered) - n_test
    boundary_ts = ordered.iloc[cut]["Start_Time"]
    while cut > 0 and ordered.iloc[cut - 1]["Start_Time"] == boundary_ts:
        cut -= 1
    train = ordered.iloc[:cut]
    test = ordered.iloc[cut:]
    info = {
        "boundary_start_time": str(boundary_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_first_start": str(train["Start_Time"].iloc[0]),
        "train_last_start": str(train["Start_Time"].iloc[-1]),
        "test_first_start": str(test["Start_Time"].iloc[0]),
        "test_last_start": str(test["Start_Time"].iloc[-1]),
        "train_max_rank": int(train["chrono_rank"].max()),
        "test_min_rank": int(test["chrono_rank"].min()),
    }
    return train, test, info


def expanding_window_folds(
    frame: pd.DataFrame, n_folds: int = 5, initial_train_fraction: float = 0.5
) -> list[dict]:
    """Expanding-window chronological CV folds (Part E).

    Fold k trains on the earliest (initial + k) portion of the ordered frame
    and validates on the next chronological block. Training always precedes
    validation in time; the window only grows forward.
    """
    ordered = add_chronological_rank(frame)
    n = len(ordered)
    start = max(int(round(n * initial_train_fraction)), 10)
    if n - start < n_folds:
        raise ValueError("n_folds too large for frame size")
    # Validation blocks PARTITION [start, n) contiguously: fold k trains on
    # [0, bounds[k]) and validates [bounds[k], bounds[k+1]). This guarantees
    # (a) training always precedes validation in time and (b) the pooled OOF
    # covers EVERY row outside the initial training window (no silent gap when
    # (n - start) is not divisible by n_folds).
    bounds = [start + (n - start) * k // n_folds for k in range(n_folds + 1)]
    folds: list[dict] = []
    for k in range(n_folds):
        train_end = bounds[k]
        val_end = bounds[k + 1]
        folds.append(
            {
                "fold": k + 1,
                "train_rank_start": 0,
                "train_rank_end_exclusive": int(train_end),
                "val_rank_start": int(train_end),
                "val_rank_end_exclusive": int(val_end),
                "train_rows": int(train_end),
                "val_rows": int(val_end - train_end),
                "train_last_start": str(ordered.iloc[train_end - 1]["Start_Time"]),
                "val_first_start": str(ordered.iloc[train_end]["Start_Time"]),
                "val_last_start": str(ordered.iloc[val_end - 1]["Start_Time"]),
            }
        )
    return folds


# ------------------------- encoding -------------------------

def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str], scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipeline = Pipeline(num_steps)
    # Categorical columns are pre-filled with an explicit sentinel by the
    # caller (pandas pd.NA breaks sklearn imputers), so no cat imputer here.
    cat_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def matrix_encoding_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into categorical and numeric sets."""
    categorical = [
        c
        for c in X.columns
        if pd.api.types.is_string_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric


# ------------------------- metrics -------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = abs_err / np.where(y_true > 0, y_true, np.nan)
    tol_10 = float(np.nanmean(rel <= 0.10) * 100.0)
    tol_20 = float(np.nanmean(rel <= 0.20) * 100.0)
    tol_30 = float(np.nanmean(rel <= 0.30) * 100.0)
    tol_2x = float(np.nanmean((y_pred <= 2 * y_true) & (y_pred >= 0.5 * y_true)) * 100.0)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.where(denom > 0, abs_err / denom, 0.0)) * 100.0)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(np.median(abs_err)),
        "within_10pct": tol_10,
        "within_20pct": tol_20,
        "within_30pct": tol_30,
        "within_2x": tol_2x,
        "smape_pct": smape,
        "abs_err_p90": float(np.percentile(abs_err, 90)),
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Pooled OOF, mean across folds, median across folds, worst fold.

    Pooled = all out-of-fold predictions treated as one sample set.
    Mean/median across folds = per-fold metric distribution.
    These are deliberately distinct and reported separately (Part F).
    """
    keys = ("mae", "rmse", "median_ae", "within_10pct", "within_20pct", "within_30pct", "within_2x", "smape_pct")
    out: dict = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = {
            "mean_across_folds": float(vals.mean()),
            "median_across_folds": float(np.median(vals)),
            "worst_fold": float(vals.max()) if k in ("mae", "rmse", "median_ae", "smape_pct") else float(vals.min()),
        }
    out["total_val_rows"] = int(sum(m["n"] for m in fold_metrics))
    out["n_folds"] = len(fold_metrics)
    return out


def _as_frame(obj) -> pd.DataFrame:
    return obj.frame if isinstance(obj, TrainingFrame) else obj


# ------------------------- leakage validation before fitting -------------------------

def validate_features_for_model(tf: TrainingFrame, feature_names: list[str]) -> None:
    """Run the authoritative Task-01 leakage boundary on the model input."""
    from dltime.feature_assembly import PREDICTION_FEATURE_GROUPS

    assert feature_names == tf.feature_names, "Model feature order must match TrainingFrame.feature_names"
    assert not any(col in tf.frame.columns and col == tf.target_column for col in feature_names)
    tf.X[feature_names].head(1)  # existence check
    from dltime.leakage import guard_frame

    guard_frame(tf.X[feature_names], feature_names, context="model-fit")


# ------------------------- prediction runner -------------------------

@dataclass
class FoldResult:
    fold: int
    train_rows: int
    val_rows: int
    metrics: dict
    fit_seconds: float


@dataclass
class EvaluationResult:
    model_name: str
    target_mode: str
    fold_results: list[FoldResult] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)
    pooled_oof: dict | None = None
    per_fold_aggregate: dict | None = None
    holdout: dict | None = None
    oof_by_row: pd.Series | None = None  # index-aligned out-of-fold predictions
    feature_names: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        p = self.pooled_oof or {}
        h = self.holdout or {}
        return (
            f"{self.model_name:<28} [{self.target_mode:>5}] "
            f"pooled MAE={p.get('mae', float('nan')):>10.2f} "
            f"MedAE={p.get('median_ae', float('nan')):>9.2f} "
            f"RMSE={p.get('rmse', float('nan')):>11.2f} "
            f"|±30%={p.get('within_30pct', float('nan')):>5.1f}% "
            f"| holdout MAE={h.get('mae', float('nan')):>10.2f} "
            f"MedAE={h.get('median_ae', float('nan')):>9.2f}"
        )


def fit_candidate_pipeline(
    model_spec: dict,
    X_train: pd.DataFrame,
    y_train_t: np.ndarray,
    tf_mode: str,
    feature_names: list[str],
) -> tuple[object, object, list[str]]:
    """Fit one candidate pipeline; return (estimator, preprocessor, feature_out).

    The serving layer (Task 04 artifact) reuses this exact path so inference
    preprocessing is byte-identical to evaluation preprocessing.
    """
    categorical, numeric = matrix_encoding_columns(X_train)
    if categorical:
        # Explicit missing sentinel (documented missing-value behaviour):
        # missing category = "information not available pre-execution".
        for c in categorical:
            X_train = X_train.copy()
            X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
    est = model_spec["estimator"]()
    if model_spec.get("needs_encoding", True) and categorical:
        pre = build_preprocessor(categorical, numeric, scale=model_spec.get("needs_scaling", False))
        Xtr_p = pre.fit_transform(X_train)
        # numeric arrays (dense); column names kept for potential diagnostics
        feature_out = list(pre.get_feature_names_out())
    else:
        Xtr_p = X_train.to_numpy(dtype=float)
        pre = None
        feature_out = list(X_train.columns)
    y_t = TargetTransformer(mode=tf_mode).transform(pd.Series(y_train_t))
    est.fit(Xtr_p, y_t)
    return est, pre, feature_out


def transform_with_preprocessor(
    pre,
    X: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> np.ndarray:
    """Apply a fitted preprocessor to new rows (serving path, Task 04)."""
    X = X.copy()
    for c in categorical:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    return pre.transform(X)


def _fit_predict_one(
    model_spec: dict,
    X_train: pd.DataFrame,
    y_train_t: np.ndarray,
    X_val: pd.DataFrame,
    tf_mode: str,
    feature_names: list[str],
):
    """Fit one model on transformed target; return raw-second predictions."""
    Xtr = X_train[feature_names].copy()
    Xva = X_val[feature_names].copy()
    categorical, numeric = matrix_encoding_columns(Xtr)
    est, pre, feature_out = fit_candidate_pipeline(model_spec, Xtr, y_train_t, tf_mode, feature_names)
    if pre is not None:
        Xva_p = transform_with_preprocessor(pre, Xva, categorical, numeric)
    else:
        Xva_p = Xva.to_numpy(dtype=float)
    t0 = time.perf_counter()
    pred_t = est.predict(Xva_p)
    fit_seconds = time.perf_counter() - t0
    pred_raw = TargetTransformer(mode=tf_mode).inverse_transform(pred_t)
    return pred_raw, fit_seconds, feature_out


def evaluate_candidate(
    name: str,
    model_spec: dict,
    tf: TrainingFrame,
    target_mode: str,
    folds: list[dict] | None = None,
    ordered_frame: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Expanding-window chronological CV + final chronological holdout."""
    feature_names = tf.feature_names
    validate_features_for_model(tf, feature_names)
    frame = ordered_frame if ordered_frame is not None else _as_frame(tf)
    # Align X/y to the chronological order of the full frame.
    X = tf.X.loc[frame.index]
    y = tf.y_raw.loc[frame.index]

    result = EvaluationResult(model_name=name, target_mode=target_mode, feature_names=list(feature_names))
    oof = pd.Series(np.nan, index=frame.index)

    fold_list = folds or expanding_window_folds(frame)
    for f in fold_list:
        train_idx = frame.index[: f["train_rank_end_exclusive"]]
        val_idx = frame.index[f["val_rank_start"] : f["val_rank_end_exclusive"]]
        pred_raw, fit_s, _ = _fit_predict_one(
            model_spec,
            X.loc[train_idx],
            y.loc[train_idx].to_numpy(),
            X.loc[val_idx],
            target_mode,
            feature_names,
        )
        m = compute_metrics(y.loc[val_idx].to_numpy(), pred_raw)
        result.fold_results.append(
            FoldResult(fold=f["fold"], train_rows=f["train_rows"], val_rows=f["val_rows"], metrics=m, fit_seconds=fit_s)
        )
        result.fold_metrics.append(m)
        oof.loc[val_idx] = pred_raw

    result.per_fold_aggregate = aggregate_fold_metrics(result.fold_metrics)
    mask = oof.notna()
    result.pooled_oof = compute_metrics(y[mask].to_numpy(), oof[mask].to_numpy())
    result.oof_by_row = oof

    # Final chronological holdout (Part E.1)
    train, test, split_info = chronological_holdout_split(frame)
    pred_hold, _, _ = _fit_predict_one(
        model_spec,
        X.loc[train.index],
        y.loc[train.index].to_numpy(),
        X.loc[test.index],
        target_mode,
        feature_names,
    )
    result.holdout = compute_metrics(y.loc[test.index].to_numpy(), pred_hold)
    result.holdout["split"] = split_info
    return result


# ------------------------- loader-history-aware evaluation -------------------------

def seen_unseen_evaluation(result: EvaluationResult, tf: TrainingFrame, frame: pd.DataFrame) -> dict:
    """Split pooled OOF metrics by whether the loader had prior history (Part G)."""
    oof = result.oof_by_row
    has_prior = tf.X.loc[frame.index, "hist_loader_prior_count"] > 0
    mask = oof.notna()
    seen_mask = mask & has_prior
    unseen_mask = mask & ~has_prior
    y = tf.y_raw.loc[frame.index]
    out: dict = {"seen_loader": None, "unseen_loader": None}
    if seen_mask.any():
        out["seen_loader"] = compute_metrics(y[seen_mask].to_numpy(), oof[seen_mask].to_numpy())
        out["seen_loader"]["n"] = int(seen_mask.sum())
    if unseen_mask.any():
        out["unseen_loader"] = compute_metrics(y[unseen_mask].to_numpy(), oof[unseen_mask].to_numpy())
        out["unseen_loader"]["n"] = int(unseen_mask.sum())
    return out


HISTORY_BINS = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, np.inf, "10+")]


def history_bin_evaluation(result: EvaluationResult, tf: TrainingFrame, frame: pd.DataFrame) -> dict:
    """Pooled OOF metrics per prior-loader-history bin (Part H)."""
    oof = result.oof_by_row
    counts = tf.X.loc[frame.index, "hist_loader_prior_count"]
    y = tf.y_raw.loc[frame.index]
    out: dict = {}
    for lo, hi, label in HISTORY_BINS:
        mask = oof.notna() & (counts >= lo) & (counts <= hi)
        if mask.sum() == 0:
            out[label] = {"n": 0}
            continue
        m = compute_metrics(y[mask].to_numpy(), oof[mask].to_numpy())
        m["n"] = int(mask.sum())
        out[label] = m
    return out


# ------------------------- synthetic evaluation probe -------------------------

def evaluate_candidate_synthetic(
    frame: pd.DataFrame,
    y: pd.Series,
    model_kind: str = "tree",
    target_mode: str = "log1p",
    n_folds: int = 4,
) -> EvaluationResult:
    """Run the same chronological evaluation machinery on an arbitrary frame.

    Used by tests to prove no future rows influence earlier folds. y must be
    index-aligned with frame.
    """
    ordered = add_chronological_rank(frame)
    y = y.loc[ordered.index]
    spec = {
        "estimator": lambda: ExtraTreesRegressor(n_estimators=60, random_state=RANDOM_STATE, n_jobs=-1),
        "needs_encoding": True,
        "needs_scaling": False,
    }
    feature_names = [c for c in ordered.columns if c not in ("chrono_rank",)]
    X = ordered[feature_names].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("string")
    result = EvaluationResult(model_name="synthetic", target_mode=target_mode, feature_names=feature_names)
    oof = pd.Series(np.nan, index=ordered.index)
    folds = expanding_window_folds(ordered, n_folds=n_folds)
    for f in folds:
        train_idx = ordered.index[: f["train_rank_end_exclusive"]]
        val_idx = ordered.index[f["val_rank_start"] : f["val_rank_end_exclusive"]]
        pred, _, _ = _fit_predict_one(spec, X.loc[train_idx], y.loc[train_idx].to_numpy(), X.loc[val_idx], target_mode, feature_names)
        m = compute_metrics(y.loc[val_idx].to_numpy(), pred)
        result.fold_results.append(FoldResult(f["fold"], f["train_rows"], f["val_rows"], m, 0.0))
        result.fold_metrics.append(m)
        oof.loc[val_idx] = pred
    result.per_fold_aggregate = aggregate_fold_metrics(result.fold_metrics)
    mask = oof.notna()
    result.pooled_oof = compute_metrics(y[mask].to_numpy(), oof[mask].to_numpy())
    result.oof_by_row = oof
    return result


# ------------------------- orchestration -------------------------

def run_model_experiments(tf: TrainingFrame, target_modes: tuple[str, ...] = ("log1p", "raw")) -> list[EvaluationResult]:
    """Evaluate baselines + candidates chronologically on the real frame."""
    full = _as_frame(tf)
    eligible = full[full["is_training_eligible"].fillna(False)]
    ordered = add_chronological_rank(eligible)
    folds = expanding_window_folds(ordered)
    results: list[EvaluationResult] = []
    for name, spec in available_candidates().items():
        for mode in target_modes:
            if mode == "raw" and name.startswith("baseline"):
                continue  # scale-free baselines identical in both target spaces
            results.append(evaluate_candidate(name, spec, tf, mode, folds=folds, ordered_frame=ordered))
    return results


# ------------------------- model artifact (Part J) -------------------------

def build_model_artifact(
    model_name: str,
    target_mode: str,
    tf: TrainingFrame,
    evaluation: EvaluationResult,
    evaluation_config: dict,
    version: str = "total-et-v3-candidate",
) -> dict:
    """Reproducible artifact/config record. NOT wired to any endpoint yet."""
    full = _as_frame(tf)
    eligible = full[full["is_training_eligible"].fillna(False)]
    frame = add_chronological_rank(eligible)
    train, _, split_info = chronological_holdout_split(frame)
    spec = available_candidates()[model_name]
    est = spec["estimator"]()
    return {
        "model_version": version,
        "status": "CANDIDATE_EVALUATED",  # not production/final by construction
        "model_type": model_name,
        "hyperparameters": est.get_params(),
        "target_transformation": target_mode,
        "feature_names_in_order": list(tf.feature_names),
        "categorical_features": list(matrix_encoding_columns(tf.X)[0]),
        "encoding": "one_hot(handle_unknown=ignore) for categoricals; median impute for numerics",
        "training_row_count": int(split_info["train_rows"]),
        "training_period": {
            "first_start": split_info["train_first_start"],
            "last_start": split_info["train_last_start"],
        },
        "evaluation": {
            "split_method": "expanding-window chronological CV + final chronological holdout",
            "config": evaluation_config,
            "pooled_oof": evaluation.pooled_oof,
            "holdout": evaluation.holdout,
        },
    }
