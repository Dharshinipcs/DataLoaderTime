"""Stage-intelligence v2 — the MAIN GOAL rebuild on the filled dataset (2026-09-29).

Goal (user's main-goal contract):
  * User enters the 11 pre-execution inputs; the system predicts the five
    operational stage durations with REAL fitted ML models (no hardcoding),
    one model per stage, each selected by a documented multi-model comparison
    (Ridge / RandomForest / ExtraTrees / HistGradientBoosting / GradientBoosting)
    with chronological expanding-window CV + a chronological holdout, plus a
    small per-family hyperparameter grid, log1p target transform, and
    MedAE-primary selection (robust on heavy-tailed durations; RMSE/MAE/±20%
    co-reported).
  * Loader-specific learning: the stage frames carry the locked AS-OF
    stage_hist_* aggregates (global / loader / loader+conn / loader+dataset /
    loader+conn+dataset + previous-value), so a known loader's own history
    drives its prediction; the same model generalizes via def_* config
    features for loaders with little/no history.
  * THREE availability cases (user contract), enforced by the threshold gate:
      1. KNOWN loader with >= STAGE_LOADER_MIN_PRIOR prior stage observations
         -> full prediction, loader-specific, sum available.
      2. UNKNOWN loader (never executed) -> global/def-feature model still
         predicts but is disclosed LOW-CONFIDENCE-WARMUP; NO total is served
         until the loader reaches the threshold with real outcomes.
      3. KNOWN loader BELOW the threshold (1..K-1 priors) -> system tells the
         user the threshold is not met; no total is fabricated.
    (Stage-level: a stage with no training rows at all — staging on the
    replaced export — stays honestly unavailable, never fabricated.)
  * Total = sum(available operational stage predictions) + approvals, where
    the two approval stages remain the contextual hierarchical STATISTICAL
    estimator (too sparse for ML; documented). When the threshold gate blocks
    the total, the system reports exactly why and what is missing.

Data contract: TRAINING runs on the FILLED dataset (train/data_filled/,
LDR_Execution_Id < 899001 = real, >= 899001 = synthetic-augmentation;
synthetic ids are EXCLUDED from any accuracy/evaluation metric by design and
used only to widen training coverage). Serving candidates must pass the
regression gate against the REAL-rows-only evaluation before activation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (dltime -> src -> backend -> root)
FILLED_DIR = PROJECT_ROOT / "train" / "data_filled"

# ---- availability / threshold contract (user main-goal spec) ----------------
STAGE_LOADER_MIN_PRIOR = 3          # priors needed for a trusted loader-specific total
SUM_VERSION = "sum-of-available-stages-v2"

CANDIDATE_FAMILIES = (
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gradient_boosting",
)

# small, documented grids (small-data regime: prefer regularized, shallow-ish)
HYPERPARAMETER_GRIDS: dict[str, tuple[dict, ...]] = {
    "ridge": ({"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0}),
    "random_forest": (
        {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.5},
        {"n_estimators": 400, "min_samples_leaf": 1, "max_features": 1.0},
    ),
    "extra_trees": (
        {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.5},
        {"n_estimators": 400, "min_samples_leaf": 1, "max_features": 1.0},
    ),
    "hist_gradient_boosting": (
        {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 10},
        {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 5},
    ),
    "gradient_boosting": (
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3},
    ),
}

RANDOM_STATE = 42
STAGE_V2_MIN_TRAIN_ROWS = 30       # below this a stage cannot be trained at all
STAGE_V2_MIN_EVAL_ROWS = 12        # chronological holdout needs >= 12 rows
GATE_VERSION = "loader-threshold-gate-v1"


def _make_estimator(kind: str, params: dict):
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import Ridge

    if kind == "ridge":
        return Ridge(**params, random_state=RANDOM_STATE)
    if kind == "random_forest":
        return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if kind == "extra_trees":
        return ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params, early_stopping=False, random_state=RANDOM_STATE)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    raise ValueError(f"unknown model family: {kind}")


# --------------------------------------------------------------------------
# Filled-source loading
# --------------------------------------------------------------------------

def load_filled_bundle():
    """SourceBundle over the FILLED workbooks (originals never touched).

    Reads train/data_filled/ directly and applies the SAME per-file schema
    validation as the real sources. (The sources data_dir override does not
    reach the SourceSpec.path property, so the bundle is built explicitly.)
    """
    import hashlib

    import dltime.sources as src_mod

    if not FILLED_DIR.exists():
        return src_mod.load_all_sources()
    detail = pd.read_excel(FILLED_DIR / "LOADER_DEFINATIONS.xlsx")
    summary = pd.read_excel(FILLED_DIR / "LOADER_DEFINATIONS_1.xlsx")
    executions = pd.read_excel(FILLED_DIR / "Loader_Executions.xlsx")
    for spec, df in (
        (src_mod.SOURCE_SPECS["loader_definitions_detail"], detail),
        (src_mod.SOURCE_SPECS["loader_definitions_summary"], summary),
        (src_mod.SOURCE_SPECS["executions"], executions),
    ):
        src_mod._validate(spec, df)
    from dltime.sources import SourceBundle

    def _prov(name, df):
        p = FILLED_DIR / src_mod.SOURCE_SPECS[name].filename
        return {
            "file": p.name,
            "role": src_mod.SOURCE_SPECS[name].role,
            "path": str(p),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }

    provenance = {
        "loader_definitions_detail": _prov("loader_definitions_detail", detail),
        "loader_definitions_summary": _prov("loader_definitions_summary", summary),
        "executions": _prov("executions", executions),
        "filled_dataset": {
            "file": str(FILLED_DIR),
            "real_exec_rows": int(real_exec_mask(executions).sum()),
            "synthetic_exec_rows": int((~real_exec_mask(executions)).sum()),
            "note": "real rows byte-identical to train/data; synthetic marked by id range",
        },
    }
    return SourceBundle(
        loader_definitions_detail=detail,
        loader_definitions_summary=summary,
        executions=executions,
        provenance=provenance,
    )


def real_exec_mask(df: pd.DataFrame) -> pd.Series:
    """True for REAL execution rows (synthetic augmentation excluded)."""
    ids = pd.to_numeric(df["LDR_Execution_Id"], errors="coerce")
    return ids < 899001


# --------------------------------------------------------------------------
# Comparison protocol (per stage)
# --------------------------------------------------------------------------

def _encode(X: pd.DataFrame, ref_columns=None):
    Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
    if ref_columns is not None:
        Xenc = Xenc.reindex(columns=ref_columns, fill_value=0)
    return Xenc


def _chronological_folds(elig: pd.DataFrame, n_folds: int = 4):
    """Expanding-window chronological folds (train strictly earlier than val).

    Small-sample cadence (documented): when the pool has fewer than 40 rows,
    fall back to 2 folds with a 60/40 boundary so tiny-but-real stages
    (pre_load on the filled dataset) still get an honest CV signal.
    """
    n = len(elig)
    if n < 40:
        n_folds = min(n_folds, 2)
    min_train = 8 if n < 40 else 10
    first_val = max(min_train, int(n * (0.6 if n < 40 else 0.5)))
    folds = []
    if n - 3 >= first_val and n_folds > 1:
        val_starts = np.linspace(first_val, n - 3, n_folds).astype(int)
        for vs in val_starts:
            folds.append({"train": slice(0, int(vs)), "val": slice(int(vs), n)})
    if not folds:
        cut = max(min_train, int(n * 0.6))
        if n - cut >= 3:
            folds = [{"train": slice(0, cut), "val": slice(cut, n)}]
    return folds


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    abs_err = np.abs(err)
    m = {
        "n": int(len(y_true)),
        "medae": float(np.median(abs_err)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "within_20pct": float((abs_err <= 0.2 * np.maximum(np.abs(y_true), 1e-9)).mean()),
    }
    return m


def _fit_prepare(elig: pd.DataFrame, feature_names: list[str]):
    X = elig[feature_names].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _fit_log1p(model, Xenc, y):
    from dltime.feature_assembly import TargetTransformer

    tt = TargetTransformer(mode="log1p")
    y_log = tt.transform(pd.Series(np.asarray(y, dtype=float)))
    model.fit(Xenc, y_log)
    return tt


def _inv_log1p(tt, pred_log):
    return tt.inverse_transform(np.asarray(pred_log, dtype=float))


def compare_stage_models(
    stage: str,
    *,
    bundle=None,
    n_folds: int = 4,
    seed: int = RANDOM_STATE,
) -> dict:
    """Multi-family comparison for ONE stage on the FILLED dataset.

    Protocol per candidate (family, hyperparameters):
      * chronological expanding-window CV on the eligible stage rows
        (log1p-transformed target; metrics on original seconds scale);
      * MedAE is the PRIMARY selection metric (robust to heavy tails),
        RMSE tie-break; every candidate's full metrics are returned;
      * final refit on the full eligible set; chronological holdout metrics
        (latest 25%, >= 12 rows) recorded for the WINNER as honest evidence.
    """
    from dltime.feature_assembly import TargetTransformer

    bundle = bundle or load_filled_bundle()
    from dltime.stage_models import build_stage_frame

    frame, feats, diag = build_stage_frame(stage=stage, bundle=bundle)
    elig_all = frame[frame["stage_eligible"].astype(bool)].copy()
    real = real_exec_mask(elig_all)
    n_real = int(real.sum())
    if n_real < STAGE_V2_MIN_EVAL_ROWS:
        return {
            "stage": stage,
            "status": "NOT_ENOUGH_REAL_EVIDENCE",
            "n_real": n_real,
            "n_synthetic": int((~real).sum()),
            "reason": (
                f"Only {n_real} REAL eligible observations for stage '{stage}' "
                f"(synthetic augmentation excluded from evaluation; minimum "
                f"{STAGE_V2_MIN_EVAL_ROWS}). Stage stays honestly unavailable."
            ),
            "candidates": [],
        }

    # chronological holdout = latest 25% of REAL rows (never synthetic)
    elig_real = elig_all[real].sort_values(
        ["Start_Time", "LDR_Execution_Id"], kind="mergesort"
    ).reset_index(drop=True)
    n_hold = max(STAGE_V2_MIN_EVAL_ROWS, int(len(elig_real) * 0.25))
    hold = elig_real.iloc[-n_hold:]
    train_pool = pd.concat([elig_all[~real], elig_real.iloc[:-n_hold]], ignore_index=True)
    # chronological order restored (synthetic rows sit AFTER all real rows in
    # time, which keeps the strictly-prior history property intact)
    train_pool = train_pool.sort_values(
        ["Start_Time", "LDR_Execution_Id"], kind="mergesort"
    ).reset_index(drop=True)

    y_hold = pd.to_numeric(hold[f"stage_{stage}_sec"], errors="coerce").to_numpy(float)

    candidates: list[dict] = []
    for family in CANDIDATE_FAMILIES:
        for grid in HYPERPARAMETER_GRIDS[family]:
            params = dict(grid)
            cv_scores: list[float] = []
            cv_rmse: list[float] = []
            try:
                # CV speed: tree families fit a reduced forest during CV; the
                # FINAL refit (winner + production) always uses the full grid
                # params. Same family/shape comparison, ~2.5x faster protocol.
                cv_params = dict(params)
                if family in ("random_forest", "extra_trees") and "n_estimators" in cv_params:
                    cv_params["n_estimators"] = 150
                for fold in _chronological_folds(train_pool, n_folds=n_folds):
                    tr = train_pool.iloc[fold["train"]]
                    va = train_pool.iloc[fold["val"]]
                    if len(va) < 3 or len(tr) < 8:
                        continue
                    Xtr, Xva = _fit_prepare(tr, feats), _fit_prepare(va, feats)
                    Xtr_e, Xva_e = _encode(Xtr), _encode(Xva, ref_columns=_encode(Xtr).columns)
                    model = _make_estimator(family, cv_params)
                    tt = _fit_log1p(model, Xtr_e, pd.to_numeric(tr[f"stage_{stage}_sec"], errors="coerce"))
                    pred = _inv_log1p(tt, model.predict(Xva_e))
                    m = _metrics(pd.to_numeric(va[f"stage_{stage}_sec"], errors="coerce").to_numpy(float), pred)
                    cv_scores.append(m["medae"])
                    cv_rmse.append(m["rmse"])
            except Exception as exc:  # noqa: BLE001
                candidates.append({
                    "family": family, "params": params, "status": "failed",
                    "error": str(exc)[:120],
                })
                continue
            if not cv_scores:
                candidates.append({
                    "family": family, "params": params, "status": "no_folds",
                })
                continue
            candidates.append({
                "family": family,
                "params": params,
                "status": "ok",
                "cv_medae": float(np.mean(cv_scores)),
                "cv_rmse": float(np.mean(cv_rmse)),
            })

    ok = [c for c in candidates if c.get("status") == "ok"]
    if not ok:
        return {
            "stage": stage, "status": "ALL_CANDIDATES_FAILED",
            "n_real": n_real, "candidates": candidates,
        }
    best = min(ok, key=lambda c: (c["cv_medae"], c["cv_rmse"]))

    # ---- winner's holdout evidence: REFIT ON REAL TRAIN ROWS ONLY ----------
    # Synthetic coverage rows are excluded from the EVIDENCE refit because
    # some synthetic clones were derived from donor executions that may sit
    # in the holdout; mixing them would flatter the metric. The PRODUCTION
    # model below still trains on the full pool (coverage is the point of
    # the fill); the reported holdout metric stays strictly real-only.
    real_train = train_pool[real_exec_mask(train_pool)]
    X_ev = _fit_prepare(real_train, feats)
    X_ev_e = _encode(X_ev)
    evidence_model = _make_estimator(best["family"], dict(best["params"]))
    tt_ev = _fit_log1p(
        evidence_model, X_ev_e, pd.to_numeric(real_train[f"stage_{stage}_sec"], errors="coerce")
    )
    hold_pred = _inv_log1p(tt_ev, evidence_model.predict(_encode(_fit_prepare(hold, feats), ref_columns=X_ev_e.columns)))
    hold_metrics = _metrics(y_hold, hold_pred)

    # final fit: full eligible pool (real holdout kept aside)
    X_pool = _fit_prepare(train_pool, feats)
    X_pool_e = _encode(X_pool)

    # production refit: ALL eligible rows (real + synthetic coverage)
    X_all = _fit_prepare(elig_all, feats)
    X_all_e = _encode(X_all)
    production = _make_estimator(best["family"], dict(best["params"]))
    tt_all = _fit_log1p(
        production, X_all_e, pd.to_numeric(elig_all[f"stage_{stage}_sec"], errors="coerce")
    )

    return {
        "stage": stage,
        "status": "OK",
        "n_real": n_real,
        "n_synthetic": int((~real).sum()),
        "n_holdout_real": int(len(hold)),
        "feature_count": len(feats),
        "candidates": candidates,
        "selected": {"family": best["family"], "params": best["params"],
                     "cv_medae": best["cv_medae"], "cv_rmse": best["cv_rmse"]},
        "holdout_metrics_real_only": hold_metrics,
        "diagnostics": {k: v for k, v in diag.items() if k != "widened_fields_top"},
    }


# --------------------------------------------------------------------------
# Fitted production bundle (serving artifact)
# --------------------------------------------------------------------------

class StageV2Predictor:
    """One stage's fitted v2 predictor (kind/params recorded for the report)."""

    def __init__(self, stage, family, params, model, feature_names, medians, enc_columns, n_real, n_syn):
        self.stage = stage
        self.family = family
        self.params = params
        self.model = model
        self.feature_names = feature_names
        self._medians = medians
        self._enc_columns = enc_columns
        self.n_real = n_real
        self.n_synthetic = n_syn
        self.version = "stage-v2"

    def predict_seconds(self, ctx_row: pd.DataFrame) -> float:
        X = ctx_row[self.feature_names].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(
                    self._medians.get(col, 0.0)
                )
        Xenc = _encode(X, ref_columns=self._enc_columns)
        from dltime.feature_assembly import TargetTransformer

        pred_log = np.asarray(self.model.predict(Xenc), dtype=float)
        return float(TargetTransformer(mode="log1p").inverse_transform(pred_log)[0])


@dataclass
class StageV2Bundle:
    version: str = "stage-v2-bundle-1"
    filled_dataset: str = str(FILLED_DIR)
    predictors: dict[str, StageV2Predictor] = field(default_factory=dict)
    comparisons: dict[str, dict] = field(default_factory=dict)
    unavailable_stages: dict[str, str] = field(default_factory=dict)


def build_stage_v2_bundle(stages=None, *, verbose=False) -> StageV2Bundle:
    """Compare, select, and fit ALL trainable stages.

    2026-09-30: the corrected export now carries real staging timing fields
    (716/792 real rows), so staging participates in the v2 comparison like
    any other stage and is auto-skipped ONLY when it genuinely has no rows.
    """
    all_stages = ("staging", "prevalidation", "transformation", "pre_load", "data_loading")
    stages = all_stages if stages is None else stages
    filled = load_filled_bundle()
    bundle = StageV2Bundle()
    for stage in all_stages:
        res = compare_stage_models(stage, bundle=filled) if stage in stages else {
            "stage": stage, "status": "EXCLUDED_BY_REQUEST",
            "reason": "excluded by caller request",
        }
        bundle.comparisons[stage] = res
        if res.get("status") != "OK":
            bundle.unavailable_stages[stage] = res.get("reason", res.get("status"))
            continue
        # refit production predictor on ALL eligible rows (real + synthetic)
        from dltime.stage_models import build_stage_frame

        frame, feats, _diag = build_stage_frame(stage=stage, bundle=filled)
        elig = frame[frame["stage_eligible"].astype(bool)]
        X = _fit_prepare(elig, feats)
        Xenc = _encode(X)
        model = _make_estimator(res["selected"]["family"], dict(res["selected"]["params"]))
        tt = _fit_log1p(model, Xenc, pd.to_numeric(elig[f"stage_{stage}_sec"], errors="coerce"))
        pred = StageV2Predictor(
            stage=stage,
            family=res["selected"]["family"],
            params=dict(res["selected"]["params"]),
            model=model,
            feature_names=list(feats),
            medians=X.select_dtypes(include=[np.number]).median(),
            enc_columns=list(Xenc.columns),
            n_real=int(res["n_real"]),
            n_syn=int(res["n_synthetic"]),
        )
        bundle.predictors[stage] = pred
        if verbose:
            print(f"[v2] {stage}: {res['selected']['family']} holdout MedAE={res['holdout_metrics_real_only']['medae']:.2f}s")
    return bundle


def save_stage_v2_bundle(bundle: StageV2Bundle, path=None) -> Path:
    import joblib

    path = Path(path) if path else PROJECT_ROOT / "backend" / "artifacts" / "stage-models-v2.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"serving_version": bundle.version, "bundle": bundle}, path)
    return path


def load_stage_v2_bundle(path=None) -> StageV2Bundle | None:
    path = Path(path) if path else PROJECT_ROOT / "backend" / "artifacts" / "stage-models-v2.pkl"
    if not path.exists():
        return None
    import sys
    import types

    import joblib

    # Artifact built by running stage_v2.py as __main__ records the bundle
    # classes under __main__; temporarily shim __main__ so pickle resolves
    # them to dltime.stage_v2 (no rebuild needed), then restore.
    real_main = sys.modules.get("__main__")
    shim = types.ModuleType("__main__")
    shim.StageV2Bundle = StageV2Bundle
    shim.StageV2Predictor = StageV2Predictor
    sys.modules["__main__"] = shim
    try:
        blob = joblib.load(path)
        bundle = blob.get("bundle")
    finally:
        sys.modules["__main__"] = real_main
    if blob.get("serving_version") != "stage-v2-bundle-1" or bundle is None:
        return None
    return bundle


# --------------------------------------------------------------------------
# Threshold gate (the THREE availability cases)
# --------------------------------------------------------------------------

def loader_history_count(combined_executions: pd.DataFrame, loader_name: str) -> int:
    """Strictly-prior TRAINING-ELIGIBLE executions of this loader (real rows only)."""
    df = combined_executions
    m = real_exec_mask(df) & df["is_training_eligible"].fillna(False).astype(bool)
    if loader_name is None or (isinstance(loader_name, float) and pd.isna(loader_name)):
        return 0
    return int((df.loc[m, "Loader_Name"].astype("string") == str(loader_name)).sum())


def assess_loader_case(
    combined_executions: pd.DataFrame,
    loader_name: str,
    *,
    min_prior: int = STAGE_LOADER_MIN_PRIOR,
) -> dict:
    """Classify the request into one of the THREE user cases."""
    n = loader_history_count(combined_executions, loader_name)
    if n == 0:
        return {
            "case": "UNKNOWN_LOADER",
            "case_number": 2,
            "prior_count": n,
            "threshold": min_prior,
            "total_available": False,
            "message": (
                f"Loader '{loader_name}' has no prior executions: the global model can "
                f"show a LOW-CONFIDENCE warm-up estimate only. No total is served until "
                f"this loader reaches {min_prior} recorded executions."
            ),
        }
    if n < min_prior:
        return {
            "case": "BELOW_THRESHOLD",
            "case_number": 3,
            "prior_count": n,
            "threshold": min_prior,
            "total_available": False,
            "message": (
                f"Loader '{loader_name}' has {n} prior execution(s) but the trust "
                f"threshold is {min_prior}: predictions are shown per stage as "
                f"indicative, but no trusted total is served yet. "
                f"{min_prior - n} more recorded execution(s) needed."
            ),
        }
    return {
        "case": "KNOWN_LOADER_TRUSTED",
        "case_number": 1,
        "prior_count": n,
        "threshold": min_prior,
        "total_available": True,
        "message": (
            f"Loader '{loader_name}' has {n} prior executions (>= threshold "
            f"{min_prior}): loader-specific predictions and the stage-sum total "
            f"are available."
        ),
    }


# --------------------------------------------------------------------------
# Sum-of-available-stages v2 (total including approvals)
# --------------------------------------------------------------------------

def sum_total_v2(stage_intel: dict, loader_case: dict) -> dict:
    """Total = sum(available operational stage ML predictions) + approvals.

    Approvals are added when their contextual statistical estimates are
    available (they are part of the total per the user contract; they are
    statistical, not ML). Fail-closed: if the gate blocks or an APPLICABLE
    operational stage is missing without an honest unavailability record, no
    total.

    2026-09-30 (user final spec): only APPLICABLE stages participate. The
    applicable set comes from the user's pre-execution entries — a loader
    whose execution runs 3 stages must never be refused because a stage it
    does not run has no prediction.
    """
    stages = stage_intel.get("stages", [])
    by_stage = {s.get("stage"): s for s in stages if isinstance(s, dict)}
    applicable = stage_intel.get("applicable_stages")
    applicable_set = set(applicable) if applicable else None
    if not loader_case.get("total_available"):
        return {
            "sum_version": SUM_VERSION,
            "sum_available": False,
            "gate": loader_case,
            "reason": loader_case["message"],
            "total_seconds": None,
        }
    unavailable = []
    skipped_not_applicable = []
    stage_sum = 0.0
    stage_parts: dict[str, float] = {}
    # 2026-09-30: staging is trained in the v2 bundle (real staging timing
    # fields merged into the export), so it participates in the total like
    # every other operational stage.
    for stage in ("staging", "prevalidation", "transformation", "pre_load", "data_loading"):
        if applicable_set is not None and stage not in applicable_set:
            skipped_not_applicable.append(stage)
            continue
        entry = by_stage.get(stage)
        if entry is None or not entry.get("available") or entry.get("estimate_seconds") is None:
            unavailable.append(stage)
            continue
        stage_sum += float(entry["estimate_seconds"])
        stage_parts[stage] = float(entry["estimate_seconds"])
    if unavailable:
        return {
            "sum_version": SUM_VERSION,
            "sum_available": False,
            "gate": loader_case,
            "unavailable_stages": unavailable,
            "reason": (
                f"Total unavailable: applicable operational stage(s) without an ML "
                f"prediction: {sorted(unavailable)}. The sum is refused (no partial total)."
            ),
            "total_seconds": None,
        }
    # 2026-10-02 (user final spec): ONLY approvals backed by the loader's OWN
    # approval history (evidence tier LOADER, included_in_total=True) join the
    # total. Broader-context estimates (CONTEXT_GROUP / GLOBAL / …) are
    # disclosed in excluded_approvals but NEVER summed — for a loader with no
    # approval history of its own, no approval enters the total.
    approvals_sum = 0.0
    approvals_detail: dict[str, float] = {}
    excluded_approvals: dict[str, dict] = {}
    for name in ("datamart_approval", "dataloading_approval"):
        if applicable_set is not None and name not in applicable_set:
            continue
        entry = by_stage.get(name)
        if not entry or not entry.get("available") or entry.get("estimate_seconds") is None:
            continue
        if entry.get("included_in_total") is False:
            excluded_approvals[name] = {
                "estimate_seconds": float(entry["estimate_seconds"]),
                "evidence_tier": entry.get("evidence_tier"),
                "n_evidence": entry.get("n_evidence"),
                "reason": entry.get("total_exclusion_reason"),
            }
            continue
        approvals_sum += float(entry["estimate_seconds"])
        approvals_detail[name] = float(entry["estimate_seconds"])
    return {
        "sum_version": SUM_VERSION,
        "sum_available": True,
        "gate": loader_case,
        "stage_parts_seconds": stage_parts,
        "approvals_seconds": approvals_detail,
        "approvals_total_seconds": round(approvals_sum, 3),
        "excluded_approvals": excluded_approvals,
        "operational_total_seconds": round(stage_sum, 3),
        "total_seconds": round(stage_sum + approvals_sum, 3),
        "applicable_stages": sorted(applicable_set) if applicable_set else None,
        "not_applicable_stages": sorted(skipped_not_applicable) if skipped_not_applicable else [],
        "note": (
            "Total = exact sum of the applicable ML stage predictions + the "
            "statistical approval estimates backed by this loader's OWN approval "
            "history (LOADER tier only). Broader-context approval evidence is "
            "disclosed but excluded from the total. Model/statistical outputs, "
            "never actuals."
        ),
    }


# --------------------------------------------------------------------------
# Comparison report (JSON-serializable technical evidence)
# --------------------------------------------------------------------------

def comparison_report(bundle: StageV2Bundle) -> dict:
    """Structured per-stage report: families compared, params, metrics, decision."""
    out: dict[str, dict] = {}
    for stage, res in bundle.comparisons.items():
        if res.get("status") == "OK":
            ranked = sorted(
                [c for c in res["candidates"] if c.get("status") == "ok"],
                key=lambda c: (c["cv_medae"], c["cv_rmse"]),
            )
            out[stage] = {
                "status": "OK",
                "n_real": res["n_real"],
                "n_synthetic": res["n_synthetic"],
                "selected": res["selected"],
                "holdout_real_only": res["holdout_metrics_real_only"],
                "leaderboard": ranked,
                "candidates_failed": [
                    c for c in res["candidates"] if c.get("status") != "ok"
                ],
            }
        else:
            out[stage] = {k: v for k, v in res.items() if k != "candidates"}
    return out


if __name__ == "__main__":  # pragma: no cover
    b = build_stage_v2_bundle(verbose=True)
    save_stage_v2_bundle(b)
    rep = comparison_report(b)
    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "stage_v2_comparison.json"
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("saved artifact + docs/stage_v2_comparison.json")
