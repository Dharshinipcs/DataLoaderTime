"""Stage-aware total model + direct-total baseline comparison (AD-31, §15-§17).

Locked architecture: the final ETA is NOT the sum of the seven stage
estimates (the EXP-032 audit shows the operational stage windows are strictly
sequential/DISJOINT with gaps — 0 overlaps across all 10 stage pairs — and
the true union of stage windows covers only ~31% (median) of the total, so a
sum cannot reconstruct the total; the remainder is inter-stage/queue time).
Instead a stage-aware TOTAL model learns the mapping

    7-stage vector (5 predicted stage durations + 2 approval estimates)
    + legitimate pre-execution context features
    + causally valid historical features
      ->  authoritative total seconds
      ->  total_timetaken_in_min * 60  (AD-34/EXP-032 authoritative target:
          Start/End-span-consistent on 757/757 rows; the legacy
          timetaken_in_sec column is corrupted — disagrees with the span on
          170/172 co-observed rows, sometimes ~1000x)

CRITICAL LEAKAGE SAFETY (§17): the stage columns fed to the stage-aware total
model during TRAINING are OUT-OF-FOLD (OOF) stage predictions produced inside
the chronological training window only — never the row's own actual stage
durations, and never actuals from any later row. At INFERENCE the stage
columns are the stage models' PREDICTIONS for the context row (plus the
approval hierarchical estimates). Actual stage durations never enter any
feature matrix.

Comparison protocol (§16): identical chronological population and identical
folds for the direct-total baseline (the existing 30-feature architecture) and
the stage-aware candidate; pooled OOF + chronological holdout + unseen-loader
+ history-depth slices. The stage-aware model can only become a CANDIDATE on
this evidence; promotion remains an explicit human lifecycle action and is
NOT performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STAGE_AWARE_VERSION = "stage-aware-total-v2-ad34"  # AD-34: corrected total target


class StageAwareError(ValueError):
    """Raised when the stage-aware evaluation cannot be performed safely."""


@dataclass
class StageAwareComparison:
    direct: dict
    stage_aware: dict
    verdict: dict
    population: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "direct_total": self.direct,
            "stage_aware_total": self.stage_aware,
            "verdict": self.verdict,
            "population": self.population,
            "version": STAGE_AWARE_VERSION,
        }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from dltime.modeling import compute_metrics

    m = compute_metrics(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y_true - y_pred) / np.where(y_true > 0, y_true, np.nan)
    m["within_20pct"] = float(np.nanmean(rel <= 0.20) * 100.0)
    return m


def _fit_predict_total(kind: str, X_train: pd.DataFrame, y_train_log: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    from dltime.stage_models import _fit_predict

    return _fit_predict(kind, X_train, y_train_log, X_val)


def _encode(train_X: pd.DataFrame, other_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.get_dummies(train_X.astype(object).where(train_X.notna(), None), dummy_na=True)
    ot = pd.get_dummies(other_X.astype(object).where(other_X.notna(), None), dummy_na=True).reindex(columns=tr.columns, fill_value=0)
    return tr, ot


# Process-level OOF cache: the OOF columns depend only on the stage frame
# (read-only within a process), not on the total-population window. Keyed by
# stage name; reset with clear_stage_oof_cache().
_STAGE_OOF_CACHE: dict[str, pd.Series] = {}


def clear_stage_oof_cache() -> None:
    """Reset the OOF prediction cache (tests / fresh-source scenarios)."""
    _STAGE_OOF_CACHE.clear()


def _stage_oof_predictions(
    stage: str,
    frame: pd.DataFrame,
    feature_names: list[str],
    rank_bounds: dict[int, tuple[int, int]],
) -> pd.Series:
    """OOF stage predictions aligned to the frame's eligible rows.

    For every eligible row of `frame` (ordered chronologically), the stage
    model is trained on strictly-earlier stage-eligible rows of the SAME
    stage frame and predicts that row. Rows lacking the minimum training
    evidence get NaN (handled downstream by the total model's imputation).
    """
    from dltime.stage_models import _fit_predict
    from dltime.feature_assembly import TargetTransformer

    if stage in _STAGE_OOF_CACHE:
        return _STAGE_OOF_CACHE[stage]

    target_col = f"stage_{stage}_sec"
    eligible_idx = frame.index[frame["stage_eligible"].astype(bool)]
    ordered = frame.loc[eligible_idx]
    X_all = ordered[feature_names].copy()
    for col in X_all.columns:
        if X_all[col].dtype == "string" or isinstance(X_all[col].dtype, pd.CategoricalDtype):
            X_all[col] = X_all[col].fillna("__MISSING__").astype("string")
        else:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
    y_log_all = TargetTransformer(mode="log1p").transform(
        pd.to_numeric(ordered[target_col], errors="coerce")
    )

    # ---- OOF cadence + estimator (documented trade-offs) --------------------
    # Block-cadence expanding OOF: refit every REFIT_EVERY rows with a model
    # trained on rows STRICTLY BEFORE the block start, then reuse it for every
    # row inside the block. Strictly-prior safety is unchanged (the model has
    # only ever seen earlier rows), while refits drop from ~300 to ~15 per
    # stage. Estimator: 60-tree ExtraTrees — the same family/inductive bias as
    # the registered 400-tree stage candidate at ~1/7 the cost; registered
    # stage-model evaluations keep the full configuration.
    #
    # Small-sample stages (AD-31a: n < 40, e.g. pre_load n=19): min_train is
    # lowered to 10 and REFIT_EVERY to 5 so the earliest rows still receive a
    # leakage-safe OOF prediction from a real fitted model; rows before
    # min_train remain NaN (total model imputes). This is an explicitly
    # DOCUMENTED APPROXIMATION: block reuse means a block's model is up to
    # REFIT_EVERY rows behind the row being predicted — training data always
    # STRICTLY precedes the block start, so no future information can leak.
    from sklearn.ensemble import ExtraTreesRegressor

    small_sample = len(ordered) < 40
    REFIT_EVERY = 5 if small_sample else 20
    min_train = 10 if small_sample else 30
    oof = pd.Series(np.nan, index=ordered.index, dtype=float)
    est = None
    enc_cols: pd.Index | None = None
    for pos in range(min_train, len(ordered)):
        if est is None or pos % REFIT_EVERY == 0:
            Xtr = X_all.iloc[:pos]
            Xtr_enc = pd.get_dummies(Xtr.astype(object).where(Xtr.notna(), None), dummy_na=True)
            enc_cols = Xtr_enc.columns
            est = ExtraTreesRegressor(n_estimators=60, random_state=42, n_jobs=1, min_samples_leaf=2)
            est.fit(Xtr_enc, y_log_all[:pos])
        Xrow = pd.get_dummies(
            X_all.iloc[pos : pos + 1].astype(object).where(X_all.iloc[pos : pos + 1].notna(), None),
            dummy_na=True,
        ).reindex(columns=enc_cols, fill_value=0)
        oof.iloc[pos] = float(
            TargetTransformer(mode="log1p").inverse_transform(est.predict(Xrow))[0]
        )
    _STAGE_OOF_CACHE[stage] = oof
    return oof


def evaluate_stage_aware_total(
    *,
    bundle=None,
    total_model_kind: str = "extra_trees",
    n_folds: int = 4,
) -> StageAwareComparison:
    """Chronological comparison: direct-total baseline vs stage-aware total.

    Population: rows satisfying the STAGE training policy (loader+sprint
    identity + completed status + positive total target) so both models see
    the identical rows; the direct model uses its 30-feature contract, the
    stage-aware model adds the 7-vector of OOF stage predictions. Folds,
    holdout, unseen-loader and history-bin slices are identical.
    """
    from dltime.stage_models import build_stage_frame
    from dltime.stage_targets import ML_STAGE_NAMES, APPROVAL_STAGE_NAMES, stage_target_column
    from dltime.feature_assembly import TargetTransformer
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.modeling import add_chronological_rank
    from dltime.sources import load_all_sources

    bundle = bundle or load_all_sources()

    # ---- base total frame on the stage population ---------------------------
    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    # AD-34 target correction: total_timetaken_in_min * 60 is the authoritative
    # total (matches the Start/End span on 757/757 rows); the legacy
    # timetaken_in_sec column is corrupted (170/172 disagreements, median ratio
    # 2.26 vs the span) and must never define the training target.
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = total_frame[pop].copy()
    y_raw = target[pop]
    y_log = TargetTransformer(mode="log1p").transform(y_raw)

    frame = add_chronological_rank(frame)
    X_direct = frame[total_feats].copy()
    for col in X_direct.columns:
        if X_direct[col].dtype == "string" or isinstance(X_direct[col].dtype, pd.CategoricalDtype):
            X_direct[col] = X_direct[col].fillna("__MISSING__").astype("string")
        else:
            X_direct[col] = pd.to_numeric(X_direct[col], errors="coerce")

    # ---- OOF stage columns (per stage, chronological, leakage-safe) ---------
    # AD-31a: ALL FIVE operational stages contribute their leakage-safe OOF
    # stage predictions — small-sample stages (n < 40) use the documented
    # small-sample cadence above instead of being dropped to NaN columns.
    stage_cols: dict[str, pd.Series] = {}
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
        if sdiag["stage_eligible_rows"] >= 10:
            oof = _stage_oof_predictions(stage, sframe, sfeats, {})
            # oof is indexed by position within the stage frame's eligible
            # ordering; map through the frame's execution ids.
            lookup = pd.Series(
                oof.values,
                index=sframe.loc[oof.index, "LDR_Execution_Id"].to_numpy(),
            )
            stage_cols[stage] = lookup
        else:
            stage_cols[stage] = pd.Series(dtype=float)

    # ---- approval OOF columns: strictly-prior hierarchical estimate ---------
    for stage in APPROVAL_STAGE_NAMES:
        col = stage_target_column(stage)
        estimates: list[float] = []
        ordered = frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
        for exec_id in ordered["LDR_Execution_Id"]:
            est = estimate_approval_stage(frame, exec_id, stage)
            estimates.append(est.estimate_seconds if est.available else np.nan)
        stage_cols[stage] = pd.Series(estimates, index=ordered.index)

    # assemble the stage-aware matrix on the total population: map each
    # execution id to its OOF stage prediction (NaN where the stage model
    # lacked the minimum chronological training evidence — the total model's
    # median imputation handles it exactly like any other missing feature).
    X_stage = X_direct.copy()
    ids = frame["LDR_Execution_Id"]
    for stage, lookup in stage_cols.items():
        colname = f"pred_stage_{stage}_sec"
        X_stage[colname] = pd.to_numeric(ids.map(lookup), errors="coerce").astype(float) if len(lookup) else np.nan

    n = len(frame)
    loader_arr = frame["Loader_Name"].astype("string").to_numpy()
    prior_count = X_direct.get("stage_hist_loader_prior_count", pd.Series(0, index=X_direct.index)).to_numpy()

    def run(model_matrix: pd.DataFrame, kind: str) -> dict:
        from dltime.modeling import expanding_window_folds

        fold_metrics: list[dict] = []
        try:
            folds_spec = expanding_window_folds(frame, n_folds=n_folds, initial_train_fraction=0.5)
        except ValueError:
            folds_spec = []
        for f in folds_spec:
            tr = slice(f["train_rank_start"], f["train_rank_end_exclusive"])
            va = slice(f["val_rank_start"], f["val_rank_end_exclusive"])
            if f["val_rows"] < 3:
                continue
            Xtr, Xva = _encode(model_matrix.iloc[tr], model_matrix.iloc[va])
            pred_log = _fit_predict_total(kind, Xtr, y_log[tr], Xva)
            pred = TargetTransformer(mode="log1p").inverse_transform(pred_log)
            fold_metrics.append({"fold": f["fold"], **_metrics(y_raw.to_numpy()[va], pred)})

        n_test = max(int(round(n * 0.25)), 12)
        cut = n - n_test
        boundary = frame.iloc[cut]["Start_Time"]
        while cut > 0 and frame.iloc[cut - 1]["Start_Time"] == boundary:
            cut -= 1
        Xtr, Xte = _encode(model_matrix.iloc[:cut], model_matrix.iloc[cut:])
        pred_log = _fit_predict_total(kind, Xtr, y_log[:cut], Xte)
        pred = TargetTransformer(mode="log1p").inverse_transform(pred_log)
        holdout = _metrics(y_raw.to_numpy()[cut:], pred)

        train_loaders = set(loader_arr[:cut].tolist())
        unseen_mask = np.array([l not in train_loaders for l in loader_arr[cut:]])
        unseen = (
            _metrics(y_raw.to_numpy()[cut:][unseen_mask], pred[unseen_mask])
            if unseen_mask.sum() >= 3
            else {"n": int(unseen_mask.sum()), "status": "INSUFFICIENT_SAMPLE"}
        )
        bins = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, np.inf, "10+")]
        hist_bins = {}
        pc_h = prior_count[cut:]
        for lo, hi, label in bins:
            m = (pc_h >= lo) & (pc_h <= hi)
            hist_bins[label] = (
                _metrics(y_raw.to_numpy()[cut:][m], pred[m])
                if m.sum() >= 3
                else {"n": int(m.sum()), "status": "INSUFFICIENT_SAMPLE"}
            )
        pooled = None
        if fold_metrics:
            keys = ("mae", "median_ae", "rmse", "within_20pct", "within_2x")
            pooled = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
            pooled["n_folds"] = len(fold_metrics)
            pooled["total_val_rows"] = int(sum(m["n"] for m in fold_metrics))
        return {
            "model_kind": kind,
            "pooled_oof": pooled,
            "holdout": holdout,
            "unseen_loader": unseen,
            "history_bins": hist_bins,
        }

    direct = run(X_direct, total_model_kind)
    stage_aware = run(X_stage, total_model_kind)

    # ---- verdict: numbers only, no auto-promotion ---------------------------
    def better(a: dict, b: dict) -> bool:
        # stage-aware (a) must not be worse on pooled MAE/RMSE and must win
        # the chronological holdout MAE to be considered a candidate.
        pa, pb = a["pooled_oof"] or {}, b["pooled_oof"] or {}
        if not pa or not pb:
            return False
        mae_ok = pa["mae"] <= pb["mae"]
        rmse_ok = pa["rmse"] <= pb["rmse"]
        hold_ok = a["holdout"]["mae"] < b["holdout"]["mae"]
        return mae_ok and rmse_ok and hold_ok

    sa_better = better(stage_aware, direct)
    verdict = {
        "stage_aware_better_by_rule": sa_better,
        "rule": "pooled MAE <=, pooled RMSE <=, holdout MAE strictly <",
        "decision": (
            "CANDIDATE_EXPERIMENTAL — eligible for explicit human lifecycle review"
            if sa_better
            else "KEEP_DIRECT_BASELINE — stage-aware remains experimental"
        ),
        "promoted": False,
        "promotion_policy": "explicit human lifecycle action only (AD-31); nothing is auto-promoted",
    }
    population = {
        "rows": int(n),
        "stage_columns": [f"pred_stage_{s}_sec" for s in ML_STAGE_NAMES]
        + [f"pred_stage_{s}_sec" for s in APPROVAL_STAGE_NAMES],
        "total_features_direct": len(total_feats),
        "total_features_stage_aware": X_stage.shape[1],
    }
    return StageAwareComparison(direct=direct, stage_aware=stage_aware, verdict=verdict, population=population)
