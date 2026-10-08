"""Stage model training + experiments (AD-31, prompt §10-§14).

Five dedicated ML stage models — staging, prevalidation, transformation,
pre_load, data_loading — each with:

  * a stage-specific target (stage_targets.build_stage_targets; AD-31 chain),
  * stage-specific eligibility (documented deviation from the total-model
    identity gate: the corrected source's Dataset_display_name is populated
    on only 36/759 raw rows, so requiring the full canonical identity would
    strand every stage model at n<=35 — the documented stage training policy
    requires (loader, sprint) identity completeness + completed status +
    positive stage target instead; EXP-028 audit),
  * stage-specific features: legitimate pre-execution configuration (def_*)
    plus AS-OF strictly-prior historical aggregates of THE SAME STAGE'S
    duration (prior count/mean/median/std per identity group; no future rows,
    no same-execution post-execution information — the context row itself is
    never usable, is_training_eligible=False),
  * log1p target transformation with expm1 inversion (locked §11),
  * chronological evaluation ONLY: expanding-window folds + final
    chronological holdout + unseen-loader + history-depth slices (§13).

Model comparison (§12) uses the existing compatible stack: ExtraTrees /
RandomForest / HistGradientBoosting / Ridge / historical median baseline.
Every stage model is registered as EXPERIMENTAL in the Model Laboratory
registry with full lineage. NO automatic promotion ever happens (§14).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STAGE_MODELS_VERSION = "stage-models-v1"

# Documented stage training-eligibility policy (AD-31; EXP-028 data audit;
# amended by the final validation pass — AD-31a).
#
# Two-tier evidence policy:
#   n >= 40                 full chronological evaluation (4 folds + holdout)
#   12 <= n < 40            SMALL-SAMPLE evaluation (2 folds + holdout): the
#                           stage still gets a REAL fitted ML model, honestly
#                           labelled with its small-sample limitation — a
#                           sparse dataset is a limitation, not permission to
#                           silently remove a required operational stage.
#   n < 12                  NOT_ENOUGH_EVIDENCE (a defensible chronological
#                           holdout of >= 12 rows is impossible).
STAGE_MIN_TRAIN_ROWS = 40
STAGE_MIN_SMALL_SAMPLE_ROWS = 12

STAGE_MODEL_CANDIDATES: dict[str, str] = {
    "extra_trees": "extra_trees",
    "random_forest": "random_forest",
    "hist_gradient_boosting": "hist_gradient_boosting",
    "ridge": "ridge",
    "historical_median_baseline": "historical_median_baseline",
}

# Chronologically-selected serving kind per stage (SELECTION records;
# holdout-MAE rule among ML candidates — AD-31a: every operational stage
# serves a real ML model; the historical median baseline stays registered as
# a non-ML reference only). 2026-09-29 v2 reality (see stage_v2.py):
# staging = UNAVAILABLE (the replaced export carries NO staging timing
# fields; 0 observations, honestly never fabricated) and pre_load is
# small-sample. The v2 bundle (stage-models-v2.pkl) supersedes this registry
# for serving display when present.
# pre_load: EXPERIMENTAL_SMALL_SAMPLE (n=19 — real
# fitted model, honest small-sample limitation).
STAGE_MODEL_SERVING_KINDS: dict[str, dict] = {
    # 2026-09-30: corrected export now carries real staging timing fields
    # (716/792 rows); staging trained in the v2 bundle (extra_trees, 326 real
    # eligible rows). Display truth for v2 serving comes from the predictors
    # themselves; this registry stays accurate for fallback/rollback paths.
    "staging": {"model_kind": "extra_trees", "status": "EXPERIMENTAL", "n_train": 326},
    "prevalidation": {"model_kind": "extra_trees", "status": "EXPERIMENTAL", "n_train": 66},
    "transformation": {"model_kind": "extra_trees", "status": "EXPERIMENTAL", "n_train": 62},
    "pre_load": {"model_kind": "extra_trees", "status": "EXPERIMENTAL_SMALL_SAMPLE", "n_train": 23},
    "data_loading": {"model_kind": "extra_trees", "status": "EXPERIMENTAL", "n_train": 323},
}


class StageModelError(ValueError):
    """Raised when a stage model cannot be built without guessing."""


# ---------------------------------------------------------------- estimators


def _make_estimator(kind: str) -> object:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400, random_state=42, n_jobs=1, min_samples_leaf=2
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=1, min_samples_leaf=2
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=42, max_iter=300)
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    raise StageModelError(f"Unknown stage estimator kind '{kind}'")


def _fit_predict(kind: str, X_train, y_train_log, X_val) -> np.ndarray:
    if kind == "historical_median_baseline":
        # Baseline: predict the training-set median of log1p(target) for every
        # row (the transparent non-ML reference every learned model must beat).
        return np.full(len(X_val), float(np.median(y_train_log)), dtype=float)
    est = _make_estimator(kind)
    est.fit(X_train, y_train_log)
    return np.asarray(est.predict(X_val), dtype=float)


# ------------------------------------------------------- stage feature frames


# Module-level stage-frame cache (corrected source is read-only within a
# process; building a stage frame costs a full foundation reconstruction,
# measured ~90 s, and evaluate_stage_model + evaluate_stage_aware_total need
# the same frames repeatedly). Keyed by stage name; callers receive the shared
# frame and MUST treat it as read-only. Tests reset via clear_stage_frame_cache().
_STAGE_FRAME_CACHE: dict[str, tuple[pd.DataFrame, list[str], dict]] = {}


def clear_stage_frame_cache() -> None:
    """Reset the stage-frame cache (tests / fresh-source scenarios)."""
    _STAGE_FRAME_CACHE.clear()


def build_stage_frame(
    *,
    bundle,
    stage: str,
    cache: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """One stage's leakage-safe training frame (cached per stage per process).

    Returns (frame, feature_names, diagnostics). The frame carries:
      identity/analytical columns, the stage target column, eligibility flag
      (stage_eligible), pre-execution def_* features, and AS-OF prior-stage
      historical aggregates (stage_hist_*) computed by a strict chronological
      streaming prefix over training-eligible rows with a positive stage
      target. The context/row itself is excluded by construction (update AFTER
      reading the row — identical pattern to dltime.historical).
    """
    from dltime.pipeline import build_foundation
    from dltime.stage_targets import stage_target_column
    from dltime.definition_features import attach_definition_features

    if cache and stage in _STAGE_FRAME_CACHE:
        return _STAGE_FRAME_CACHE[stage]

    target_col = stage_target_column(stage)

    foundation = build_foundation(bundle)
    # The foundation frame already carries the stage-target columns
    # (pipeline.py Part F — single source of truth) plus definition features
    # are attached here per-stage frame (they are derived, not stored).
    df = foundation.frames["executions_eligible"]

    # definition (pre-execution configuration) features by canonical identity
    df = attach_definition_features(
        df, bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )

    # ---- stage training eligibility (documented stage policy, AD-31) ------
    completed = df["LDR_Status"].isin(
        {
            "SUCCEED",
            "COMPLETED_WITH_WARNING",
            "COMPLETED_WITH_ERROR",
            "COMPLETED_WITH_ERROR_AND_WARNING",
            "ERROR",
        }
    )
    id_ok = df["Loader_Name"].notna() & df["Sprint"].notna()
    target = pd.to_numeric(df[target_col], errors="coerce")
    positive = target > 0
    df["stage_eligible"] = (completed & id_ok & positive).fillna(False)
    reasons = []
    for i in df.index:
        r = []
        if not bool(id_ok.iloc[i]):
            r.append("incomplete_identity")
        if not bool(completed.iloc[i]):
            r.append("status_not_completed")
        if not bool(positive.iloc[i]):
            r.append("missing_or_nonpositive_stage_target")
        reasons.append(";".join(r))
    df["stage_eligibility_reasons"] = reasons

    # ---- AS-OF prior-stage historical aggregates (strictly prior only) -----
    ordered = df.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    ordered["_ts"] = pd.to_datetime(ordered["Start_Time"], errors="coerce")
    if ordered["_ts"].isna().any():
        raise StageModelError("Every execution needs Start_Time for chronological stage history.")

    usable = ordered["stage_eligible"].astype(bool) & (pd.to_numeric(ordered[target_col], errors="coerce") > 0)
    values = pd.to_numeric(ordered[target_col], errors="coerce")

    loader = ordered["Loader_Name"].astype("string")
    sprint = ordered["Sprint"].astype("string")
    dataset = ordered["Dataset_display_name"].astype("string")
    conn = ordered["execution_connection"].astype("string") if "execution_connection" in ordered.columns else pd.Series(pd.NA, index=ordered.index, dtype="string")

    seen_all: list[float] = []
    seen_loader: dict[str, list[float]] = {}
    seen_loader_conn: dict[tuple[str, str], list[float]] = {}
    seen_loader_dataset: dict[tuple[str, str], list[float]] = {}
    seen_composite: dict[tuple[str, str, str], list[float]] = {}
    prev_value: dict[str, float] = {}

    def stats(vals: list[float]) -> tuple[int, float, float, float]:
        n = len(vals)
        if n == 0:
            return 0, np.nan, np.nan, np.nan
        s = pd.Series(vals, dtype="float64")
        return n, float(s.mean()), float(s.median()), float(s.std(ddof=1)) if n >= 2 else np.nan

    rows: list[dict] = []
    for i in range(len(ordered)):
        L, S, D, C = loader.iat[i], sprint.iat[i], dataset.iat[i], conn.iat[i]
        kL = str(L) if pd.notna(L) else None
        row: dict[str, object] = {}
        for prefix, vals in (
            ("stage_hist_global", seen_all),
            ("stage_hist_loader", seen_loader.get(kL, [])),
            ("stage_hist_loader_conn", seen_loader_conn.get((kL, str(C)) if kL and pd.notna(C) else None, [])),
            ("stage_hist_loader_dataset", seen_loader_dataset.get((kL, str(D)) if kL and pd.notna(D) else None, [])),
            ("stage_hist_composite", seen_composite.get((str(D), str(S), str(L)) if all(pd.notna(x) for x in (D, S, L)) else None, [])),
        ):
            n, mean, med, std = stats(vals)
            row[f"{prefix}_prior_count"] = n
            row[f"{prefix}_prior_mean_sec"] = mean
            row[f"{prefix}_prior_median_sec"] = med
            row[f"{prefix}_prior_std_sec"] = std
        row["stage_hist_prev_stage_sec"] = (
            prev_value.get(kL) if kL else np.nan
        )
        rows.append(row)
        # update AFTER reading the row (strictly-prior guarantee)
        if bool(usable.iat[i]):
            v = float(values.iat[i])
            seen_all.append(v)
            if kL:
                seen_loader.setdefault(kL, []).append(v)
                if pd.notna(C):
                    seen_loader_conn.setdefault((kL, str(C)), []).append(v)
                if pd.notna(D):
                    seen_loader_dataset.setdefault((kL, str(D)), []).append(v)
                prev_value[kL] = v
            if kL and pd.notna(D) and pd.notna(S):
                seen_composite.setdefault((str(D), str(S), str(L)), []).append(v)

    hist_df = pd.DataFrame(rows, index=ordered.index)
    ordered = pd.concat([ordered.drop(columns=["_ts"]), hist_df], axis=1)

    FEATURE_NAMES = (
        [c for c in ordered.columns if c.startswith("def_") and c != "def_match"]
        + [c for c in ordered.columns if c.startswith("stage_hist_")]
    )
    diagnostics = {
        "stage": stage,
        "target_column": target_col,
        "executions": int(len(ordered)),
        "stage_eligible_rows": int(ordered["stage_eligible"].sum()),
        "target_source_counts": ordered[f"stage_{stage}_source"].value_counts().to_dict(),
        "feature_count": len(FEATURE_NAMES),
    }
    result = (ordered, FEATURE_NAMES, diagnostics)
    if cache:
        _STAGE_FRAME_CACHE[stage] = result
    return result


# ------------------------------------------------------------------ evaluation


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from dltime.modeling import compute_metrics

    m = compute_metrics(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y_true - y_pred) / np.where(y_true > 0, y_true, np.nan)
    m["within_20pct"] = float(np.nanmean(rel <= 0.20) * 100.0)
    return m


@dataclass
class StageEvaluation:
    stage: str
    model_name: str
    n_train_total: int
    folds: list[dict] = field(default_factory=list)
    pooled_oof: dict | None = None
    holdout: dict | None = None
    unseen_loader: dict | None = None
    history_bins: dict = field(default_factory=dict)
    status: str = "OK"  # OK | NOT_IMPLEMENTED
    status_reason: str | None = None

    def to_payload(self) -> dict:
        return {
            "stage": self.stage,
            "model_name": self.model_name,
            "status": self.status,
            "status_reason": self.status_reason,
            "n_train_total": self.n_train_total,
            "folds": self.folds,
            "pooled_oof": self.pooled_oof,
            "holdout": self.holdout,
            "unseen_loader": self.unseen_loader,
            "history_bins": self.history_bins,
        }


def evaluate_stage_model(
    stage: str,
    model_kind: str,
    *,
    frame: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    n_folds: int = 4,
) -> StageEvaluation:
    """Chronological evaluation of one candidate for one stage (§13).

    Strict protocol: expanding-window folds (train strictly earlier than
    validation), then a final chronological holdout (latest 25%, min 12 rows).
    All metrics computed on the ORIGINAL seconds scale after expm1 inversion.
    """
    from dltime.modeling import expanding_window_folds, add_chronological_rank
    from dltime.feature_assembly import TargetTransformer

    if frame is None or feature_names is None:
        from dltime.sources import load_all_sources

        frame, feature_names, _diag = build_stage_frame(
            stage=stage, bundle=load_all_sources()
        )
    target_col = f"stage_{stage}_sec"
    eligible = frame[frame["stage_eligible"].astype(bool)].copy()
    n_total = int(len(eligible))
    if n_total < STAGE_MIN_SMALL_SAMPLE_ROWS:
        return StageEvaluation(
            stage=stage,
            model_name=model_kind,
            n_train_total=n_total,
            status="NOT_ENOUGH_EVIDENCE",
            status_reason=(
                f"Insufficient eligible stage observations: {n_total} < "
                f"{STAGE_MIN_SMALL_SAMPLE_ROWS} (a defensible chronological "
                "holdout of >= 12 rows is impossible)."
            ),
        )
    small_sample = n_total < STAGE_MIN_TRAIN_ROWS
    if small_sample:
        n_folds = 2  # documented small-sample cadence (AD-31a)

    eligible = add_chronological_rank(eligible)
    X_all = eligible[feature_names].copy()
    for col in X_all.columns:
        if X_all[col].dtype == "string" or isinstance(X_all[col].dtype, pd.CategoricalDtype):
            X_all[col] = X_all[col].fillna("__MISSING__").astype("string")
        else:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
    y_all = pd.to_numeric(eligible[target_col], errors="coerce").to_numpy(dtype=float)
    transformer = TargetTransformer(mode="log1p")
    y_log_all = transformer.transform(pd.Series(y_all))

    loader_arr = eligible["Loader_Name"].astype("string").to_numpy()
    prior_count = X_all.get("stage_hist_loader_prior_count", pd.Series(0, index=X_all.index)).to_numpy()

    # ---- expanding-window folds ----
    try:
        folds_spec = expanding_window_folds(eligible, n_folds=n_folds, initial_train_fraction=0.5)
    except ValueError:
        folds_spec = []
    fold_metrics: list[dict] = []
    for f in folds_spec:
        tr = slice(f["train_rank_start"], f["train_rank_end_exclusive"])
        va = slice(f["val_rank_start"], f["val_rank_end_exclusive"])
        if f["val_rows"] < 3:
            continue
        Xtr = X_all.iloc[tr].copy()
        Xtr = pd.get_dummies(Xtr.astype(object).where(Xtr.notna(), None), dummy_na=True)
        Xva = pd.get_dummies(X_all.iloc[va].astype(object).where(X_all.iloc[va].notna(), None), dummy_na=True).reindex(columns=Xtr.columns, fill_value=0)
        pred_log = _fit_predict(model_kind, Xtr, y_log_all[tr], Xva)
        pred = transformer.inverse_transform(pred_log)
        fold_metrics.append({"fold": f["fold"], **_metrics(y_all[va], pred)})

    # ---- final chronological holdout (25%, min 12) ----
    n_test = max(int(round(n_total * 0.25)), 12)
    cut = n_total - n_test
    boundary = eligible.iloc[cut]["Start_Time"]
    while cut > 0 and eligible.iloc[cut - 1]["Start_Time"] == boundary:
        cut -= 1
    Xtr = pd.get_dummies(X_all.iloc[:cut].astype(object).where(X_all.iloc[:cut].notna(), None), dummy_na=True)
    Xte = pd.get_dummies(X_all.iloc[cut:].astype(object).where(X_all.iloc[cut:].notna(), None), dummy_na=True).reindex(columns=Xtr.columns, fill_value=0)
    pred_log = _fit_predict(model_kind, Xtr, y_log_all[:cut], Xte)
    pred = transformer.inverse_transform(pred_log)
    holdout = _metrics(y_all[cut:], pred)

    # ---- unseen/new-loader slice on the holdout window ----
    train_loaders = set(loader_arr[:cut].tolist())
    unseen_mask = np.array([l not in train_loaders for l in loader_arr[cut:]])
    unseen = _metrics(y_all[cut:][unseen_mask], pred[unseen_mask]) if unseen_mask.sum() >= 3 else {"n": int(unseen_mask.sum()), "status": "INSUFFICIENT_SAMPLE"}

    # ---- history-depth slices on the holdout window ----
    bins = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, np.inf, "10+")]
    hist_bins: dict[str, dict] = {}
    pc_h = prior_count[cut:]
    for lo, hi, label in bins:
        m = (pc_h >= lo) & (pc_h <= hi)
        if m.sum() >= 3:
            hist_bins[label] = _metrics(y_all[cut:][m], pred[m])
        else:
            hist_bins[label] = {"n": int(m.sum()), "status": "INSUFFICIENT_SAMPLE"}

    pooled = None
    if fold_metrics:
        keys = ("mae", "median_ae", "rmse", "within_20pct", "within_2x")
        pooled = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
        pooled["n_folds"] = len(fold_metrics)
        pooled["total_val_rows"] = int(sum(m["n"] for m in fold_metrics))

    evaluation = StageEvaluation(
        stage=stage,
        model_name=model_kind,
        n_train_total=n_total,
        folds=fold_metrics,
        pooled_oof=pooled,
        holdout=holdout,
        unseen_loader=unseen,
        history_bins=hist_bins,
    )
    if small_sample:
        evaluation.status = "OK_SMALL_SAMPLE"
        evaluation.status_reason = (
            f"Small-sample chronological evaluation: n={n_total} < "
            f"{STAGE_MIN_TRAIN_ROWS}; {len(fold_metrics)} fold(s) + "
            f"{int(n_total - cut)}-row holdout. Metrics are honest but "
            "high-variance; the model remains an experimental candidate "
            "with a disclosed small-sample limitation."
        )
    return evaluation


# ------------------------------------------------------- registry integration


def register_stage_model_experiments(
    stage: str,
    evaluations: dict[str, StageEvaluation],
    chosen_kind: str | None,
) -> list[str]:
    """Append stage-model experiment records to the Model Laboratory registry.

    Records are EXPERIMENTAL facts (never lifecycle actions, never promotion).
    Returns the experiment IDs created.
    """
    from dltime.model_lab import ExperimentRegistry

    registry = ExperimentRegistry()
    ids: list[str] = []
    # Experiment-ID generation: registry is append-only with unique IDs, so a
    # re-run after a policy change writes a new versioned series
    # (…-v2, …-v3, …) while the original records stay as history.
    existing = {e.get("experiment_id") for e in registry.list()}

    def _eid(base: str) -> str:
        if base not in existing:
            existing.add(base)
            return base
        v = 2
        while f"{base}-v{v}" in existing:
            v += 1
        final = f"{base}-v{v}"
        existing.add(final)
        return final

    for kind, ev in evaluations.items():
        payload = {
            "experiment_id": _eid(f"EXP-STAGE-{stage.upper()}-{kind}"),
            "kind": "stage_model",
            "model_family": f"stage_{stage}_{kind}",
            "target_transformation": "log1p",
            "feature_set_version": f"stage-{stage}-v1",
            "feature_names": [],
            "hyperparameters": {},
            "training_population": {"stage_eligible_rows": ev.n_train_total, "policy": "AD-31 stage training policy (loader+sprint identity, completed status, positive stage target)"},
            "validation_method": "expanding-window chronological CV + final chronological holdout",
            "metrics": {
                "pooled_oof": ev.pooled_oof,
                "holdout": ev.holdout,
                "unseen_loader": ev.unseen_loader,
                "history_bins": ev.history_bins,
            },
            "status": ev.status,
            "decision": (
                "EXPERIMENTAL_FACT_NO_LIFECYCLE_ACTION"
                if ev.status in ("OK", "OK_SMALL_SAMPLE")
                else "INSUFFICIENT_EVIDENCE"
            ),
            "rationale": (
                f"Stage model experiment for '{stage}' ({kind}); recorded as an "
                "experimental fact. No lifecycle action taken (AD-31: stage "
                "models never auto-promote)."
            ),
        }
        record = registry.register(payload)
        ids.append(record["experiment_id"])
    if chosen_kind:
        payload = {
            "experiment_id": _eid(f"EXP-STAGE-{stage.upper()}-SELECTION"),
            "kind": "stage_model_selection",
            "model_family": f"stage_{stage}_selection",
            "target_transformation": "log1p",
            "feature_set_version": f"stage-{stage}-v1",
            "feature_names": [],
            "hyperparameters": {},
            "training_population": {},
            "validation_method": "chronological comparison across candidates",
            "metrics": {k: v.to_payload().get("holdout") for k, v in evaluations.items()},
            "status": "SELECTED_CANDIDATE" if chosen_kind else "NO_CANDIDATE_SELECTED",
            "decision": "SELECTED_FOR_REGISTRY_ONLY" if chosen_kind else "NO_SELECTION",
            "rationale": (
                f"Chronological evidence selection for stage '{stage}': "
                f"{chosen_kind}. Selection records the experimental candidate "
                "only — never a promotion."
            ),
        }
        record = registry.register(payload)
        ids.append(record["experiment_id"])
    return ids


def register_stage_aware_total_experiment(comparison) -> str:
    """Record the stage-aware-total vs direct-total chronological comparison.

    Experimental fact only — never a lifecycle action (AD-31: promotion is an
    explicit human decision). Versioned ID (…-v2, …-v3) when the base ID
    already exists (append-only registry).
    """
    from dltime.model_lab import ExperimentRegistry

    registry = ExperimentRegistry()
    existing = {e.get("experiment_id") for e in registry.list()}
    base = "EXP-STAGE-AWARE-TOTAL-v1"
    eid = base
    v = 2
    while eid in existing:
        eid = f"{base.replace('-v1', '')}-v{v}"
        v += 1
    record = registry.register(
        {
            "experiment_id": eid,
            "kind": "stage_aware_total",
            "model_family": "stage_aware_total_extra_trees",
            "target_transformation": "log1p",
            "feature_set_version": "direct-30 + OOF 7-stage vector (stage-aware-total)",
            "feature_names": [
                f"pred_stage_{s}_sec"
                for s in (
                    "staging", "prevalidation", "transformation", "pre_load",
                    "data_loading", "datamart_approval", "dataloading_approval",
                )
            ],
            "hyperparameters": {},
            "training_population": {
                "rows": comparison.population["rows"],
                "policy": "stage population: loader+sprint identity + completed status + positive total target (identical for both models)",
                "total_features_direct": comparison.population["total_features_direct"],
                "total_features_stage_aware": comparison.population["total_features_stage_aware"],
            },
            "validation_method": "expanding-window chronological CV + final chronological holdout (identical population/folds for both architectures)",
            "metrics": {
                "direct_total": comparison.direct,
                "stage_aware_total": comparison.stage_aware,
            },
            "status": "EVALUATED",
            "decision": comparison.verdict["decision"],
            "rationale": (
                "Chronological comparison of the stage-aware total architecture "
                "(7-stage OOF vector + pre-execution context) against the direct "
                "total baseline on the identical population. "
                + str(comparison.verdict.get("decision"))
                + ". No automatic promotion (AD-31)."
            ),
        }
    )
    return record["experiment_id"]
