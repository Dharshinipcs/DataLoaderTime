"""Feature/model experiment engine (Module 7 / Prompt 4; AD-28).

Runs governed experiments on top of the EXISTING Task 03 chronological
machinery (`modeling.expanding_window_folds`, `chronological_holdout_split`,
`_fit_predict_one` path via fit_candidate_pipeline). Every evaluation is
strictly chronological; no random split is introduced here.

Key experiment (§9): the Prompt-2-deferred `def_preval_name_nunique` derived
from the legitimate definition field `preval_trans_display_name` — the count
of distinct prevalidation/transformation component labels configured for an
identity. Deterministic, available before execution, leakage-safe
(DERIVED_FROM_PRE_EXECUTION), computed as-of each execution from the
definition tables (configuration data, not execution outcomes).

Also provides: served-population evaluation (≥3 prior eligible executions,
matching the AD-16 serving gate), slice evaluation (§14), and the formalized
feature-group ablation (§10). Nothing here promotes a feature or a model;
results are recorded as experiment evidence for human decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dltime.modeling import (
    add_chronological_rank,
    chronological_holdout_split,
    compute_metrics,
    expanding_window_folds,
)

NEW_FEATURE = "def_preval_name_nunique"
FEATURE_SET_V30 = "pre-execution+historical-v30"
FEATURE_SET_V31 = "pre-execution+historical-v31+preval_name_nunique"


def compute_preval_name_nunique(
    definitions_detail: pd.DataFrame,
    executions: pd.DataFrame,
) -> pd.Series:
    """As-of `def_preval_name_nunique` for each execution row.

    Definition-derived (pre-execution configuration): the number of distinct
    `preval_trans_display_name` component labels configured for the
    execution's (sprint, loader, dataset-set). Identical join semantics to
    the existing set-variant definition features; NaN when the definition
    has no matching component rows (never fabricated). No execution outcome
    is consulted.
    """
    det = definitions_detail.copy()
    det["_s"] = det["sprint_name"].map(lambda v: str(v).strip() if pd.notna(v) else None)
    det["_l"] = det["loader_display_name"].map(lambda v: str(v).strip() if pd.notna(v) else None)
    det["_d"] = det["datasetname"].map(lambda v: str(v).strip() if pd.notna(v) else None)
    det["_p"] = det["preval_trans_display_name"].map(lambda v: str(v).strip() if pd.notna(v) else None)
    det = det.dropna(subset=["_s", "_l"])
    pool_by_sl = {k: g for k, g in det.groupby(["_s", "_l"])}

    values: list[float] = []
    for r in executions.itertuples(index=False):
        sprint = str(getattr(r, "Sprint")).strip() if pd.notna(getattr(r, "Sprint")) else None
        loader = str(getattr(r, "Loader_Name")).strip() if pd.notna(getattr(r, "Loader_Name")) else None
        val = np.nan
        if sprint and loader:
            pool = pool_by_sl.get((sprint, loader))
            if pool is not None:
                ds = str(getattr(r, "Dataset_display_name")).strip() if pd.notna(getattr(r, "Dataset_display_name")) else None
                sub = pool[pool["_d"] == ds] if ds else pool
                if sub.empty and ds:
                    # representative display name may differ from datasetname
                    sub = pool[pool["_d"].str.contains(ds, regex=False, na=False)]
                names = sub["_p"].dropna()
                if not names.empty:
                    val = float(names.nunique())
        values.append(val)
    return pd.Series(values, index=executions.index, name=NEW_FEATURE)


def _run_chrono_eval(
    X: pd.DataFrame, y: pd.Series, frame: pd.DataFrame, *, n_estimators: int = 200
) -> dict[str, Any]:
    """Chronological expanding-window CV + holdout for one feature matrix.

    Uses the candidate ExtraTrees config through the shared evaluation path
    (`fit_candidate_pipeline`), preserving byte-identical methodology with
    the Task 03/06 experiments.
    """
    from dltime.modeling import _fit_predict_one

    spec = {
        "estimator": lambda: __import__(
            "sklearn.ensemble", fromlist=["ExtraTreesRegressor"]
        ).ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=-1
        ),
        "needs_encoding": True,
        "needs_scaling": False,
    }
    feature_names = list(X.columns)
    ordered = add_chronological_rank(frame)
    Xo, yo = X.loc[ordered.index], y.loc[ordered.index]
    folds = expanding_window_folds(ordered)
    oof = pd.Series(np.nan, index=ordered.index)
    fold_meds: list[float] = []
    for f in folds:
        train_idx = ordered.index[: f["train_rank_end_exclusive"]]
        val_idx = ordered.index[f["val_rank_start"] : f["val_rank_end_exclusive"]]
        pred_raw, _s, _f = _fit_predict_one(
            spec, Xo.loc[train_idx], yo.loc[train_idx].to_numpy(), Xo.loc[val_idx], "log1p", feature_names
        )
        m = compute_metrics(yo.loc[val_idx].to_numpy(), pred_raw)
        fold_meds.append(m["median_ae"])
        oof.loc[val_idx] = pred_raw
    mask = oof.notna()
    pooled = compute_metrics(yo[mask].to_numpy(), oof[mask].to_numpy())

    # final chronological holdout metrics
    train, test, _info = chronological_holdout_split(ordered)
    pred_h, _s2, _f2 = _fit_predict_one(
        spec, Xo.loc[train.index], yo.loc[train.index].to_numpy(), Xo.loc[test.index], "log1p", feature_names
    )
    holdout = compute_metrics(yo.loc[test.index].to_numpy(), pred_h)
    return {
        "pooled_oof": pooled,
        "holdout": holdout,
        "fold_median_aes": [round(v, 3) for v in fold_meds],
        "fold_medae_worst": max(fold_meds) if fold_meds else None,
        "oof": oof,
        "ordered_index": ordered.index,
    }


def _slice_metrics(oof: pd.Series, ordered_index, frame: pd.DataFrame, y: pd.Series, mask: pd.Series) -> dict[str, Any] | None:
    idx = [i for i in ordered_index if mask.get(i, False) and not pd.isna(oof.get(i, np.nan))]
    if not idx:
        return None
    m = compute_metrics(y.loc[idx].to_numpy(), oof.loc[idx].to_numpy())
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def run_preval_name_experiment(
    tf,
    bundle,
    *,
    min_prior_served: int = 3,
) -> dict[str, Any]:
    """§9 required experiment: 30-feature baseline vs 31-feature candidate.

    Chronological CV + holdout + served-population (≥3 priors) + slices.
    Returns the full evidence record; no adoption decision is made here.
    """
    frame = tf.frame
    eligible = frame[frame["is_training_eligible"].fillna(False)]
    y = tf.y_raw

    # candidate feature (as-of, definition-derived)
    new_feat = compute_preval_name_nunique(bundle.loader_definitions_detail, eligible)
    coverage = float(new_feat.notna().mean())

    base = _run_chrono_eval(tf.X.loc[eligible.index], y.loc[eligible.index], eligible)
    X31 = tf.X.loc[eligible.index].copy()
    X31[NEW_FEATURE] = new_feat
    cand = _run_chrono_eval(X31, y.loc[eligible.index], eligible)

    def _population_block(res: dict[str, Any]) -> dict[str, Any]:
        oof, oidx = res["oof"], res["ordered_index"]
        prior = frame.loc[oidx, "hist_loader_prior_count"]
        served = prior >= min_prior_served
        seen = prior > 0
        out: dict[str, Any] = {
            "served_population": _slice_metrics(oof, oidx, frame, y, served),
            "known_loader": _slice_metrics(oof, oidx, frame, y, seen),
            "unseen_loader": _slice_metrics(oof, oidx, frame, y, ~seen),
            "history_buckets": {},
        }
        for label, m in (
            ("0", prior == 0),
            ("1-2", (prior >= 1) & (prior <= 2)),
            ("3-4", (prior >= 3) & (prior <= 4)),
            ("5-9", (prior >= 5) & (prior <= 9)),
            ("10+", prior >= 10),
        ):
            sm = _slice_metrics(oof, oidx, frame, y, m)
            if sm:
                out["history_buckets"][label] = sm
        return out

    record: dict[str, Any] = {
        "experiment": "preval-name-nunique-v1",
        "feature_added": NEW_FEATURE,
        "source_field": "preval_trans_display_name",
        "feature_coverage_on_training_rows": round(coverage, 4),
        "baseline": {
            "feature_set_version": FEATURE_SET_V30,
            "feature_count": len(tf.feature_names),
            "pooled_oof": base["pooled_oof"],
            "holdout": base["holdout"],
            "fold_median_aes": base["fold_median_aes"],
            **_population_block(base),
        },
        "candidate": {
            "feature_set_version": FEATURE_SET_V31,
            "feature_count": len(tf.feature_names) + 1,
            "pooled_oof": cand["pooled_oof"],
            "holdout": cand["holdout"],
            "fold_median_aes": cand["fold_median_aes"],
            **_population_block(cand),
        },
        "decision": None,
        "decision_note": (
            "Evidence only. Adoption changes the serving artifact and requires the "
            "governed promotion process (human-gated); nothing is adopted here."
        ),
    }
    return record


def run_group_ablation(tf) -> dict[str, Any]:
    """§10 formalized feature-group ablation over the existing registry groups."""
    from dltime.feature_assembly import PREDICTION_FEATURE_GROUPS

    groups = list(PREDICTION_FEATURE_GROUPS.keys())
    full = tf.frame
    eligible = full[full["is_training_eligible"].fillna(False)]
    results: dict[str, Any] = {}
    base = _run_chrono_eval(tf.X.loc[eligible.index], tf.y_raw.loc[eligible.index], eligible)
    results["ALL_FEATURES"] = {
        "pooled_oof": base["pooled_oof"],
        "holdout": base["holdout"],
        "fold_median_aes": base["fold_median_aes"],
    }
    for g in groups:
        keep = [f for gg, cols in PREDICTION_FEATURE_GROUPS.items() if gg != g for f in cols]
        res = _run_chrono_eval(tf.X.loc[eligible.index, keep], tf.y_raw.loc[eligible.index], eligible)
        results[f"REMOVE_{g}"] = {
            "pooled_oof": res["pooled_oof"],
            "holdout": res["holdout"],
            "fold_median_aes": res["fold_median_aes"],
        }
    return results


def build_experiment_record(
    *,
    experiment_id: str,
    kind: str,
    model_family: str,
    target_mode: str,
    hyperparameters: dict[str, Any],
    feature_set_version: str,
    feature_names: list[str],
    training_population: dict[str, Any],
    metrics: dict[str, Any],
    slices: dict[str, Any] | None = None,
    artifact_ref: str | None = None,
    parent_model: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Assemble a laboratory experiment record (§3 fields)."""
    ordered = add_chronological_rank(
        training_population.get("frame")  # not used; placeholder for typing
        if isinstance(training_population.get("frame"), pd.DataFrame)
        else pd.DataFrame()
    ) if False else None
    from dltime.feature_assembly import build_training_frame  # noqa: F401

    rec = {
        "schema_version": "t13-v1",
        "experiment_id": experiment_id,
        "kind": kind,
        "created_at": None,  # set by ExperimentRegistry.register
        "model_family": model_family,
        "target_transformation": target_mode,
        "hyperparameters": hyperparameters,
        "feature_set_version": feature_set_version,
        "feature_names": feature_names,
        "training_population": training_population,
        "validation_method": {
            "type": "expanding-window chronological CV + final chronological holdout",
            "config": {"holdout_fraction": 0.25, "min_holdout": 25, "n_folds": 5},
            "random_seed": 42,
            "policy_versions": {
                "chronology": "AD-11",
                "eligibility": "AD-16 (min 3 prior eligible executions)",
                "target": "Task 01 execution-level timetaken_in_sec",
            },
        },
        "metrics": metrics,
        "slices": slices or {},
        "artifact_reference": artifact_ref,
        "parent_model_version": parent_model,
        "status": "EVALUATED",
        "decision": None,
        "rationale": notes,
    }
    return rec
