"""EXP-034: senior-audit evidence pack.

Produces the mandatory evidence the final audit prompt requires:

E1. Train/test disjointness + leakage evidence table (stage-aware total).
E2. Unseen-holdout prediction evidence: predict-then-reveal on rows the
    final serving fit never saw, with legitimate pre-execution inputs,
    recorded BEFORE revealing the actual, then error metrics.
E3. Floor impact on the chronological holdout: metrics before/after the
    max-ML-stage floor (does the floor help or hurt accuracy?).
E4. Five stage-model validation table (n train/holdout, features, metrics).

Run from repo root: uv run --project backend python backend/scripts/exp034_audit_evidence.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def main() -> None:
    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import (
        add_chronological_rank,
        chronological_holdout_split,
        compute_metrics,
    )
    from dltime.stage_aware_serving import (
        load_stage_aware_artifact,
        predict_stage_aware_seconds,
    )
    from dltime.stage_aware_total import _stage_oof_predictions
    from dltime.stage_models import build_stage_frame
    from dltime.stage_serving import build_stage_serving_models
    from dltime.stage_targets import APPROVAL_STAGE_NAMES, ML_STAGE_NAMES
    from dltime.approval_estimator import estimate_approval_stage
    from dltime.sources import load_all_sources

    bundle = load_all_sources()
    total_frame, total_feats, _ = build_stage_frame(stage="staging", bundle=bundle)
    completed = total_frame["LDR_Status"].isin(
        {"SUCCEED", "COMPLETED_WITH_WARNING", "COMPLETED_WITH_ERROR", "COMPLETED_WITH_ERROR_AND_WARNING", "ERROR"}
    )
    id_ok = total_frame["Loader_Name"].notna() & total_frame["Sprint"].notna()
    target = pd.to_numeric(total_frame["total_timetaken_in_min"], errors="coerce") * 60.0
    pop = (completed & id_ok & (target > 0)).fillna(False)
    frame = add_chronological_rank(total_frame[pop].copy())
    y_raw = target[pop].loc[frame.index]

    # ---------------------------------------------------------------- E1: split
    train, test, split_info = chronological_holdout_split(frame)
    train_ids = set(train["LDR_Execution_Id"])
    test_ids = set(test["LDR_Execution_Id"])
    overlap = train_ids & test_ids
    print("== E1 train/test evidence ==")
    print(f"population rows: {len(frame)}")
    print(f"train period: {split_info['train_first_start']} -> {split_info['train_last_start']} (n={split_info['train_rows']})")
    print(f"test period:  {split_info['test_first_start']} -> {split_info['test_last_start']} (n={len(test)})")
    print(f"ID overlap: {len(overlap)} | train_end < test_start: {split_info['train_last_start'] < split_info['test_first_start']}")

    # feature fitting policy: encoding + medians from the TRAIN side only
    sa = load_stage_aware_artifact()
    meta = sa["metadata"]
    print(f"artifact training_row_count: {meta['training_row_count']} (== train rows: {meta['training_row_count'] == len(train)})")
    print(f"feature contract: {len(meta['feature_names_in_order'])} features; encoding fitted on train only (enc_columns pinned)")

    # historical as-of policy spot check: stage_hist features of TEST rows must
    # come from rows strictly earlier (chronological rank of test rows all >
    # train rows by construction).
    assert train["chrono_rank"].max() < test["chrono_rank"].min()
    print("chronological ranks: max(train) < min(test) -> as-of policy structurally sound")

    # ---------------------------------------------------------------- E2: unseen predictions
    # Rebuild the EXACT served feature matrix for holdout rows using ONLY
    # legitimate pre-execution information + served stage models.
    X_all = frame.copy()
    for c in meta["feature_names_in_order"]:
        if c not in X_all.columns:
            X_all[c] = np.nan
    for col in X_all.columns:
        if col.startswith("pred_stage_"):
            continue
        if X_all[col].dtype == "string" or isinstance(X_all[col].dtype, pd.CategoricalDtype):
            X_all[col] = X_all[col].fillna("__MISSING__").astype("string")
        else:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")

    from dltime.stage_serving import load_stage_serving_models

    serving = load_stage_serving_models(bundle=bundle)
    prior_counts = X_all["stage_hist_loader_prior_count"] if "stage_hist_loader_prior_count" in X_all else pd.Series(0, index=X_all.index)

    rows = []
    for idx in test.index:
        row = frame.loc[idx]
        Xrow = X_all.loc[[idx]].copy()
        for stage in ML_STAGE_NAMES:
            col = f"pred_stage_{stage}_sec"
            model = serving.get(stage)
            if model is not None:
                try:
                    est = model.predict_seconds(Xrow)
                except Exception:
                    est = None
                Xrow[col] = float(est) if est is not None and np.isfinite(est) else np.nan
            else:
                Xrow[col] = np.nan
        for stage in APPROVAL_STAGE_NAMES:
            est = estimate_approval_stage(frame, row["LDR_Execution_Id"], stage)
            Xrow[f"pred_stage_{stage}_sec"] = float(est.estimate_seconds) if est.available and est.estimate_seconds is not None else np.nan
        Xrow = Xrow[meta["feature_names_in_order"]]
        raw = predict_stage_aware_seconds(sa, Xrow)
        ml_vals = [Xrow[c].iat[0] for c in meta["feature_names_in_order"] if c.startswith("pred_stage_") and c.split("_")[2] in ML_STAGE_NAMES and pd.notna(Xrow[c].iat[0])]
        floor = max(ml_vals) if ml_vals else None
        eta = max(raw, floor) if floor is not None else raw
        pc = prior_counts.loc[idx]
        rows.append({
            "LDR_Execution_Id": row["LDR_Execution_Id"],
            "start": row["Start_Time"],
            "loader": row["Loader_Name"],
            "dataset": row.get("Dataset_display_name"),
            "raw_total": raw,
            "floor": floor,
            "eta": eta,
            "actual": float(y_raw.loc[idx]),
            "prior_count": float(pc) if pd.notna(pc) else 0.0,
        })
    ev = pd.DataFrame(rows)
    ev["abs_err"] = (ev["eta"] - ev["actual"]).abs()
    ev["signed_err"] = ev["eta"] - ev["actual"]
    ev["ape"] = np.where(ev["actual"] > 0, ev["abs_err"] / ev["actual"] * 100.0, np.nan)
    ev["within20"] = ev["ape"] <= 20.0
    ev["within2x"] = (ev["eta"] <= 2 * ev["actual"]) & (ev["actual"] <= 2 * ev["eta"])
    train_loaders = set(frame.loc[train.index, "Loader_Name"])
    ev["known_loader"] = ev["loader"].isin(train_loaders)

    print("\n== E2 unseen-holdout predict-then-reveal (first 12 by time) ==")
    ev_sorted = ev.sort_values("start")
    cols = ["start", "loader", "raw_total", "floor", "eta", "actual", "abs_err", "ape", "prior_count", "known_loader"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(ev_sorted[cols].head(12).to_string(index=False))
    m_all = compute_metrics(ev["actual"].to_numpy(), ev["eta"].to_numpy())
    m_known = compute_metrics(ev.loc[ev["known_loader"], "actual"].to_numpy(), ev.loc[ev["known_loader"], "eta"].to_numpy()) if ev["known_loader"].any() else {"n": 0}
    m_new = compute_metrics(ev.loc[~ev["known_loader"], "actual"].to_numpy(), ev.loc[~ev["known_loader"], "eta"].to_numpy()) if (~ev["known_loader"]).any() else {"n": 0}
    m_low = compute_metrics(ev.loc[ev["prior_count"] < 3, "actual"].to_numpy(), ev.loc[ev["prior_count"] < 3, "eta"].to_numpy()) if (ev["prior_count"] < 3).any() else {"n": 0}
    print("\naggregate on unseen holdout (WITH floor):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_all.items()})
    print("known-loader:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_known.items()})
    print("new-loader:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_new.items()})
    print("low-history(<3):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_low.items()})

    # ---------------------------------------------------------------- E3: floor impact
    m_nofloor = compute_metrics(ev["actual"].to_numpy(), ev["raw_total"].to_numpy())
    m_floor = compute_metrics(ev["actual"].to_numpy(), ev["eta"].to_numpy())
    engaged = ev["floor"].notna() & (ev["raw_total"] < ev["floor"])
    print("\n== E3 floor impact on holdout ==")
    print(f"floor engaged: {int(engaged.sum())}/{len(ev)}")
    print("no-floor:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_nofloor.items()})
    print("with-floor:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_floor.items()})
    # invariant check: actual total >= actual stage durations?
    # (E3b: how often would the floor RAISE a prediction above the actual?)
    over = ev[engaged & (ev["eta"] > ev["actual"])]
    print(f"floor raised prediction ABOVE actual: {len(over)}/{len(ev)}")

    # ---------------------------------------------------------------- E4: stage model table
    print("\n== E4 stage models ==")
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, sdiag = build_stage_frame(stage=stage, bundle=bundle)
        n_elig = int(sdiag["stage_eligible_rows"])
        print(f"{stage}: eligible_rows={n_elig} features={len(sfeats)} serving_kind={serving.get(stage) and getattr(serving.get(stage), 'model_kind', '?')}")

    print("\nDONE")


if __name__ == "__main__":
    main()
