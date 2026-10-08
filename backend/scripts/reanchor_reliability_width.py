"""One-off evidence script: re-anchor reliability width-factor thresholds.

The locked product presentation change (additive interval: point +/- Q90
seconds) changed the meaning of `interval_width_seconds`. The EXP-016
width-factor anchors (median relative width ~10x at 90%) were derived on the
OLD multiplicative presentation. This script re-derives the relative width on
the NEW additive presentation over the SAME calibration population (EXP-036
clean holdout): width = 2 * Q90 seconds (constant), factor = width / point.
Frozen stage models are used exactly as in the serving calibration path;
nothing is retrained and no artifact is written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAL_PATH = ROOT / "backend" / "artifacts" / "unc-log1p-conformal-v1-sum-of-five-stages.json"


def main() -> None:
    sys.path.insert(0, str(ROOT / "backend" / "src"))
    import numpy as np
    import pandas as pd

    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import chronological_holdout_split
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import ML_STAGE_NAMES
    from dltime.sources import load_all_sources
    from dltime.sum_total_calibration import _encode_fit

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    q90 = float(cal["quantiles_abs_log_residual"]["90"])
    q80 = float(cal["quantiles_abs_log_residual"]["80"])

    spec = importlib.util.spec_from_file_location(
        "exp036_clean_holdout", ROOT / "backend" / "scripts" / "exp036_clean_holdout.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bundle = load_all_sources()
    frame, y_raw, _feats = mod.build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)

    tt = TargetTransformer(mode="log1p")
    stage_predictors: dict[str, _StagePredictor] = {}
    stage_frame_by_id = {}
    for stage in ML_STAGE_NAMES:
        sframe, sfeats, _diag = build_stage_frame(stage=stage, bundle=bundle)
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

    factors90: list[float] = []
    for eid in test["LDR_Execution_Id"]:
        total = 0.0
        ok = True
        for stage in ML_STAGE_NAMES:
            try:
                total += float(stage_predictors[stage].predict_seconds(stage_frame_by_id[stage].loc[[eid]]))
            except Exception:
                ok = False
                break
        if not ok or total <= 0:
            continue
        factors90.append((2.0 * q90) / total)

    a = np.asarray(factors90)
    out = {
        "purpose": "Re-anchor assess_reliability width-factor thresholds for the ADDITIVE interval presentation",
        "calibration_version": cal["calibration_version"],
        "calibration_rows": cal["n_calibration_rows"],
        "q90_seconds": q90,
        "interval_width_seconds_90": round(2.0 * q90, 4),
        "width_factor": {
            "n": int(a.size),
            "median": round(float(np.median(a)), 3),
            "p90": round(float(np.quantile(a, 0.9)), 3),
            "max": round(float(a.max()), 3),
        },
        "prior_anchors_multiplicative": {"median_rel_width_90": "~10.0x", "median_rel_width_80": "~5.0x"},
        "recommended_anchors_additive": {
            "LOW_at_or_above": round(float(np.quantile(a, 0.9)) * 1.2, 2),
            "MEDIUM_at_or_above": round(float(np.median(a)) * 1.5, 2),
        },
        "split_info": {
            k: split_info[k]
            for k in ("train_first_start", "train_last_start", "test_first_start", "test_last_start")
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
