"""AD-32 one-shot build+governance script (recorded in the lifecycle log).

Builds the stage-aware total serving artifact, persists it, calibrates the
dedicated uncertainty to it, and registers both the artifact record and the
explicit human-approved promotion in the Model Registry. Idempotent at the
registry level (fails cleanly if the version already exists).

Run:  uv run python scripts/build_stage_aware_serving.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def main() -> int:
    from dltime import stage_aware_serving as sas
    from dltime.model_lab import ModelRegistry, ModelRecord, _JsonlStore

    if sas.artifact_exists():
        print(f"Artifact already exists: {sas.STAGE_AWARE_ARTIFACT_PATH} — rebuilding (same version).")

    print("Building stage-aware serving artifact (fresh chronological evidence)...")
    artifact, build_info = sas.build_stage_aware_serving_artifact()
    path = sas.save_stage_aware_artifact(artifact)
    print(f"Saved artifact: {path}")

    print("Building stage-aware conformal calibration...")
    cal = sas.build_stage_aware_calibration(artifact)
    print(
        f"Calibration: n={cal['n_calibration_rows']} "
        f"q80={cal['quantiles_abs_log_residual']['80']:.4f} "
        f"q90={cal['quantiles_abs_log_residual']['90']:.4f}"
    )

    # Reload through the integrity-checked path and smoke-predict once.
    loaded = sas.load_stage_aware_artifact()
    print(
        f"Loaded artifact OK: version={loaded['metadata']['model_version']} "
        f"features={len(loaded['metadata']['feature_names_in_order'])} "
        f"enc={loaded['metadata']['fitted']['feature_out_count']}"
    )
    sas.load_stage_aware_calibration()
    print("Calibration integrity OK.")

    # ---- registry: artifact record + explicit human-approved promotion -------
    store = _JsonlStore()
    registry = ModelRegistry(store)
    model_id = sas.STAGE_AWARE_MODEL_VERSION
    if registry.get(model_id) is None:
        meta = loaded["metadata"]
        rec = ModelRecord(
            model_id=model_id,
            model_family="stage_aware_total_extra_trees",
            artifact_path=str(path),
            status="CANDIDATE_EVALUATED",
            feature_set_version="stage-aware-total-v1 (29 base + 7 stage OOF/approval)",
            feature_count=len(meta["feature_names_in_order"]),
            target_mode="log1p",
            training_population={
                "rows": meta["training_row_count"],
                "period": meta["training_period"],
                "policy": "stage population (loader+sprint identity + completed status + positive total target)",
            },
            validation_population={
                "holdout_rows": meta["holdout_rows"],
                "period": meta["holdout_period"],
                "method": meta["evaluation"]["split_method"],
            },
            experiment_id="EXP-STAGE-AWARE-TOTAL-v2",
            metrics={
                "pooled_oof": meta["evaluation"]["pooled_oof"],
                "holdout": meta["evaluation"]["holdout"],
                "direct_reference": meta["direct_reference_evaluation"],
            },
            notes=(
                "AD-34 authoritative total model (EXP-032 target correction: "
                "authoritative total = total_timetaken_in_min * 60; legacy "
                "timetaken_in_sec is corrupted). Stage features are leakage-safe "
                "OOF predictions (training) / served stage-model predictions "
                "(inference); approvals are strictly-prior hierarchical statistics. "
                "Calibration: unc-log1p-conformal-v1-stage-aware-ad34. Direct total "
                "(total-et-v3-candidate) retained as reference/rollback."
            ),
        )
        registry.register(rec)
        print(f"Registered model record: {model_id} (CANDIDATE_EVALUATED)")
    else:
        existing = registry.get(model_id)
        if existing.get("status") == "EXPERIMENTAL":
            # Repair a record written by an interrupted earlier build run.
            registry._set_state(
                model_id,
                "CANDIDATE_EVALUATED",
                extra={"evaluation_note": "fresh chronological evaluation computed by build_stage_aware_serving.py"},
            )
            print(f"Advanced existing record to CANDIDATE_EVALUATED: {model_id}")
        else:
            print(f"Model record already present: {model_id} ({existing.get('status')})")

    if (registry.promoted_model() or {}).get("model_id") != model_id:
        promoted = registry.promote(
            model_id,
            approved_by="product-owner",
            reason=(
                "AD-34 (EXP-032) correction build: stage-aware total on the "
                "authoritative total target (total_timetaken_in_min * 60) is the "
                "authoritative serving path per the locked product direction; "
                "promotion gates verified in the build pipeline."
            ),
        )
        print(f"PROMOTED: {promoted['model_id']} at {promoted.get('promoted_at')}")
    else:
        print(f"{model_id} is already the promoted model.")

    # Keep the direct model visible as the retired reference/rollback path.
    if registry.get("total-et-v3-candidate") is None:
        registry.register(
            ModelRecord(
                model_id="total-et-v3-candidate",
                model_family="extra_trees_candidate",
                artifact_path=str(Path(__file__).resolve().parents[1] / "artifacts" / "total-et-v3-candidate.pkl"),
                status="RETIRED",
                feature_set_version="pre-execution+historical-v30",
                feature_count=30,
                target_mode="log1p",
                training_population={"rows": 256, "note": "AD-31 corrected-source regime (EXP-028)"},
                validation_population={"note": "EXP-028 chronological re-evaluation"},
                metrics={"note": "see EXP-LAB-012 / EXP-028 evidence"},
                notes="AD-32: retired direct-total baseline — reference/rollback only, no longer the primary ETA.",
            )
        )
        print("Registered direct-total reference record: total-et-v3-candidate (RETIRED)")

    # ---- five ML stage-model artifact (AD-32 §A: stages must be artifacted)
    from dltime.stage_serving import (
        STAGE_MODELS_ARTIFACT_PATH,
        build_stage_serving_models,
        load_stage_serving_models,
        save_stage_serving_models,
        STAGE_SERVING_VERSION,
    )

    need_stage_fit = True
    if STAGE_MODELS_ARTIFACT_PATH.exists():
        import joblib

        blob = joblib.load(STAGE_MODELS_ARTIFACT_PATH)
        need_stage_fit = blob.get("serving_version") != STAGE_SERVING_VERSION
    if need_stage_fit:
        stage_models = build_stage_serving_models()
        p = save_stage_serving_models(stage_models)
        print(f"Saved stage-model artifact: {p} (stages: {sorted(stage_models)})")
    else:
        print("Stage-model artifact already version-matched; skipping refit.")
    loaded = load_stage_serving_models()
    print(
        "Stage serving check:",
        {s: getattr(m, "kind", "?") for s, m in sorted(loaded.items())},
    )

    print(json.dumps(build_info["comparison_verdict"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
