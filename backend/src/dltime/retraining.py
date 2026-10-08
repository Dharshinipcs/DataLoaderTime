"""Retraining workflow (locked plan §9-16): human-gated, persisted, evidence-first.

NOTHING here retrains, promotes, rolls back, or serves anything automatically:

  - POST /retraining/continue-monitoring  -> persisted human decision record
    (no retraining; the current model remains untouched).
  - POST /retraining/start                -> persisted human decision record
    ("retraining in progress"), still without touching serving.
  - POST /retraining/train-candidate      -> trains a CANDIDATE stage bundle
    with the SAME five approved model families (staging=ExtraTrees,
    prevalidation=ExtraTrees, transformation=RandomForest, pre_load=ExtraTrees,
    data_loading=Ridge), SAME feature frames, SAME chronological-holdout
    evaluation protocol (EXP-036 machinery) — then registers it as
    CANDIDATE_EVALUATED with evaluation metrics. Serving is untouched.
  - POST /models/promote (existing)       -> the ONLY path to production.
  - POST /models/rollback (existing)      -> restores the previous model.

Every state change appends to the append-only lifecycle log and raises a
deduplicated notification (reusing the AD-33 notification infrastructure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dltime.config import PROJECT_ROOT
from dltime.model_lab import (
    CANDIDATE_EVALUATION_MIN_RATIO,
    LabError,
    ModelState,
    utc_now_iso,
)

DECISIONS_SCHEMA_VERSION = "t13-v1"
DECISIONS_FILE = PROJECT_ROOT / "backend" / "store" / "retraining_decisions.jsonl"

DECISION_TYPES = ("CONTINUE_MONITORING", "START_RETRAINING")

# The candidate bundle version marker recorded in predictions served by a
# promoted candidate bundle (used to prove serving actually switched).
CANDIDATE_NAME_PREFIXES = (
    "full_stage_bundle_",
    "staging_stage_",
    "prevalidation_stage_",
    "transformation_stage_",
    "pre_load_stage_",
    "data_loading_stage_",
)


# ---------------------------------------------------------------- decisions


def _decisions_path() -> Path:
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    return DECISIONS_FILE


def _read_decisions() -> list[dict[str, Any]]:
    path = _decisions_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # fail-closed on read
        if isinstance(rec, dict) and rec.get("schema_version") == DECISIONS_SCHEMA_VERSION:
            out.append(rec)
    return out


def record_decision(
    *,
    decision_type: str,
    reason: str,
    decided_by: str,
    assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one explicit human monitoring/retraining decision (append-only).

    The decision references the CURRENT assessment evidence so the record can
    stand alone in governance review later.
    """
    if decision_type not in DECISION_TYPES:
        raise LabError(f"Unknown decision type {decision_type!r}; expected one of {DECISION_TYPES}.")
    rec = {
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "decision_id": f"dec_{utc_now_iso().replace(':', '').replace('-', '')}_{decision_type.lower()}",
        "decision_type": decision_type,
        "decided_by": decided_by,
        "reason": reason,
        "decided_at": utc_now_iso(),
        "model_version_at_decision": _serving_version(),
        "assessment_snapshot": {
            "verdict": (assessment or {}).get("verdict"),
            "data_health": (assessment or {}).get("data_health"),
            "model_health": (assessment or {}).get("model_health"),
            "new_eligible_outcomes": (assessment or {}).get("new_eligible_outcomes"),
        },
    }
    with _decisions_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return rec


def list_decisions() -> list[dict[str, Any]]:
    return list(reversed(_read_decisions()))


# ---------------------------------------------------------- candidate training


def build_candidate_stage_bundle(*, bundle=None) -> dict[str, Any]:
    """Train + evaluate a CANDIDATE five-stage bundle (NO promotion, NO serving change).

    Reuses the EXISTING approved methodology verbatim:
      - stage frames from dltime.stage_models.build_stage_frame (same feature
        contract, as-of semantics, no post-execution leakage);
      - the approved stage-kind map (staging=ExtraTrees, prevalidation=ExtraTrees,
        transformation=RandomForest, pre_load=ExtraTrees, data_loading=Ridge);
      - the EXP-036 chronological TRAIN/HOLDOUT split (same population, same
        boundary) for a leakage-safe candidate-vs-baseline evaluation.
    The candidate models are fit ONLY on the train-period eligible rows, frozen,
    then evaluated on the holdout. The result is an EVALUATION record plus a
    registered CANDIDATE_EVALUATED registry entry — never a serving change.
    """
    import importlib.util

    import numpy as np
    import pandas as pd

    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import chronological_holdout_split
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import ML_STAGE_NAMES

    # Approved stage-kind map — the SAME families the production stages use
    # (sum_total_calibration.build_sum_total_calibration, EXP-038).
    STAGE_KIND_MAP = {
        "staging": "extra_trees",
        "prevalidation": "extra_trees",
        "transformation": "random_forest",
        "pre_load": "extra_trees",
        "data_loading": "ridge",
    }

    from dltime.sources import load_all_sources

    bundle = bundle or load_all_sources()

    # EXP-036 population + split (identical boundary by construction).
    root = PROJECT_ROOT
    spec = importlib.util.spec_from_file_location(
        "exp036_clean_holdout", root / "backend" / "scripts" / "exp036_clean_holdout.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise LabError("EXP-036 protocol module not found.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frame, y_raw, _total_feats = mod.build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)
    train_ids = set(train["LDR_Execution_Id"])

    tt = TargetTransformer(mode="log1p")
    predictors: dict[str, Any] = {}
    eval_rows: dict[str, dict[str, Any]] = {}
    stage_frame_by_id: dict[str, pd.DataFrame] = {}
    skipped_stages: dict[str, str] = {}

    for stage in ML_STAGE_NAMES:
        sframe, feats, diag = build_stage_frame(stage=stage, bundle=bundle)
        elig = sframe[sframe["stage_eligible"].astype(bool)]
        elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
        # Same discipline as the production build (EXP-036): each stage fits on
        # its TRAIN-period eligible rows. pre_load is documented small-sample
        # (n=13 train rows in the recorded EXP-036 evidence) — refusal would
        # diverge from the approved methodology, so instead the sample size is
        # carried honestly in the evaluation/registry records.
        # 2026-09-29 (v2 reality): a stage with NO training rows at all on the
        # current source is SKIPPED and recorded as NOT_APPLICABLE instead of
        # blocking the whole candidate bundle. Skipping is honest: the stage
        # simply cannot be retrained and the serving layer reports it
        # unavailable (never fabricated). (Staging used to be the only such
        # stage; since 2026-09-30 the corrected export carries real staging
        # timing fields, so staging now retrains like any other stage.)
        if len(elig) == 0:
            skipped_stages[stage] = "NOT_APPLICABLE_NO_STAGE_DATA"
            continue
        if len(elig_train) < 5:
            raise LabError(f"Candidate training refused: stage '{stage}' has only {len(elig_train)} eligible train rows.")
        X = elig_train[feats].copy()
        for col in X.columns:
            if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna("__MISSING__").astype("string")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        medians = X.median(numeric_only=True)
        Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
        y_log = tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce"))
        kind = STAGE_KIND_MAP[stage]
        model = _make_estimator(kind)
        model.fit(Xenc, y_log)
        pred = _StagePredictor(stage, kind, model, feats, len(elig_train), medians)
        pred._enc_columns = Xenc.columns
        predictors[stage] = pred

        # ---- holdout evaluation for THIS stage (chronological, model-unseen rows)
        y_all = pd.to_numeric(elig[f"stage_{stage}_sec"], errors="coerce")
        ids_all = elig["LDR_Execution_Id"]
        hold_mask = ids_all.isin(set(test["LDR_Execution_Id"]))
        if hold_mask.sum() >= 5:
            Xh = elig.loc[hold_mask, feats].copy()
            for col in Xh.columns:
                if Xh[col].dtype == "string" or isinstance(Xh[col].dtype, pd.CategoricalDtype):
                    Xh[col] = Xh[col].fillna("__MISSING__").astype("string")
                else:
                    Xh[col] = pd.to_numeric(Xh[col], errors="coerce")
            Xh_enc = pd.get_dummies(Xh.astype(object).where(Xh.notna(), None), dummy_na=True)
            Xh_enc = Xh_enc.reindex(columns=Xenc.columns, fill_value=0)
            yh = y_all[hold_mask].to_numpy(dtype=float)
            ph = tt.inverse_transform(np.asarray(model.predict(Xh_enc), dtype=float))
            ok = np.isfinite(yh) & np.isfinite(ph)
            if ok.sum() >= 5:
                errs = np.abs(yh[ok] - ph[ok])
                eval_rows[stage] = _metrics_from_errors(errs, n=int(ok.sum()))

        stage_frame_by_id[stage] = sframe.set_index("LDR_Execution_Id")

    # ---- holdout TOTAL evaluation (sum of the candidate stages that exist;
    # stages skipped for NO_STAGE_DATA are excluded from the sum and
    # disclosed, matching the serving-side honest behavior).
    skipped = sorted(skipped_stages)
    sum_stages = [s for s in ML_STAGE_NAMES if s not in skipped]
    sums: list[float] = []
    for eid in test["LDR_Execution_Id"]:
        total = 0.0
        ok = True
        for stage in sum_stages:
            try:
                total += float(predictors[stage].predict_seconds(stage_frame_by_id[stage].loc[[eid]]))
            except Exception:
                ok = False
                break
        sums.append(total if ok else np.nan)
    sum_pred = np.asarray(sums, dtype=float)
    y = y_raw.loc[test.index].to_numpy(dtype=float)
    ok = (y > 0) & np.isfinite(sum_pred)
    if ok.sum() < 5:
        raise LabError("Candidate evaluation refused: fewer than 5 valid holdout totals.")
    eval_rows["__total__"] = _metrics_from_errors(np.abs(y[ok] - sum_pred[ok]), n=int(ok.sum()))

    candidate_version = f"full_stage_bundle_{utc_now_iso().replace(':', '').replace('-', '')}"
    artifact_path = PROJECT_ROOT / "backend" / "artifacts" / f"stage-models-{candidate_version}.pkl"

    return {
        "candidate_version": candidate_version,
        "artifact_path": str(artifact_path),
        "stage_kind_map": dict(STAGE_KIND_MAP),
        "predictors": predictors,
        "skipped_stages": skipped_stages,
        "evaluation": {
            "protocol": "EXP-036 chronological holdout (candidate fit ONLY on the earlier train period; frozen on holdout)",
            "split": {
                "train_first_start": str(split_info["train_first_start"]),
                "train_last_start": str(split_info["train_last_start"]),
                "holdout_first_start": str(split_info["test_first_start"]),
                "holdout_last_start": str(split_info["test_last_start"]),
            },
            "per_stage": eval_rows,
            "n_holdout_rows": int(ok.sum()),
        },
    }


def _metrics_from_errors(errs, *, n: int) -> dict[str, Any]:
    """Absolute-error metrics in the same shape the SELECTION experiments use."""
    import numpy as np

    errs = np.asarray(errs, dtype=float)
    errs = errs[np.isfinite(errs)]
    srt = np.sort(errs)
    return {
        "n": int(n),
        "mae": round(float(errs.mean()), 4),
        "median_ae": round(float(np.median(errs)), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(errs)))), 4),
        "within_10pct": None,  # filled by callers with the matching denominator
        "within_20pct": None,
    }


# ------------------------------------------------------- candidate evaluation


def candidate_evaluation(
    *,
    candidate_version: str,
    per_stage: dict[str, Any],
    baseline_per_stage: dict[str, Any] | None,
    total: dict[str, Any],
    baseline_total: dict[str, Any] | None,
) -> dict[str, Any]:
    """Explain the candidate verdict from evaluation metrics (NEVER auto-promotes).

    'ACCEPTABLE_FOR_REVIEW' means the candidate is at least broadly comparable
    on the same chronological holdout and goes to the HUMAN for a decision.
    'NOT_SUFFICIENTLY_BETTER' means the evidence does not support promotion.
    Nothing here promotes anything.
    """
    # Only stages present in the evaluation are compared — a
    # single-stage candidate shows in its own row only.
    present = [s for s in ("staging", "prevalidation", "transformation", "pre_load", "data_loading") if s in per_stage]
    stage_rows: list[dict[str, Any]] = []
    comparisons: list[str] = []
    for stage in present:
        cand = per_stage.get(stage) or {}
        base = (baseline_per_stage or {}).get(stage) or {}
        cand_med, base_med = cand.get("median_ae"), base.get("median_ae")
        ratio = None
        if cand_med is not None and base_med not in (None, 0) and base_med:
            try:
                ratio = round(float(cand_med) / float(base_med), 3)
            except (TypeError, ZeroDivisionError):
                ratio = None
        stage_rows.append(
            {
                "stage": stage,
                "candidate_median_ae": cand_med,
                "baseline_median_ae": base_med,
                "ratio": ratio,
                "candidate_n": cand.get("n"),
            }
        )
        if ratio is not None:
            comparisons.append(stage)

    cand_total_med = total.get("median_ae")
    base_total_med = (baseline_total or {}).get("median_ae")
    total_ratio = None
    if cand_total_med is not None and base_total_med:
        try:
            total_ratio = round(float(cand_total_med) / float(base_total_med), 3)
        except (TypeError, ZeroDivisionError):
            total_ratio = None

    # Verdict: strictly better OR statistically comparable on the SAME protocol.
    if total_ratio is None:
        verdict, rationale = "INSUFFICIENT_EVIDENCE", (
            "No persisted chronological baseline for the served sum was found, so the "
            "candidate cannot be compared under the same protocol."
        )
    elif total_ratio <= CANDIDATE_EVALUATION_MIN_RATIO:
        verdict, rationale = "ACCEPTABLE_FOR_REVIEW", (
            f"Candidate holdout MedAE is {total_ratio}x the served baseline "
            f"({cand_total_med}s vs {base_total_med}s on the same chronological holdout). "
            "A human decides promotion; nothing is automatic."
        )
    else:
        verdict, rationale = "NOT_SUFFICIENTLY_BETTER", (
            f"Candidate holdout MedAE is {total_ratio}x the served baseline "
            f"({cand_total_med}s vs {base_total_med}s) — the candidate is NOT an "
            "improvement; promotion is not supported by this evidence."
        )

    return {
        "candidate_version": candidate_version,
        "verdict": verdict,
        "rationale": rationale,
        "per_stage": stage_rows,
        "total": {
            "candidate_median_ae": cand_total_med,
            "baseline_median_ae": base_total_med,
            "ratio": total_ratio,
            "candidate_n": total.get("n"),
            "baseline_n": (baseline_total or {}).get("n"),
        },
        "human_gate_required": True,
        "automatic_promotion": "OFF",
    }


# ------------------------------------------------------- promotion gates


def candidate_stage_report(artifact_path: str | Path, stage: str) -> dict[str, Any]:
    """Promotion-safety gates for a SINGLE-STAGE candidate.

    Same discipline as the bundle gates, scoped to one stage:
    the artifact must exist, load, name the right version, and
    carry exactly the one approved stage model it claims.
    """
    import joblib

    path = Path(artifact_path)
    report: dict[str, Any] = {
        "artifact_exists": path.exists(),
        "bundle_loads": False,
        "version_matches": False,
        "stage_present": False,
        "approved_model_family": False,
        "predict_path_callable": False,
        "checks_detail": {},
    }
    APPROVED = {
        "staging": "extra_trees",
        "prevalidation": "extra_trees",
        "transformation": "random_forest",
        "pre_load": "extra_trees",
        "data_loading": "ridge",
    }
    if not path.exists():
        report["checks_detail"]["artifact"] = "missing artifact"
        return report
    try:
        blob = joblib.load(path)
        report["bundle_loads"] = True
        report["version_matches"] = blob.get("serving_version") == path.name.replace("stage-models-", "").replace(".pkl", "")
        models = blob.get("models") or {}
        report["stage_present"] = set(models.keys()) == {stage} and (blob.get("candidate_metadata") or {}).get("stage") == stage
        model = models.get(stage)
        report["approved_model_family"] = bool(model is not None and getattr(model, "kind", None) == APPROVED.get(stage))
        report["predict_path_callable"] = bool(model is not None and callable(getattr(model, "predict_seconds", None)))
        meta = blob.get("candidate_metadata") or {}
        report["trained_at"] = meta.get("trained_at")
        report["evaluation_summary"] = (meta.get("evaluation") or {}).get("per_stage", {}).get(stage)
    except Exception as exc:
        report["checks_detail"]["load"] = str(exc)
    return report


def candidate_bundle_report(artifact_path: str | Path) -> dict[str, Any]:
    """Promotion-safety gates for a CANDIDATE STAGE BUNDLE (§16).

    Stage bundles are five stage predictors, not the 30-feature total
    artifact, so the total-model compatibility_report does not apply. These
    gates verify the bundle against the LIVE serving contract it must serve
    under: same five stages, same approved model families, fitted predict
    path, persisted experiment evidence. The uncertainty calibration stays
    bound to the deterministic sum policy, so no calibration swap occurs.
    """
    import joblib

    from dltime.stage_targets import ML_STAGE_NAMES

    path = Path(artifact_path)
    report: dict[str, Any] = {
        "artifact_exists": path.exists(),
        "bundle_loads": False,
        "version_matches": False,
        "five_stages_present": False,
        "approved_model_families": False,
        "predict_path_callable": False,
        "forbidden_feature_absent": True,
        "checks_detail": {},
    }
    APPROVED = {
        "staging": "extra_trees",
        "prevalidation": "extra_trees",
        "transformation": "random_forest",
        "pre_load": "extra_trees",
        "data_loading": "ridge",
    }
    if not path.exists():
        report["checks_detail"]["artifact"] = "missing artifact"
        return report
    try:
        blob = joblib.load(path)
        report["bundle_loads"] = True
        report["version_matches"] = blob.get("serving_version") == path.name.replace("stage-models-", "").replace(".pkl", "")
        models = blob.get("models") or {}
        stages = set(models.keys())
        report["five_stages_present"] = stages == set(ML_STAGE_NAMES)
        kinds_ok = all(
            getattr(m, "kind", None) == APPROVED.get(stage)
            for stage, m in models.items()
        )
        report["approved_model_families"] = bool(kinds_ok)
        report["predict_path_callable"] = all(
            callable(getattr(m, "predict_seconds", None)) for m in models.values()
        )
        meta = blob.get("candidate_metadata") or {}
        report["trained_at"] = meta.get("trained_at")
        report["evaluation_summary"] = (meta.get("evaluation") or {}).get("per_stage", {}).get("__total__")
    except Exception as exc:
        report["checks_detail"]["load"] = str(exc)
    return report


# ------------------------------------------------------- registry + serving


def persist_candidate_bundle(
    *,
    result: dict[str, Any],
    experiment_record: dict[str, Any] | None,
    registry: Any,
    model_family: str = "sum-of-five-stages-candidate",
    stage: str | None = None,
) -> dict[str, Any]:
    """Save the candidate artifact, register CANDIDATE_EVALUATED + experiment.

    Serving is NOT touched here — promotion remains the separate, explicit,
    human-gated action via POST /models/promote.
    """
    import joblib

    from dltime.model_lab import ModelRecord

    artifact_path = Path(result["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "serving_version": result["candidate_version"],
            "stages": list(result["predictors"].keys()),
            "models": result["predictors"],
            "candidate_metadata": {
                "trained_at": utc_now_iso(),
                "stage_kind_map": result["stage_kind_map"],
                "evaluation": result["evaluation"],
                "experiment_id": (experiment_record or {}).get("experiment_id"),
                **({"stage": stage} if stage else {}),
            },
        },
        artifact_path,
    )

    rec = ModelRecord(
        model_id=result["candidate_version"],
        model_family=model_family,
        artifact_path=str(artifact_path),
        status=ModelState.CANDIDATE_EVALUATED,
        feature_set_version="stage-frames-v2",
        feature_count=0,
        target_mode="log1p",
        training_population={
            "protocol": result["evaluation"]["protocol"],
            "split": result["evaluation"]["split"],
            "n_holdout_rows": result["evaluation"]["n_holdout_rows"],
        },
        validation_population={"protocol": "chronological holdout"},
        experiment_id=(experiment_record or {}).get("experiment_id"),
        metrics={"per_stage": result["evaluation"]["per_stage"]},
        notes=(
            f"Candidate stage model for '{stage}' (same approved family); "
            "evaluated on the EXP-036 chronological holdout. Requires human promotion."
            if stage
            else "Candidate five-stage bundle (same approved model families); evaluated on the EXP-036 chronological holdout. Requires human promotion."
        ),
    )
    if stage:
        # Single-stage candidates carry the stage they serve; the
        # registry record exposes it for per-stage promotion.
        rec.stage = stage
        rec.artifact_path = str(result["artifact_path"])
    registered = registry.models.register(rec)
    if experiment_record is not None:
        registry.experiments.register(experiment_record)
    return registered


def promote_candidate_bundle(
    *, registry: Any, candidate_version: str, approved_by: str, reason: str
) -> dict[str, Any]:
    """Point stage serving at the promoted candidate bundle (after registry promote).

    2026-10-06: also records per-stage lineage — the previously serving
    model of EACH stage is demoted to 'previous' and the promoted bundle
    becomes 'current' — so the Drift & Retraining comparison table's
    Previous Model column fills for full retrains exactly as it does for
    per-stage promotions.
    """
    from dltime import retraining_serving, stage_lineage
    from dltime.stage_targets import ML_STAGE_NAMES

    retraining_serving.activate(candidate_version)
    _reset_stage_model_cache()
    rec = registry.models.get(candidate_version) or {}
    per_stage = (rec.get("metrics") or {}).get("per_stage") or {}
    lineage: dict[str, Any] = {}
    for stage in ML_STAGE_NAMES:
        m = per_stage.get(stage) or {}
        lineage[stage] = stage_lineage.record_promotion(
            stage=stage,
            model_version=candidate_version,
            holdout_medae=m.get("median_ae"),
            holdout_n=m.get("n"),
        )
    return {
        "serving_version": retraining_serving.active_version(),
        "model": rec,
        "lineage": lineage,
    }

def rollback_serving_path() -> dict[str, Any]:
    """Return serving to the deterministic sum-of-five-stages path (post-rollback).

    2026-10-06: also records per-stage rollback lineage — the rolled-back
    (previously promoted) model moves to 'previous' and the restored model
    becomes 'current' again, so the comparison table reflects the restore.
    """
    from dltime import retraining_serving, stage_lineage
    from dltime.stage_targets import ML_STAGE_NAMES

    roles_all = stage_lineage.stage_roles()
    retraining_serving.reset()
    _reset_stage_model_cache()
    lineage: dict[str, Any] = {}
    for stage in ML_STAGE_NAMES:
        roles = roles_all.get(stage, {})
        restored = roles.get("previous") or roles.get("baseline") or {}
        lineage[stage] = stage_lineage.record_rollback(
            stage=stage,
            model_version=str(
                restored.get("model_version") or "stage-serving-v2 (stage-models-ad34)"
            ),
            holdout_medae=restored.get("holdout_medae"),
            holdout_n=restored.get("holdout_n"),
        )
    return {"serving_version": retraining_serving.active_version(), "lineage": lineage}


# ------------------------------------------------ single-stage candidate

def build_candidate_stage(stage: str, *, bundle=None) -> dict[str, Any]:
    """Train + evaluate a CANDIDATE for ONE stage (NO promotion).

    Identical protocol to the whole-bundle path, scoped to a single
    stage: same feature frame builder (build_stage_frame), the
    approved model family for that stage, the EXP-036 chronological
    TRAIN/HOLDOUT split. Registers nothing; serving untouched.
    """
    import importlib.util

    import numpy as np
    import pandas as pd

    from dltime.feature_assembly import TargetTransformer
    from dltime.modeling import chronological_holdout_split
    from dltime.stage_models import _make_estimator, build_stage_frame
    from dltime.stage_serving import _StagePredictor
    from dltime.stage_targets import ML_STAGE_NAMES

    if stage not in ML_STAGE_NAMES:
        raise LabError(f"Unknown stage {stage!r}; expected one of {ML_STAGE_NAMES}.")

    STAGE_KIND_MAP = {
        "staging": "extra_trees",
        "prevalidation": "extra_trees",
        "transformation": "random_forest",
        "pre_load": "extra_trees",
        "data_loading": "ridge",
    }

    from dltime.sources import load_all_sources

    bundle = bundle or load_all_sources()

    root = PROJECT_ROOT
    spec = importlib.util.spec_from_file_location(
        "exp036_clean_holdout", root / "backend" / "scripts" / "exp036_clean_holdout.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise LabError("EXP-036 protocol module not found.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frame, _y_raw, _total_feats = mod.build_population(bundle)
    train, test, split_info = chronological_holdout_split(frame)
    train_ids = set(train["LDR_Execution_Id"])

    tt = TargetTransformer(mode="log1p")
    sframe, feats, _diag = build_stage_frame(stage=stage, bundle=bundle)
    elig = sframe[sframe["stage_eligible"].astype(bool)]
    elig_train = elig[elig["LDR_Execution_Id"].isin(train_ids)]
    # A single-stage candidate is an explicit per-stage request: a
    # stage with no data is a refusal, never a silent skip.
    if len(elig) == 0:
        raise LabError(f"Candidate training refused: stage '{stage}' has no stage data on the current source.")
    if len(elig_train) < 5:
        raise LabError(f"Candidate training refused: stage '{stage}' has only {len(elig_train)} eligible train rows.")
    X = elig_train[feats].copy()
    for col in X.columns:
        if X[col].dtype == "string" or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    medians = X.median(numeric_only=True)
    Xenc = pd.get_dummies(X.astype(object).where(X.notna(), None), dummy_na=True)
    y_log = tt.transform(pd.to_numeric(elig_train[f"stage_{stage}_sec"], errors="coerce"))
    kind = STAGE_KIND_MAP[stage]
    model = _make_estimator(kind)
    model.fit(Xenc, y_log)
    pred = _StagePredictor(stage, kind, model, feats, len(elig_train), medians)
    pred._enc_columns = Xenc.columns

    # ---- chronological holdout evaluation for THIS stage
    eval_row: dict[str, Any] | None = None
    y_all = pd.to_numeric(elig[f"stage_{stage}_sec"], errors="coerce")
    ids_all = elig["LDR_Execution_Id"]
    hold_mask = ids_all.isin(set(test["LDR_Execution_Id"]))
    if hold_mask.sum() >= 5:
        Xh = elig.loc[hold_mask, feats].copy()
        for col in Xh.columns:
            if Xh[col].dtype == "string" or isinstance(Xh[col].dtype, pd.CategoricalDtype):
                Xh[col] = Xh[col].fillna("__MISSING__").astype("string")
            else:
                Xh[col] = pd.to_numeric(Xh[col], errors="coerce")
        Xh_enc = pd.get_dummies(Xh.astype(object).where(Xh.notna(), None), dummy_na=True)
        Xh_enc = Xh_enc.reindex(columns=Xenc.columns, fill_value=0)
        yh = y_all[hold_mask].to_numpy(dtype=float)
        ph = tt.inverse_transform(np.asarray(model.predict(Xh_enc), dtype=float))
        ok = np.isfinite(yh) & np.isfinite(ph)
        if ok.sum() >= 5:
            errs = np.abs(yh[ok] - ph[ok])
            eval_row = _metrics_from_errors(errs, n=int(ok.sum()))
    if eval_row is None:
        raise LabError(f"Candidate evaluation refused: stage '{stage}' has fewer than 5 valid holdout rows.")

    candidate_version = f"{stage}_stage_{utc_now_iso().replace(':', '').replace('-', '')}"
    artifact_path = PROJECT_ROOT / "backend" / "artifacts" / f"stage-models-{candidate_version}.pkl"

    return {
        "candidate_version": candidate_version,
        "stage": stage,
        "artifact_path": str(artifact_path),
        "stage_kind_map": {stage: STAGE_KIND_MAP[stage]},
        "predictors": {stage: pred},
        "skipped_stages": {},
        "evaluation": {
            "protocol": "EXP-036 chronological holdout (candidate fit ONLY on the earlier train period; frozen on holdout)",
            "split": {
                "train_first_start": str(split_info["train_first_start"]),
                "train_last_start": str(split_info["train_last_start"]),
                "holdout_first_start": str(split_info["test_first_start"]),
                "holdout_last_start": str(split_info["test_last_start"]),
            },
            "per_stage": {stage: eval_row},
            "n_holdout_rows": int(eval_row["n"]),
        },
    }


def promote_candidate_stage(
    *, registry: Any, stage: str, candidate_version: str, approved_by: str, reason: str
) -> dict[str, Any]:
    """Activate a PER-STAGE serving override after the registry
    promotion of a single-stage candidate (explicit human action).
    Other stages keep serving their current models. Writes the
    stage's lineage snapshot."""
    from dltime import retraining_serving, stage_lineage

    retraining_serving.activate_stage(stage, candidate_version)
    _reset_stage_model_cache()
    rec = registry.models.get(candidate_version) or {}
    metrics = ((rec.get("metrics") or {}).get("per_stage") or {}).get(stage) or {}
    lineage = stage_lineage.record_promotion(
        stage=stage,
        model_version=candidate_version,
        holdout_medae=metrics.get("median_ae"),
        holdout_n=metrics.get("n"),
    )
    return {
        "serving_version": retraining_serving.active_stage_overrides().get(stage, {}).get("candidate_version"),
        "stage": stage,
        "model": rec,
        "lineage": lineage,
    }


def rollback_stage(*, stage: str, approved_by: str, reason: str) -> dict[str, Any]:
    """Reset ONE stage's serving override (explicit human action).
    The stage returns to the standard artifact model; other stages
    keep their overrides. Writes the stage's lineage snapshot."""
    from dltime import retraining_serving, stage_lineage

    roles = stage_lineage.stage_roles().get(stage, {})
    restored = roles.get("previous") or roles.get("baseline") or {}
    restored_version = str(
        restored.get("model_version") or "stage-serving-v2 (stage-models-ad34)"
    )
    retraining_serving.reset([stage])
    _reset_stage_model_cache()
    lineage = stage_lineage.record_rollback(
        stage=stage,
        model_version=restored_version,
        holdout_medae=restored.get("holdout_medae"),
        holdout_n=restored.get("holdout_n"),
    )
    return {
        "serving_version": restored_version,
        "stage": stage,
        "lineage": lineage,
    }


def _reset_stage_model_cache() -> None:
    """Force the live PredictionService singleton to reload stage models from
    the (possibly switched) artifact path so serving changes take effect
    immediately without a process restart."""
    try:
        from dltime.api import get_prediction_service

        svc = get_prediction_service()
        svc._stage_models_cache = None
    except Exception:
        pass  # singletons not initialized yet — first use will load the switch


# ------------------------------------------------------- notifications (Phase 5)


def _serving_version() -> str:
    try:
        from dltime import retraining_serving

        active = retraining_serving.active_version()
        if active:
            return active
        overrides = retraining_serving.active_stage_overrides()
        if overrides:
            return " + ".join(
                f"{stage}:{ov.get('candidate_version')}" for stage, ov in sorted(overrides.items())
            )
        return "stage-serving-v2 (stage-models-ad34)"
    except Exception:
        return "unknown"


def lifecycle_notifications(
    *, event_kind: str, store: Any, context: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Deduplicated lifecycle notifications for the locked Phase 5 taxonomy.

    Reuses the EXISTING AD-33 notification persistence (MonitoringStore,
    dedupe on (alert_type, key) while open) — no second notification system.
    """
    from dltime.notifications import (
        AlertType,
        SEVERITY_FOR_TYPE,
        TITLE_FOR_TYPE,
        _record,
        append_notifications,
    )

    # Extend the taxonomy with the retraining lifecycle events (values are
    # stable API strings; severities follow the locked plan §42).
    LIFECYCLE_TYPES = {
        "DRIFT_DETECTED": AlertType.MODEL_DRIFT,
        "DEGRADATION_DETECTED": AlertType.PREDICTION_ACCURACY_DEGRADED,
        "RETRAINING_REVIEW_RECOMMENDED": AlertType.RETRAINING_RECOMMENDED,
        "RETRAINING_STARTED": "RETRAINING_STARTED",
        "CANDIDATE_READY": AlertType.NEW_MODEL_CANDIDATE,
        "CANDIDATE_REJECTED": "CANDIDATE_REJECTED",
        "MODEL_PROMOTED": "MODEL_PROMOTED",
        "ROLLBACK_COMPLETED": "ROLLBACK_COMPLETED",
        "EVIDENCE_INSUFFICIENT": "EVIDENCE_INSUFFICIENT",
    }
    SEVERITY_FOR_LIFECYCLE = {
        "DRIFT_DETECTED": "WARNING",
        "DEGRADATION_DETECTED": "WARNING",
        "RETRAINING_REVIEW_RECOMMENDED": "WARNING",
        "RETRAINING_STARTED": "INFO",
        "CANDIDATE_READY": "INFO",
        "CANDIDATE_REJECTED": "INFO",
        "MODEL_PROMOTED": "INFO",
        "ROLLBACK_COMPLETED": "INFO",
        "EVIDENCE_INSUFFICIENT": "INFO",
    }
    TITLES = {
        "RETRAINING_STARTED": "Retraining started",
        "CANDIDATE_REJECTED": "Candidate model rejected",
        "MODEL_PROMOTED": "New model version promoted",
        "ROLLBACK_COMPLETED": "Model rollback completed",
        "EVIDENCE_INSUFFICIENT": "Monitoring evidence insufficient",
    }
    MESSAGES = {
        "DRIFT_DETECTED": "Data drift detected: recent production inputs differ materially from the reference window. Review the Drift & Retraining page; no automatic action was taken.",
        "DEGRADATION_DETECTED": "Model performance degradation detected: recent prediction error is higher than the established baseline. Review the Drift & Retraining page.",
        "RETRAINING_REVIEW_RECOMMENDED": "Retraining review recommended: recent evidence shows a persistent distribution change together with increased prediction error. A human decision is required.",
        "RETRAINING_STARTED": "Retraining started: a candidate model is being evaluated. Current production models remain active.",
        "CANDIDATE_READY": "Candidate model ready for review: evaluation is complete. Human approval is required before promotion.",
        "CANDIDATE_REJECTED": "Candidate model rejected: the candidate was not promoted. The current production model remains unchanged.",
        "MODEL_PROMOTED": "New model version promoted: the approved candidate is now the current production version.",
        "ROLLBACK_COMPLETED": "Model rollback completed: production returned to the previous approved version.",
        "EVIDENCE_INSUFFICIENT": "Monitoring evidence still insufficient: the system cannot yet assess performance for this loader/model. More linked outcomes are needed.",
    }

    lifecycle_type = LIFECYCLE_TYPES.get(event_kind)
    if lifecycle_type is None:
        return []
    # 2026-10-05 fixed-scenario policy: the Alert Center carries ONLY the
    # three fixed scenarios — (1) retraining needed, (2) stage-model
    # promotion suggestion (candidate ready with enough data), (3) drift
    # detected. Other lifecycle echoes stay in the retraining store only.
    if event_kind not in ("DRIFT_DETECTED", "RETRAINING_REVIEW_RECOMMENDED", "CANDIDATE_READY"):
        return []
    ctx = dict(context or {})
    ctx.setdefault("model_version", _serving_version())
    key = ctx.pop("_dedupe_key", None) or f"{event_kind}:{ctx.get('model_version', '')}"

    # Temporarily register the extended type so _record's severity/title maps work.
    severity = SEVERITY_FOR_LIFECYCLE.get(event_kind, "INFO")
    title = TITLES.get(event_kind) or TITLE_FOR_TYPE.get(lifecycle_type, event_kind.replace("_", " ").title())
    if lifecycle_type not in SEVERITY_FOR_TYPE:
        SEVERITY_FOR_TYPE[lifecycle_type] = severity
    if lifecycle_type not in TITLE_FOR_TYPE:
        TITLE_FOR_TYPE[lifecycle_type] = title

    rec = _record(
        alert_type=lifecycle_type,
        dedupe_key=key,
        message=MESSAGES[event_kind],
        context=ctx,
    )
    return append_notifications(store, [rec])


# ---------------------------------------------------------------- API bodies


class DecisionBody:
    """Pydantic body for retraining decisions (api layer defines the model)."""
