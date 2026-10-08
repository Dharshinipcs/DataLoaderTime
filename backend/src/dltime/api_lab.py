"""Module 7 API surface (Model Laboratory / registry / drift / readiness).

Read-only inspection endpoints + EXPLICIT human-gated actions. Promotion and
rollback are POST actions that require an explicit ``approved_by`` + ``reason``
in the body (local approval action — the product has no user-authenticated
identity mechanism, and none is invented). Nothing trains or promotes
automatically.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dltime.model_lab import LabError, _median


class GovernanceActionBody(BaseModel):
    """Human-approval body for promotion / rollback / rejection."""

    model_config = {"extra": "forbid"}

    approved_by: str = Field(min_length=1, description="Local approver identity (explicit action, not an auth system)")
    reason: str = Field(min_length=1, description="Why this action is taken; persisted in the lifecycle log")
    stages: list[str] | None = Field(
        default=None,
        description="Rollback: reset ONLY these stages' serving overrides (per-stage rollback). Omit for the full registry rollback.",
    )


class RetrainingDecisionBody(BaseModel):
    """Human decision body: continue monitoring OR start retraining."""

    model_config = {"extra": "forbid"}

    decision_type: str = Field(description="CONTINUE_MONITORING or START_RETRAINING")
    reason: str = Field(min_length=1)
    decided_by: str = Field(min_length=1)


class CandidateTrainBody(BaseModel):
    """Body for the human-gated candidate-training action."""

    model_config = {"extra": "forbid"}

    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    stages: list[str] | None = Field(
        default=None,
        description="Optional stage list for PER-STAGE retraining (one single-stage candidate per stage). Default: all five stages (today's whole-bundle path, unchanged).",
    )


def wire_model_lab(app, get_registry, get_lab_context) -> None:
    """Attach Module 7 endpoints to the existing FastAPI app.

    get_registry / get_lab_context are lru-cached providers supplied by api.py
    so tests can override them cleanly.
    """
    import json
    from pathlib import Path

    @app.get("/laboratory/experiments")
    def laboratory_experiments() -> dict:
        """Registered experiment records (append-only lineage)."""
        return {"experiments": get_registry().experiments.list()}

    @app.get("/laboratory/experiments/{experiment_id}")
    def laboratory_experiment_detail(experiment_id: str) -> dict:
        rec = get_registry().experiments.get(experiment_id)
        if rec is None:
            raise HTTP404(f"Experiment '{experiment_id}' not found.")
        return rec

    @app.get("/models")
    def models_list() -> dict:
        from dltime.model_lab import ModelState

        reg = get_registry()
        return {
            "models": reg.models.list(),
            "promoted_model_id": (reg.models.promoted_model() or {}).get("model_id"),
            "lifecycle_states": [
                ModelState.EXPERIMENTAL,
                ModelState.CANDIDATE_EVALUATED,
                ModelState.PROMOTED,
                ModelState.RETIRED,
                ModelState.REJECTED,
            ],
        }

    @app.get("/models/status")
    def models_status() -> dict:
        reg = get_registry()
        promoted = reg.models.promoted_model()
        # Sum-total policy (EXP-038): the SERVING total is the deterministic
        # five-stage sum; the stage-aware and direct total models are both
        # retired reference/rollback paths.
        from dltime.sum_total_serving import SUM_TOTAL_MODEL_VERSION

        direct = reg.models.get("total-et-v3-candidate")
        return {
            "serving_model_version": SUM_TOTAL_MODEL_VERSION,
            "serving_role": "AUTHORITATIVE_TOTAL",
            # Governance status: the registry is "promoted" whenever a
            # governed entry currently holds PROMOTED (today: the
            # stage-serving candidate promoted during the Oct-5 manual
            # lifecycle test) — not tied to one historical model id.
            "serving_is_promoted": bool(promoted),
            "promoted_model": promoted,
            "reference_model": {
                "model_id": "total-et-v3-candidate",
                "status": (direct or {}).get("status"),
                "role": "REFERENCE / ROLLBACK / HISTORICAL BASELINE",
            },
            "note": (
                "The authoritative serving total is the deterministic sum of the "
                "five operational stage predictions (EXP-038 product decision; "
                "no learned total model, no floor, approvals excluded). The "
                "stage-aware and direct total models remain registered as "
                "reference/rollback evidence and are never the primary ETA."
            ),
        }

    @app.get("/models/compatibility/{model_id}")
    def models_compatibility(model_id: str) -> dict:
        rec = get_registry().models.get(model_id)
        if rec is None:
            raise HTTP404(f"Model '{model_id}' not registered.")
        from dltime.model_lab import compatibility_report

        return compatibility_report(rec.get("artifact_path", ""))

    @app.post("/models/promote")
    def models_promote(model_id: str, body: GovernanceActionBody) -> dict:
        reg = get_registry()
        rec = reg.models.get(model_id)
        if rec is None:
            raise HTTP404(f"Model '{model_id}' not registered.")
        from dltime.retraining import CANDIDATE_NAME_PREFIXES

        if str(model_id).startswith(CANDIDATE_NAME_PREFIXES):
            from dltime.retraining import candidate_bundle_report, candidate_stage_report

            # Single-stage candidates are gated against the
            # single-stage serving contract; bundles against the
            # five-stage contract.
            if rec.get("model_family") == "stage-candidate":
                comp = candidate_stage_report(rec.get("artifact_path", ""), rec.get("stage"))
            else:
                comp = candidate_bundle_report(rec.get("artifact_path", ""))
        else:
            from dltime.model_lab import compatibility_report

            comp = compatibility_report(rec.get("artifact_path", ""))
        blocking = [k for k, v in comp.items() if v is False]
        if blocking:
            raise HTTP422(f"Promotion refused: compatibility gates failed: {blocking}")
        try:
            updated = reg.models.promote(
                model_id, compatibility=comp, approved_by=body.approved_by, reason=body.reason
            )
        except LabError as exc:
            raise HTTP422(str(exc))
        # Candidate BUNDLES carry their own serving artifact: switch the stage
        # serving path to the promoted bundle (explicit, persisted, after the
        # registry promotion succeeded).
        serving_switched = False
        from dltime.retraining import CANDIDATE_NAME_PREFIXES

        if str(model_id).startswith(CANDIDATE_NAME_PREFIXES):
            from dltime import retraining

            family = rec.get("model_family")
            if family == "stage-candidate":
                # PER-STAGE promotion: activates ONLY this stage's
                # serving override (every other stage keeps serving
                # its current model) and writes this stage's
                # lineage snapshot.
                stage = rec.get("stage")
                if not stage:
                    raise HTTP422(
                        "Promotion refused: single-stage candidate record carries no stage."
                    )
                retraining.promote_candidate_stage(
                    registry=reg, stage=stage, candidate_version=model_id,
                    approved_by=body.approved_by, reason=body.reason,
                )
                serving_switched = True
            elif family == "sum-of-five-stages-candidate":
                # Bundle gates: the uncertainty calibration is bound to the
                # deterministic sum policy (sum-of-five-stages-v1), so a candidate
                # bundle serves under the SAME total policy — no calibration swap
                # is possible or needed. Refuse anything else.
                retraining.promote_candidate_bundle(
                    registry=reg, candidate_version=model_id, approved_by=body.approved_by, reason=body.reason
                )
                serving_switched = True
            else:
                raise HTTP422(
                    "Promotion refused: unknown candidate family "
                    f"{family!r} (expected 'stage-candidate' or 'sum-of-five-stages-candidate')."
                )
        _notify_lifecycle("MODEL_PROMOTED", {"model_id": model_id, "previous": (updated.get("promotion") or {}).get("previous_model_id"), "reason": body.reason, "approved_by": body.approved_by})
        return {"promoted": updated, "compatibility": comp, "serving_switched": serving_switched}

    @app.post("/models/rollback")
    def models_rollback(body: GovernanceActionBody) -> dict:
        # ---- per-stage rollback (optional ``stages``): reset ONLY
        # the named stages' serving overrides; other stages keep
        # serving their current models. The registry-level rollback
        # below stays the whole-model path.
        if body.stages:
            from dltime.stage_targets import ML_STAGE_NAMES

            invalid = [s for s in body.stages if s not in ML_STAGE_NAMES]
            if invalid:
                raise HTTP422(
                    f"Unknown stage(s) {invalid}; expected a subset of {ML_STAGE_NAMES}."
                )
            from dltime import retraining as _retraining

            restored_stages: dict[str, dict[str, Any]] = {}
            for stage in body.stages:
                restored_stages[stage] = _retraining.rollback_stage(
                    stage=stage, approved_by=body.approved_by, reason=body.reason
                )
            _notify_lifecycle(
                "ROLLBACK_COMPLETED",
                {"restored_stages": list(body.stages), "reason": body.reason, "approved_by": body.approved_by},
            )
            return {"restored": None, "restored_stages": restored_stages, "serving_reset": True}
        try:
            restored = get_registry().models.rollback(approved_by=body.approved_by, reason=body.reason)
        except LabError as exc:
            raise HTTP422(str(exc))
        # If serving was switched to a (now retired) candidate bundle, return
        # serving to the standard deterministic sum path.
        from dltime import retraining_serving

        serving_reset = False
        if retraining_serving.active_version() is not None:
            from dltime.retraining import rollback_serving_path

            rollback_serving_path()
            serving_reset = True
        _notify_lifecycle("ROLLBACK_COMPLETED", {"restored": restored.get("model_id"), "reason": body.reason, "approved_by": body.approved_by})
        return {"restored": restored, "serving_reset": serving_reset}

    @app.post("/models/{model_id}/reject")
    def models_reject(model_id: str, body: GovernanceActionBody) -> dict:
        try:
            updated = get_registry().models.reject(model_id, reason=body.reason, approved_by=body.approved_by)
        except LabError as exc:
            raise HTTP422(str(exc))
        _notify_lifecycle("CANDIDATE_REJECTED", {"model_id": model_id, "reason": body.reason, "approved_by": body.approved_by})
        return {"rejected": updated}

    @app.get("/drift/features")
    def drift_features() -> dict:
        ctx = get_lab_context()
        from dltime.model_lab import feature_drift

        return feature_drift(
            ctx["reference_frame"],
            ctx["current_frame"],
            feature_names=ctx["feature_names"],
            categorical_features=ctx["categorical_features"],
        )

    @app.get("/drift/population")
    def drift_population() -> dict:
        ctx = get_lab_context()
        from dltime.model_lab import population_drift

        return population_drift(ctx["reference_frame"], ctx["current_frame"])

    @app.get("/drift/performance")
    def drift_performance() -> dict:
        store = get_lab_context()["monitoring_store"]
        from dltime.model_lab import prediction_performance_drift

        feedback, _w = store.read_feedback_records()
        return prediction_performance_drift(feedback)

    @app.get("/retraining/readiness")
    def retraining_readiness_endpoint() -> dict:
        ctx = get_lab_context()
        from dltime.model_lab import (
            feature_drift,
            population_drift,
            retraining_readiness,
        )

        feedback, _w = ctx["monitoring_store"].read_feedback_records()
        fd = feature_drift(
            ctx["reference_frame"],
            ctx["current_frame"],
            feature_names=ctx["feature_names"],
            categorical_features=ctx["categorical_features"],
        )
        pd_ = population_drift(ctx["reference_frame"], ctx["current_frame"])
        outcomes, _w2 = ctx["monitoring_store"].read_actual_outcomes()
        eligible = sum(1 for o in outcomes if o.get("training_eligible"))
        return retraining_readiness(
            registry=get_registry().models,
            matched_feedback=feedback,
            feature_drift_report=fd,
            population_drift_report=pd_,
            eligible_actual_count=eligible,
        )

    @app.get("/drift/overview")
    def drift_overview_endpoint(loader: str | None = None) -> dict:
        """Composed, evidence-backed Drift & Retraining payload (locked plan §30).

        Reuses the EXISTING drift/performance/readiness signals — no second
        monitoring system, no new statistics, nothing trains or promotes.
        Optional ``loader`` scope (§10) re-runs the SAME feature/population
        drift on that loader's reference/current frames; the standard
        small-sample policy guards apply (never a verdict without evidence).
        """
        ctx = get_lab_context()
        from dltime.model_lab import (
            drift_overview,
            feature_drift,
            population_drift,
            prediction_performance_drift,
        )

        store = ctx["monitoring_store"]
        feedback, _w = store.read_feedback_records()
        events, _w3 = store.read_prediction_events()
        ref_frame = ctx["reference_frame"]
        cur_frame = ctx["current_frame"]
        scope_payload: dict[str, Any] = {"kind": "system", "value": None}
        if loader:
            scope_payload = {"kind": "loader", "value": loader}
            if "Loader_Name" in ref_frame.columns:
                ref_frame = ref_frame[ref_frame["Loader_Name"].astype(str) == loader]
            if "Loader_Name" in cur_frame.columns:
                cur_frame = cur_frame[cur_frame["Loader_Name"].astype(str) == loader]
        fd = feature_drift(
            ref_frame,
            cur_frame,
            feature_names=ctx["feature_names"],
            categorical_features=ctx["categorical_features"],
        )
        pd_ = population_drift(ref_frame, cur_frame)
        perf = prediction_performance_drift(feedback)
        outcomes, _w2 = store.read_actual_outcomes()
        eligible = sum(1 for o in outcomes if o.get("training_eligible"))
        result = drift_overview(
            registry=get_registry().models,
            matched_feedback=feedback,
            prediction_events=events,
            feature_drift_report=fd,
            population_drift_report=pd_,
            performance_report=perf,
            selection_experiments=get_registry().experiments.list(),
            reference_window={
                "first_created": (ctx.get("reference_window") or {}).get("first_start"),
                "last_created": (ctx.get("reference_window") or {}).get("last_start"),
            },
            eligible_actual_count=eligible,
            serving_epochs=_serving_epochs(get_registry()),
        )
        result["scope"] = scope_payload
        return result

    @app.get("/monitoring/stage-lineage")
    def stage_lineage_status() -> dict:
        """Per-stage Baseline / Previous / Current lineage — the source
        for the Drift & Retraining comparison table — plus the active
        candidate row. Seeds the lineage ONCE for today's serving."""
        from dltime import retraining_serving, stage_lineage
        from dltime.stage_targets import ML_STAGE_NAMES

        reg = get_registry()
        # One-time seed: today's serving models + their chronological
        # holdout baselines (no-op once the lineage holds records).
        stage_lineage.seed(_selection_baselines(reg))
        roles = stage_lineage.stage_roles()
        overrides = retraining_serving.active_stage_overrides()
        bundle = retraining_serving.switch_record() or {}
        active_candidate: dict[str, Any] | None = None
        if overrides:
            activated = [str(ov.get("activated_at") or "") for ov in overrides.values()]
            active_candidate = {
                "kind": "stage-candidate",
                "stages": sorted(overrides),
                "candidate_versions": {
                    stage: ov.get("candidate_version") for stage, ov in overrides.items()
                },
                "activated_at": max(activated) if activated else None,
            }
        elif bundle.get("candidate_version"):
            active_candidate = {
                "kind": "bundle-candidate",
                "candidate_version": bundle.get("candidate_version"),
                "activated_at": bundle.get("activated_at"),
            }
        return {
            "schema_version": stage_lineage.LINEAGE_SCHEMA_VERSION,
            "stages": {
                stage: {
                    "baseline": roles.get(stage, {}).get("baseline"),
                    "previous": roles.get(stage, {}).get("previous"),
                    "current": roles.get(stage, {}).get("current"),
                }
                for stage in ML_STAGE_NAMES
            },
            "active_candidate": active_candidate,
        }

    @app.get("/models/lifecycle")
    def models_lifecycle() -> dict:
        """Human-readable model lifecycle timeline (§27) from the EXISTING
        append-only governance event log + registry — no new persistence."""
        from dltime.model_lab import ModelState

        reg = get_registry()
        out: list[dict[str, Any]] = []
        for ev in reg.models.events.list():
            et = str(ev.get("event_type", ""))
            payload = ev.get("payload") or {}
            entry: dict[str, Any] = {"at": ev.get("at"), "model_id": ev.get("model_id"), "event_type": et}
            if et == "REGISTERED":
                entry["kind"] = "registered"
                entry["label"] = f"Model '{ev.get('model_id')}' registered"
            elif et == "PROMOTION_APPROVED" or et == ModelState.PROMOTED:
                entry["kind"] = "promoted"
                prev = payload.get("previous") or payload.get("previous_model_id")
                entry["label"] = (
                    f"Model '{ev.get('model_id')}' approved and promoted"
                    + (f" (replacing '{prev}')" if prev and prev != "null" else "")
                )
            elif et == "ROLLBACK":
                entry["kind"] = "rollback"
                entry["label"] = (
                    f"Rolled back to '{ev.get('model_id')}'"
                    + (f" (from '{payload.get('from_model')}')" if payload.get("from_model") else "")
                )
            elif et == ModelState.RETIRED:
                entry["kind"] = "retired"
                entry["label"] = f"Model '{ev.get('model_id')}' retired"
            elif et == ModelState.REJECTED:
                entry["kind"] = "rejected"
                entry["label"] = f"Model '{ev.get('model_id')}' rejected"
            else:
                entry["kind"] = "event"
                entry["label"] = f"{ev.get('model_id')}: {et.lower().replace('_', ' ')}"
            if payload.get("reason"):
                entry["reason"] = payload["reason"]
            out.append(entry)
        return {"events": out}

    # ------------------------------------------------ retraining workflow (§9-16)

    def _current_serving_version() -> str:
        """The version string of what stage serving would load right now."""
        try:
            from dltime import retraining_serving

            active = retraining_serving.active_version()
            if active:
                return active
        except Exception:
            pass
        return "sum-of-five-stages-v1 (stage-serving-v2)"

    def _serving_epochs(reg) -> dict[str, dict[str, Any]]:
        """Per-stage serving epoch — the moment the now-serving stage
        model was activated. An active per-stage override carries its
        own switch time ('serving-switch'); an active whole-bundle
        switch shares its activation time; otherwise the registry
        promoted model's promotion time ('registry')."""
        from dltime import retraining_serving
        from dltime.stage_targets import ML_STAGE_NAMES

        promoted = reg.models.promoted_model() or {}
        registry_epoch: dict[str, Any] = {
            "model_id": promoted.get("model_id"),
            "since": promoted.get("promoted_at"),
            "source": "registry",
        }
        bundle = retraining_serving.switch_record() or {}
        bundle_epoch: dict[str, Any] | None = None
        if bundle.get("activated_at"):
            bundle_epoch = {
                "model_id": bundle.get("candidate_version"),
                "since": bundle.get("activated_at"),
                "source": "serving-switch",
            }
        overrides = retraining_serving.active_stage_overrides()
        epochs: dict[str, dict[str, Any]] = {}
        for stage in ML_STAGE_NAMES:
            ov = overrides.get(stage)
            if ov and ov.get("activated_at"):
                epochs[stage] = {
                    "model_id": ov.get("candidate_version"),
                    "since": ov.get("activated_at"),
                    "source": "serving-switch",
                }
            elif bundle_epoch is not None:
                epochs[stage] = dict(bundle_epoch)
            else:
                epochs[stage] = dict(registry_epoch)
        return epochs

    def _notify_lifecycle(event_kind: str, context: dict[str, Any]) -> None:
        """Best-effort deduplicated lifecycle notification (never blocks the action)."""
        try:
            from dltime import retraining

            store = get_lab_context()["monitoring_store"]
            retraining.lifecycle_notifications(event_kind=event_kind, store=store, context=context)
        except Exception:
            pass

    @app.get("/retraining/decisions")
    def retraining_decisions() -> dict:
        from dltime.retraining import list_decisions

        return {"decisions": list_decisions()}

    @app.post("/retraining/decide")
    def retraining_decide(body: RetrainingDecisionBody) -> dict:
        """Persist the human monitoring decision (§9/§10). Nothing trains here."""
        from dltime.retraining import record_decision

        try:
            rec = record_decision(
                decision_type=body.decision_type,
                reason=body.reason,
                decided_by=body.decided_by,
            )
        except LabError as exc:
            raise HTTP422(str(exc))
        if body.decision_type == "START_RETRAINING":
            _notify_lifecycle("RETRAINING_STARTED", {"reason": body.reason, "decided_by": body.decided_by})
        return {"decision": rec}

    @app.post("/retraining/train-candidate")
    def retraining_train_candidate(body: CandidateTrainBody) -> dict:
        """Human-gated candidate training + evaluation (§11-13).

        Uses the SAME approved stage families / frames / chronological protocol.
        Registers a CANDIDATE_EVALUATED model + experiment record; SERVING IS
        NOT TOUCHED. A separate human promotion is the only path to production.
        """
        from dltime import retraining
        from dltime.model_lab import utc_now_iso
        from dltime.stage_targets import ML_STAGE_NAMES

        reg = get_registry()

        # ---- per-stage retraining (optional ``stages``): one
        # single-stage candidate per requested stage, identical
        # protocol scoped to that stage. Serving is NOT touched.
        if body.stages:
            invalid = [s for s in body.stages if s not in ML_STAGE_NAMES]
            if invalid:
                raise HTTP422(
                    f"Unknown stage(s) {invalid}; expected a subset of {ML_STAGE_NAMES}."
                )
            from dltime.model_lab import CANDIDATE_EVALUATION_MIN_RATIO

            registered_all: list[dict[str, Any]] = []
            evaluations: list[dict[str, Any]] = []
            for stage in body.stages:
                result = retraining.build_candidate_stage(stage)
                evaluation = retraining.candidate_evaluation(
                    candidate_version=result["candidate_version"],
                    per_stage=result["evaluation"]["per_stage"],
                    baseline_per_stage=_selection_baselines(reg),
                    total={},
                    baseline_total=None,
                )
                # A single-stage candidate is judged on ITS stage's
                # holdout ratio (one stage does not move the total).
                row = (evaluation["per_stage"] or [{}])[0]
                ratio = row.get("ratio")
                if ratio is None:
                    evaluation["verdict"] = "INSUFFICIENT_EVIDENCE"
                    evaluation["rationale"] = (
                        f"Stage '{stage}' has no persisted chronological-holdout baseline to compare against."
                    )
                elif ratio <= CANDIDATE_EVALUATION_MIN_RATIO:
                    evaluation["verdict"] = "ACCEPTABLE_FOR_REVIEW"
                    evaluation["rationale"] = (
                        f"Stage '{stage}' holdout MedAE is {ratio}x the stage's chronological-holdout baseline "
                        f"({row.get('candidate_median_ae')}s vs {row.get('baseline_median_ae')}s on the same protocol). "
                        "A human decides promotion; nothing is automatic."
                    )
                else:
                    evaluation["verdict"] = "NOT_SUFFICIENTLY_BETTER"
                    evaluation["rationale"] = (
                        f"Stage '{stage}' holdout MedAE is {ratio}x the stage's chronological-holdout baseline "
                        f"({row.get('candidate_median_ae')}s vs {row.get('baseline_median_ae')}s) — the candidate is NOT "
                        "an improvement for this stage; promotion is not supported by this evidence."
                    )
                experiment_record = {
                    "experiment_id": f"EXP-RETRAIN-{result['candidate_version']}",
                    "created_at": utc_now_iso(),
                    "model_family": "stage-candidate",
                    "target_transformation": "log1p",
                    "hyperparameters": {
                        "stage_kind_map": result["stage_kind_map"],
                        "note": "same approved family for this stage; estimator hyperparameters unchanged from the serving build",
                    },
                    "feature_set_version": "stage-frames-v2",
                    "feature_names": [],
                    "training_population": {
                        "protocol": result["evaluation"]["protocol"],
                        "split": result["evaluation"]["split"],
                    },
                    "validation_method": "chronological holdout (EXP-036 machinery)",
                    "metrics": result["evaluation"]["per_stage"],
                    "status": "EVALUATED",
                    "decision": evaluation["verdict"],
                    "rationale": evaluation["rationale"],
                }
                registered = retraining.persist_candidate_bundle(
                    result=result,
                    experiment_record=experiment_record,
                    registry=reg,
                    model_family="stage-candidate",
                    stage=stage,
                )
                registered_all.append(registered)
                evaluations.append(evaluation)
            _notify_lifecycle(
                "CANDIDATE_READY",
                {
                    "model_id": registered_all[-1]["model_id"],
                    "stages": list(body.stages),
                    "verdict": evaluations[-1]["verdict"],
                },
            )
            single = len(registered_all) == 1
            return {
                "registered": registered_all[0] if single else registered_all,
                "evaluation": evaluations[0] if single else evaluations,
                "stages": list(body.stages),
                "serving_untouched": True,
                "serving_version_before": _current_serving_version(),
            }

        try:
            result = retraining.build_candidate_stage_bundle()
        except LabError as exc:
            raise HTTP422(str(exc))

        evaluation = retraining.candidate_evaluation(
            candidate_version=result["candidate_version"],
            per_stage=result["evaluation"]["per_stage"],
            baseline_per_stage=_selection_baselines(reg),
            total=result["evaluation"]["per_stage"].get("__total__", {}),
            baseline_total=None,
        )
        # Total baseline: compare against the CURRENT serving bundle rebuilt on
        # the same holdout is expensive; the honest comparison uses the
        # candidate's own holdout vs the SELECTION stage baselines, and the
        # total row is compared against the EXP-036 holdout MedAE recorded in
        # the final validation evidence when available.
        exp_medae = _exp036_total_medae()
        if exp_medae is not None and evaluation["total"]["candidate_median_ae"] is not None:
            evaluation["total"]["baseline_median_ae"] = exp_medae
            evaluation["total"]["ratio"] = (
                round(evaluation["total"]["candidate_median_ae"] / exp_medae, 3) if exp_medae else None
            )
            if evaluation["total"]["ratio"] is not None:
                from dltime.model_lab import CANDIDATE_EVALUATION_MIN_RATIO

                if evaluation["total"]["ratio"] <= CANDIDATE_EVALUATION_MIN_RATIO:
                    evaluation["verdict"] = "ACCEPTABLE_FOR_REVIEW"
                    evaluation["rationale"] = (
                        f"Candidate holdout MedAE is {evaluation['total']['ratio']}x the served baseline "
                        f"({evaluation['total']['candidate_median_ae']}s vs {exp_medae}s on the same chronological protocol). "
                        "A human decides promotion; nothing is automatic."
                    )
                else:
                    evaluation["verdict"] = "NOT_SUFFICIENTLY_BETTER"
                    evaluation["rationale"] = (
                        f"Candidate holdout MedAE is {evaluation['total']['ratio']}x the served baseline "
                        f"({evaluation['total']['candidate_median_ae']}s vs {exp_medae}s) — the candidate is NOT an "
                        "improvement; promotion is not supported by this evidence."
                    )

        experiment_record = {
            "experiment_id": f"EXP-RETRAIN-{result['candidate_version']}",
            "created_at": utc_now_iso(),
            "model_family": "sum-of-five-stages-candidate",
            "target_transformation": "log1p",
            "hyperparameters": {
                "stage_kind_map": result["stage_kind_map"],
                "note": "same approved families; estimator hyperparameters unchanged from the serving build",
            },
            "feature_set_version": "stage-frames-v2",
            "feature_names": [],
            "training_population": {
                "protocol": result["evaluation"]["protocol"],
                "split": result["evaluation"]["split"],
            },
            "validation_method": "chronological holdout (EXP-036 machinery)",
            "metrics": result["evaluation"]["per_stage"],
            "status": "EVALUATED",
            "decision": evaluation["verdict"],
            "rationale": evaluation["rationale"],
        }
        try:
            registered = retraining.persist_candidate_bundle(
                result=result, experiment_record=experiment_record, registry=reg
            )
        except LabError as exc:
            raise HTTP422(str(exc))
        _notify_lifecycle("CANDIDATE_READY", {"model_id": registered["model_id"], "verdict": evaluation["verdict"]})
        return {
            "registered": registered,
            "evaluation": evaluation,
            "serving_untouched": True,
            "serving_version_before": _current_serving_version(),
        }

    @app.get("/retraining/serving-status")
    def retraining_serving_status() -> dict:
        from dltime import retraining_serving

        return {
            "serving_version": _current_serving_version(),
            "candidate_override_active": retraining_serving.active_version() is not None,
            "candidate_version": retraining_serving.active_version(),
        }

    def _selection_baselines(reg) -> dict[str, dict[str, Any]]:
        """Per-stage chronological-holdout MedAE baselines from the SELECTION experiments."""
        out: dict[str, dict[str, Any]] = {}
        suffixes = {
            "staging": "STAGING",
            "prevalidation": "PREVALIDATION",
            "transformation": "TRANSFORMATION",
            "pre_load": "PRE_LOAD",
            "data_loading": "DATA_LOADING",
        }
        for key, suffix in suffixes.items():
            best = None
            for rec_ in reg.experiments.list():
                eid = str(rec_.get("experiment_id", ""))
                if eid.startswith("EXP-STAGE-" + suffix + "-") and eid.endswith("SELECTION"):
                    metrics = rec_.get("metrics") or {}
                    medians = [m.get("median_ae") for m in metrics.values() if isinstance(m, dict) and m.get("median_ae") is not None]
                    if medians:
                        best = _median(medians)
            if best is not None:
                out[key] = {"median_ae": best}
        return out

    def _exp036_total_medae() -> float | None:
        """Chronological-holdout MedAE of the served sum (recorded EXP-036 evidence)."""
        try:
            path = Path(__file__).resolve().parents[2] / "artifacts" / "registry" / "served_baseline.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                v = data.get("median_ae")
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return None


def HTTP404(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=404, detail=msg)


def HTTP422(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=422, detail=msg)
