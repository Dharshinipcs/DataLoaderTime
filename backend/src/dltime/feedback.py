"""Prediction monitoring + actual-outcome feedback (Task 11, AD-24).

The operational loop the locked lifecycle describes:

    prediction (Task 04 serving path, UNCHANGED)
      -> prediction event persisted (ALL states, append-only)
      -> later, an actual execution outcome is ingested
      -> link when DEFENSIBLE (exact reference only — see PART A evidence)
      -> total + stage error computed on the matched pair
      -> training eligibility decided by the EXISTING Task 01 rules
      -> monitoring aggregates read-only

PART A linkage evidence (investigated from the real data, recorded in
EXP-021): identity+time fallback is UNSAFE for this dataset — same-identity
executions collide heavily (57.1% of same-identity same-DAY groups contain
>1 execution; up to 32 executions of one identity on a single day), and
`ldr_workflow_execution_id` is present on 100% of executions but only 66%
distinct. The internal synthetic context id is NOT a real-world execution
reference (it exists only inside the prediction request), so linkage is
EXACT-WORKFLOW-REFERENCE ONLY:

    1. EXACT_WORKFLOW_EXECUTION_ID   caller supplies ldr_workflow_execution_id;
                                     unambiguous only when exactly ONE
                                     non-already-linked event shares it
    2. IDENTITY_MOST_RECENT          2026-10-05: fallback for manual entries
                                     that carry no execution reference — link
                                     the most recent unlinked event with
                                     EXACTLY equal canonical identity; no
                                     time-window guessing
    3. else UNMATCHED

AMBIGUOUS when a reference matches several linkable prediction events;
already-linked predictions are never silently re-linked; identical
re-submissions are idempotent; conflicting content for the same execution is
recorded as a conflict, never overwritten.

Post-execution values live ONLY here: they can never enter the pre-execution
prediction feature path (the request schema has no such fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dltime.store import (
    SCHEMA_VERSION,
    MonitoringStore,
    StoreWarning,
    content_hash,
    utc_now_iso,
)

# ------------------------------------------------------------------ constants

LINK_STATUS_MATCHED = "MATCHED"
LINK_STATUS_UNMATCHED = "UNMATCHED"
LINK_STATUS_AMBIGUOUS = "AMBIGUOUS"

LINK_METHOD_WORKFLOW_ID = "EXACT_WORKFLOW_EXECUTION_ID"
LINK_METHOD_IDENTITY = "IDENTITY_MOST_RECENT"

FEEDBACK_SCHEMA_VERSION = SCHEMA_VERSION


# ------------------------------------------------------------- prediction events


def event_id_for(payload: dict[str, Any], created_at: str) -> str:
    """Deterministic, unique-enough event id: timestamp + content digest."""
    return f"pe_{created_at.replace(':', '').replace('-', '')}_{content_hash(payload)}"


def prediction_event_from_result(
    result,  # PredictionResult
    *,
    planned_start_time: str | None,
) -> dict[str, Any]:
    """Build the audit record for ONE prediction attempt (ANY state).

    Captured from the exact response the caller received plus the internal
    feature vector already exposed in the response contract (``features``) —
    no second feature computation, no leakage (these are the same
    pre-execution values the prediction used). Stage estimates are stored as
    the SUPPLEMENTARY evidence they are, never as the total.
    """
    state = result.state
    payload = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "created_at": None,  # filled by caller below
        "event_state": state,
        "request": {
            "loader_name": result.inputs.get("loader_name"),
            "dataset": result.inputs.get("dataset"),
            "sprint": result.inputs.get("sprint"),
            "connection": result.inputs.get("connection"),
            "workflow_execution_id": result.inputs.get("workflow_execution_id"),
            "planned_start_time": planned_start_time,
        },
        "identity": dict(result.identity or {}),
        "prediction": None,
        "explanation_digest": None,
        "stage_intelligence": None,
        "features": dict(result.features) if result.features else None,
        "warnings": list(result.warnings or []),
    }
    if state == "PREDICTED":
        unc = result.uncertainty or {}
        risk = result.risk or {}
        explanation = result.explanation or {}
        payload["prediction"] = {
            "predicted_total_seconds": result.predicted_seconds,
            "model_version": result.model_version,
            "model_status": result.model_status,
            "serving_role": getattr(result, "serving_role", None),
            "direct_reference_seconds": getattr(result, "direct_reference_seconds", None),
            "stage_aware_reference_seconds": getattr(result, "stage_aware_reference_seconds", None),
            "prior_count": (result.history or {}).get("prior_loader_executions"),
            "interval_lower_seconds": unc.get("lower_seconds"),
            "interval_upper_seconds": unc.get("upper_seconds"),
            "interval_level": unc.get("interval_level"),
            "risk_label": risk.get("label"),
            "risk_policy_version": risk.get("policy_version"),
        }
        # compact explanation evidence: additivity + top-3 ranked contributors
        tops = [
            {
                "feature": f.get("feature"),
                "display_name": f.get("display_name"),
                "direction": f.get("direction"),
                "shap_value_log1p": f.get("shap_value_log1p"),
            }
            for f in (explanation.get("top_features") or [])[:3]
        ]
        payload["explanation_digest"] = {
            "method": explanation.get("method"),
            "shap_version": explanation.get("shap_version"),
            "model_output_log1p": explanation.get("model_output_log1p"),
            "additivity_abs_error": explanation.get("additivity_abs_error"),
            "top_contributors": tops,
        }
        if result.stage_intelligence and result.stage_intelligence.get("available"):
            payload["stage_intelligence"] = {
                "note": (
                    "five operational stage predictions; their exact sum IS the "
                    "served total (approvals excluded)"
                ),
                "stages": [
                    {
                        "stage": s.get("stage"),
                        "available": bool(s.get("available")),
                        "estimate_seconds": s.get("estimate_seconds"),
                        "evidence_count": s.get("evidence_count"),
                    }
                    for s in result.stage_intelligence.get("stages", [])
                ],
            }
    return payload


def finalize_event_id(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill created_at + deterministic event_id; returns the finished record."""
    created = utc_now_iso()
    payload["created_at"] = created
    payload["event_id"] = event_id_for(payload, created)
    return payload


# ------------------------------------------------------------- actual outcomes


@dataclass(frozen=True)
class ActualOutcomeRecord:
    outcome_id: str
    outcome_hash: str
    execution_id: str | None
    workflow_execution_id: str | None
    identity: dict[str, Any]
    timing: dict[str, Any]
    actual: dict[str, Any]
    stage_actuals: dict[str, float]
    training_eligible: bool
    training_eligibility_reason: str
    stored_at: str
    warnings: list[str]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def build_actual_outcome(
    *,
    loader_name: str,
    dataset: str,
    sprint: str,
    connection: str,
    start_time: str,
    end_time: str | None,
    ldr_execution_id: str | None = None,
    workflow_execution_id: str | None = None,
    actual_total_seconds: float | None = None,
    status: str | None = None,
    stage_actuals: dict[str, float] | None = None,
    record_counts: dict[str, int | None] | None = None,
    stage_minutes: dict[str, float | None] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate + build an actual-outcome record (NOT yet persisted).

    Total-duration authority: the caller reports ``actual_total_seconds`` as
    recorded by the same execution-level measure the Task 01 target uses
    (``timetaken_in_sec``). Training eligibility reuses the EXISTING Task 01
    semantics — completed status (AD-08 policy) + finite positive target.
    Nothing is invented: a non-positive/missing target is stored for audit
    but marked NOT TRAINING ELIGIBLE.

    AD-32 §26/§35: XLSX-faithful post-execution fields. ``stage_minutes``
    carries the source's ``*_timetaken_in_min`` values converted to the
    canonical SECONDS stage map (min x 60 — identical semantics to the
    stage-target source policy). ``record_counts`` carries the source's
    record/error counts verbatim. Data-quality problems (timestamp order,
    record-count consistency) are appended as ``warnings`` and never block
    audit storage — but they are surfaced to the notification center.
    """
    errors: list[str] = []
    quality_warnings: list[str] = []
    if not loader_name or not dataset or not sprint or not connection:
        errors.append("missing_required_identity_field")
    total = _num(actual_total_seconds)
    if total is None:
        errors.append("missing_or_invalid_actual_total_seconds")
    start_norm = str(start_time).strip() if start_time else ""
    if not start_norm:
        errors.append("missing_start_time")
    if errors:
        return None, ";".join(errors)

    # ---- AD-32 §35 data-quality validation (warn, never silently accept) --
    end_norm = str(end_time).strip() if end_time else None
    start_ts = pd.to_datetime(start_norm, errors="coerce")
    end_ts = pd.to_datetime(end_norm, errors="coerce") if end_norm else None
    if end_norm is not None and (end_ts is pd.NaT or start_ts is pd.NaT or end_ts < start_ts):
        quality_warnings.append("timestamp_order_suspicious: end_time precedes start_time or is unparseable")
    if total is not None and end_ts is not None and start_ts is not pd.NaT and end_ts is not pd.NaT and end_ts >= start_ts:
        span_sec = (end_ts - start_ts).total_seconds()
        if span_sec > 0 and total > span_sec * 3:
            quality_warnings.append("duration_exceeds_timestamp_span_3x")
    counts = {k: v for k, v in (record_counts or {}).items() if v is not None}
    if counts.get("total_records") is not None and counts.get("success_records") is not None and counts.get("error_records") is not None:
        if counts["success_records"] + counts["error_records"] != counts["total_records"]:
            quality_warnings.append("record_count_inconsistent: success + error != total")

    status_clean = (str(status).strip() if status else None)
    known_status = status_clean in {
        "SUCCEED",
        "COMPLETED_WITH_WARNING",
        "COMPLETED_WITH_ERROR",
        "COMPLETED_WITH_ERROR_AND_WARNING",
        "ERROR",
        "STOPPED",
        "KILLED",
    }
    # Task 01 AD-08 status policy (completed statuses only)
    completed = status_clean in {
        "SUCCEED",
        "COMPLETED_WITH_WARNING",
        "COMPLETED_WITH_ERROR",
        "COMPLETED_WITH_ERROR_AND_WARNING",
    }
    target_valid = total is not None and total > 0
    identity_complete = True  # enforced above
    eligible = bool(target_valid and completed and identity_complete and known_status)
    if not known_status:
        reason = "UNKNOWN_STATUS"
    elif not target_valid:
        reason = "INVALID_TARGET"
    elif not completed:
        reason = "STATUS_NOT_TRAINING_ELIGIBLE"
    else:
        reason = "ELIGIBLE"

    payload = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "stored_at": utc_now_iso(),
        "outcome_hash": "",  # filled below over the content fields
        "execution_ref": {
            "ldr_execution_id": str(ldr_execution_id).strip() if ldr_execution_id else None,
            "workflow_execution_id": str(workflow_execution_id).strip() if workflow_execution_id else None,
        },
        "identity": {
            "loader_name": str(loader_name).strip(),
            "dataset": str(dataset).strip(),
            "sprint": str(sprint).strip(),
            "connection": str(connection).strip(),
        },
        "timing": {"start_time": start_norm, "end_time": str(end_time).strip() if end_time else None},
        "actual": {
            "actual_total_seconds": total,
            "status": status_clean,
        },
        "record_counts": dict(counts),  # XLSX-faithful post-execution counts
        "stage_minutes_reported": {
            str(k).strip(): _num(v)
            for k, v in (stage_minutes or {}).items()
            if _num(v) is not None
        },
        "stage_actuals": {
            str(k).strip(): v
            for k, v in (stage_actuals or {}).items()
            if _num(v) is not None
        },
        "training_eligible": eligible,
        "training_eligibility_reason": reason,
        "data_quality_warnings": quality_warnings,
    }
    content_fields = {k: v for k, v in payload.items() if k not in ("stored_at", "outcome_hash")}
    payload["outcome_hash"] = content_hash(content_fields)
    payload["outcome_id"] = f"ao_{payload['outcome_hash']}"
    return payload, None# ------------------------------------------------------------------- linkage


@dataclass
class LinkageCandidateView:
    """Minimal read model of a persisted prediction event for linkage."""

    event_id: str
    event_state: str
    loader_name: str | None
    dataset: str | None
    sprint: str | None
    connection: str | None  # resolved authoritative connection when available
    workflow_execution_id: str | None
    created_at: str = ""    # event timestamp for the identity fallback (most recent wins)


def _candidate_view(event: dict[str, Any]) -> LinkageCandidateView:
    req = event.get("request") or {}
    ident = event.get("identity") or {}
    return LinkageCandidateView(
        event_id=event.get("event_id", ""),
        event_state=event.get("event_state", ""),
        loader_name=req.get("loader_name"),
        dataset=req.get("dataset"),
        sprint=req.get("sprint"),
        connection=ident.get("resolved_connection") or req.get("connection"),
        workflow_execution_id=req.get("workflow_execution_id"),
        created_at=str(event.get("created_at") or event.get("created") or ""),
    )


def _identity_matches(a: LinkageCandidateView, b: dict[str, Any]) -> bool:
    """Canonical identity equality (Dataset + Connection + Sprint + Loader)."""
    return (
        (a.loader_name or "").strip().lower() == str(b.get("loader_name", "")).strip().lower()
        and (a.dataset or "").strip().lower() == str(b.get("dataset", "")).strip().lower()
        and (a.sprint or "").strip().lower() == str(b.get("sprint", "")).strip().lower()
        and (a.connection or "").strip().lower() == str(b.get("connection", "")).strip().lower()
    )


def link_actual_to_predictions(
    actual: dict[str, Any],
    prediction_events: list[dict[str, Any]],
    existing_feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    """Link one actual outcome to persisted prediction events (deterministic).

    Hierarchy (PART A/B evidence): exact workflow reference (unique match
    only), else UNMATCHED. Identity equality is a REQUIRED GUARD — a
    reference match with a different canonical identity is not defensible.
    Already-linked prediction events are never silently re-linked; a
    reference hitting several linkable events yields AMBIGUOUS (never
    silently chosen).
    """
    already_linked = {r.get("prediction_event_id") for r in existing_feedback if r.get("linkage_status") == LINK_STATUS_MATCHED}
    candidates = [_candidate_view(e) for e in prediction_events]

    linkable = [
        c
        for c in candidates
        if c.event_id not in already_linked
        and _identity_matches(c, actual["identity"])
        and c.event_state in {"PREDICTED", "INSUFFICIENT_HISTORY"}
    ]

    matched: list[LinkageCandidateView] = []

    ref_wf = (actual.get("execution_ref") or {}).get("workflow_execution_id")
    if ref_wf:
        by_wf = [c for c in linkable if c.workflow_execution_id == str(ref_wf)]
        if len(by_wf) == 1:
            matched = by_wf
        elif len(by_wf) > 1:
            return {
                "linkage_status": LINK_STATUS_AMBIGUOUS,
                "linkage_method": LINK_METHOD_WORKFLOW_ID,
                "prediction_event_ids": [c.event_id for c in by_wf],
                "evidence": {
                    "reference": str(ref_wf),
                    "reason": "multiple linkable prediction events share this workflow reference",
                },
                "linked_at": utc_now_iso(),
            }

    # 2026-10-05 (user request): manual outcome entries usually carry NO
    # execution reference. Fall back to canonical identity ONLY (never
    # identity+time windowing — EXP-021 remains the guard): link the most
    # recent unlinked prediction event whose identity is exactly equal.
    if not matched and linkable:
        target = max(linkable, key=lambda c: (c.created_at, c.event_id))
        return {
            "linkage_status": LINK_STATUS_MATCHED,
            "linkage_method": LINK_METHOD_IDENTITY,
            "prediction_event_ids": [target.event_id],
            "evidence": {
                "reference": None,
                "reason": (
                    "no execution/workflow reference was supplied; linked to the "
                    "most recent unlinked prediction event with exactly equal "
                    "canonical identity (no time-window matching)"
                ),
                "identity": actual["identity"],
            },
            "linked_at": utc_now_iso(),
        }

    if not matched:
        # Distinguish WHY (2026-10-05 user request — the UI must be
        # able to tell the operator exactly why a manual entry did
        # not link): no equal-identity event at all, vs equal-identity
        # events that are all already linked to a prior outcome.
        same_identity = [c for c in candidates if _identity_matches(c, actual["identity"])]
        if same_identity:
            reason = (
                f"{len(same_identity)} prediction event(s) share this exact identity but "
                "all are already linked to a previously recorded outcome; "
                "an outcome is linked once"
            )
        else:
            reason = (
                "no prediction event with this exact identity "
                "(loader + dataset + sprint + connection) exists in the monitoring "
                "store — record the outcome only after a prediction was made for "
                "the same loader, dataset, sprint and connection"
            )
        return {
            "linkage_status": LINK_STATUS_UNMATCHED,
            "linkage_method": None,
            "prediction_event_ids": [],
            "evidence": {"reason": reason},
            "linked_at": utc_now_iso(),
        }

    target = matched[0]
    return {
        "linkage_status": LINK_STATUS_MATCHED,
        "linkage_method": LINK_METHOD_WORKFLOW_ID,
        "prediction_event_ids": [target.event_id],
        "evidence": {
            "reference": (actual.get("execution_ref") or {}).get("workflow_execution_id"),
            "identity": actual["identity"],
        },
        "linked_at": utc_now_iso(),
    }


# --------------------------------------------------------------- error math


def compute_total_error(predicted: float, actual: float) -> dict[str, Any]:
    """Operational prediction-vs-actual error (NOT Task 03 evaluation).

    signed_error = predicted - actual; absolute_error = |signed|;
    percentage error only when actual > 0; log-space absolute error mirrors
    the AD-20 model space. Tolerances only when mathematically meaningful.
    """
    signed = float(predicted) - float(actual)
    absolute = abs(signed)
    meaningful = actual is not None and float(actual) > 0
    pct = (absolute / float(actual)) * 100.0 if meaningful else None
    return {
        "signed_error_seconds": round(signed, 6),
        "absolute_error_seconds": round(absolute, 6),
        "absolute_percentage_error": round(pct, 4) if pct is not None else None,
        "log_absolute_error": round(abs(_log1p(predicted) - _log1p(actual)), 6),
        "within_10_percent": bool(pct <= 10.0) if pct is not None else None,
        "within_20_percent": bool(pct <= 20.0) if pct is not None else None,
        "within_30_percent": bool(pct <= 30.0) if pct is not None else None,
        "within_2x": bool(float(predicted) <= 2.0 * float(actual) and float(actual) <= 2.0 * float(predicted)) if meaningful else None,
    }


def _log1p(v: float) -> float:
    import math

    return math.log1p(max(float(v), 0.0))


def compute_stage_errors(
    predicted_stages: list[dict[str, Any]] | None,
    stage_actuals: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    """Per-model error block ONLY where BOTH sides exist.

    Covers the FIVE operational ML stages AND the TWO approval analytics
    (datamart_approval / dataloading_approval) — every predicted component of
    the total is scored against its own actual so each model can be improved
    individually on retraining. Missing stage actuals are never fabricated;
    nothing is summed into or compared against the authoritative total.
    """
    out: dict[str, dict[str, Any]] = {}
    if not predicted_stages or not stage_actuals:
        return out
    for s in predicted_stages:
        if not s.get("available"):
            continue
        est = s.get("estimate_seconds")
        act = stage_actuals.get(str(s.get("stage")))
        if est is None or act is None or _num(act) is None:
            continue
        a = float(act)
        p = float(est)
        signed = p - a
        pct = (abs(signed) / a * 100.0) if a > 0 else None
        out[str(s["stage"])] = {
            "predicted_seconds": p,
            "actual_seconds": a,
            "signed_error_seconds": round(signed, 6),
            "absolute_error_seconds": round(abs(signed), 6),
            "absolute_percentage_error": round(pct, 4) if pct is not None else None,
            # same method as the authoritative total error (log-space mirrors
            # the models' log1p training objective; AD-20)
            "log_absolute_error": round(abs(_log1p(p) - _log1p(a)), 6),
            "within_20_percent": bool(pct <= 20.0) if pct is not None else None,
            "within_2x": bool(p <= 2.0 * a and a <= 2.0 * p),
        }
    return out


# ------------------------------------------------------------ feedback record


def build_feedback_record(
    *,
    actual: dict[str, Any],
    linkage: dict[str, Any],
    predicted_event: dict[str, Any] | None,
    stage_errors: dict[str, dict[str, Any]] | None = None,
    data_quality_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Append-only feedback record: link + errors + training eligibility.

    Identity/eligibility of the ACTUAL govern training eligibility; an
    INSUFFICIENT_HISTORY prediction event never becomes a training
    observation merely because an actual arrived (the actual's own Task 01
    semantics decide eligibility for future training data, independent of
    the prediction event's serving gate).
    """
    total_errors: dict[str, Any] | None = None
    if (
        linkage.get("linkage_status") == LINK_STATUS_MATCHED
        and predicted_event is not None
        and (predicted_event.get("prediction") or {}).get("predicted_total_seconds") is not None
        and actual.get("actual", {}).get("actual_total_seconds") is not None
    ):
        total_errors = compute_total_error(
            float(predicted_event["prediction"]["predicted_total_seconds"]),
            float(actual["actual"]["actual_total_seconds"]),
        )
    stage_errors = stage_errors or {}
    return {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "feedback_id": f"fb_{content_hash({'p': linkage.get('prediction_event_ids'), 'a': actual.get('outcome_hash'), 'l': linkage.get('linkage_status')})}",
        "prediction_event_id": (linkage.get("prediction_event_ids") or [None])[0],
        "outcome_id": actual.get("outcome_id"),
        "linkage": {
            "status": linkage.get("linkage_status"),
            "method": linkage.get("linkage_method"),
            "evidence": linkage.get("evidence"),
            "linked_at": linkage.get("linked_at"),
        },
        "identity": actual.get("identity"),
        "timing": actual.get("timing"),
        "total_error": total_errors,
        "stage_errors": stage_errors,
        "record_counts": actual.get("record_counts"),
        "data_quality_warnings": data_quality_warnings or actual.get("data_quality_warnings") or [],
        "training_eligible": bool(actual.get("training_eligible")),
        "training_eligibility_reason": actual.get("training_eligibility_reason"),
        "created_at": utc_now_iso(),
    }


# ------------------------------------------------------------ monitoring read


def monitoring_summary(
    prediction_events: list[dict[str, Any]],
    actual_outcomes: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    loader_filter: str | None = None,
) -> dict[str, Any]:
    """Read-only operational aggregate (explicitly NOT Task 03 evaluation)."""
    def _in_loader(rec: dict[str, Any]) -> bool:
        if not loader_filter:
            return True
        return str((rec.get("request") or rec.get("identity") or {}).get("loader_name", "")).strip().lower() == loader_filter.strip().lower()

    events = [e for e in prediction_events if _in_loader(e)]
    outcomes = [a for a in actual_outcomes if _in_loader(a)]
    feedback = [f for f in feedback_records if _in_loader(f)]

    state_counts: dict[str, int] = {}
    for e in events:
        state_counts[e.get("event_state", "?")] = state_counts.get(e.get("event_state", "?"), 0) + 1

    matched = [f for f in feedback if f.get("linkage", {}).get("status") == LINK_STATUS_MATCHED and f.get("total_error")]
    unmatched = [f for f in feedback if f.get("linkage", {}).get("status") == LINK_STATUS_UNMATCHED]
    ambiguous = [f for f in feedback if f.get("linkage", {}).get("status") == LINK_STATUS_AMBIGUOUS]

    metrics: dict[str, Any] | None = None
    slices: dict[str, Any] = {}
    if matched:
        abs_errs = [f["total_error"]["absolute_error_seconds"] for f in matched]
        signed_errs = [f["total_error"]["signed_error_seconds"] for f in matched]
        pcts = [f["total_error"]["absolute_percentage_error"] for f in matched if f["total_error"]["absolute_percentage_error"] is not None]
        n_pct = len(pcts)
        rmse = (sum(e * e for e in abs_errs) / len(abs_errs)) ** 0.5

        def _median(v: list[float]) -> float:
            s = sorted(v)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

        metrics = {
            "sample_count": len(matched),
            "mae_seconds": round(sum(abs_errs) / len(abs_errs), 6),
            "median_absolute_error_seconds": round(_median(abs_errs), 6),
            "rmse_seconds": round(rmse, 6),
            "mean_signed_error_seconds": round(sum(signed_errs) / len(signed_errs), 6),
            "mape_percent": round(sum(pcts) / n_pct, 4) if pcts else None,
            "within_10_percent": round(100.0 * sum(1 for f in matched if f["total_error"]["within_10_percent"]) / len(matched), 2),
            "within_20_percent": round(100.0 * sum(1 for f in matched if f["total_error"]["within_20_percent"]) / len(matched), 2),
            "within_30_percent": round(100.0 * sum(1 for f in matched if f["total_error"]["within_30_percent"]) / len(matched), 2),
            "within_2x": round(100.0 * sum(1 for f in matched if f["total_error"]["within_2x"]) / len(matched), 2),
            # §18 denominators: percentage-based metrics use only pairs where
            # the percentage error is mathematically meaningful (actual > 0);
            # tolerance rates use the full matched sample. Never silently
            # drop unexplained records.
            "denominators": {
                "matched_pairs": len(matched),
                "pairs_with_percentage_error": n_pct,
            },
        }

        # ---- §17 slices (empty slices are omitted, never reported as zeros)
        event_by_id = {e.get("event_id"): e for e in events}

        def _slice_metrics(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not recs:
                return None
            errs = [r["total_error"]["absolute_error_seconds"] for r in recs]
            return {
                "sample_count": len(recs),
                "mae_seconds": round(sum(errs) / len(errs), 6),
                "median_absolute_error_seconds": round(_median(errs), 6),
            }

        # 2026-10-03: the reliability label was removed from the system —
        # error slices are by history bucket / loader / stage only.

        # error by history bucket (EXP-012 bins, reporting-only)
        def _bucket(n) -> str:
            if n is None:
                return "unknown"
            n = int(n)
            if n == 0:
                return "0"
            if n <= 2:
                return "1-2"
            if n <= 4:
                return "3-4"
            if n <= 9:
                return "5-9"
            return "10+"

        by_hist: dict[str, list[dict[str, Any]]] = {}
        for f in matched:
            ev = event_by_id.get(f.get("prediction_event_id")) or {}
            prior = (ev.get("prediction") or {}).get("prior_count")
            by_hist.setdefault(_bucket(prior), []).append(f)
        if by_hist:
            slices["by_history_bucket"] = {k: _slice_metrics(v) for k, v in sorted(by_hist.items())}

        # error by loader (loader filter OFF only; with the filter on this
        # would duplicate the top-level metrics)
        if not loader_filter:
            by_loader: dict[str, list[dict[str, Any]]] = {}
            for f in matched:
                name = str((f.get("identity") or {}).get("loader_name") or "unknown")
                by_loader.setdefault(name, []).append(f)
            if by_loader:
                slices["by_loader"] = {k: _slice_metrics(v) for k, v in sorted(by_loader.items())}

        # stage error aggregates (supplementary diagnostics only)
        stage_agg: dict[str, dict[str, Any]] = {}
        stage_avail: dict[str, int] = {}
        for f in matched:
            ev = event_by_id.get(f.get("prediction_event_id")) or {}
            ev_stages = ((ev.get("stage_intelligence") or {}).get("stages")) or []
            for s in ev_stages:
                st = str(s.get("stage"))
                if s.get("available"):
                    stage_avail[st] = stage_avail.get(st, 0) + 1
            for st, se in (f.get("stage_errors") or {}).items():
                agg = stage_agg.setdefault(st, {"errs": [], "sample_count": 0})
                agg["errs"].append(se["absolute_error_seconds"])
                agg["sample_count"] += 1
        if stage_agg:
            stage_metrics: dict[str, Any] = {}
            for st in sorted(stage_agg):
                agg = stage_agg[st]
                stage_metrics[st] = {
                    "sample_count": agg["sample_count"],
                    "mae_seconds": round(sum(agg["errs"]) / len(agg["errs"]), 6),
                    "median_absolute_error_seconds": round(_median(agg["errs"]), 6),
                    # availability coverage among matched pairs (denominator =
                    # matched pairs whose prediction event carried stage intel)
                    "available_on_matched": stage_avail.get(st, 0),
                    "scored_on_matched": agg["sample_count"],
                }
            slices["stage_metrics"] = stage_metrics
        if slices:
            metrics["slices"] = slices

    eligible_actuals = [a for a in outcomes if a.get("training_eligible")]
    reason_counts: dict[str, int] = {}
    for a in outcomes:
        r = a.get("training_eligibility_reason", "?")
        reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "loader_filter": loader_filter,
        "label": "Operational prediction-vs-actual performance (NOT Task 03 chronological model evaluation)",
        "prediction_attempts": len(events),
        "state_counts": state_counts,
        "actual_outcomes_received": len(outcomes),
        "linkage": {
            "matched": len(matched),
            "unmatched": len(unmatched),
            "ambiguous": len(ambiguous),
            "matched_without_prediction_value": len(
                [f for f in feedback if f.get("linkage", {}).get("status") == LINK_STATUS_MATCHED and not f.get("total_error")]
            ),
        },
        "metrics": metrics,
        "training_feedback": {
            "eligible_actual_count": len(eligible_actuals),
            "not_eligible_count": len(outcomes) - len(eligible_actuals),
            "reason_counts": reason_counts,
        },
    }
