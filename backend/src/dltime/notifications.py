"""System-level notification / alert center (AD-33, §21-§24).

A persisted notification feed derived EXCLUSIVELY from signals that already
exist in the architecture. This module invents no new detection logic: every
alert type wraps an existing subsystem verdict —

    prediction risk labels            (dltime.risk)
    prediction event states           (Task 11 store records)
    drift monitoring statuses         (model_lab.feature/population_drift)
    retraining readiness state        (model_lab.retraining_readiness)
    model registry lifecycle          (model_lab.ModelRegistry)
    feedback data-quality warnings    (Task 11 actual-outcome ingestion)
    stage serving lifecycle labels    (stage_models / stage_serving)

Design rules:
- NOTIFICATIONS ARE OBSERVATIONS, NOT ACTIONS: nothing here retrains,
  promotes, or mutates any model. The feed is read/write only in its own
  read/unread bookkeeping.
- DEDUPLICATED: one open notification per ``(alert_type, dedupe_key)``. A
  notification the user has read does not suppress future occurrences — an
  unresolved condition deserves renewed attention — but identical open
  notifications are never duplicated.
- PERSISTED: append-mostly JSONL under ``backend/store`` via MonitoringStore;
  marking read rewrites the file (the only rewrite, and it never drops
  records).
- HONEST: severities, messages, and context come from the observed records;
  no synthetic alerts are generated to populate the UI.
"""

from __future__ import annotations

from typing import Any

from dltime import stage_models
from dltime.store import MonitoringStore, content_hash, utc_now_iso

# --------------------------------------------------------------- taxonomy


class AlertType:
    """Closed alert taxonomy (AD-33 §22). Values are stable API strings."""

    NEW_LOADER = "NEW_LOADER"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    PREDICTION_ACCURACY_DEGRADED = "PREDICTION_ACCURACY_DEGRADED"
    MODEL_DRIFT = "MODEL_DRIFT"
    RETRAINING_RECOMMENDED = "RETRAINING_RECOMMENDED"
    NEW_MODEL_CANDIDATE = "NEW_MODEL_CANDIDATE"
    TRAINING_DATA_QUALITY = "TRAINING_DATA_QUALITY"
    STAGE_DATA_COVERAGE = "STAGE_DATA_COVERAGE"
    MODEL_STALE = "MODEL_STALE"
    UNUSUAL_LOADER_EXECUTION_PATTERN = "UNUSUAL_LOADER_EXECUTION_PATTERN"


ALL_ALERT_TYPES: tuple[str, ...] = (
    AlertType.NEW_LOADER,
    AlertType.INSUFFICIENT_HISTORY,
    AlertType.OUT_OF_DISTRIBUTION,
    AlertType.PREDICTION_ACCURACY_DEGRADED,
    AlertType.MODEL_DRIFT,
    AlertType.RETRAINING_RECOMMENDED,
    AlertType.NEW_MODEL_CANDIDATE,
    AlertType.TRAINING_DATA_QUALITY,
    AlertType.STAGE_DATA_COVERAGE,
    AlertType.MODEL_STALE,
    AlertType.UNUSUAL_LOADER_EXECUTION_PATTERN,
)


class Severity:
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Severity per alert type (color is only a supporting signal in the UI;
# severity label + icon are always rendered textually).
SEVERITY_FOR_TYPE: dict[str, str] = {
    AlertType.NEW_LOADER: Severity.INFO,
    AlertType.INSUFFICIENT_HISTORY: Severity.WARNING,
    AlertType.OUT_OF_DISTRIBUTION: Severity.HIGH,
    AlertType.PREDICTION_ACCURACY_DEGRADED: Severity.WARNING,
    AlertType.MODEL_DRIFT: Severity.HIGH,
    AlertType.RETRAINING_RECOMMENDED: Severity.WARNING,
    AlertType.NEW_MODEL_CANDIDATE: Severity.INFO,
    AlertType.TRAINING_DATA_QUALITY: Severity.WARNING,
    AlertType.STAGE_DATA_COVERAGE: Severity.WARNING,
    AlertType.MODEL_STALE: Severity.INFO,
    AlertType.UNUSUAL_LOADER_EXECUTION_PATTERN: Severity.WARNING,
}

# Human-readable titles per alert type (the message carries the specifics).
TITLE_FOR_TYPE: dict[str, str] = {
    AlertType.NEW_LOADER: "New loader detected",
    AlertType.INSUFFICIENT_HISTORY: "Insufficient loader history",
    AlertType.OUT_OF_DISTRIBUTION: "Out-of-distribution prediction",
    AlertType.PREDICTION_ACCURACY_DEGRADED: "Prediction accuracy degraded",
    AlertType.MODEL_DRIFT: "Model drift detected",
    AlertType.RETRAINING_RECOMMENDED: "Retraining recommended",
    AlertType.NEW_MODEL_CANDIDATE: "New model candidate available",
    AlertType.TRAINING_DATA_QUALITY: "Training data quality problem",
    AlertType.STAGE_DATA_COVERAGE: "Stage data coverage problem",
    AlertType.MODEL_STALE: "Model training snapshot is stale",
    AlertType.UNUSUAL_LOADER_EXECUTION_PATTERN: "Unusual loader execution pattern",
}


def _notification_id(dedupe_key: str, created_at: str) -> str:
    return f"n_{content_hash({'k': dedupe_key, 't': created_at})}"


def _record(
    *, alert_type: str, dedupe_key: str, message: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    created = utc_now_iso()
    from dltime.store import SCHEMA_VERSION

    return {
        # store schema gate (uniform across all store files)
        "schema_version": SCHEMA_VERSION,
        "notification_id": _notification_id(dedupe_key, created),
        "notification_key": dedupe_key,
        "alert_type": alert_type,
        "severity": SEVERITY_FOR_TYPE[alert_type],
        "title": TITLE_FOR_TYPE[alert_type],
        "message": message,
        "context": context or {},
        "created_at": created,
        "read": False,
        "read_at": None,
    }


# ------------------------------------------------- event-derived alerts (cheap)


def alerts_from_prediction_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed-scenario policy (2026-10-05): prediction events generate NO alerts.

    The Alert Center carries ONLY the three fixed scenarios — retraining
    needed, stage-model promotion suggestion, drift detected. Per-prediction
    signals (new loader, insufficient history, risk labels) are never
    persisted, so the bell cannot flood.
    """
    return []


def alerts_from_actual_outcome(
    outcome: dict[str, Any], *, eligible_outcome_count: int | None = None
) -> list[dict[str, Any]]:
    """Fixed-scenario policy (2026-10-05): actual outcomes generate NO alerts.

    Data-quality and model-stale signals stay visible on the dashboards
    (Actual Outcomes, Retraining Center); they do not enter the bell.
    """
    return []


# ------------------------------------------- system-level evaluation (on demand)


def _served_model_version() -> str:
    """The authoritative served model version (artifact metadata block)."""
    try:
        from dltime.stage_aware_serving import load_stage_aware_artifact

        artifact = load_stage_aware_artifact()
        version = artifact.get("metadata", {}).get(
            "model_version", artifact.get("model_version")
        )
        return str(version or "unknown")
    except Exception:
        return "unknown"


def _stage_coverage_alerts() -> list[dict[str, Any]]:
    """STAGE_DATA_COVERAGE from the stage serving lifecycle labels.

    Uses the existing honest lifecycle states (AD-31a small-sample policy):
    any stage whose served status discloses limited evidence raises one
    WARNING per stage. No new threshold is introduced.
    """
    out: list[dict[str, Any]] = []
    for stage, spec in stage_models.STAGE_MODEL_SERVING_KINDS.items():
        status = str(spec.get("status") or "")
        if "SMALL_SAMPLE" not in status and status != "EXPERIMENTAL":
            continue
        train_n = spec.get("train_n")
        out.append(
            _record(
                alert_type=AlertType.STAGE_DATA_COVERAGE,
                dedupe_key=f"{AlertType.STAGE_DATA_COVERAGE}:{stage}",
                message=(
                    f"Stage '{stage}' is served with lifecycle status {status}"
                    + (f" (training n={train_n})" if train_n is not None else "")
                    + ". Stage estimates for this stage carry limited evidence and "
                    "are disclosed as such in every prediction."
                ),
                context={"stage": stage, "lifecycle_status": status, "train_n": train_n},
            )
        )
    return out


def _registry_alerts(registry: Any | None) -> list[dict[str, Any]]:
    """NEW_MODEL_CANDIDATE from registry CANDIDATE_EVALUATED records."""
    if registry is None:
        return []
    out: list[dict[str, Any]] = []
    try:
        candidates = [
            rec
            for rec in registry.list()
            if rec.get("status") == "CANDIDATE_EVALUATED"
        ]
    except Exception:
        return out
    for rec in candidates:
        model_id = rec.get("model_id")
        out.append(
            _record(
                alert_type=AlertType.NEW_MODEL_CANDIDATE,
                dedupe_key=f"{AlertType.NEW_MODEL_CANDIDATE}:{model_id}",
                message=(
                    f"Model '{model_id}' is registered as CANDIDATE_EVALUATED and "
                    "awaits an explicit human promotion decision in the Model "
                    "Registry. Nothing is promoted automatically."
                ),
                context={"model_id": model_id, "model_family": rec.get("model_family")},
            )
        )
    return out


def evaluate_system_notifications(
    *,
    drift_feature_status: str | None = None,
    drift_population_status: str | None = None,
    readiness: dict[str, Any] | None = None,
    performance_degraded: bool | None = None,
    registry: Any | None = None,
) -> list[dict[str, Any]]:
    """System-level alerts from EXISTING monitoring verdicts (§24).

    Only the subsystems whose verdicts are passed in are assessed; callers
    reuse the same drift/readiness computations the dashboards already run,
    so no thresholds are redefined here.
    """
    out: list[dict[str, Any]] = []
    model_version = _served_model_version()

    for kind, status in (
        ("feature", drift_feature_status),
        ("population", drift_population_status),
    ):
        if status == "SIGNIFICANT_DRIFT":
            out.append(
                _record(
                    alert_type=AlertType.MODEL_DRIFT,
                    dedupe_key=f"{AlertType.MODEL_DRIFT}:{kind}",
                    message=(
                        f"{kind.capitalize()}-level drift monitoring reported "
                        "SIGNIFICANT_DRIFT for the current window. Inspect the "
                        "Drift Monitoring view; no automatic action was taken."
                    ),
                    context={"drift_kind": kind, "status": status, "model_version": model_version},
                )
            )

    if readiness:
        state = readiness.get("readiness_state")
        signals = readiness.get("signals") or {}
        if state == "RETRAINING_RECOMMENDED_FOR_REVIEW":
            out.append(
                _record(
                    alert_type=AlertType.RETRAINING_RECOMMENDED,
                    dedupe_key=AlertType.RETRAINING_RECOMMENDED,
                    message=(
                        "Retraining readiness reached RETRAINING_RECOMMENDED_FOR_REVIEW "
                        "from existing signals (drift/performance). Review the "
                        "Retraining Center; retraining requires explicit human approval."
                    ),
                    context={"readiness_state": state, "model_version": model_version},
                )
            )
    # 2026-10-05 fixed-scenario policy: MODEL_STALE and
    # PREDICTION_ACCURACY_DEGRADED no longer generate notifications. The
    # Alert Center carries ONLY: drift detected (above), retraining needed
    # (above), and the stage-model promotion suggestion (registry candidates
    # below). All other signals stay on their dashboards.
    out.extend(_registry_alerts(registry))
    return out


# ---------------------------------------------------------------- persistence


def append_notifications(store: MonitoringStore, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append only records whose (type, key) has no currently-open notification."""
    existing, _warnings = store.read_notifications()
    open_keys = {rec.get("notification_key") for rec in existing if not rec.get("read")}
    fresh = [rec for rec in records if rec.get("notification_key") not in open_keys]
    for rec in fresh:
        store.append_notification(rec)
    return fresh


def list_notifications(store: MonitoringStore) -> dict[str, Any]:
    """Active feed, newest first, with the unread count for the bell badge."""
    records, warnings = store.read_notifications()
    ordered = sorted(records, key=lambda r: str(r.get("created_at")), reverse=True)
    return {
        "notifications": ordered,
        "unread_count": sum(1 for r in ordered if not r.get("read")),
        "read_warnings": [w.message for w in warnings],
    }


def mark_notification_read(store: MonitoringStore, notification_id: str) -> dict[str, Any]:
    records, _warnings = store.read_notifications()
    hit = False
    for rec in records:
        if rec.get("notification_id") == notification_id and not rec.get("read"):
            rec["read"] = True
            rec["read_at"] = utc_now_iso()
            hit = True
    if hit:
        store.rewrite_notifications(records)
    return {"marked": hit, **list_notifications(store)}


def mark_all_notifications_read(store: MonitoringStore) -> dict[str, Any]:
    records, _warnings = store.read_notifications()
    changed = False
    now = utc_now_iso()
    for rec in records:
        if not rec.get("read"):
            rec["read"] = True
            rec["read_at"] = now
            changed = True
    if changed:
        store.rewrite_notifications(records)
    return {"marked": changed, **list_notifications(store)}
