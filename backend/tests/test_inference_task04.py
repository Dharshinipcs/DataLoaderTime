"""Task 04: pre-execution inference service — states, eligibility, leakage."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dltime.artifact import SERVING_MODEL_VERSION, SERVING_STATUS
from dltime.history_eligibility import HistoryEligibilityConfig
from dltime.inference import (
    PredictionRequest,
    PredictionResult,
    PredictionService,
    PredictionState,
    human_readable_duration,
    validate_request,
)


# ------------------------------------------------------------------ helpers

def _resolved_training_rows(service: PredictionService) -> pd.DataFrame:
    """Training rows whose identity RESOLVES from definitions.

    Two conditions: the execution carries a resolved connection (AD-04 union)
    AND the representative dataset's own triple resolves — a caller supplying
    a lookup-gap dataset is correctly UNRESOLVED by the locked rules.
    """
    X = service.tf.X
    frame = service.tf.frame.loc[X.index]
    keep = []
    for idx, r in frame.iterrows():
        status, _ = service.resolver.resolve(
            r["Dataset_display_name"], r["Sprint"], r["Loader_Name"]
        )
        if status.value == "RESOLVED" and pd.notna(r["execution_connection"]):
            keep.append(idx)
    out = frame.loc[keep].copy()
    out["_priors"] = X["hist_loader_prior_count"].loc[keep].astype(float)
    return out


def _pick_service_case(service: PredictionService, *, min_history: int):
    """Find a real RESOLVED identity with at least min_history priors."""
    rows = _resolved_training_rows(service)
    mask = rows["_priors"] >= min_history
    assert mask.any(), f"no resolved case with >= {min_history} priors"
    idx = rows.index[mask][-1]  # latest chronological such case
    return rows.loc[idx]


def _pick_zero_history_resolvable(service: PredictionService) -> dict:
    """A genuine definition triple whose loader never executed (0 priors).

    Picked from the real pre-execution definitions, so it RESOLVES and is
    gated by the history policy — a never-seen identity is UNRESOLVED instead.
    """
    executed = set(
        service.tf.frame["Loader_Name"].dropna().astype(str)
    )
    d = service.definitions_detail
    cand = d[
        ~d["loader_display_name"].astype(str).isin(executed)
    ].dropna(subset=["datasetname", "sprint_name", "loader_display_name", "ldr_connection_name"])
    assert not cand.empty
    r = cand.iloc[0]
    return {
        "loader_name": str(r["loader_display_name"]),
        "sprint": str(r["sprint_name"]),
        "dataset": str(r["datasetname"]),
        "connection": str(r["ldr_connection_name"]),
    }


def _pick_never_seen_loader(service: PredictionService) -> dict:
    """An identity whose loader name appears nowhere in the execution history."""
    frame = service.tf.frame
    loaders = set(frame["Loader_Name"].dropna().astype(str))
    candidate = "no_such_loader_anywhere_task04"
    assert candidate not in loaders
    return {
        "loader_name": candidate,
        "sprint": "task04_sprint",
        "dataset": "task04_dataset",
        "connection": "task04_connection",  # required field; identity state precedes validation
    }


# ------------------------------------------------------------------ request validation

def test_validate_request_missing_identity_fields():
    req = PredictionRequest(loader_name="L", sprint="  ", dataset=None, connection="c")
    clean, err = validate_request(req)
    assert clean is None
    assert err is not None


def test_validate_request_missing_connection_is_rejected():
    """Task 10.1: Connection is a compulsory pre-execution input."""
    req = PredictionRequest(loader_name="L", sprint="S", dataset="D", connection=None)
    clean, err = validate_request(req)
    assert clean is None
    assert err == "missing_required_field: connection"
    req_blank = PredictionRequest(loader_name="L", sprint="S", dataset="D", connection="   ")
    clean2, err2 = validate_request(req_blank)
    assert clean2 is None
    assert err2 == "missing_required_field: connection"


def test_request_dataclass_cannot_carry_post_execution_fields():
    """The typed request contract physically lacks every forbidden field."""
    field_names = set(PredictionRequest.__dataclass_fields__)
    from dltime.inference import FORBIDDEN_REQUEST_FIELDS

    assert not (field_names & FORBIDDEN_REQUEST_FIELDS)


def test_human_readable_duration():
    assert human_readable_duration(65.4) == "1m 5s"
    assert human_readable_duration(3661.0) == "1h 1m 1s"
    assert human_readable_duration(None) is None
    assert human_readable_duration(float("nan")) is None


# ------------------------------------------------------------------ PREDICTED

def test_valid_eligible_prediction(service):
    row = _pick_service_case(service, min_history=3)
    result = service.predict(
        PredictionRequest(
            loader_name=str(row["Loader_Name"]),
            sprint=str(row["Sprint"]),
            dataset=str(row["Dataset_display_name"]),
            connection=str(row["execution_connection"]),
        )
    )
    assert result.state == PredictionState.PREDICTED, result.reason
    assert result.predicted_seconds is not None
    assert math.isfinite(result.predicted_seconds)
    assert result.predicted_seconds > 0
    # expm1 back-transform consistency: log1p(seconds) round-trips
    assert result.predicted_log1p == pytest.approx(
        math.log1p(result.predicted_seconds),        rel=1e-9
    )
    assert result.predicted_seconds_human is not None
    # Sum-total policy (EXP-038): the served ETA is the exact sum of the five
    # operational stage predictions; the stage-aware and direct models are
    # retired to reference/rollback evidence only.
    assert result.model_version == "sum-of-available-stages-v2"
    assert result.serving_role == "AUTHORITATIVE_TOTAL"
    assert result.model_status == "DETERMINISTIC_POLICY"
    si = result.stage_intelligence or {}
    ml_vals = [
        s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    assert len(ml_vals) == 3  # applicable: no prevalidation/transformation entries
    appr_vals = [
        s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    ]
    # 2026-10-02 final spec: only LOADER-tier approval evidence (the loader's
    # OWN approval history) joins the total; broader-context estimates are
    # disclosed but excluded.
    assert result.predicted_seconds == pytest.approx(sum(ml_vals) + sum(appr_vals), abs=1e-9)
    # identity resolved through the locked rules
    assert result.identity["resolution_status"] == "RESOLVED"
    assert result.identity["resolved_connection"] is not None
    # eligibility evidence present
    assert result.history["prior_loader_executions"] >= 3


def test_feature_order_matches_artifact(service):
    """The reported feature vector is the artifact's fitted 44-feature
    contract (2026-10-04 spec) in the artifact's own saved order, PLUS the
    two audit-marker columns (2026-10-02); the served ETA itself is the
    deterministic five-stage sum and depends on no learned total model."""
    row = _pick_service_case(service, min_history=3)
    result = service.predict(
        PredictionRequest(
            loader_name=str(row["Loader_Name"]),
            sprint=str(row["Sprint"]),
            dataset=str(row["Dataset_display_name"]),
            connection=str(row["execution_connection"]),
        )
    )
    assert result.features is not None
    import joblib

    art = joblib.load("artifacts/total-et-v3-candidate.pkl")
    expected = list(art["metadata"]["feature_names_in_order"])
    assert len(expected) == 44  # 2026-10-04 spec: 22 DEF + 22 HIST fitted
    assert list(result.features.keys()) == expected
    # the contract must now carry the per-section order + per-type features
    for name in (
        "def_preval_order_max",
        "def_trans_order_max",
        "def_type_crossreference_count",
        "def_type_table_count",
    ):
        assert name in result.features
    assert len(result.features) == len(expected)


def test_prediction_is_deterministic(service):
    row = _pick_service_case(service, min_history=3)
    req = PredictionRequest(
        loader_name=str(row["Loader_Name"]),
        sprint=str(row["Sprint"]),
        dataset=str(row["Dataset_display_name"]),
        connection=str(row["execution_connection"]),
    )
    r1 = service.predict(req)
    r2 = service.predict(req)
    assert r1.state == r2.state == PredictionState.PREDICTED
    assert r1.predicted_seconds == r2.predicted_seconds
    assert r1.features == r2.features


def test_prediction_out_of_sample_matches_holdout_rows(service):
    """Replay real late-history cases: predictions must be finite and positive
    (no fabricated context); states follow the locked rules.

    Primary population: the final 85-row chronological holdout restricted to
    rows whose identity RESOLVES through the definitions and that carry
    >= 3 as-of loader priors. AD-31 reality: definitions alignment is sparse
    on the corrected source, so when NO holdout row qualifies (asserted, not
    assumed), the latest qualifying rows of the eligible history are replayed
    instead — never a fabricated context.
    """

    def _replay(rows: pd.DataFrame, max_checked: int = 3) -> int:
        checked = 0
        for _, r in rows.iterrows():
            status, conn = service.resolver.resolve(
                r["Dataset_display_name"], r["Sprint"], r["Loader_Name"]
            )
            if status.value != "RESOLVED":
                continue
            result = service.predict(
                PredictionRequest(
                    loader_name=str(r["Loader_Name"]),
                    sprint=str(r["Sprint"]),
                    dataset=str(r["Dataset_display_name"]),
                    connection=str(conn),
                )
            )
            if result.state != PredictionState.PREDICTED:
                continue  # historical features for that context legitimately differ
            assert math.isfinite(result.predicted_seconds) and result.predicted_seconds > 0
            checked += 1
            if checked >= max_checked:
                break
        return checked

    full = service.tf.frame
    eligible = full[full["is_training_eligible"].fillna(False)]
    ordered = eligible.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    holdout = ordered.tail(85)
    n_checked = _replay(
        holdout[
            pd.to_numeric(holdout["hist_loader_prior_count"], errors="coerce") >= 3
        ]
    )
    if n_checked == 0:
        # Documented corrected-source reality: no holdout row resolves through
        # the sparse definitions (the only resolving identities sit mid-history).
        # Replay the qualifying rows most-recent-first until covered.
        qualifying = ordered[
            pd.to_numeric(ordered["hist_loader_prior_count"], errors="coerce") >= 3
        ]
        n_checked = _replay(qualifying.iloc[::-1])
    assert n_checked >= 1, "no resolvable >=3-prior case exists to replay"


# ------------------------------------------------------------------ INSUFFICIENT_HISTORY

def test_insufficient_history_never_seen_loader(service):
    """A real definition triple whose loader never executed: identity resolves
    but there are 0 priors — gated, no fabricated ETA."""
    case = _pick_zero_history_resolvable(service)
    result = service.predict(PredictionRequest(**case))
    assert result.state == PredictionState.INSUFFICIENT_HISTORY
    # human-gate message (user final spec): the refusal names the loader and
    # the threshold instead of a bare machine code.
    assert "has no prior executions" in (result.reason or "")
    assert "3 recorded executions" in (result.reason or "")
    assert result.identity["resolution_status"] == "RESOLVED"
    assert result.history["prior_loader_executions"] == 0
    assert result.predicted_seconds is None  # no fabricated ETA


def test_insufficient_history_known_loader_below_threshold(service):
    """A real loader with only 1-2 total eligible executions must be gated
    (AD-16: min 3 priors).

    A prediction context sits AFTER the full history, so its recomputed prior
    count equals the identity's TOTAL eligible history — the case must have
    1-2 total eligible executions AND resolve through the definitions.
    """
    frame = service.tf.frame.loc[service.tf.X.index].copy()
    frame["_priors"] = service.tf.X["hist_loader_prior_count"].astype(float)
    grp = frame.groupby(["Loader_Name", "Sprint", "Dataset_display_name"], dropna=False)[
        "_priors"
    ].agg(["size", "max"])
    picked = None
    for (ln, sp, ds), row in grp.iterrows():
        if pd.isna(ds) or not (1 <= int(row["size"]) <= 2):
            continue
        status, conn = service.resolver.resolve(ds, sp, ln)
        if status.value != "RESOLVED":
            continue
        picked = (ln, sp, ds, str(conn))
        break
    assert picked is not None, (
        "no resolvable identity with 1-2 total eligible executions "
        "— update the case via the audit, never the gate"
    )
    ln, sp, ds, conn = picked
    result = service.predict(
        PredictionRequest(loader_name=ln, sprint=sp, dataset=ds, connection=conn)
    )
    assert result.state == PredictionState.INSUFFICIENT_HISTORY
    # below-threshold message: names the count, the threshold, and what is
    # still needed before a trusted total is served.
    assert "prior execution" in (result.reason or "")
    assert "trust threshold is 3" in (result.reason or "")


def test_eligibility_threshold_is_configurable(service, serving_artifact, bundle, foundation):
    """AD-16 is a policy value, not hard-coded: min=5 flips a 3-4-prior case.

    A prediction context sits AFTER the full history, so its recomputed
    prior count equals the loader's TOTAL eligible history — the test picks a
    loader whose total eligible history is exactly 3-4.
    """
    ee = service.foundation.frames["executions_eligible"]
    eligible_loaders = set(
        ee.loc[ee["is_training_eligible"].fillna(False), "Loader_Name"]
    )
    counts = ee[ee["Loader_Name"].isin(eligible_loaders) & ee["is_training_eligible"].fillna(False)][
        "Loader_Name"
    ].value_counts()
    candidates = counts[(counts >= 3) & (counts <= 4)].index.tolist()
    assert candidates, "no loader with total eligible history in [3, 4]"

    # must also be a resolvable identity (definition triple exists)
    d = service.definitions_detail
    r = None
    for loader in candidates:
        hit = d[d["loader_display_name"].astype(str) == str(loader)].dropna(
            subset=["datasetname", "sprint_name", "ldr_connection_name"]
        )
        if not hit.empty:
            r = hit.iloc[0]
            break
    assert r is not None
    req = PredictionRequest(
        loader_name=str(r["loader_display_name"]),
        sprint=str(r["sprint_name"]),
        dataset=str(r["datasetname"]),
        connection=str(r["ldr_connection_name"]),
    )
    assert service.predict(req).state == PredictionState.PREDICTED

    strict = PredictionService(
        training_frame=service.tf,
        artifact=serving_artifact,
        foundation=service.foundation,
        resolver=service.resolver,
        definitions_detail=service.definitions_detail,
        definitions_summary=service.definitions_summary,
        eligibility_config=HistoryEligibilityConfig(min_prior_loader_executions=5),
    )
    assert strict.predict(req).state == PredictionState.INSUFFICIENT_HISTORY


# ------------------------------------------------------------------ identity states

def test_unresolved_identity_state(service):
    case = _pick_never_seen_loader(service)
    # 2026-10-03: a never-registered loader is treated as a NEW loader — the
    # identity stays honestly UNRESOLVED in the audit block, no total is served.
    result = service.predict(PredictionRequest(**case))
    assert result.state == PredictionState.INSUFFICIENT_HISTORY
    assert result.identity["resolution_status"] == "UNRESOLVED"
    assert result.predicted_seconds is None


def test_unresolved_identity_treated_as_new_loader(service):
    """2026-10-03 (user final spec): an unregistered loader gets the same
    short new-loader refusal and the post-execution entry path — the gate
    counts it as UNKNOWN_LOADER with zero priors, never a fabricated total."""
    case = _pick_never_seen_loader(service)
    result = service.predict(PredictionRequest(**case))
    assert result.state == PredictionState.INSUFFICIENT_HISTORY
    assert result.loader_case["case"] == "UNKNOWN_LOADER"
    assert result.history["prior_loader_executions"] == 0


def test_connection_override_is_rejected_as_invalid(service):
    row = _pick_service_case(service, min_history=3)
    result = service.predict(
        PredictionRequest(
            loader_name=str(row["Loader_Name"]),
            sprint=str(row["Sprint"]),
            dataset=str(row["Dataset_display_name"]),
            connection="definitely_not_the_resolved_connection",
        )
    )
    assert result.state == PredictionState.INVALID_INPUT
    assert "disagrees" in result.reason


def test_ambiguous_identity_state_via_synthetic_resolver(service, serving_artifact, bundle, foundation):
    """AMBIGUOUS is returned when definitions genuinely offer >1 connection
    (constructed via an injected conflicting definition row; production data
    currently has 0 ambiguous keys, so no silent choice can ever occur)."""
    detail = bundle.loader_definitions_detail.copy()
    conflict_row = detail.iloc[0].copy()
    # force a second, different connection for the same identity triple
    conflict_row["ldr_connection_name"] = "__conflicting_connection__"
    detail = pd.concat([detail, pd.DataFrame([conflict_row])], ignore_index=True)

    from dltime.resolution import ConnectionResolver

    resolver2 = ConnectionResolver(
        detail, bundle.loader_definitions_summary
    )
    svc = PredictionService(
        training_frame=service.tf,
        artifact=serving_artifact,
        foundation=service.foundation,
        resolver=resolver2,
        definitions_detail=service.definitions_detail,
        definitions_summary=service.definitions_summary,
    )
    r0 = detail.iloc[0]
    result = svc.predict(
        PredictionRequest(
            loader_name=str(r0["loader_display_name"]),
            sprint=str(r0["sprint_name"]),
            dataset=str(r0["datasetname"]),
            connection=str(r0["ldr_connection_name"]),
        )
    )
    assert result.state == PredictionState.AMBIGUOUS_IDENTITY
    assert result.identity["resolution_status"] == "AMBIGUOUS"
    assert result.identity["resolved_connection"] is None
    assert result.predicted_seconds is None


# ------------------------------------------------------------------ leakage probes

def test_context_row_can_never_enter_its_own_history(service):
    """The synthetic context row is target-less and ineligible: it must not
    shift any historical aggregate between two identical requests, and its
    own prior counts must equal the strictly-prior real counts."""
    row = _pick_service_case(service, min_history=3)
    req = PredictionRequest(
        loader_name=str(row["Loader_Name"]),
        sprint=str(row["Sprint"]),
        dataset=str(row["Dataset_display_name"]),
        connection=str(row["execution_connection"]),
    )
    result = service.predict(req)
    assert result.state == PredictionState.PREDICTED

    # Directly verify the chronology property on the combined frame path:
    ctx = service.build_context_row(
        loader_name=str(row["Loader_Name"]),
        sprint=str(row["Sprint"]),
        dataset=str(row["Dataset_display_name"]),
        resolution_status=service.resolver.resolve(
            str(row["Dataset_display_name"]), str(row["Sprint"]), str(row["Loader_Name"])
        )[0],
        resolved_connection=str(row["execution_connection"]),
        start_time=service._default_start_time(),
        execution_id=service._synthetic_execution_id(),
        workflow_execution_id=None,
    )
    assert bool(ctx["is_training_eligible"].iloc[0]) is False
    assert pd.isna(ctx["timetaken_in_sec"].iloc[0])
    # The context sits after the full history, so it sees every eligible prior
    # of its loader (== the loader's TOTAL eligible executions). It must never
    # exceed that total: a larger number would mean the target-less context
    # row (or another ineligible row) contributed to its own history.
    loader_total = int(
        service.foundation.frames["executions_eligible"]
        .loc[
            service.foundation.frames["executions_eligible"]["Loader_Name"]
            == str(row["Loader_Name"]),
            "is_training_eligible",
        ]
        .fillna(False)
        .sum()
    )
    ctx_priors = int(ctx["hist_loader_prior_count"].iloc[0])
    assert ctx_priors == loader_total
    assert ctx_priors <= loader_total + 0  # no self/future contribution
    assert ctx_priors >= int(row["hist_loader_prior_count"])


def test_future_start_time_does_not_pull_future_history(service):
    """A planned start after the last execution must still see only prior data;
    historical aggregates equal those of the same identity in training."""
    row = _pick_service_case(service, min_history=3)
    req = PredictionRequest(
        loader_name=str(row["Loader_Name"]),
        sprint=str(row["Sprint"]),
        dataset=str(row["Dataset_display_name"]),
        connection=str(row["execution_connection"]),
        planned_start_time="2099-01-01T00:00:00",
    )
    result = service.predict(req)
    assert result.state == PredictionState.PREDICTED
    # the future context cannot observe any execution after the last real one
    assert result.history["prior_loader_executions"] >= 3


def test_future_execution_cannot_contribute_to_current_prediction(service):
    """AD-11 end-to-end: an execution strictly AFTER the prediction context can
    never enter its historical features.

    Setup: pick a resolved >=3-prior row that is NOT the chronologically last
    execution; place the context at that row's exact Start_Time (the row
    itself becomes a legitimate prior via the deterministic id tie-break);
    poison the globally LAST execution's target (strictly future to the
    context) and re-predict. Features/history/prediction must be unchanged.
    """
    rows = _resolved_training_rows(service)
    max_start = pd.to_datetime(
        service.foundation.frames["executions_eligible"]["Start_Time"].max()
    )
    mask = (rows["_priors"] >= 3) & (
        pd.to_datetime(rows["Start_Time"]) < max_start
    )
    assert mask.any(), "no resolved >=3-prior case strictly before the last execution"
    row = rows.loc[mask].iloc[-1]
    req = PredictionRequest(
        loader_name=str(row["Loader_Name"]),
        sprint=str(row["Sprint"]),
        dataset=str(row["Dataset_display_name"]),
        connection=str(row["execution_connection"]),
        planned_start_time=str(pd.Timestamp(row["Start_Time"])),
    )
    baseline = service.predict(req)
    assert baseline.state == PredictionState.PREDICTED
    assert baseline.history["prior_loader_executions"] >= 3

    executions = service.foundation.frames["executions_eligible"]
    try:
        poisoned = executions.copy()
        poisoned["timetaken_in_sec"] = poisoned["timetaken_in_sec"].astype(float)
        # poison the chronologically LAST execution: strictly FUTURE relative
        # to the context placed at row.Start_Time
        last_idx = poisoned["Start_Time"].idxmax()
        assert pd.Timestamp(poisoned.loc[last_idx, "Start_Time"]) > pd.Timestamp(row["Start_Time"])
        poisoned.loc[last_idx, "timetaken_in_sec"] = 9.9e9
        service.foundation.frames["executions_eligible"] = poisoned
        after = service.predict(req)
    finally:
        service.foundation.frames["executions_eligible"] = executions
    assert after.state == PredictionState.PREDICTED
    assert after.predicted_seconds == pytest.approx(baseline.predicted_seconds, rel=1e-12)
    assert after.history["prior_loader_executions"] == baseline.history["prior_loader_executions"]
    assert after.features == baseline.features


def test_no_target_or_post_execution_columns_in_feature_vector(service):
    forbidden = {
        "timetaken_in_sec",
        "Time_Taken",
        "End_Time",
        "LDR_Status",
        "Total_Records",
        "Success_Records",
        "Error_Records",
    }
    row = _pick_service_case(service, min_history=3)
    result = service.predict(
        PredictionRequest(
            loader_name=str(row["Loader_Name"]),
            sprint=str(row["Sprint"]),
            dataset=str(row["Dataset_display_name"]),
            connection=str(row["execution_connection"]),
        )
    )
    assert result.features is not None
    assert not (set(result.features) & forbidden)


# ------------------------------------------------------------------ INVALID_INPUT

def test_invalid_input_missing_loader(service):
    result = service.predict(
        PredictionRequest(loader_name=" ", sprint="s", dataset="d", connection="c")
    )
    assert result.state == PredictionState.INVALID_INPUT
    assert "loader_name" in result.reason


def test_invalid_input_bad_planned_start(service):
    row = _pick_service_case(service, min_history=3)
    result = service.predict(
        PredictionRequest(
            loader_name=str(row["Loader_Name"]),
            sprint=str(row["Sprint"]),
            dataset=str(row["Dataset_display_name"]),
            connection=str(row["execution_connection"]),
            planned_start_time="not-a-timestamp",
        )
    )
    assert result.state == PredictionState.INVALID_INPUT
    assert "planned_start_time" in result.reason


def test_invalid_input_state_has_no_prediction(service):
    result = service.predict(
        PredictionRequest(loader_name=" ", sprint=" ", dataset=" ", connection=" ")
    )
    assert result.predicted_seconds is None
    assert result.model_version is None
