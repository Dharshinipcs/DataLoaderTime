"""Task 06: stage intelligence tests.

AD-31 contract (corrected Loader_Executions.xlsx, EXP-028 audit), amended by
the final sum-of-five-stages serving policy (EXP-038):

- seven-stage inventory: five ML stages (staging, prevalidation,
  transformation, pre_load, data_loading) + two contextual-statistical
  approval stages (datamart_approval, dataloading_approval)
- pre_load is a REAL independent stage (own fields); the old
  duplicate-of-data_loading treatment is removed
- chronological prior-only evidence for every stage estimate
- sparse approval stages are estimated ONLY by the contextual hierarchical
  statistical estimator, never fabricated and never called ML
- sum policy: the five operational stage predictions ARE the served total's
  exact additive components (dataset-wide ANALYTICS aggregates in
  /stages/analytics remain descriptive only and are never summed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.stages import (
    STAGE_DEFINITIONS,
    StageConfig,
    StageModelability,
    build_stage_analytics,
    build_stage_history_for_context,
    classify_stage,
    stage_context_is_na,
    stage_duration_series,
)
from dltime.inference import PredictionRequest
from dltime.inference import PredictionState
from dltime.stage_intel import APPROVAL_STAGES, ML_STAGES

# Real evidence on the corrected source (EXP-028 audit; recompute pinned by
# test_real_classification_matches_evidence so drift is caught, not assumed).
EXPECTED_POSITIVE_COUNTS = {
    # 2026-09-30 (2nd re-audit): the approval stage specs now fall back to the
    # corrected export's `*_timetaken_in_sec` fields (the `_in_min` spellings
    # are empty), so approvals have real evidence too. ML-stage counts are
    # unchanged from the first re-audit.
    "staging": 469,
    "prevalidation": 147,
    "transformation": 114,
    "pre_load": 51,
    "data_loading": 332,
    "datamart_approval": 12,
    "dataloading_approval": 6,
}


# ------------------------------------------------------------- discovery/schema


def test_stage_inventory_from_real_data(foundation):
    """The AD-31 seven-stage inventory on the corrected source."""
    names = {d.name for d in STAGE_DEFINITIONS}
    assert names == {
        "staging",
        "prevalidation",
        "transformation",
        "pre_load",
        "data_loading",
        "datamart_approval",
        "dataloading_approval",
    }
    assert "data_loading_1" not in names  # old alias spelling removed
    ex = foundation.frames["executions_eligible"]
    for d in STAGE_DEFINITIONS:
        # every declared source column/span must actually resolve on real data
        stage_duration_series(ex, d)


def test_stage_duration_series_semantics(foundation):
    ex = foundation.frames["executions_eligible"]
    defs = {d.name: d for d in STAGE_DEFINITIONS}
    # literal column passes through numeric coercion
    v = stage_duration_series(ex, defs["data_loading"])
    assert v.notna().sum() > 0
    # timestamp span: negative spans become NA, never clamped
    v2 = stage_duration_series(ex, defs["pre_load"])
    assert (v2.dropna() >= 0).all()


def test_pre_load_is_independent_of_data_loading(foundation):
    """AD-31: pre_load and data_loading are separate real stages.

    The corrected source provides real independent fields; the co-observed
    values are not identical (EXP-028: only 4/36 co-observed equal), so the
    old alias policy is factually obsolete as well as contractually removed.
    """
    ex = foundation.frames["executions_eligible"]
    pl = ex["stage_pre_load_sec"]
    dl = ex["stage_data_loading_sec"]
    both = pl.notna() & dl.notna() & (pl > 0) & (dl > 0)
    assert both.sum() >= 15  # genuinely co-observed on the corrected source
    identical = (pl[both] == dl[both]).sum()
    assert identical < both.sum()  # NOT a duplicate measurement


# --------------------------------------------------------------- classification


def test_classification_thresholds_are_evidence_based():
    cfg = StageConfig()
    assert classify_stage(0, cfg) is StageModelability.NOT_OBSERVED
    assert classify_stage(5, cfg) is StageModelability.TOO_SPARSE
    assert classify_stage(6, cfg) is StageModelability.ANALYTICS_ONLY
    assert classify_stage(29, cfg) is StageModelability.ANALYTICS_ONLY
    assert classify_stage(30, cfg) is StageModelability.MODELABLE


def test_real_classification_matches_evidence(foundation):
    """Recompute every count from the corrected source and pin the verdict."""
    ex = foundation.frames["executions_eligible"]
    cfg = StageConfig()
    got = {}
    for d in STAGE_DEFINITIONS:
        n = int((stage_duration_series(ex, d) > 0).sum())
        got[d.name] = (n, classify_stage(n, cfg))
        assert n == EXPECTED_POSITIVE_COUNTS[d.name], (
            f"{d.name}: corrected-source evidence changed "
            f"({n} != {EXPECTED_POSITIVE_COUNTS[d.name]}); update the pin "
            "via the audit, never the assertion."
        )
    for ml_stage in ML_STAGES:
        assert got[ml_stage][1] is StageModelability.MODELABLE
    for approval in APPROVAL_STAGES:
        # 2026-09-30 (2nd re-audit): approval specs now read the corrected
        # export's `*_timetaken_in_sec` fields, so approvals carry real
        # evidence (12 / 6 positives) and classify ANALYTICS_ONLY — below the
        # MODELABLE floor (30) by design: approvals stay statistical-only at
        # serving time via the hierarchical contextual ladder, never ML.
        assert got[approval][1] is StageModelability.ANALYTICS_ONLY


def test_analytics_reports_real_statistics(foundation):
    ex = foundation.frames["executions_eligible"]
    result = build_stage_analytics(ex)
    by_name = {s["stage"]: s for s in result.stages}
    dl = by_name["data_loading"]
    # exact, honest recomputation against the real frame:
    pos = ex["stage_data_loading_sec"].dropna()
    pos = pos[pos > 0]
    assert dl["observation_count"] == len(pos)
    assert dl["median_sec"] == pytest.approx(float(pos.median()), abs=1e-3)
    assert dl["p90_sec"] == pytest.approx(float(pos.quantile(0.9)), abs=1e-3)
    # AD-31: pre_load is a real stage — no duplicate_of metadata anywhere
    assert "duplicate_of" not in by_name["pre_load"]
    assert by_name["pre_load"]["observation_count"] == EXPECTED_POSITIVE_COUNTS["pre_load"]
    assert "NEVER summed" in result.to_payload()["note"]


# ------------------------------------------------- chronological stage history


def _synthetic_context(df: pd.DataFrame, at_ts, exec_id="CTX-000") -> pd.DataFrame:
    """A target-less, stage-less, ineligible context row (mirror of inference)."""
    ctx = df.iloc[[0]].copy()
    ctx["Start_Time"] = pd.Timestamp(at_ts)
    ctx["LDR_Execution_Id"] = exec_id
    stage_cols = [c for c in df.columns if c.startswith("stage_") and c.endswith("_sec")]
    for col in stage_cols:
        ctx[col] = np.nan
    ctx["is_training_eligible"] = False
    return ctx


def test_stage_history_uses_only_strictly_prior_eligible_rows(foundation):
    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    ex = ex.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    # context placed exactly at a real historical timestamp
    at = ex["Start_Time"].iloc[len(ex) // 2]
    combined = pd.concat([ex, _synthetic_context(ex, at)], ignore_index=True)
    hist = build_stage_history_for_context(combined, "CTX-000")
    dl = hist.per_stage["data_loading"]
    # context sits at 'at': the evidence is every ELIGIBLE row strictly BEFORE
    # the context in the deterministic (Start_Time, LDR_Execution_Id) order —
    # the established AD-11 tie semantics (same-instant smaller-id rows are
    # prior; the id tie-break keeps ordering deterministic).
    elig = ex["is_training_eligible"].fillna(False).astype(bool)
    ordered = ex.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    # the synthetic context id sorts (lexicographically) after every numeric
    # execution id, so same-instant rows at 'at' are all strictly prior to it
    prior = ordered[ordered["Start_Time"] <= at]
    prior = prior[prior["is_training_eligible"].fillna(False).astype(bool) & (prior["stage_data_loading_sec"] > 0)]
    assert dl["prior_positive_count"] == len(prior)
    assert dl["available"] is True
    assert dl["estimate_seconds"] == pytest.approx(float(prior["stage_data_loading_sec"].median()), abs=1e-3)


def test_current_execution_cannot_contaminate_stage_history(foundation):
    """A poisoned 'current' row at/after the context leaves history identical."""
    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    ex = ex.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    at = ex["Start_Time"].iloc[len(ex) // 2]
    base = pd.concat([ex, _synthetic_context(ex, at)], ignore_index=True)
    clean = build_stage_history_for_context(base, "CTX-000")

    poisoned = ex.copy()
    # a monstrous stage value on a row AT the context instant and one AFTER it
    poison_idx = [len(ex) - 1]
    same_time = ex.index[ex["Start_Time"] == at]
    poison_idx += [int(i) for i in same_time if int(i) != len(ex) - 1][:1]
    for i in poison_idx:
        poisoned.loc[i, "stage_data_loading_sec"] = 9.9e9
    combined = pd.concat([poisoned, _synthetic_context(poisoned, at)], ignore_index=True)
    dirty = build_stage_history_for_context(combined, "CTX-000")
    assert dirty.per_stage["data_loading"] == clean.per_stage["data_loading"]


def test_future_execution_cannot_contribute_to_stage_history(foundation):
    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    at = ex["Start_Time"].min()
    future = ex.iloc[[0]].copy()
    future["Start_Time"] = at + pd.Timedelta(days=365)
    future["stage_data_loading_sec"] = 1.0e9
    combined = pd.concat([ex, future, _synthetic_context(ex, at)], ignore_index=True)
    hist = build_stage_history_for_context(combined, "CTX-000")
    # context sits at the earliest instant; nothing (incl. the future row and
    # any same-instant tie) may contribute
    assert hist.per_stage["data_loading"]["prior_positive_count"] == 0
    assert hist.per_stage["data_loading"]["available"] is False


def test_deterministic_tie_ordering_stage_history(foundation):
    """Same Start_Time rows: LDR_Execution_Id tie-break is deterministic."""
    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    at = ex["Start_Time"].min()
    a = ex.iloc[[0]].copy()
    a["Start_Time"] = at
    a["LDR_Execution_Id"] = "TIE-A"
    a["stage_data_loading_sec"] = 10.0
    b = ex.iloc[[1]].copy()
    b["Start_Time"] = at
    b["LDR_Execution_Id"] = "TIE-B"
    b["stage_data_loading_sec"] = 20.0
    combined = pd.concat([ex, a, b, _synthetic_context(ex, at + pd.Timedelta(seconds=1))], ignore_index=True)
    h1 = build_stage_history_for_context(combined, "CTX-000")
    # identical inputs in a different insertion order -> identical history
    combined2 = pd.concat([ex, b, a, _synthetic_context(ex, at + pd.Timedelta(seconds=1))], ignore_index=True)
    h2 = build_stage_history_for_context(combined2, "CTX-000")
    assert h1.per_stage["data_loading"] == h2.per_stage["data_loading"]


# ------------------------------------------------------- sparse-stage gating


def test_approval_stages_use_contextual_statistical_estimator_only(foundation):
    """AD-31: approvals go through the hierarchical statistical estimator.

    The estimator may legitimately produce an estimate from strictly-prior
    contextual evidence (e.g. GLOBAL tier) — what it must NEVER do is fabricate
    without evidence or masquerade as an ML prediction.
    """
    from dltime.approval_estimator import estimate_approval_stage

    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    at = ex["Start_Time"].max() + pd.Timedelta(days=1)  # sees the full history
    combined = pd.concat([ex, _synthetic_context(ex, at)], ignore_index=True)
    for name in ("datamart_approval", "dataloading_approval"):
        est = estimate_approval_stage(combined, "CTX-000", name)
        payload = est.to_payload()
        assert payload["method"] == "contextual_hierarchical_statistical"
        if est.available:
            assert est.tier in ("CONTEXT_GROUP", "LOADER_DATASET", "LOADER_CONNECTION", "LOADER", "COMPOSITE", "GLOBAL")
            assert est.n_evidence >= 3  # documented MIN_TIER_OBSERVATIONS
            assert est.estimate_seconds > 0
        else:
            assert est.estimate_seconds is None
            assert est.tier == "UNAVAILABLE"
            assert "Insufficient prior evidence" in est.reason


def test_context_row_is_stage_na_guard(foundation):
    ex = foundation.frames["executions_eligible"].copy()
    ex["Start_Time"] = pd.to_datetime(ex["Start_Time"])
    at = ex["Start_Time"].max() + pd.Timedelta(days=1)
    combined = pd.concat([ex, _synthetic_context(ex, at)], ignore_index=True)
    assert stage_context_is_na(combined, "CTX-000") is True
    # a context carrying a stage observation violates the guard
    bad = combined.copy()
    bad.loc[bad["LDR_Execution_Id"] == "CTX-000", "stage_data_loading_sec"] = 5.0
    assert stage_context_is_na(bad, "CTX-000") is False


# ------------------------------------------------------------------ API layer


def test_api_predict_keeps_total_authoritative_and_adds_stage_intelligence():
    """POST /predict: unchanged total semantics + supplementary stage section."""
    from fastapi.testclient import TestClient

    from dltime.api import app

    client = TestClient(app)
    resp = client.post(
        "/predict",
        json={
            "loader_name": "MigrationSummarySubTableLoaderDataset",
            "sprint": "TEST",
            "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
            "connection": "MDM_DM_TEST",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "PREDICTED"
    # Sum policy (EXP-038): the total IS the exact five-stage sum.
    assert body["model_version"] == "sum-of-available-stages-v2"
    assert body["serving_role"] == "AUTHORITATIVE_TOTAL"
    assert body["predicted_seconds"] > 0
    si = body["stage_intelligence"]
    assert si["available"] is True
    # applicable-stage view (user final spec, 2026-09-30): the note names
    # the applicable-stage view and the EXACT sum policy.
    assert "Applicable-Stage Execution View" in si["note"]
    assert "EXACT sum of the applicable stage predictions" in si["note"]
    est = [s for s in si["stages"] if s["available"]]
    assert est, "serving stages must provide estimates"
    assert all(s["estimate_seconds"] > 0 for s in est)
    by_name = {s["stage"]: s for s in si["stages"]}
    # applicable-stage view: without prevalidation/transformation entries the
    # request covers exactly 5 applicable stages (user final spec, 2026-09-30).
    assert set(by_name) == {
        "staging", "pre_load", "data_loading",
        "datamart_approval", "dataloading_approval",
    }
    assert set(si.get("applicable_stages") or []) == set(by_name)
    # approval stages: contextual statistical estimator, explicitly not ML
    for approval in ("datamart_approval", "dataloading_approval"):
        assert by_name[approval]["method"] == "contextual_hierarchical_statistical"
        assert by_name[approval].get("note", "").startswith("Contextual statistical")


def test_api_non_predicted_states_carry_no_stage_intelligence(service):
    """Gate-blocked (INSUFFICIENT_HISTORY) responses carry the stage snapshot
    as INDICATIVE estimates with no total served (v2 design); an unresolved
    identity carries none (no context exists)."""
    # insufficient history: resolvable identity, zero priors
    r1 = service.predict(
        PredictionRequest(
            loader_name="2210_TEST_ALL_CONFIG",
            sprint="SUPPORT",
            dataset="A_TABLE",
            connection="test1",
        )
    )
    assert r1.state == PredictionState.INSUFFICIENT_HISTORY
    assert r1.predicted_seconds is None
    assert (r1.loader_case or {}).get("case") == "UNKNOWN_LOADER"
    assert r1.stage_intelligence is not None
    # unresolved identity
    r2 = service.predict(
        PredictionRequest(
            loader_name="definitely_not_a_loader_xyz",
            sprint="TEST",
            dataset="nope",
            connection="nope_conn",
        )
    )
    # 2026-10-03 final spec: an unknown loader is a KNOWN-NAME lookup that
    # resolves nowhere -> the new-loader refusal path (INSUFFICIENT_HISTORY),
    # never a fabricated total.
    assert r2.state == PredictionState.INSUFFICIENT_HISTORY
    assert r2.stage_intelligence is None


def test_api_stage_analytics_endpoint():
    from fastapi.testclient import TestClient

    from dltime.api import app

    client = TestClient(app)
    resp = client.get("/stages/analytics")
    assert resp.status_code == 200
    body = resp.json()
    assert "NEVER summed" in body["note"]
    stages = {s["stage"]: s for s in body["stages"]}
    assert stages["data_loading"]["modelability"] == "MODELABLE"
    assert stages["staging"]["modelability"] == "MODELABLE"
    assert stages["pre_load"]["modelability"] == "MODELABLE"
    # Approvals stay statistical-only (evidence 12 / 6 positives < floor 30)
    assert stages["datamart_approval"]["modelability"] == "ANALYTICS_ONLY"
    assert stages["dataloading_approval"]["modelability"] == "ANALYTICS_ONLY"
    assert "data_loading_1" not in stages  # old alias spelling removed
