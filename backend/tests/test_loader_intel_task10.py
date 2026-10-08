"""Task 10 tests: Loader Intelligence + adaptive history evidence + cold-start context.

Covers: evidence-level classification, small-sample honesty, trend basis,
cold-start context (present AND absent), unknown/invalid identities,
identity disambiguation, and regressions that /predict behavior (Task
04-09 outputs) is untouched by this intelligence layer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dltime.inference import PredictionRequest, PredictionState
from dltime.loader_intel import (
    EvidenceLevel,
    LOADER_INTEL_VERSION,
    build_loader_intel,
    evidence_level_for,
)

# Real identities established from the actual corrected data (AD-31 regime).
STRONG = "MigrationSummarySubTableLoaderDataset"  # 10 eligible
STRONG_SPRINT = "TEST"
STRONG_DATASET = "Migration_Summary_Sub_Fact_Table_LDR_Datase_new"
EMERGING = "sample_ldr_case"  # 2 eligible
SUFFICIENT = "2902_latest_loader"  # 6 eligible
NO_HISTORY = "2210_TEST_ALL_CONFIG"  # definition loader, 0 eligible
UNKNOWN = "definitely_not_a_real_loader_xyz"
MULTI_SPRINT = "DGE Migration Summary Main Fact Table Loader_new"  # 2 sprints, 4 eligible
SINGLE_EXEC = "Asset Master AS91 HANA S20"  # exactly 1 eligible execution


@pytest.fixture(scope="module")
def client():
    from dltime.api import app

    with TestClient(app) as c:
        yield c


def _intel(client, name):
    r = client.get(f"/loaders/{name}")
    assert r.status_code == 200
    return r.json()


# ------------------------------------------------------------ level rules

def test_evidence_level_boundaries():
    assert evidence_level_for(0) == EvidenceLevel.NO_HISTORY
    assert evidence_level_for(1) == EvidenceLevel.EMERGING
    assert evidence_level_for(2) == EvidenceLevel.EMERGING
    assert evidence_level_for(3) == EvidenceLevel.SUFFICIENT
    assert evidence_level_for(9) == EvidenceLevel.SUFFICIENT
    assert evidence_level_for(10) == EvidenceLevel.STRONG


def test_policy_version_present(client):
    d = _intel(client, STRONG)
    assert d["loader_intel_version"] == LOADER_INTEL_VERSION


# ------------------------------------------------------- known/edge cases

def test_strong_loader_full_intelligence(client):
    d = _intel(client, STRONG)
    assert d["known"] is True
    assert d["evidence_level"] == EvidenceLevel.STRONG
    h = d["history"]
    assert h["eligible_execution_count"] == 11
    assert h["show_typical"] is True
    assert h["median_sec"] is not None and h["cv"] is not None and h["p90_sec"] is not None
    # trend exists and cites the documented basis (n>=6)
    assert h["trend"]["label"] in {"stable", "increasing", "decreasing"}
    assert "EXP-019" in h["trend"]["basis"] and "n>=6" in h["trend"]["basis"]
    assert d["eta_eligible"] is True
    assert d["eta_gate"]["satisfied"] is True
    assert d["cold_start"]["applicable"] is False


def test_sufficient_loader_matches_eta_population(client):
    d = _intel(client, SUFFICIENT)
    assert d["evidence_level"] == EvidenceLevel.SUFFICIENT
    assert d["history"]["eligible_execution_count"] == 6
    assert d["history"]["show_typical"] is True
    assert d["eta_eligible"] is True
    assert d["history"].get("trend") is not None  # n=6 >= TREND_MIN


def test_emerging_loader_hides_typical_duration(client):
    d = _intel(client, EMERGING)
    assert d["evidence_level"] == EvidenceLevel.EMERGING
    h = d["history"]
    assert h["eligible_execution_count"] == 2
    # honesty rule: no typical-duration statistics for tiny samples
    assert h["show_typical"] is False
    assert "median_sec" not in h and "mean_sec" not in h and "cv" not in h
    assert any("typical-duration" in lim for lim in d["limitations"])
    assert d["eta_eligible"] is False
    assert "requires at least 3" in d["eta_gate"]["explanation"]


def test_no_history_loader_with_definition_context(client):
    d = _intel(client, NO_HISTORY)
    assert d["evidence_level"] == EvidenceLevel.NO_HISTORY
    assert d["history"]["eligible_execution_count"] == 0
    assert d["eta_eligible"] is False
    assert d["definition_context"]["matched"] is True
    assert "A_TABLE" in d["definition_context"]["datasets"]
    assert d["cold_start"]["applicable"] is True
    assert d["cold_start"]["eta_produced"] is False
    # 2026-09-30: on the corrected export this loader's resolved connection
    # ('test1') has 5 eligible executions by 2 other loaders — above the
    # 3-observation honesty threshold, so the descriptive contextual
    # statistic IS shown (clearly labelled same-connection context, never a
    # loader-specific claim). eta stays withheld (0 own priors).
    ctx = d["cold_start"]["contextual_intelligence"]
    assert ctx is not None and ctx["kind"] == "CONTEXTUAL_SAME_CONNECTION"
    assert ctx["observation_count"] == 5 and ctx["distinct_loaders"] == 2


def _cold_start_unit_frame():
    """Synthetic frame: cold loader with genuine >=3 other-loader context."""
    import pandas as pd

    rows = [
        # the cold loader itself: no eligible executions
        dict(LDR_Execution_Id="C0", Loader_Name="Cold_Ldr", Sprint="S1",
             Dataset_display_name="DS_X", Start_Time="2026-01-10", execution_connection="CNX",
             is_training_eligible=False, timetaken_in_sec=None,
             execution_resolution_status="UNRESOLVED"),
        # four other-loader eligible executions on the same dataset (context)
        dict(LDR_Execution_Id="C1", Loader_Name="Other_A", Sprint="S1",
             Dataset_display_name="DS_X", Start_Time="2026-01-01", execution_connection="CNX",
             is_training_eligible=True, timetaken_in_sec=10.0,
             execution_resolution_status="RESOLVED"),
        dict(LDR_Execution_Id="C2", Loader_Name="Other_B", Sprint="S1",
             Dataset_display_name="DS_X", Start_Time="2026-01-02", execution_connection="CNX",
             is_training_eligible=True, timetaken_in_sec=20.0,
             execution_resolution_status="RESOLVED"),
        dict(LDR_Execution_Id="C3", Loader_Name="Other_A", Sprint="S1",
             Dataset_display_name="DS_X", Start_Time="2026-01-03", execution_connection="CNX",
             is_training_eligible=True, timetaken_in_sec=30.0,
             execution_resolution_status="RESOLVED"),
        dict(LDR_Execution_Id="C4", Loader_Name="Other_C", Sprint="S1",
             Dataset_display_name="DS_other", Start_Time="2026-01-04", execution_connection="CNX",
             is_training_eligible=True, timetaken_in_sec=40.0,
             execution_resolution_status="RESOLVED"),
    ]
    return pd.DataFrame(rows)


def test_cold_start_with_genuine_context_is_contextual_not_prediction():
    """Unit-level (AD-31: no live loader reaches the contextual threshold):
    a cold loader with >=3 other-loader same-dataset eligible executions gets
    CONTEXTUAL intelligence — clearly labelled, never an ETA."""
    import pandas as pd

    from dltime.loader_intel import build_loader_intel

    definitions = pd.DataFrame(
        [
            dict(loader_display_name="Cold_Ldr", datasetname="DS_X", sprint_name="S1", ldr_connection_name="CNX"),
        ]
    )
    d = build_loader_intel(
        loader_name="Cold_Ldr",
        frame=_cold_start_unit_frame(),
        definitions_detail=definitions,
    )
    cs = d["cold_start"]
    assert cs["applicable"] is True
    assert cs["eta_produced"] is False
    ctx = cs["contextual_intelligence"]
    assert ctx is not None
    assert ctx["kind"] == "CONTEXTUAL_SAME_DATASET"
    assert ctx["observation_count"] >= 3
    assert "NOT a prediction" in ctx["disclaimer"]
    assert "NEVER used as its ETA" in ctx["disclaimer"]
    assert d["eta_eligible"] is False


def test_multi_sprint_loader_lists_all(client):
    d = _intel(client, MULTI_SPRINT)
    assert len(d["context"]["sprints"]) >= 2
    assert d["history"]["eligible_execution_count"] >= 3


def test_single_execution_loader_hides_typical(client):
    d = _intel(client, SINGLE_EXEC)  # exactly 1 eligible execution
    h = d["history"]
    assert h["eligible_execution_count"] == 1
    assert h["show_typical"] is False
    assert "median_sec" not in h
    assert "trend" not in h
    assert any("Only one eligible execution" in lim for lim in d["limitations"])


def test_min_history_loader_is_eta_eligible(client):
    d = _intel(client, "DGE Migration Summary Main Fact Table Loader_new")  # 4 eligible
    assert d["eta_eligible"] is True
    assert d["evidence_level"] == EvidenceLevel.SUFFICIENT


# ------------------------------------------------------ identity semantics

def test_identity_decoding_and_missing_name(client):
    import urllib.parse

    # spaces decode correctly
    d = _intel(client, urllib.parse.quote("Asset Master AS91 HANA S20"))
    assert d["loader_name"] == "Asset Master AS91 HANA S20"
    assert d["evidence_level"] == EvidenceLevel.EMERGING
    # an empty-ish lookup returns the NO_HISTORY schema (FastAPI route misses -> 404)
    r = client.get("/loaders/%20")
    assert r.status_code in (200, 404)


def test_loader_intel_is_read_only_for_prediction(service):
    """The intelligence layer must not alter any Task 04-09 output."""
    base = service.predict(
        PredictionRequest(
            loader_name=STRONG,
            sprint=STRONG_SPRINT,
            dataset=STRONG_DATASET,
            connection="MDM_DM_TEST",
        )
    )
    assert base.state == PredictionState.PREDICTED
    # AD-32: served ETA = authoritative stage sum; direct reference constant intact.
    assert base.serving_role == "AUTHORITATIVE_TOTAL"
    assert base.direct_reference_seconds == pytest.approx(16.289350640581837, rel=1e-9)
    # applicable-stage policy (user final spec, 2026-09-30): this request has
    # NO prevalidation/transformation entries, so the ML part is the applicable
    # 3-stage sum (0.6 + 10.388 + 0.477 = 11.465 s) plus the GLOBAL-tier approval
    # statistical estimates (11.465 + 11.465). Interval: point ± Q90 against
    # the rebuilt 44-feature conformal calibration (2026-10-06 rebuild of
    # unc-log1p-conformal-v1.json, fc=44): 11.465 ± 3.035 -> 8.43 - 14.5.
    assert base.predicted_seconds == pytest.approx(11.465, abs=0.01)
    assert (base.uncertainty or {}).get("lower_seconds") == pytest.approx(8.43, abs=0.01)
    assert (base.uncertainty or {}).get("upper_seconds") == pytest.approx(14.5, abs=0.01)
    assert (base.uncertainty or {}).get("interval_width_seconds") == pytest.approx(6.07, abs=0.01)
    assert base.explanation is not None
    assert base.risk is not None
    assert base.stage_intelligence is not None
    # and the insufficient-history contract is intact
    ins = service.predict(
        PredictionRequest(
            loader_name=NO_HISTORY,
            sprint="SUPPORT",
            dataset="A_TABLE",
            connection="test1",
        )
    )
    assert ins.state == PredictionState.INSUFFICIENT_HISTORY
    assert ins.predicted_seconds is None
