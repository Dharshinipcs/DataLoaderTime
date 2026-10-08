"""Evidence-sentence explainability (evidence-v1) — user final spec 2026-10-01.

Deterministic template sentences, no AI/LLM:
  - Layer 1: one plain-English sentence per applicable ML stage comparing the
    prediction against the loader's OWN strictly-prior history (count +
    median, the exact stage_hist_<stage>_loader_* features the model saw),
    plus one statistical line per available approval stage.
  - Layer 2: what-if counterfactuals for prevalidation/transformation stages
    the user did not enter — quoting that stage's own model estimate.
  - Layer 3 (untouched): the SHAP ranked contributions remain the technical
    backing; both live in the same Prediction Insights panel.

The sentences must quote ONLY numbers computed from the request — verifiable
against the Loader Intelligence page. Never a probability, never a confidence,
never an invented number.
"""

from __future__ import annotations

import pytest

from dltime.explain_evidence import build_evidence_explanation


# ------------------------------------------------------------------ helpers
def _stage_entry(name, seconds, method="ml_stage_model", **extra):
    base = {
        "stage": name,
        "estimate_seconds": seconds,
        "available": seconds is not None,
        "method": method,
    }
    base.update(extra)
    return base


# ------------------------------------------------------------------ Layer 1
def test_in_line_history_sentence():
    payload = build_evidence_explanation(
        stage_intel={"stages": [_stage_entry("staging", 0.6)]},
        unfiltered_stage_estimates={},
        history={"prior_loader_executions": 11},
        ctx_stage_history={"staging": {"prior_count": 10, "prior_median": 0.62}},
        total_seconds=11.465,
        applicable=["staging"],
        not_applicable=["prevalidation", "transformation"],
    )
    assert payload["available"] is True
    assert len(payload["sentences"]) == 1
    s = payload["sentences"][0]
    assert "Staging" in s
    assert "10 time(s)" in s
    assert "0.6s" in s  # typical history time
    assert "in line with your usual" in s


def test_deviation_sentence_names_direction_and_pct():
    payload = build_evidence_explanation(
        stage_intel={"stages": [_stage_entry("data_loading", 0.477)]},
        unfiltered_stage_estimates={},
        history={"prior_loader_executions": 11},
        ctx_stage_history={"data_loading": {"prior_count": 11, "prior_median": 1.9}},
        total_seconds=11.465,
        applicable=["data_loading"],
        not_applicable=[],
    )
    s = payload["sentences"][0]
    assert "11 time(s)" in s
    assert "1.9s" in s
    assert "lower than your usual" in s  # 2026-10-04 retrain: data_loading now predicts below the typical
    assert "%" in s


def test_no_history_sentence_is_honest():
    payload = build_evidence_explanation(
        stage_intel={"stages": [_stage_entry("pre_load", 10.388)]},
        unfiltered_stage_estimates={},
        history={"prior_loader_executions": 0},
        ctx_stage_history={"pre_load": {"prior_count": 0, "prior_median": None}},
        total_seconds=10.388,
        applicable=["pre_load"],
        not_applicable=[],
    )
    s = payload["sentences"][0]
    assert "no history exists for this loader" in s
    assert "10s" in s  # 2026-10-04 retrain: pre_load prediction moved to ~10.4s


def test_approval_sentences_carry_tier_and_count():
    payload = build_evidence_explanation(
        stage_intel={
            "stages": [
                _stage_entry(
                    "datamart_approval",
                    98.592,
                    method="contextual_hierarchical_statistical",
                    evidence_tier="GLOBAL",
                    evidence_tier_label="Global historical median",
                    n_evidence=7,
                )
            ]
        },
        unfiltered_stage_estimates={},
        history={"prior_loader_executions": 11},
        ctx_stage_history={},
        total_seconds=11.465,
        applicable=["datamart_approval"],
        not_applicable=[],
    )
    s = payload["sentences"][0]
    assert "1m 39s" in s
    assert "7 matching past approval(s)" in s
    assert "global historical median" in s
    assert payload["approval_sentence_count"] == 1


# ------------------------------------------------------------------ Layer 2
def test_what_if_quotes_unentered_stage_estimate():
    payload = build_evidence_explanation(
        stage_intel={"stages": []},
        unfiltered_stage_estimates={"prevalidation": 86.215, "transformation": 5.33},
        history={"prior_loader_executions": 11},
        ctx_stage_history={},
        total_seconds=11.465,
        applicable=["staging"],
        not_applicable=["prevalidation", "transformation"],
    )
    assert len(payload["what_ifs"]) == 2
    assert "1m 26s" in payload["what_ifs"][0]
    assert "prevalidation" in payload["what_ifs"][0]


def test_no_what_if_when_stage_entered():
    payload = build_evidence_explanation(
        stage_intel={"stages": []},
        unfiltered_stage_estimates={"prevalidation": 86.215},
        history={"prior_loader_executions": 11},
        ctx_stage_history={},
        total_seconds=93.0,
        applicable=["staging", "prevalidation"],
        not_applicable=[],
    )
    assert payload["what_ifs"] == []


def test_unavailable_estimate_produces_no_invented_number():
    payload = build_evidence_explanation(
        stage_intel={"stages": []},
        unfiltered_stage_estimates={"prevalidation": None},
        history={"prior_loader_executions": 3},
        ctx_stage_history={},
        total_seconds=6.0,
        applicable=["staging"],
        not_applicable=["prevalidation"],
    )
    assert payload["what_ifs"] == []
    assert payload["available"] is False or payload["sentences"]


# ------------------------------------------------------------------ contract
def test_payload_never_contains_confidence_or_probability():
    payload = build_evidence_explanation(
        stage_intel={"stages": [_stage_entry("staging", 0.6)]},
        unfiltered_stage_estimates={},
        history={"prior_loader_executions": 5},
        ctx_stage_history={"staging": {"prior_count": 4, "prior_median": 0.55}},
        total_seconds=6.0,
        applicable=["staging"],
        not_applicable=[],
    )
    blob = (payload["summary"] + " ".join(payload["sentences"] + payload["what_ifs"])).lower()
    for banned in ("confidence", "probability", "% sure", "likely"):
        assert banned not in blob


# ------------------------------------------------------- real serving path
@pytest.fixture(scope="module")
def _svc():
    from dltime.inference import PredictionService

    return PredictionService.from_sources()


def test_real_prediction_carries_evidence_section(_svc):
    from dltime.inference import PredictionRequest

    r = _svc.predict(
        PredictionRequest(
            loader_name="MigrationSummarySubTableLoaderDataset",
            sprint="TEST",
            dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
            connection="MDM_DM_TEST",
        )
    )
    assert r.state == "PREDICTED"
    ee = r.evidence_explanation
    assert ee is not None and ee["available"] is True
    assert ee["evidence_version"] == "evidence-v1"
    # one sentence per applicable ML stage; approval sentences appear ONLY
    # when the estimate comes from the loader's OWN approval history — this
    # reference loader has none (its estimates are GLOBAL fallback), so zero
    # approval sentences are emitted (2026-10-03 user final spec).
    assert ee["ml_sentence_count"] == 3  # staging + pre_load + data_loading
    assert ee["approval_sentence_count"] == 0
    assert len(ee["what_ifs"]) == 2  # prevalidation + transformation not entered
    # every ML sentence mentions a count/median drawn from the real history
    assert any("10 time(s)" in s for s in ee["sentences"])
    # the payload serializes through to_dict (API shape)
    d = r.to_dict()
    assert d["evidence_explanation"]["summary"]
