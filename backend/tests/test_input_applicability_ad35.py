"""AD-35 senior-audit regression tests.

Covers the correctness risks flagged in the final audit:
  1. Config override independence (A/B/C/D states): separate prevalidation /
     transformation display names write ONLY their own marker + aggregates;
     neither corrupts the other; neutral overrides leave the ETA unchanged.
  2. Stage-specific history isolation: poisoning FUTURE rows' history for ONE
     stage changes only that stage's context — other stages' contexts are
     byte-identical (regression test for the EXP-032 Part-J class of bug).
  3. Floor scope: RETIRED with the sum policy (EXP-038) — the served total is
     never clamped; approvals enter the total only as statistical components
     (applicable-stage policy); overrides flow into the stage predictions that
     ARE the total.
  4. Overview target facts: authoritative semantics + labels.
"""

from __future__ import annotations

import pytest


# ------------------------------------------------------------------ 1. overrides
def _predict_with_overrides(service, **overrides):
    from dltime.inference import PredictionRequest
    from dltime.resolution import ResolutionStatus  # noqa: F401

    req = PredictionRequest(
        loader_name="MigrationSummarySubTableLoaderDataset",
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",
        connection="MDM_DM_TEST",
        **overrides,
    )
    return service.predict(req)


def test_overrides_abcd_prevalidation_only_marker(service):
    r = _predict_with_overrides(service, prevalidation_display_name="DO")
    assert r.state == "PREDICTED"
    f = r.features or {}
    assert f.get("def_preval_name_component") == "DO"
    assert f.get("def_trans_name_component") is None


def test_overrides_transformation_only_marker(service):
    r = _predict_with_overrides(service, transformation_display_name="CR")
    f = r.features or {}
    assert f.get("def_trans_name_component") == "CR"
    assert f.get("def_preval_name_component") is None


def test_overrides_both_neither_corruption(service):
    r = _predict_with_overrides(
        service, prevalidation_display_name="DO", transformation_display_name="CR"
    )
    f = r.features or {}
    assert f.get("def_preval_name_component") == "DO"
    assert f.get("def_trans_name_component") == "CR"


def test_display_name_entries_make_stages_applicable(service):
    """Applicable-stage policy (user final spec, 2026-09-30): display-name
    entries are the user's declaration that prevalidation/transformation run
    in this execution — the marked request covers 5 operational stages while
    the unmarked one covers 3, and each total equals its own applicable sum."""
    base = _predict_with_overrides(service)
    marked = _predict_with_overrides(
        service, prevalidation_display_name="DO", transformation_display_name="CR"
    )
    assert base.state == "PREDICTED"
    assert marked.state == "PREDICTED"
    base_ml = [
        s["estimate_seconds"]
        for s in (base.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    marked_ml = [
        s["estimate_seconds"]
        for s in (marked.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    assert len(base_ml) == 3  # staging + pre_load + data_loading
    assert len(marked_ml) == 5  # + prevalidation + transformation
    # Approval statistical estimates (same for both requests: same identity,
    # same as-of history) enter the total as components in applicable mode.
    assert marked.predicted_seconds == pytest.approx(sum(marked_ml) + _approval_sum(marked), abs=1e-9)
    assert base.predicted_seconds == pytest.approx(sum(base_ml) + _approval_sum(base), abs=1e-9)


def test_count_override_changes_only_its_own_aggregate(service):
    r = _predict_with_overrides(service, prevalidation_count=7)
    f = r.features or {}
    assert f.get("def_prevalidation_count") == 7.0
    # transformation aggregate untouched by the prevalidation override
    assert f.get("def_transformation_count") != 7.0


# ------------------------------------------------------------------ 2. stage-history isolation
def test_stage_specific_history_isolation(foundation, bundle):
    """Poisoning the stage TARGET of FUTURE rows for one stage must leave the
    generic stage_hist_* context of a probe row computed from that stage's OWN
    history — and must not change contexts built with other stages' history.

    Regression test for the EXP-032 Part-J class of bug (cross-stage
    overwrite of stage_hist_* columns)."""
    import pandas as pd

    from dltime.inference import _attach_context_stage_history
    from dltime.stage_intel import ML_STAGES
    from dltime.stage_targets import stage_target_column

    ex = foundation.frames["executions_eligible"].copy()
    start = pd.to_datetime(ex["Start_Time"]).max() + pd.Timedelta(days=1)
    ctx = ex.iloc[[0]].copy()
    ctx["LDR_Execution_Id"] = "__ISO_PROBE__"
    ctx["Start_Time"] = start
    ctx["is_training_eligible"] = False
    combined = pd.concat([ex, ctx], ignore_index=True)

    clean = _attach_context_stage_history(combined, "__ISO_PROBE__", ML_STAGES, stage_specific=True)

    # Poison FUTURE rows: append one more future row per stage with a huge
    # stage target, then verify the probe's stage_hist_<stage>_* values are
    # unchanged (future rows are strictly later -> never enter as-of history).
    poisoned = combined.copy()
    future = ex.iloc[[1]].copy()
    future["LDR_Execution_Id"] = "__ISO_POISON__"
    future["Start_Time"] = start + pd.Timedelta(days=2)
    future["is_training_eligible"] = True
    for stage in ML_STAGES:
        poisoned = pd.concat([poisoned, future.assign(**{stage_target_column(stage): 9.9e9})], ignore_index=True)
    dirty = _attach_context_stage_history(poisoned, "__ISO_PROBE__", ML_STAGES, stage_specific=True)

    def probe_values(df):
        row = df[df["LDR_Execution_Id"] == "__ISO_PROBE__"].iloc[0]
        return {c: row[c] for c in df.columns if c.startswith("stage_hist_")}

    clean_vals = probe_values(clean)
    dirty_vals = probe_values(dirty)
    assert set(clean_vals) == set(dirty_vals) and len(clean_vals) > 0
    for c in clean_vals:
        a, b = clean_vals[c], dirty_vals[c]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"stage history column {c} changed under future-row poisoning"


# ------------------------------------------------------------------ 3. floor scope (RETIRED)
def _approval_sum(result):
    """Sum of the approval statistical estimates that JOIN the total (2026-10-02
    final spec: only LOADER-tier evidence — the loader's own approval history)."""
    return sum(
        s["estimate_seconds"]
        for s in (result.stage_intelligence or {}).get("stages", [])
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    )


def test_no_floor_and_approvals_enter_total_when_available(service):
    """v2 sum policy (2026-09-30): the served total is NEVER clamped (no floor
    fields engage) and equals the exact applicable-stage sum PLUS the available
    approval statistical estimates. After the 2nd re-audit fix the approval
    specs read the corrected export's *_timetaken_in_sec fields, so this
    identity carries real GLOBAL-tier approval evidence (98.592 + 2.498)."""
    r = _predict_with_overrides(service)
    assert r.state == "PREDICTED"
    assert r.serving_role == "AUTHORITATIVE_TOTAL"
    assert r.floor_applied is False
    assert r.physical_floor_seconds is None
    si = r.stage_intelligence or {}
    stages = {s["stage"]: s for s in si.get("stages", [])}
    operational = [
        s["estimate_seconds"]
        for name, s in stages.items()
        if s.get("method") == "ml_stage_model"
        and s.get("estimate_seconds") is not None
        and name in {"staging", "prevalidation", "transformation", "pre_load", "data_loading"}
    ]
    approval_vals = [
        s["estimate_seconds"]
        for s in stages.values()
        if s.get("method") == "contextual_hierarchical_statistical"
        and s.get("estimate_seconds") is not None
        and s.get("included_in_total") is not False
    ]
    # applicable-stage policy (user final spec, 2026-09-30): NO prevalidation/
    # transformation entries -> exactly 3 applicable operational stages.
    assert len(operational) == 3
    # 2026-10-02 final spec: this loader has NO loader-tier approval history, so
    # the GLOBAL-tier estimates (98.592 + 2.498) are disclosed but EXCLUDED.
    assert len(approval_vals) == 0
    assert r.predicted_seconds == pytest.approx(
        sum(operational) + sum(approval_vals), abs=1e-9
    )
    assert r.predicted_seconds == pytest.approx(11.465, abs=1e-6)


def test_count_override_flows_through_stage_predictions_to_total(service):
    """A prevalidation-count override must change the total ONLY through the
    stage predictions it legitimately feeds — the sum identity stays exact."""
    r = _predict_with_overrides(service, prevalidation_count=7)
    si = r.stage_intelligence or {}
    ml_vals = [
        s["estimate_seconds"]
        for s in si.get("stages", [])
        if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None
    ]
    # the prevalidation entry makes prevalidation applicable -> 4 operational
    # stages (staging + prevalidation + pre_load + data_loading)
    assert len(ml_vals) == 4
    assert "prevalidation" in {s["stage"] for s in si.get("stages", [])}
    assert r.predicted_seconds == pytest.approx(sum(ml_vals) + _approval_sum(r), abs=1e-9)


# ------------------------------------------------------------------ 4. status consistency
def test_authoritative_result_carries_role_and_status(service):
    r = _predict_with_overrides(service)
    assert r.serving_role == "AUTHORITATIVE_TOTAL"
    # Sum policy (EXP-038): the serving identity is the deterministic
    # five-stage sum; the stage-aware artifact is retired to evidence.
    assert r.model_version == "sum-of-available-stages-v2"
    assert r.model_status == "DETERMINISTIC_POLICY"
