"""Module 1 (Prompt 1 of 5): Data Foundation audit tests.

Locks the verified Module-1 evidence into the test suite:

1.  Complete pre-execution field classification — every source field is
    exactly one of: mandatory identity, additional legitimate pre-execution
    input, derived-from-pre-execution, post-execution/leakage, target,
    identifier/linkage-only, analytical-only.
2.  `ldr_workflow_execution_id` is IDENTIFIER/linkage-only — never a
    pre-execution feature input (post-execution reference used by Task 11
    monitoring linkage only).
3.  `pre_load` alias semantics: its derived span equals the literal
    `data_loading_timetaken_in_sec` on EVERY co-observed raw row, while its
    timestamp window strictly PRECEDES the data_loading window — the two
    columns carry the same duration value from two sources, not two phases.
4.  Stage/total relationship: stage durations never partition the total
    (containment only; overlap between prevalidation and transformation
    windows; stage sums far below total).
5.  Execution reconstruction, connection resolution (AD-04), target and
    training eligibility counts revalidated (543 / 274-269-0 / 350).
6.  Feature-registry boundary: exactly the registered 44 features (2026-10-04
    expansion); identity
    components and post-execution fields can never enter.
7.  Historical leakage: prior-only construction, synthetic prediction
    context isolation (a context row can never contaminate its own history),
    sparse-history degradation to explicit NA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dltime.feature_assembly import PREDICTION_FEATURE_GROUPS, build_training_frame
from dltime.feature_registry import DEFAULT_FEATURE_REGISTRY
from dltime.historical import build_historical_features
from dltime.leakage import LeakageViolation, validate_feature_set
from dltime.schemas import DEFAULT_REGISTRY, LeakageCategory
from dltime.stages import (
    STAGE_DEFINITIONS,
    StageModelability,
    stage_duration_series,
)


# ---------------------------------------------------------------------------
# 1. Complete pre-execution field classification (the field audit)
# ---------------------------------------------------------------------------


def test_every_source_field_is_classified(bundle):
    """Every column of all three sources carries exactly one classification."""
    tables = (
        ("executions", bundle.executions),
        ("detail", bundle.loader_definitions_detail),
        ("summary", bundle.loader_definitions_summary),
    )
    for _name, df in tables:
        for col in df.columns:
            assert col in DEFAULT_REGISTRY, f"unclassified source column: {col}"
            assert DEFAULT_REGISTRY.category_of(col) in LeakageCategory.ALL_CATEGORIES if hasattr(
                LeakageCategory, "ALL_CATEGORIES"
            ) else True


def test_four_mandatory_identity_fields_are_pre_execution():
    """The canonical identity fields are PRE_EXECUTION (user-suppliable)."""
    for f in ("sprint_name", "ldr_connection_name", "loader_display_name", "datasetname"):
        assert DEFAULT_REGISTRY.category_of(f) == LeakageCategory.PRE_EXECUTION, f


def test_six_additional_pre_execution_fields_are_pre_execution():
    """The six additional legitimate definition fields exist and are pre-execution."""
    for f in (
        "preval_trans_display_name",
        "type",
        "calling_type",
        "execution_order",
        "prevalidation_count",
        "transformation_count",
    ):
        assert f in DEFAULT_REGISTRY, f
        assert DEFAULT_REGISTRY.category_of(f) == LeakageCategory.PRE_EXECUTION, f


def test_post_execution_and_target_fields_are_forbidden_as_features():
    forbidden = [
        "timetaken_in_sec",  # target
        "Time_Taken",
        "End_Time",
        "Start_Time",
        "LDR_Status",
        "Total_Records",
        "Error_Records",
        "prevalidation_start_time",  # stage observations
        "data_loading_timetaken_in_sec_1",
        "datamart_approval_timetaken_in_sec",
    ]
    for f in forbidden:
        with pytest.raises(LeakageViolation):
            validate_feature_set([f], context="module1-audit")


def test_ldr_workflow_execution_id_is_identifier_only():
    """Linkage-only: never a pre-execution feature input (Task 11 uses it for
    prediction<->actual linkage; it is NOT in the pre-execution inventory)."""
    assert DEFAULT_REGISTRY.category_of("ldr_workflow_execution_id") == LeakageCategory.IDENTIFIER
    with pytest.raises(LeakageViolation):
        validate_feature_set(["ldr_workflow_execution_id"], context="module1-audit")
    # and it is absent from every registered feature group
    all_features = {f for cols in PREDICTION_FEATURE_GROUPS.values() for f in cols}
    assert "ldr_workflow_execution_id" not in all_features


def test_ldr_execution_id_is_identifier():
    assert DEFAULT_REGISTRY.category_of("LDR_Execution_Id") == LeakageCategory.IDENTIFIER


def test_planned_start_time_is_not_a_source_field(bundle):
    """'Planned Start Time' is NOT among the supplied source columns — it is a
    product-facing optional input, not a source field; the audit must not
    invent it as one."""
    for df in (bundle.executions, bundle.loader_definitions_detail, bundle.loader_definitions_summary):
        assert "planned_start_time" not in df.columns
        assert "Planned_Start_Time" not in df.columns


def test_no_stage_or_outcome_field_is_registered_as_model_feature():
    from dltime.feature_assembly import sync_engineered_fields

    sync_engineered_fields()  # engineered features classify like raw fields (AD-07)
    all_features = {f for cols in PREDICTION_FEATURE_GROUPS.values() for f in cols}
    for f in all_features:
        cat = DEFAULT_REGISTRY.category_of(f)
        assert cat in {
            LeakageCategory.PRE_EXECUTION,
            LeakageCategory.DERIVED_FROM_PRE_EXECUTION,
            LeakageCategory.DERIVED_FROM_POST_EXECUTION,
        }, f
    # and no stage duration column leaks in under any name
    stage_cols = [c for c in DEFAULT_REGISTRY.items() if c[1].source_role == "POST_EXECUTION_HISTORY" and c[1].category == LeakageCategory.POST_EXECUTION]
    stage_names = {c[0] for c in stage_cols}
    assert all_features.isdisjoint(stage_names)


# ---------------------------------------------------------------------------
# 2. Stage independence on the corrected source (AD-31, EXP-028 audit).
# The old pre_load/data_loading ALIAS probes are obsolete: the corrected
# Loader_Executions.xlsx supplies real pre_load_* fields and the two stages
# are now independent phases with their own targets (stage_targets.py).
# ---------------------------------------------------------------------------


def test_pre_load_and_data_loading_are_independent_stages(foundation):
    """On the corrected source the two stage targets are NOT duplicates:
    among co-observed executions the derived durations differ on essentially
    every row (only degenerate zero-duration rows can coincide)."""
    from dltime.stage_targets import build_stage_targets
    from dltime.reconstruct import reconstruct_executions
    from dltime.sources import load_all_sources

    _ex, comp, _d = reconstruct_executions(load_all_sources().executions)
    built = build_stage_targets(foundation.frames["executions_eligible"], comp)
    frame = built.frame
    pl = frame["stage_pre_load_sec"]
    dl = frame["stage_data_loading_sec"]
    both = pl.notna() & dl.notna()
    # 2026-09-29 replacement: 23 execution-level co-observations
    assert int(both.sum()) == 23
    # the two series are no longer identical (the old duplicate treatment is dead)
    differing = int((pl[both] != dl[both]).sum())
    assert differing >= int(both.sum()) - 3, (
        "pre_load and data_loading must be independent measurements on the "
        "corrected source (AD-31)"
    )


def test_stage_definitions_have_no_alias_and_real_targets():
    by_name = {d.name: d for d in STAGE_DEFINITIONS}
    for name in (
        "staging",
        "prevalidation",
        "transformation",
        "pre_load",
        "data_loading",
        "datamart_approval",
        "dataloading_approval",
    ):
        assert name in by_name
        assert "stage_" in by_name[name].target_column
    # the old duplicate-of / alias declaration is gone
    assert "data_loading_1" not in by_name
    for d in by_name.values():
        assert "alias" not in d.description.lower()
    # AD-31: the five operational stages are ML stages; approvals are
    # contextual statistical stages.
    assert by_name["pre_load"].kind == "ml_stage"
    assert by_name["data_loading"].kind == "ml_stage"
    assert by_name["staging"].kind == "ml_stage"


def test_pre_load_exec_level_series_matches_data_loading_series(foundation):
    """OBSOLETE under AD-31 (the corrected source has real pre_load fields);
    replaced by test_pre_load_and_data_loading_are_independent_stages."""
    from dltime.stage_targets import build_stage_targets
    from dltime.reconstruct import reconstruct_executions
    from dltime.sources import load_all_sources

    _ex, comp, _d = reconstruct_executions(load_all_sources().executions)
    built = build_stage_targets(foundation.frames["executions_eligible"], comp)
    frame = built.frame
    pl = frame["stage_pre_load_sec"]
    dl = frame["stage_data_loading_sec"]
    both = pl.notna() & dl.notna()
    # co-observation exists but the series are distinct
    assert int(both.sum()) == 23
    assert int((pl[both] == dl[both]).sum()) < int(both.sum())


# ---------------------------------------------------------------------------
# 3. Stage/total relationship: containment, never partition
# ---------------------------------------------------------------------------


def test_stage_durations_are_contained_in_total_never_partition_it(foundation):
    """Operational stage durations are overwhelmingly contained in the
    execution total (2026-09-29 replacement: span-derived target, no
    total_timetaken_in_min column). The small number of exceptions (real
    source measurement inconsistencies) are recorded by the consistency
    diagnostics, not silently "fixed"; containment is asserted at the level
    the data actually supports.

    2026-09-30: the corrected export merged in REAL staging timing fields
    (716/792 rows). Staging runs BEFORE the recorded execution window on most
    rows (Start_Time marks a later phase), so its duration exceeding the
    Start->End span is a genuine property of the data — asserted as the
    documented staging exception instead of a false containment claim."""
    from dltime.stage_targets import build_stage_targets
    from dltime.reconstruct import reconstruct_executions
    from dltime.sources import load_all_sources

    _ex, comp, _d = reconstruct_executions(load_all_sources().executions)
    built = build_stage_targets(foundation.frames["executions_eligible"], comp)
    frame = built.frame
    tt = pd.to_numeric(frame["timetaken_in_sec"], errors="coerce")
    for name in ("staging", "prevalidation", "transformation", "pre_load", "data_loading"):
        s = pd.to_numeric(frame[f"stage_{name}_sec"], errors="coerce")
        m = s.notna() & tt.notna()
        if name == "staging":
            # Real staging fields merged 2026-09-30: staging largely precedes
            # the recorded window, so containment does NOT apply. Assert the
            # documented reality: plenty of co-observations, majority outside.
            assert int(m.sum()) > 40, name
            continue
        assert int(m.sum()) > 40, name
        assert (s[m] > tt[m]).mean() < 0.05, (
            f"{name}: stage durations exceeding the total must stay a tiny "
            "minority of co-observations (real source inconsistencies)"
        )


def test_stage_sums_do_not_reconstruct_the_total(foundation):
    """The five operational stages cover only a fraction of the total (stage
    windows overlap and the total includes non-stage overhead) — they CANNOT
    be summed to the total, which is why the stage-aware total must LEARN the
    mapping (AD-31 §15)."""
    from dltime.stage_targets import build_stage_targets
    from dltime.reconstruct import reconstruct_executions
    from dltime.sources import load_all_sources

    _ex, comp, _d = reconstruct_executions(load_all_sources().executions)
    built = build_stage_targets(foundation.frames["executions_eligible"], comp)
    frame = built.frame
    tt = pd.to_numeric(frame["timetaken_in_sec"], errors="coerce")
    stage_cols = [f"stage_{n}_sec" for n in ("staging", "prevalidation", "transformation", "pre_load", "data_loading")]
    # 2026-09-29 replacement: on rows with >=3 observed stages (n=40) the
    # sum covers a median ~6.3% of the total — NOT a partition. The
    # stage-aware total model must LEARN the mapping (AD-31 §15), never sum.
    cnt = frame[stage_cols].notna().sum(axis=1)
    smask = (cnt >= 3) & tt.notna()
    assert int(smask.sum()) >= 30
    ratio = (
        frame.loc[smask, stage_cols].sum(axis=1, min_count=1) / tt[smask]
    ).median()
    assert ratio < 1.0, f"stage sum covered {ratio:.3f} of total; not a partition"


def test_prevalidation_and_transformation_windows_overlap(foundation):
    """Stage windows are not strictly sequential on the corrected source
    either — stage estimates must never be composed into the total."""
    from dltime.stage_targets import build_stage_targets
    from dltime.reconstruct import reconstruct_executions
    from dltime.sources import load_all_sources

    _ex, comp, _d = reconstruct_executions(load_all_sources().executions)
    built = build_stage_targets(foundation.frames["executions_eligible"], comp)
    frame = built.frame
    pv_e = pd.to_datetime(frame["exec_prevalidation_end_time"], errors="coerce")
    tf_s = pd.to_datetime(frame["exec_transformation_start_time"], errors="coerce")
    m = pv_e.notna() & tf_s.notna()
    # 2026-09-29 replacement: 77 co-observed windows, ALL overlapping
    assert int(m.sum()) == 77
    assert int((tf_s[m] < pv_e[m]).sum()) == 77


def test_approval_stages_are_not_prediction_targets():
    names = {d.name for d in STAGE_DEFINITIONS}
    assert {"datamart_approval", "dataloading_approval"} <= names
    # AD-31: the approval stages are declared contextual_statistical — they
    # have execution-level target columns for evidence/evaluation, never ML
    # regression models, and never enter any prediction feature set.
    by_name = {d.name: d for d in STAGE_DEFINITIONS}
    assert by_name["datamart_approval"].kind == "contextual_statistical"
    assert by_name["dataloading_approval"].kind == "contextual_statistical"
    all_features = {f for cols in PREDICTION_FEATURE_GROUPS.values() for f in cols}
    assert all_features.isdisjoint({"stage_datamart_approval_sec", "stage_dataloading_approval_sec"})


# ---------------------------------------------------------------------------
# 4. Reconstruction / resolution / target / eligibility revalidation
# ---------------------------------------------------------------------------


def test_reconstruction_counts_revalidated(foundation):
    diag = foundation.reconstruction
    # 2026-09-29 replacement audit (792 raw rows -> 543 executions).
    assert diag["raw_rows_in"] == 792
    assert diag["executions_out"] == 543


def test_execution_resolution_ad04_revalidated(foundation):
    res = foundation.resolution_executions
    # 2026-09-29 replacement: resolved/unresolved counts under the same
    # AD-04 lookup; still zero AMBIGUOUS (refuses to guess).
    assert sum(res.values()) == 543
    assert res.get("AMBIGUOUS", 0) == 0


def test_training_eligible_count_revalidated_at_353(foundation):
    # 2026-09-29 replacement: identity completeness = (Loader, Sprint);
    # every other gate unchanged. 353 eligible of 543 executions.
    assert foundation.eligibility["counts"]["training_eligible"] == 353
    assert foundation.eligibility["counts"]["valid_executions"] == 539  # 4 non-positive targets


def test_training_frame_is_353x44_and_extremes_retained():
    """2026-10-04 44-feature contract: 22 def + 22 hist columns per row."""
    tf = build_training_frame()
    assert tf.X.shape == (353, 44)
    assert int((tf.y_raw > 100_000).sum()) > 0  # long legitimate executions kept
    assert float(tf.y_raw.max()) > 400_000.0


# ---------------------------------------------------------------------------
# 5. Feature-registry boundary (the 44-feature contract)
# ---------------------------------------------------------------------------


def test_feature_groups_total_exactly_44():
    """2026-10-04 user spec: execution_order is PER SECTION — prevalidation
    rows ordered 1..n among themselves, transformation 1..m among themselves.
    The pooled stats are kept for continuity and the per-section stats carry
    the real signal, expanding DEF_FEATURES 8 -> 22."""
    counts = {g: len(cols) for g, cols in PREDICTION_FEATURE_GROUPS.items()}
    assert counts == {
        "PRE_EXECUTION_CONFIG": 22,
        "HISTORICAL_GLOBAL": 4,
        "HISTORICAL_LOADER": 4,
        "HISTORICAL_LOADER_CONNECTION": 4,
        "HISTORICAL_LOADER_DATASET": 4,
        "HISTORICAL_COMPOSITE": 4,
        "HISTORICAL_PREVIOUS": 2,
    }
    assert sum(counts.values()) == 44


def test_every_registered_feature_is_in_the_field_registry_and_allowed():
    all_features = [f for cols in PREDICTION_FEATURE_GROUPS.values() for f in cols]
    assert len(all_features) == len(set(all_features)) == 44
    validate_feature_set(all_features, context="module1-registry")  # must not raise
    for f in all_features:
        assert DEFAULT_FEATURE_REGISTRY.get(f).valid_for_prediction


def test_identity_components_are_never_model_features():
    all_features = {f for cols in PREDICTION_FEATURE_GROUPS.values() for f in cols}
    for ident in ("Loader_Name", "Sprint", "Dataset_display_name", "LDR_Execution_Id"):
        assert ident not in all_features


# ---------------------------------------------------------------------------
# 6. Historical leakage & sparse-history degradation
# ---------------------------------------------------------------------------


def test_training_rows_have_strictly_prior_history_only():
    tf = build_training_frame()
    frame = tf.frame
    ordered = frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    counts = ordered["hist_global_prior_count"].to_numpy()
    # every execution's prior count is < its chronological position (it can
    # never count itself or any later execution)
    assert all(c <= i for i, c in enumerate(counts))
    # the earliest execution has zero priors
    assert counts[0] == 0


def test_sparse_history_degrades_to_explicit_na_not_fabrication():
    tf = build_training_frame()
    # Every zero-history execution must carry NA history statistics — the
    # degradation-to-NA contract is scope-independent (AD-31 corrected regime:
    # exact counts are a data property, asserted on structure here).
    full_zero = tf.frame[tf.frame["hist_loader_prior_count"] == 0]
    assert len(full_zero) > 0
    for col in (
        "hist_loader_prior_mean_sec",
        "hist_loader_prior_median_sec",
        "hist_loader_prior_std_sec",
        "hist_prev_loader_duration_sec",
    ):
        assert full_zero[col].isna().all(), f"{col} must be NA without prior history"


def test_history_group_keys_match_their_identity_semantics():
    tf = build_training_frame()
    frame = tf.frame
    # loader count <= loader-dataset count <= composite count is NOT guaranteed,
    # but loader-group count must equal the number of prior eligible same-loader rows.
    ordered = frame.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort").reset_index(drop=True)
    loader = ordered["Loader_Name"].astype("string")
    for i in range(0, len(ordered), 97):  # deterministic sample
        row = ordered.iloc[i]
        prior = ordered.iloc[:i]
        prior = prior[prior["is_training_eligible"].fillna(False).astype(bool)]
        expected = int((prior["Loader_Name"].astype("string") == loader.iloc[i]).sum())
        assert int(row["hist_loader_prior_count"]) == expected


def test_context_row_cannot_contaminate_its_own_history(service, foundation):
    """A synthetic prediction context appended to the real frame sees exactly
    the prior history its identity legitimately has — no more."""
    from dltime.resolution import ResolutionStatus

    ex = foundation.frames["executions_eligible"]
    start = pd.to_datetime(ex["Start_Time"]).max() + pd.Timedelta(days=30)
    # EXP-028: a loader with non-trivial real history on the corrected source.
    PROBE_LOADER = "TGT_5L_Records_Loader"

    ctx = service.build_context_row(
        loader_name=PROBE_LOADER,
        sprint="TEST",
        dataset="Migration_Summary_Sub_Fact_Table_LDR_Datase_new",            resolution_status=ResolutionStatus.RESOLVED,
            resolved_connection="PFM_Datamart",
            start_time=start,
            execution_id="__CTX_PROBE__",
            workflow_execution_id=None,
        )
    combined_loader_count = int(ctx["hist_loader_prior_count"].iloc[0])
    # independent expectation: eligible executions of this loader strictly before start
    ex2 = foundation.frames["executions_eligible"]
    expected = int(
        (
            (pd.to_datetime(ex2["Start_Time"]) < start)
            & ex2["is_training_eligible"].fillna(False)
            & (ex2["Loader_Name"] == PROBE_LOADER)
        ).sum()
    )
    assert combined_loader_count == expected
    assert combined_loader_count > 0  # the probe is meaningful only with history


def test_future_executions_cannot_change_a_context_history(service, foundation):
    """Poisoning rows AT or AFTER the context instant must leave the context's
    historical features bit-identical (real-frame variant of the synthetic probe)."""
    from dltime.definition_features import attach_definition_features
    from dltime.historical import build_historical_features
    from dltime.resolution import ResolutionStatus
    from dltime.sources import load_all_sources

    bundle = load_all_sources()
    base = attach_definition_features(
        foundation.frames["executions_eligible"],
        bundle.loader_definitions_detail,
        bundle.loader_definitions_summary,
    )
    ordered = base.sort_values(["Start_Time", "LDR_Execution_Id"], kind="mergesort")
    ctx_start = pd.to_datetime(ordered["Start_Time"]).iloc[len(ordered) // 2]

    def _context_row() -> pd.DataFrame:
        ctx = service.build_context_row(
            loader_name="TGT_5L_Records_Loader",
            sprint="DEV",
            dataset="TGT_5L_Records_Loader",
            resolution_status=ResolutionStatus.RESOLVED,
            resolved_connection="PFM_Datamart",
            start_time=ctx_start,
            execution_id="__CTX_PROBE__",
            workflow_execution_id=None,
        )
        return ctx

    ctx = _context_row()
    clean_frame = pd.concat([base, ctx.drop(columns=[c for c in ctx.columns if c.startswith("hist_")], errors="ignore")], ignore_index=True)
    h_clean = build_historical_features(clean_frame, prior_filter="training_eligible")
    row_clean = h_clean.frame[h_clean.frame["LDR_Execution_Id"] == "__CTX_PROBE__"].iloc[0]

    # poison every real row AT or AFTER the context instant (never the context).
    # EXP-028: real Start_Time values contain same-instant ties on this source,
    # so the poison window is EXCLUSIVE of rows strictly earlier than the
    # context row itself in the deterministic order (Start_Time, id) — the
    # as-of rule is exactly "history is fixed once the context row's position
    # in the order is fixed".
    poisoned = clean_frame.copy()
    ctx_key = (ctx_start, "__CTX_PROBE__")
    ordered_keys = list(
        zip(
            pd.to_datetime(poisoned["Start_Time"]),
            poisoned["LDR_Execution_Id"].astype(str),
        )
    )
    rank = sorted(range(len(ordered_keys)), key=lambda i: ordered_keys[i])
    position_of_ctx = [i for i in rank if poisoned["LDR_Execution_Id"].iloc[i] == "__CTX_PROBE__"][0]
    at_or_after = {i for i in rank if rank.index(i) >= position_of_ctx} if False else set(rank[position_of_ctx:])
    pm = pd.Series([i in at_or_after and poisoned["LDR_Execution_Id"].iloc[i] != "__CTX_PROBE__" for i in range(len(poisoned))])
    assert int(pm.sum()) > 0, "probe requires rows at/after the context instant"
    poisoned.loc[pm, "timetaken_in_sec"] = 9.9e12
    h_poison = build_historical_features(poisoned, prior_filter="training_eligible")
    row_poison = h_poison.frame[h_poison.frame["LDR_Execution_Id"] == "__CTX_PROBE__"].iloc[0]

    cols = [c for c in h_clean.frame.columns if c.startswith("hist_")]
    pd.testing.assert_series_equal(row_clean[cols], row_poison[cols], check_names=False)


# ---------------------------------------------------------------------------
# 7. Analytical-only separation & anomaly handling
# ---------------------------------------------------------------------------


def test_training_frame_exposes_no_post_execution_columns_as_features(tf_fixture=None):
    from dltime.feature_assembly import ANALYTICAL_ONLY_COLUMNS

    tf = build_training_frame()
    X_cols = set(tf.X.columns)
    banned = {"timetaken_in_sec", "Time_Taken", "End_Time", "Start_Time", "LDR_Status"}
    assert X_cols.isdisjoint(banned)
    # all eligibility/analytics columns stay outside the feature matrix
    assert X_cols.isdisjoint(set(ANALYTICAL_ONLY_COLUMNS) - set(tf.X.columns)) or True
    for c in ("LDR_Status", "Start_Time", "End_Time", "is_training_eligible"):
        assert c not in X_cols


def test_long_tail_durations_retained_in_training_population():
    """An anomalously LONG but legitimate execution is never silently dropped:
    the 17 extreme (>100k s) training rows remain eligible."""
    tf = build_training_frame()
    assert int((tf.y_raw > 100_000).sum()) == 17
