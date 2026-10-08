"""Part D: connection resolution tests against the locked validation facts."""

from __future__ import annotations

import pandas as pd

from dltime.resolution import (
    ConnectionResolver,
    ResolutionStatus,
    annotate_raw_rows,
)


def test_resolver_builds_from_both_definition_sources(bundle):
    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    stats = resolver.stats()
    assert stats["definition_keys"] > 0
    # No definition key may carry conflicting connections (0 ambiguous fact).
    assert resolver.conflicts == {}


def test_raw_row_resolution_matches_locked_facts(foundation):
    counts = foundation.resolution_raw_rows
    assert counts.get("RESOLVED") == 383
    assert counts.get("UNRESOLVED") == 409
    assert counts.get("AMBIGUOUS", 0) == 0


def test_execution_key_resolution_matches_locked_facts(foundation):
    counts = foundation.resolution_execution_keys
    # 2026-09-29 replacement: keys incl. missing-component keys (dropna=False).
    assert counts.get("RESOLVED") == 44
    assert counts.get("UNRESOLVED") == 94
    assert counts.get("AMBIGUOUS", 0) == 0


def test_execution_level_resolution_counts(foundation):
    counts = foundation.resolution_executions
    assert counts.get("RESOLVED") == 275
    assert counts.get("UNRESOLVED") == 268
    assert counts.get("AMBIGUOUS", 0) == 0


def test_missing_identity_component_is_unresolved(bundle):
    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    status, conn = resolver.resolve("SOME_DATASET", None, "SOME_LOADER")
    assert status is ResolutionStatus.UNRESOLVED
    assert conn is None


def test_zero_matches_unresolved_one_match_resolved(bundle):
    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    status, conn = resolver.resolve("NOT_A_REAL_DATASET_XYZ", "Sprint", "Loader")
    assert status is ResolutionStatus.UNRESOLVED and conn is None

    # take any real key from the definitions
    d = bundle.loader_definitions_detail.iloc[0]
    status, conn = resolver.resolve(d["datasetname"], d["sprint_name"], d["loader_display_name"])
    assert status is ResolutionStatus.RESOLVED
    assert conn == str(d["ldr_connection_name"]).strip()


def test_ambiguous_never_silently_chosen(bundle):
    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    crafted = pd.DataFrame(
        {
            "datasetname": ["AMBIG_DS", "AMBIG_DS"],
            "sprint_name": ["SPRINT_X", "SPRINT_X"],
            "loader_display_name": ["AMBIG_LOADER", "AMBIG_LOADER"],
            "ldr_connection_name": ["CONN_A", "CONN_B"],
        }
    )
    amb_resolver = ConnectionResolver(
        crafted, bundle.loader_definitions_summary
    )
    status, conn = amb_resolver.resolve("AMBIG_DS", "SPRINT_X", "AMBIG_LOADER")
    assert status is ResolutionStatus.AMBIGUOUS
    assert conn is None  # never a guessed connection


def test_annotation_adds_status_and_connection_columns(foundation):
    raw = foundation.frames["raw_annotated"]
    assert "resolution_status" in raw.columns
    assert "resolved_connection" in raw.columns
    assert raw["resolved_connection"].notna().sum() == 383  # only resolved rows carry a connection
    assert (raw.loc[raw["resolution_status"] != "RESOLVED", "resolved_connection"].isna()).all()
