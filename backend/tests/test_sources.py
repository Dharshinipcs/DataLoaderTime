"""Part B: source loading + schema validation against the real files."""

from __future__ import annotations

import pandas as pd

from dltime.sources import (
    SOURCE_SPECS,
    SourceSchemaError,
    SourceRole,
    load_all_sources,
)


def test_three_sources_load_from_real_files(bundle):
    assert set(bundle.provenance) == {
        "loader_definitions_detail",
        "loader_definitions_summary",
        "executions",
    }
    for name, prov in bundle.provenance.items():
        assert prov["rows"] > 0
        assert prov["columns"] > 0
        assert len(prov["sha256"]) == 64


def test_source_roles_are_declared(bundle):
    assert bundle.provenance["loader_definitions_detail"]["role"] == SourceRole.PRE_EXECUTION_DEFINITION_DETAIL
    assert bundle.provenance["loader_definitions_summary"]["role"] == SourceRole.PRE_EXECUTION_DEFINITION_SUMMARY
    assert bundle.provenance["executions"]["role"] == SourceRole.POST_EXECUTION_HISTORY


def test_known_shapes_match_real_data(bundle):
    # Locked facts revalidated against reality (not hard-coded into the code paths).
    assert bundle.provenance["loader_definitions_detail"]["rows"] == 1480
    assert bundle.provenance["loader_definitions_detail"]["columns"] == 8
    assert bundle.provenance["loader_definitions_summary"]["rows"] == 278
    assert bundle.provenance["loader_definitions_summary"]["columns"] == 6
    # AD-31 corrected executions source; 2026-09-30 merge adds the real
    # staging timing fields (staging_start_time, staging_end_time,
    # staging_timetaken_in_min) by LDR_Execution_Id -> 42 columns.
    assert bundle.provenance["executions"]["rows"] == 792
    assert bundle.provenance["executions"]["columns"] == 42


def test_missing_required_column_fails_clearly(bundle, tmp_path):
    broken = bundle.executions.drop(columns=["LDR_Execution_Id"])
    (tmp_path / "Loader_Executions.xlsx").write_bytes(b"broken")
    # Schema validation is column-based; simulate by asserting the error path
    # through the public validator rather than writing a corrupt Excel file.
    from dltime.sources import _validate

    spec = SOURCE_SPECS["executions"]
    try:
        _validate(spec, broken)
    except SourceSchemaError as exc:
        assert "LDR_Execution_Id" in str(exc)
    else:
        raise AssertionError("Expected SourceSchemaError for missing LDR_Execution_Id")


def test_all_declared_columns_exist_in_real_files(bundle):
    for spec in SOURCE_SPECS.values():
        df = getattr(
            bundle,
            {
                "loader_definitions_detail": "loader_definitions_detail",
                "loader_definitions_summary": "loader_definitions_summary",
                "executions": "executions",
            }[spec.name],
        )
        for col in spec.required_columns:
            assert col in df.columns, f"{spec.name} missing declared column {col}"


def test_read_only_guarantee_smoke(bundle):
    # Loading twice returns equal content (no in-place mutation of loaded frames).
    again = load_all_sources()
    pd.testing.assert_frame_equal(bundle.executions, again.executions)
    pd.testing.assert_frame_equal(
        bundle.loader_definitions_detail, again.loader_definitions_detail
    )
