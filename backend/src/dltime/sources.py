"""Read-only loading of the three real source Excel files.

The raw files under train/data are the source of truth and are NEVER modified.
Every load validates the expected schema explicitly and fails with a clear
error if required columns are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from dltime.config import DATA_DIR


class SourceSchemaError(RuntimeError):
    """Raised when a source file is missing or does not match its expected schema."""


class SourceRole:
    """Logical roles of the source files (see ARCHITECTURE_DECISIONS.md, AD-02)."""

    PRE_EXECUTION_DEFINITION_DETAIL = "PRE_EXECUTION_DEFINITION_DETAIL"
    PRE_EXECUTION_DEFINITION_SUMMARY = "PRE_EXECUTION_DEFINITION_SUMMARY"
    POST_EXECUTION_HISTORY = "POST_EXECUTION_HISTORY"


@dataclass(frozen=True)
class SourceSpec:
    """Machine-readable specification of one source file."""

    name: str
    filename: str
    role: str
    required_columns: tuple[str, ...]
    all_columns: tuple[str, ...]  # empty tuple = no full-column assertion

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


SOURCE_SPECS: dict[str, SourceSpec] = {
    "loader_definitions_detail": SourceSpec(
        name="loader_definitions_detail",
        filename="LOADER_DEFINATIONS.xlsx",
        role=SourceRole.PRE_EXECUTION_DEFINITION_DETAIL,
        required_columns=(
            "sprint_name",
            "ldr_connection_name",
            "loader_display_name",
            "datasetname",
            "preval_trans_display_name",
            "type",
            "calling_type",
            "execution_order",
        ),
        all_columns=(),
    ),
    "loader_definitions_summary": SourceSpec(
        name="loader_definitions_summary",
        filename="LOADER_DEFINATIONS_1.xlsx",
        role=SourceRole.PRE_EXECUTION_DEFINITION_SUMMARY,
        required_columns=(
            "sprint_name",
            "ldr_connection_name",
            "loader_display_name",
            "datasetname",
            "prevalidation_count",
            "transformation_count",
        ),
        all_columns=(),
    ),
    "executions": SourceSpec(
        name="executions",
        filename="Loader_Executions.xlsx",
        role=SourceRole.POST_EXECUTION_HISTORY,
        required_columns=(
            "LDR_Execution_Id",
            "Loader_Name",
            "LDR_Status",
            "Sprint",
            "Dataset_display_name",
            "ldr_workflow_execution_id",
            "Start_Time",
            "End_Time",
            "timetaken_in_sec",
            "Total_Records",
            "Success_Records",
            "Error_Records",
        ),
        all_columns=(),
    ),
}


def _validate(spec: SourceSpec, df: pd.DataFrame) -> None:
    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        raise SourceSchemaError(
            f"Source '{spec.name}' ({spec.path.name}) is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    if spec.all_columns:
        extra = [c for c in df.columns if c not in spec.all_columns]
        if extra:
            raise SourceSchemaError(
                f"Source '{spec.name}' ({spec.path.name}) has unexpected extra columns: {extra}. "
                f"Expected exactly: {list(spec.all_columns)}"
            )


def _read_source(spec: SourceSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise SourceSchemaError(f"Source file not found: {spec.path}")
    df = pd.read_excel(spec.path)
    _validate(spec, df)
    return df


@dataclass(frozen=True)
class SourceBundle:
    """All three validated source tables plus provenance metadata."""

    loader_definitions_detail: pd.DataFrame
    loader_definitions_summary: pd.DataFrame
    executions: pd.DataFrame
    provenance: dict[str, dict[str, object]]


def _provenance(spec: SourceSpec, df: pd.DataFrame) -> dict[str, object]:
    digest = sha256(spec.path.read_bytes()).hexdigest()
    return {
        "file": spec.filename,
        "role": spec.role,
        "path": str(spec.path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256": digest,
    }


def load_all_sources(data_dir: Path | None = None) -> SourceBundle:
    """Load all three source files read-only with schema validation.

    Args:
        data_dir: optional override of the data directory (used by tests).

    Returns:
        SourceBundle with the three raw DataFrames (unmodified except dtype
        inference done by the Excel reader) and per-file provenance.
    """
    if data_dir is not None:
        # Rebind DATA_DIR for this load without mutating global state for other callers.
        import dltime.config as cfg

        original = cfg.DATA_DIR
        try:
            cfg.DATA_DIR = data_dir
            return _load_all()
        finally:
            cfg.DATA_DIR = original
    return _load_all()


def _load_all() -> SourceBundle:
    detail = _read_source(SOURCE_SPECS["loader_definitions_detail"])
    summary = _read_source(SOURCE_SPECS["loader_definitions_summary"])
    executions = _read_source(SOURCE_SPECS["executions"])
    provenance = {
        spec.name: _provenance(spec, df)
        for spec, df in (
            (SOURCE_SPECS["loader_definitions_detail"], detail),
            (SOURCE_SPECS["loader_definitions_summary"], summary),
            (SOURCE_SPECS["executions"], executions),
        )
    }
    return SourceBundle(
        loader_definitions_detail=detail,
        loader_definitions_summary=summary,
        executions=executions,
        provenance=provenance,
    )
