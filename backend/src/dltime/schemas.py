"""Leakage classification registry (locked rule, ARCHITECTURE_DECISIONS.md AD-07).

Every field consumed downstream must be classifiable into exactly one of the
categories below. The classification is declared once, here, and is enforced
by tests plus the runtime guards in leakage.py.

Categories
----------
PRE_EXECUTION                  known before the current execution starts
POST_EXECUTION                 only known after the current execution finishes
DERIVED_FROM_PRE_EXECUTION     computed exclusively from pre-execution inputs
DERIVED_FROM_POST_EXECUTION    computed from history (allowed only from PRIOR
                               executions at prediction time)
TARGET                         the training label; never a prediction input
IDENTIFIER                     identifies records; identity components may be
                               used for lookup/joins, never as model features
                               without an explicit locked decision
ANALYTICAL_ONLY                reporting/diagnostics; must never feed features
"""

from __future__ import annotations

from dataclasses import dataclass, field


class LeakageCategory:
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_EXECUTION = "POST_EXECUTION"
    DERIVED_FROM_PRE_EXECUTION = "DERIVED_FROM_PRE_EXECUTION"
    DERIVED_FROM_POST_EXECUTION = "DERIVED_FROM_POST_EXECUTION"
    TARGET = "TARGET"
    IDENTIFIER = "IDENTIFIER"
    ANALYTICAL_ONLY = "ANALYTICAL_ONLY"


ALL_CATEGORIES = frozenset(
    {
        LeakageCategory.PRE_EXECUTION,
        LeakageCategory.POST_EXECUTION,
        LeakageCategory.DERIVED_FROM_PRE_EXECUTION,
        LeakageCategory.DERIVED_FROM_POST_EXECUTION,
        LeakageCategory.TARGET,
        LeakageCategory.IDENTIFIER,
        LeakageCategory.ANALYTICAL_ONLY,
    }
)

# Categories that may legitimately enter a pre-execution feature set.
ALLOWED_FOR_PREDICTION = frozenset(
    {
        LeakageCategory.PRE_EXECUTION,
        LeakageCategory.DERIVED_FROM_PRE_EXECUTION,
        LeakageCategory.DERIVED_FROM_POST_EXECUTION,  # historical features only (prior executions)
    }
)


class RegistryError(KeyError):
    """Raised when a field has no registered leakage classification."""


@dataclass(frozen=True)
class FieldClassification:
    name: str
    category: str
    note: str = ""
    source_role: str = ""  # e.g. POST_EXECUTION_HISTORY
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in ALL_CATEGORIES:
            raise ValueError(f"Unknown leakage category '{self.category}' for field '{self.name}'")


class FieldRegistry:
    """Single authoritative classification table for raw source fields."""

    def __init__(self) -> None:
        self._fields: dict[str, FieldClassification] = {}

    def register(self, fc: FieldClassification) -> None:
        existing = self._fields.get(fc.name)
        if existing is not None:
            if existing.category != fc.category:
                raise ValueError(
                    f"Conflicting classification for field '{fc.name}': "
                    f"{existing.category} vs {fc.category}"
                )
            # Same category from another source: keep the first classification.
            return
        self._fields[fc.name] = fc

    def get(self, name: str) -> FieldClassification:
        if name not in self._fields:
            raise RegistryError(
                f"Field '{name}' is not classified. Every downstream field must be "
                "registered in dltime.schemas (leakage boundary, AD-07)."
            )
        return self._fields[name]

    def category_of(self, name: str) -> str:
        return self.get(name).category

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def items(self):
        return sorted(self._fields.items(), key=lambda kv: (kv[1].category, kv[0]))

    def __len__(self) -> int:
        return len(self._fields)


def build_default_registry() -> FieldRegistry:
    """Classify every raw field of the three sources from the real schemas."""
    reg = FieldRegistry()
    HIST = "POST_EXECUTION_HISTORY"
    DEF = "PRE_EXECUTION_DEFINITION_DETAIL"
    DEFS = "PRE_EXECUTION_DEFINITION_SUMMARY"

    # ---- Execution history (Loader_Executions.xlsx) -------------------------
    hist = {
        "LDR_Execution_Id": LeakageCategory.IDENTIFIER,
        "ldr_workflow_execution_id": LeakageCategory.IDENTIFIER,
        "Loader_Name": LeakageCategory.IDENTIFIER,  # identity component (lookup/join)
        "Sprint": LeakageCategory.IDENTIFIER,  # identity component
        # 2026-09-29 source replacement: Connection_Name joins Sprint/Loader/
        # Dataset as the fourth identity component of the 4-key.
        "Connection_Name": LeakageCategory.IDENTIFIER,  # identity component
        "Dataset_display_name": LeakageCategory.IDENTIFIER,  # identity component
        # Execution outcome & timing: only known after execution finishes.
        "LDR_Status": LeakageCategory.POST_EXECUTION,
        "Start_Time": LeakageCategory.POST_EXECUTION,
        "End_Time": LeakageCategory.POST_EXECUTION,
        "Time_Taken": LeakageCategory.POST_EXECUTION,
        "timetaken_in_sec": LeakageCategory.TARGET,
        # Record/error counters are outcomes of the run.
        "Total_Records": LeakageCategory.POST_EXECUTION,
        "Success_Records": LeakageCategory.POST_EXECUTION,
        "Error_Records": LeakageCategory.POST_EXECUTION,
        "Prevalidation_Errors": LeakageCategory.POST_EXECUTION,
        "Format_Errors": LeakageCategory.POST_EXECUTION,
        "Constraint_Errors": LeakageCategory.POST_EXECUTION,
        "Application_Errors": LeakageCategory.POST_EXECUTION,
        "Cascade_Errors": LeakageCategory.POST_EXECUTION,
        # Stage-level observations: measured during/after the run.
        "prevalidation_start_time": LeakageCategory.POST_EXECUTION,
        "prevalidation_end_time": LeakageCategory.POST_EXECUTION,
        "prevaliadtion_timetaken": LeakageCategory.POST_EXECUTION,
        "prevaliadtion_timetaken_in_sec": LeakageCategory.POST_EXECUTION,
        "transformation_start_time": LeakageCategory.POST_EXECUTION,
        "transformation_end_time": LeakageCategory.POST_EXECUTION,
        "transformation_timetaken": LeakageCategory.POST_EXECUTION,
        "transformation_timetaken_in_sec": LeakageCategory.POST_EXECUTION,
        "pre_load_start_time": LeakageCategory.POST_EXECUTION,
        "pre_load_end_time": LeakageCategory.POST_EXECUTION,
        "pre_load_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "staging_start_time": LeakageCategory.POST_EXECUTION,
        "staging_end_time": LeakageCategory.POST_EXECUTION,
        "staging_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "prevalidation_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "transformation_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "data_loading_timetaken": LeakageCategory.POST_EXECUTION,
        "data_loading_timetaken_in_sec": LeakageCategory.POST_EXECUTION,
        "data_loading_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "data_loading_start_time": LeakageCategory.POST_EXECUTION,
        "data_loading_end_time": LeakageCategory.POST_EXECUTION,
        "data_loading_timetaken_1": LeakageCategory.POST_EXECUTION,
        "data_loading_timetaken_in_sec_1": LeakageCategory.POST_EXECUTION,
        "datamart_approval_timetaken": LeakageCategory.POST_EXECUTION,
        "datamart_approval_timetaken_in_sec": LeakageCategory.POST_EXECUTION,
        "datamart_approval_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "dataloading_approval_timetaken": LeakageCategory.POST_EXECUTION,
        "dataloading_approval_timetaken_in_sec": LeakageCategory.POST_EXECUTION,
        "dataloading_approval_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        # AD-31 corrected-source total-target chain (EXP-028 audit): the legacy
        # per-component timetaken_in_sec column is provably inconsistent with
        # the Start/End span; the authoritative total target is
        # total_timetaken_in_min * 60 (validated against the span).
        "total_timetaken_in_min": LeakageCategory.POST_EXECUTION,
        "target_source": LeakageCategory.POST_EXECUTION,
        "target_span_rel_diff": LeakageCategory.POST_EXECUTION,
        # Stray spreadsheet artifacts present in the corrected source (dropped
        # by the reconstruct policy layer; classified for registry totality).
        "Unnamed: 42": LeakageCategory.POST_EXECUTION,
        "Unnamed: 43": LeakageCategory.POST_EXECUTION,
    }
    for name, cat in hist.items():
        reg.register(FieldClassification(name, cat, source_role=HIST))

    # ---- Definition detail (LOADER_DEFINATIONS.xlsx) ------------------------
    for name in (
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
    ):
        reg.register(FieldClassification(name, LeakageCategory.PRE_EXECUTION, source_role=DEF))
    for name in ("preval_trans_display_name", "type", "calling_type", "execution_order"):
        reg.register(
            FieldClassification(
                name,
                LeakageCategory.PRE_EXECUTION,
                note="loader design/configuration, fixed before execution",
                source_role=DEF,
            )
        )

    # ---- Definition summary (LOADER_DEFINATIONS_1.xlsx) ---------------------
    for name in (
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
    ):
        reg.register(FieldClassification(name, LeakageCategory.PRE_EXECUTION, source_role=DEFS))
    for name in ("prevalidation_count", "transformation_count"):
        reg.register(
            FieldClassification(
                name,
                LeakageCategory.PRE_EXECUTION,
                note="config counts known from the loader definition before execution",
                source_role=DEFS,
            )
        )

    return reg


DEFAULT_REGISTRY: FieldRegistry = build_default_registry()
