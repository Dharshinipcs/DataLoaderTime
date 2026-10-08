"""Connection resolution: canonical business identity -> connection.

Canonical identity (LOCKED): Dataset + Connection + Sprint + Loader Name.
The post-execution history has no connection column, so the connection is
resolved from the pre-execution definition sources using the other three
identity fields (Dataset + Sprint + Loader).

Resolution rules (LOCKED, never silently relaxed):
    0 matching definition records  -> UNRESOLVED
    1 matching definition record   -> RESOLVED
    >1 matching definition records -> AMBIGUOUS (never silently choose)

Normalization: string strip + case-sensitive comparison, matching the
completed investigation that produced the locked validation facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class ResolutionStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"


# Canonical identity field names used across the codebase (execution side).
CANONICAL_IDENTITY_FIELDS = ("dataset", "sprint", "loader")


@dataclass(frozen=True)
class IdentityKey:
    """A canonical business identity without connection (lookup key)."""

    dataset: str | None
    sprint: str | None
    loader: str | None

    def is_complete(self) -> bool:
        return all(v is not None and str(v).strip() != "" for v in (self.dataset, self.sprint, self.loader))


def _norm(value) -> str | None:
    """Normalize an identity value: NaN/None/blank -> None, else stripped string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None


class ConnectionResolver:
    """Resolves connection for identity keys from pre-execution definitions.

    Build once from the definition sources; query many times. Immutable after
    construction (frozen lookup tables), so resolution can never silently
    change behavior at query time.
    """

    def __init__(self, definitions_detail: pd.DataFrame, definitions_summary: pd.DataFrame) -> None:
        tables = [definitions_detail, definitions_summary]
        self._lookup: dict[tuple[str, str, str], set[str]] = {}
        self._records_per_key: dict[tuple[str, str, str], int] = {}
        self._conflicts: dict[tuple[str, str, str], set[str]] = {}

        for table in tables:
            required = {"datasetname", "sprint_name", "loader_display_name", "ldr_connection_name"}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"Definition table missing columns for resolution: {sorted(missing)}")
            for row in table.itertuples(index=False):
                key = (
                    _norm(row.datasetname),
                    _norm(row.sprint_name),
                    _norm(row.loader_display_name),
                )
                conn = _norm(row.ldr_connection_name)
                if None in key or conn is None:
                    continue  # definition rows without full identity cannot support resolution
                self._lookup.setdefault(key, set()).add(conn)
                self._records_per_key[key] = self._records_per_key.get(key, 0) + 1

        # A key whose definition records disagree on connection is ambiguous,
        # even though each individual record resolves to exactly one connection.
        for key, conns in self._lookup.items():
            if len(conns) > 1:
                self._conflicts[key] = conns

    def resolve(self, dataset, sprint, loader) -> tuple[ResolutionStatus, str | None]:
        """Resolve one identity to (status, connection_or_None). Never guesses."""
        key = (_norm(dataset), _norm(sprint), _norm(loader))
        if any(part is None for part in key):
            return ResolutionStatus.UNRESOLVED, None
        conns = self._lookup.get(key, set())
        if len(conns) == 1:
            return ResolutionStatus.RESOLVED, next(iter(conns))
        if len(conns) > 1:
            return ResolutionStatus.AMBIGUOUS, None
        return ResolutionStatus.UNRESOLVED, None

    def resolve_identity(self, identity: IdentityKey) -> tuple[ResolutionStatus, str | None]:
        return self.resolve(identity.dataset, identity.sprint, identity.loader)

    @property
    def conflicts(self) -> dict[tuple[str, str, str], set[str]]:
        return dict(self._conflicts)

    def stats(self) -> dict[str, int]:
        keys = len(self._lookup)
        return {
            "definition_keys": keys,
            "ambiguous_definition_keys": len(self._conflicts),
        }


def annotate_raw_rows(
    executions: pd.DataFrame,
    resolver: ConnectionResolver,
    *,
    dataset_col: str = "Dataset_display_name",
    sprint_col: str = "Sprint",
    loader_col: str = "Loader_Name",
) -> pd.DataFrame:
    """Return a copy of raw execution rows with connection-resolution columns.

    Adds:
        resolution_status: UNRESOLVED / RESOLVED / AMBIGUOUS (row level)
        resolved_connection: connection string when RESOLVED, else <NA>
    """
    out = executions.copy()
    statuses: list[str] = []
    connections: list[str | pd.NA] = []
    for d, s, l in zip(out[dataset_col], out[sprint_col], out[loader_col]):
        status, conn = resolver.resolve(d, s, l)
        statuses.append(status.value)
        connections.append(conn if conn is not None else pd.NA)
    out["resolution_status"] = pd.Series(statuses, index=out.index, dtype="string")
    out["resolved_connection"] = pd.Series(connections, index=out.index, dtype="string")
    return out


def summarize_resolution(counts: dict[str, int]) -> str:
    """Human-readable one-line summary of a resolution-count mapping."""
    total = sum(counts.values())
    return (
        f"RESOLVED={counts.get('RESOLVED', 0)} / UNRESOLVED={counts.get('UNRESOLVED', 0)} / "
        f"AMBIGUOUS={counts.get('AMBIGUOUS', 0)} (total {total})"
    )
