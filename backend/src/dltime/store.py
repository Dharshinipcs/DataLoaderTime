"""Append-only local JSONL persistence for prediction monitoring (Task 11).

A small repository abstraction over three append-only files under
``backend/store/``:

- ``prediction_events.jsonl``  one record per prediction attempt (all states)
- ``actual_outcomes.jsonl``    one record per ingested actual execution outcome
- ``feedback_records.jsonl``   prediction<->actual link + error + eligibility

Design rules (AD-24):
- APPEND-ONLY: records are never mutated or deleted in place. A linkage is a
  separate feedback record that REFERENCES a prediction event id; a duplicate
  submission is recognized by content hash and reported idempotently instead
  of being written twice.
- FAIL-SAFE: persistence failures never alter or fail the prediction itself
  (same isolation pattern as Task 07/08/09 enrichment layers).
- FAIL-CLOSED ON READ: a corrupt line is skipped with a captured warning, not
  silently treated as data.
- VERSIONED: every record carries ``schema_version``; the store rejects
  records of a different major schema on read (skip-with-warning).

The store holds no secrets and contains only derived application data; the
original Excel sources, training data, and model artifacts are never touched.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dltime.config import PROJECT_ROOT

STORE_DIR: Path = PROJECT_ROOT / "backend" / "store"
SCHEMA_VERSION = "t11-v1"
# Prompt 3 (§14/§15): prediction events must carry the caller linkage
# reference inside the persisted identity block so /monitoring/events is a
# complete audit view (previously the reference lived only in `request`).
# Linkage metadata ONLY — it never enters model features, the feature
# registry, or explanations (test-enforced).
LINKAGE_REFERENCE_FIELD = "ldr_workflow_execution_id"
SCHEMA_VERSION_MINOR = 1  # additive: identity carries the linkage reference

PREDICTION_EVENTS_FILE = "prediction_events.jsonl"
ACTUAL_OUTCOMES_FILE = "actual_outcomes.jsonl"
FEEDBACK_RECORDS_FILE = "feedback_records.jsonl"
# AD-33 (§21): persisted notification feed with read/unread bookkeeping.
# Append-mostly: individual occurrences are appended; marking read is the only
# rewrite and never drops records.
NOTIFICATIONS_FILE = "notifications.jsonl"


def utc_now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision (audit-friendly, sortable)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(payload: dict[str, Any]) -> str:
    """Stable content hash for idempotency/conflict checks (deterministic)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class StoreWarning:
    file: str
    line: int
    message: str


class MonitoringStore:
    """Append-only JSONL repository (thread-safe within this process)."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory) if directory is not None else resolve_store_dir()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- internals
    def _path(self, name: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name

    def _append(self, name: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self._path(name).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def _iter(self, name: str) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
        """Yield (line_number, record_or_None, warning_or_None); fail-closed on read."""
        path = self._path(name)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield i, None, f"skipped corrupt JSON line: {exc}"
                    continue
                if not isinstance(rec, dict):
                    yield i, None, "skipped non-object record"
                    continue
                if rec.get("schema_version") != SCHEMA_VERSION:
                    yield i, None, (
                        f"skipped record with foreign schema_version "
                        f"{rec.get('schema_version')!r}"
                    )
                    continue
                yield i, rec, None

    def _read_all(self, name: str) -> tuple[list[dict[str, Any]], list[StoreWarning]]:
        records: list[dict[str, Any]] = []
        warnings: list[StoreWarning] = []
        for line_no, rec, warn in self._iter(name):
            if rec is not None:
                records.append(rec)
            else:
                warnings.append(StoreWarning(name, line_no, warn or "unknown"))
        return records, warnings

    # ------------------------------------------------------ prediction events
    def append_prediction_event(self, event: dict[str, Any]) -> None:
        self._append(PREDICTION_EVENTS_FILE, event)

    def read_prediction_events(self) -> tuple[list[dict[str, Any]], list[StoreWarning]]:
        return self._read_all(PREDICTION_EVENTS_FILE)

    def find_prediction_event(self, event_id: str) -> dict[str, Any] | None:
        events, _warnings = self.read_prediction_events()
        for rec in events:
            if rec.get("event_id") == event_id:
                return rec
        return None

    # ------------------------------------------------------- actual outcomes
    def append_actual_outcome(self, outcome: dict[str, Any]) -> None:
        self._append(ACTUAL_OUTCOMES_FILE, outcome)

    def read_actual_outcomes(self) -> tuple[list[dict[str, Any]], list[StoreWarning]]:
        return self._read_all(ACTUAL_OUTCOMES_FILE)

    # ------------------------------------------------------ feedback records
    def append_feedback_record(self, record: dict[str, Any]) -> None:
        self._append(FEEDBACK_RECORDS_FILE, record)

    def read_feedback_records(self) -> tuple[list[dict[str, Any]], list[StoreWarning]]:
        return self._read_all(FEEDBACK_RECORDS_FILE)

    # ---------------------------------------------------------- notifications
    def append_notification(self, record: dict[str, Any]) -> None:
        self._append(NOTIFICATIONS_FILE, record)

    def read_notifications(self) -> tuple[list[dict[str, Any]], list[StoreWarning]]:
        return self._read_all(NOTIFICATIONS_FILE)

    def rewrite_notifications(self, records: list[dict[str, Any]]) -> None:
        """Persist read/unread transitions (the one sanctioned rewrite; append-only
        files are untouched). Records are never dropped."""
        import json as _json

        lines = [
            _json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
            for rec in records
        ]
        with self._lock:
            with self._path(NOTIFICATIONS_FILE).open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
                fh.flush()


def resolve_store_dir() -> Path:
    """Store location, resolved at call time (module-level so tests can redirect
    every instantiation hermetically; callers may also pass an explicit dir)."""
    return STORE_DIR


def store_ref() -> MonitoringStore:
    """Default process-wide store (backend/store)."""
    return MonitoringStore()
