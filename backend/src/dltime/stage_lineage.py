"""Stage performance lineage (append-only).

Every promote/rollback appends one snapshot per AFFECTED stage to
artifacts/registry/stage_performance_lineage.jsonl:

    {"stage": ..., "model_version": ..., "holdout_medae": ...,
     "holdout_n": ..., "recent_medae": ..., "role":
     "baseline"|"previous"|"current", "at": ...}

The lineage is the source for the Baseline / Previous / Current
columns of the Drift & Retraining comparison table. Role semantics:

  - "current": the model serving that stage right now.
  - "previous": the model that served the stage before the current
    one (demoted on promote, restored on rollback).
  - "baseline": the original chronological-holdout baseline the
    first serving model was measured against.

On promote: the stage's current snapshot is re-appended as
"previous" and the promoted candidate becomes "current". Rollback
reverses it. Nothing here ever promotes, rolls back, or serves —
the caller (retraining / api_lab) performs the actual action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dltime.config import PROJECT_ROOT

LINEAGE_SCHEMA_VERSION = "stage-lineage-v1"
LINEAGE_FILE = PROJECT_ROOT / "backend" / "artifacts" / "registry" / "stage_performance_lineage.jsonl"

ROLES = ("baseline", "previous", "current")
STAGE_NAMES = ["staging", "prevalidation", "transformation", "pre_load", "data_loading"]


def _ensure_file() -> Path:
    LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    return LINEAGE_FILE


def read_snapshots() -> list[dict[str, Any]]:
    """All lineage snapshots, oldest first (append-only read)."""
    if not LINEAGE_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in LINEAGE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # fail-closed on read
        if isinstance(rec, dict) and rec.get("schema_version") == LINEAGE_SCHEMA_VERSION:
            out.append(rec)
    return out


def _append(rec: dict[str, Any]) -> dict[str, Any]:
    _ensure_file().parent.mkdir(parents=True, exist_ok=True)
    with _ensure_file().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return rec


def stage_roles() -> dict[str, dict[str, dict[str, Any]]]:
    """Latest snapshot per (stage, role): {stage: {role: snapshot}}."""
    out: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in STAGE_NAMES}
    for rec in read_snapshots():
        stage = str(rec.get("stage", ""))
        role = str(rec.get("role", ""))
        if stage in out and role in ROLES:
            out[stage][role] = rec
    return out


def snapshot(
    *,
    stage: str,
    role: str,
    model_version: str,
    holdout_medae: float | None = None,
    holdout_n: int | None = None,
    recent_medae: float | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Append one lineage snapshot (explicit action only)."""
    from dltime.model_lab import utc_now_iso

    if role not in ROLES:
        raise ValueError(f"Unknown lineage role {role!r}.")
    rec = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "stage": str(stage),
        "role": role,
        "model_version": str(model_version),
        "holdout_medae": holdout_medae,
        "holdout_n": holdout_n,
        "recent_medae": recent_medae,
        "at": at or utc_now_iso(),
    }
    return _append(rec)


def record_promotion(
    *,
    stage: str,
    model_version: str,
    holdout_medae: float | None = None,
    holdout_n: int | None = None,
    recent_medae: float | None = None,
    at: str | None = None,
) -> list[dict[str, Any]]:
    """Lineage for a stage promotion: the serving model is demoted
    to 'previous' and the promoted candidate becomes 'current'."""
    written: list[dict[str, Any]] = []
    roles = stage_roles().get(stage, {})
    current = roles.get("current")
    if current and current.get("model_version") != model_version:
        written.append(
            snapshot(
                stage=stage,
                role="previous",
                model_version=current["model_version"],
                holdout_medae=current.get("holdout_medae"),
                holdout_n=current.get("holdout_n"),
                recent_medae=current.get("recent_medae"),
                at=at,
            )
        )
    written.append(
        snapshot(
            stage=stage,
            role="current",
            model_version=model_version,
            holdout_medae=holdout_medae,
            holdout_n=holdout_n,
            recent_medae=recent_medae,
            at=at,
        )
    )
    return written


def record_rollback(
    *,
    stage: str,
    model_version: str,
    holdout_medae: float | None = None,
    holdout_n: int | None = None,
    recent_medae: float | None = None,
    at: str | None = None,
) -> list[dict[str, Any]]:
    """Lineage for a stage rollback: the current model is demoted to
    'previous' and the restored model becomes 'current' again."""
    written: list[dict[str, Any]] = []
    roles = stage_roles().get(stage, {})
    current = roles.get("current")
    if current and current.get("model_version") != model_version:
        written.append(
            snapshot(
                stage=stage,
                role="previous",
                model_version=current["model_version"],
                holdout_medae=current.get("holdout_medae"),
                holdout_n=current.get("holdout_n"),
                recent_medae=current.get("recent_medae"),
                at=at,
            )
        )
    written.append(
        snapshot(
            stage=stage,
            role="current",
            model_version=model_version,
            holdout_medae=holdout_medae,
            holdout_n=holdout_n,
            recent_medae=recent_medae,
            at=at,
        )
    )
    return written


def seed(baseline_per_stage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Seed the lineage ONCE for today's serving: every stage gets a
    'baseline' snapshot (its chronological-holdout baseline) and a
    'current' snapshot (the model serving today). A no-op when the
    lineage already holds records."""
    if read_snapshots():
        return []
    from dltime.model_lab import utc_now_iso

    written: list[dict[str, Any]] = []
    for stage in STAGE_NAMES:
        base = baseline_per_stage.get(stage) or {}
        medae = base.get("median_ae")
        n = base.get("n")
        written.append(
            snapshot(
                stage=stage,
                role="baseline",
                model_version="stage-serving-v2 (stage-models-ad34)",
                holdout_medae=medae,
                holdout_n=n,
                at=utc_now_iso(),
            )
        )
        written.append(
            snapshot(
                stage=stage,
                role="current",
                model_version="stage-serving-v2 (stage-models-ad34)",
                holdout_medae=medae,
                holdout_n=n,
                at=utc_now_iso(),
            )
        )
    return written
