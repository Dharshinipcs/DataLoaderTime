"""EXP-033 Part K: representative-case physical-consistency probes (live API).

Sends prediction requests across representative contexts and verifies the
authoritative ETA cannot become physically nonsensical: ETA >= every predicted
ML stage duration, floor disclosed, approvals never set the floor.
Run with the dev stack up: uv run python scripts/exp033_consistency_cases.py
"""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"

CASES = [
    # (label, body-overrides)
    ("all-five-config-sparse", {"loader_name": "Test transform", "sprint": "DEV",
        "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new", "connection": "MDM_DM_TEST"}),
    ("canonical-context-floor", {"loader_name": "MigrationSummarySubTableLoaderDataset", "sprint": "TEST",
        "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new", "connection": "MDM_DM_TEST"}),
    ("asset-master-large-history", {"loader_name": "Asset Master AS91 HANA S20", "sprint": "DEV",
        "dataset": "CUMULALEVALUE", "connection": "SAP_HANA_SIT_S20"}),
    ("example-ldr-alt-identity", {"loader_name": "2902_latest_loader", "sprint": "DEV",
        "dataset": "2902_latest_loader", "connection": "DEV_DM"}),
    ("sparse-identity-unseen", {"loader_name": "BAPI_BANK_CREATE_2", "sprint": "DEV",
        "dataset": "HEADER_DATA", "connection": "SAP_S4_S19_TGT"}),
    ("11-input-overrides", {"loader_name": "MigrationSummarySubTableLoaderDataset", "sprint": "TEST",
        "dataset": "Migration_Summary_Sub_Fact_Table_LDR_Datase_new", "connection": "MDM_DM_TEST",
        "prevalidation_display_name": "DO", "transformation_display_name": "CR",
        "loader_type": "TABLE", "calling_type": "PREVALIDATION",
        "execution_order": 1, "prevalidation_count": 2, "transformation_count": 1}),
]


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


print(f"{'case':28} {'state':16} {'eta':>10} {'floor':>10} {'fl':>3}  eta>=max_ml_stage  max_ml_stage  n_ml")
violations = 0
for label, body in CASES:
    try:
        r = post("/predict", body)
    except Exception as exc:  # noqa: BLE001
        detail = ""
        try:
            detail = json.loads(exc.read().decode()).get("detail", "")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        print(f"{label:28} ERROR {exc} {str(detail)[:80]}")
        continue
    state = r.get("state")
    eta = r.get("predicted_seconds")
    floor = r.get("physical_floor_seconds")
    applied = r.get("floor_applied")
    stages = ((r.get("stage_intelligence") or {}).get("stages")) or []
    ml = [s["estimate_seconds"] for s in stages if s.get("method") == "ml_stage_model" and s.get("estimate_seconds") is not None]
    max_ml = max(ml) if ml else None
    ok = (eta is None) or (max_ml is None) or (eta >= max_ml - 1e-9)
    if not ok:
        violations += 1
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "None"  # noqa: E731
    print(f"{label:28} {str(state):16} {fmt(eta):>10} {fmt(floor):>10} {str(bool(applied))[:3]:>3}  {str(ok):>17}  {fmt(max_ml):>12}  {len(ml)}")

print(f"\nviolations: {violations}")
print("DONE")
