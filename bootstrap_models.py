"""First-boot bootstrap for Render free tier (no persistent disk).

Uses the SAME code paths as the UI: /retraining/train-candidate then
/models/promote. Idempotent — skips when a promoted model already exists.
Run:  python bootstrap_models.py   (after deps installed, before uvicorn)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from dltime.api import create_app  # noqa: E402


def main() -> int:
    client = TestClient(create_app())

    status = client.get("/models/status").json()
    models = (status.get("models") or []) if isinstance(status, dict) else []
    promoted = [
        m for m in models
        if str(m.get("lifecycle_state") or m.get("status") or "").upper() == "PROMOTED"
    ]
    if promoted:
        print(f"[bootstrap] promoted model exists ({promoted[0].get('model_id')}); nothing to do.")
        return 0

    print("[bootstrap] no promoted model — training full-bundle candidate ...")
    r = client.post(
        "/retraining/train-candidate",
        json={"reason": "First-boot bootstrap (Render free tier, no persistent disk)."},
    )
    if r.status_code != 200:
        print(f"[bootstrap] train-candidate failed: {r.status_code} {r.text[:300]}")
        return 1
    body = r.json()
    registered = body.get("registered")
    model_id = (
        registered[0].get("model_id") if isinstance(registered, list) else registered.get("model_id")
    )
    print(f"[bootstrap] candidate ready: {model_id}")

    p = client.post(
        "/models/promote",
        params={"model_id": model_id},
        json={"reason": "Auto-bootstrap promotion (first boot)."},
    )
    if p.status_code != 200:
        print(f"[bootstrap] promote failed: {p.status_code} {p.text[:300]}")
        return 1
    print(f"[bootstrap] promoted: {model_id}. App is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
