"""EXP-035: product input-contract verification probe.

Verifies (§6 of the productization prompt) that every legitimate user-facing
pre-execution input actually reaches the derived feature representation, and
quantifies (§7) how much of the served configuration representation depends on
the original configuration workbooks vs the user's own inputs.

Run with the stack up: uv run python scripts/exp035_input_contract_probe.py
"""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"

IDENTITY = {
    "loader_name": "test_acs_d_test",
    "sprint": "DEV",
    "dataset": "test_acs_d_test",
    "connection": "PFM_Datamart",
}


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def features(body: dict) -> dict:
    r = post("/predict", body)
    assert r.get("state") == "PREDICTED", r.get("state")
    return r.get("features") or {}


def main() -> None:
    base = features(dict(IDENTITY))
    probes = {
        "E calling type (echo)": ({"calling_type": "PREVALIDATION"},
            ["def_dominant_calling_type"]),
        "A prevalidation display name": ({"prevalidation_display_name": "PrimaryFlag"},
            ["def_preval_name_component"]),
        "B transformation display name": ({"transformation_display_name": "after"},
            ["def_trans_name_component"]),
        "C execution order": ({"execution_order": 9},
            ["def_execution_order_max", "def_execution_order_mean"]),
        "D prevalidation count": ({"prevalidation_count": 11},
            ["def_prevalidation_count"]),
        "D2 transformation count": ({"transformation_count": 7},
            ["def_transformation_count"]),
        "E2 type": ({"loader_type": "CrossReference"},
            ["def_type_nunique"]),
    }

    print("== input -> derived feature delta probe (§6) ==")
    failures = 0
    for label, (overrides, expect_cols) in probes.items():
        f = features({**IDENTITY, **overrides})
        changed = {c: (base.get(c), f.get(c)) for c in expect_cols if base.get(c) != f.get(c)}
        untouched_ok = all(
            f.get(c) == base.get(c)
            for c in (
                "def_preval_name_component", "def_trans_name_component",
                "def_execution_order_max", "def_prevalidation_count",
                "def_transformation_count", "def_dominant_calling_type", "def_type_nunique",
            )
            if c not in expect_cols
        )
        # ETA may change when the override maps to a fitted feature (counts,
        # order, calling type); markers alone must not move it (AD-35 test).
        eta = post("/predict", {**IDENTITY, **overrides}).get("predicted_seconds")
        ok = bool(changed) and untouched_ok
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {label}: changed={changed} untouched_ok={untouched_ok} eta={eta}")

    # §7: prediction must not silently substitute training dataset for user input.
    # The identity keys are still needed for RESOLUTION + canonical history
    # (documented product contract), but every user-supplied configuration
    # value must appear verbatim in the derived representation.
    print("\n== user input echo verification (§7) ==")
    r = post("/predict", {**IDENTITY,
        "prevalidation_display_name": "MY_OWN_COMPONENT",
        "transformation_display_name": "MY_TRANS_01",
        "execution_order": 4,
        "prevalidation_count": 3,
        "transformation_count": 5,
        "calling_type": "PREVALIDATION",
        "loader_type": "Dataset"})
    f = r.get("features") or {}
    ident = r.get("identity") or {}
    supplied = (ident.get("supplied_configuration") or {})
    checks = [
        ("prev name echoed", f.get("def_preval_name_component") == "MY_OWN_COMPONENT"),
        ("trans name echoed", f.get("def_trans_name_component") == "MY_TRANS_01"),
        ("order applied", f.get("def_execution_order_max") == 4.0),
        ("prev count applied", f.get("def_prevalidation_count") == 3.0),
        ("trans count applied", f.get("def_transformation_count") == 5.0),
        ("calling type applied", f.get("def_dominant_calling_type") == "PREVALIDATION"),
        ("supplied_configuration echo", supplied.get("prevalidation_display_name") == "MY_OWN_COMPONENT"),
    ]
    for label, ok in checks:
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {label}")

    print(f"\nfailures: {failures}")
    print("DONE")


if __name__ == "__main__":
    main()
