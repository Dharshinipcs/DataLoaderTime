"""EXP-034b: resolved-configuration applicability audit.

For each canonical configuration (loader+sprint+dataset), produce the
applicable component manifest (per calling_type: n components, distinct
display names, types, execution-order range) from the REAL definitions.
This is the evidence base for the Studio applicability display and for
the realistic acceptance scenarios (Cases A-J).
"""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

DEF = "train/data/LOADER_DEFINATIONS.xlsx"
DEF_SUM = "train/data/LOADER_DEFINATIONS_1.xlsx"


def config_manifest(detail: pd.DataFrame, summary: pd.DataFrame, loader: str, sprint: str, dataset: str) -> dict | None:
    d = detail[
        (detail["loader_display_name"].astype(str).str.strip() == loader)
        & (detail["sprint_name"].astype(str).str.strip() == sprint)
        & (detail["datasetname"].astype(str).str.strip() == dataset)
    ]
    if d.empty:
        return None
    s = summary[
        (summary["loader_display_name"].astype(str).str.strip() == loader)
        & (summary["sprint_name"].astype(str).str.strip() == sprint)
        & (summary["datasetname"].astype(str).str.strip() == dataset)
    ]
    manifest: dict = {"components": {}, "summary": {}}
    for ct, g in d.groupby("calling_type"):
        g = g.sort_values("execution_order")
        manifest["components"][str(ct)] = {
            "n_rows": int(len(g)),
            "distinct_display_names": sorted(g["preval_trans_display_name"].dropna().astype(str).unique().tolist())[:8],
            "n_distinct_display_names": int(g["preval_trans_display_name"].nunique()),
            "types": sorted(g["type"].dropna().astype(str).unique().tolist())[:6],
            "execution_orders": [float(x) for x in pd.to_numeric(g["execution_order"], errors="coerce").dropna().tolist()[:8]],
        }
    for _, r in s.iterrows():
        manifest["summary"] = {
            "prevalidation_count": None if pd.isna(r.get("prevalidation_count")) else float(r["prevalidation_count"]),
            "transformation_count": None if pd.isna(r.get("transformation_count")) else float(r["transformation_count"]),
        }
    return manifest


def main() -> None:
    d1 = pd.read_excel(DEF)
    d2 = pd.read_excel(DEF_SUM)
    configs = d2.drop_duplicates(subset=["loader_display_name", "sprint_name", "datasetname"])

    # realistic acceptance scenarios from REAL data
    scenarios = [
        ("A: staging+prevalidation+transformation (multi-component)", "Asset Master AS91 HANA S20", "DEV", "CUMULALEVALUE"),
        ("G: both prev+trans (explicit multi)", "test_acs_d_test", "DEV", None),
        ("E: many prevalidation components", "Customer Master", "DEV", "CUSTOMER_PROFILE_DEMO"),
        ("F: multiple transformation components", "2210_TEST_ALL_CONFIG", "SUPPORT", "A_TABLE"),
        ("H: known loader w/ history", "MigrationSummarySubTableLoaderDataset", "TEST", "Migration_Summary_Sub_Fact_Table_LDR_Datase_new"),
    ]
    print("== resolved configuration manifests (real definitions) ==")
    for label, loader, sprint, dataset in scenarios:
        if dataset is None:
            row = d2[(d2["loader_display_name"].astype(str).str.strip() == loader) & (d2["sprint_name"].astype(str).str.strip() == sprint)]
            if row.empty:
                print(f"{label}: NO CONFIG FOUND for {loader}")
                continue
            dataset = str(row.iloc[0]["datasetname"])
        m = config_manifest(d1, d2, loader, sprint, dataset)
        if m is None:
            print(f"{label}: config not found in detail ({loader}/{sprint}/{dataset})")
            continue
        print(f"\n{label}\n  config: {loader} / {sprint} / {dataset}")
        for ct, info in m["components"].items():
            print(f"  {ct}: n_rows={info['n_rows']} distinct_names={info['n_distinct_display_names']} sample={info['distinct_display_names'][:4]} types={info['types'][:3]} orders={info['execution_orders'][:6]}")
        print(f"  summary counts: {m['summary']}")

    # how many configs have NO transformation at all (Case C) and NO prevalidation?
    card = d1.groupby(["loader_display_name", "sprint_name", "datasetname", "calling_type"]).size().unstack(fill_value=0)
    only_prev = ((card.get("PREVALIDATION", pd.Series(0, index=card.index)) > 0) & (card.get("TRANSFORMATION", pd.Series(0, index=card.index)) == 0)).sum()
    only_trans = ((card.get("TRANSFORMATION", pd.Series(0, index=card.index)) > 0) & (card.get("PREVALIDATION", pd.Series(0, index=card.index)) == 0)).sum()
    print(f"\nconfigs with PREVALIDATION only: {only_prev}/{len(card)}")
    print(f"configs with TRANSFORMATION only: {only_trans}/{len(card)}")

    print("\nDONE")


if __name__ == "__main__":
    main()
