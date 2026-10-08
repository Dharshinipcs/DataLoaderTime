"""EXP-032 Part F: LOADER_DEFINATIONS.xlsx semantics audit (read-only).

Determines whether preval_trans_display_name is (A) mutually exclusive,
(B) a combined record carrying both, or (C) splittable via other columns.
"""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

D1 = pd.read_excel("train/data/LOADER_DEFINATIONS.xlsx")
D2 = pd.read_excel("train/data/LOADER_DEFINATIONS_1.xlsx")

print("=== detail schema ===")
print(list(D1.columns))
print(f"rows: {len(D1)}")
print("\n=== summary schema ===")
print(list(D2.columns))
print(f"rows: {len(D2)}")

# ---- column-level profile of the detail file
for col in D1.columns:
    nn = D1[col].notna().sum()
    nu = D1[col].nunique(dropna=True)
    print(f"\ncolumn {col!r}: non-null={nn} ({nn/len(D1):.1%}) unique={nu}")
    if nu <= 12:
        print("   values:", D1[col].value_counts(dropna=False).head(12).to_dict())
    else:
        print("   sample:", D1[col].dropna().astype(str).head(5).tolist())

# ---- coexistence analysis
print("\n=== coexistence analysis ===")
# Which columns indicate prevalidation vs transformation?
cand = [c for c in D1.columns if any(k in str(c).lower() for k in ("preval", "trans", "type", "call", "order", "count", "name"))]
print("candidate columns:", cand)

# Does any row carry BOTH prevalidation-ish and transformation-ish markers?
def has_transform(row) -> bool:
    for c in D1.columns:
        v = str(row.get(c, "")).lower()
        if "transform" in v and "preval" not in v:
            return True
    return False

def has_preval(row) -> bool:
    for c in D1.columns:
        v = str(row.get(c, "")).lower()
        if "preval" in v:
            return True
    return False

flags = D1.apply(lambda r: pd.Series({"has_preval": has_preval(r), "has_trans": has_trans_marker(r) if False else has_transform(r)}), axis=1)
both = (flags["has_preval"] & flags["has_trans"]).sum()
only_p = (flags["has_preval"] & ~flags["has_trans"]).sum()
only_t = (~flags["has_preval"] & flags["has_trans"]).sum()
neither = (~flags["has_preval"] & ~flags["has_trans"]).sum()
print(f"rows with preval marker: {flags['has_preval'].sum()} | transform marker: {flags['has_trans'].sum()}")
print(f"BOTH markers: {both} ({both/len(D1):.1%}) | preval-only: {only_p} | trans-only: {only_t} | neither: {neither}")

# ---- type/calling_type semantics
for col in ("type", "calling_type", "Type", "Calling_Type", "call_type"):
    if col in D1.columns:
        print(f"\n{col} value counts:", D1[col].value_counts(dropna=False).head(15).to_dict())

# ---- per-loader: multiple prevalidation / transformation definitions?
gk = "loader_display_name" if "loader_display_name" in D1.columns else None
if gk is None:
    for c in D1.columns:
        if "loader" in c.lower():
            gk = c
            break
print(f"\ngrouping by: {gk}")
if gk:
    per_loader_rows = D1.groupby(gk).size()
    print(f"loaders: {per_loader_rows.size} | rows per loader: min={per_loader_rows.min()} median={per_loader_rows.median()} max={per_loader_rows.max()}")
    # rows that look preval vs trans per loader
    D1f = D1.join(flags)
    per_loader_both = D1f.groupby(gk).apply(lambda g: int((g['has_preval'] & g['has_trans']).any()))
    print(f"loaders with AT LEAST ONE row having both markers: {per_loader_both.sum()}/{per_loader_both.size}")

# ---- example rows
print("\n=== example rows (first 6, key columns only) ===")
key_cols = [c for c in D1.columns if any(k in str(c).lower() for k in ("loader", "dataset", "sprint", "connect", "preval", "trans", "type", "order", "count"))][:12]
print(D1[key_cols].head(6).to_string())

print("\n=== example rows WITH BOTH markers ===")
mask_both = flags["has_preval"] & flags["has_trans"]
if mask_both.any():
    print(D1.loc[mask_both, key_cols].head(4).to_string())
else:
    print("(none)")

print("\n=== preval_trans_display_name distinct sample ===")
ptcol = [c for c in D1.columns if "preval" in c.lower()]
for c in ptcol:
    vals = D1[c].dropna().astype(str)
    trans = vals[vals.str.lower().str.contains("trans")]
    preval = vals[vals.str.lower().str.contains("preval")]
    other = vals[~vals.str.lower().str.contains("trans|preval")]
    print(f"{c}: n={len(vals)} | contains-trans={len(trans)} | contains-preval={len(preval)} | other={len(other)}")
    print("  other sample:", other.head(6).tolist())
    print("  trans sample:", trans.head(4).tolist())

print("\nDONE")
