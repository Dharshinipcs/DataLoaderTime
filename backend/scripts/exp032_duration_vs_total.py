"""EXP-032: stage DURATION-FIELD vs total consistency (decides the floor rule).

The containment invariant held for stage WINDOWS (spans). Stage targets come
from duration FIELDS. If duration fields themselves exceed the total, a
serving-side floor `total >= max(stage predictions)` would fight the data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_excel("train/data/Loader_Executions.xlsx")
total = pd.to_numeric(df["total_timetaken_in_min"], errors="coerce") * 60.0

DUR = {
    "staging": "staging_timetaken_in_min",
    "prevalidation": "prevalidation_timetaken_in_min",
    "transformation": "transformation_timetaken_in_min",
    "pre_load": "pre_load_timetaken_in_min",
    "data_loading": "data_loading_timetaken_in_min",
}

print("== per-stage: duration-field vs total ==")
viol_total = 0
checked = 0
for k, c in DUR.items():
    d = pd.to_numeric(df[c], errors="coerce") * 60.0
    m = d.notna() & total.notna()
    exceed = int((d[m] > total[m]).sum())
    viol_total += exceed
    checked += int(m.sum())
    ratio = (d[m] / total[m]).replace([np.inf, -np.inf], np.nan)
    print(f"{k}: n={m.sum()} exceed_total={exceed} ({exceed/max(m.sum(),1):.1%}) median(d/total)={ratio.median():.3f} p90={ratio.quantile(.9):.3f} max={ratio.max():.2f}")

print(f"\nrows where ANY stage duration-field > total: {viol_total}/{checked} checks")

# span vs duration-field consistency per stage
TS = {
    "staging": ("staging_start_time", "staging_end_time"),
    "prevalidation": ("prevalidation_start_time", "prevalidation_end_time"),
    "transformation": ("transformation_start_time", "transformation_end_time"),
    "pre_load": ("pre_load_start_time", "pre_load_end_time"),
    "data_loading": ("data_loading_start_time", "data_loading_end_time"),
}
print("\n== duration-field vs timestamp-span consistency (per stage) ==")
for k, c in DUR.items():
    s, e = TS[k]
    span = (pd.to_datetime(df[e], errors="coerce") - pd.to_datetime(df[s], errors="coerce")).dt.total_seconds()
    d = pd.to_numeric(df[c], errors="coerce") * 60.0
    m = span.notna() & (span > 0) & d.notna()
    r = (span[m] / d[m]).replace([np.inf, -np.inf], np.nan)
    agree = int(r.between(0.9, 1.1).sum())
    print(f"{k}: n={m.sum()} span/duration median={r.median():.3f} agree(0.9-1.1)={agree} ({agree/max(m.sum(),1):.0%})")

# approval durations vs total
print("\n== approvals vs total ==")
for c in ("datamart_approval_timetaken_in_min", "dataloading_approval_timetaken_in_min"):
    d = pd.to_numeric(df[c], errors="coerce") * 60.0
    m = d.notna() & total.notna()
    exceed = int((d[m] > total[m]).sum())
    print(f"{c}: n={m.sum()} exceed_total={exceed} median(d/total)={(d[m]/total[m]).median():.4f}")

print("\nDONE")
