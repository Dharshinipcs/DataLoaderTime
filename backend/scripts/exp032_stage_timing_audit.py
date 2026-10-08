"""EXP-032 forensic audit: stage-timing physics on the corrected source.

Read-only analysis of train/data/Loader_Executions.xlsx using the SAME
reconstruction semantics as the project (timestamp parsing, seconds units,
no fabrication). Writes no data; prints an audit report.

Questions answered (prompt Part B):
 B1 execution-level timing table (windows, gaps, overlaps, union, ratios)
 B2 actual stage ordering from timestamps
 B3 pairwise overlap analysis (5 operational stages)
 B4 union-vs-sum-vs-total physics
 B5 containment invariant (total >= stage, total >= union)
 B6 skipped-stage analysis (raw rows vs executions, missingness semantics)
 B7 approval-stage timing evidence (duration-only)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XLSX = "train/data/Loader_Executions.xlsx"
df = pd.read_excel(XLSX)
print(f"raw rows: {len(df)}")

# ---------------------------------------------------------------- helpers
def ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def dur_min(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def span_sec(s: pd.Series, e: pd.Series) -> pd.Series:
    d = (e - s).dt.total_seconds()
    return d.where(d > 0)  # non-positive spans unusable

# ---------------------------------------------------------------- windows
STAGES = {
    "staging": ("staging_start_time", "staging_end_time", "staging_timetaken_in_min"),
    "prevalidation": ("prevalidation_start_time", "prevalidation_end_time", "prevalidation_timetaken_in_min"),
    "transformation": ("transformation_start_time", "transformation_end_time", "transformation_timetaken_in_min"),
    "pre_load": ("pre_load_start_time", "pre_load_end_time", "pre_load_timetaken_in_min"),
    "data_loading": ("data_loading_start_time", "data_loading_end_time", "data_loading_timetaken_in_min"),
}

starts = {k: ts(df[v[0]]) for k, v in STAGES.items()}
ends = {k: ts(df[v[1]]) for k, v in STAGES.items()}
mins = {k: dur_min(df[v[2]]) for k, v in STAGES.items()}
spans = {k: span_sec(starts[k], ends[k]) for k in STAGES}

t_start = ts(df["Start_Time"])
t_end = ts(df["End_Time"])
total_span = span_sec(t_start, t_end)
total_min = dur_min(df["total_timetaken_in_min"])
# authoritative total per project semantics: total_timetaken_in_min x 60
total_target = total_min * 60.0
legacy_sec = dur_min(df["timetaken_in_sec"])

print("\n== total-source cross-check ==")
ok = total_min.notna() & total_span.notna()
if ok.any():
    ratio = (total_span[ok] / (total_min[ok] * 60.0)).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"span/min*60 ratio: n={len(ratio)} median={ratio.median():.4f} p25={ratio.quantile(.25):.4f} p75={ratio.quantile(.75):.4f}")
    agree = (ratio.between(0.95, 1.05)).sum()
    print(f"span agrees with total_timetaken_in_min (0.95-1.05): {agree}/{len(ratio)}")
ok2 = total_min.notna() & legacy_sec.notna()
if ok2.any():
    r2 = (legacy_sec[ok2] / (total_min[ok2] * 60.0)).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"legacy_sec/min*60 ratio: n={len(r2)} median={r2.median():.4f}")

# --------------------------------------------- B1 execution-level table
rows = []
for i in range(len(df)):
    windows = {}
    for k in STAGES:
        s, e, sp = starts[k].iat[i], ends[k].iat[i], spans[k].iat[i]
        if pd.notna(s) and pd.notna(e) and pd.notna(sp) and sp > 0:
            windows[k] = (s, e, sp)
    if not windows:
        continue
    tt = total_target.iat[i] if pd.notna(total_target.iat[i]) else np.nan
    t0, t1 = t_start.iat[i], t_end.iat[i]
    # interval union (true union, not max-min)
    ivs = sorted((s, e) for s, e, _ in windows.values())
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    union = sum((e - s).total_seconds() for s, e in merged)
    order = [k for k, _ in sorted(windows.items(), key=lambda kv: kv[1][0])]
    sum_d = sum(sp for _, _, sp in windows.values())
    max_d = max(sp for _, _, sp in windows.values())
    # containment in overall window
    inside = 0
    for s, e, _ in windows.values():
        if pd.notna(t0) and pd.notna(t1) and (s >= t0) and (e <= t1):
            inside += 1
    rows.append({
        "row": i,
        "n_stages": len(windows),
        "stages": "+".join(sorted(windows)),
        "order": ">".join(order),
        "total_sec": tt,
        "total_span_sec": total_span.iat[i],
        "sum_stage": sum_d,
        "union": union,
        "max_stage": max_d,
        "total_over_sum": tt / sum_d if pd.notna(tt) and sum_d > 0 else np.nan,
        "total_over_union": tt / union if pd.notna(tt) and union > 0 else np.nan,
        "sum_over_total": sum_d / tt if pd.notna(tt) and tt > 0 else np.nan,
        "union_over_total": union / tt if pd.notna(tt) and tt > 0 else np.nan,
        "stage_outside_total": len(windows) - inside,
        "windows": windows,
    })

ex = pd.DataFrame(rows)
print(f"\n== B1 execution-level: {len(ex)} executions with >=1 valid stage window ==")
print(ex["n_stages"].value_counts().sort_index().to_dict())
print("stage-combination counts:", ex["stages"].value_counts().head(12).to_dict())

both = ex.dropna(subset=["total_sec"])
print(f"with authoritative total: {len(both)}")
for col in ["total_over_sum", "total_over_union", "sum_over_total", "union_over_total"]:
    s = both[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s):
        print(f"{col}: n={len(s)} median={s.median():.4f} p25={s.quantile(.25):.4f} p75={s.quantile(.75):.4f} p90={s.quantile(.9):.4f}")

# --------------------------------------------- B2 ordering
print("\n== B2 stage ordering (from stage starts) ==")
print("order counts:", ex["order"].value_counts().to_dict())

SEQ = [("staging", "prevalidation"), ("prevalidation", "transformation"), ("transformation", "pre_load"), ("pre_load", "data_loading")]
print("\nStage | Next | ordered | reversed | overlap | gap | missing(pairs evaluable)")
for a, b in SEQ:
    ev = ex[ex["stages"].str.contains(a) & ex["stages"].str.contains(b)]
    ordered = reversed_ = overlap = gap = 0
    for _, r in ev.iterrows():
        sa, ea, da = r["windows"][a]
        sb, eb, db = r["windows"][b]
        if sa < sb:
            ordered += 1
        elif sb < sa:
            reversed_ += 1
        ov = max(0.0, (min(ea, eb) - max(sa, sb)).total_seconds())
        if ov > 0:
            overlap += 1
        elif sa < sb and (sb - ea).total_seconds() > 0:
            gap += 1
    print(f"{a} | {b} | {ordered} | {reversed_} | {overlap} | {gap} | {len(ev)} evaluable")

# --------------------------------------------- B3 pairwise overlaps
print("\n== B3 pairwise overlap (operational stages) ==")
keys = list(STAGES)
hdr = "pair | n | overlap_n | rate | median_ov | mean_ov | p90_ov | seq_gap | touching | reversed"
print(hdr)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        ovs, gaps, touch, rev = [], 0, 0, 0
        for _, r in ex.iterrows():
            w = r["windows"]
            if a not in w or b not in w:
                continue
            sa, ea, _ = w[a]
            sb, eb, _ = w[b]
            ov = max(0.0, (min(ea, eb) - max(sa, sb)).total_seconds())
            ovs.append(ov)
            if sa < sb and (sb - ea).total_seconds() > 0:
                gaps += 1
            elif abs((sb - ea).total_seconds()) < 1 and (sa < sb):
                touch += 1
            elif sb < sa:
                rev += 1
        if ovs:
            arr = np.array(ovs)
            n_ov = int((arr > 0).sum())
            print(f"{a}|{b} | {len(arr)} | {n_ov} | {n_ov/len(arr):.2%} | {np.median(arr):.1f} | {arr.mean():.1f} | {np.percentile(arr,90):.1f} | {gaps} | {touch} | {rev}")

# --------------------------------------------- B5 containment invariant
print("\n== B5 physical invariant: total >= stage, total >= union ==")
viol_stage = viol_union = checked = 0
examples = []
for _, r in both.iterrows():
    tt = r["total_sec"]
    checked += 1
    worst = max(sp for _, _, sp in r["windows"].values())
    if tt < worst:
        viol_stage += 1
        if len(examples) < 5:
            examples.append((r["row"], round(tt, 1), round(worst, 1), r["stages"]))
    if pd.notna(r["union"]) and tt < r["union"]:
        viol_union += 1
print(f"checked={checked} total<max_stage: {viol_stage} ({viol_stage/checked:.1%}) | total<union: {viol_union} ({viol_union/checked:.1%})")
print("examples (row, total, worst_stage, stages):", examples)

# also against span-based total where available
vsp = ex.dropna(subset=["total_span_sec"])
v2 = sum(1 for _, r in vsp.iterrows() if r["total_span_sec"] < max(sp for _, _, sp in r["windows"].values()))
print(f"vs span-total: checked={len(vsp)} violations={v2}")

# --------------------------------------------- B6 skipped stages
print("\n== B6 skipped-stage analysis ==")
N = len(df)
for k in STAGES:
    s, e, m = STAGES[k]
    raw_any = int((df[s].notna() | df[e].notna() | mins[k].notna()).sum())
    raw_dur = int(mins[k].notna().sum())
    raw_span = int((spans[k] > 0).sum())
    in_ex = int(ex["stages"].str.contains(k).sum())
    print(f"{k}: raw rows any-evidence={raw_any} ({raw_any/N:.1%}) | duration={raw_dur} | valid span={raw_span} | executions w/ window={in_ex} ({in_ex/len(ex):.1%} of {len(ex)}) | missing={N-in_ex} ({(N-in_ex)/N:.1%})")

# missingness vs execution status
if "LDR_Status" in df.columns:
    print("\nstage presence by status (any evidence %):")
    for k in STAGES:
        s, e, m = STAGES[k]
        anyev = df[s].notna() | df[e].notna() | mins[k].notna()
        t = df.assign(_any=anyev).groupby("LDR_Status")["_any"].mean()
        print(f"  {k}: {t.round(3).to_dict()}")

# --------------------------------------------- B7 approvals
print("\n== B7 approvals (duration-only) ==")
for col in ["datamart_approval_timetaken_in_min", "dataloading_approval_timetaken_in_min"]:
    if col in df.columns:
        v = dur_min(df[col])
        pos = v[v > 0]
        print(f"{col}: n={v.notna().sum()} positive={len(pos)} median_min={pos.median():.3f} p75={pos.quantile(.75):.3f} max={pos.max():.3f}")

# correlation with total where both exist
for col in ["datamart_approval_timetaken_in_min", "dataloading_approval_timetaken_in_min"]:
    if col in df.columns:
        v = dur_min(df[col]) * 60.0
        m = v.notna() & total_target.notna()
        if m.sum() >= 5:
            c = np.corrcoef(v[m], total_target[m])[0, 1]
            print(f"corr({col}x60, total): n={m.sum()} r={c:.3f}")

print("\nDONE")
