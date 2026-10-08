"""EXP-033 final-forensic-closure audit (Parts A, B, C evidence).

Read-only analysis of the corrected Loader_Executions.xlsx with the SAME
unit semantics as the project (durations in seconds = minutes*60; timestamps
coerced; no fabrication; every number printed is measured).

Two lenses, both reported:
  ROW lens      — each raw component row (759 rows) carrying stage timestamps
  EXECUTION lens— grouped by LDR_Execution_Id (512 executions), windows merged

Part A — stage ORDER forensics:
  A1 observed stage sequence per execution
  A2 data-driven pattern taxonomy + per-loader signatures
  A3 execution_order (pre-execution config) vs observed runtime order
  A4 config-vs-runtime disagreement quantification

Part B — zero-overlap vs complete-coverage:
  B1 overlap / touching / gap structure between consecutive stage windows
  B2 coverage: union(stage windows) vs Start_Time->End_Time span per execution
  B3 coverage by pattern category and by loader (not pooled only)
  B4 head/tail uninstrumented segments + data-quality outliers

Part C — total-composition candidate evidence:
  measured total vs (sum of stages, max stage) and gap mass decomposition;
  containment invariant (floor premise) per pattern category.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XLSX = "train/data/Loader_Executions.xlsx"
DEF = "train/data/LOADER_DEFINATIONS.xlsx"

df = pd.read_excel(XLSX)
N_EXEC = df["LDR_Execution_Id"].nunique()
print(f"raw rows: {len(df)}; executions: {N_EXEC}")

OPS = ["staging", "prevalidation", "transformation", "pre_load", "data_loading"]
TS = {
    "staging": ("staging_start_time", "staging_end_time"),
    "prevalidation": ("prevalidation_start_time", "prevalidation_end_time"),
    "transformation": ("transformation_start_time", "transformation_end_time"),
    "pre_load": ("pre_load_start_time", "pre_load_end_time"),
    "data_loading": ("data_loading_start_time", "data_loading_end_time"),
}
DUR_MIN = {
    "staging": "staging_timetaken_in_min",
    "prevalidation": "prevalidation_timetaken_in_min",
    "transformation": "transformation_timetaken_in_min",
    "pre_load": "pre_load_timetaken_in_min",
    "data_loading": "data_loading_timetaken_in_min",
}


def ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


starts = {k: ts(df[TS[k][0]]) for k in OPS}
ends = {k: ts(df[TS[k][1]]) for k in OPS}
durs = {k: pd.to_numeric(df[DUR_MIN[k]], errors="coerce") * 60.0 for k in OPS}

t_start = ts(df["Start_Time"])
t_end = ts(df["End_Time"])
total_min = pd.to_numeric(df["total_timetaken_in_min"], errors="coerce")
total_target = total_min * 60.0


def windows_of(i: int) -> dict:
    w = {}
    for k in OPS:
        s, e = starts[k].iat[i], ends[k].iat[i]
        if pd.notna(s) and pd.notna(e) and e > s:
            w[k] = (s, e)
    return w


def union_of(windows: dict) -> tuple[float, float]:
    """(union length, sum of window lengths) — identical iff disjoint."""
    if not windows:
        return 0.0, 0.0
    ivs = sorted(windows.values())
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    u = sum((e - s).total_seconds() for s, e in merged)
    total = sum((e - s).total_seconds() for s, e in windows.values())
    return u, total


def pair_structure(windows: dict) -> dict:
    ks = sorted(windows, key=lambda k: windows[k][0])
    overlap = touching = gap = 0
    for a, b in zip(ks, ks[1:]):
        d = (windows[a][1] - windows[b][0]).total_seconds()
        if d > 0:
            overlap += 1
        elif d > -1.0:
            touching += 1
        else:
            gap += 1
    return {"overlap": overlap, "touching": touching, "gap": gap}


def categorize(present: frozenset, n: int) -> str:
    if n == 0:
        return "none"
    if n == len(OPS):
        return "full_sequence"
    if n == 1:
        return "single_stage"
    return "partial_multi_stage"


# ---------------------------------------------------------------- build both lenses
def build_rows(idx) -> list[dict]:
    out = []
    for i in idx:
        w = windows_of(i)
        if not w:
            continue
        span = (
            (t_end.iat[i] - t_start.iat[i]).total_seconds()
            if pd.notna(t_start.iat[i]) and pd.notna(t_end.iat[i]) and t_end.iat[i] > t_start.iat[i]
            else np.nan
        )
        head = (min(s for s, _ in w.values()) - t_start.iat[i]).total_seconds() if pd.notna(t_start.iat[i]) else np.nan
        tail = (t_end.iat[i] - max(e for _, e in w.values())).total_seconds() if pd.notna(t_end.iat[i]) else np.nan
        out.append({
            "i": i,
            "LDR_Execution_Id": df["LDR_Execution_Id"].iat[i],
            "loader": df["Loader_Name"].iat[i],
            "sprint": df["Sprint"].iat[i],
            "status": df["LDR_Status"].iat[i],
            "windows": w,
            "present": frozenset(w),
            "order": tuple(sorted(w, key=lambda k: w[k][0])),
            "n": len(w),
            "category": categorize(frozenset(w), len(w)),
            "total_sec": total_target.iat[i] if pd.notna(total_target.iat[i]) else np.nan,
            "span_sec": span,
            "head": head,
            "tail": tail,
        })
    return out


row_view = build_rows(range(len(df)))
print(f"\nROW lens: raw rows with >=1 valid stage window: {len(row_view)}/{len(df)}")

# execution lens: merge windows across each execution's rows
exec_ids = []
exec_meta = []
exec_windows = []
for exec_id, g in df.groupby("LDR_Execution_Id", sort=True):
    merged: dict = {}
    for i in g.index:
        for k, (s, e) in windows_of(i).items():
            if k not in merged:
                merged[k] = (s, e)
            else:
                s0, e0 = merged[k]
                merged[k] = (min(s0, s), max(e0, e))
    i0 = g.index[0]
    span = (
        (t_end.iat[i0] - t_start.iat[i0]).total_seconds()
        if pd.notna(t_start.iat[i0]) and pd.notna(t_end.iat[i0]) and t_end.iat[i0] > t_start.iat[i0]
        else np.nan
    )
    head = (min(s for s, _ in merged.values()) - t_start.iat[i0]).total_seconds() if merged and pd.notna(t_start.iat[i0]) else np.nan
    tail = (t_end.iat[i0] - max(e for _, e in merged.values())).total_seconds() if merged and pd.notna(t_end.iat[i0]) else np.nan
    exec_ids.append(exec_id)
    exec_meta.append({
        "loader": df["Loader_Name"].iat[i0],
        "sprint": df["Sprint"].iat[i0],
        "status": df["LDR_Status"].iat[i0],
        "present": frozenset(merged),
        "order": tuple(sorted(merged, key=lambda k: merged[k][0])) if merged else (),
        "n": len(merged),
        "category": categorize(frozenset(merged), len(merged)),
        "windows": merged,
        "total_sec": total_target.iat[i0] if pd.notna(total_target.iat[i0]) else np.nan,
        "span_sec": span,
        "head": head,
        "tail": tail,
    })
    exec_windows.append(merged)

ex = pd.DataFrame(exec_meta)
ex["LDR_Execution_Id"] = exec_ids
N = len(ex)
with_win = int((ex["n"] > 0).sum())
print(f"EXECUTION lens: executions with >=1 merged valid stage window: {with_win}/{N} ({with_win/N:.1%})")

for label, view in (("ROW", pd.DataFrame(row_view)), ("EXECUTION", ex)):
    v = view[view["n"] > 0]
    print(f"\n== A2 [{label}] stage-presence rates ==")
    for k in OPS:
        c = sum(1 for p in v["present"] if k in p)
        print(f"  {k}: {c}/{len(v)} ({c/len(v):.1%})")
    print(f"\n== A2 [{label}] observed-order distribution ==")
    for n in sorted(v["n"].unique()):
        sub = v[v["n"] == n]
        vc = sub["order"].value_counts()
        print(f" n={n}: {len(sub)} execs, {len(vc)} distinct orders")
        for o, c in vc.head(8).items():
            print(f"    {'>'.join(o)}: {c}")
    print(f"\n== A2 [{label}] pattern categories ==")
    print(v["category"].value_counts().to_dict())
    # order diversity WITHIN a fixed stage set (does order vary for same stages?)
    same_set = v[v["n"] >= 2].groupby(v["present"].apply(lambda p: "+".join(sorted(p))))["order"].nunique()
    multi = same_set[same_set > 1]
    print(f"stage-sets observed with >1 distinct runtime order: {len(multi)}")
    for sig, c in multi.items():
        orders = v[v["n"] >= 2][v["n"] >= 2]["present"].apply(lambda p: "+".join(sorted(p)))
        oc = v[orders == sig]["order"].value_counts()
        print(f"    {sig}: {oc.to_dict()}")

# per-loader signature (execution lens)
print("\n== A2 per-loader stage-set signature (executions lens, loaders >=5) ==")
vc = ex[ex["n"] > 0].groupby("loader")["present"].apply(lambda s: Counter(s).most_common(3))
for loader, cnt in vc.items():
    if sum(c for _, c in cnt) >= 5:
        top = "; ".join(f"{'+'.join(sorted(p))}:{c}" for p, c in cnt)
        print(f"  {loader}: {top}")

# ---------------------------------------------------------------- A3 config vs runtime
print("\n== A3 execution_order config vs runtime order ==")
d1 = pd.read_excel(DEF)
print(f"detail rows: {len(d1)}; calling_type vocabulary: {sorted(d1['calling_type'].astype(str).str.upper().unique().tolist())}")

# config sequences per configuration key
conf_orders: dict = {}
for key, g in d1.groupby(["loader_display_name", "sprint_name"]):
    seq = g.sort_values("execution_order")["calling_type"].astype(str).str.upper().tolist()
    conf_orders.setdefault(key, set()).add(tuple(seq))

# how many configs describe BOTH prev and trans, and their relative order
both_orders = Counter()
for key, seqs in conf_orders.items():
    for s in seqs:
        if "PREVALIDATION" in s and "TRANSFORMATION" in s:
            first = "PREVALIDATION" if s.index("PREVALIDATION") < s.index("TRANSFORMATION") else "TRANSFORMATION"
            both_orders[(key[0], first)] += 1
cfg_first = Counter()
for (loader, first), c in both_orders.items():
    cfg_first[first] += c
print(f"config sequences containing BOTH prev+trans: {sum(both_orders.values())} across {len(both_orders)} (loader,first) combos")
print(f"which calling type is configured FIRST (component-sequence level): {dict(cfg_first)}")
loaders_cfg_prev_first = {loader for (loader, first), c in both_orders.items() if first == "PREVALIDATION" for _ in range(c)}
print(f"config sequences placing PREVALIDATION before TRANSFORMATION: {cfg_first.get('PREVALIDATION', 0)}")

# runtime: executions with both windows
rt_both = ex[ex["present"].apply(lambda p: {"prevalidation", "transformation"} <= p)]
rt_order = Counter()
for o in rt_both["order"]:
    a, b = o.index("prevalidation"), o.index("transformation")
    rt_order["transformation_first" if b < a else "prevalidation_first"] += 1
print(f"runtime executions with BOTH prev+trans windows: {len(rt_both)} -> {dict(rt_order)}")

# config-vs-runtime per matched (loader, sprint) where config places preval first
matched = compared = conflict = 0
ex_conflict = []
for _, r in rt_both.iterrows():
    seqs = conf_orders.get((r["loader"], r["sprint"]))
    if not seqs:
        continue
    matched += 1
    # any config variant placing PREVALIDATION first for this exact config?
    if any(s.index("PREVALIDATION") < s.index("TRANSFORMATION") for s in seqs if "PREVALIDATION" in s and "TRANSFORMATION" in s):
        compared += 1
        b = r["order"].index("transformation")
        a = r["order"].index("prevalidation")
        if b < a:  # runtime transformation first -> conflicts with a preval-first config
            conflict += 1
            if len(ex_conflict) < 3:
                ex_conflict.append((r["loader"], r["sprint"], ">".join(r["order"])))
print(f"executions (both windows) matched to a (loader,sprint) config: {matched}")
print(f"  of these, config places PREVALIDATION before TRANSFORMATION in >=1 variant: {compared}")
print(f"  runtime shows TRANSFORMATION first anyway (config-vs-runtime conflict): {conflict}")
print(f"  examples: {ex_conflict}")
print(
    "\nA4 verdict inputs: config execution_order describes ONLY the PREVALIDATION/TRANSFORMATION "
    "component sequence (no staging/pre_load/data_loading rows exist in the definitions detail); "
    "runtime order of ALL operational stages is decided by timestamps."
)

# ---------------------------------------------------------------- B coverage
def window_len(w) -> float:
    return (w[1] - w[0]).total_seconds() if isinstance(w, tuple) else float((w[1] - w[0]) / np.timedelta64(1, "s"))


for label, view in (("ROW", pd.DataFrame(row_view)), ("EXECUTION", ex)):
    v = view[view["n"] > 0].copy()
    v[["union", "sum_w"]] = [union_of(w) for w in v["windows"]]
    ps = [pair_structure(w) for w in v["windows"]]
    v["overlap"] = [p["overlap"] for p in ps]
    v["touching"] = [p["touching"] for p in ps]
    v["gap"] = [p["gap"] for p in ps]
    v["max_w"] = [max(window_len(w) for w in wins.values()) for wins in v["windows"]]
    print(f"\n== B1 [{label}] consecutive-window structure ==")
    print(f"execs: {len(v)} | pairs: {v['overlap'].sum() + v['touching'].sum() + v['gap'].sum()}")
    print(f"overlap pairs: {int(v['overlap'].sum())} | touching: {int(v['touching'].sum())} | gaps: {int(v['gap'].sum())}")

    hs = v[v["span_sec"].notna() & (v["span_sec"] > 0)].copy()
    print(f"\n== B2 [{label}] union(stage windows) vs Start->End span (n={len(hs)}) ==")
    full = int((hs["union"] >= hs["span_sec"] - 0.5).sum())
    within60 = int((hs["span_sec"] - hs["union"] <= 60).sum())
    rr = (hs["union"] / hs["span_sec"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"union>=span-0.5s (complete coverage): {full}/{len(hs)} ({full/len(hs):.1%}) | within 60s: {within60}")
    print(f"union/span ratio: median={rr.median():.4f} p25={rr.quantile(.25):.4f} p75={rr.quantile(.75):.4f} p90={rr.quantile(.9):.4f}")
    print(f"head gap: median={hs['head'].median():.1f}s p90={hs['head'].quantile(.9):.1f}s | tail gap: median={hs['tail'].median():.1f}s p90={hs['tail'].quantile(.9):.1f}s")
    big = hs[hs["tail"] > 86400]
    print(f"data-quality outliers (tail gap > 1 day): {len(big)}; of these with valid total: {int(big['total_sec'].notna().sum())}")

    bt = hs[hs["total_sec"].notna() & (hs["total_sec"] > 0)].copy()
    r2 = (bt["union"] / bt["total_sec"]).replace([np.inf, -np.inf], np.nan).dropna()
    r3 = (bt["sum_w"] / bt["total_sec"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n== B2b [{label}] vs authoritative total (n={len(bt)}) ==")
    print(f"union/total: median={r2.median():.4f} p25={r2.quantile(.25):.4f} p75={r2.quantile(.75):.4f}")
    print(f"sum(windows)/total: median={r3.median():.4f} max={r3.max():.4f} | sum>=0.999*total: {int((r3 >= 0.999).sum())}/{len(r3)}")
    print(f"sum==union check (disjointness): max|sum-union| = {(bt['sum_w'] - bt['union']).abs().max():.6f}s")

    print(f"\n== B3 [{label}] coverage BY pattern category ==")
    for cat, g in hs.groupby("category"):
        if len(g) < 3:
            continue
        rq = (g["union"] / g["span_sec"]).replace([np.inf, -np.inf], np.nan).dropna()
        fc = (g["union"] >= g["span_sec"] - 0.5).mean()
        print(f"  {cat}: n={len(g)} union/span median={rq.median():.4f} p75={rq.quantile(.75):.4f} complete-coverage={fc:.1%}")

    print(f"\n== B3b [{label}] coverage BY loader (>=5 rows) ==")
    for loader, g in hs.groupby("loader"):
        if len(g) < 5:
            continue
        rq = (g["union"] / g["span_sec"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rq) < 5:
            continue
        print(f"  {loader}: n={len(g)} median={rq.median():.4f} p25={rq.quantile(.25):.4f} p75={rq.quantile(.75):.4f}")

    # containment / floor premise per category
    print(f"\n== B5 [{label}] containment invariant: total >= max(stage window) & >= union ==")
    for cat, g in bt.groupby("category"):
        if len(g) < 3:
            continue
        viol_w = int((g["total_sec"] < g["max_w"]).sum())
        viol_u = int((g["total_sec"] < g["union"] - 1e-6).sum())
        print(f"  {cat}: n={len(g)} total<max_window: {viol_w} | total<union: {viol_u}")
    viol_w_all = int((bt["total_sec"] < bt["max_w"]).sum())
    viol_u_all = int((bt["total_sec"] < bt["union"] - 1e-6).sum())
    print(f"  ALL: n={len(bt)} total<max_window: {viol_w_all} | total<union: {viol_u_all}")

    # duration FIELDS vs total (the actual floor premise uses predicted durations)
    print(f"\n== B6 [{label}] duration FIELDS vs total (floor premise) ==")
    checks = viol = 0
    for i in range(len(df)):
        tt = total_target.iat[i]
        if pd.isna(tt):
            continue
        for k in OPS:
            d = durs[k].iat[i]
            if pd.notna(d):
                checks += 1
                if d > tt:
                    viol += 1
    print(f"  duration-field > total violations: {viol}/{checks}")
    # approvals
    for c in ("datamart_approval_timetaken_in_min", "dataloading_approval_timetaken_in_min"):
        a = pd.to_numeric(df[c], errors="coerce") * 60.0
        m = a.notna() & total_target.notna()
        print(f"  {c}: n={m.sum()} exceed_total={int((a[m] > total_target[m]).sum())}")

print("\n== B4 gap-mass decomposition (EXECUTION lens, medians over rows w/ span) ==")
v = ex[(ex["n"] > 0) & ex["span_sec"].notna() & (ex["span_sec"] > 0)].copy()
v[["union", "sum_w"]] = [union_of(w) for w in v["windows"]]
print(f"  span median={v['span_sec'].median():.1f}s | union median={v['union'].median():.1f}s")
print(f"  head median={v['head'].median():.1f}s | tail median={v['tail'].median():.1f}s")
print(f"  implied internal-gap median={v['span_sec'].median() - v['union'].median():.1f}s")
print(f"  uninstrumented mass ~= {1 - (v['union'].median() / v['span_sec'].median()):.1%} of the span (median)")

# canonical loader example
canon = v[v["loader"] == "MigrationSummarySubTableLoaderDataset"]
if len(canon):
    rq = (canon["union"] / canon["span_sec"]).dropna()
    print(f"\n  canonical loader (MigrationSummarySubTableLoaderDataset): n={len(canon)} union/span median={rq.median():.4f} orders={Counter(canon['order']).most_common(3)}")
print('DONE')
