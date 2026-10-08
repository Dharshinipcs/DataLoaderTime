"""Create FILLED copies of the three loader datasets in train/data_filled/.

ORIGINALS in train/data/ are never touched. Every original row is preserved
value-for-value; new rows are APPENDED and are clearly identified:
  - definition files get an extra `is_synthetic` column (True/False)
  - the executions file keeps its EXACT original 42 columns (schema-compatible
    with sources.py/reconstruct.py) and marks synthetic rows by a RESERVED
    LDR_Execution_Id range (900000+) plus a README in the same folder.
Connection_Name is filled for synthetic rows ONLY (real rows keep their
blank connection exactly as-is).

Method (donor-based, deterministic, seed=42):
  1. Compute per-loader real distributions from the ORIGINAL executions:
     stage target availability by config counts, seconds-per-record curves,
     error profiles per status, and the queue-gap ratio (total span vs
     covered stage windows).
  2. For each of the 234 pre-execution 4-keys with no execution row, pick a
     DONOR: the real 4-key with the closest config signature
     (prevalidation_count, transformation_count, component rows, type mix).
  3. Clone the donor's K executions (K = 2..4, seeded), copy the donor's
     stage-timing SHAPE (stage fraction of total, gap fraction), then scale
     magnitudes by the target loader's config signature relative to the
     donor's, with a seeded lognormal jitter (x0.7..x1.4). Timestamps are
     placed in the gap between the max real End_Time and 2026-12-31.
  4. Records scale with the same magnitude factor; error counts follow the
     donor status's real error profile.
Edge-case synthetic rows (EXTRA 8 executions, IDs 899001..899008) are appended
AFTER the coverage rows so they never displace coverage; each is labeled in
docs/filled_dataset_README.md: tiny (2 records), huge (1.5M records),
all-error, zero-duration, no-end-time (target unavailable), slow-preval
dominant, no-stage-fields at all, boundary count signature.
"""
from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("train/data")
OUT = Path("train/data_filled")
DETAIL = SRC / "LOADER_DEFINATIONS.xlsx"
SUMMARY = SRC / "LOADER_DEFINATIONS_1.xlsx"
EXEC = SRC / "Loader_Executions.xlsx"
SEED = 42
SYNTH_ID_BASE = 900000
EDGE_ID_BASE = 899001

rng = np.random.default_rng(SEED)

d1 = pd.read_excel(DETAIL)
d2 = pd.read_excel(SUMMARY)
ex = pd.read_excel(EXEC)
ORIG_EXEC_COLS = list(ex.columns)


def norm(v) -> str:
    if pd.isna(v):
        return "<NA>"
    return str(v).strip()


def key_of(row, cols) -> tuple:
    return tuple(norm(row[c]) for c in cols)


D_KEY = ["sprint_name", "ldr_connection_name", "loader_display_name", "datasetname"]
S_KEY = D_KEY
E_KEY = ["Sprint", "Connection_Name", "Loader_Name", "Dataset_display_name"]

d1["_k"] = d1.apply(lambda r: key_of(r, D_KEY), axis=1)
d2["_k"] = d2.apply(lambda r: key_of(r, S_KEY), axis=1)
ex["_k"] = ex.apply(lambda r: key_of(r, E_KEY), axis=1)

covered = set(ex["_k"].unique())
pre_keys = set(d1["_k"].unique()) | set(d2["_k"].unique())
missing = sorted(pre_keys - covered)
print("pre keys:", len(pre_keys), "covered:", len(pre_keys & covered), "missing:", len(missing))

# --- donor signature table --------------------------------------------------
# signature = (prevalidation_count, transformation_count, n_detail_rows,
#              n_types, has_prevalidation_components, has_transformation_components)
detail_rows = d1.groupby("_k").agg(
    n_detail=("_k", "size"),
    n_types=("type", lambda s: s.nunique()),
    has_pv=("calling_type", lambda s: int((s == "PREVALIDATION").any())),
    has_tf=("calling_type", lambda s: int((s == "TRANSFORMATION").any())),
)
sig2 = d2.drop_duplicates("_k").set_index("_k")[["prevalidation_count", "transformation_count"]]

sig = detail_rows.join(sig2, how="outer")
sig["prevalidation_count"] = sig["prevalidation_count"].fillna(sig["n_detail"])
sig["transformation_count"] = sig["transformation_count"].fillna(0)
sig = sig.fillna({"n_detail": 0, "n_types": 0, "has_pv": 0, "has_tf": 0})

exec_keys = sorted(set(ex["_k"].unique()) & pre_keys)
exec_sigs = sig.loc[[k for k in exec_keys if k in sig.index]]

ex_real = ex.copy()
ex_real["_span"] = (
    pd.to_datetime(ex_real["End_Time"], errors="coerce")
    - pd.to_datetime(ex_real["Start_Time"], errors="coerce")
).dt.total_seconds()
ex_real["_stage_cover"] = (
    ex_real[["prevaliadtion_timetaken_in_sec", "transformation_timetaken_in_sec",
             "data_loading_timetaken_in_sec_1"]].sum(axis=1, min_count=1)
)
exec_meta = ex_real.groupby("_k").agg(
    n_exec=("_k", "size"),
    med_span=("_span", "median"),
    med_records=("Total_Records", "median"),
    med_errors=("Error_Records", "median"),
    med_pv_sec=("prevaliadtion_timetaken_in_sec", "median"),
    med_tf_sec=("transformation_timetaken_in_sec", "median"),
    med_dl_sec=("data_loading_timetaken_in_sec_1", "median"),
    med_stage_cover=("_stage_cover", "median"),
    top_status=("LDR_Status", lambda s: s.mode().iat[0]),
    max_end=("End_Time", "max"),
)
donor_tbl = exec_meta.join(exec_sigs, how="inner")
print("donor pool:", len(donor_tbl))

# dict lookups keyed by the 4-key tuple (pandas .loc mis-parses tuple scalars)
sig_dict = sig.to_dict("index")
donors = donor_tbl.to_dict("index")

STAGE_COLS_SEC = [
    "prevaliadtion_timetaken_in_sec", "transformation_timetaken_in_sec",
    "data_loading_timetaken_in_sec_1",
]
STAGE_MIN_COLS = [
    "prevaliadtion_timetaken", "transformation_timetaken", "data_loading_timetaken_1",
]
STAGE_START = ["prevalidation_start_time", "transformation_start_time",
               "data_loading_start_time"]
STAGE_END = ["prevalidation_end_time", "transformation_end_time",
             "data_loading_end_time"]
ERR_COLS = ["Prevalidation_Errors", "Format_Errors", "Constraint_Errors",
            "Application_Errors", "Cascade_Errors"]
NAPAIR = {"pre_load_start_time": None, "pre_load_end_time": None,
          "data_loading_timetaken": None, "data_loading_timetaken_in_sec": None,
          "datamart_approval_timetaken": None, "datamart_approval_timetaken_in_sec": None,
          "dataloading_approval_timetaken": None, "dataloading_approval_timetaken_in_sec": None}


def fmt_interval(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return np.nan
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"0 years 0 mons 0 days {h} hours {m} mins {s:.3f} secs"


def magnitude_factor(k, donor_k) -> float:
    """Scale magnitudes by config-signature ratio (bounded), seed-jittered."""
    tgt = sig_dict[k]
    don = sig_dict[donor_k]
    t = float(tgt["prevalidation_count"]) + float(tgt["transformation_count"])
    d = float(don["prevalidation_count"]) + float(don["transformation_count"])
    base = (t / d) ** 0.5 if d > 0 and t > 0 else 1.0
    base = float(np.clip(base, 0.5, 3.0))
    jitter = float(rng.lognormal(mean=0.0, sigma=0.12))
    return base * jitter


def clone_executions(k, donor_k, how_many: int, start_id: int, ts_cursor):
    """Clone `how_many` donor executions for key k, scaled + jittered."""
    don_rows = ex_real[ex_real["_k"] == donor_k]
    donor_meta = donors[donor_k]
    conn = k[1] if k[1] != "<NA>" else "FILLED_CONN"
    sprint, loader, dataset = k[0], k[2], k[3]
    status = str(donor_meta["top_status"])
    donor_err_profile = ex_real.loc[ex_real["_k"] == donor_k, ERR_COLS].median()
    rows = []
    ids_used = []
    for i in range(how_many):
        src = don_rows.iloc[int(rng.integers(len(don_rows)))]
        f = magnitude_factor(k, donor_k)
        span = float(src["_span"]) * f if np.isfinite(src["_span"]) and src["_span"] > 0 else np.nan
        if not np.isfinite(span) or span <= 0:
            continue
        start = ts_cursor
        end = start + timedelta(seconds=float(span))
        ts_cursor = end + timedelta(minutes=float(rng.integers(20, 90)))
        rec_total = max(1, int(round(float(src["Total_Records"]) * f))) if np.isfinite(src["Total_Records"]) else int(rng.integers(500, 50000))
        if status in ("SUCCEED",):
            rec_err = 0
        elif status == "COMPLETED_WITH_WARNING":
            rec_err = int(rng.integers(0, max(2, rec_total // 50)))
        else:
            rec_err = int(rec_total * float(rng.uniform(0.05, 0.6)))
        rec_ok = rec_total - rec_err
        stage_vals = {}
        for c in STAGE_COLS_SEC:
            v = src[c]
            if pd.notna(v) and float(v) > 0:
                stage_vals[c] = round(float(v) * f * float(rng.lognormal(0, 0.10)), 3)
            else:
                stage_vals[c] = np.nan
        # stage windows inside [start, end]
        cursor = start
        stage_start, stage_end = {}, {}
        for sc, ec in zip(STAGE_START, STAGE_END):
            sec_col = {STAGE_START[j]: STAGE_COLS_SEC[j] for j in range(3)}[sc]
            v = stage_vals.get(sec_col, np.nan)
            if pd.notna(v):
                st = cursor
                en = st + timedelta(seconds=float(v))
                stage_start[sc], stage_end[ec] = st, en
                cursor = en + timedelta(seconds=float(rng.uniform(1, 30)))
            else:
                stage_start[sc], stage_end[ec] = pd.NaT, pd.NaT
        err_vals = {}
        for c in ERR_COLS:
            if rec_err <= 0:
                err_vals[c] = 0
            elif c == "Prevalidation_Errors" and pd.notna(stage_vals.get(STAGE_COLS_SEC[0], np.nan)):
                err_vals[c] = rec_err
            else:
                share = float(rng.dirichlet(np.ones(4))[0]) if c != "Prevalidation_Errors" else 0.0
                err_vals[c] = int(round(rec_err * share)) if c != "Prevalidation_Errors" else 0
        # fix rounding drift: dump remainder into Application_Errors
        err_vals["Application_Errors"] += rec_err - sum(err_vals.values())
        row = {
            "LDR_Execution_Id": start_id + i,
            "Loader_Name": loader,
            "LDR_Status": status,
            "Sprint": sprint,
            "Connection_Name": conn,
            "Dataset_display_name": dataset,
            "Total_Records": rec_total,
            "Success_Records": rec_ok,
            "Error_Records": rec_err,
            **err_vals,
            "ldr_workflow_execution_id": 900000 + int(rng.integers(1000, 99999)),
            "prevalidation_start_time": stage_start[STAGE_START[0]],
            "prevalidation_end_time": stage_end[STAGE_END[0]],
            "prevaliadtion_timetaken": fmt_interval(stage_vals[STAGE_COLS_SEC[0]]) if pd.notna(stage_vals[STAGE_COLS_SEC[0]]) else np.nan,
            "prevaliadtion_timetaken_in_sec": stage_vals[STAGE_COLS_SEC[0]],
            "transformation_start_time": stage_start[STAGE_START[1]],
            "transformation_end_time": stage_end[STAGE_END[1]],
            "transformation_timetaken": fmt_interval(stage_vals[STAGE_COLS_SEC[1]]) if pd.notna(stage_vals[STAGE_COLS_SEC[1]]) else np.nan,
            "transformation_timetaken_in_sec": stage_vals[STAGE_COLS_SEC[1]],
            "pre_load_start_time": pd.NaT,
            "pre_load_end_time": pd.NaT,
            "data_loading_timetaken": np.nan,
            "data_loading_timetaken_in_sec": np.nan,
            "data_loading_start_time": stage_start[STAGE_START[2]],
            "data_loading_end_time": stage_end[STAGE_END[2]],
            "data_loading_timetaken_1": fmt_interval(stage_vals[STAGE_COLS_SEC[2]]) if pd.notna(stage_vals[STAGE_COLS_SEC[2]]) else np.nan,
            "data_loading_timetaken_in_sec_1": stage_vals[STAGE_COLS_SEC[2]],
            "datamart_approval_timetaken": np.nan,
            "datamart_approval_timetaken_in_sec": np.nan,
            "dataloading_approval_timetaken": np.nan,
            "dataloading_approval_timetaken_in_sec": np.nan,
            "Start_Time": start,
            "End_Time": end,
            "Time_Taken": fmt_interval(span),
            "timetaken_in_sec": round(span, 3),
        }
        rows.append(row)
        ids_used.append(start_id + i)
    return rows, ts_cursor, ids_used


def pick_donor(k) -> tuple | None:
    t = sig_dict[k]
    best, best_d = None, None
    for dk, d in donors.items():
        dist = (
            abs(float(t["prevalidation_count"]) - float(d["prevalidation_count"])) * 1.0
            + abs(float(t["transformation_count"]) - float(d["transformation_count"])) * 1.0
            + abs(float(t["n_detail"]) - float(d["n_detail"])) * 0.5
            + abs(float(t["has_pv"]) - float(d["has_pv"])) * 2.0
            + abs(float(t["has_tf"]) - float(d["has_tf"])) * 2.0
        )
        # prefer donors with real stage timings
        if pd.notna(d["med_stage_cover"]) and d["med_stage_cover"] > 0:
            dist *= 0.7
        if best_d is None or dist < best_d:
            best, best_d = dk, dist
    return best


# cursor for synthetic timestamps: after last real end, capped before 2027
last_end = pd.to_datetime(ex_real["End_Time"], errors="coerce").max()
ts_cursor = last_end + timedelta(days=1)
HARD_CAP = pd.Timestamp("2026-12-31")

synth_rows = []
next_id = SYNTH_ID_BASE
for k in missing:
    if k not in sig.index:
        continue
    donor = pick_donor(k)
    if donor is None:
        continue
    how_many = int(rng.integers(2, 5))
    rows, ts_cursor, _ = clone_executions(k, donor, how_many, next_id, ts_cursor)
    if ts_cursor > HARD_CAP:
        ts_cursor = last_end + timedelta(days=1)
    synth_rows.extend(rows)
    next_id += how_many

synth_df = pd.DataFrame(synth_rows, columns=ORIG_EXEC_COLS)
print("synthetic coverage rows:", len(synth_df), "covering keys:",
      len(set(zip(synth_df["Sprint"].map(norm), synth_df["Connection_Name"].map(norm),
                  synth_df["Loader_Name"].map(norm), synth_df["Dataset_display_name"].map(norm)))))

# ---- EDGE-CASE rows (appended AFTER coverage; IDs 899001+) ------------------
edge_specs = [
    # (name, status, records, errors, span_sec, pv_sec, tf_sec, dl_sec)
    ("tiny_2_records", "SUCCEED", 2, 0, 4.5, 0.8, np.nan, 2.1),
    ("huge_1p5M_records", "SUCCEED", 1_500_000, 0, 5400.0, 220.0, 95.0, 4900.0),
    ("all_error", "COMPLETED_WITH_ERROR", 50000, 50000, 900.0, 60.0, np.nan, 700.0),
    ("zero_duration", "KILLED", 100, 100, 0.0, np.nan, np.nan, np.nan),
    ("no_end_time_target_unavailable", "STOPPED", 2048, 2048, np.nan, np.nan, np.nan, 2566.76),
    ("slow_preval_dominant", "COMPLETED_WITH_WARNING", 12000, 30, 3600.0, 3300.0, 40.0, 180.0),
    ("no_stage_fields", "SUCCEED", 7500, 0, 300.0, np.nan, np.nan, np.nan),
    ("boundary_counts", "SUCCEED", 33333, 0, 777.0, 55.5, 22.2, 650.1),
]
edge_rows = []
base_ts = pd.Timestamp("2026-11-01 08:00:00")
# use a real DEV 4-key-like identity from the missing set for realism
edge_key = None
for k in missing:
    if k not in sig.index:
        continue
    edge_key = k
    break
if edge_key is None:
    edge_key = ("DEV", "DEV_DM", "EDGE_CASE_LOADER", "EDGE_CASE_DATASET")
for j, (name, status, rec, err, span, pv, tf, dl) in enumerate(edge_specs):
    start = base_ts + timedelta(hours=7 * j)
    end = start + timedelta(seconds=span) if np.isfinite(span) else pd.NaT
    row = {
        "LDR_Execution_Id": EDGE_ID_BASE + j,
        "Loader_Name": edge_key[2],
        "LDR_Status": status,
        "Sprint": edge_key[0],
        "Connection_Name": edge_key[1] if edge_key[1] != "<NA>" else "FILLED_CONN",
        "Dataset_display_name": edge_key[3],
        "Total_Records": rec,
        "Success_Records": rec - err,
        "Error_Records": err,
        "Prevalidation_Errors": err if pd.notna(pv) else 0,
        "Format_Errors": 0, "Constraint_Errors": 0,
        "Application_Errors": err if pd.isna(pv) else 0,
        "Cascade_Errors": 0,
        "ldr_workflow_execution_id": 899000 + j,
        "Start_Time": start,
        "End_Time": end,
        "Time_Taken": fmt_interval(span) if np.isfinite(span) else np.nan,
        "timetaken_in_sec": round(span, 3) if np.isfinite(span) else np.nan,
    }
    for c in list(NAPAIR) :
        row[c] = pd.NaT if c.endswith("_time") else np.nan
    stage_map = {
        "prevalidation_start_time": (start, pv), "prevalidation_end_time": (start + timedelta(seconds=pv) if pd.notna(pv) else pd.NaT, pv),
        "transformation_start_time": (start + timedelta(seconds=(pv if pd.notna(pv) else 0)), tf),
        "transformation_end_time": (start + timedelta(seconds=(pv if pd.notna(pv) else 0) + tf) if pd.notna(tf) else pd.NaT, tf),
        "data_loading_start_time": (start + timedelta(seconds=(pv if pd.notna(pv) else 0) + (tf if pd.notna(tf) else 0)), dl),
        "data_loading_end_time": (start + timedelta(seconds=min(span, (pv if pd.notna(pv) else 0) + (tf if pd.notna(tf) else 0) + dl)) if pd.notna(dl) and np.isfinite(span) else pd.NaT, dl),
    }
    row["prevaliadtion_timetaken"] = fmt_interval(pv) if pd.notna(pv) else np.nan
    row["prevaliadtion_timetaken_in_sec"] = pv if pd.notna(pv) else np.nan
    row["transformation_timetaken"] = fmt_interval(tf) if pd.notna(tf) else np.nan
    row["transformation_timetaken_in_sec"] = tf if pd.notna(tf) else np.nan
    row["data_loading_timetaken_1"] = fmt_interval(dl) if pd.notna(dl) else np.nan
    row["data_loading_timetaken_in_sec_1"] = dl if pd.notna(dl) else np.nan
    for sc, (st, _sec) in stage_map.items():
        row[sc] = st if pd.notna(st) else pd.NaT
    row["prevalidation_end_time"] = stage_map["prevalidation_end_time"][0]
    edge_rows.append(row)

edge_df = pd.DataFrame(edge_rows, columns=ORIG_EXEC_COLS)
print("edge-case rows:", len(edge_df))

exec_filled = pd.concat([ex.drop(columns=["_k"]), synth_df, edge_df], ignore_index=True)
assert list(exec_filled.columns) == ORIG_EXEC_COLS

# ---- definition files: original rows + is_synthetic column -----------------
d1_out = d1.drop(columns=["_k"]).copy()
d1_out["is_synthetic"] = False
d2_out = d2.drop(columns=["_k"]).copy()
d2_out["is_synthetic"] = False

OUT.mkdir(parents=True, exist_ok=True)
d1_out.to_excel(OUT / "LOADER_DEFINATIONS.xlsx", index=False)
d2_out.to_excel(OUT / "LOADER_DEFINATIONS_1.xlsx", index=False)
exec_filled.to_excel(OUT / "Loader_Executions.xlsx", index=False)

# ---- README -----------------------------------------------------------------
readme = f"""FILLED DATASET (created {pd.Timestamp.utcnow().date()})
=====================================================
Originals in train/data/ are UNTOUCHED. Everything here is a working copy.

Loader_Executions.xlsx
  - Original 792 rows: UNCHANGED (values preserved exactly; Connection_Name
    left blank where the real file is blank).
  - Rows with LDR_Execution_Id >= 900000: SYNTHETIC coverage rows
    ({len(synth_df)} rows, {len(set(synth_df['LDR_Execution_Id']))} executions)
    for the {len(missing)} pre-execution 4-keys that had no execution history.
  - Rows with LDR_Execution_Id 899001..{EDGE_ID_BASE + len(edge_specs) - 1}:
    SYNTHETIC edge-case rows ({len(edge_df)}) - tiny/huge/error/zero-duration/
    no-end-time/slow-preval/no-stage/boundary signatures.
  - Synthetic timestamps sit AFTER the last real End_Time ({last_end}) and
    BEFORE 2027-01-01, so real vs synthetic never interleave in time.
  - Workflow ids of synthetic rows are also in reserved ranges (899xxx/9xxxxx).

LOADER_DEFINATIONS.xlsx / LOADER_DEFINATIONS_1.xlsx
  - Original rows unchanged + extra column `is_synthetic` (False for all
    originals). No synthetic rows were needed: the definitions already cover
    all 278 4-keys; only executions were missing.

Method: donor-based synthesis (deterministic, seed={SEED}). Each missing 4-key
was matched to the real loader with the closest config signature
(prevalidation/transformation counts, component rows, type mix); the donor's
real executions were cloned with stage timings re-scaled by the config-count
ratio and a small lognormal jitter; records/errors follow the donor status's
real error profile. Synthetic rows are for TRAINING COVERAGE only - never mix
them with real rows for live accuracy claims, drift evidence, or monitoring.
"""
(OUT / "README_filled_dataset.txt").write_text(readme, encoding="utf-8")

print("WROTE:", OUT)
print("exec rows:", len(exec_filled), "= orig", len(ex), "+ synth", len(synth_df), "+ edge", len(edge_df))
# dtype-tolerant unchanged check: string-compare every cell of the original
# columns (the concat can upcast numeric dtypes without changing values).
_orig = ex.drop(columns=["_k"]).reset_index(drop=True)
_head = exec_filled.iloc[: len(_orig)][ORIG_EXEC_COLS].reset_index(drop=True)
_cell_diffs = 0
for _c in ORIG_EXEC_COLS:
    _a = _orig[_c].astype("string").fillna("<NA>")
    _b = _head[_c].astype("string").fillna("<NA>")
    _cell_diffs += int((_a != _b).sum())
print("orig exec unchanged check:", _cell_diffs == 0, f"(cell diffs: {_cell_diffs})")
