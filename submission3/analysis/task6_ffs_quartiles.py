from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"
OUT = ROOT / "submission3" / "results"
OUT.mkdir(exist_ok=True)

YEAR = 2018

def clean_cols(cols):
    return [re.sub(r"\s+", " ", str(c).replace("\n", " ").strip().lower()) for c in cols]

def find_header_row_with_code(xlsx_path, sheet_name="FFS18", max_rows=120):
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=max_rows)
    for r in range(raw.shape[0]):
        row_text = " ".join(str(x).strip().lower() for x in raw.iloc[r].tolist() if pd.notna(x))
        if ("code" in row_text) and ("state" in row_text) and ("county" in row_text):
            return r
    return None

def digits_only(s):
    s = str(s)
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    return s

def zfill5(x):
    d = digits_only(x)
    return d.zfill(5) if d != "" else np.nan

# ---------------------------------------------------
# 1) Load county-level data from Task 5 (FIPS)
# ---------------------------------------------------
county_path = OUT / "task5_county_level.csv"
if not county_path.exists():
    raise FileNotFoundError(f"Missing {county_path}. Run updated Task 5 first.")

county = pd.read_csv(county_path)
required = {"fips", "county_avg_bid_proxy", "treatment"}
missing = required - set(county.columns)
if missing:
    raise KeyError(f"task5_county_level.csv missing columns: {missing}")

county = county.copy()
county["fips"] = county["fips"].apply(zfill5)

print(f"Task5 counties: {len(county):,}")
print("Task5 FIPS samples:", county["fips"].dropna().unique()[:10])

# ---------------------------------------------------
# 2) Load FFS18.xlsx (SSA CODE + per-capita costs)
# ---------------------------------------------------
ffs_path = DATA / "FFS18.xlsx"
if not ffs_path.exists():
    raise FileNotFoundError("FFS18.xlsx not found in data/input.")

sheet = "FFS18"
hdr = find_header_row_with_code(ffs_path, sheet_name=sheet)
if hdr is None:
    raise ValueError("Could not detect header row in FFS18.xlsx.")

print(f"\nUsing FFS file: {ffs_path.name} | sheet={sheet} | header_row={hdr}")

ffs = pd.read_excel(ffs_path, sheet_name=sheet, header=hdr)
ffs.columns = clean_cols(ffs.columns)

if "code" not in ffs.columns:
    raise KeyError(f"'code' column not found. Columns: {list(ffs.columns)[:40]}")

# Find per-capita columns
pa_col = None
pb_col = None
for c in ffs.columns:
    if "part a total per capita" in c and "w/o" not in c:
        pa_col = c
    if "part b total per capita" in c:
        pb_col = c

if pa_col is None or pb_col is None:
    raise KeyError(
        f"Could not detect Part A/B per capita columns.\nColumns sample: {list(ffs.columns)[:80]}"
    )

ffs = ffs.copy()
# IMPORTANT: Treat CODE as SSA (not FIPS)
ffs["ssa"] = ffs["code"].apply(zfill5)

ffs[pa_col] = pd.to_numeric(ffs[pa_col], errors="coerce")
ffs[pb_col] = pd.to_numeric(ffs[pb_col], errors="coerce")
ffs = ffs.dropna(subset=["ssa", pa_col, pb_col])

ffs["ffs_cost"] = ffs[pa_col] + ffs[pb_col]
ffs = ffs[["ssa", "ffs_cost"]]

print(f"FFS rows (usable): {len(ffs):,}")
print("FFS SSA samples:", ffs["ssa"].unique()[:10])

# ---------------------------------------------------
# 3) Build SSA -> FIPS mapping using penetration file (has both)
# ---------------------------------------------------
pen_path = DATA / f"State_County_Penetration_MA_{YEAR}_12.csv"
if not pen_path.exists():
    raise FileNotFoundError(f"Missing penetration file: {pen_path}")

pen = pd.read_csv(pen_path, low_memory=False)
pen.columns = clean_cols(pen.columns)

# penetration file has: fips, ssa (based on your earlier output)
if "fips" not in pen.columns or "ssa" not in pen.columns:
    raise KeyError(f"Penetration file missing fips/ssa. Columns: {list(pen.columns)[:50]}")

pen = pen.copy()
pen["fips"] = pen["fips"].apply(zfill5)
pen["ssa"] = pen["ssa"].apply(zfill5)

# unique mapping (ssa -> fips) at county level
ssa_map = pen[["ssa", "fips"]].drop_duplicates()

print(f"\nSSA->FIPS map rows: {len(ssa_map):,}")
print("Map samples:", ssa_map.head(10).to_string(index=False))

# ---------------------------------------------------
# 4) Convert FFS SSA to FIPS, then merge with Task5 counties
# ---------------------------------------------------
ffs_fips = ffs.merge(ssa_map, on="ssa", how="left").dropna(subset=["fips"])
print(f"FFS rows after SSA->FIPS mapping: {len(ffs_fips):,}")

df = county.merge(ffs_fips[["fips", "ffs_cost"]], on="fips", how="inner")
print(f"Rows after merging Task5 county + FFS (via SSA map): {len(df):,}")

if len(df) == 0:
    raise ValueError("Merge still 0 after SSA->FIPS mapping. Something is off with codes.")

# ---------------------------------------------------
# 5) Quartiles + indicators
# ---------------------------------------------------
# duplicates='drop' protects us if ties create identical bin edges
df["ffs_quartile"] = pd.qcut(df["ffs_cost"], 4, labels=[1, 2, 3, 4], duplicates="drop")

# If duplicates were dropped, you might get <4 bins; handle that gracefully
df["ffs_quartile"] = df["ffs_quartile"].astype("float").astype("Int64")

for q in [1, 2, 3, 4]:
    df[f"ffs_q{q}"] = (df["ffs_quartile"] == q).astype(int)

# ---------------------------------------------------
# 6) Table: avg bid by treatment within each quartile
# ---------------------------------------------------
rows = []
for q in [1, 2, 3, 4]:
    sub = df[df["ffs_quartile"] == q]
    if len(sub) == 0:
        rows.append({
            "ffs_quartile": q,
            "avg_bid_competitive": np.nan,
            "avg_bid_uncompetitive": np.nan,
            "diff_uncomp_minus_comp": np.nan,
            "n_counties": 0
        })
        continue

    avg_c = sub.loc[sub["treatment"] == 0, "county_avg_bid_proxy"].mean()
    avg_t = sub.loc[sub["treatment"] == 1, "county_avg_bid_proxy"].mean()
    rows.append({
        "ffs_quartile": q,
        "avg_bid_competitive": avg_c,
        "avg_bid_uncompetitive": avg_t,
        "diff_uncomp_minus_comp": avg_t - avg_c,
        "n_counties": len(sub)
    })

table = pd.DataFrame(rows)

print("\n=== Task 6 Results (2018) ===")
print(table)

table.to_csv(OUT / "task6_avg_bid_by_ffs_quartile.csv", index=False)
df.to_csv(OUT / "task6_county_level_with_ffs.csv", index=False)

print("\nSaved files:")
print(OUT / "task6_avg_bid_by_ffs_quartile.csv")
print(OUT / "task6_county_level_with_ffs.csv")