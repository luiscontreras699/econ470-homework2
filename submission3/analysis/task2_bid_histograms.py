from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_IN = ROOT / "data" / "input"
OUT = ROOT / "submission3" / "results"
OUT.mkdir(exist_ok=True)

CMSPAY = DATA_IN / "payment_2014" / "cms-payment"
YEARS = [2014, 2018]

def clean_text(x) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_cols(cols):
    return [clean_text(c).lower() for c in cols]

def read_csv_enc(path: Path, **kwargs):
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="cp1252", errors="replace", **kwargs)

def norm_contract(x: str) -> str:
    return clean_text(x).upper()

def norm_plan(x: str) -> str:
    s = clean_text(x)
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return ""
    return digits.zfill(3)

def to_number(x):
    s = clean_text(x)
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    return pd.to_numeric(s, errors="coerce")

def choose_best_premium_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "premium" in c]
    if not candidates:
        raise KeyError("No premium-like columns found.")
    best_col = None
    best_nonmissing = -1
    for c in candidates:
        nonmissing = df[c].map(to_number).notna().sum()
        if nonmissing > best_nonmissing:
            best_nonmissing = nonmissing
            best_col = c
    print(f"  Selected premium column: '{best_col}' (numeric non-missing={best_nonmissing:,})")
    return best_col

def load_landscape_premium(year: int) -> pd.DataFrame:
    files = sorted(DATA_IN.glob(f"{year}LandscapeSource file MA_*.csv"))
    if not files:
        raise FileNotFoundError(f"No LandscapeSource files found for {year} in {DATA_IN}")

    dfs = []
    for f in files:
        df = read_csv_enc(f, skiprows=5)
        df.columns = clean_cols(df.columns)
        dfs.append(df)
    land = pd.concat(dfs, ignore_index=True)

    if "contract id" not in land.columns or "plan id" not in land.columns:
        raise KeyError(f"[{year}] Landscape missing contract id / plan id. Columns sample: {land.columns[:40]}")

    prem_col = choose_best_premium_column(land)

    out = land[["contract id", "plan id", prem_col]].copy()
    out.columns = ["contract_id", "plan_id", "premium"]
    out["contract_id"] = out["contract_id"].map(norm_contract)
    out["plan_id"] = out["plan_id"].map(norm_plan)
    out["premium"] = out["premium"].map(to_number)

    out = out.dropna(subset=["premium"])
    out = out[(out["contract_id"] != "") & (out["plan_id"] != "")]
    return out

def find_planlevel_xlsx(year: int) -> Path:
    year_dir = CMSPAY / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Missing folder: {year_dir}")

    candidates = sorted(
        [p for p in year_dir.glob("*.xlsx") if ("partc" in p.name.lower() and "plan" in p.name.lower())],
        key=lambda p: p.name.lower()
    )
    if not candidates:
        all_files = "\n  ".join(sorted([p.name for p in year_dir.glob("*")]))
        raise FileNotFoundError(f"[{year}] No PartC Plan Level .xlsx found.\nFiles:\n  {all_files}")
    return candidates[0]

def find_header_row_payment(xlsx_path: Path, sheet="result.srx", max_rows=250):
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, nrows=max_rows)
    for r in range(raw.shape[0]):
        row_text = " ".join(clean_text(v).lower() for v in raw.iloc[r].tolist())
        if ("contract number" in row_text) and ("plan benefit" in row_text):
            return r
    return None

def load_rebate(year: int) -> pd.DataFrame:
    xlsx_path = find_planlevel_xlsx(year)
    sheet = "result.srx"
    header_row = find_header_row_payment(xlsx_path, sheet=sheet)
    if header_row is None:
        raise ValueError(f"[{year}] Could not find header row in {xlsx_path.name} sheet '{sheet}'")

    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row)
    df.columns = clean_cols(df.columns)

    # 2014/2018 both show "Contract Number" and "Plan Benefit Package" in these files
    contract_col = "contract number"
    plan_col = "plan benefit package"

    rebate_col = next((c for c in df.columns if "average rebate" in c), None)
    if rebate_col is None:
        rebate_col = next((c for c in df.columns if "rebate" in c), None)
    if rebate_col is None:
        raise KeyError(f"[{year}] Could not find rebate column. Columns sample: {df.columns[:80]}")

    out = df[[contract_col, plan_col, rebate_col]].copy()
    out.columns = ["contract_id", "plan_id", "rebate"]
    out["contract_id"] = out["contract_id"].map(norm_contract)
    out["plan_id"] = out["plan_id"].map(norm_plan)
    out["rebate"] = pd.to_numeric(out["rebate"], errors="coerce")

    out = out.dropna(subset=["rebate"])
    out = out[(out["contract_id"] != "") & (out["plan_id"] != "")]
    return out

def save_hist(series, title, path):
    plt.figure(figsize=(9, 5))
    plt.hist(series, bins=40)
    plt.title(title)
    plt.xlabel("Bid (premium + rebate)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def overlap_debug(year, prem, reb):
    prem_keys = prem[["contract_id", "plan_id"]].drop_duplicates()
    reb_keys = reb[["contract_id", "plan_id"]].drop_duplicates()

    prem_keys.to_csv(OUT / f"task2_keys_landscape_{year}.csv", index=False)
    reb_keys.to_csv(OUT / f"task2_keys_payment_{year}.csv", index=False)

    overlap = prem_keys.merge(reb_keys, on=["contract_id", "plan_id"], how="inner")
    print(f"[{year}] unique keys landscape: {len(prem_keys):,}")
    print(f"[{year}] unique keys payment:   {len(reb_keys):,}")
    print(f"[{year}] unique key overlap:    {len(overlap):,}")

    # show some examples of non-overlap (first 10)
    only_prem = prem_keys.merge(reb_keys, on=["contract_id","plan_id"], how="left", indicator=True)
    only_prem = only_prem[only_prem["_merge"] == "left_only"].drop(columns=["_merge"]).head(10)

    only_reb = reb_keys.merge(prem_keys, on=["contract_id","plan_id"], how="left", indicator=True)
    only_reb = only_reb[only_reb["_merge"] == "left_only"].drop(columns=["_merge"]).head(10)

    print(f"[{year}] sample keys only in landscape (10):")
    print(only_prem.to_string(index=False))
    print(f"[{year}] sample keys only in payment (10):")
    print(only_reb.to_string(index=False))

for year in YEARS:
    print(f"\n=== Processing {year} ===")
    prem = load_landscape_premium(year)
    reb = load_rebate(year)

    # Always run overlap debug for 2018 (and safe for 2014)
    if year == 2018:
        overlap_debug(year, prem, reb)

    merged = prem.merge(reb, on=["contract_id", "plan_id"], how="inner")
    merged["bid"] = merged["premium"] + merged["rebate"]
    merged = merged.dropna(subset=["bid"])

    print(f"[{year}] landscape rows: {len(prem):,} | payment rows: {len(reb):,} | merged rows: {len(merged):,}")

    merged[["contract_id", "plan_id", "premium", "rebate", "bid"]].to_csv(
        OUT / f"task2_bids_{year}.csv", index=False
    )

    save_hist(merged["bid"], f"Plan Bid Distribution – {year}", OUT / f"task2_bid_hist_{year}.png")
    print(f"[{year}] saved task2_bids_{year}.csv and task2_bid_hist_{year}.png")

print("\nTask 2 complete!")