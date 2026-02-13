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

def norm_contract(x) -> str:
    return "" if pd.isna(x) else str(x).strip().upper()

def norm_plan(x) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    digits = re.sub(r"\D", "", s)
    return "" if digits == "" else digits.zfill(3)

def detect_col(cols, must_contain_any):
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in must_contain_any):
            return c
    return None

# ------------------------------------------------------------
# 1) Load enrollment (county-plan enrollment) for 2018
# ------------------------------------------------------------
enr_files = sorted(DATA.glob(f"CPSC_Enrollment_Info_{YEAR}_*.csv"))
if not enr_files:
    raise FileNotFoundError(f"No enrollment files found for {YEAR} in {DATA}")

dfs = []
for f in enr_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = clean_cols(df.columns)
    dfs.append(df)

enr = pd.concat(dfs, ignore_index=True)

contract_col = detect_col(enr.columns, ["contract id", "contract number"])
plan_col = detect_col(enr.columns, ["plan id", "plan"])
fips_col = detect_col(enr.columns, ["fips"])
enroll_col = detect_col(enr.columns, ["enrollment"])

if not all([contract_col, plan_col, fips_col, enroll_col]):
    raise KeyError(
        f"Enrollment columns not detected.\n"
        f"Detected contract={contract_col}, plan={plan_col}, fips={fips_col}, enroll={enroll_col}\n"
        f"Columns: {enr.columns}"
    )

enr = enr.rename(columns={
    contract_col: "contract_id",
    plan_col: "plan_id",
    fips_col: "fips",
    enroll_col: "enrollment"
})

enr["contract_id"] = enr["contract_id"].map(norm_contract)
enr["plan_id"] = enr["plan_id"].map(norm_plan)
enr["enrollment"] = pd.to_numeric(enr["enrollment"], errors="coerce")
enr = enr.dropna(subset=["fips", "enrollment"])
enr = enr[(enr["contract_id"] != "") & (enr["plan_id"] != "")]

# total enrollment by county-plan across months
enr_cp = enr.groupby(["fips", "contract_id", "plan_id"], as_index=False)["enrollment"].sum()

# ------------------------------------------------------------
# 2) Compute county HHI from enrollment shares
# ------------------------------------------------------------
county_total = enr_cp.groupby("fips", as_index=False)["enrollment"].sum().rename(columns={"enrollment": "total_enrollment"})
tmp = enr_cp.merge(county_total, on="fips", how="inner")
tmp["share"] = tmp["enrollment"] / tmp["total_enrollment"]
hhi = tmp.groupby("fips", as_index=False)["share"].apply(lambda s: (s**2).sum()).rename(columns={"share": "hhi"})

p33 = np.percentile(hhi["hhi"], 33)
p66 = np.percentile(hhi["hhi"], 66)

hhi["treatment"] = np.where(hhi["hhi"] >= p66, 1,
                            np.where(hhi["hhi"] <= p33, 0, np.nan))
hhi = hhi.dropna(subset=["treatment"])
hhi["treatment"] = hhi["treatment"].astype(int)

print(f"HHI cutoffs for {YEAR}: p33={p33:.6f}, p66={p66:.6f}")
print(f"Counties kept (bottom+top terciles): {len(hhi):,}")

# ------------------------------------------------------------
# 3) Load plan-level payment file and compute bid proxy
# ------------------------------------------------------------
pay_dir = DATA / "payment_2014" / "cms-payment" / str(YEAR)
if not pay_dir.exists():
    raise FileNotFoundError(f"Missing payment folder: {pay_dir}")

xlsx_candidates = sorted([p for p in pay_dir.glob("*.xlsx") if "partc" in p.name.lower() and "plan" in p.name.lower()])
if not xlsx_candidates:
    raise FileNotFoundError(f"No PartC Plan Level xlsx found in {pay_dir}")

xlsx_path = xlsx_candidates[0]

raw = pd.read_excel(xlsx_path, sheet_name="result.srx", header=None, nrows=250)
header_row = None
for r in range(raw.shape[0]):
    row_text = " ".join(str(v).lower() for v in raw.iloc[r].tolist())
    if ("contract number" in row_text) and ("plan benefit" in row_text):
        header_row = r
        break
if header_row is None:
    raise ValueError(f"Could not detect header row in {xlsx_path.name} (sheet result.srx)")

pay = pd.read_excel(xlsx_path, sheet_name="result.srx", header=header_row)
pay.columns = clean_cols(pay.columns)

contract_col = detect_col(pay.columns, ["contract number", "contract"])
plan_col = detect_col(pay.columns, ["plan benefit", "pbp", "package"])
rebate_col = detect_col(pay.columns, ["average rebate"])
abpay_col = detect_col(pay.columns, ["average a/b", "average ab", "a/b pm/pm", "ab pm/pm"])

if not all([contract_col, plan_col, rebate_col, abpay_col]):
    raise KeyError(
        f"Payment columns not detected in {xlsx_path.name}.\n"
        f"Detected contract={contract_col}, plan={plan_col}, rebate={rebate_col}, abpay={abpay_col}\n"
        f"Columns sample: {list(pay.columns)[:60]}"
    )

pay = pay.rename(columns={
    contract_col: "contract_id",
    plan_col: "plan_id",
    rebate_col: "rebate_pmpm",
    abpay_col: "ab_payment_pmpm"
})

pay["contract_id"] = pay["contract_id"].map(norm_contract)
pay["plan_id"] = pay["plan_id"].map(norm_plan)
pay["rebate_pmpm"] = pd.to_numeric(pay["rebate_pmpm"], errors="coerce")
pay["ab_payment_pmpm"] = pd.to_numeric(pay["ab_payment_pmpm"], errors="coerce")

pay = pay.dropna(subset=["rebate_pmpm", "ab_payment_pmpm"])
pay = pay[(pay["contract_id"] != "") & (pay["plan_id"] != "")]

pay["bid_proxy"] = pay["rebate_pmpm"] + pay["ab_payment_pmpm"]
pay_plan = pay[["contract_id", "plan_id", "bid_proxy"]].drop_duplicates()

# ------------------------------------------------------------
# 4) Merge bid proxy onto county-plan enrollment, compute county avg bid proxy
# ------------------------------------------------------------
cp = enr_cp.merge(pay_plan, on=["contract_id", "plan_id"], how="inner")

print(f"County-plan enrollment rows: {len(enr_cp):,}")
print(f"After merging payment bid proxy: {len(cp):,}")

if len(cp) == 0:
    raise ValueError("Merge is empty. Enrollment plan IDs not matching payment plan IDs.")

county_bid = cp.groupby("fips").apply(
    lambda g: np.average(g["bid_proxy"], weights=g["enrollment"])
).reset_index(name="county_avg_bid_proxy")

# ------------------------------------------------------------
# 5) Merge treatment and compute group averages
# ------------------------------------------------------------
county_bid = county_bid.merge(hhi[["fips", "treatment", "hhi"]], on="fips", how="inner")

# SAVE county-level output for later tasks
county_bid.to_csv(OUT / "task5_county_level.csv", index=False)

avg_comp = county_bid.loc[county_bid["treatment"] == 0, "county_avg_bid_proxy"].mean()
avg_uncomp = county_bid.loc[county_bid["treatment"] == 1, "county_avg_bid_proxy"].mean()

print("\n=== Task 5 Results (2018) ===")
print(f"Average bid (proxy) in COMPETITIVE counties:   {avg_comp:.2f}")
print(f"Average bid (proxy) in UNCOMPETITIVE counties: {avg_uncomp:.2f}")
print(f"Difference (uncompetitive - competitive):     {(avg_uncomp - avg_comp):.2f}")

summary = pd.DataFrame({
    "group": ["competitive (bottom tercile HHI)", "uncompetitive (top tercile HHI)"],
    "avg_bid_proxy": [avg_comp, avg_uncomp]
})
summary.to_csv(OUT / "task5_avg_bid_by_competition.csv", index=False)

print(f"\nSaved: {OUT / 'task5_avg_bid_by_competition.csv'}")
print(f"Saved: {OUT / 'task5_county_level.csv'}")