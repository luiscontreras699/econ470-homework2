from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"
OUT = ROOT / "submission3" / "results"
OUT.mkdir(exist_ok=True)

YEARS = range(2014, 2020)

def read_csv_safe(path):
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except:
            continue
    return pd.read_csv(path, low_memory=False, encoding="cp1252", errors="replace")

all_counts = []

for year in YEARS:
    print(f"\nProcessing {year}...")

    # -----------------------
    # Load Contract Info (Plan Data)
    # -----------------------
    contract_files = sorted(DATA.glob(f"CPSC_Contract_Info_{year}_*.csv"))
    contract_dfs = [read_csv_safe(f) for f in contract_files]
    contract = pd.concat(contract_dfs, ignore_index=True)

    # Normalize column names
    contract.columns = [c.strip().lower() for c in contract.columns]

    # Filters
    contract = contract[~contract["plan id"].astype(str).str.startswith("8")]
    contract = contract[contract["snp plan"].astype(str).str.lower() != "yes"]
    contract = contract[contract["plan type"].str.lower() != "pdp"]

    # Keep only needed columns
    contract = contract[["contract id", "plan id"]].drop_duplicates()

    # -----------------------
    # Load Service Area
    # -----------------------
    sa_files = sorted(DATA.glob(f"MA_Cnty_SA_{year}_*.csv"))
    sa_dfs = [read_csv_safe(f) for f in sa_files]
    sa = pd.concat(sa_dfs, ignore_index=True)

    sa.columns = [c.strip().lower() for c in sa.columns]

    # Merge
    merged = sa.merge(
        contract,
        on=["contract id"],
        how="inner"
    )

    # County plan counts
    counts = (
        merged.groupby(["fips"])
        .agg(plan_count=("plan id", "nunique"))
        .reset_index()
    )

    counts["year"] = year
    all_counts.append(counts)

# Combine all years
final_counts = pd.concat(all_counts, ignore_index=True)

# Save CSV
final_counts.to_csv(OUT / "task1_plan_counts_by_county_year.csv", index=False)

# Boxplot
plt.figure(figsize=(10, 6))
final_counts.boxplot(column="plan_count", by="year")
plt.title("Plan Counts by County (2014-2019)")
plt.suptitle("")
plt.xlabel("Year")
plt.ylabel("Number of Plans")
plt.savefig(OUT / "task1_boxplot.png")
plt.close()

print("\nTask 1 complete!")