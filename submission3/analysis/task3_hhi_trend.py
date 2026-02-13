from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"
OUT = ROOT / "submission3" / "results"
OUT.mkdir(exist_ok=True)

YEARS = range(2014, 2020)

def clean_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_enrollment_year(year):
    files = sorted(DATA.glob(f"CPSC_Enrollment_Info_{year}_*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df = clean_cols(df)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df = df.rename(columns={
        "contract id": "contract_id",
        "plan id": "plan_id",
        "fips state county code": "fips",
        "enrollment": "enrollment"
    })

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df = df.dropna(subset=["enrollment", "fips"])
    return df

def compute_hhi_year(year):
    enr = load_enrollment_year(year)

    # total enrollment per county
    county_totals = enr.groupby("fips")["enrollment"].sum().reset_index(name="total")

    enr = enr.merge(county_totals, on="fips")
    enr["share"] = enr["enrollment"] / enr["total"]
    enr["share_sq"] = enr["share"] ** 2

    hhi_county = enr.groupby("fips")["share_sq"].sum().reset_index(name="hhi")
    avg_hhi = hhi_county["hhi"].mean()
    return avg_hhi

results = []

for y in YEARS:
    print(f"Processing {y}...")
    hhi = compute_hhi_year(y)
    results.append({"year": y, "avg_hhi": hhi})

df_res = pd.DataFrame(results)
df_res.to_csv(OUT / "task3_hhi_trend.csv", index=False)

plt.figure(figsize=(8,5))
plt.plot(df_res["year"], df_res["avg_hhi"], marker="o")
plt.title("Average HHI Over Time (2014–2019)")
plt.xlabel("Year")
plt.ylabel("Average HHI")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "task3_hhi_trend.png", dpi=200)
plt.close()

print("Task 3 complete.")