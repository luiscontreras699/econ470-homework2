from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"
OUT = ROOT / "submission3" / "results"
OUT.mkdir(exist_ok=True)

YEARS = range(2014, 2020)

def clean_cols(df):
    df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]
    return df

def pick_december_file(year: int) -> Path:
    dec = DATA / f"State_County_Penetration_MA_{year}_12.csv"
    if dec.exists():
        return dec
    files = sorted(DATA.glob(f"State_County_Penetration_MA_{year}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No penetration files found for {year} in {DATA}")
    return files[-1]

def to_num_series(s: pd.Series) -> pd.Series:
    # handles "35.2%" and commas
    s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def normalize_share(share: pd.Series) -> pd.Series:
    # If stored as percent 0-100, convert to 0-1
    med = share.median(skipna=True)
    if pd.notna(med) and med > 1:
        share = share / 100.0
    return share

def load_penetration_share(year: int) -> pd.DataFrame:
    f = pick_december_file(year)
    df = pd.read_csv(f, low_memory=False)
    df = clean_cols(df)

    print(f"\n[{year}] Using file: {f.name}")
    print(f"[{year}] Columns: {list(df.columns)}")

    if "penetration" in df.columns:
        share = to_num_series(df["penetration"])
        share = normalize_share(share)
        df["share"] = share
        # Debug stats
        print(f"[{year}] penetration numeric stats: min={df['share'].min(skipna=True)} "
              f"median={df['share'].median(skipna=True)} max={df['share'].max(skipna=True)}")
    elif ("enrolled" in df.columns) and ("eligibles" in df.columns):
        enrolled = pd.to_numeric(df["enrolled"], errors="coerce")
        eligibles = pd.to_numeric(df["eligibles"], errors="coerce")
        df["share"] = enrolled / eligibles
    else:
        raise KeyError(
            f"[{year}] Expected columns not found. Need 'penetration' or both 'enrolled' and 'eligibles'. "
            f"Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["share"])

    # Now apply reasonable bounds after normalization:
    df = df[(df["share"] >= 0) & (df["share"] <= 1)]

    print(f"[{year}] Non-missing share rows (after normalization): {len(df):,}")
    return df[["share"]].copy()

results = []
for y in YEARS:
    print(f"\nProcessing {y}...")
    dfy = load_penetration_share(y)
    results.append({"year": y, "avg_share": dfy["share"].mean()})

df_res = pd.DataFrame(results)
df_res.to_csv(OUT / "task4_ma_share_trend.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(df_res["year"], df_res["avg_share"], marker="o")
plt.title("Average Medicare Advantage Share Over Time (2014–2019)")
plt.xlabel("Year")
plt.ylabel("Average MA Share")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "task4_ma_share_trend.png", dpi=200)
plt.close()

print("\nTask 4 complete.")
print(df_res)