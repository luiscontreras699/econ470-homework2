from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"

YEARS = [2014, 2018]

def read_head(path, n=3):
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, nrows=n, low_memory=False, encoding=enc)
        except Exception:
            continue
    return None

for year in YEARS:
    print(f"\n==================== {year} ====================")
    year_files = sorted([p for p in DATA.rglob("*.csv") if str(year) in p.name])

    premium_hits = []
    rebate_hits = []

    for f in year_files:
        df = read_head(f, n=5)
        if df is None:
            continue

        cols = [c.lower() for c in df.columns]

        if any("premium" in c for c in cols):
            premium_hits.append((f.name, df.columns.tolist()))

        if any("rebate" in c for c in cols):
            rebate_hits.append((f.name, df.columns.tolist()))

    print(f"\nFiles with PREMIUM-like columns: {len(premium_hits)}")
    for name, cols in premium_hits[:25]:
        prem_cols = [c for c in cols if "Prem" in str(c) or "prem" in str(c)]
        print("\n ", name)
        print("   premium cols:", prem_cols if prem_cols else cols[:30])

    print(f"\nFiles with REBATE-like columns: {len(rebate_hits)}")
    for name, cols in rebate_hits[:25]:
        reb_cols = [c for c in cols if "Reb" in str(c) or "reb" in str(c)]
        print("\n ", name)
        print("   rebate cols:", reb_cols if reb_cols else cols[:30])