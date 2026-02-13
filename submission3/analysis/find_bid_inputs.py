from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"

YEARS = [2014, 2018]

# keywords to search for in headers/cells
PREM_KEYS = ["premium", "part c", "partc", "monthly prem", "plan prem"]
REB_KEYS = ["rebate", "rebate amt", "rebate amount"]
BID_KEYS = ["bid", "plan bid"]

def read_csv_try(path, skiprows=0):
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, nrows=5, low_memory=False, encoding=enc, skiprows=skiprows)
        except Exception:
            continue
    return None

def file_contains_keywords_in_first_rows(path, keywords, max_skip=50):
    # Try to detect headers not on row 0 by skipping 0..max_skip
    for s in range(0, max_skip + 1):
        df = read_csv_try(path, skiprows=s)
        if df is None:
            continue
        cols = [str(c).lower() for c in df.columns]
        if any(any(k in c for k in keywords) for c in cols):
            return True, s, cols
    return False, None, None

def scan_year(year):
    print(f"\n==================== {year} ====================")

    year_files = [p for p in DATA.rglob("*") if p.is_file() and str(year) in p.name]
    year_files = sorted(year_files, key=lambda p: p.name)

    prem_hits, reb_hits, bid_hits = [], [], []

    for f in year_files:
        if f.suffix.lower() != ".csv":
            continue

        ok, skip, cols = file_contains_keywords_in_first_rows(f, PREM_KEYS)
        if ok:
            prem_hits.append((f.name, skip, cols))

        ok, skip, cols = file_contains_keywords_in_first_rows(f, REB_KEYS)
        if ok:
            reb_hits.append((f.name, skip, cols))

        ok, skip, cols = file_contains_keywords_in_first_rows(f, BID_KEYS)
        if ok:
            bid_hits.append((f.name, skip, cols))

    print(f"\nCSV files with PREMIUM-like headers (detected via skiprows up to 50): {len(prem_hits)}")
    for name, skip, cols in prem_hits[:15]:
        print(f"  {name}  | skiprows={skip}  | cols={cols[:20]}")

    print(f"\nCSV files with REBATE-like headers (detected via skiprows up to 50): {len(reb_hits)}")
    for name, skip, cols in reb_hits[:15]:
        print(f"  {name}  | skiprows={skip}  | cols={cols[:20]}")

    print(f"\nCSV files with BID-like headers (detected via skiprows up to 50): {len(bid_hits)}")
    for name, skip, cols in bid_hits[:15]:
        print(f"  {name}  | skiprows={skip}  | cols={cols[:20]}")

for y in YEARS:
    scan_year(y)

print("\nDONE.")