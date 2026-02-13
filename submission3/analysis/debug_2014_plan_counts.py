from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"

print("\n=== DEBUG: LISTING ALL CSV FILES ===\n")

all_csvs = sorted(DATA.rglob("*.csv"))
print(f"Total CSV files found: {len(all_csvs)}\n")

for f in all_csvs[:200]:
    print(f.name)

print("\n=== SEARCHING FOR FILES THAT CONTAIN PLAN ID / PBP ===\n")

hits = []

for f in all_csvs:
    try:
        # read only first few rows, flexible encoding
        try:
            df = pd.read_csv(f, nrows=5, low_memory=False, encoding="utf-8")
        except:
            try:
                df = pd.read_csv(f, nrows=5, low_memory=False, encoding="cp1252")
            except:
                df = pd.read_csv(f, nrows=5, low_memory=False, encoding="latin1")

        cols_lower = [c.lower() for c in df.columns]

        if any(x in cols_lower for x in [
            "plan id", "plan_id", "pbp", "pbpid", "pbp id", "plan", "benefit"
        ]):
            hits.append((f.name, df.columns.tolist()))

    except Exception:
        pass

print(f"Files that might contain Plan ID / PBP columns: {len(hits)}\n")

for name, cols in hits[:25]:
    print("\nFILE:", name)
    print("COLUMNS:", cols[:25])

print("\n=== DONE DEBUGGING ===\n")