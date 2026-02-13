from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "input"

PATTERNS = [
    r"rebate",
    r"risk",
    r"risk.*rebate",
    r"rr",
    r"rebates",
    r"riskadjust",
    r"risk_adj",
    r"plan.*payment",
    r"payment",
]

all_files = sorted([p for p in DATA.rglob("*") if p.is_file()])

hits = []
for p in all_files:
    name = p.name.lower()
    if any(re.search(pat, name) for pat in PATTERNS):
        hits.append(p)

print(f"Total files scanned: {len(all_files)}")
print(f"Matches for risk/rebate/payment keywords: {len(hits)}\n")

for p in hits[:200]:
    print(p.name)

if not hits:
    print("\nNo files matched. This strongly suggests the risk/rebate files are not in data/input yet.")