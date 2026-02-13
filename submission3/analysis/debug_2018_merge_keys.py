from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission3" / "results"

land = pd.read_csv(OUT / "task2_bids_2018.csv")  # this is merged; small
# Instead read the prem and rebate intermediate by reusing what we already saved:
# We'll reconstruct from your script outputs by reloading the raw inputs again: