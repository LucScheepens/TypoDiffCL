"""
Compute dataset statistics for the IBM AML benchmark table.
Reads HI/LI × Small/Medium/Large files and prints #Accounts, #Trans,
#Illicit, Illicit %, #Networks.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "IBM"
CONFIGS = ["HI", "LI"]
SCALES  = ["Small", "Medium", "Large"]
CHUNK   = 500_000  # rows per chunk for large transaction files


def count_accounts(path: Path) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for _ in f:
            total += 1
    return total


def count_transactions(path: Path) -> tuple[int, int]:
    """Returns (total_transactions, illicit_transactions)."""
    total, illicit = 0, 0
    for chunk in pd.read_csv(path, usecols=["Is Laundering"], chunksize=CHUNK):
        total   += len(chunk)
        illicit += int(chunk["Is Laundering"].sum())
    return total, illicit


def count_networks(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                count += 1
    return count


rows = []
for config in CONFIGS:
    for scale in SCALES:
        prefix = f"{config}-{scale}"
        acc_file  = DATA_DIR / f"{prefix}_accounts.csv"
        trs_file  = DATA_DIR / f"{prefix}_Trans.csv"
        pat_file  = DATA_DIR / f"{prefix}_Patterns.txt"

        print(f"Processing {prefix}...", flush=True)

        n_accounts = count_accounts(acc_file)  if acc_file.exists()  else None
        n_trans, n_illicit = count_transactions(trs_file) if trs_file.exists() else (None, None)
        n_networks = count_networks(pat_file)  if pat_file.exists()  else None

        illicit_pct = (
            f"{n_illicit / n_trans * 100:.2f}%"
            if n_trans else "—"
        )

        rows.append({
            "Config":    config,
            "Scale":     scale,
            "#Accounts": n_accounts,
            "#Trans.":   n_trans,
            "#Illicit":  n_illicit,
            "Illicit %": illicit_pct,
            "#Networks": n_networks,
        })

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
