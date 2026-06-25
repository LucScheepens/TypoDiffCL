"""
Compute dataset statistics for the IBM AML benchmark table and the Elliptic
Bitcoin dataset. Prints #Accounts/#Nodes, #Trans/#Edges, #Illicit, Illicit %,
#Networks/#Timesteps.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "IBM"
ELLIPTIC_DIR = Path(__file__).parent / "data" / "elliptic_bitcoin_dataset"
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

# ── Elliptic Bitcoin Dataset ──────────────────────────────────────────────────
print("\nProcessing Elliptic...", flush=True)

classes_path   = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
edgelist_path  = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
features_path  = ELLIPTIC_DIR / "elliptic_txs_features.csv"

classes  = pd.read_csv(classes_path)
n_nodes  = len(classes)
n_edges  = sum(1 for _ in open(edgelist_path, "r", encoding="utf-8")) - 1  # minus header

labeled   = classes[classes["class"] != "unknown"]
n_illicit = int((labeled["class"] == "1").sum())
n_licit   = int((labeled["class"] == "2").sum())
illicit_pct = f"{n_illicit / (n_illicit + n_licit) * 100:.2f}%"

# timestep is the second column (no header) of the features file
features_head = pd.read_csv(features_path, header=None, usecols=[1])
n_timesteps = features_head[1].nunique()

elliptic_row = {
    "Dataset":    "Elliptic Bitcoin",
    "#Nodes":     n_nodes,
    "#Edges":     n_edges,
    "#Illicit":   n_illicit,
    "#Licit":     n_licit,
    "Illicit %":  illicit_pct,
    "#Networks":  n_timesteps,
}

print("\n" + pd.DataFrame([elliptic_row]).to_string(index=False))
