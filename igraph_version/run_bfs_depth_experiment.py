"""
Experiment: BFS Depth Sensitivity for Elliptic Bitcoin Dataset
═══════════════════════════════════════════════════════════════
Tests how BFS subgraph extraction depth (2, 3, 4, 6 hops) affects
downstream classifier performance on the Elliptic dataset.

Motivation
──────────
The thesis observed low Elliptic macro-F1 (~0.114) and attributed it to
small, sparse ego-subgraphs.  This sweep validates whether increasing
the BFS hop depth yields richer subgraphs and higher F1.

Usage
─────
  cd igraph_version
  python run_bfs_depth_experiment.py

Results are written to:
  results/classifier_comparison_elliptic_depth_<D>.csv
A summary table is printed to stdout and saved to:
  results/bfs_depth_summary.csv
"""

import subprocess
import sys
import csv
import os
from pathlib import Path

# Force UTF-8 output so Unicode characters in evaluate_classifiers.py don't crash
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT   = Path(__file__).resolve().parent / "generation" / "evaluate_classifiers.py"
RESULTS  = Path(__file__).resolve().parent / "results"
DEPTHS   = [2, 3, 4, 6]          # BFS hop depths to test
MODELS   = None                   # None = all classifiers; or e.g. ["GIN", "DeepSets"]


def run_depth(depth: int) -> Path:
    label  = f"depth_{depth}"
    outcsv = RESULTS / f"classifier_comparison_elliptic_{label}.csv"
    if outcsv.exists():
        print(f"\n[depth={depth}] Cache found — skipping run ({outcsv.name})")
        return outcsv

    cmd = [
        sys.executable, str(SCRIPT),
        "--dataset",        "elliptic",
        "--elliptic-depth", str(depth),
        "--ablation-label", label,
    ]
    if MODELS:
        cmd += ["--models"] + MODELS

    print(f"\n{'='*60}")
    print(f" Running depth={depth}  ->  {outcsv.name}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, check=True, env=env)
    return outcsv


def read_csv_results(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_summary(depths: list[int]) -> list[dict]:
    summary = []
    for depth in depths:
        label  = f"depth_{depth}"
        outcsv = RESULTS / f"classifier_comparison_elliptic_{label}.csv"
        if not outcsv.exists():
            print(f"  WARNING: {outcsv.name} not found — depth {depth} skipped in summary")
            continue
        rows = read_csv_results(outcsv)
        for row in rows:
            if row.get("condition", "").lower() == "baseline":
                summary.append({
                    "depth":     depth,
                    "model":     row["model"],
                    "f1_mean":   float(row["f1_mean"]),
                    "f1_std":    float(row["f1_std"]),
                    "auc_mean":  float(row["auc_mean"]),
                    "auc_std":   float(row["auc_std"]),
                })
    return summary


def print_table(summary: list[dict]) -> None:
    models = sorted(set(r["model"] for r in summary))
    depths = sorted(set(r["depth"] for r in summary))

    # F1 table
    print("\n" + "="*70)
    print("  Macro-F1 by BFS Depth  (Elliptic, baseline, mean ± std)")
    print("="*70)
    header = f"{'Model':<22}" + "".join(f"  depth={d}" for d in depths)
    print(header)
    print("-"*len(header))
    for m in models:
        row_str = f"{m:<22}"
        for d in depths:
            hit = next((r for r in summary if r["model"] == m and r["depth"] == d), None)
            if hit:
                row_str += f"  {hit['f1_mean']:.4f}±{hit['f1_std']:.4f}"
            else:
                row_str += "  -            "
        print(row_str)

    # AUC table
    print("\n" + "="*70)
    print("  AUROC by BFS Depth  (Elliptic, baseline, mean ± std)")
    print("="*70)
    print(header)
    print("-"*len(header))
    for m in models:
        row_str = f"{m:<22}"
        for d in depths:
            hit = next((r for r in summary if r["model"] == m and r["depth"] == d), None)
            if hit:
                row_str += f"  {hit['auc_mean']:.4f}±{hit['auc_std']:.4f}"
            else:
                row_str += "  -            "
        print(row_str)
    print()


def save_summary_csv(summary: list[dict]) -> None:
    out = RESULTS / "bfs_depth_summary.csv"
    if not summary:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to {out}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for depth in DEPTHS:
        run_depth(depth)

    summary = build_summary(DEPTHS)
    print_table(summary)
    save_summary_csv(summary)


if __name__ == "__main__":
    main()
