"""
Experiment: Diffusion Starting Timestep (t0) Sensitivity on Elliptic
══════════════════════════════════════════════════════════════════════
Tests how the diffusion starting timestep t0 ∈ {50, 100, 150, 200, 250}
affects augmentation quality and downstream classifier performance on
the Elliptic Bitcoin dataset.

Motivation
──────────
t0 controls how much noise is injected before the reverse denoising
process begins.  Too low → generated graphs look like real laundering
(low diversity).  Too high → generated graphs are mostly noise (low
realism).  The thesis uses t0=150 (30% of T=500); this sweep formally
validates that choice and provides ablation evidence for the paper.

Requires
────────
Pre-trained Elliptic models in:
  checkpoints/elliptic/diffusion/model.pt
  checkpoints/simclr_elliptic/best_model.pt  (or checkpoints/elliptic/simclr/)

Usage
─────
  cd igraph_version
  python run_t0_sensitivity_experiment.py

Results are written to:
  results/classifier_comparison_elliptic_t0_<T>.csv
A summary table is printed and saved to:
  results/t0_sensitivity_summary.csv
"""

import subprocess
import sys
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT   = Path(__file__).resolve().parent / "generation" / "evaluate_classifiers.py"
RESULTS  = Path(__file__).resolve().parent / "results"
T0_VALUES = [50, 100, 150, 200, 250]   # t_start values to sweep
N_GEN     = 40                          # generated graphs per run (keep small for speed)
MODELS    = None                        # None = all classifiers


def run_t0(t0: int) -> Path:
    label  = f"t0_{t0}"
    outcsv = RESULTS / f"classifier_comparison_elliptic_{label}.csv"
    if outcsv.exists():
        print(f"\n[t0={t0}] Cache found — skipping run ({outcsv.name})")
        return outcsv

    cmd = [
        sys.executable, str(SCRIPT),
        "--dataset",        "elliptic",
        "--augment",
        "--t-start",        str(t0),
        "--n-gen",          str(N_GEN),
        "--ablation-label", label,
    ]
    if MODELS:
        cmd += ["--models"] + MODELS

    print(f"\n{'='*60}")
    print(f" Running t0={t0}  (n_gen={N_GEN})  ->  {outcsv.name}")
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


def build_summary(t0_values: list[int]) -> list[dict]:
    summary = []
    for t0 in t0_values:
        label  = f"t0_{t0}"
        outcsv = RESULTS / f"classifier_comparison_elliptic_{label}.csv"
        if not outcsv.exists():
            print(f"  WARNING: {outcsv.name} not found — t0={t0} skipped")
            continue
        rows = read_csv_results(outcsv)
        for row in rows:
            summary.append({
                "t0":        t0,
                "model":     row["model"],
                "condition": row.get("condition", ""),
                "f1_mean":   float(row["f1_mean"]),
                "f1_std":    float(row["f1_std"]),
                "auc_mean":  float(row["auc_mean"]),
                "auc_std":   float(row["auc_std"]),
            })
    return summary


def print_table(summary: list[dict]) -> None:
    models    = sorted(set(r["model"] for r in summary))
    t0_values = sorted(set(r["t0"] for r in summary))

    for cond in ("baseline", "augmented"):
        cond_rows = [r for r in summary if r["condition"].lower() == cond]
        if not cond_rows:
            continue

        label_str = "Macro-F1" if cond == "baseline" else "Macro-F1 with augmentation"
        print(f"\n{'='*70}")
        print(f"  {label_str} by t0  ({cond}, Elliptic, mean ± std)")
        print(f"{'='*70}")
        header = f"{'Model':<22}" + "".join(f"  t0={t}" for t in t0_values)
        print(header)
        print("-"*len(header))
        for m in models:
            row_str = f"{m:<22}"
            for t in t0_values:
                hit = next((r for r in cond_rows
                            if r["model"] == m and r["t0"] == t), None)
                if hit:
                    row_str += f"  {hit['f1_mean']:.4f}±{hit['f1_std']:.4f}"
                else:
                    row_str += "  -            "
            print(row_str)
    print()


def save_summary_csv(summary: list[dict]) -> None:
    out = RESULTS / "t0_sensitivity_summary.csv"
    if not summary:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to {out}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for t0 in T0_VALUES:
        run_t0(t0)

    summary = build_summary(T0_VALUES)
    print_table(summary)
    save_summary_csv(summary)


if __name__ == "__main__":
    main()
