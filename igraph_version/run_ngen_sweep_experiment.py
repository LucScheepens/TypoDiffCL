"""
Experiment: N_GEN Sweep — FraudGT with multiple augmentation methods (Ethereum)
═══════════════════════════════════════════════════════════════════════════════
Tests how the number of generated graphs n_gen ∈ {100, 200, 400, 800, 1600, 3200}
affects FraudGT performance on the Ethereum phishing dataset, comparing four
augmentation strategies:
  - diffusion : TypoDiffCL (guided DDPM + SimCLR, our method)
  - gan       : WGAN-GP in mean-pooled feature space
  - graphsmote: k-NN interpolation in feature space
  - diga      : unconditional DDPM in feature space

The top-100 highest-degree anchor nodes are removed from the test set so that
evaluation focuses on non-hub, harder-to-classify nodes.

Usage
─────
  cd igraph_version
  python run_ngen_sweep_experiment.py

Results are written to:
  results/classifier_comparison_ethereum_ngen_<N>_<method>.csv
A summary table is printed and saved to:
  results/ngen_sweep_summary.csv
"""

import subprocess
import sys
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT          = Path(__file__).resolve().parent / "generation" / "evaluate_classifiers.py"
RESULTS         = Path(__file__).resolve().parent / "results"
NGEN_VALUES     = [100, 200, 400, 800, 1600, 3200]
AUGMENT_METHODS = ["diffusion", "gan", "graphsmote", "diga"]
T_START         = 150        # thesis default t0 (only used for diffusion method)
MODELS          = ["FraudGT"]
N_RUNS          = 3
REMOVE_TOP_DEG  = 100        # remove top-100 highest-degree anchor nodes from test
LOW_DATA        = 0.01       # subsample training set to 1% (~139 graphs) for speed


def run_ngen_method(n_gen: int, method: str) -> Path:
    label  = f"ngen_{n_gen}_{method}"
    outcsv = RESULTS / f"classifier_comparison_ethereum_{label}.csv"
    if outcsv.exists():
        print(f"\n[n_gen={n_gen}, {method}] Cache found -- skipping ({outcsv.name})")
        return outcsv

    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--dataset",           "ethereum",
        "--augment",
        "--augment-method",    method,
        "--t-start",           str(T_START),
        "--n-gen",             str(n_gen),
        "--ablation-label",    label,
        "--n-runs",            str(N_RUNS),
        "--remove-top-degree", str(REMOVE_TOP_DEG),
        "--low-data",          str(LOW_DATA),
    ]
    if MODELS:
        cmd += ["--models"] + MODELS

    print(f"\n{'='*60}")
    print(f" n_gen={n_gen}  method={method}  ->  {outcsv.name}")
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


def build_summary() -> list[dict]:
    summary = []
    for n_gen in NGEN_VALUES:
        for method in AUGMENT_METHODS:
            label  = f"ngen_{n_gen}_{method}"
            outcsv = RESULTS / f"classifier_comparison_ethereum_{label}.csv"
            if not outcsv.exists():
                print(f"  WARNING: {outcsv.name} not found -- skipped")
                continue
            rows = read_csv_results(outcsv)
            for row in rows:
                if row.get("condition", "").lower() == "augmented":
                    summary.append({
                        "n_gen":    n_gen,
                        "method":   method,
                        "model":    row["model"],
                        "f1_mean":  float(row["f1_mean"]),
                        "f1_std":   float(row["f1_std"]),
                        "auc_mean": float(row["auc_mean"]),
                        "auc_std":  float(row["auc_std"]),
                    })
    return summary


def print_table(summary: list[dict]) -> None:
    if not summary:
        print("No results to display.")
        return

    print(f"\n{'='*72}")
    print(f"  FraudGT Macro-F1 by n_gen and augmentation method")
    print(f"  (Ethereum, augmented, mean over {N_RUNS} seeds, top-{REMOVE_TOP_DEG} hubs removed)")
    print(f"{'='*72}")
    header = f"{'Method':<14}" + "".join(f"  n={n:>4}" for n in NGEN_VALUES)
    print(header)
    print("-" * len(header))
    for method in AUGMENT_METHODS:
        row_str = f"{method:<14}"
        for n in NGEN_VALUES:
            hit = next((r for r in summary
                        if r["method"] == method and r["n_gen"] == n), None)
            if hit:
                row_str += f"  {hit['f1_mean']:.4f}"
            else:
                row_str += "  ------"
        print(row_str)

    print(f"\n{'='*72}")
    print(f"  FraudGT AUC-ROC by n_gen and augmentation method")
    print(f"{'='*72}")
    print(header)
    print("-" * len(header))
    for method in AUGMENT_METHODS:
        row_str = f"{method:<14}"
        for n in NGEN_VALUES:
            hit = next((r for r in summary
                        if r["method"] == method and r["n_gen"] == n), None)
            if hit:
                row_str += f"  {hit['auc_mean']:.4f}"
            else:
                row_str += "  ------"
        print(row_str)
    print()


def save_summary_csv(summary: list[dict]) -> None:
    out = RESULTS / "ngen_sweep_summary.csv"
    if not summary:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to {out}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for n_gen in NGEN_VALUES:
        for method in AUGMENT_METHODS:
            run_ngen_method(n_gen, method)

    summary = build_summary()
    print_table(summary)
    save_summary_csv(summary)


if __name__ == "__main__":
    main()
