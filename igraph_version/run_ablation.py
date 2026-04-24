"""
run_ablation.py
──────────────────────────────────────────────────────────────────────────────
Orchestrates the full ablation study for the SimCLR + diffusion pipeline.

Two families of ablations are run:

  A. SimCLR encoder ablations  (require re-training the encoder)
     Vary: augmentation type, supervised contrastive loss, diffusion views
     Each variant trains a new encoder and then evaluates on the classifier.

  B. Generation ablations  (use the default trained encoder, vary generation)
     Vary: guidance scale, novelty weight, degree penalty
     No retraining needed — only evaluate_classifiers.py is re-run.

Results are written to:
  checkpoints/simclr_elliptic_ablation/<condition>/best_model.pt  (encoders)
  results/classifier_comparison_elliptic_<condition>.csv          (metrics)

Usage
─────
  # Run everything (train all encoder variants + all generation ablations)
  python run_ablation.py --dataset elliptic --n-gen 40

  # Skip encoder retraining if checkpoints already exist
  python run_ablation.py --dataset elliptic --n-gen 40 --skip-training

  # Run generation ablations only
  python run_ablation.py --dataset elliptic --n-gen 40 --gen-only

  # Quick smoke test with fewer epochs and graphs
  python run_ablation.py --dataset elliptic --n-gen 10 --epochs 10 --low-data 0.2
"""

import argparse
import subprocess
import sys
import time
import csv
import json
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

EVAL_SCRIPT             = BASE_DIR / "generation" / "evaluate_classifiers.py"
ELLIPTIC_TRAIN_SCRIPT   = BASE_DIR / "elliptic_simclr_train_ablation.py"
IBM_TRAIN_SCRIPT        = BASE_DIR / "ibm_simclr_train_ablation.py"


# ── Ablation condition definitions ────────────────────────────────────────────

# Elliptic conditions — may use --p-edge-drop and --p-feat-mask
# Each entry: (condition_name, description, extra_train_args)
ELLIPTIC_ENCODER_CONDITIONS = [
    (
        "full",
        "Full pipeline: edge-drop + feat-mask + diffusion aug + SupCon",
        [],
    ),
    (
        "no_supcon",
        "No supervised contrastive loss (NT-Xent only)",
        ["--supcon-weight", "0.0"],
    ),
    (
        "no_diffusion_aug",
        "No diffusion augmentation in SimCLR (structural aug only)",
        ["--p-diffusion", "0.0"],
    ),
    (
        "edge_drop_only",
        "Only edge dropout augmentation (no feat-mask, no diffusion)",
        ["--p-feat-mask", "0.0", "--p-diffusion", "0.0"],
    ),
    (
        "feat_mask_only",
        "Only feature masking augmentation (no edge-drop, no diffusion)",
        ["--p-edge-drop", "0.0", "--p-diffusion", "0.0"],
    ),
    (
        "diffusion_aug_only",
        "Diffusion augmentation for both views (no structural aug)",
        ["--p-edge-drop", "0.0", "--p-feat-mask", "0.0", "--p-diffusion", "1.0"],
    ),
    (
        "no_aug",
        "No augmentation at all in SimCLR (identity views + SupCon)",
        ["--p-edge-drop", "0.0", "--p-feat-mask", "0.0", "--p-diffusion", "0.0"],
    ),
    (
        "ntxent_only",
        "NT-Xent only, no SupCon, no diffusion aug",
        ["--supcon-weight", "0.0", "--p-diffusion", "0.0"],
    ),
    # ── Option A / C ─────────────────────────────────────────────────────────
    (
        "diff_multistep",
        "Option A: multi-step DDIM views (15 steps, t_start=0.5T)",
        ["--view-type", "multistep"],
    ),
    (
        "diff_guided",
        "Option C: class-guided DDIM views (15 steps, guidance_scale=1.5)",
        ["--view-type", "guided"],
    ),
    (
        "diff_multistep_to_guided",
        "Option A+C: warmup on multistep then switch to guided",
        ["--view-type", "multistep_to_guided"],
    ),
]

# IBM conditions — IBM augmentation uses augment_network_view_fast internally,
# so only p_diffusion and supcon_weight are exposed as knobs.
IBM_ENCODER_CONDITIONS = [
    (
        "full",
        "Full pipeline: struct aug (crop/edge-del/node-del/node-add) + diffusion aug + SupCon",
        [],
    ),
    (
        "no_supcon",
        "No supervised contrastive loss (NT-Xent only)",
        ["--supcon-weight", "0.0"],
    ),
    (
        "no_diffusion_aug",
        "No diffusion augmentation in SimCLR (structural aug only, p_diffusion=0)",
        ["--p-diffusion", "0.0"],
    ),
    (
        "diffusion_aug_only",
        "Diffusion augmentation for both views (p_diffusion=1.0)",
        ["--p-diffusion", "1.0"],
    ),
    (
        "ntxent_only",
        "NT-Xent only, no SupCon, no diffusion aug",
        ["--supcon-weight", "0.0", "--p-diffusion", "0.0"],
    ),
    # ── Option A / C ─────────────────────────────────────────────────────────
    (
        "diff_multistep",
        "Option A: multi-step DDIM views (15 steps, t_start=0.5T)",
        ["--view-type", "multistep"],
    ),
    (
        "diff_guided",
        "Option C: class-guided DDIM views (15 steps, guidance_scale=1.5)",
        ["--view-type", "guided"],
    ),
    (
        "diff_multistep_to_guided",
        "Option A+C: warmup on multistep then switch to guided",
        ["--view-type", "multistep_to_guided"],
    ),
]

# Kept for backwards compatibility — used when dataset is not "ibm"
ENCODER_CONDITIONS = ELLIPTIC_ENCODER_CONDITIONS

# Each entry: (condition_name, description, extra_eval_args)
# These all use the DEFAULT trained encoder (checkpoints/simclr_elliptic)
GENERATION_CONDITIONS = [
    (
        "gen_full_guided",
        "Full guided generation (classification + novelty + degree penalty)",
        [],
    ),
    (
        "gen_unguided",
        "Unguided diffusion (no classifier guidance, no novelty, no degree penalty)",
        ["--guidance-scale", "0.0", "--novelty-weight", "0.0", "--degree-penalty", "0.0"],
    ),
    (
        "gen_no_novelty",
        "Guided generation without novelty repulsion",
        ["--novelty-weight", "0.0"],
    ),
    (
        "gen_no_degree_pen",
        "Guided generation without degree/density penalty",
        ["--degree-penalty", "0.0"],
    ),
    (
        "gen_classif_only",
        "Classification guidance only (no novelty, no degree penalty)",
        ["--novelty-weight", "0.0", "--degree-penalty", "0.0"],
    ),
    (
        "gen_high_guidance",
        "High guidance scale (4.0 instead of default 2.0)",
        ["--guidance-scale", "4.0"],
    ),
    (
        "gen_low_guidance",
        "Low guidance scale (0.5)",
        ["--guidance-scale", "0.5"],
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd, label):
    print(f"\n{'─'*60}")
    print(f"Running: {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + [str(c) for c in cmd],
        check=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] returncode={result.returncode}  ({elapsed:.0f}s)")
        return False
    print(f"  [OK]  ({elapsed:.0f}s)")
    return True


def _read_csv_results(condition_label, dataset="elliptic"):
    """Read the metrics CSV produced by evaluate_classifiers for a given condition."""
    path = RESULT_DIR / f"classifier_comparison_{dataset}_{condition_label}.csv"
    if not path.exists():
        return None
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['condition']}"
            rows[key] = {
                "auc_mean": float(row["auc_mean"]),
                "auc_std":  float(row["auc_std"]),
                "f1_mean":  float(row["f1_mean"]),
                "f1_std":   float(row["f1_std"]),
            }
    return rows


def _print_summary(all_results):
    """Print a consolidated summary table of all ablation results."""
    print("\n" + "="*100)
    print("ABLATION STUDY SUMMARY")
    print("="*100)

    # Header
    header = f"{'Condition':<30}  {'Model':<20}  {'Split':<12}  {'AUC':>8}  {'F1':>8}"
    print(header)
    print("-"*100)

    for condition, results in sorted(all_results.items()):
        if results is None:
            print(f"  {condition:<28}  [NO RESULTS]")
            continue
        first = True
        for key, metrics in sorted(results.items()):
            model, split = key.rsplit("_", 1)
            label = condition if first else ""
            first = False
            print(f"  {label:<28}  {model:<20}  {split:<12}  "
                  f"{metrics['auc_mean']:>6.3f}±{metrics['auc_std']:.3f}  "
                  f"{metrics['f1_mean']:>6.3f}±{metrics['f1_std']:.3f}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run full ablation study for SimCLR + diffusion pipeline."
    )
    parser.add_argument("--dataset", choices=["elliptic", "ibm", "both"],
                        default="elliptic",
                        help="Dataset (default: elliptic). "
                             "IBM uses ibm_simclr_train_ablation.py for encoder training; "
                             "Elliptic uses elliptic_simclr_train_ablation.py.")
    parser.add_argument("--n-gen",    type=int,   default=40,
                        help="Generated graphs per condition (default 40)")
    parser.add_argument("--low-data", type=float, default=1.0,
                        help="Training data fraction, e.g. 0.2 for low-data regime")
    parser.add_argument("--epochs",   type=int,   default=10,
                        help="SimCLR training epochs per condition (default 100)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip SimCLR retraining if checkpoints already exist")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only run generation ablations (skip encoder ablations)")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Only run encoder ablations (skip generation ablations)")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run only these named conditions (subset of all)")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Select dataset-specific config ───────────────────────────────────────
    if args.dataset == "ibm":
        train_script      = IBM_TRAIN_SCRIPT
        encoder_conditions = IBM_ENCODER_CONDITIONS
        ckpt_root         = BASE_DIR / "checkpoints" / "simclr_ibm_ablation"
    else:
        # "elliptic" or "both" — run encoder training on Elliptic
        train_script      = ELLIPTIC_TRAIN_SCRIPT
        encoder_conditions = ELLIPTIC_ENCODER_CONDITIONS
        ckpt_root         = BASE_DIR / "checkpoints" / "simclr_elliptic_ablation"

    ckpt_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    failures    = []

    # ── A. Encoder ablations ─────────────────────────────────────────────────
    if not args.gen_only:
        print("\n" + "="*60)
        print(f"PART A: SimCLR Encoder Ablations  [{args.dataset}]")
        print("="*60)

        for condition, description, extra_train_args in encoder_conditions:
            if args.conditions and condition not in args.conditions:
                continue

            ckpt_dir  = ckpt_root / condition
            best_ckpt = ckpt_dir / "best_model.pt"

            # ── A1. Train encoder ────────────────────────────────────────────
            if not (args.skip_training and best_ckpt.exists()):
                ok = _run(
                    [train_script,
                     "--condition", condition,
                     "--epochs",    str(args.epochs),
                     ] + extra_train_args,
                    f"Train encoder [{args.dataset}]: {condition} — {description}",
                )
                if not ok:
                    failures.append(f"train:{condition}")
                    continue
            else:
                print(f"\n[skip] Encoder checkpoint exists for '{condition}'")

            # ── A2. Evaluate with augmentation ───────────────────────────────
            label = f"enc_{condition}"
            ok = _run(
                [EVAL_SCRIPT,
                 "--dataset",        args.dataset,
                 "--augment",
                 "--n-gen",          str(args.n_gen),
                 "--encoder-dir",    str(ckpt_dir),
                 "--ablation-label", label,
                 "--low-data",       str(args.low_data),
                 ],
                f"Evaluate (augmented): {condition}",
            )
            if not ok:
                failures.append(f"eval:{condition}")
            all_results[label] = _read_csv_results(label, args.dataset)

        # ── A3. Baseline (no augmentation) using default encoder ─────────────
        if not args.conditions or "baseline" in args.conditions:
            label = "baseline"
            ok = _run(
                [EVAL_SCRIPT,
                 "--dataset",        args.dataset,
                 "--ablation-label", label,
                 "--low-data",       str(args.low_data),
                 ],
                "Evaluate (baseline, no augmentation)",
            )
            if not ok:
                failures.append("eval:baseline")
            all_results[label] = _read_csv_results(label, args.dataset)

    # ── B. Generation ablations ──────────────────────────────────────────────
    if not args.encoder_only:
        print("\n" + "="*60)
        print("PART B: Generation Ablations  (default encoder)")
        print("="*60)

        for condition, description, extra_eval_args in GENERATION_CONDITIONS:
            if args.conditions and condition not in args.conditions:
                continue

            ok = _run(
                [EVAL_SCRIPT,
                 "--dataset",        args.dataset,
                 "--augment",
                 "--n-gen",          str(args.n_gen),
                 "--ablation-label", condition,
                 "--low-data",       str(args.low_data),
                 ] + extra_eval_args,
                f"Evaluate: {condition} — {description}",
            )
            if not ok:
                failures.append(f"eval:{condition}")
            all_results[condition] = _read_csv_results(condition, args.dataset)

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(all_results)

    summary_path = RESULT_DIR / f"ablation_summary_{args.dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {k: v for k, v in all_results.items() if v is not None},
            f, indent=2,
        )
    print(f"\nSummary JSON → {summary_path}")

    if failures:
        print(f"\nFailed conditions: {failures}")
        sys.exit(1)
    else:
        print("\nAll conditions completed successfully.")


if __name__ == "__main__":
    main()
