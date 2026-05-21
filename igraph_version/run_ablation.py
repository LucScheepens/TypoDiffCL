"""
run_ablation.py
──────────────────────────────────────────────────────────────────────────────
Orchestrates the full ablation study for the SimCLR + diffusion pipeline.

Three families of ablations are run:

  A. SimCLR encoder ablations  (require re-training the encoder)
     Vary: augmentation type, supervised contrastive loss, diffusion views
     Each variant trains a new encoder and then evaluates on the classifier.

  B. Generation ablations  (use the default trained encoder, vary generation)
     Vary: guidance scale, novelty weight, degree penalty
     No retraining needed — only evaluate_classifiers.py is re-run.

  C. AML pattern feature ablations  (IBM only, require re-training)
     Systematically zero out each group of pattern features (fan-in/out,
     stack, cycle, scatter-gather, bipartite) to isolate their contribution.
     Compares pat_full vs. pat_none vs. each individual group removed.

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

  # Run only AML pattern feature ablations (IBM, Part C)
  python run_ablation.py --dataset ibm --n-gen 40 --pattern-only

  # Run a specific pattern condition
  python run_ablation.py --dataset ibm --n-gen 40 --pattern-only --conditions pat_none pat_full

  # Run all ablations but evaluate only one classifier (much faster)
  python run_ablation.py --dataset ibm --n-gen 40 --classifier ExSTraQt

  # Combine with other filters
  python run_ablation.py --dataset elliptic --n-gen 40 --gen-only --classifier GIN GraphSAGE
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

# ── AML pattern feature ablations (IBM only — requires directed tx data) ──────
# Each entry removes one group of AML typological pattern features (cols 11-18)
# and retrains the encoder.  Comparing each variant to "full" shows the
# individual contribution of each pattern type to classification performance.
IBM_PATTERN_CONDITIONS = [
    (
        "pat_full",
        "All 8 AML pattern features active (reference for pattern ablation)",
        [],
    ),
    (
        "pat_none",
        "No AML pattern features at all (cols 11-18 zeroed) — measures total contribution",
        ["--no-pattern-features"],
    ),
    (
        "pat_no_fan",
        "No fan-out / fan-in / asymmetry features (cols 11-13 zeroed)",
        ["--no-fan-features"],
    ),
    (
        "pat_no_stack",
        "No stack / passthrough / chain-depth features (cols 14-15 zeroed)",
        ["--no-stack-features"],
    ),
    (
        "pat_no_cycle",
        "No in-cycle feature (col 16 zeroed) — round-tripping / U-turn signal",
        ["--no-cycle-feature"],
    ),
    (
        "pat_no_sg",
        "No scatter-gather score (col 17 zeroed) — bipartite fan-out→fan-in bridge",
        ["--no-sg-feature"],
    ),
    (
        "pat_no_bipartite",
        "No graph-level bipartite score (col 18 zeroed)",
        ["--no-bipartite-feature"],
    ),
]

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
    parser.add_argument("--low-data", type=float, default=0.01,
                        help="Training data fraction, e.g. 0.2 for low-data regime")
    parser.add_argument("--epochs",   type=int,   default=100,
                        help="SimCLR training epochs per condition (default 100)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip SimCLR retraining if checkpoints already exist")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only run generation ablations (skip encoder ablations)")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Only run encoder ablations (skip generation ablations)")
    parser.add_argument("--pattern-only", action="store_true",
                        help="Only run AML pattern feature ablations (IBM only, skip other parts)")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run only these named conditions (subset of all)")
    parser.add_argument("--classifier", nargs="+", default=None,
                        metavar="MODEL",
                        help="Restrict evaluation to these classifiers, e.g. "
                             "--classifier GIN ExSTraQt. "
                             "Valid: GIN, GraphTransformer, GraphSAGE, DeepSets, "
                             "FraudGT, ExSTraQt. Default: all.")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Extra args forwarded to every evaluate_classifiers.py call
    _classifier_args = (["--models"] + args.classifier) if args.classifier else []

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
                 ] + _classifier_args,
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
                 ] + _classifier_args,
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
                 ] + extra_eval_args + _classifier_args,
                f"Evaluate: {condition} — {description}",
            )
            if not ok:
                failures.append(f"eval:{condition}")
            all_results[condition] = _read_csv_results(condition, args.dataset)

    # ── C. AML pattern feature ablations (IBM only) ──────────────────────────
    run_pattern = (
        not args.gen_only
        and not args.encoder_only
        and (args.pattern_only or args.dataset in ("ibm", "both"))
    )
    if run_pattern:
        print("\n" + "="*60)
        print("PART C: AML Pattern Feature Ablations  [IBM]")
        print("  Trains one encoder per pattern group, each with that group zeroed out.")
        print("  Compare every row to pat_full to see individual feature contributions.")
        print("="*60)

        pat_ckpt_root = BASE_DIR / "checkpoints" / "simclr_ibm_ablation"
        pat_ckpt_root.mkdir(parents=True, exist_ok=True)

        for condition, description, extra_train_args in IBM_PATTERN_CONDITIONS:
            if args.conditions and condition not in args.conditions:
                continue

            ckpt_dir  = pat_ckpt_root / condition
            best_ckpt = ckpt_dir / "best_model.pt"

            # C1. Train encoder with selected pattern features zeroed out
            if not (args.skip_training and best_ckpt.exists()):
                ok = _run(
                    [IBM_TRAIN_SCRIPT,
                     "--condition", condition,
                     "--epochs",    str(args.epochs),
                     ] + extra_train_args,
                    f"Train IBM encoder [pattern ablation]: {condition} — {description}",
                )
                if not ok:
                    failures.append(f"train:{condition}")
                    continue
            else:
                print(f"\n[skip] Pattern ablation checkpoint exists for '{condition}'")

            # C2. Evaluate
            label = f"pat_{condition}" if not condition.startswith("pat_") else condition
            ok = _run(
                [EVAL_SCRIPT,
                 "--dataset",        "ibm",
                 "--augment",
                 "--n-gen",          str(args.n_gen),
                 "--encoder-dir",    str(ckpt_dir),
                 "--ablation-label", label,
                 "--low-data",       str(args.low_data),
                 ] + _classifier_args,
                f"Evaluate (pattern ablation): {condition}",
            )
            if not ok:
                failures.append(f"eval:{condition}")
            all_results[label] = _read_csv_results(label, "ibm")

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
