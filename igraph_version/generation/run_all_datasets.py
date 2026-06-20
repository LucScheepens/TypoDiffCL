"""
run_all_datasets.py — Train and evaluate the full pipeline on every IBM HI/LI
dataset variant (HI-Small, HI-Medium, HI-Large, LI-Small, LI-Medium, LI-Large).

For each dataset the script runs, in order:
  1. Diffusion training        (train.py --phase diffusion)
  2. SimCLR training           (train.py --phase simclr)
  3. Guided generation         (generation/test.py)
  4. Classifier evaluation     (generation/evaluate_classifiers.py --augment)

Checkpoints are saved to checkpoints/<dataset-name>/diffusion/ and simclr/.
Results are saved to results/<dataset-name>/.
Classifier CSVs are saved to results/classifier_comparison_ibm_<dataset-name>.csv.

Usage:
    # Full run on all 6 datasets
    python generation/run_all_datasets.py

    # Only small datasets (fast smoke-test)
    python generation/run_all_datasets.py --sizes small

    # Only HI datasets
    python generation/run_all_datasets.py --variants HI

    # Skip (re)training if checkpoints already exist
    python generation/run_all_datasets.py --skip-training

    # Train only, no generation or classifier evaluation
    python generation/run_all_datasets.py --train-only

    # Skip classifier evaluation (generation only)
    python generation/run_all_datasets.py --no-clf

    # Evaluate only (checkpoints must already exist)
    python generation/run_all_datasets.py --eval-only

    # Control generation and classifier args
    python generation/run_all_datasets.py --n-gen 16 --n-gen-clf 40 --sep-check
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT_DIR / "train.py"
TEST_SCRIPT  = Path(__file__).resolve().parent / "test.py"
EVAL_SCRIPT  = Path(__file__).resolve().parent / "evaluate_classifiers.py"
DATA_DIR     = Path(__file__).resolve().parent.parent.parent / "data" / "IBM"

DATASETS = [
    ("HI-Small",  DATA_DIR / "HI-Small_Trans.csv"),
    ("HI-Medium", DATA_DIR / "HI-Medium_Trans.csv"),
    ("HI-Large",  DATA_DIR / "HI-Large_Trans.csv"),
    ("LI-Small",  DATA_DIR / "LI-Small_Trans.csv"),
    ("LI-Medium", DATA_DIR / "LI-Medium_Trans.csv"),
    ("LI-Large",  DATA_DIR / "LI-Large_Trans.csv"),
]

SIZE_MAP = {"small": "Small", "medium": "Medium", "large": "Large"}


def _run(cmd: list, label: str) -> tuple[bool, float]:
    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    t0  = time.time()
    ret = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=str(ROOT_DIR))
    elapsed = time.time() - t0
    ok = ret.returncode == 0
    status = "OK" if ok else f"FAILED (exit {ret.returncode})"
    print(f"  [{status}]  {elapsed/60:.1f} min")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Train + evaluate the IBM pipeline on all HI/LI dataset variants."
    )
    parser.add_argument(
        "--sizes", nargs="+", choices=["small", "medium", "large"],
        default=["small", "medium", "large"],
        help="Dataset sizes to include (default: all three)",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=["HI", "LI"],
        default=["HI", "LI"],
        help="Dataset ratio variants to include (default: both)",
    )
    # ── training control ──────────────────────────────────────────────────────
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training if checkpoints already exist for a dataset")
    parser.add_argument("--train-only", action="store_true",
                        help="Run training only, skip generation and classifier evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run generation + classifier evaluation only (checkpoints must exist)")
    parser.add_argument("--no-clf", action="store_true",
                        help="Skip classifier evaluation; run training + generation only")
    parser.add_argument("--phase", choices=["diffusion", "simclr", "all"], default="all",
                        help="Training phase to run (default: all)")
    # ── generation args (passed through to test.py) ───────────────────────────
    parser.add_argument("--n-gen", type=int, default=8,
                        help="Networks to generate for the quality evaluation step (default 8)")
    parser.add_argument("--t-start", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--novelty-weight", type=float, default=None)
    parser.add_argument("--degree-penalty", type=float, default=None)
    parser.add_argument("--sep-check", action="store_true",
                        help="Run embedding separation diagnostic")
    parser.add_argument("--tune-guidance", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=15)
    # ── classifier evaluation args ────────────────────────────────────────────
    parser.add_argument("--n-gen-clf", type=int, default=40,
                        help="Augmentation graphs generated for classifier training (default 40)")
    parser.add_argument("--low-data", type=float, default=1.0, metavar="FRAC",
                        help="Subsample training set to this fraction before augmenting "
                             "(default 1.0 = full set)")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                        help="Restrict classifier evaluation to these models, e.g. "
                             "--models GIN GraphSAGE. Default: all.")
    parser.add_argument("--elliptic", action="store_true",
                        help="Also run the Elliptic Bitcoin dataset pipeline "
                             "(train, generate, classify) after all IBM datasets. "
                             "Uses checkpoints/diffusion_elliptic/ and checkpoints/simclr_elliptic/.")
    parser.add_argument("--only-elliptic", action="store_true",
                        help="Run ONLY the Elliptic Bitcoin pipeline — skip all IBM datasets. "
                             "Implies --elliptic.")
    args = parser.parse_args()

    if args.only_elliptic:
        args.elliptic = True

    # ── filter datasets ───────────────────────────────────────────────────────
    wanted_sizes    = {SIZE_MAP[s] for s in args.sizes}
    wanted_variants = set(args.variants)
    selected = [] if args.only_elliptic else [
        (name, path) for name, path in DATASETS
        if any(name.startswith(v) for v in wanted_variants)
        and any(name.endswith(s) for s in wanted_sizes)
    ]

    if not selected and not args.elliptic:
        print("No datasets matched the given --sizes / --variants filters.")
        sys.exit(1)

    run_clf = not args.train_only and not args.no_clf

    print(f"Datasets ({len(selected)}): {', '.join(n for n, _ in selected)}")
    steps = []
    if not args.eval_only:
        steps.append(f"train (phase={args.phase})")
    if not args.train_only:
        steps.append("generate")
    if run_clf:
        steps.append("classify")
    print(f"Steps: {' → '.join(steps)}")

    # ── shared args ───────────────────────────────────────────────────────────
    gen_extra: list[str] = ["--n-gen", str(args.n_gen)]
    if args.t_start is not None:
        gen_extra += ["--t-start", str(args.t_start)]
    if args.guidance_scale is not None:
        gen_extra += ["--guidance-scale", str(args.guidance_scale)]
    if args.novelty_weight is not None:
        gen_extra += ["--novelty-weight", str(args.novelty_weight)]
    if args.degree_penalty is not None:
        gen_extra += ["--degree-penalty", str(args.degree_penalty)]
    if args.sep_check:
        gen_extra.append("--sep-check")
    if args.tune_guidance:
        gen_extra += ["--tune-guidance", "--tune-trials", str(args.tune_trials)]

    clf_extra: list[str] = ["--n-gen", str(args.n_gen_clf)]
    if args.low_data < 1.0:
        clf_extra += ["--low-data", str(args.low_data)]
    if args.models:
        clf_extra += ["--models"] + args.models

    # ── per-dataset loop ──────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    total_start = time.time()

    for name, csv_path in selected:
        print(f"\n{'='*64}")
        print(f"  Dataset: {name}")
        print(f"{'='*64}")

        if not csv_path.exists():
            print(f"  [SKIP] file not found: {csv_path}")
            results[name] = {"status": "skipped (file not found)"}
            continue

        ckpt_dir    = ROOT_DIR / "checkpoints" / name
        results_dir = ROOT_DIR / "results"    / name
        dataset_results: dict = {"train": None, "gen": None, "clf": None}

        # ── 1. Training ───────────────────────────────────────────────────────
        if not args.eval_only:
            diff_ckpt   = ckpt_dir / "diffusion" / "model.pt"
            simclr_ckpt = ckpt_dir / "simclr"    / "best_model.pt"
            ckpts_exist = diff_ckpt.exists() and simclr_ckpt.exists()

            if args.skip_training and ckpts_exist:
                print(f"  [skip training] checkpoints found in {ckpt_dir}")
                dataset_results["train"] = "skipped (exists)"
            else:
                ok, elapsed = _run(
                    [TRAIN_SCRIPT,
                     "--dataset", "ibm",
                     "--ibm-csv", str(csv_path),
                     "--ckpt-dir", str(ckpt_dir),
                     "--phase", args.phase],
                    f"Train [{name}]  phase={args.phase}",
                )
                dataset_results["train"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"
                if not ok:
                    results[name] = dataset_results
                    continue  # skip eval if training failed

        # ── 2. Guided generation / quality evaluation ─────────────────────────
        if not args.train_only:
            ok, elapsed = _run(
                [TEST_SCRIPT,
                 "--dataset", "ibm",
                 "--ibm-csv", str(csv_path),
                 "--ckpt-dir", str(ckpt_dir),
                 "--results-dir", str(results_dir)]
                + gen_extra,
                f"Generate [{name}]",
            )
            dataset_results["gen"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"

        # ── 3. Classifier evaluation ──────────────────────────────────────────
        if run_clf:
            ok, elapsed = _run(
                [EVAL_SCRIPT,
                 "--dataset", "ibm",
                 "--ibm-csv", str(csv_path),
                 "--ckpt-dir", str(ckpt_dir),
                 "--augment",
                 "--ablation-label", name]
                + clf_extra,
                f"Classify [{name}]  (results/classifier_comparison_ibm_{name}.csv)",
            )
            dataset_results["clf"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"

        results[name] = dataset_results

    # ── Elliptic dataset ──────────────────────────────────────────────────────
    if args.elliptic:
        name = "elliptic"
        print(f"\n{'='*64}")
        print(f"  Dataset: {name}  (Elliptic Bitcoin)")
        print(f"{'='*64}")

        ell_diff_ckpt   = ROOT_DIR / "checkpoints" / "diffusion_elliptic" / "model.pt"
        ell_simclr_ckpt = ROOT_DIR / "checkpoints" / "simclr_elliptic"    / "best_model.pt"
        ell_results: dict = {"train": None, "gen": None, "clf": None}
        ell_ok = True

        # 1. Training
        if not args.eval_only:
            ckpts_exist = ell_diff_ckpt.exists() and ell_simclr_ckpt.exists()
            if args.skip_training and ckpts_exist:
                print(f"  [skip training] checkpoints found in {ell_diff_ckpt.parent.parent}")
                ell_results["train"] = "skipped (exists)"
            else:
                ok, elapsed = _run(
                    [TRAIN_SCRIPT, "--dataset", "elliptic", "--phase", args.phase],
                    f"Train [elliptic]  phase={args.phase}",
                )
                ell_results["train"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"
                if not ok:
                    ell_ok = False

        # 2. Guided generation / quality evaluation
        if ell_ok and not args.train_only:
            ok, elapsed = _run(
                [TEST_SCRIPT, "--dataset", "elliptic"] + gen_extra,
                "Generate [elliptic]",
            )
            ell_results["gen"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"

        # 3. Classifier evaluation
        if ell_ok and run_clf:
            ok, elapsed = _run(
                [EVAL_SCRIPT,
                 "--dataset", "elliptic",
                 "--augment"]
                + clf_extra,
                "Classify [elliptic]  (results/classifier_comparison_elliptic.csv)",
            )
            ell_results["clf"] = f"{'ok' if ok else 'FAILED'}  ({elapsed/60:.1f} min)"

        results[name] = ell_results

    # ── summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*64}")
    print(f"Summary  (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*64}")
    for name, r in results.items():
        if "status" in r:
            print(f"  ✗  {name:15s}  {r['status']}")
            continue
        parts = []
        if r["train"] is not None:
            parts.append(f"train={r['train']}")
        if r["gen"] is not None:
            parts.append(f"gen={r['gen']}")
        if r["clf"] is not None:
            parts.append(f"clf={r['clf']}")
        all_ok = all(
            v is None or v.startswith("ok") or v.startswith("skipped")
            for v in (r["train"], r["gen"], r["clf"])
        )
        print(f"  {'✓' if all_ok else '✗'}  {name:15s}  {'  '.join(parts)}")

    any_failed = any(
        "status" not in r and
        any(v is not None and "FAILED" in v for v in (r.get("train"), r.get("gen"), r.get("clf")))
        for r in results.values()
    )
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
