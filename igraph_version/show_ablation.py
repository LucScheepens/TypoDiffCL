"""
show_ablation.py
────────────────────────────────────────────────────────────────────────────
Read all ablation CSVs from results/ and print a formatted summary.

Usage
─────
  python show_ablation.py                 # all datasets, all conditions
  python show_ablation.py --dataset ibm   # IBM only
  python show_ablation.py --dataset elliptic
  python show_ablation.py --metric f1     # sort / highlight by F1 instead of AUC
  python show_ablation.py --models GIN GraphSAGE  # show only these models
"""

import argparse
import csv
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Which conditions to show together — kept in logical display order ─────────
SECTION_ORDER = [
    # Encoder ablations
    ("enc_full",             "Full pipeline"),
    ("enc_no_diffusion_aug", "No diffusion aug"),
    ("enc_diffusion_aug_only","Diffusion aug only"),
    ("enc_no_supcon",        "No SupCon"),
    ("enc_ntxent_only",      "NT-Xent only"),
    ("enc_edge_drop_only",   "Edge-drop only"),
    ("enc_feat_mask_only",   "Feat-mask only"),
    ("enc_no_aug",                  "No augmentation"),
    ("enc_diff_multistep",          "Opt-A: multistep DDIM"),
    ("enc_diff_guided",             "Opt-C: guided DDIM"),
    ("enc_diff_multistep_to_guided","Opt-A+C: warmup then guided"),
    # Generation ablations
    ("gen_full_guided",      "Gen: full guided"),
    ("gen_unguided",         "Gen: unguided"),
    ("gen_classif_only",     "Gen: classif only"),
    ("gen_no_novelty",       "Gen: no novelty"),
    ("gen_no_degree_pen",    "Gen: no degree pen"),
    ("gen_high_guidance",    "Gen: high guidance"),
    ("gen_low_guidance",     "Gen: low guidance"),
    # Standalone baselines / other
    ("baseline",             "Baseline (no aug)"),
    ("best",                 "Best (selected)"),
    ("selected",             "Selected"),
    ("ld20",                 "Low-data 20%"),
]
LABEL_MAP = {k: v for k, v in SECTION_ORDER}


def _infer_condition_label(stem):
    """
    Extract the condition name from a CSV filename.
    classifier_comparison_ibm_enc_full  →  ('ibm', 'enc_full')
    classifier_comparison_elliptic_gen_unguided → ('elliptic', 'gen_unguided')
    """
    # Strip prefix
    s = stem.replace("classifier_comparison_", "")
    for ds in ("ibm_ld20", "elliptic_ld20", "ibm", "elliptic"):
        if s.startswith(ds + "_"):
            condition = s[len(ds) + 1:]
            dataset   = ds
            return dataset, condition
        if s == ds:
            return ds, "baseline"
    return "unknown", s


def load_all_csvs(results_dir, dataset_filter=None):
    """
    Returns dict: {(dataset, condition_label): {model: {metric: (mean, std)}}}
    """
    data = {}
    for path in sorted(results_dir.glob("classifier_comparison_*.csv")):
        dataset, condition = _infer_condition_label(path.stem)
        if dataset_filter and dataset_filter not in dataset:
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                cond  = row["condition"]   # 'baseline' or 'augmented'
                key   = (dataset, condition)
                data.setdefault(key, {}).setdefault(model, {})[cond] = {
                    "auc": (float(row["auc_mean"]), float(row["auc_std"])),
                    "f1":  (float(row["f1_mean"]),  float(row["f1_std"])),
                }
    return data


def _fmt(mean, std):
    return f"{mean:.3f}+/-{std:.3f}"


def _delta(val, ref):
    """Format a delta vs reference with +/- sign."""
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.3f}"


def print_table(data, metric="auc", models_filter=None):
    """
    Print one table per dataset.
    Rows = conditions, columns = models × {baseline, augmented, Δ}.
    """
    # Group by dataset
    datasets = sorted({ds for ds, _ in data.keys()})

    for dataset in datasets:
        keys = [(ds, cond) for ds, cond in data.keys() if ds == dataset]
        if not keys:
            continue

        # Determine which models appear
        all_models = []
        for k in keys:
            for m in data[k]:
                if m not in all_models:
                    all_models.append(m)
        if models_filter:
            all_models = [m for m in all_models if m in models_filter]
        if not all_models:
            continue

        # Build ordered condition list
        cond_labels = {cond for _, cond in keys}
        ordered = [c for c, _ in SECTION_ORDER if c in cond_labels]
        ordered += sorted(cond_labels - set(ordered))   # anything not in SECTION_ORDER last

        col_w  = 14   # width per model×split column
        lbl_w  = 22   # condition label width
        n_cols = len(all_models)

        # ── header ─────────────────────────────────────────────────────────
        sep   = "-" * (lbl_w + n_cols * (col_w * 2 + 3) + 2)
        sep2  = "=" * len(sep)
        title = f"  DATASET: {dataset.upper()}   metric: {metric.upper()}"
        print()
        print(sep2)
        print(title)
        print(sep2)

        # model names row
        hdr = f"{'Condition':<{lbl_w}}"
        for m in all_models:
            hdr += f"  {m:^{col_w * 2 + 1}}"
        print(hdr)

        # baseline / augmented / Δ sub-header
        sub = " " * lbl_w
        for _ in all_models:
            sub += f"  {'baseline':^{col_w}} {'augmented':^{col_w}}"
        print(sub)
        print(sep)

        for cond in ordered:
            key = (dataset, cond)
            if key not in data:
                continue
            label = LABEL_MAP.get(cond, cond)

            # Highlight the three diffusion comparison rows
            marker = " *" if cond in ("enc_full", "enc_no_diffusion_aug",
                                       "enc_diffusion_aug_only") else "  "
            row_str = f"{marker}{label:<{lbl_w - 2}}"

            for model in all_models:
                model_data = data[key].get(model, {})
                base_val = model_data.get("baseline", {}).get(metric)
                aug_val  = model_data.get("augmented", {}).get(metric)

                if base_val:
                    row_str += f"  {_fmt(*base_val):^{col_w}}"
                else:
                    row_str += f"  {'n/a':^{col_w}}"

                if aug_val:
                    row_str += f" {_fmt(*aug_val):^{col_w}}"
                else:
                    row_str += f" {'n/a':^{col_w}}"

            print(row_str)

        print(sep)
        print("  * = key diffusion comparison rows")

        # ── Δ summary: augmented - baseline for the diffusion rows ─────────
        print()
        print(f"  Delta augmented - baseline  ({metric.upper()})")
        delta_sep = "-" * (lbl_w + n_cols * (col_w + 2))
        print(f"  {'Condition':<{lbl_w - 2}}", end="")
        for m in all_models:
            print(f"  {m:^{col_w}}", end="")
        print()
        print("  " + delta_sep)

        for cond in ordered:
            key = (dataset, cond)
            if key not in data:
                continue
            label = LABEL_MAP.get(cond, cond)
            row_str = f"  {label:<{lbl_w - 2}}"
            has_delta = False
            for model in all_models:
                model_data = data[key].get(model, {})
                base_val = model_data.get("baseline", {}).get(metric)
                aug_val  = model_data.get("augmented", {}).get(metric)
                if base_val and aug_val:
                    row_str += f"  {_delta(aug_val[0], base_val[0]):^{col_w}}"
                    has_delta = True
                else:
                    row_str += f"  {'n/a':^{col_w}}"
            if has_delta:
                print(row_str)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Print formatted ablation study summary from results CSVs."
    )
    parser.add_argument("--dataset",  default=None,
                        choices=["ibm", "elliptic", "ibm_ld20", "elliptic_ld20"],
                        help="Filter to one dataset (default: all)")
    parser.add_argument("--metric",   default="auc", choices=["auc", "f1"],
                        help="Metric to display (default: auc)")
    parser.add_argument("--models",   nargs="*", default=None,
                        metavar="MODEL",
                        help="Show only these model names, e.g. --models GIN GraphSAGE")
    args = parser.parse_args()

    data = load_all_csvs(RESULTS_DIR, dataset_filter=args.dataset)

    if not data:
        print(f"No CSVs found in {RESULTS_DIR}")
        return

    print_table(data, metric=args.metric, models_filter=args.models)


if __name__ == "__main__":
    main()
