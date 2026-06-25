"""
plot_ablation.py
────────────────────────────────────────────────────────────────────────────
Visualise ablation results from results/classifier_comparison_*.csv.

Produces four figures saved to results/plots/:
  ablation_encoder_auc.png      — encoder conditions, AUC baseline vs aug
  ablation_generation_auc.png   — generation conditions, AUC baseline vs aug
  ablation_delta_heatmap.png    — Δ(aug-baseline) heatmap, all conditions × models
  ablation_f1_summary.png       — F1 summary for both ablation groups side-by-side

Usage
─────
  python plot_ablation.py                    # IBM + Elliptic, metric=AUC
  python plot_ablation.py --dataset ibm
  python plot_ablation.py --metric f1
  python plot_ablation.py --models GIN GraphSAGE
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "ablation_full"
PLOTS_DIR   = RESULTS_DIR / "plots"

# ── condition ordering & labels ──────────────────────────────────────────────
ENCODER_CONDITIONS = [
    ("enc_full",                  "Full pipeline"),
    ("enc_no_diffusion_aug",      "No diffusion aug"),
    ("enc_diffusion_aug_only",    "Diffusion aug only"),
    ("enc_no_supcon",             "No SupCon"),
    ("enc_ntxent_only",           "NT-Xent only"),
    ("enc_edge_drop_only",        "Edge-drop only"),
    ("enc_feat_mask_only",        "Feat-mask only"),
    ("enc_no_aug",                "No augmentation"),
    ("enc_diff_multistep",        "Multistep DDIM"),
    ("enc_diff_guided",           "Guided DDIM"),
    ("enc_diff_multistep_to_guided", "Warmup→Guided"),
]

GENERATION_CONDITIONS = [
    ("gen_full_guided",   "Full guided"),
    ("gen_unguided",      "Unguided"),
    ("gen_classif_only",  "Classif only"),
    ("gen_no_novelty",    "No novelty"),
    ("gen_no_degree_pen", "No degree pen"),
    ("gen_high_guidance", "High guidance"),
    ("gen_low_guidance",  "Low guidance"),
]

PATTERN_CONDITIONS = [
    ("pat_full",         "Full patterns"),
    ("pat_none",         "No patterns"),
    ("pat_no_bipartite", "No bipartite"),
    ("pat_no_cycle",     "No cycle"),
    ("pat_no_fan",       "No fan"),
    ("pat_no_sg",        "No SG"),
    ("pat_no_stack",     "No stack"),
]

OTHER_CONDITIONS = [
    ("baseline", "Baseline"),
    ("best",     "Best selected"),
    ("selected", "Selected"),
    ("ld20",     "Low-data 20%"),
]

ALL_CONDITIONS = ENCODER_CONDITIONS + GENERATION_CONDITIONS + PATTERN_CONDITIONS + OTHER_CONDITIONS
LABEL_MAP      = {k: v for k, v in ALL_CONDITIONS}

# ── colours ──────────────────────────────────────────────────────────────────
DATASET_COLOURS = {
    "ibm":          ("#2563EB", "#93C5FD"),   # blue dark/light
    "elliptic":     ("#16A34A", "#86EFAC"),   # green dark/light
    "ibm_ld10":     ("#DC2626", "#FCA5A5"),   # red dark/light
    "ibm_ld1":      ("#0D9488", "#99F6E4"),   # teal
    "ibm_ld20":     ("#9333EA", "#D8B4FE"),   # purple
    "elliptic_ld20":("#EA580C", "#FED7AA"),   # orange
}
DEFAULT_COLOURS = ("#374151", "#9CA3AF")


# ── data loading (mirrors show_ablation.py) ───────────────────────────────────

def _infer_condition(stem):
    s = stem.replace("classifier_comparison_", "")
    # More-specific prefixes must come before shorter ones (ibm_ld10 before ibm)
    for ds in ("ibm_ld20", "ibm_ld10", "ibm_ld1", "elliptic_ld20", "ibm", "elliptic"):
        if s.startswith(ds + "_"):
            return ds, s[len(ds) + 1:]
        if s == ds:
            return ds, "baseline"
    return "unknown", s


def load_all_csvs(results_dir, ablation_dir=None, dataset_filter=None):
    """
    Returns {(dataset, condition): {model: {"baseline": {...}, "augmented": {...}}}}
    Each inner dict has keys: auc, f1, prec, rec  →  (mean, std).

    If ablation_dir is given and differs from results_dir, its CSVs are also
    loaded (used to pull root-level encoder/pattern ablation files into a
    data-dir run).
    """
    dirs = [results_dir]
    if ablation_dir and Path(ablation_dir).resolve() != Path(results_dir).resolve():
        dirs.append(Path(ablation_dir))

    data = {}
    for scan_dir in dirs:
        for path in sorted(scan_dir.glob("classifier_comparison_*.csv")):
            dataset, condition = _infer_condition(path.stem)
            if dataset_filter and dataset_filter not in dataset:
                continue
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    model = row["model"]
                    cond  = row["condition"]
                    key   = (dataset, condition)
                    entry = data.setdefault(key, {}).setdefault(model, {})
                    entry[cond] = {
                        "auc":  (float(row["auc_mean"]),  float(row["auc_std"])),
                        "f1":   (float(row["f1_mean"]),   float(row["f1_std"])),
                        "prec": (float(row["prec_mean"]), float(row["prec_std"])),
                        "rec":  (float(row["rec_mean"]),  float(row["rec_std"])),
                    }
    return data


# ── plotting helpers ──────────────────────────────────────────────────────────

def _bar_group(ax, x_pos, baseline_val, aug_val,
               baseline_err, aug_err,
               col_base, col_aug, bar_w=0.35):
    ax.bar(x_pos - bar_w / 2, baseline_val, bar_w,
           yerr=baseline_err, color=col_base,
           capsize=3, error_kw={"lw": 1}, zorder=3)
    ax.bar(x_pos + bar_w / 2, aug_val,      bar_w,
           yerr=aug_err,      color=col_aug,
           capsize=3, error_kw={"lw": 1}, zorder=3)


def _style_ax(ax, xtick_labels, xtick_pos, ylabel, title, ylim_low=None):
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylim_low is not None:
        ax.set_ylim(bottom=ylim_low)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_legend(ax, col_base, col_aug, label_base="Baseline", label_aug="Augmented"):
    ax.legend(handles=[
        mpatches.Patch(color=col_base, label=label_base),
        mpatches.Patch(color=col_aug,  label=label_aug),
    ], fontsize=8, loc="lower right")


# ── Figure 1 & 2: grouped bar charts ─────────────────────────────────────────

def plot_bar_ablation(data, conditions, metric, datasets, title_suffix, out_path):
    """
    One subplot per dataset.  Rows = conditions, bars = baseline vs augmented,
    one group per GNN model stacked side-by-side within each condition.
    """
    n_ds   = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(max(10, len(conditions) * 1.2) * n_ds, 5),
                             sharey=False, squeeze=False)

    for col_idx, dataset in enumerate(datasets):
        ax     = axes[0, col_idx]
        colors = DATASET_COLOURS.get(dataset, DEFAULT_COLOURS)

        # Collect models that appear in this dataset for these conditions
        models = []
        for cond, _ in conditions:
            key = (dataset, cond)
            if key in data:
                for m in data[key]:
                    if m not in models:
                        models.append(m)

        n_models = len(models)
        if n_models == 0:
            ax.set_visible(False)
            continue

        # x positions: one group per condition
        n_conds  = len(conditions)
        group_w  = n_models * 0.4 + 0.3
        x_groups = np.arange(n_conds) * group_w

        for ci, (cond, label) in enumerate(conditions):
            key = (dataset, cond)
            if key not in data:
                continue
            for mi, model in enumerate(models):
                model_data = data[key].get(model, {})
                base = model_data.get("baseline", {}).get(metric)
                aug  = model_data.get("augmented", {}).get(metric)
                if base is None or aug is None:
                    continue

                offset  = (mi - (n_models - 1) / 2) * 0.35
                x_base  = x_groups[ci] + offset - 0.09
                x_aug   = x_groups[ci] + offset + 0.09
                bw      = 0.15

                ax.bar(x_base, base[0], bw, yerr=base[1], color=colors[0],
                       alpha=0.6 + 0.4 * (mi / max(n_models - 1, 1)),
                       capsize=2, error_kw={"lw": 0.8}, zorder=3, label=None)
                ax.bar(x_aug,  aug[0],  bw, yerr=aug[1],  color=colors[1],
                       alpha=0.6 + 0.4 * (mi / max(n_models - 1, 1)),
                       capsize=2, error_kw={"lw": 0.8}, zorder=3, label=None)

        cond_labels = [lbl for _, lbl in conditions]
        _style_ax(ax, cond_labels, x_groups,
                  ylabel=metric.upper(),
                  title=f"{dataset.upper()} — {title_suffix}",
                  ylim_low=max(0, min(
                      v
                      for cond, _ in conditions
                      for m_data in (data.get((dataset, cond), {}).values())
                      for v in [m_data.get("baseline", {}).get(metric, (0.5,))[0],
                                m_data.get("augmented", {}).get(metric, (0.5,))[0]]
                  ) - 0.05))

        # Legend: model names (baseline=dark, augmented=light)
        handles = []
        for mi, model in enumerate(models):
            alpha = 0.6 + 0.4 * (mi / max(n_models - 1, 1))
            handles.append(mpatches.Patch(color=colors[0], alpha=alpha, label=f"{model} base"))
            handles.append(mpatches.Patch(color=colors[1], alpha=alpha, label=f"{model} aug"))
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  ncol=2, framealpha=0.8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Ablation Study — {title_suffix} ({metric.upper()})",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 3: delta heatmap ───────────────────────────────────────────────────

def plot_delta_heatmap(data, metric, datasets, out_path):
    """
    Rows = models, columns = ablation conditions (merged across datasets).
    Cell colour = Δ metric (augmented − baseline).
    For each (model, condition) cell the value is taken from whichever
    dataset contains that model's data for that condition, so models
    trained on different dataset splits still share the same columns.
    Models with no data at all are dropped.
    """
    all_conds = ENCODER_CONDITIONS + GENERATION_CONDITIONS + PATTERN_CONDITIONS + OTHER_CONDITIONS

    # Collect all models in stable insertion order
    all_models = []
    for key in data:
        for m in data[key]:
            if m not in all_models:
                all_models.append(m)

    # Build column list by condition name, merging across datasets
    seen_conds = set()
    col_labels = []
    col_conds  = []
    for cond, lbl in all_conds:
        if cond in seen_conds:
            continue
        has_any = any(
            data.get((ds, cond), {}).get(m, {}).get("baseline") and
            data.get((ds, cond), {}).get(m, {}).get("augmented")
            for ds in datasets for m in all_models
        )
        if has_any:
            col_labels.append(lbl)
            col_conds.append(cond)
            seen_conds.add(cond)

    if not col_conds:
        print("No data for heatmap.")
        return

    # Build matrix: rows = models, columns = conditions
    # Each cell picks its value from whatever dataset holds that model's data
    matrix = []
    for model in all_models:
        row = []
        for cond in col_conds:
            val = np.nan
            for ds in datasets:
                md   = data.get((ds, cond), {}).get(model, {})
                base = md.get("baseline", {}).get(metric)
                aug  = md.get("augmented", {}).get(metric)
                if base and aug:
                    val = aug[0] - base[0]
                    break
            row.append(val)
        matrix.append(row)

    mat = np.array(matrix, dtype=float)

    # Drop models that are entirely NaN, then transpose: conditions → rows, models → columns
    non_nan_cols = [i for i in range(mat.shape[0]) if not np.all(np.isnan(mat[i]))]
    mat          = mat[non_nan_cols].T          # shape: (n_conditions, n_models)
    x_labels     = [all_models[i] for i in non_nan_cols]   # columns = models
    y_labels     = col_labels                               # rows    = conditions

    n_r, n_c = mat.shape
    fig, ax = plt.subplots(figsize=(max(4, n_c * 1.4), max(4, n_r * 0.45)))

    vmax = max(0.05, np.nanmax(np.abs(mat)))
    im   = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                     vmin=-vmax, vmax=vmax)

    for r in range(n_r):
        for c in range(n_c):
            v = mat[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")

    ax.set_yticks(range(n_r))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_title(f"Δ {metric.upper()} (augmented − baseline)",
                 fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label=f"Δ {metric.upper()}", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 4: F1 summary side-by-side ────────────────────────────────────────

def plot_f1_summary(data, datasets, out_path):
    """
    Side-by-side bar chart: F1 augmented for encoder ablations (left)
    and generation ablations (right), one dataset per panel row.
    """
    n_ds   = len(datasets)
    fig, axes = plt.subplots(n_ds, 3,
                             figsize=(24, 4 * n_ds),
                             squeeze=False)

    groups = [
        (ENCODER_CONDITIONS,    "Encoder ablation"),
        (GENERATION_CONDITIONS, "Generation ablation"),
        (PATTERN_CONDITIONS,    "Pattern ablation"),
    ]

    for row_i, dataset in enumerate(datasets):
        colors = DATASET_COLOURS.get(dataset, DEFAULT_COLOURS)

        for col_i, (conditions, group_title) in enumerate(groups):
            ax = axes[row_i, col_i]

            models = []
            for cond, _ in conditions:
                key = (dataset, cond)
                if key in data:
                    for m in data[key]:
                        if m not in models:
                            models.append(m)

            cond_keys   = [c for c, _ in conditions if (dataset, c) in data]
            cond_labels = [LABEL_MAP.get(c, c) for c in cond_keys]
            x           = np.arange(len(cond_keys))
            bar_w       = 0.8 / max(len(models), 1)

            for mi, model in enumerate(models):
                f1_vals = []
                f1_errs = []
                for cond in cond_keys:
                    md  = data.get((dataset, cond), {}).get(model, {})
                    aug = md.get("augmented", {}).get("f1")
                    if aug:
                        f1_vals.append(aug[0])
                        f1_errs.append(aug[1])
                    else:
                        f1_vals.append(0.0)
                        f1_errs.append(0.0)

                offset = (mi - (len(models) - 1) / 2) * bar_w
                alpha  = 0.55 + 0.45 * (mi / max(len(models) - 1, 1))
                ax.bar(x + offset, f1_vals, bar_w * 0.9,
                       yerr=f1_errs, label=model,
                       color=colors[mi % 2], alpha=alpha,
                       capsize=3, error_kw={"lw": 0.8}, zorder=3)

            _style_ax(ax, cond_labels, x,
                      ylabel="F1 (augmented)",
                      title=f"{dataset.upper()} — {group_title}")
            if models:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.8)

    fig.suptitle("F1 Score — Augmented Condition by Ablation Group",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 5: scatter baseline vs augmented ──────────────────────────────────

def plot_scatter(data, metric, datasets, out_path):
    """
    Scatter: x = baseline metric, y = augmented metric.
    Points above the diagonal improved with augmentation.
    Colour by dataset, marker by condition type.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    markers = {"enc": "o", "gen": "s", "pat": "D", "other": "^"}
    all_vals = []
    labeled_datasets = set()

    for dataset in datasets:
        colors = DATASET_COLOURS.get(dataset, DEFAULT_COLOURS)

        for cond, lbl in ALL_CONDITIONS:
            key = (dataset, cond)
            if key not in data:
                continue
            if cond.startswith("enc"):
                mtype = "enc"
            elif cond.startswith("gen"):
                mtype = "gen"
            elif cond.startswith("pat"):
                mtype = "pat"
            else:
                mtype = "other"
            mk = markers[mtype]

            for model, md in data[key].items():
                base = md.get("baseline", {}).get(metric)
                aug  = md.get("augmented", {}).get(metric)
                if base and aug:
                    add_label = dataset not in labeled_datasets
                    ax.scatter(base[0], aug[0],
                               color=colors[0], marker=mk,
                               s=50, alpha=0.75, zorder=3,
                               label=dataset if add_label else "")
                    if add_label:
                        labeled_datasets.add(dataset)
                    ax.annotate(f"{lbl[:12]}\n{model}",
                                (base[0], aug[0]),
                                fontsize=5, alpha=0.6,
                                textcoords="offset points", xytext=(3, 3))
                    all_vals.extend([base[0], aug[0]])

    if not all_vals:
        plt.close(fig)
        return

    lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="No change")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                    alpha=0.04, color="green", label="Aug improves")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                    alpha=0.04, color="red",   label="Aug hurts")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Baseline {metric.upper()}", fontsize=10)
    ax.set_ylabel(f"Augmented {metric.upper()}", fontsize=10)
    ax.set_title(f"Baseline vs Augmented {metric.upper()} — all conditions",
                 fontsize=11, fontweight="bold")

    # Custom legend for dataset colours and marker types
    legend_handles = []
    for ds in datasets:
        c = DATASET_COLOURS.get(ds, DEFAULT_COLOURS)[0]
        legend_handles.append(mpatches.Patch(color=c, label=ds.upper()))
    for mtype, mk in markers.items():
        legend_handles.append(plt.Line2D([0], [0], marker=mk, color="gray",
                                          linestyle="None", ms=7,
                                          label=f"{mtype} conditions"))
    legend_handles.append(plt.Line2D([0], [0], color="k", ls="--", label="No change"))
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results.")
    parser.add_argument("--dataset", default=None,
                        choices=["ibm", "elliptic", "ibm_ld10", "ibm_ld20", "elliptic_ld20"],
                        help="Filter to one dataset (default: all)")
    parser.add_argument("--metric",  default="auc", choices=["auc", "f1", "prec", "rec"])
    parser.add_argument("--models",  nargs="*", default=None, metavar="MODEL")
    parser.add_argument("--data-dir", default=None, metavar="DIR",
                        help="Folder containing the main comparison CSVs (e.g. 'results/ibm 10000'). "
                             "Root-level ablation files are always loaded from results/ as well.")
    args = parser.parse_args()

    # Resolve directories
    if args.data_dir:
        data_dir  = Path(args.data_dir)
        plots_dir = data_dir / "plots"
    else:
        data_dir  = RESULTS_DIR
        plots_dir = PLOTS_DIR

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load: primary data from data_dir; ablation files always from RESULTS_DIR
    data = load_all_csvs(data_dir, ablation_dir=RESULTS_DIR, dataset_filter=args.dataset)
    if not data:
        print(f"No CSVs found in {data_dir}")
        sys.exit(1)

    # Filter models
    if args.models:
        filtered = {}
        for key, mdict in data.items():
            filt = {m: v for m, v in mdict.items() if m in args.models}
            if filt:
                filtered[key] = filt
        data = filtered
        if not data:
            print(f"No data remaining after filtering models to {args.models}")
            sys.exit(1)

    # Determine which datasets are present
    datasets = sorted({ds for ds, _ in data.keys()})
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d]

    m = args.metric

    # Figure 1 — encoder ablation bar chart
    plot_bar_ablation(
        data, ENCODER_CONDITIONS, m, datasets,
        title_suffix="Encoder ablation",
        out_path=plots_dir / f"ablation_encoder_{m}.png",
    )

    # Figure 2 — generation ablation bar chart
    plot_bar_ablation(
        data, GENERATION_CONDITIONS, m, datasets,
        title_suffix="Generation ablation",
        out_path=plots_dir / f"ablation_generation_{m}.png",
    )

    # Figure 2b — pattern ablation bar chart
    plot_bar_ablation(
        data, PATTERN_CONDITIONS, m, datasets,
        title_suffix="Pattern ablation",
        out_path=plots_dir / f"ablation_pattern_{m}.png",
    )

    # Figure 3 — delta heatmap
    plot_delta_heatmap(
        data, m, datasets,
        out_path=plots_dir / f"ablation_delta_heatmap_{m}.png",
    )

    # Figure 4 — F1 summary
    plot_f1_summary(
        data, datasets,
        out_path=plots_dir / "ablation_f1_summary.png",
    )

    # Figure 5 — scatter baseline vs augmented
    plot_scatter(
        data, m, datasets,
        out_path=plots_dir / f"ablation_scatter_{m}.png",
    )

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
