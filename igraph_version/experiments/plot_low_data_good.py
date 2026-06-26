"""Plot recovery curves from low_data_results_good.csv."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results" / "low_data_results_good.csv"
OUT_PATH = HERE / "results" / "low_data_curves_good2.png"

LABEL_MAP = {"diffusion": "TypoDiffCL"}

COLOURS = {
    "baseline":   "#4a90d9",
    "diffusion":  "#e05c2e",
    "graphsmote": "#2ecc71",
    "gan":        "#9b59b6",
    "vae":        "#f39c12",
    "diga":       "#1abc9c",
}
MARKERS = {
    "baseline":   "o",
    "diffusion":  "s",
    "graphsmote": "^",
    "gan":        "D",
    "vae":        "P",
    "diga":       "X",
}

with open(CSV_PATH, newline="") as f:
    records = [
        {k: (float(v) if k != "condition" else v) for k, v in row.items()}
        for row in csv.DictReader(f)
    ]

conditions = list(dict.fromkeys(r["condition"] for r in records))
fracs = sorted(set(r["fraction"] for r in records))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for metric, ax, ylabel in [("auc", axes[0], "AUC-ROC"), ("f1", axes[1], "F1 Score")]:
    for cond in conditions:
        xs, ys, errs = [], [], []
        for frac in fracs:
            match = [r for r in records if r["fraction"] == frac and r["condition"] == cond]
            if match:
                xs.append(frac * 100)
                ys.append(match[0][f"{metric}_mean"])
                errs.append(match[0][f"{metric}_std"])
        if xs:
            label = LABEL_MAP.get(cond, cond)
            ax.errorbar(xs, ys, yerr=errs, label=label,
                        color=COLOURS.get(cond, "grey"), marker=MARKERS.get(cond, "o"),
                        linewidth=1.8, markersize=6, capsize=3)
    ax.set_xlabel("Training laundering fraction (%)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=13)
    ax.set_xlim(0, 105)
    ax.set_xticks([5, 10, 25, 50, 100])
    ax.set_xticklabels(["5", "10", "25%", "50%", "100%"])
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Low-Data Regime: Augmentation Recovery Curves", fontsize=12, y=1.01)
fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
print(f"Plot saved -> {OUT_PATH}")
plt.close(fig)
