#!/bin/bash

#SBATCH --job-name=aml_ethereum
#SBATCH --output=logs/ethereum_%j.txt
#SBATCH --error=logs/ethereum_%j.err
#SBATCH --partition=tue.gpu.q
#SBATCH --time=09:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Python/3.11.3-GCCcore-12.3.0

source ~/venvs/igraph311/bin/activate

# ── Sanity checks ─────────────────────────────────────────────────────────────
which python
python --version
python -c "import igraph; print('igraph:', igraph.__file__)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR="$HOME/"

mkdir -p "$WORK_DIR/logs"

cd "$WORK_DIR"

# ── Generation & classifier settings ─────────────────────────────────────────
# Time budget breakdown (single A100/V100, 9-hour wall clock):
#
#   Pickle load + ego-subgraph extraction  ~30 min  (first run; cached afterwards)
#     Loading 1.2 GB MulDiGraph.pkl + BFS over 3M-node graph for ~1165 labeled nodes
#   Diffusion training                     ~60 min  (150 epochs, NODE_DIM=6)
#   SimCLR training                        ~20 min  (80 epochs, ~1k graphs)
#   Guided generation                      ~20 min  (N_GEN_CLF graphs)
#   Classifier evaluation                  ~90 min  (GIN, GT, SAGE, DeepSets, FraudGT
#                                                    × baseline + augmented × 3 seeds)
#   ─────────────────────────────────────────────────────────────────────────
#   Total estimate                         ~3.5 h   (well within 9-hour limit)
#
# Epoch counts are set in igraph_version/train.py CONFIG:
#   DIFF_ETH_EPOCHS    = 150
#   SIMCLR_ETH_EPOCHS  = 80
#
N_GEN_CLF=200   # graphs generated for classifier augmentation
LOW_DATA=1.0    # fraction of training set kept (1.0 = full)

echo "================================================================"
echo " run_ethereum — Ethereum Phishing Transaction Network"
echo " n-gen-clf : $N_GEN_CLF  |  low-data: $LOW_DATA"
echo "================================================================"

# ── Step 1: Train diffusion model ─────────────────────────────────────────────
echo ""
echo "----------------------------------------------------------------"
echo " STEP 1: Diffusion training (ethereum)"
echo "----------------------------------------------------------------"
python igraph_version/train.py --dataset ethereum --phase diffusion
DIFF_EXIT=$?
if [ $DIFF_EXIT -ne 0 ]; then
    echo "ERROR: Diffusion training failed (exit $DIFF_EXIT). Aborting."
    exit $DIFF_EXIT
fi
echo "Diffusion training complete."

# ── Step 2: Train SimCLR encoder ─────────────────────────────────────────────
echo ""
echo "----------------------------------------------------------------"
echo " STEP 2: SimCLR training (ethereum)"
echo "----------------------------------------------------------------"
python igraph_version/train.py --dataset ethereum --phase simclr
SIMCLR_EXIT=$?
if [ $SIMCLR_EXIT -ne 0 ]; then
    echo "ERROR: SimCLR training failed (exit $SIMCLR_EXIT). Aborting."
    exit $SIMCLR_EXIT
fi
echo "SimCLR training complete."

# ── Step 3: Classifier evaluation (baseline + augmented) ─────────────────────
echo ""
echo "----------------------------------------------------------------"
echo " STEP 3: Classifier evaluation (baseline + augmented)"
echo "----------------------------------------------------------------"
python igraph_version/generation/evaluate_classifiers.py \
    --dataset   ethereum \
    --augment \
    --n-gen     $N_GEN_CLF \
    --low-data  $LOW_DATA
CLF_EXIT=$?
if [ $CLF_EXIT -ne 0 ]; then
    echo "ERROR: Classifier evaluation failed (exit $CLF_EXIT)."
    exit $CLF_EXIT
fi

echo ""
echo "================================================================"
echo " run_ethereum finished successfully."
echo " Results: igraph_version/results/classifier_comparison_ethereum.csv"
echo "================================================================"
exit 0
