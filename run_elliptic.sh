#!/bin/bash

#SBATCH --job-name=aml_elliptic
#SBATCH --output=logs/elliptic_%j.txt
#SBATCH --error=logs/elliptic_%j.err
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

# ── Classifier settings ───────────────────────────────────────────────────────
# Skips training and generation — checkpoints must already exist at:
#   igraph_version/checkpoints/diffusion_elliptic/model.pt
#   igraph_version/checkpoints/simclr_elliptic/best_model.pt
#
# Classifier evaluation time estimate (single GPU):
#   6 classifiers × baseline+augmented × 3 seeds  ~90 min
#
N_GEN_CLF=400   # graphs generated for augmentation (evaluate_classifiers.py)
LOW_DATA=1.0    # fraction of training set kept  (1.0 = full)

echo "================================================================"
echo " run_elliptic — classifier evaluation only"
echo " n-gen-clf: $N_GEN_CLF  |  low-data: $LOW_DATA"
echo "================================================================"

python igraph_version/generation/evaluate_classifiers.py \
    --dataset   elliptic \
    --augment \
    --n-gen     $N_GEN_CLF \
    --low-data  $LOW_DATA

EXIT=$?
echo "================================================================"
echo " run_elliptic finished with exit code $EXIT"
echo "================================================================"
exit $EXIT
