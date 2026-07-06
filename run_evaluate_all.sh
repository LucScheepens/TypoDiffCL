#!/bin/bash

#SBATCH --job-name=aml_eval_all
#SBATCH --output=logs/eval_all_%j.txt
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --partition=tue.gpu.q
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Python/3.11.3-GCCcore-12.3.0

source ~/venvs/igraph311/bin/activate

which python
python --version
python -c "import igraph; print('igraph:', igraph.__file__)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR="$HOME/"
mkdir -p "${WORK_DIR}logs"
cd "$WORK_DIR"

EVAL_SCRIPT="igraph_version/generation/evaluate_classifiers.py"
IBM_DATA_DIR="${WORK_DIR}data/IBM"
CKPT_BASE="${WORK_DIR}igraph_version/checkpoints"

# ── Settings ──────────────────────────────────────────────────────────────────
# Checkpoints must already exist at:
#   $CKPT_BASE/<dataset>/diffusion/model.pt
#   $CKPT_BASE/<dataset>/simclr/best_model.pt  (or epoch_*.pt)
#   $CKPT_BASE/diffusion_elliptic/model.pt
#   $CKPT_BASE/simclr_elliptic/best_model.pt
#
# Runtime estimate (single GPU, remove-hubs=10):
#   6 IBM variants × ~90 min + elliptic ~90 min  ≈  11–12 h
#
N_GEN_CLF=400   # generated graphs used for augmentation in each run
LOW_DATA=1.0    # fraction of training data to keep (1.0 = full)
REMOVE_HUBS=10  # top-N highest-degree accounts removed before ego extraction

# ── IBM datasets ──────────────────────────────────────────────────────────────
# Each entry: "<dataset-name>"
IBM_DATASETS=(
    "HI-Small"
    "HI-Medium"
    "HI-Large"
    "LI-Small"
    "LI-Medium"
    "LI-Large"
)

FAILED=()

for NAME in "${IBM_DATASETS[@]}"; do
    CSV_PATH="${IBM_DATA_DIR}/${NAME}_Trans.csv"
    CKPT_DIR="${CKPT_BASE}/${NAME}"

    echo "================================================================"
    echo " IBM dataset: ${NAME}"
    echo " csv:  ${CSV_PATH}"
    echo " ckpt: ${CKPT_DIR}"
    echo " n-gen-clf: ${N_GEN_CLF}  |  low-data: ${LOW_DATA}  |  remove-hubs: ${REMOVE_HUBS}"
    echo "================================================================"

    if [ ! -f "${CSV_PATH}" ]; then
        echo "[SKIP] CSV not found: ${CSV_PATH}"
        FAILED+=("${NAME} (csv missing)")
        continue
    fi
    if [ ! -f "${CKPT_DIR}/diffusion/model.pt" ]; then
        echo "[SKIP] diffusion checkpoint not found: ${CKPT_DIR}/diffusion/model.pt"
        FAILED+=("${NAME} (diffusion ckpt missing)")
        continue
    fi

    python "${EVAL_SCRIPT}" \
        --dataset      ibm \
        --ibm-csv      "${CSV_PATH}" \
        --ckpt-dir     "${CKPT_DIR}" \
        --augment \
        --n-gen        "${N_GEN_CLF}" \
        --low-data     "${LOW_DATA}" \
        --remove-hubs  "${REMOVE_HUBS}" \
        --ablation-label "${NAME}"

    EXIT=$?
    echo "----------------------------------------------------------------"
    echo " ${NAME} finished — exit code ${EXIT}"
    echo "----------------------------------------------------------------"
    [ "${EXIT}" -ne 0 ] && FAILED+=("${NAME} (exit ${EXIT})")
done

# ── Elliptic dataset ──────────────────────────────────────────────────────────
echo "================================================================"
echo " Elliptic Bitcoin dataset"
echo " n-gen-clf: ${N_GEN_CLF}  |  low-data: ${LOW_DATA}"
echo " (hub removal not applied — Elliptic graphs are pre-processed)"
echo "================================================================"

ELLIPTIC_DIFF_CKPT="${CKPT_BASE}/diffusion_elliptic/model.pt"
ELLIPTIC_SIMCLR_CKPT="${CKPT_BASE}/simclr_elliptic/best_model.pt"

if [ ! -f "${ELLIPTIC_DIFF_CKPT}" ]; then
    echo "[SKIP] Elliptic diffusion checkpoint not found: ${ELLIPTIC_DIFF_CKPT}"
    FAILED+=("elliptic (diffusion ckpt missing)")
else

python "${EVAL_SCRIPT}" \
    --dataset  elliptic \
    --augment \
    --n-gen    "${N_GEN_CLF}" \
    --low-data "${LOW_DATA}"

EXIT=$?
echo "----------------------------------------------------------------"
echo " elliptic finished — exit code ${EXIT}"
echo "----------------------------------------------------------------"
[ "${EXIT}" -ne 0 ] && FAILED+=("elliptic (exit ${EXIT})")

fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo " run_evaluate_all — done"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo " All datasets completed successfully."
    EXIT_FINAL=0
else
    echo " Failed or skipped:"
    for ITEM in "${FAILED[@]}"; do
        echo "   ✗  ${ITEM}"
    done
    EXIT_FINAL=1
fi
echo "================================================================"
exit "${EXIT_FINAL}"
