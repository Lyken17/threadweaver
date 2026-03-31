#!/usr/bin/env bash
# Self-training SFT wrapper
# Usage: ./scripts/run_self_training_sft.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SFT_DIR="${REPO_ROOT}/threadweaver_sft"
TRAIN_DATA="${SFT_DIR}/data/self_training_17k"
LABEL="sft-self-training"
UID_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SFT_DIR}/ckpts/Q3-8B-131072-${LABEL}-${UID_STAMP}"

echo "=== Self-Training SFT ==="
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

WRAPPER_DIR="${REPO_ROOT}/.tmp_scripts"
mkdir -p "$WRAPPER_DIR"
WRAPPER="${WRAPPER_DIR}/sft_train_${LABEL}_${UID_STAMP}.sh"
cat > "$WRAPPER" <<EOFWRAPPER
#!/usr/bin/env bash
set -eo pipefail
source /home/ligengz/anaconda3/etc/profile.d/conda.sh
set +u
conda activate tw
set -u

cd "${SFT_DIR}"
export TRAIN_DATA="${TRAIN_DATA}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export WANDB_PROJECT="threadweaver-sft"
export WANDB_RUN_NAME="${LABEL}-${UID_STAMP}"

bash train.sh --save_strategy="steps" --save_steps=100
echo "Training complete. Checkpoint saved to: ${OUTPUT_DIR}"
EOFWRAPPER
chmod +x "$WRAPPER"

TIME="${EAI_TIME:-4:00:00}"
echo "Submitting to SLURM via eai-run (time=${TIME})..."
eai-run -i -J "ralph/${LABEL}" --time "${TIME}" --pty bash "$WRAPPER"
