#!/usr/bin/env bash
# SFT training wrapper for ThreadWeaver
# Usage: ./scripts/run_sft.sh <variant> [extra args for train.sh]
#   variant: "1x" or "8x"
# Submits via eai-run to SLURM GPU node.
set -euo pipefail

VARIANT="${1:?Usage: $0 <1x|8x> [extra train.sh args]}"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SFT_DIR="${REPO_ROOT}/threadweaver_sft"

case "$VARIANT" in
  1x)
    TRAIN_DATA="${SFT_DIR}/data/polaris_1st_sft/polaris_data_53K_1_1k_1000samples_v111/sample_964"
    LABEL="sft-1x"
    ;;
  8x)
    TRAIN_DATA="${SFT_DIR}/data/polaris_1st_sft/polaris_data_53K_1_1k_1000samples_v111/sample_964_8x"
    LABEL="sft-8x"
    ;;
  *)
    echo "Error: variant must be '1x' or '8x', got '$VARIANT'" >&2
    exit 1
    ;;
esac

UID_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SFT_DIR}/ckpts/Q3-8B-131072-${LABEL}-${UID_STAMP}"

echo "=== SFT Training (${VARIANT}) ==="
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Extra args: $*"

# Create a wrapper script on shared filesystem (GPU node can't see /tmp)
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

RESUME_ARG=""
if [ -n "\${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume_from_checkpoint=\${RESUME_FROM}"
fi
bash train.sh --save_strategy="steps" --save_steps=100 --save_only_model=False \${RESUME_ARG} \$@
echo "Training complete. Checkpoint saved to: ${OUTPUT_DIR}"
EOFWRAPPER
chmod +x "$WRAPPER"

TIME="${EAI_TIME:-4:00:00}"
echo "Submitting to SLURM via eai-run (time=${TIME})..."
eai-run -i -J "ralph/${LABEL}" --time "${TIME}" --pty bash "$WRAPPER" "$@"
