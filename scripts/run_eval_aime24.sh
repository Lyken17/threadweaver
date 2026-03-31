#!/usr/bin/env bash
# AIME'24 evaluation wrapper for SFT checkpoints
# Usage: ./scripts/run_eval_aime24.sh <model_path> [label]
# Submits via eai-run to SLURM GPU node.
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> [label]}"
LABEL="${2:-eval}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SFT_DIR="${REPO_ROOT}/threadweaver_sft"
AIME_DATA="${SFT_DIR}/data/aime2024"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Resolve to absolute path
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$(cd "$REPO_ROOT" && realpath "$MODEL_PATH")"
fi

UID_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${LABEL}_${UID_STAMP}.txt"

echo "=== AIME'24 Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Data:  ${AIME_DATA}"
echo "Log:   ${LOG_FILE}"

# Create wrapper script on shared filesystem (GPU node can't see /tmp)
WRAPPER_DIR="${REPO_ROOT}/.tmp_scripts"
mkdir -p "$WRAPPER_DIR"
WRAPPER="${WRAPPER_DIR}/eval_aime_${LABEL}_${UID_STAMP}.sh"
cat > "$WRAPPER" <<EOFWRAPPER
#!/usr/bin/env bash
set -eo pipefail
source /home/ligengz/anaconda3/etc/profile.d/conda.sh
set +u
conda activate tw
set -u

cd "${SFT_DIR}"

python src/simple_eval.py \
    --model_name "${MODEL_PATH}" \
    --launch_server \
    --template-type model \
    --branching-generate \
    --max-context-length 40960 \
    --data-type "${AIME_DATA}" \
    --n_samples 8 \
    --bfloat16 \
    --timeout 600 \
    --verbose 2 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Evaluation complete. Log saved to: ${LOG_FILE} ==="
EOFWRAPPER
chmod +x "$WRAPPER"

echo "Submitting to SLURM via eai-run..."
eai-run -i -J "ralph/${LABEL}" --pty bash "$WRAPPER"
