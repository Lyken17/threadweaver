#!/usr/bin/env bash
# Chain 8x SFT training across multiple 4h SLURM jobs.
# Each job loads model weights from the previous checkpoint and trains
# for max_steps steps. Optimizer/scheduler reset each job but model
# knowledge carries forward.
#
# Usage: ./scripts/chain_8x_training.sh [starting_checkpoint_path]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SFT_DIR="${REPO_ROOT}/threadweaver_sft"
TRAIN_DATA="${SFT_DIR}/data/polaris_1st_sft/polaris_data_53K_1_1k_1000samples_v111/sample_964_8x"

# Total steps for 1 epoch: 7712 / (8 GPUs * 2 grad_accum) = 482
TOTAL_STEPS=482
STEPS_PER_JOB=140  # conservative to leave time for checkpoint save

# Starting checkpoint (model weights only) - resolve to absolute
PREV_CKPT="$(realpath "${1:-${SFT_DIR}/ckpts/Q3-8B-131072-sft-8x-20260401_120033/checkpoint-100}")"
COMPLETED_STEPS="${2:-100}"  # steps already done (pass as 2nd arg)

echo "=== Chained 8x SFT Training ==="
echo "Starting from: ${PREV_CKPT}"
echo "Steps completed: ${COMPLETED_STEPS}/${TOTAL_STEPS}"
echo "Steps per job: ${STEPS_PER_JOB}"

JOB_NUM=1
while [ "$COMPLETED_STEPS" -lt "$TOTAL_STEPS" ]; do
    REMAINING=$((TOTAL_STEPS - COMPLETED_STEPS))
    STEPS=$((REMAINING < STEPS_PER_JOB ? REMAINING : STEPS_PER_JOB))

    UID_STAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${SFT_DIR}/ckpts/Q3-8B-131072-sft-8x-chain-job${JOB_NUM}-${UID_STAMP}"

    echo ""
    echo "=== Job ${JOB_NUM}: steps ${COMPLETED_STEPS}+${STEPS} (target: ${TOTAL_STEPS}) ==="
    echo "  Base model: ${PREV_CKPT}"
    echo "  Max steps: ${STEPS}"
    echo "  Output: ${OUTPUT_DIR}"

    WRAPPER="${REPO_ROOT}/.tmp_scripts/chain_8x_job${JOB_NUM}_${UID_STAMP}.sh"
    mkdir -p "$(dirname "$WRAPPER")"
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
export WANDB_RUN_NAME="sft-8x-chain-job${JOB_NUM}"

# Use previous checkpoint as base model, train for max_steps
bash train.sh \
    --model_name="${PREV_CKPT}" \
    --max_steps=${STEPS} \
    --save_strategy="steps" \
    --save_steps=${STEPS} \
    --learning_rate=5e-6 \
    --warmup_ratio=0.02

echo "Job ${JOB_NUM} complete. Output: ${OUTPUT_DIR}"
EOFWRAPPER
    chmod +x "$WRAPPER"

    echo "  Submitting job ${JOB_NUM}..."
    eai-run -i -J "ralph/sft-8x-j${JOB_NUM}" --pty bash "$WRAPPER"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: Job ${JOB_NUM} failed with exit code ${EXIT_CODE}"
        exit 1
    fi

    # Find the checkpoint from this job
    LATEST_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        # No checkpoint subdir, use output_dir itself (save_only_model puts files in root)
        LATEST_CKPT="${OUTPUT_DIR}"
    fi

    # Verify checkpoint has model files
    if [ ! -f "${LATEST_CKPT}/config.json" ]; then
        echo "ERROR: No valid checkpoint found in ${OUTPUT_DIR}"
        exit 1
    fi

    echo "  Job ${JOB_NUM} done. Checkpoint: ${LATEST_CKPT}"
    PREV_CKPT="${LATEST_CKPT}"
    COMPLETED_STEPS=$((COMPLETED_STEPS + STEPS))
    JOB_NUM=$((JOB_NUM + 1))
done

echo ""
echo "=== All jobs complete ==="
echo "Final checkpoint: ${PREV_CKPT}"
echo "Total steps: ${COMPLETED_STEPS}/${TOTAL_STEPS}"

# Copy final checkpoint to a clean location
FINAL_DIR="${SFT_DIR}/ckpts/Q3-8B-131072-sft-8x-complete"
if [ -d "$FINAL_DIR" ]; then
    rm -rf "$FINAL_DIR"
fi
cp -r "${PREV_CKPT}" "${FINAL_DIR}"
echo "Final model copied to: ${FINAL_DIR}"
