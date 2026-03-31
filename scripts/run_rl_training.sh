#!/usr/bin/env bash
# RL training wrapper (P-GRPO)
# Usage: ./scripts/run_rl_training.sh <model_path> <variant>
#   variant: "mean-centered" or "std-norm"
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <mean-centered|std-norm>}"
VARIANT="${2:?Usage: $0 <model_path> <mean-centered|std-norm>}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RL_DIR="${REPO_ROOT}/threadweaver_rl"
SFT_DIR="${REPO_ROOT}/threadweaver_sft"
RL_DATA="${SFT_DIR}/data/polaris_rl"

# Resolve model path
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$(cd "$REPO_ROOT" && realpath "$MODEL_PATH")"
fi

case "$VARIANT" in
  mean-centered)
    NORM_STD="False"
    LABEL="rl-mean-centered"
    ;;
  std-norm)
    NORM_STD="True"
    LABEL="rl-std-norm"
    ;;
  *)
    echo "Error: variant must be 'mean-centered' or 'std-norm'" >&2
    exit 1
    ;;
esac

UID_STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT="${LABEL}-${UID_STAMP}"

echo "=== RL Training (${VARIANT}) ==="
echo "Model: ${MODEL_PATH}"
echo "norm_adv_by_std_in_grpo: ${NORM_STD}"
echo "RL Data: ${RL_DATA}"

WRAPPER_DIR="${REPO_ROOT}/.tmp_scripts"
mkdir -p "$WRAPPER_DIR"
WRAPPER="${WRAPPER_DIR}/rl_train_${LABEL}_${UID_STAMP}.sh"
cat > "$WRAPPER" <<EOFWRAPPER
#!/usr/bin/env bash
set -eo pipefail
source /home/ligengz/anaconda3/etc/profile.d/conda.sh
set +u
conda activate tw
set -u

cd "${RL_DIR}"
export VLLM_USE_V1=1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${RL_DATA}/train.parquet" \
    data.val_files="${RL_DATA}/val.parquet" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=9216 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.ppo_mini_batch_size=null \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${NORM_STD} \
    algorithm.broadcast_from_last=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard','wandb']" \
    trainer.project_name='threadweaver-rl' \
    trainer.experiment_name="${EXPERIMENT}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="8" \
    trainer.nnodes="1" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    reward_model.config.acceleration_ratio_reward=1.0 \
    reward_model.config.acceleration_ratio_reward_factor=0.5 \
    reward_model.config.acceleration_ratio_clip_max=0.2 \
    reward_model.config.version=v2 \
    reward_model.config.require_think_end=False \
    reward_model.config.strip_comma_from_answer=True \
    reward_model.reward_manager_type=reward_manager_with_server \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.enable_parallel_branching=True \
    actor_rollout_ref.rollout.agent_return_expanded_sequences=True \
    actor_rollout_ref.rollout.agent.no_conclusion=true \
    actor_rollout_ref.rollout.mode=async

echo "RL training complete."
EOFWRAPPER
chmod +x "$WRAPPER"

TIME="${EAI_TIME:-4:00:00}"
echo "Submitting to SLURM via eai-run (time=${TIME})..."
eai-run -i -J "ralph/${LABEL}" --time "${TIME}" --pty bash "$WRAPPER"
