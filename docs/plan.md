# Reproduce the ThreadWeaver Ablation Studies

## Goal Description

Reproduce the two Ablation Studies tables from the ThreadWeaver README on AIME'24:

**Table 1 — Impact of Self-Training & RL:**

| Model Configuration | Format Correctness | AIME24 Accuracy | Token Latency |
| :--- | :--- | :--- | :--- |
| Qwen3-8B + 1st SFT (959 samples) | 56.4% | 74.5% | 17.6k |
| Qwen3-8B + Self-Training (17k samples) | 77.0% | 74.0% | 17.3k |
| **Qwen3-8B + Self-Training + RL** | **72.4%** | **79.9%** | **16.9k** |

**Table 2 — Reward Normalization (P-GRPO):**

| Setting | Accuracy | Mean Longest Thread |
| :--- | :--- | :--- |
| With Std. Normalization | 74.79% | 18.7k |
| **Mean-Centered Only (Ours)** | **79.90%** | **16.9k** |

**Current Phase: SFT only.** Launch the 1st SFT phase first using pre-processed data from HuggingFace, evaluate on AIME'24, and confirm accuracy is within 2pp of the ablation table reference (74.5%). Two dataset variants are available — `sample_964` (1x) and `sample_964_8x` (8x) — and both must be trained and compared. RL phases will follow once SFT results are validated.

Pre-processed parallel trajectory data is available on HuggingFace at `longlian/threadweaver_public_code_reproduced` under subdirectory `polaris_data_53K_1_1k_1000samples_v111/`:
- `sample_964` — 1x dataset (~964 samples)
- `sample_964_8x` — 8x augmented dataset

For RL training data (future phase), follow the README to use Polaris-53K. The target cluster uses SLURM for job scheduling. GPU-requiring commands must be wrapped with `eai-run` since the login node does not have GPU access. The conda environment is named `tw` with Python 3.12.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Metrics use tolerance-based targets: reference numbers from the README Ablation Studies serve as guidelines, with pass criteria allowing reasonable variance (within ~2 percentage points for AIME'24 accuracy).

- AC-1: Environment setup completes and smoke test passes
  - Positive Tests (expected to PASS):
    - SLURM cluster detected: `which srun && which squeue && which sinfo` all succeed
    - Conda env `tw` created: `conda activate tw && python --version` shows Python 3.12
    - CPU-only imports succeed on login node: `python -c "import transformers, trl, deepspeed, accelerate, datasets"` succeeds
    - GPU smoke test via `eai-run`: `eai-run -i -J ralph/smoke-test --pty bash -c "conda activate tw && python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())'"` prints 8
    - `python -c "import ray; import vllm; import sglang"` succeeds
    - `python -c "import flash_attn"` succeeds (must run on GPU node via `eai-run`)
    - `python -c "import verl"` succeeds (from `threadweaver_rl/` directory)
    - OpenAI API key is accessible: `OPENAI_KEY_PATH` or `OPENAI_API_KEY` set
  - Negative Tests (expected to FAIL):
    - Running GPU smoke test directly on login node (without `eai-run`) fails — no GPU available
    - Importing `flash_attn` on CPU-only login node raises ImportError
  - AC-1.1: If single-env smoke test fails on package compatibility, fall back to separate SFT and RL environments
    - Positive: Each environment passes its respective subset of import checks
    - Negative: Cross-environment imports (e.g., veRL in SFT env) may fail — this is acceptable in fallback mode

- AC-2: Training data prepared for both SFT and RL phases
  - Positive Tests (expected to PASS):
    - Pre-processed SFT data downloaded from HuggingFace (`longlian/threadweaver_public_code_reproduced`, subdirectory `polaris_data_53K_1_1k_1000samples_step5_v1_v1_v1`)
    - Downloaded dataset loads successfully and contains ~1000 samples with `qwen_text` field
    - Polaris-53K parquet downloaded and MD5 verified (`58e1e523f9946956f68055a374d43a46`) for RL data preparation
  - Negative Tests (expected to FAIL):
    - HuggingFace download fails with invalid dataset path
    - Dataset missing `qwen_text` field fails SFT training
  - AC-2.1: First SFT datasets (1x and 8x) ready from HuggingFace download
    - Positive: Both `sample_964` (1x, ~964 samples) and `sample_964_8x` (8x) downloaded from `polaris_data_53K_1_1k_1000samples_v111/`, each exposing `qwen_text` field
    - Negative: Dataset with incompatible schema for `train.sh`
  - AC-2.2: Self-Training dataset (17k samples) prepared
    - Positive: Expanded dataset with ~17k samples after self-training data augmentation using 1st SFT checkpoint
    - Negative: Dataset with fewer than 15k samples indicates insufficient self-training scaling
  - AC-2.3: RL training data prepared from Polaris-53K
    - Positive: Parquet dataset with `prompt` and `data_source` columns suitable for veRL
    - Negative: Missing required columns causes veRL data loading to fail

- AC-3: Qwen3-8B model downloaded and context length extended
  - Positive Tests (expected to PASS):
    - `Qwen/Qwen3-8B-131072/config.json` contains `"max_position_embeddings": 131072`
    - Model loads successfully with `AutoModelForCausalLM.from_pretrained()`
    - Tokenizer loads and encodes/decodes correctly
  - Negative Tests (expected to FAIL):
    - Unmodified config still shows `"max_position_embeddings": 40960`
    - Model path that doesn't exist raises OSError

- AC-4: 1st SFT training completes on both 1x and 8x datasets and checkpoints evaluate on AIME'24
  - Positive Tests (expected to PASS):
    - SFT training on `sample_964` (1x) dataset runs to completion via `eai-run`
    - SFT training on `sample_964_8x` (8x) dataset runs to completion via `eai-run`
    - Both checkpoints are loadable with `AutoModelForCausalLM.from_pretrained()`
    - AIME'24 evaluation for each: Reference is Format Correctness ~56.4%, Accuracy ~74.5%, Token Latency ~17.6k (tolerance: within **2pp** for accuracy)
    - 8x vs 1x accuracy comparison documented
  - Negative Tests (expected to FAIL):
    - Untrained base model produces near-zero format correctness
    - Training on empty dataset fails immediately

- AC-5: Self-Training SFT (17k samples) completes and checkpoint evaluates on AIME'24
  - Positive Tests (expected to PASS):
    - SFT training on 17k-sample self-training dataset runs to completion
    - Format Correctness improves to ~77.0% (higher than 1st SFT's 56.4%)
    - AIME'24 Accuracy ~74.0%, Token Latency ~17.3k (tolerance: within ~2pp for accuracy)
  - Negative Tests (expected to FAIL):
    - Format Correctness lower than 1st SFT indicates self-training regression

- AC-6: RL training (P-GRPO, mean-centered) completes and checkpoint evaluates on AIME'24
  - Positive Tests (expected to PASS):
    - RL training with `algorithm.norm_adv_by_std_in_grpo=False` (mean-centered) runs on self-training SFT checkpoint via `eai-run`
    - `reward/mean` and `metrics/correct_rate` improve over training
    - AIME'24 evaluation produces: Accuracy ~79.9%, Token Latency ~16.9k (tolerance: within ~2pp for accuracy)
    - Token Latency decreases compared to SFT-only checkpoints
  - Negative Tests (expected to FAIL):
    - RL training without SFT checkpoint as initialization fails
    - Training with misconfigured reward produces zero rewards

- AC-7: RL ablation with std normalization evaluates on AIME'24
  - Positive Tests (expected to PASS):
    - RL training with `algorithm.norm_adv_by_std_in_grpo=True` (standard normalization) runs on same SFT checkpoint
    - AIME'24 evaluation produces: Accuracy ~74.79%, Mean Longest Thread ~18.7k
    - Accuracy is lower than mean-centered variant (~79.9%), confirming P-GRPO advantage
  - Negative Tests (expected to FAIL):
    - Std-normalization variant outperforming mean-centered on both accuracy and latency would contradict paper findings

- AC-8: AIME'24 evaluation infrastructure works for all checkpoints
  - Positive Tests (expected to PASS):
    - AIME'24 dataset obtained and formatted as parquet with `prompt`, `data_source`, `reward_model.ground_truth` fields
    - `simple_eval.py --branching-generate --data-type aime` runs on each checkpoint and produces format correctness, accuracy, and token latency metrics
    - All five rows of the two ablation tables are populated with reproducible numbers
  - Negative Tests (expected to FAIL):
    - Evaluation without AIME parquet fails with file-not-found
    - Evaluation on non-parallel model with `--branching-generate` produces malformed output

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The reproduction covers the full Ablation Studies pipeline: environment setup, Polaris-53K data generation (5-stage), two SFT training runs (959-sample 1st SFT and 17k-sample self-training), two RL training runs (mean-centered P-GRPO and std-normalization ablation), and AIME'24 evaluation for all five checkpoints producing all rows of both ablation tables. Output artifacts for each milestone are clearly documented. A synthetic multiplication sanity check is included as a quick validation before the full pipeline.

### Lower Bound (Minimum Acceptable Scope)

The reproduction produces all five data points across the two ablation tables with tolerance-based accuracy. All eight acceptance criteria are met. Intermediate sanity checks may be skipped if the pipeline is progressing correctly.

### Allowed Choices

- Can use: conda environment `tw` (Python 3.12) for isolation; `eai-run` for GPU job submission on SLURM cluster; single node with 8x80GB GPUs; DeepSpeed ZeRO-3 (as configured in repo); vLLM for RL rollout; SGLang for SFT evaluation; W&B or TensorBoard for monitoring; OpenAI API for data generation pipeline
- Cannot use: multi-node setup for the primary path (documented as single-node); alternative base models (must use Qwen3-8B); alternative RL algorithms (must use P-GRPO via veRL); direct GPU access on the login node (must use `eai-run`)

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Environment**: Create the `tw` conda environment with Python 3.12 (`conda create -n tw python=3.12 -y`). Install PyTorch 2.6.0+cu124, then SFT deps, RL deps, data generation deps, and flash-attn. Configure OpenAI API key.

2. **Data Preparation**:
   - Download pre-processed 1st SFT data from HuggingFace: `longlian/threadweaver_public_code_reproduced` (subdirectory `polaris_data_53K_1_1k_1000samples_v111/`):
     - `sample_964` — 1x dataset (~964 samples)
     - `sample_964_8x` — 8x augmented dataset
   - Download Polaris-53K parquet and verify MD5 (needed for RL data preparation in future phase)
   - The 5-stage data generation pipeline can be skipped for the 1st SFT phase since pre-processed data is available
   - Self-training expansion to 17k samples still requires generating additional data using the 1st SFT checkpoint (future phase)

3. **1st SFT (1x and 8x)**: Train on both `sample_964` (1x) and `sample_964_8x` (8x) datasets using `train.sh` with `TRAIN_DATA` pointing to each parquet. Evaluate both on AIME'24 and compare 8x vs 1x accuracy. Target: within 2pp of reference 74.5% AIME'24 accuracy.

4. **Self-Training SFT (17k samples)**: Use the 1st SFT checkpoint to generate more parallel trajectories, filter for format/correctness, scale to ~17k samples. Retrain SFT on the expanded dataset. Evaluate on AIME'24.

5. **RL (Mean-Centered P-GRPO)**: From `threadweaver_rl/`, run `verl.trainer.main_ppo` with `algorithm.norm_adv_by_std_in_grpo=False` using the self-training SFT checkpoint. Evaluate on AIME'24.

6. **RL Ablation (Std Normalization)**: Same RL setup but with `algorithm.norm_adv_by_std_in_grpo=True`. Evaluate on AIME'24 to confirm the P-GRPO advantage.

7. **AIME'24 Evaluation**: Run `simple_eval.py --branching-generate --data-type aime` on each of the checkpoints (1st SFT, self-training SFT, RL mean-centered, RL std-norm) collecting format correctness, accuracy, and token latency.

### Relevant References

- `data_generation/src/generate_trajectories.py` — Sequential trajectory generation from base model
- `data_generation/src/generated_data_to_training_data_polaris.py` — Filter and format training data
- `data_generation/run.sh` — 5-stage parallel trajectory refinement pipeline
- `data_generation/src/collect_trajectories.py` — Trajectory collection (Stage 0)
- `data_generation/scripts/step1.sh` through `step5.sh` — Individual pipeline stages
- `threadweaver_sft/train.sh` — SFT training launcher with DeepSpeed config
- `threadweaver_sft/src/simple_eval.py` — Evaluation with SGLang backend and branching generation
- `threadweaver_sft/src/sft_threadweaver.py` — SFT training implementation with trie-based collation
- `threadweaver_sft/configs/deepspeed_zero3_offload.json` — DeepSpeed ZeRO-3 config
- `threadweaver_rl/verl/trainer/main_ppo.py` — RL training entry point
- `threadweaver_rl/deepscaler/rewards/math_rewardv2.py` — Reward function with acceleration ratio
- `threadweaver_rl/README.md` — RL quickstart guide with single-node training command

## Dependencies and Sequence

### Milestones

1. **Environment Setup**: Install all dependencies in a single environment on the SLURM cluster
   - Verify SLURM cluster: `which srun && which squeue && which sinfo`
   - Create conda environment: `conda create -n tw python=3.12 -y && conda activate tw`
   - Install PyTorch 2.6.0 with CUDA 12.4 support
   - Install SFT deps, RL deps, data generation deps
   - Install flash-attn with `--no-build-isolation`
   - Configure OpenAI API key (`OPENAI_KEY_PATH` or `OPENAI_API_KEY`)
   - Run smoke tests (CPU on login node, GPU via `eai-run`)
   - **Output artifact**: Working `tw` conda environment
   - **Depends on**: Nothing

2. **Model Preparation**: Download and configure base model
   - Download Qwen3-8B via `huggingface-cli download`
   - Extend context length to 131072 via `sed` on `config.json`
   - **Output artifact**: `Qwen/Qwen3-8B-131072/`
   - **Depends on**: Milestone 1

3. **Data Preparation**: Obtain training data for SFT (and RL later)
   - Download pre-processed 1st SFT data from HuggingFace (`longlian/threadweaver_public_code_reproduced`, subdir `polaris_data_53K_1_1k_1000samples_v111/`):
     - `sample_964` — 1x dataset (~964 samples)
     - `sample_964_8x` — 8x augmented dataset
   - Verify both datasets load and expose the `qwen_text` field
   - Download Polaris-53K parquet and verify MD5 (for RL data preparation in future phase)
   - **Output artifacts**: 1st SFT training parquets (1x and 8x), Polaris-53K raw parquet
   - **Depends on**: Milestone 1

4. **1st SFT Training + AIME'24 Eval**: Train on both 1x and 8x datasets, evaluate, compare
   - Run `TRAIN_DATA=<sample_964-parquet> ./train.sh` via `eai-run` (1x)
   - Run `TRAIN_DATA=<sample_964_8x-parquet> ./train.sh` via `eai-run` (8x)
   - Evaluate both checkpoints on AIME'24: reference is ~56.4% format correctness, ~74.5% accuracy, ~17.6k token latency (tolerance: within **2pp** for accuracy)
   - Compare 8x vs 1x accuracy to determine which variant to carry forward
   - **Output artifacts**: 1st SFT checkpoints (1x and 8x), AIME'24 eval results for both, comparison summary (Row 1 of Table 1)
   - **Depends on**: Milestones 1, 2, 3

5. **Self-Training Data Expansion**: Scale to 17k samples
   - Use 1st SFT checkpoint to generate additional parallel trajectories
   - Filter for format correctness and answer correctness
   - Combine with original data to reach ~17k samples
   - **Output artifact**: 17k-sample self-training parquet
   - **Depends on**: Milestone 4

6. **Self-Training SFT + AIME'24 Eval**: Train on 17k samples, evaluate
   - Run `TRAIN_DATA=<17k-sample-parquet> ./train.sh` via `eai-run`
   - Evaluate on AIME'24: expect ~77.0% format correctness, ~74.0% accuracy, ~17.3k token latency
   - **Output artifacts**: Self-training SFT checkpoint, AIME'24 eval results (Row 2 of Table 1)
   - **Depends on**: Milestones 1, 2, 5

7. **RL Training (Mean-Centered P-GRPO) + AIME'24 Eval**: RL on self-training checkpoint
   - Run `verl.trainer.main_ppo` with `algorithm.norm_adv_by_std_in_grpo=False` via `eai-run`
   - Evaluate on AIME'24: expect ~72.4% format correctness, ~79.9% accuracy, ~16.9k token latency
   - **Output artifacts**: RL checkpoint (mean-centered), AIME'24 eval results (Row 3 of Table 1, Row 2 of Table 2)
   - **Depends on**: Milestones 1, 6

8. **RL Ablation (Std Normalization) + AIME'24 Eval**: RL with standard normalization
   - Run `verl.trainer.main_ppo` with `algorithm.norm_adv_by_std_in_grpo=True` via `eai-run`
   - Evaluate on AIME'24: expect ~74.79% accuracy, ~18.7k mean longest thread
   - **Output artifacts**: RL checkpoint (std-norm), AIME'24 eval results (Row 1 of Table 2)
   - **Depends on**: Milestones 1, 6

9. **AIME'24 Evaluation Infrastructure**: Obtain and prepare AIME'24 dataset
   - Source AIME 2024 problems and format as parquet (`prompt`, `data_source: "aime"`, `reward_model.ground_truth`)
   - Verify `simple_eval.py` works with AIME parquet
   - This milestone is needed before any AIME eval in Milestones 4, 6, 7, 8
   - **Output artifact**: `aime2024.parquet`
   - **Depends on**: Milestone 1

Milestones 7 and 8 (the two RL variants) can run in parallel since they share the same SFT starting checkpoint. Milestone 9 (AIME data) must be ready before any evaluation step.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write environment setup script: conda env creation, PyTorch install, SFT/RL/datagen deps, flash-attn, OpenAI API config, smoke test | AC-1 | coding | - |
| task2 | Analyze dependency conflicts between SFT, RL, and data generation packages; document fallback two-env strategy | AC-1, AC-1.1 | analyze | task1 |
| task3 | Write model download and context extension commands with verification | AC-3 | coding | task1 |
| task4 | Write AIME'24 dataset preparation: obtain problems, format as parquet with required schema, verify with simple_eval.py | AC-8 | coding | task1 |
| task5 | Write data preparation: download both 1x (`sample_964`) and 8x (`sample_964_8x`) SFT data from HuggingFace, download Polaris-53K for RL, verify formats | AC-2, AC-2.1, AC-2.3 | coding | task1, task3 |
| task6 | Analyze downloaded HF datasets (1x and 8x): verify sample counts, schema (`qwen_text` field), compatibility with `train.sh` | AC-2, AC-2.1 | analyze | task5 |
| task7 | Write 1st SFT training commands for both 1x and 8x datasets, AIME'24 evaluation for both, and 8x vs 1x accuracy comparison | AC-4 | coding | task4, task5 |
| task8 | Write self-training data expansion procedure: use 1st SFT checkpoint to generate more data, filter, scale to 17k | AC-2.2, AC-5 | coding | task7 |
| task9 | Analyze self-training methodology: verify filtering criteria, expected yield rate, format correctness improvement | AC-2.2, AC-5 | analyze | task8 |
| task10 | Write self-training SFT (17k samples) and AIME'24 evaluation commands | AC-5 | coding | task8 |
| task11 | Write RL training command (mean-centered P-GRPO, `norm_adv_by_std_in_grpo=False`) and AIME'24 evaluation | AC-6 | coding | task10 |
| task12 | Write RL ablation command (std normalization, `norm_adv_by_std_in_grpo=True`) and AIME'24 evaluation | AC-7 | coding | task10 |
| task13 | Analyze RL reward config and P-GRPO normalization variants: verify expected metric differences between the two variants | AC-6, AC-7 | analyze | task11, task12 |
| task14 | Compile final ablation tables from all AIME'24 evaluations; compare against reference numbers | AC-4 to AC-8 | coding | task11, task12 |
| task15 | Final review: verify all commands, paths, and ablation table numbers are consistent end-to-end | AC-1 to AC-8 | analyze | task14 |

## Claude-Codex Deliberation

### Agreements

- Single-environment setup is the correct default; two environments are a fallback only if optional veRL extras conflict
- Tolerance-based metrics (~5pp) are appropriate for reproduction across different hardware/seeds
- flash-attn installation and SLURM cluster config are legitimate operational risks
- Hardware assumption of single node with 8x80GB GPUs matches the documented setup
- veRL should be run from source directory (`cd threadweaver_rl && python -m verl...`), not installed as editable package
- The Ablation Studies tables require the full Polaris-53K pipeline, not synthetic multiplication
- AIME'24 dataset must be sourced externally and formatted as parquet with specific schema
- The P-GRPO ablation requires two RL training runs differing only in `norm_adv_by_std_in_grpo`

### Resolved Disagreements

- **Separate vs single environment**: Codex v1 initially suggested separate environments. Resolution: single environment is default, with two-env fallback documented.

- **Acceleration ratio definition**: Codex noted the code-level `acceleration_ratio` (used in rewards) differs from the README's token latency / speedup definition. Resolution: ACs use the README metric definitions (token latency for Table 1, mean longest thread for Table 2).

- **Metric strictness**: Codex recommended tolerance-based targets. Resolution: adopted ~5pp tolerance with reference numbers from ablation tables as targets.

- **Scope change**: Initial plan targeted synthetic multiplication only. User updated scope to Ablation Studies tables, requiring the full Polaris-53K pipeline and AIME'24 evaluation. Resolution: plan fully restructured to target all five rows across both tables.

### Convergence Status

- Final Status: `converged`
- Rounds: 2 (initial convergence), then scope updated by user to target Ablation Studies
- Plan restructured to reflect the updated goal

## Pending User Decisions

- DEC-1: Reproduction scope
  - Claude Position: Initially multiplication-only; updated to full Ablation Studies tables
  - Codex Position: Originally agreed on multiplication-only
  - Tradeoff Summary: Ablation Studies require Polaris-53K pipeline + OpenAI API + AIME'24 external dataset
  - Decision Status: `Ablation Studies tables` (user updated)

- DEC-2: Hardware configuration
  - Claude Position: Single node, 8x80GB GPUs as documented
  - Codex Position: Agrees; matches repo documentation
  - Tradeoff Summary: Single node is simpler; multi-node adds SLURM/Ray complexity
  - Decision Status: `Single node, 8x80G GPUs` (user confirmed)

- DEC-3: Acceptance metric philosophy
  - Claude Position: Tolerance-based (within ~2pp of reference for AIME'24 accuracy)
  - Codex Position: Recommends tolerance-based with reference numbers retained as targets
  - Tradeoff Summary: 2pp tolerance balances reproducibility with meaningful accuracy validation
  - Decision Status: `Tolerance-based, 2pp` (user updated)

- DEC-4: Environment starting point
  - Claude Position: Fresh environment setup from scratch
  - Codex Position: N/A — open question
  - Tradeoff Summary: Fresh setup is more reproducible; existing stack saves setup effort
  - Decision Status: `Fresh environment` (user confirmed)

- DEC-5: AIME'24 dataset source
  - Claude Position: Must be obtained externally; format as parquet with `prompt`, `data_source: "aime"`, `reward_model.ground_truth`
  - Codex Position: N/A — open question
  - Tradeoff Summary: Dataset is not in the repo; user needs to provide or source it
  - Decision Status: `PENDING`

- DEC-6: OpenAI API access for data generation
  - Claude Position: Not needed for 1st SFT (pre-processed data on HuggingFace). May be needed for self-training data expansion if using the 5-stage pipeline to process new trajectories.
  - Codex Position: Flagged as prerequisite in v1 analysis (before HF data was known)
  - Tradeoff Summary: 1st SFT data is available on HF; self-training expansion may need OpenAI API for the refinement stages, or could use the 1st SFT model directly for generation + format/correctness filtering
  - Decision Status: `PENDING`

## Implementation Notes

### SLURM Cluster Execution

This cluster uses SLURM for scheduling. The login node has no GPU access; all GPU-requiring commands must be wrapped with `eai-run`.

**Pattern for GPU commands:**
```bash
# Python scripts requiring GPU
eai-run -i -J ralph/{job-name} --pty bash -c "conda activate tw && python a.py"

# Shell scripts requiring GPU
eai-run -i -J ralph/{job-name} --pty bash -c "conda activate tw && bash run.sh"

# GPU diagnostics
eai-run -i -J ralph/{job-name} --pty nvidia-smi
```

**Commands that require `eai-run`:** trajectory generation (`generate_trajectories.py`), SFT training (`train.sh`), SFT evaluation (`simple_eval.py`), RL training (`python3 -m verl.trainer.main_ppo`), and any script that imports `torch.cuda`.

**Commands safe on login node:** `conda create/install`, `pip install`, `huggingface-cli download`, `sed`, non-GPU Python imports, queue monitoring (`squeue`).

**Job queue monitoring:** After submitting via `eai-run`, check status with `squeue`. Poll every 1 minute if expected launch is within 5 minutes, every 5 minutes otherwise. The last column timestamp shows expected launch time.

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### SFT Training Data Format
- `train.sh` accepts a parquet dataset via `TRAIN_DATA` env var; the dataset must expose a `qwen_text` field.
- The pre-processed HuggingFace datasets (`longlian/threadweaver_public_code_reproduced`, subdir `polaris_data_53K_1_1k_1000samples_v111/sample_964` for 1x and `sample_964_8x` for 8x) provide this data for the 1st SFT phase, avoiding the need to run the full 5-stage data generation pipeline. Both variants should be trained and compared.
- For RL, veRL expects Parquet with `prompt` and `data_source` columns; follow the README to prepare data from Polaris-53K.

### Known Version Tensions
- veRL `setup.py` optional extras pin different versions of TRL (<=0.9.6) and SGLang (0.4.10.post2) than the README install commands (TRL 0.19.0, SGLang 0.4.6.post1). The documented install commands work; avoid installing veRL optional extras (`pip install -e .[vllm]` or `pip install -e .[sglang]`).

### Ablation-Specific Notes
- The two RL runs (mean-centered vs std-normalization) differ ONLY in `algorithm.norm_adv_by_std_in_grpo` (False vs True). All other config is identical.
- Token latency in Table 1 and Mean Longest Thread in Table 2 both refer to `num_tokens_in_the_longest_thread` from the parallel stats computation.
- Format Correctness measures whether the model produces valid `<Parallel>`, `<Thread>`, `<Outlines>` structure.
- Self-training involves using the 1st SFT checkpoint to generate parallel trajectories on additional data, then filtering for both format correctness and answer correctness before combining with original training data.

--- Original Design Draft Start ---

# Reproduce the baseline

## Setup the Env

## Launch the SFT exps

## Eval the SFT checkpoint on AIME'24

## Launch the RL exps

## EVal the RL checkpoint on AIME'24

--- Original Design Draft End ---
