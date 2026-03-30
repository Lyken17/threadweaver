# ThreadWeaver Agents

This document describes the key agents (actors, controllers, and workers) in the ThreadWeaver framework and how they interact across the data generation, training, and inference pipelines.

---

## 0. SLURM Cluster Setup

### Detect whether you are running on a SLURM cluster

```bash
which srun
which squeue
which sinfo
```

If all of `srun`, `squeue`, and `sinfo` are installed, you are running on a SLURM cluster. Otherwise you can ignore the SLURM-related instructions below.

### Determine login-node vs. worker-node

Check whether the environment has `SLURM_JOB_ID` set or `nvidia-smi` is available. If neither is present, you are on a **login node**; otherwise you are on a **worker node**.

### Allocate a GPU node for GPU-related tasks

The login node can handle most tasks, but if a program (`*.py`, `*.sh`, or commands like `nvidia-smi`) requires GPU access, you must allocate a GPU node. Wrap the command with `eai-run`:

```bash
# Python scripts
python a.py            -> eai-run -i -J ralph/{a-suitable-job-name} --pty python a.py

# uv-managed scripts
uv run a.py            -> eai-run -i -J ralph/{a-suitable-job-name} --pty uv run a.py

# GPU diagnostics
nvidia-smi             -> eai-run -i -J ralph/{a-suitable-job-name} --pty nvidia-smi

# Shell scripts requiring GPUs
bash run.sh            -> eai-run -i -J ralph/{a-suitable-job-name} --pty bash run.sh
```

### Check queue status and running time

After submitting a job to SLURM, check its status with `squeue`:

```
Sat Mar 28 11:40:04 2026
     ACCOUNT          JOBID    PARTITION     USER    STATE     TIME TIME_LIMI NODES NAME {NODELIST(REASON) START_TIME}
 nvr_elm_llm        8947377 interactive,  ligengz  PENDING     0:00   4:00:00     1 nvr_elm_llm:dev/eai-test {(QOSMaxJobsPerUserLimit) 2026-03-28T15:39:39}
```

The last timestamp is the expected launch time. Check the job status:
- Every **1 minute** if the expected launch time is within 5 minutes
- Every **5 minutes** otherwise

When running jobs with a time limit (e.g., 5 minutes), ensure the job gets enough runtime after allocation. The time count starts once SLURM allocates the resources, from submission.

### Conda environment

This repo is recommended to run with conda. Create and activate the `tw` environment before running any experiments:

```bash
conda create -n tw python=3.12 -y
conda activate tw
```

All commands (training, data generation, evaluation, etc.) should be run inside the `tw` conda environment. When combining with `eai-run`, activate the environment inside the job:

```bash
eai-run -i -J ralph/{job-name} --pty bash -c "conda activate tw && python a.py"
```

---

## Architecture Overview

```
                         +-----------------------+
                         |   Data Generation     |
                         |   Pipeline (5 stages) |
                         +-----------+-----------+
                                     |
                              parallel trajectories
                                     |
                    +----------------+----------------+
                    |                                 |
          +---------v----------+          +-----------v---------+
          |   SFT Trainer      |          |   RL Trainer        |
          |   (Trie Collator)  |          |   (P-GRPO)          |
          +--------------------+          +-----+------+--------+
                                                |      |
                                          +-----v-+  +-v--------+
                                          | Actor  |  | Reward   |
                                          | Rollout|  | Manager  |
                                          +---+----+  +----------+
                                              |
                                   +----------v-----------+
                                   | ParallelBranching    |
                                   | Controller           |
                                   | (Inference Agent)    |
                                   +----------------------+
```

---

## 1. Inference Agents

### ParallelBranchingController

**Location:** `threadweaver_rl/verl/experimental/agent_loop/single_turn_agent_loop.py`

The central inference-time agent. It orchestrates the fork-join execution of parallel reasoning by driving a 3-state state machine over a standard vLLM/SGLang completion API.

**States:**
| State | Behavior |
|---|---|
| `SEQUENTIAL` | Autoregressive decoding until `</Outlines>` or EOS |
| `PARALLEL` | Spawns concurrent completion requests for each `<Thread>`, one per outline entry |
| `DONE` | Generation complete |

**Lifecycle:**
1. Receives prompt tokens, begins in `SEQUENTIAL` state
2. Decodes until `</Outlines>` is emitted, then parses outline numbers via regex (`<Outline> N:`)
3. Transitions to `PARALLEL` -- issues `asyncio.gather(*tasks)` to run all thread completions concurrently
4. Each branch gets its own prompt: `context + "\n<Thread>\n{outline_num}:"`, stops on `</Thread>` or EOS
5. Joins branch outputs into a single token sequence, appends `</Parallel>`
6. Returns to `SEQUENTIAL` for the next segment (or terminates)

**Key config:** `AgentLoopConfig` (`threadweaver_rl/verl/workers/config/rollout.py`)
- `enable_parallel_branching: bool` -- toggles fork-join behavior
- `num_workers: int` -- concurrency for the agent loop
- `no_conclusion: bool` -- skip `<Conclusion>` block generation

---

## 2. Training Agents

### SFT Trainer (Supervised Fine-Tuning)

**Location:** `threadweaver_sft/src/sft_threadweaver.py`

Trains the base model to produce the parallel trajectory format. Built on TRL's `SFTTrainer` with a custom data collator.

**Key behaviors:**
- Initializes special token embeddings (`<Parallel>`, `<Thread>`, `<Outlines>`, etc.)
- Uses the **PrefixTreeDataCollator** (see below) to construct ancestor-only attention masks
- Supports Qwen, Llama, and DeepSeek chat templates
- Distributed via FSDP or DeepSpeed ZeRO-2/3

### PrefixTreeDataCollator

**Location:** `threadweaver_sft/src/prefix_tree_utils_v1.py`

The core training-inference alignment agent. Flattens a parallel reasoning tree into a single training sequence while enforcing information isolation between threads.

**How it works:**
1. **Extraction** -- identifies all `(context, completion)` pairs the inference state machine would produce
2. **Trie construction** -- inserts token sequences into a prefix tree; shared prefixes become shared ancestors, threads become sibling branches
3. **Ancestor-only masking** -- generates an attention mask where token `i` can attend to token `j` iff `j` is an ancestor of `i` in the trie (computed via entry/exit DFS timestamps)
4. **Position IDs** -- assigns position IDs that respect the trie depth, not the flat sequence order

This ensures threads cannot leak information to each other during training, perfectly matching the independent parallel generation at inference time.

### RL Trainer (P-GRPO)

**Location:** `threadweaver_rl/verl/trainer/main_ppo.py` (entry point)

Reinforcement learning agent built on the VERL framework. Optimizes the SFT-initialized model for both correctness and parallelization efficiency.

**Training loop:**
1. **Rollout** -- generate parallel reasoning trajectories using the `ParallelBranchingController`
2. **Reward** -- score each trajectory (see Reward Manager below)
3. **Advantage** -- compute mean-centered advantages (`A_i = r_i - mean(r)`, no std normalization)
4. **Policy update** -- GRPO gradient step with clipped ratios (`clip_ratio_low=0.2`, `clip_ratio_high=0.28`)

**Why mean-centered?** Standard variance normalization causes the acceleration reward term to dominate when all rollouts are correct (since correctness becomes constant). Mean-centering preserves the relative weight of the acceleration signal.

---

## 3. Reward Agents

### Reward Manager

**Location:** `threadweaver_rl/deepscaler/rewards/math_rewardv2.py`

Computes per-trajectory rewards with two components:

**Correctness reward (`r_correct`):**
- Extracts the answer from `\boxed{}` in the model output
- Grades via SymPy symbolic comparison and MathD verification
- `+1.0` correct, `0.0` incorrect

**Acceleration reward (`r_accel`):**
- Calls `get_parallel_stats()` -- a one-pass parser over token IDs that computes:
  - `sequential_cost`: total tokens if everything ran sequentially
  - `parallel_cost`: tokens on the critical path (longest thread per block + sequential segments)
  - `acceleration_ratio = sequential_cost / parallel_cost`
- `r_accel = factor * min(acceleration_ratio - 1.0, clip_max)`
- Defaults: `factor=0.5`, `clip_max=0.2`

**Combined:** `reward = r_correct + r_accel`

The reward manager runs as a server (`reward_manager_with_server`) for distributed training, communicating with rollout workers.

---

## 4. Data Generation Agents

### Stage Pipeline

**Location:** `data_generation/src/`

A 5-stage pipeline where each stage acts as an independent processing agent, transforming sequential reasoning into parallel trajectories.

| Stage | Agent Script | Role |
|---|---|---|
| 0 | `collect_trajectories.py` | Samples and exports trajectories from source dataset |
| 1 | `gpt.py` + `step1-prompt_v1.txt` | Extracts hierarchical step structure (S1, S1.1, S2, ...) with line ranges |
| 2 | `extract_v1.py` | Identifies parallelizable segments, inserts `<Parallel>` / `<Thread>` markers |
| 3 | `rewrite-context_v1.py` | Rewrites thread content to be self-contained (removes cross-thread dependencies) |
| 4 | `generate-outline_v1.py` | Generates `<Outlines>` with numbered `<Outline>` entries for each parallel block |
| 5 | `filter-format-correct-and-obtain-stats.py` | Validates structure, filters malformed trajectories, reports quality stats |

**Stage 1** uses GPT-4/5 for analysis. **Stages 2-4** perform surgical rewriting -- they do not regenerate content, preserving original reasoning quality. **Stage 5** is a deterministic quality gate.

### Trajectory Generator

**Location:** `data_generation/src/generate_trajectories.py`

Generates initial sequential reasoning chains by running inference on the base model (via SGLang). Produces the raw material that the 5-stage pipeline transforms into parallel trajectories.

---

## 5. Rollout Workers

### BaseRollout / vLLM Rollout

**Location:** `threadweaver_rl/verl/workers/rollout/base.py`, `threadweaver_rl/verl/workers/rollout/sglang_rollout/`

Manages the inference engine lifecycle during RL training:
- Loads model weights into vLLM/SGLang serving engine
- Handles weight updates from the actor after each policy gradient step
- Coordinates with the `ParallelBranchingController` for fork-join generation
- Supports tensor parallelism and prefix caching

### FSDP Workers

**Location:** `threadweaver_rl/verl/workers/fsdp_workers.py`

Distributed training workers that handle:
- Actor model forward/backward passes with FSDP sharding
- Reference model inference (for KL penalty, if enabled)
- Gradient synchronization across nodes
- Checkpoint saving/loading

---

## 6. Evaluation Agent

### SimpleEval

**Location:** `threadweaver_sft/src/simple_eval.py`

A lightweight evaluation agent that:
1. Launches an SGLang server with the trained model
2. Runs parallel generation with branching on a test dataset
3. Computes correctness rewards and acceleration ratios
4. Reports Pass@1 accuracy and speedup metrics

---

## Agent Interaction Flow

### Training (RL)

```
PPO Trainer
  |
  +-- Actor Worker (FSDP)
  |     +-- generates policy gradients
  |     +-- updates model weights
  |
  +-- Rollout Worker (vLLM)
  |     +-- ParallelBranchingController
  |     |     +-- SEQUENTIAL decode
  |     |     +-- PARALLEL spawn (asyncio.gather)
  |     |     +-- JOIN and continue
  |     +-- returns trajectories with expanded sequences
  |
  +-- Reward Manager (server)
  |     +-- correctness grading (SymPy)
  |     +-- acceleration ratio computation
  |     +-- combined reward
  |
  +-- Reference Worker (FSDP, optional)
        +-- computes reference log-probs for KL
```

### Inference (Deployment)

```
User Prompt
  |
  v
ParallelBranchingController
  |
  +-- [SEQ] decode until </Outlines>
  |
  +-- [PARSE] extract outline numbers
  |
  +-- [PAR] spawn N threads (standard completion API)
  |     +-- Thread 1: "1: ..."  -->  vLLM/SGLang
  |     +-- Thread 2: "2: ..."  -->  vLLM/SGLang
  |     +-- Thread N: "N: ..."  -->  vLLM/SGLang
  |
  +-- [JOIN] concatenate threads + </Parallel>
  |
  +-- [SEQ] continue until next block or EOS
  |
  v
Final Response
```
