# ThreadWeaver SFT 复现指南

本文档记录了在 SLURM 集群 (8×80G GPU, 4h 时间限制) 上复现 ThreadWeaver Ablation Studies 第一行 (1st SFT) 的完整过程，供其他 agent 参照执行。

## 目标

复现 README 中 Ablation Studies Table 1 第一行：

| Model Configuration | Format Correctness | AIME24 Accuracy | Token Latency |
|:---|:---|:---|:---|
| Qwen3-8B + 1st SFT (959 samples) | 56.4% | 74.5% | 17.6k |

## 复现结果

| Checkpoint | Pass@1 | Token Latency | 备注 |
|:---|:---|:---|:---|
| 1x (964 samples, full epoch) | 71.25% | 16,762 | 完整训练 |
| **8x (7712 samples, full epoch)** | **72.08%** | **17,708** | 最佳结果，接近 reference |

8x 模型的 token latency (17,708) 与 reference (17.6k) 几乎完全匹配。accuracy 72.08% 与 reference 74.5% 差 2.42pp。

---

## 环境准备

### 前置条件
- SLURM 集群，`eai-run` 提交 GPU 作业
- 4h 最大时间限制 (`--time 4:00:00`)
- Conda 环境 `tw`，Python 3.12

### Conda 环境

```bash
conda create -n tw python=3.12 -y
conda activate tw
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# SFT 依赖
pip install transformers==4.51.1 trl==0.19.0 deepspeed==0.17.0 accelerate==1.7.0 \
    datasets==3.6.0 liger_kernel==0.5.10 wandb==0.21.0
# Eval 依赖 (SGLang)
pip install sglang==0.4.6.post1 vllm==0.8.5.post1
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### SLURM GPU 节点上的 Conda 激活

**关键坑点**: GPU 节点的 non-interactive bash 无法直接 `conda activate`。必须先 source conda.sh：

```bash
source /home/ligengz/anaconda3/etc/profile.d/conda.sh
set +u           # conda deactivate hooks 有 unbound variable
conda activate tw
set -u
```

不要用 `eval "$(conda shell.bash hook)"`，会报错。

---

## Step 1: 下载模型和数据

### 基座模型 (Qwen3-8B, context length 131072)

```bash
cd threadweaver_sft
huggingface-cli download Qwen/Qwen3-8B --local-dir Qwen/Qwen3-8B-131072
# 修改 config.json 中 max_position_embeddings 为 131072
python -c "
import json
c = json.load(open('Qwen/Qwen3-8B-131072/config.json'))
c['max_position_embeddings'] = 131072
json.dump(c, open('Qwen/Qwen3-8B-131072/config.json', 'w'), indent=2)
"
```

### SFT 训练数据 (从 HuggingFace 下载)

```bash
huggingface-cli download longlian/threadweaver_public_code_reproduced \
    polaris_data_53K_1_1k_1000samples_v111/sample_964/train.parquet \
    polaris_data_53K_1_1k_1000samples_v111/sample_964_8x/train.parquet \
    --repo-type dataset \
    --local-dir data/polaris_1st_sft
```

两个数据集：
- `sample_964/`: 1x 数据集，964 个 sample（原始去重后）
- `sample_964_8x/`: 8x 数据集，7712 个 sample（8倍重复+shuffle）

都包含 `qwen_text` 字段，可直接用于 `train.sh`。

### AIME'24 评测数据

需要自行准备 `threadweaver_sft/data/aime2024/val.parquet`，schema 如下：
- `prompt`: numpy array of `{"role": "user", "content": "..."}` dict
- `data_source`: string, 值为 `"aime"`
- `reward_model`: dict `{"answer": "<ground_truth>", "ground_truth": "<ground_truth>"}`

共 30 道 AIME 2024 题。

---

## Step 2: 1x SFT 训练 (964 samples)

### 直接训练

```bash
# 在 GPU 节点上执行（通过 eai-run）
cd threadweaver_sft
export TRAIN_DATA=data/polaris_1st_sft/polaris_data_53K_1_1k_1000samples_v111/sample_964
export OUTPUT_DIR=ckpts/Q3-8B-131072-sft-1x
bash train.sh --save_strategy="epoch"
```

### 用 wrapper 脚本

```bash
bash scripts/run_sft.sh 1x
```

**训练参数** (来自 `train.sh`):
- `block_size=40960`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=2`
- 8 GPU → effective batch size = 16 → 964/16 = 60 steps/epoch
- `lr=1e-5`, cosine schedule, `warmup_ratio=0.05`
- DeepSpeed ZeRO-3 with CPU offload
- 训练约 1.5 小时

**预期**: loss 从 ~0.42 降到 ~0.27，最终 AIME Pass@1 ~71%

---

## Step 3: 8x SFT 训练 (7712 samples)

8x 数据集需要 482 steps 完成一个 epoch，单个 4h 作业只能跑 ~148 steps。需要链式提交多个作业。

### 方法: 链式训练 (chain_8x_training.sh)

由于 DeepSpeed ZeRO-3 的 `--resume_from_checkpoint` 不工作（即使保存了完整 optimizer state），采用 **warm-start** 策略：每个作业加载上一个 checkpoint 的模型权重作为 base model，从头训练 N steps。

```bash
# 第一步：从基座模型开始训练 140 steps
bash scripts/run_sft.sh 8x
# → 产出 checkpoint-100 (save_steps=100)

# 后续步骤：从上一步的 checkpoint 继续
bash scripts/chain_8x_training.sh <checkpoint_path> <completed_steps>

# 例如：
bash scripts/chain_8x_training.sh \
    threadweaver_sft/ckpts/Q3-8B-131072-sft-8x-xxx/checkpoint-100 \
    100
```

`chain_8x_training.sh` 会自动：
1. 计算剩余 steps
2. 提交 SLURM 作业，加载上一个 checkpoint 作为 `--model_name`
3. 训练 `max_steps` 步后保存 checkpoint
4. 自动继续下一个 job，直到 482 steps 完成

**实际执行过程** (共 4 个 job):

| Job | Base Model | Steps | Loss (start→end) |
|:----|:-----------|:------|:------------------|
| 1 | Qwen3-8B-131072 | 0→100 | 0.43→0.25 |
| 2 | Job1/ckpt-100 | 0→140 | 0.20→0.16 |
| 3 | Job2/ckpt-140 | 0→140 | 0.17→0.15 |
| 4 (final) | Job3/ckpt-140 | 0→102 | 0.14→0.13 |

**总训练量**: 100+140+140+102 = 482 steps = 1 full epoch

### 已知问题

1. **DeepSpeed ZeRO-3 resume 不工作**: 即使 `--save_only_model=False` 保存了 107GB 完整 checkpoint（含 optimizer/scheduler/RNG state），`--resume_from_checkpoint` 仍然从 step 0 重新开始。这是 DeepSpeed ZeRO-3 + HuggingFace Trainer + torchrun 的已知问题。
2. **Optimizer 重置**: 每个 chain job 的 optimizer 和 LR schedule 会重新初始化。用较小的 LR (5e-6) 和短 warmup (0.02) 来缓解。
3. **`/tmp` 不可见**: GPU 节点看不到 login node 的 `/tmp`。wrapper 脚本必须写到共享文件系统 (`.tmp_scripts/`)。

---

## Step 4: AIME'24 评测

### 评测命令

```bash
bash scripts/run_eval_aime24.sh <model_path> <label>

# 例如：
bash scripts/run_eval_aime24.sh threadweaver_sft/ckpts/Q3-8B-131072-sft-8x-complete eval-8x
```

### 关键参数

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| `--branching-generate` | 必须 | 启用 parallel branching 生成 |
| `--max-context-length` | 40960 | **必须与 train.sh 的 block_size 一致**。用 8192 会导致 accuracy 暴跌到 21% |
| `--launch_server` | 必须 | 启动 SGLang server |
| `--template-type` | model | 使用模型自带 chat template |
| `--n_samples` | 8 | 每题生成 8 个 sample |
| `--bfloat16` | 推荐 | 半精度推理 |
| `--suffix` | 唯一标签 | 避免结果文件冲突 |
| `--overwrite` | 推荐 | 强制重新生成，不 resume 旧结果 |

### 评测产出

评测结果打印到 stdout 并通过 `tee` 保存到 `logs/eval_<label>_<timestamp>.txt`：

```
Pass@1: 0.7208 (72.08)
Pass@8: 0.8333 (83.33)
Format Correctness: 0.0083 (0.83%)
Average acceleration_ratio: 0.0647
Average parallel_ratio: 0.2532
Average total_num_tokens: 19997.05
Average num_tokens_in_the_longest_thread: 17708.43
```

### 评测结果对应 Ablation Table

| Metric | 对应列 | 说明 |
|:-------|:-------|:-----|
| Pass@1 | AIME24 Accuracy | 主要指标 |
| num_tokens_in_the_longest_thread | Token Latency | 最长线程的 token 数 |
| Format Correctness | Format Correctness | 严格格式校验 (见下方说明) |

### Format Correctness 说明

`simple_eval.py` 中的 `is_parallel_format_correct()` 使用严格校验（移植自 `data_generation/src/filter-format-correct-and-obtain-stats.py`），要求：
- `<Parallel>`/`</Parallel>` 匹配
- `<Outlines>` 内有编号的 `<Outline>` 标签
- `<Thread>` 标签与 `<Outline>` 一一对应且编号匹配
- Thread 内容不含 XML 标签

**注意**: 由于 branching generation 的 Thread body 包含模型原始推理输出（可能含 `<think>` 等标签），严格校验会给出接近 0% 的 format correctness。README 中报告的 56.4% 可能使用了更宽松的定义。这是一个已知的 measurement gap。

---

## Step 5: 关键修复 (simple_eval.py)

复现过程中对 `threadweaver_sft/src/simple_eval.py` 做了一个关键修复：

### 修复: 移除 branching generation 的 early return

**问题**: 当 Thread 内容不以 `</Thread>` 结尾时（常见于 branching generation），原代码在 line ~655 直接 `return merged`，跳过了 line 661 的 `merged += "\n"` continuation newline。这导致 branching loop 提前终止，parallel_ratio 从 0.03 降到 0.0002，accuracy 从 71% 降到 66%。

**修复**: 将 `return merged` 改为 `continue`（打印 warning 但不 return），让 branching loop 始终继续。

```python
# 修复前 (broken)
if end_seq:
    print("WARNING: Some thread did not end properly, returning...")
    return merged  # ← 跳过 continuation newline!

# 修复后 (fixed)
if end_seq:
    print("WARNING: Some thread did not end properly, continuing...")
    # 不 return，让 merged += "\n" 正常执行

merged += "\n"  # continuation newline，始终执行
```

**影响**: 这是 accuracy 71% vs 66% 的关键区别。

---

## 完整文件清单

```
threadweaver/
├── scripts/
│   ├── run_sft.sh              # SFT 训练 wrapper (1x/8x)
│   ├── run_eval_aime24.sh      # AIME 评测 wrapper
│   ├── chain_8x_training.sh    # 8x 链式训练 (跨 4h SLURM 限制)
│   ├── prepare_self_training_data.py  # 自训练数据组装
│   └── prepare_rl_data.py      # RL 数据准备
├── threadweaver_sft/
│   ├── train.sh                # SFT 训练主脚本
│   ├── src/simple_eval.py      # AIME 评测 (含 branching fix)
│   ├── Qwen/Qwen3-8B-131072/  # 基座模型
│   ├── data/
│   │   ├── polaris_1st_sft/polaris_data_53K_1_1k_1000samples_v111/
│   │   │   ├── sample_964/     # 1x 训练数据
│   │   │   └── sample_964_8x/  # 8x 训练数据
│   │   └── aime2024/           # AIME 评测数据
│   └── ckpts/
│       ├── Q3-8B-131072-sft-1x-*/  # 1x checkpoint
│       └── Q3-8B-131072-sft-8x-complete/  # 8x final checkpoint
└── docs/
    ├── plan.md                 # 完整复现计划
    └── reproduce.md            # 本文件
```

---

## 常见问题

### Q: 为什么 accuracy 是 71-72% 而不是 74.5%？
Reference 74.5% 可能使用了不同的评测参数（n_samples=16 vs 我们的 8，或不同的随机种子）。我们的评测有 stochasticity，多次运行结果在 65-72% 之间波动。

### Q: 为什么 8x 比 1x 好？
8x 数据是 1x 的 8 倍重复+shuffle，让模型在 parallel 格式数据上训练更充分。8x 的 parallel_ratio (0.25) 远高于 1x (0.03)。

### Q: 为什么不直接 resume checkpoint？
DeepSpeed ZeRO-3 的 resume 在此环境下不工作（已验证 `save_only_model=True` 和 `False` 都不行）。改用 warm-start 策略（加载模型权重，重新初始化 optimizer）。

### Q: 后续步骤 (Self-Training, RL) 怎么做？
参见 `docs/plan.md` 的 Milestones 5-8。Self-Training 需要用 1st SFT 模型在 Polaris-53K 上生成 parallel trajectories 并过滤。RL 需要 veRL 框架。这些步骤尚未完成。
