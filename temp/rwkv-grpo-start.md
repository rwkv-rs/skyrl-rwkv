# RWKV7 GRPO 快速开始

这份说明用于从 SkyRL 项目根目录启动 RWKV7 G1j 1.5B 的 GSM8K GRPO。它是功能验收 recipe，不是吞吐量 benchmark。

## 1. 进入项目并准备环境

```bash
cd ~/Projects/MachineLearning/skyrl-rwkv
uv sync --extra rwkv
```

只使用这个项目自己的 `.venv`。不要复用其他项目的 Python 环境。

## 2. 下载模型

recipe 固定使用下面的模型 revision：

- 仓库：`rwkv-rs/rwkv7-g1-st`
- revision：`e1a670a5523742b5cfe8cb6759c1eb8f1d88b637`
- 子目录：`rwkv7-g1j-1.5b-20260831-ctx16384`
- 架构：RWKV7，`model_type=rwkv`
- WKV 模式：`fp32io16`

```bash
MODEL_REPO=rwkv-rs/rwkv7-g1-st
MODEL_REVISION=e1a670a5523742b5cfe8cb6759c1eb8f1d88b637
MODEL_SUBFOLDER=rwkv7-g1j-1.5b-20260831-ctx16384
MODEL_ROOT="$HOME/models/rwkv7-g1-st-$MODEL_REVISION"

hf download "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --include "$MODEL_SUBFOLDER/*" \
  --local-dir "$MODEL_ROOT"

export MODEL_DIR="$MODEL_ROOT/$MODEL_SUBFOLDER"
```

检查配置和 Safetensors：

```bash
MODEL_DIR="$MODEL_DIR" uv run --no-sync --extra rwkv python -c \
  'import json, os; from pathlib import Path; p=Path(os.environ["MODEL_DIR"]); c=json.loads((p/"config.json").read_text()); assert c["model_type"]=="rwkv"; assert c["architecture_version"]=="rwkv7"; assert c["wkv_mode"]=="fp32io16"; print(c["architectures"], c["wkv_mode"])'

sha256sum "$MODEL_DIR"/*.safetensors
```

把 revision、配置和 Safetensors 哈希保存在实验记录中。Trainer 和 vLLM 必须使用同一个 `MODEL_DIR`。

## 3. 准备 GSM8K

```bash
uv run --no-sync --extra rwkv examples/train/gsm8k/gsm8k_dataset.py \
  --output_dir "$HOME/data/gsm8k"
```

默认需要：

```text
$HOME/data/gsm8k/train.parquet
$HOME/data/gsm8k/validation.parquet
```

## 4. 配置 W&B（可选但推荐）

在项目根目录创建只对当前用户可读的 `.env`：

```bash
install -m 0600 /dev/null .env
${EDITOR:-vi} .env
```

写入：

```text
WANDB_API_KEY=你的_W&B_API_key
```

不要把 `.env` 提交到 Git，也不要把 token 放进命令行、日志或截图。只检查权限和 Git 忽略状态：

```bash
stat -c '%a %n' .env
git check-ignore -v .env
```

## 5. 启动 8 卡 GRPO

先确认机器上的 GPU 没有被其他任务占用：

```bash
nvidia-smi
```

从项目根目录启动：

```bash
MODEL_DIR="$MODEL_DIR" \
NUM_GPUS=8 \
MICRO_BATCH_SIZE=2 \
bash examples/train/rwkv/run_rwkv_gsm8k.sh
```

脚本默认配置如下：

- SkyRL FSDP Trainer + vLLM-RWKV inference engines
- GRPO，50 个 optimizer steps
- 8 个 TP1 colocated inference engines
- `train_batch_size=64`，`policy_mini_batch_size=64`
- 每卡 micro-batch 为 `2`
- Trainer BF16，vLLM FP16，`wkv_mode=fp32io16`
- gradient checkpointing 开启
- rollout：`temperature=1.0`、`top_p=0.95`、`top_k=-1`，关闭 penalty
- eval：`temperature=0.96`、`top_p=0.76`、`top_k=32`，presence/frequency penalty 为 `1.0/0.1`，`penalty_decay=0.988`
- 默认 prompt template：`bot + open_think`

默认 run 名称是：

```text
rwkv7-g1j-1.5b-gsm8k-grpo-50step
```

如果显存不足，只把每卡 micro-batch 调为 `1`：

```bash
MICRO_BATCH_SIZE=1 bash examples/train/rwkv/run_rwkv_gsm8k.sh
```

不要因此修改全局 `train_batch_size` 或 `policy_mini_batch_size`。

## 6. 查看输出

默认输出目录为：

```text
$HOME/skyrl-rwkv-runs/rwkv7-g1j-1.5b-gsm8k-grpo-50step/
```

主要产物：

```text
checkpoints/global_step_50/       # 可恢复 checkpoint
exports/global_step_50/policy/    # Hugging Face 导出
exports/dumped_evals/              # GSM8K eval JSONL 和聚合结果
logs/                              # trainer、inference、router 日志
```

运行时可另开终端查看日志：

```bash
tail -f "$HOME/skyrl-rwkv-runs/rwkv7-g1j-1.5b-gsm8k-grpo-50step/run.log"
```

完成判断至少应包括：50/50 optimizer steps、每步 trainer→inference 权重同步、step 50 checkpoint/export/eval，以及 W&B 中的训练、reward、eval、资源和 logprob 曲线。

## 7. 停止任务

优先在启动任务的终端按 `Ctrl-C`。停止后确认 GPU 已释放：

```bash
nvidia-smi
```

不要使用 `killall python`、`ray stop --force` 等会影响其他项目或会话的宽范围命令。
