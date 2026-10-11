# Sampling SFT：专家轨迹 → 模型专属数据 → 原生 SFT

按 [Finetuning with Sampling（arXiv:2610.02140v1）](https://arxiv.org/abs/2610.02140)
和[官方代码](https://github.com/aakaran/finetuning-with-sampling/tree/aa7080885f114fdc5d141f19c09f53008a1c061e)
实现离线数据链路，详见[阅读笔记](../../../notes/other_works/FinetuningWithSampling.md)。

**实现的是附录 C.2 / 公开代码的贪心近似，不是精确 Metropolis–Hastings。**
每块先在专家条件下续写，再从当前回答的随机切点重写 suffix；只接受无专家条件下
**平均 token logprob 更高**的候选。SFT loss、原生 Trainer、其他模型及 RL rollout 配置均不改变。

所有命令从仓库根目录运行，使用本项目 `.venv`。先按
[`examples/train/rwkv/README.md`](../rwkv/README.md)准备 RWKV 环境与 FlashRWKV2 cache。
API 阶段不加载本地模型；生成请求会产生服务商费用，建议先用 `--limit 8`。

## 1. 用 `.env` 的 DS_MODEL 生成专家轨迹

实际使用三个已有键（不在代码中配置凭证）：

```dotenv
DS_MODEL_URL=<OpenAI-compatible base URL, or .../chat/completions>
DS_MODEL_NAME=<teacher model name>
DS_MODEL_API_KEY=<API key>
```

UltraData-RL-2609 的原始格式为 `uuid/query/ground_truth/domain`。
目录读取所有子目录中的 JSONL/Parquet shard；以下示例只选 Math：

```bash
uv run --isolated --no-project --env-file .env .venv/bin/python \
  -m examples.train.sampling_sft.generate_expert_traces \
  --dataset "$HOME/data/UltraData-RL-2609/files" \
  --domain Math --limit 8 --concurrency 4 \
  --output outputs/sampling_sft/expert_math.jsonl \
  > /tmp/results_expert_math.log 2>&1
```

完成小规模验证后可去掉 `--limit`。继续已完成的小批次或中断任务，加 `--resume`；
允许扩大 `--limit` / 调整并发，但不允许在同一输出文件中混用不同模型或提示。
新增 `--domain Code` 可以同时生成代码题轨迹；不会执行模型返回的代码。

其他题集支持显式指定字段和 HF revision，例如：

```bash
uv run --isolated --no-project --env-file .env .venv/bin/python \
  -m examples.train.sampling_sft.generate_expert_traces \
  --dataset hendrycks/competition_math --dataset-split train \
  --question-key problem --answer-key solution --limit 8 \
  --output outputs/sampling_sft/expert_other.jsonl \
  > /tmp/results_expert_other.log 2>&1
```

提供 HF ID 时使用 `datasets` streaming，`--dataset-config` / `--revision` 可固定配置和版本。
缺少 ID 列时按题目文本 SHA256 生成稳定 UUID。`--num-shards N --shard-index K` 生成互不重叠的 shard。
`--limit` 在 domain 过滤后、分片与 resume 前应用；因此恢复同一个限量任务不会额外选新题。
输出目录应位于题集目录之外，避免将生成结果当作源题集再次扫描。

每个成功记录保存原题、gold、`reasoning_content`、`content`、两者以单个换行符 `"\n"` 拼接的 `solution`、teacher model
和 token usage。只接受有非空最终答案且 `finish_reason=stop` 的 API 响应；限长、不完整响应或
请求最终失败会停止该次任务，已写出的成功记录可恢复。API 请求不发送 `max_tokens` 或
`max_completion_tokens`，使用服务商默认输出上限。SDK 使用有界并发、显式超时和有限重试。

每个输出旁有 `.jsonl.meta.json`，保存恢复配置，不保存 API key 或原始 endpoint URL。
成功结果逐条 flush/fsync；恢复时只移除末尾未完整写入的 JSONL 记录，不忽略损坏的完整记录。
同一输出只能有一个 writer。分片任务应各用独立输出文件。

## 2. 用目标 learner 做 projection-inspired sampling

Nemotron-SFT-Math-v4 已包含专家轨迹，可以直接输入 Parquet 目录：

```bash
uv run --isolated --no-project .venv/bin/python \
  -m examples.train.sampling_sft.boost \
  --input "$HOME/Datasets/Nemotron-SFT-Math-v4" --subset cot \
  --model "$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600" \
  --max-model-len 1048576 --max-new-tokens 8192 --block-num 32 \
  --output outputs/sampling_sft/nemotron_train.jsonl \
  > /tmp/results_sampling_nemotron.log 2>&1
```

保留完整 `reasoning_content` 和 `content`，两者用 `"\n"` 拼接，不按 checkpoint
标注的训练上下文筛选或截断。保留原始 user message 的数学输出要求。同题的不同
轨迹按 `problem` 哈希进入同一 held-out partition。多卡采样使用每卡 TP1，分别指定
`--num-shards N --shard-index K` 与独立输出文件；训练/验证需保持相同分片和 seed。
vLLM 使用 8192-token chunked prefill。

模型必须是准备训练的具体 checkpoint，不是 DS teacher。RWKV 示例明确使用：

```bash
MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"

uv run --isolated --no-project .venv/bin/python \
  -m examples.train.sampling_sft.boost \
  --input outputs/sampling_sft/expert_math.jsonl \
  --model "$MODEL_DIR" --partition train \
  --output outputs/sampling_sft/train.jsonl \
  > /tmp/results_sampling_train.log 2>&1

uv run --isolated --no-project .venv/bin/python \
  -m examples.train.sampling_sft.boost \
  --input outputs/sampling_sft/expert_math.jsonl \
  --model "$MODEL_DIR" --partition validation \
  --output outputs/sampling_sft/validation.jsonl \
  > /tmp/results_sampling_validation.log 2>&1
```

默认按 `SHA256(seed:uuid)` 保留 5% questions 作为 validation；两次调用的 seed 和
`--validation-fraction` 必须相同。很小的 smoke set 可能没有 validation question；正式训练
前确认两个 partition 均包含完整样本。`--partition all` 可输出全部数据用于单次链路验收。
validation 仅用于 SFT held-out loss，不是新任务 reward / accuracy 评测。

默认值：

- `--max-new-tokens 1856 --block-num 32 --mcmc-steps 10`；实际每块58 tokens。
- `--mcmc-steps 0` 是逐块续写的无搜索 rewrite 对照。
- Proposal：temperature0.6、top-p1、top-k-1；RWKV 使用 repetition penalty1.0，其他模型默认1.1。
  当前 RWKV Rapid-Sampling 后端不支持 multiplicative repetition penalty；禁用该 penalty，保留 suffix 搜索与原始评分。
  Scoring：只有原题、temperature1、无截断采样限制，读取 raw prompt logprobs。
- `--max-model-len 32768` 包含专家提示、prefix 与 suffix；输入不截断，超预算报错。
  专家 reasoning 很长时需要明确增大预算。采样算力与序列长度近似平方增长。
- RWKV 使用 `wkv_mode=fp32io16`、FP16 inference weights、FP32 recurrent state、TP1、prefix caching；
  在加载前复用既有 FlashRWKV2 cache preflight。
- 默认 `bot/open_think`；通过 `--rwkv-prompt-template {bot,assistant,function_calling}` 和
  `--rwkv-generation-prompt {open_think,fake_think}` 明确选择模板。其他模型不传 RWKV 模板参数；
  非 chat 模型可以指定 `--model-type base`。HF model 可用 `--revision` 固定版本。
- RWKV `bot` proposal 使用原题的 chat template，再追加一次已有 response IDs：
  `User✿{question}✿\nBot✿<think{response_prefix}`。原题后直接结束 User 消息，
  专家解答与 `Partial response` 不进入这个上下文。完整专家推理仍保存在输出的 `solution` 中；
  此路径做 question-only greedy suffix search，不再做专家条件重写。
  生成 `✿`（token ID 10060）时结束 Bot 回答，并追加 EOS（token ID 0）；模型直接生成 EOS 也会停止。
  separator 与 EOS 保留在 `response_ids` 和训练 loss mask 中。RWKV 输出 metadata 版本为 2，
  旧版拼接产生的记录应使用独立输出路径保留。

恢复加 `--resume`，不重新生成已完成 UUID。切点 RNG 与 vLLM proposal seed 按 UUID 固定，
与输入顺序/已完成题数无关；这不保证跨 GPU/kernel/vLLM 版本 bitwise identical。
同一输出必须保持 checkpoint、tokenizer、源专家数据及 metadata 配置不变。

输出每条包含原题、专家 solution、`gen`、精确 `response_ids`、base 平均 logprob、接受计数与耗时。
完整回答的原生 SFT 字段为：

```json
{"input_ids": [7, 8, 3, 0], "loss_mask": [0, 0, 1, 1]}
```

`input_ids` 是 **只有原题的真实 prompt IDs + 生成 response IDs**，不含专家解答/重写指令。
mask 只监督 response，包含生成的 EOS。限长而未完成的记录保留用于审计/resume，但
`input_ids/loss_mask` 置空，让原生 loader 剔除；不会将不完整回答伪装成成功轨迹。

这一最小移植与作者一样靠提示词保留信息，不在提案过程中强制验证正确性；完整回答也不自动
等于正确答案。正式训练前应按任务的原生 verifier 审查专家与 boosted 数据的正确率和中间事实。
不记录历史 proposal logprobs 冒充最终 sampler density，也不导入 teacher-logit trace 分支。

## 3. 使用原生 SkyRL SFT

不需要新的 SFT trainer 或 dataset class，直接消费 `pretokenized_dataset_paths`：

```bash
MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"

RAY_ADDRESS=local uv run --isolated --no-project --env-file .env .venv/bin/python \
  -m skyrl.train.main_sft \
  strategy=fsdp model.path="$MODEL_DIR" \
  'pretokenized_dataset_paths=[outputs/sampling_sft/train.jsonl]' \
  'eval_pretokenized_dataset_paths=[outputs/sampling_sft/validation.jsonl]' \
  remove_microbatch_padding=false sequence_parallel_size=1 \
  rwkv_chunk_size=1024 \
  'model_config_kwargs={wkv_mode: fp32io16}' \
  num_epochs=2 batch_size=16 micro_train_batch_size_per_gpu=1 \
  placement.num_nodes=1 placement.num_gpus_per_node=1 \
  optimizer_config.lr=5e-5 optimizer_config.max_grad_norm=1.0 \
  optimizer_config.scheduler=cosine \
  max_length=null ckpt_path=outputs/sampling_sft/checkpoints \
  ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
  export_path=outputs/sampling_sft/hf_exports logger=wandb \
  project_name=rwkv_sampling_sft run_name=rwkv7-g1k-1.5b-20260930-ctx25600 \
  > /tmp/results_sampling_sft.log 2>&1
```

`max_length=null` 保留所有完整 token 序列，不再二次截断；按真实模型/显存预算准备数据。
`rwkv_chunk_size=1024` 使用现有 Transformers `RwkvCache` 的 WKV 与 token-shift
状态分块执行，checkpoint 重算每个 chunk 的 forward 和 LM-head logprobs，保留完整
跨块梯度，不 detach 历史。FSDP ranks 使用相同 chunk 次数，样本间独立初始化状态。
`rwkv_chunk_size=0` 保留既有整段路径。
此处 eval 记录 held-out SFT loss；reward/accuracy 曲线应另用既有 RWKV evaluation 流程，
不能将 SFT loss 当作 reward 验收。W&B 使用用户已有配置。GPU 数与 batch size 需匹配实际资源。

## 文件与测试

- `generate_expert_traces.py`：DS API 专家轨迹采集。
- `data.py`：流式源题集、确定性划分、可恢复 JSONL。
- `boost.py`：固定 learner inference、RWKV preflight 与 exact-token 数据输出。
- `skyrl/train/utils/projection_sampling.py`：proposal/base scoring/greedy suffix-search 核心。

CPU 契约测试不调用真实 API、不加载 GPU、不启动 Ray：

```bash
RAY_ADDRESS=local uv run --isolated --no-project .venv/bin/python -m pytest \
  --confcutdir=tests/backends/skyrl_train/rwkv \
  tests/backends/skyrl_train/rwkv/test_sampling_sft.py -q \
  > /tmp/results_sampling_sft_tests.log 2>&1
```
