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

Nemotron-SFT-Math-v4 已包含专家轨迹，可以直接输入 Parquet 目录。
以下执行参数来自 PRO6000 的批量采样对照：

```bash
uv run --isolated --no-project .venv/bin/python \
  -m examples.train.sampling_sft.boost \
  --input "$HOME/Datasets/Nemotron-SFT-Math-v4" --subset cot \
  --model "$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600" \
  --max-model-len 1048576 --max-new-tokens 8192 --block-num 8 --mcmc-steps 1 \
  --batch-size 512 --gpu-memory-utilization 0.94 --skip-repetition \
  --max-num-batched-tokens 32768 --no-enforce-eager \
  --output outputs/sampling_sft/nemotron_train.jsonl \
  > /tmp/results_sampling_nemotron.log 2>&1
```

保留完整 `reasoning_content` 和 `content`，两者用 `"\n"` 拼接，不按 checkpoint
标注的训练上下文筛选或截断。保留原始 user message 的数学输出要求。同题的不同
轨迹按 `problem` 哈希进入同一 held-out partition。多卡采样使用每卡 TP1，分别指定
`--num-shards N --shard-index K` 与独立输出文件；训练/验证需保持相同分片和 seed。
此示例使用 32768-token chunked prefill。

`rwkv-sha-pro6000x8` 的本地 snapshot 为
`84d42ad0cb960f07f951b9baa9ed2b46a5a18c66`，包含 545,431 条轨迹、95,170 道不同原题。
其中 CoT 有 285,516 条轨迹、89,406 道原题；该训练配方消费完整 CoT 分区。
以 seed 0、validation fraction 0.05 按原题划分后，训练源为 271,385 条轨迹、
84,990 道原题，验证源为 14,131 条轨迹、4,416 道原题。验收或调参命令中的
`--limit` 只控制实验规模，生产命令省略该参数。

`--batch-size N` 将 N 条独立轨迹的 proposal 和 question-only scoring 合批交给
vLLM，每条轨迹保留自己的 UUID seed、prefix 和贪心接受状态。默认 1；多卡仍然
每卡 TP1、独立输出 shard。`--max-num-batched-tokens` 控制 prefill token budget，
`--no-enforce-eager` 启用 vLLM 的图执行路径；这两个执行参数不修改解码分布。
采样 `--resume` 允许调整 `--batch-size`、`--max-num-batched-tokens`、
`--enforce-eager` 和 `--gpu-memory-utilization`，保留已落盘 UUID，
并将调整前后的完整配置与已完成记录数追加到 `.jsonl.meta.history.jsonl`。
模型、数据源、seed、prompt template、解码参数及采样算法配置仍须一致。
不同批次与 kernel 路径可能产生不同回答，运行阶段记录用于保留这些执行差异。

RWKV 在每个采样批次内缓存已格式化的原题和专家条件前缀，续写仍直接追加
实际 response token IDs；批次结束后释放缓存。复读检测先匹配 16-token n-gram，
再比较完整重复片段，保持“至少 16 tokens 的片段连续三次重复”的判定不变。
checkpoint 的 Bot 模板会去掉 user 内容中的空行和行尾空白；OpenThink 的生成前缀
止于 `Bot✿<think`，随后直接拼接模型已生成的 response IDs，包括开头的 `>`。

2026-10-11 在同一批 256 道代表性训练题上的单卡对照，使用上述 checkpoint、
8 blocks、每块 1 次 suffix proposal、图执行和 0.94 显存预算；耗时包含模型启动：

| batch size | 耗时（秒） | 正常结束 | 正常结束 / 小时 | GPU 利用率中位数 | 显存峰值（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 1400.55 | 107 | 275.03 | 85% | 88.15 |
| 128 | 969.40 | 99 | 367.65 | 85% | 88.46 |
| 256 | 749.19 | 104 | 499.74 | 88% | 88.84 |

三组均正常退出；这里的正常结束只表示生成了终止 token，正确性与格式另行审计。
批量和 kernel 执行方式可能产生不同回答，比较使用有效回答产出速度。

随后在同一批 1024 道未完成原题上比较完整采样流程，固定每卡 400W、
8192-token 输出预算、8 blocks、每块 1 次 suffix proposal。以下耗时只计采样，
不含模型启动和审计文件写入；显存峰值来自采样期间的 1Hz NVML 监控。

| batch | prefill | 显存预算 | 前缀缓存 | 候选 / 小时 | 严格验算通过 / 小时 | GPU 利用率均值 | 显存峰值（GiB） |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 256 | 16384 | 0.94 | 关闭 | 1043.80 | 48.93 | 83.22% | 87.92 |
| 512 | 32768 | 0.94 | 关闭 | 1170.41 | 61.72 | 84.20% | 85.22 |
| 1024 | 16384 | 0.94 | 关闭 | 1023.44 | 49.97 | 87.94% | 88.53 |
| 1024 | 32768 | 0.94 | 关闭 | 1062.69 | 56.04 | 87.25% | 85.84 |
| 1024 | 32768 | 0.97 | 关闭 | 1083.86 | 57.16 | 86.63% | 88.67 |
| 1024 | 32768 | 0.94 | 开启 | 1132.25 | 51.97 | 93.61% | 85.81 |
| 1024 | 32768 | 0.97 | 开启 | 1087.76 | 50.99 | 93.80% | 88.67 |

各配置在不同 GPU 上运行；同型号 GPU 的运行频率和冷却条件仍可能不同。
这里的严格验算通过要求完整 `</think>`、最终 boxed answer、模型生成 `✿` 终止、
训练序列末尾为 `✿` + EOS 0，以及最终答案与 gold 等价；只检查最终答案，不证明推理步骤正确。
这些指标用于比较采样和可训练样本产出速度，不表示模型数学准确率有所提高。

生产采样采用每卡 batch 512、prefill 32768、0.94 显存预算、前缀缓存和图执行。
其余未完成的并发对照已按用户要求停止，全部 8 张 GPU 用于全量 CoT 生产采样；
采样和筛选完成后，管线自动启动已经验收的原生 infctx SFT 配置。

生产数据准备使用 `--skip-repetition`：检测到任意至少 16 tokens 的片段连续三次重复时，
立即跳过当前轨迹，记录 `finish_reason=repetition`，`input_ids/loss_mask` 为空。
复读 suffix 候选直接拒绝，不执行评分，不替换健康轨迹。跳过记录仍写入 JSONL，
以便统计和 resume；不会追加 EOS 将复读伪装成完整回答。该过滤是显式选项，
不改变其他任务的默认采样行为。

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

默认按 `SHA256(seed:question_id)` 保留 5% questions 作为 validation；缺少 question ID
的通用输入使用 UUID。两次调用的 seed 和
`--validation-fraction` 必须相同。很小的 smoke set 可能没有 validation question；正式训练
前确认两个 partition 均包含完整样本。`--partition all` 可输出全部数据用于单次链路验收。
validation 仅用于 SFT held-out loss，不是新任务 reward / accuracy 评测。

默认值：

- `--max-new-tokens 1856 --block-num 32 --mcmc-steps 10`；实际每块58 tokens。
- `--mcmc-steps 0` 是逐块续写的无搜索 rewrite 对照。
- RWKV Proposal 从 checkpoint 的 `generation_config.json` 读取 temperature、top-p、top-k、
  presence penalty、frequency penalty 和 penalty decay。`rwkv7-g1k-1.5b-20260930-ctx25600`
  的默认值为 0.96、0.76、32、1.0、0.1、0.988；multiplicative repetition penalty 为 1.0。
  其他模型的 Proposal 保持 temperature0.6、top-p1、top-k-1、repetition penalty1.1。
  Scoring：只有原题、temperature1、无截断采样限制，读取 raw prompt logprobs。
- `--max-model-len 32768` 包含专家提示、prefix 与 suffix；输入不截断，超预算报错。
  专家 reasoning 很长时需要明确增大预算，并实测完整专家提示与 suffix 搜索的吞吐。
- RWKV 使用 `wkv_mode=fp32io16`、FP16 inference weights、FP32 recurrent state、TP1、prefix caching；
  在加载前复用既有 FlashRWKV2 cache preflight。
- 默认 `bot/open_think`；通过 `--rwkv-prompt-template {bot,assistant,function_calling}` 和
  `--rwkv-generation-prompt {open_think,fake_think}` 明确选择模板。其他模型不传 RWKV 模板参数；
  非 chat 模型可以指定 `--model-type base`。HF model 可用 `--revision` 固定版本。
- RWKV `bot` proposal 的 User 内容为 `Expert solution:\n{solution}\nQuestion:\n{question}`，
  经 checkpoint 的 chat template 渲染后，再追加一次已有 response IDs：
  `User✿Expert solution:\n{solution}\nQuestion:\n{question}✿\nBot✿<think{response_prefix}`。
  完整专家推理与答案使用单个换行符拼接，原题后直接结束 User 消息；已有回答只出现在 Bot 前缀中。
  Proposal 保留专家条件，评分和最终 SFT 则使用只有原题的上下文。
  生成 `✿`（token ID 10060）时结束 Bot 回答，并追加 EOS（token ID 0）；模型直接生成 EOS 也会停止。
  separator 与 EOS 保留在 `response_ids` 和训练 loss mask 中。RWKV 输出 metadata 版本为 5。

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

此次 Nemotron 生产配置在采样审计 JSONL 之外，另外生成训练和验证 Parquet。
使用现有 `skyrl_gym.envs.gsm8k.utils.compute_strict_score` 对完整回答的最终 boxed 答案
与 `expected_answer` 做数学等价校验；仅保留 `strict_reward=1` 的记录，导出原始
`uuid/question_id/input_ids/loss_mask`，不重新 tokenize。复读、限长、格式不完整、
数学未通过记录都保留在审计输出中。此处校验的是最终答案；对部分符号等式、集合和
文字答案会漏判，不作为全集数学准确率。生产任务按原题 hash 将 Parquet 划入训练和验证，
消费全量源轨迹，生成后的可训练数量以实际筛选统计为准。

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
  rwkv_chunk_size=4096 \
  'model_config_kwargs={wkv_mode: fp32io16}' \
  num_epochs=1 batch_size=128 micro_train_batch_size_per_gpu=16 \
  placement.num_nodes=1 placement.num_gpus_per_node=8 \
  fsdp_config.fsdp_size=8 fsdp_config.reshard_after_forward=false \
  fsdp_config.sync_gradients_each_microbatch=false \
  optimizer_config.lr=1e-6 optimizer_config.max_grad_norm=1.0 \
  optimizer_config.scheduler=constant optimizer_config.num_warmup_steps=0 \
  optimizer_config.offload_after_step=false \
  max_length=null ckpt_path=outputs/sampling_sft/checkpoints \
  ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
  eval_before_train=true max_ckpts_to_keep=2 num_workers=0 \
  export_path=outputs/sampling_sft/hf_exports logger=wandb \
  project_name=rwkv_sampling_sft run_name=rwkv7-g1k-1.5b-20260930-ctx25600 \
  > /tmp/results_sampling_sft.log 2>&1
```

`max_length=null` 保留所有完整 token 序列，不再二次截断；按真实模型/显存预算准备数据。
`rwkv_chunk_size=4096` 使用现有 Transformers `RwkvCache` 的 WKV 与 token-shift
状态分块执行，checkpoint 重算每个 chunk 的 forward 和 LM-head logprobs，保留完整
跨块梯度，不 detach 历史。FSDP ranks 使用相同 chunk 次数，样本间独立初始化状态。
`rwkv_chunk_size=0` 保留既有整段路径。
此处 eval 记录 held-out SFT loss；reward/accuracy 曲线应另用既有 RWKV evaluation 流程，
不能将 SFT loss 当作 reward 验收。W&B 使用用户已有配置。GPU 数与 batch size 需匹配实际资源。

PRO6000×8 的硬件对照使用固定 128 条真实完整回答，共 293,236 个非 padding tokens，
每组执行 3 个优化步，以下吞吐取第 2、3 步的均值：

| microbatch / GPU | chunk size | FSDP size | reshard / 每个 microbatch 同步 | 真实 tokens / 秒 | 显存峰值（GiB） |
| ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 1024 | 8 | true / true | 12,885 | 13.58 |
| 4 | 1024 | 8 | false / false | 15,651 | 19.50 |
| 16 | 2048 | 8 | false / false | 18,760 | 33.72 |
| 16 | 4096 | 8 | false / false | 18,765 | 53.63 |
| 16 | 8192 | 8 | false / false | 18,072 | 87.27 |
| 16 | 4096 | 1 | false / false | 18,667 | 71.35 |

六组均正常完成，梯度有限且非零；相同初始权重的首步 loss 最大差异约 `4.6e-5`。
这组短实验选择 microbatch 16、chunk 4096、FSDP size 8，关闭 forward reshard 和
每个 microbatch 的梯度同步。2048 与 4096 的吞吐十分接近，表中差异不足以证明
4096 在所有输入上更快。全量 CoT 原题的 UTF-8 最大长度为 3471 bytes，
加上 8192-token 回答与模板预算后，完整 SFT 序列的保守 token 上界为 11,791。

学习率对照封存 52 条数学校验通过的训练回答和 7 条独立验证回答，global batch 32、
microbatch 4，其余优化器与 infctx 配置相同。每组从原始 checkpoint 开始训练 8 步，
AdamW betas 为 `[0.9, 0.999]`，weight decay 为 `0.01`，梯度裁剪为 `1.0`，
使用 constant schedule、无 warmup。四组基线验证 loss 均为 `0.447872`：

| learning rate | 第 8 步验证 loss |
| --- | ---: |
| `1e-6` | 0.412276 |
| `3e-6` | 0.458720 |
| `1e-5` | 0.568710 |
| `3e-5` | 0.892523 |

因此生产候选使用 `1e-6`；该结论来自小规模验证集与 global batch 32 的实验。
生产的 global batch 128 通过了硬件对照，后续通过全量独立验证分区监测学习表现。
所有训练从同一个原始 checkpoint 初始化，调参导出权重用于作答验收。

同一组 16 道独立验证题还进行了 question-only 生成对照，不提供专家解答，
使用 checkpoint 的六项默认解码参数、相同逐题 seed 和 8192-token 输出预算：

| checkpoint | 正常结束 | 完整推理与最终答案格式 | 最终答案通过数学 verifier |
| --- | ---: | ---: | ---: |
| 原始权重 | 2 | 2 | 0 |
| `1e-6`，8 steps | 7 | 7 | 0 |
| `3e-6`，8 steps | 9 | 9 | 1 |
| `1e-5`，8 steps | 0 | 0 | 0 |
| `3e-5`，8 steps | 3 | 2 | 0 |

所选 `1e-6` 模型的 7 条完整回答均无精确片段复读，实际尾部为 token IDs `[10060, 0]`。
正常结束有所改善，数学能力提升尚未在这组生成对照中得到证明。

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
