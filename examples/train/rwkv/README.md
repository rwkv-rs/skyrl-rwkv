# RWKV7 GSM8K GRPO

This recipe runs the native SkyRL FSDP Trainer, vLLM InferenceEngines, weight sync, and existing GSM8K Environment. It is a 50-step functional acceptance run, not a throughput benchmark or convergence claim.

## Artifact contract

- Repository: `rwkv-rs/rwkv7-g1-st`
- Subfolder / default weight: `rwkv7-g1k-1.5b-20260930-ctx25600`
- Revision: `b672701cccf4bddcc66f65c8c9a468b776c9836a`
- Source repository: `BlinkDL/rwkv7-g1`
- Source revision: `cd67fb95fa9e2ce8757f8d21d713e74a9c788118`
- Source file: `rwkv7-g1k-1.5b-20260930-ctx25600.pth`
- Source SHA256: `dfc1fc450641d5e00c5f412b5530a17b2a734ac02bde04d98c57608f5daaf773`
- HF converter: `rwkv-rs/transformers-rwkv`, commit `25674160e9786b191aa3642045f3615c05502af3`
- Tokenizer: `rwkv-rs/rwkv7-g1-st`, revision `fd122cc7244c28db19beceb398aa033c35576b71`
- Architecture: `rwkv7`
- `model_type`: `rwkv`
- `wkv_mode`: `fp32io16`
- Expected weight dtype: BF16 Safetensors

The `[rwkv]` extra pins the matching Transformers, tokenizer, vLLM-RWKV, FlashRWKV2, and torch cu130 dependency chain. Use the project-local `.venv`; do not reuse another project's environment.

```bash
uv sync --extra rwkv
```

The launcher uses the persistent `$HOME/.cache/skyrl-rwkv/torch_extensions` cache and refuses to start if the FlashRWKV2 artifact is absent or stale. Set `MAX_JOBS=2` (or another low value) for the one-time build to limit Ninja memory use. Training workers only load the verified cached artifact.

## Full UltraData rollout

`rollout.py` defaults to **every row in every JSONL shard**, with 64 samples per question,
using `rwkv7-g1k-1.5b-20260930-ctx25600`. It reads
`$HOME/data/UltraData-RL-2609/files/{Math,Code,Long_Context,Knowledge}`. A positive
`--limit` is an explicit, domain-balanced smoke-test limit; the default `--limit 0` is unlimited.

After preparing this project's RWKV `.venv` and FlashRWKV2 cache:

```bash
uv run --isolated --no-project .venv/bin/python examples/train/rwkv/rollout.py \
  > /tmp/results_ultradata_rollout.log 2>&1
```

By default, every CUDA-visible GPU gets one TP1 engine with `max_num_seqs=1024`.
The HTTP window is twice the running-sequence capacity so completed sequences are immediately
replaced: **16,384 in-flight generations across eight engines**, with at most 8,192 sequences
running on the GPUs. `CUDA_VISIBLE_DEVICES` restricts placement; `--base-url` reuses comma-separated
external engines instead of starting local ones. `--concurrency`, `--max-num-seqs`, and
`--score-workers` are explicit overrides.

Prompts are rendered and tokenized once per question. Serialized token-input requests are reused
for its 64 samples, which are assigned to the least-loaded engine together for prefix reuse.
The engines use `--skip-tokenizer-init`, `generation_config=vllm`, FP16 weights, FP32 recurrent
states, prefix caching, and 8,192-token chunked prefill. Token-only completions are decoded locally;
Math retains the exact final generated token for strict EOS checking. Evaluation sampling remains
temperature 0.96, top-p 0.76, top-k 32, presence penalty 1.0, frequency penalty 0.1, and penalty decay 0.988.

The default serving budget is **1,048,576 tokens**, including up to 8,192 generated tokens.
RWKV has no fixed positional embedding table, so chunked recurrent inference supports inputs
beyond the checkpoint's 25,600-token training context. This serving budget is not a claim of
long-context accuracy. Prompts are never truncated. A row beyond an explicitly configured budget
is recorded as `prompt_too_long`, rather than silently excluded from the dataset.

Reading, inference, CPU grading, and result writing run continuously through bounded queues;
the script never materializes the full dataset or all generation tasks. Graders reread golden
Code cases from their source byte offsets rather than retaining or repeatedly transferring them.
Repeated final programs within a question reuse their verdict while all 64 samples still count.
The automatic CPU budget leaves two cores per engine and uses spawned grading processes
(48 on the 128-logical-CPU, eight-GPU test host).

Each run gets a timestamped model-specific output directory containing:

- `correct_counts_by_question.jsonl`: incrementally flushed UUID, domain, prompt length, verdict counts,
  and reason counts for **every completed question**. `summary.json` references this file instead
  of keeping an unbounded UUID map in memory.
- `summary.json`: atomically refreshed every 30 seconds with domain metrics, 0–64 histograms,
  generation/token counters, queue depths, endpoints, and the exact runtime configuration.
- `pass_rate_histogram.svg` and up to 20 UUID-disjoint examples in each of
  `correct_samples.jsonl`, `wrong_samples.jsonl`, and `unanswered_samples.jsonl`, written on exit.
- `vllm_gpu_*.log` and `vllm.pids`: logs and identities of engines owned by this run.

Existing result files are not overwritten. Completion, failure, and SIGTERM shut down only engines
started by this invocation; externally supplied engines remain running.

## Download and verify the model

The authoritative HF repository already contains the converted BF16 G1k checkpoint. No local
conversion is needed. Both training launchers default to this exact release; `MODEL_DIR` remains
an explicit override. Keep the downloaded `PROVENANCE.md` and shard checksums with the experiment.

```bash
MODEL_REPO=rwkv-rs/rwkv7-g1-st
MODEL_REVISION=b672701cccf4bddcc66f65c8c9a468b776c9836a
MODEL_SUBFOLDER=rwkv7-g1k-1.5b-20260930-ctx25600
MODEL_ROOT="$HOME/Weights/RWKV/hf"

hf download "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --include "$MODEL_SUBFOLDER/*" \
  --local-dir "$MODEL_ROOT"

MODEL_DIR="$MODEL_ROOT/$MODEL_SUBFOLDER"
MODEL_DIR="$MODEL_DIR" uv run --no-sync --extra rwkv python -c \
  'import json, os; from pathlib import Path; p=Path(os.environ["MODEL_DIR"]); c=json.loads((p/"config.json").read_text()); assert c["model_type"]=="rwkv"; assert c["architecture_version"]=="rwkv7"; assert c["wkv_mode"]=="fp32io16"; print(c["architectures"], c["wkv_mode"])'
sha256sum "$MODEL_DIR"/*.safetensors
```

Prepare the native FlashRWKV2 extension once, in the same project environment and cache used by training. This invokes the vLLM-RWKV loader and the actual native build; importing `FlashRWKV2` alone is not sufficient:

```bash
MAX_JOBS=2 bash examples/train/rwkv/run_rwkv_grpo.sh --prepare-flashrwkv2
```

The local model directory is `$HOME/Weights/RWKV/hf/$MODEL_SUBFOLDER`; keep the pinned source/converter revisions and resulting Safetensors checksums with the run report. The Trainer and every vLLM engine must receive the same `MODEL_DIR`.

## Verify the three prompt styles

The model's own `chat_template.jinja` accepts `rwkv_prompt_template` values `bot`, `assistant`, and `function_calling`, plus `rwkv_generation_prompt` values `open_think` and `fake_think`. Training uses `bot + open_think`.

```bash
MODEL_DIR="$MODEL_DIR" uv run --no-sync --extra rwkv python - <<'PY'
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"], trust_remote_code=False)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
for style in ("bot", "assistant", "function_calling"):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        rwkv_prompt_template=style,
        rwkv_generation_prompt="open_think",
    )
    assert "<think" in prompt
    print(style, repr(prompt))
PY
```

## Prepare GSM8K

```bash
uv run --no-sync --extra rwkv examples/train/gsm8k/gsm8k_dataset.py \
  --output_dir "$HOME/data/gsm8k"
```

## Private W&B configuration

Create the ignored project-root `.env` without putting credentials in shell history or logs:

```bash
install -m 0600 /dev/null .env
${EDITOR:-vi} .env
```

Add `WANDB_API_KEY=...` inside `.env`. Never commit the file or include its contents in logs. Verify only its mode and ignored status:

```bash
stat -c '%a %n' .env
git check-ignore -v .env
```

The launch script loads this project-root `.env` without printing its contents.

## Run

The defaults use a BF16 Trainer, FP16 vLLM-RWKV inference, 8 GPUs, eight TP1 colocated inference engines, GRPO, `train_batch_size=64`, four samples per prompt, global policy mini-batch 64, per-GPU micro-batch 2, gradient checkpointing, and one optimizer step per training step. Step 50 runs evaluation and writes both a resumable checkpoint and an HF export.

```bash
MODEL_DIR="$MODEL_DIR" bash examples/train/rwkv/run_rwkv_grpo.sh
```

Choose `MICRO_BATCH_SIZE` for the available GPU memory while keeping global train and policy mini-batch sizes fixed. Successful acceptance requires all 50 optimizer steps, every trainer-to-inference weight update, aligned rollout token/logprob/loss-mask lengths, finite logprob-difference metrics without abnormal jumps, step-50 checkpoint/export/eval artifacts, and an online W&B run containing training, reward, evaluation, system-resource, and logprob-alignment curves.

## FlashREINFORCE and algorithm ablations

FlashREINFORCE is available through the native policy-loss registry. It uses one rollout per
prompt, batch-centered signed rewards, the actual sampler logprob, a sequence-level Bernoulli-KL
admission gate, and `sequence_mean` trajectory normalization. Its reference-free base objective
therefore disables the KL/reference-model terms:

Use the dedicated FlashREINFORCE entry point; do not pass FlashREINFORCE overrides to the GRPO GSM8K entry point:

```bash
MODEL_DIR="$MODEL_DIR" bash examples/train/rwkv/run_rwkv_flashreinforce.sh
```

The synchronous entry point above keeps `trainer.train_batch_size == trainer.policy_mini_batch_size`; each
fresh batch is consumed by exactly one optimizer step. `MINI_BATCH_SIZE` sets both sizes (default 128);
`MICRO_BATCH_SIZE` controls per-GPU memory use (default 32 for this RWKV7-1.5B recipe). `NUM_POLICY_GPUS` and
`NUM_INFERENCE_GPUS` independently select training GPUs and TP1 inference engines, each falling
back to `NUM_GPUS` (default 8). Unequal counts require `trainer.placement.colocate_all=false`.
The optimizer is AdamW with LR `1e-6`, cosine decay, weight decay `0.1`, and gradient clip `1.0`.
Training sampling is temperature `1.0`, top-p `1.0`, top-k `-1`, without penalties; evaluation
sampling is unchanged. The launcher keeps generated action tokens
in the loss mask for truncated rollouts; those rollouts retain zero reward and contribute signed
failure feedback. The FlashREINFORCE launcher enables the opt-in strict GSM8K reward: the response
must contain one non-empty thought, exactly one `</think>`, a final `\boxed{...}` answer
(or a GSM8K `####` answer) after it, a real EOS token, and no truncation. The extracted answer
is checked with `math_verify.verify(..., strict=False)` for mathematical equivalence, not string equality.
`examples/train/rwkv/rollout.py` uses the same `gsm8k_rwkv.utils.compute_strict_score` for its Math domain
and requests generated token IDs to verify the actual final EOS token. Other domains and non-strict
GSM8K rewards retain their existing scoring.
For the asynchronous pipeline, use the existing
`examples.train.fully_async.main_fully_async` entrypoint, set `trainer.fully_async.enabled=true` and
`generator.batched=false`; also pass `environment.skyrl_gym.gsm8k_rwkv.strict_reward=true`. The default gate is `3e-3`; monitor
`policy/loss_metrics/flashreinforce/acceptance_rate` and the rollout/trainer logprob-difference
metrics before tuning it.

BPO is implemented as a separate `policy_loss_type=bpo` / `advantage_estimator=bpo` ablation. For
example, on the synchronous RWKV script use `trainer.algorithm.use_kl_loss=false`,
`trainer.algorithm.use_kl_in_reward=false`, and `generator.n_samples_per_prompt=4` in addition to
those two loss/estimator overrides. BPO requires at least two sibling rollouts because its prompt-value
estimate is a group mean, so it is not mathematically interchangeable with one-rollout
FlashREINFORCE.

Score Centering is opt-in through `trainer.algorithm.flashreinforce.score_centering=true`.
The sampler captures raw support logprobs independently of sampling `top_k`, and the Trainer
replays logits over the same support (128 candidates by default). For the untruncated training
sampling distribution, use `generator.sampling_params.temperature=1.0`,
`generator.sampling_params.top_p=1.0`, `generator.sampling_params.top_k=-1`, and
`generator.sampling_params.additional_kwargs=null`.

The FlashREINFORCE launcher uses token-in/token-out inference with
`generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true`. SkyRL retains the
local HF tokenizer for prompt templates and response decoding; numeric token IDs, EOS, and
logprobs are unchanged. This avoids vLLM decoding and UTF-8-correcting every support candidate
at each generation step on the API event loop. Support capture also uses `FINAL_ONLY` outputs
while retaining all per-token sampled scores and support rows.

## FlashREINFORCE resource tuning

The RWKV FlashREINFORCE launcher keeps parameters unsharded after forward and uses
`trainer.policy.fsdp_config.sync_gradients_each_microbatch=false`. Intermediate microbatches
accumulate gradients locally and retain full parameters; only the final microbatch synchronizes
gradients and restores parameter sharding before the optimizer step. The generic FSDP default
remains synchronization after every microbatch. Native CPU offload is incompatible with deferred
synchronization. Loss normalization, clipping, and one optimizer update per full batch are unchanged.

On the 96 GiB RTX PRO 6000 host, batch 768 across six training GPUs with microbatch 32 uses four
accumulation rounds per GPU; the two separate TP1 inference engines retain token-only support capture.
Before deferred synchronization was added, retaining parameters after forward gave approximately
16 seconds of policy training versus 105 seconds with microbatch 2. These LR=0 diagnostics do not
establish convergence or a globally optimal GPU split. Measure useful tokens and admitted trajectories
per GPU-hour, and retain memory headroom at the maximum 512+1024 token length.

The hyperparameter reference is the Qwen2.5-Math-1.5B experiment in
[FlashREINFORCE, Appendix A.3, Table 13](https://www.researchgate.net/publication/414274571_FlashREINFORCE_FLASHREINFORCE_CRITIC-FREE_SINGLE-ROLLOUT_ASYNCHRONOUS_RL_FOR_AGENTIC_LANGUAGE_MODELS):
128 distinct prompts, one rollout per prompt, one full-batch update without reuse, AdamW
LR `1e-6`, cosine decay, weight decay `0.1`, gradient clip `1.0`, and gate `3e-3`.
The RWKV/GSM8K token budgets and evaluation sampling remain unchanged. Batch 512 below is a
resource-tuning candidate, not a paper-reproduction setting; do not linearly scale its LR.
[Score Centering, Appendix B.1](https://arxiv.org/html/2609.20807v1#A2.SS1) instead uses SGD
LR `0.01` and eight completions per prompt for GRPO experiments. Its correction does not
require importing that optimizer or sibling-rollout count into FlashREINFORCE.

The following candidate separates two FSDP training GPUs from one TP1 inference GPU. It
reserves three rather than eight GPUs, but fit, throughput, and training quality still require
a healthy-host benchmark. Start with `MINI_BATCH_SIZE=128` for the paper-sized baseline, then
512; compare admitted trajectories per GPU-hour and reward versus total rollouts, not just
seconds per optimizer step. In the 512 configuration, micro-batch 2 on two training ranks means
128 forward/backward accumulation rounds per update. A larger global batch amortizes weight
sync but does not imply faster steps or allow a larger micro-batch automatically.

From the local checkout, use the verified SSH launcher after host memory faults are repaired:

```bash
NUM_POLICY_GPUS=2 NUM_INFERENCE_GPUS=1 MINI_BATCH_SIZE=512 MICRO_BATCH_SIZE=2 \
SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=64 \
TRAINING_ENTRYPOINT=examples.train.fully_async.main_fully_async \
RUN_NAME=rwkv7-g1k-1.5b-20260930-ctx25600-gsm8k-flashreinforce-sc-2train-1infer-b512 \
./temp/run.sh \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=1 \
  trainer.fully_async.num_parallel_generation_workers=512 \
  trainer.fully_async.save_at_epoch_end=false \
  trainer.algorithm.flashreinforce.score_centering=true \
  trainer.algorithm.flashreinforce.sequence_kl_threshold=0.003 \
  trainer.max_training_steps=1000 trainer.epochs=1000 \
  generator.inference_engine.max_num_seqs=128 \
  > /tmp/results_rwkv_flashreinforce_sc_2train_1infer_b512.log 2>&1
```

For batch 128, also change `num_parallel_generation_workers` to 128 and use a distinct run name.
Generation workers are async request tasks, not GPU replicas; the trainer requires their count
between `MINI_BATCH_SIZE` and `MINI_BATCH_SIZE * (max_staleness_steps + 1)`. `max_num_seqs=128`
bounds simultaneously active sequences on the single inference engine.
`SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=64` independently caps in-flight HTTP generation requests
per engine; remaining generation tasks wait on the client's semaphore rather than overwhelming
the router. The SSH launcher forwards this opt-in override. Staleness control is
capacity-based: still inspect the actual `async/staleness_*` metrics and admission rate.
Keep the gate and raw 128-candidate Score-Centering support unchanged.

Benchmark micro-batches at the same global batch, retaining gradient checkpointing
and monitoring peak memory. Also compare one training GPU plus one inference GPU if it fits;
two plus one is not asserted to be optimal. Measure `timing/policy_train`,
`timing/sync_weights_pause_generation`, `timing/sync_weights_only_transfer`,
`timing/sync_weights_resume_generation`, and buffer wait alongside GPU utilization.

`trainer.fully_async.save_at_epoch_end=false` keeps 50-step evaluation/checkpoint/export
intervals and the final save, while disabling additional epoch-boundary saves. Its default is
`true`, preserving other fully-async callers. The earlier saves at steps 116, 232, and 348
were epoch boundaries (`floor(7473 / 64) = 116`); batch 512 would shorten epochs to 14 updates
and otherwise create substantially more weight files. Existing retention settings are unchanged.

## RWKV trajectory logging

`trainer.trajectory_logger=rwkv` writes accumulating W&B tables in `INCREMENTAL` mode: each
write validates and serializes only new rows. Table processing and subsequent metric writes
share one background worker to preserve step/commit ordering. Its queue is bounded to 16
writes with backpressure, and `Tracking.finish()` drains pending writes before finishing
W&B uploads. Other trajectory loggers retain their existing logging behavior.
The W&B run workspace displays the latest 100 table increments; detailed rollout
JSONL dumps retain every step.

Sample formatting remains bounded by `trainer.num_logger_train_samples` (20 in the
FlashREINFORCE launcher). Detailed rollout JSONL dumps, 50-step evaluation, checkpoints,
and HF exports are independent of the table writer.
