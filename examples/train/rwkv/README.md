# RWKV7 GSM8K GRPO

This recipe runs the native SkyRL FSDP Trainer, vLLM InferenceEngines, weight sync, and existing GSM8K Environment. It is a 50-step functional acceptance run, not a throughput benchmark or convergence claim.

## Artifact contract

- Repository: `rwkv-rs/rwkv7-g1-st`
- Subfolder: `rwkv7-g1j-1.5b-20260831-ctx16384`
- Revision: `e1a670a5523742b5cfe8cb6759c1eb8f1d88b637`
- Architecture: `rwkv7`
- `model_type`: `rwkv`
- `wkv_mode`: `fp32io16`
- Expected weight dtype: BF16 Safetensors

The `[rwkv]` extra pins the matching Transformers, tokenizer, vLLM-RWKV, FlashRWKV2, and torch cu130 dependency chain. Use the project-local `.venv`; do not reuse another project's environment.

```bash
uv sync --extra rwkv
```

The launcher uses the persistent `$HOME/.cache/skyrl-rwkv/torch_extensions` cache and refuses to start if the FlashRWKV2 artifact is absent or stale. Set `MAX_JOBS=2` (or another low value) for the one-time build to limit Ninja memory use. Training workers only load the verified cached artifact.

## Download and verify the model

```bash
MODEL_REPO=rwkv-rs/rwkv7-g1-st
MODEL_REVISION=e1a670a5523742b5cfe8cb6759c1eb8f1d88b637
MODEL_SUBFOLDER=rwkv7-g1j-1.5b-20260831-ctx16384
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

The local model directory is `$HOME/Weights/RWKV/hf/$MODEL_SUBFOLDER`; keep the pinned revision and resulting Safetensors checksums with the run report. The Trainer and every vLLM engine must receive the same `MODEL_DIR`.

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

If micro-batch 2 does not fit, record the OOM and retry with `MICRO_BATCH_SIZE=1`; do not change the global train or policy mini-batch sizes. Successful acceptance requires all 50 optimizer steps, every trainer-to-inference weight update, aligned rollout token/logprob/loss-mask lengths, finite logprob-difference metrics without abnormal jumps, step-50 checkpoint/export/eval artifacts, and an online W&B run containing training, reward, evaluation, system-resource, and logprob-alignment curves.

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
fresh batch is consumed by exactly one optimizer step. The launcher keeps generated action tokens
in the loss mask for truncated rollouts; those rollouts retain zero reward and contribute signed
failure feedback. The FlashREINFORCE launcher enables the opt-in strict GSM8K reward: the response
must contain one non-empty thought, an answer after `</think>`, a real EOS token, and no truncation.
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

## RWKV trajectory logging

`trainer.trajectory_logger=rwkv` writes accumulating W&B tables in `INCREMENTAL` mode: each
write validates and serializes only new rows. Table processing and subsequent metric writes
share one background worker to preserve step/commit ordering. Its queue is bounded to 16
writes with backpressure, and `Tracking.finish()` drains pending writes before finishing
W&B uploads. Other trajectory loggers retain their existing logging behavior.

Sample formatting remains bounded by `trainer.num_logger_train_samples` (20 in the
FlashREINFORCE launcher). Detailed rollout JSONL dumps, 50-step evaluation, checkpoints,
and HF exports are independent of the table writer.
