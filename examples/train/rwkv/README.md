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

## Download and verify the model

```bash
MODEL_REPO=rwkv-rs/rwkv7-g1-st
MODEL_REVISION=e1a670a5523742b5cfe8cb6759c1eb8f1d88b637
MODEL_SUBFOLDER=rwkv7-g1j-1.5b-20260831-ctx16384
MODEL_ROOT="$HOME/models/rwkv7-g1-st-$MODEL_REVISION"

hf download "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --include "$MODEL_SUBFOLDER/*" \
  --local-dir "$MODEL_ROOT"

MODEL_DIR="$MODEL_ROOT/$MODEL_SUBFOLDER"
MODEL_DIR="$MODEL_DIR" uv run --isolated --extra rwkv python -c \
  'import json, os; from pathlib import Path; p=Path(os.environ["MODEL_DIR"]); c=json.loads((p/"config.json").read_text()); assert c["model_type"]=="rwkv"; assert c["architecture_version"]=="rwkv7"; assert c["wkv_mode"]=="fp32io16"; print(c["architectures"], c["wkv_mode"])'
sha256sum "$MODEL_DIR"/*.safetensors
```

Keep the revision and the resulting Safetensors checksums with the run report. The Trainer and every vLLM engine must receive the same `MODEL_DIR`.

## Verify the three prompt styles

The model's own `chat_template.jinja` accepts `rwkv_prompt_template` values `bot`, `assistant`, and `function_calling`, plus `rwkv_generation_prompt` values `open_think` and `fake_think`. Training uses `bot + open_think`.

```bash
MODEL_DIR="$MODEL_DIR" uv run --isolated --extra rwkv python - <<'PY'
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
uv run --isolated --extra skyrl-train examples/train/gsm8k/gsm8k_dataset.py \
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

## Run

The defaults use 8 GPUs, eight TP1 colocated inference engines, GRPO, `train_batch_size=64`, four samples per prompt, global policy mini-batch 64, per-GPU micro-batch 2, gradient checkpointing, and one optimizer step per training step. Step 50 runs evaluation and writes both a resumable checkpoint and an HF export.

```bash
MODEL_DIR="$MODEL_DIR" bash examples/train/rwkv/run_rwkv_gsm8k.sh
```

If micro-batch 2 does not fit, record the OOM and retry with `MICRO_BATCH_SIZE=1`; do not change the global train or policy mini-batch sizes. Successful acceptance requires all 50 optimizer steps, every trainer-to-inference weight update, aligned rollout token/logprob/loss-mask lengths, finite logprob-difference metrics without abnormal jumps, step-50 checkpoint/export/eval artifacts, and an online W&B run containing training, reward, evaluation, system-resource, and logprob-alignment curves.
