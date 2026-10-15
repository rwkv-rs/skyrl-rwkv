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
Before feeding data, each owned engine must pass a deterministic strict-Math/EOS smoke check;
a failing engine stops the invocation before the full request queue is created.
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

Math uses the shared strict scorer and mathematical equivalence, preserving commas in coordinates
and sets while normalizing thousands separators only in scalar expressions. Knowledge and Long_Context use
normalized final-answer text/option matching, not semantic judging of free-form paraphrases.
Code executes the extracted program against the golden stdin/stdout cases.

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
- With `--save-rollouts`, `rollouts/<domain>/<UUID-SHA256>.jsonl`: all scored samples,
  with exact prompt/response token IDs, sample indices, rewards, and termination reasons.
  Full response IDs are staged on disk as they arrive, not accumulated in RAM. Each
  question shard is atomically published before its completed-count record; resume
  replaces any unmarked shard instead of appending duplicate samples.

Existing result files are not overwritten. To resume an interrupted run:

```bash
uv run --isolated --no-project .venv/bin/python examples/train/rwkv/rollout.py \
  --output outputs/<existing-run> --resume > /tmp/results_ultradata_resume.log 2>&1
```

Resume restores counts, histograms, reasons, and saved examples, then skips completed UUIDs before
tokenization. Unfinished questions are regenerated with all 64 samples; each completed question is
appended exactly once. The model, question limit, context/generation budgets, and sampling settings
must match the saved run, including `scoring_version`. Version 2 preserves mathematical commas;
older results must use a new output directory rather than mixing scores across versions.
Generation counters include work from earlier attempts, while domain
rollout counts contain only completed question groups. Resume retains only completed UUIDs, not
prompts, golden cases, or a per-question score map.

Completion, failure, and SIGTERM shut down only engines started by this invocation;
externally supplied engines remain running.

## Offline value pretraining

`skyrl.train.main_value` trains an independent token-level scalar critic from offline
rollouts, using the native critic loader and `FSDPStrategy` (no inference engines or
Ray cluster). The data/alignment implementation is adapted from
[ReDiPPO standard critic pretraining](https://github.com/cii030/ReDiPPO/tree/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining),
under Apache-2.0 (see the repository `LICENSE`). It does not include ReDiPPO's dual
critics. This is critic initialization, not the full online SAO algorithm.

### Collect rollouts and build data

Use `--sampling rollout` to match the training policy (temperature 1, top-p 1,
top-k -1, no penalties). The existing evaluation sampling remains the default.
This explicit 128-question run is a small data-collection example; omit `--limit`
to collect the full UltraData corpus. Each question still has 64 scored samples.

```bash
MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"
ROLLOUT_DIR=outputs/rwkv7-g1k-1.5b-20260930-ctx25600-value-rollouts
DATA_DIR="$HOME/data/rwkv-value"

uv run --isolated --no-project .venv/bin/python examples/train/rwkv/rollout.py \
  --model "$MODEL_DIR" --output "$ROLLOUT_DIR" \
  --sampling rollout --save-rollouts --limit 128 \
  > /tmp/results_rwkv_value_rollouts.log 2>&1

uv run --isolated --no-project .venv/bin/python examples/train/rwkv/build_value_data.py \
  --input "$ROLLOUT_DIR" --output "$DATA_DIR" \
  --validation-fraction 0.05 --seed 42 --max-length 8192 \
  > /tmp/results_rwkv_build_value_data.log 2>&1
```

The builder streams complete question shards into `train.jsonl`, `validation.jsonl`,
and `manifest.json`, published together in a new output directory. UUID-hash splitting
keeps all samples of each question in one split and is stable on repeated builds.
It retains positive and zero rewards without balancing. Empty, over-length, and
truncated samples without explicit bootstrap returns are excluded and counted in the
manifest. Thus the default MC dataset is conditioned on completion within the length
budget; it does not claim to estimate values for excluded truncated trajectories.
The manifest records the source model, sampling configuration, split seed, and counts.

### Dataset contract and pretraining

Store exact rollout token IDs, not re-tokenized decoded text, as unpadded JSONL,
Parquet, Arrow, or a HuggingFace `Dataset.save_to_disk` directory:

```json
{"input_ids": [11, 12, 13, 14, 15, 16], "loss_mask": [0, 0, 1, 0, 0, 1], "reward": 1.0, "truncated": false}
```

- `input_ids` includes the prompt and all assistant/tool turns.
- `loss_mask` covers the entire sequence: 1 for generated action tokens, 0 for prompt
  and tool-observation tokens. The first token must be context (mask 0).
- For completed trajectories with an undiscounted terminal reward, every action gets
  the same `reward` target. For discounted or intermediate rewards, supply explicit
  full-sequence `returns` instead. Truncated trajectories require `returns`, including
  the chosen bootstrap; the trainer does not invent a terminal target.
- Values are aligned before the target action: the first response token uses the
  final prompt state. Tool observations remain in recurrent context but not the loss.
- Over-length rows are dropped rather than cutting completed trajectories. Split
  training/validation by question UUID so repeated rollouts do not leak across splits.

`build_value_data.py` concatenates stored prompt/response IDs without re-tokenization.
For this single-turn collector it masks the prompt to 0 and generated actions to 1;
explicit `response_mask` fields in source rows preserve tool-observation exclusions.
The trainer consumes these exact tokens and full-sequence masks.

After preparing the project-local RWKV environment and FlashRWKV2 cache:

```bash
MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"
TORCH_EXTENSIONS_DIR="$HOME/.cache/skyrl-rwkv/torch_extensions" \
uv run --isolated --no-project .venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m skyrl.train.main_value \
  --model "$MODEL_DIR" \
  --train-data "$HOME/data/rwkv-value/train.jsonl" \
  --val-data "$HOME/data/rwkv-value/validation.jsonl" \
  --output-dir outputs/rwkv7-g1k-1.5b-20260930-ctx25600-value \
  --epochs 2 --lr 5e-6 --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 8192 --gradient-checkpointing \
  > /tmp/results_rwkv_value_pretraining.log 2>&1
```

Default: native BF16 RWKV, `wkv_mode=fp32io16`, raw scalar predictions and FP32 masked
MSE, globally normalized over action tokens across ranks and accumulation windows.
Unlike ReDiPPO's binary-outcome sigmoid, raw values match SkyRL's online critic and
arbitrary reward scales. Parameters, optimizer, and HF weights are saved and held-out
evaluation runs every 50 optimizer steps and at the final step. W&B records training
MSE, token/sequence validation MSE, value/target means, and mask-aware explained
variance (`--report-to none` disables it).

`step_N/` holds native FSDP checkpoints and `step_N/huggingface/` holds HF exports;
`best/` holds the critic with the lowest held-out sequence MSE. Load `best/` through
`get_llm_for_sequence_regression(path, "critic")`, or set
`trainer.critic.model.path=<output-dir>/best` for an online critic. Keep the actor's
model path separate so online initialization does not reset the pretrained value head.
The offline entry point starts a fresh optimizer; it does not resume interrupted runs.

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

## On-policy distillation

`run_rwkv_on_policy_distill.sh` uses the native
`examples.train.on_policy_distillation.main_on_policy_distill` entry point to distill
`rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora` into
`rwkv7-g1k-1.5b-20260930-ctx25600`.
The student generates completions; the frozen teacher occupies the reference-model role
and scores the same token IDs. The native trainer replaces task rewards with masked
`teacher_logprob - student_logprob`, passes them through the `no_op` advantage estimator,
and updates the student with `importance_sampling` loss. Task answer scores remain
available for evaluation. No critic is used.

Prepare the project-local RWKV environment and FlashRWKV2 cache as described above.
The teacher path must contain the complete merged AntiDoom HF checkpoint. Download the
student from the original release revision:

```bash
MODEL_ROOT="$HOME/Weights/RWKV/hf"
hf download rwkv-rs/rwkv7-g1-st \
  --revision b672701cccf4bddcc66f65c8c9a468b776c9836a \
  --include 'rwkv7-g1k-1.5b-20260930-ctx25600/*' \
  --local-dir "$MODEL_ROOT" \
  > /tmp/results_rwkv_opd_download.log 2>&1
```

The recipe uses the existing local UltraData-RL-2609 corpus for training and a prepared
AIME 2024 Parquet for evaluation. Convert all Math, Code, Long_Context, and Knowledge
shards without collecting new rollouts:

```bash
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.prepare_opd \
  --input "$HOME/data/UltraData-RL-2609/files" \
  --output "$HOME/data/ultradata-opd/train.parquet" \
  > /tmp/results_rwkv_opd_prepare_data.log 2>&1
```

The converter preserves each question's UUID and domain and writes domain counts to
`train.manifest.json`. It reuses the native AIME single-turn environment; training
rewards are replaced by the OPD teacher signal. Only Math rows carry a mathematical
answer; the other domains have an empty answer and their task scores are not accuracy
metrics. Code questions are distilled without executing test cases. The native prompt
length filter applies after conversion. `TEST_FILE` defaults to
`$HOME/data/dapo/aime-2024-cleaned.parquet`; it can point to another native AIME Parquet.

Run from the project root after configuring W&B in `.env`:

```bash
STUDENT_MODEL="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600" \
TEACHER_MODEL="$HOME/Weights/RWKV/hf/rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora" \
  bash examples/train/rwkv/run_rwkv_on_policy_distill.sh \
  > /tmp/results_rwkv_opd.log 2>&1
```

Defaults: 8 colocated GPUs, eight TP1 student inference engines, BF16 training and
teacher scoring with `wkv_mode=fp32io16`, FP16 inference, gradient checkpointing,
512 prompts per batch, one completion per prompt, policy mini-batch 512, per-GPU
micro-batch 8 for training and 4 for forward passes, and AdamW LR `2e-6`.
Each fresh rollout batch produces one optimizer
step followed by native student-to-inference weight sync; the teacher stays frozen.
Prompts use `bot + open_think`, with a 17408-token prompt budget and an 8192-token
generation budget. Training sampling uses temperature `1`, top-p `1`, top-k `-1`,
and disabled penalties. Evaluation uses temperature `0.96`, top-p `0.76`, top-k `32`,
presence penalty `1.0`, frequency penalty `0.1`, penalty decay `0.988`, and 32 samples
per prompt.

RWKV Rapid-Sampling uses `raw_logprobs`. The launcher selects it explicitly;
the inference setup requires an unmodified training distribution so these equal
the sampling logprobs used by OPD. Evaluation logprobs do not enter the loss.
Each engine has a 2-GiB recurrent-state cache budget and up to 128 active requests.

The default run covers one complete epoch. Evaluation runs before training and
every 50 steps; resumable checkpoints and HF exports are saved every 50 steps.
Training metrics, evaluation scores, and sampled trajectories are uploaded to W&B;
detailed training rollout dumps are written under `$OUTPUT_ROOT/exports/dumped_rollouts`.
`OUTPUT_ROOT` defaults to `$HOME/skyrl-rwkv-runs/$RUN_NAME`.

`STUDENT_MODEL`, `TEACHER_MODEL`, `DATA_DIR`, `TRAIN_FILE`, `TEST_FILE`, `NUM_GPUS`,
`MICRO_BATCH_SIZE`, `MICRO_FORWARD_BATCH_SIZE`, `MINI_BATCH_SIZE`, `LR`, `USE_NORA`,
`LORA_RANK`, `LORA_ALPHA`, `LOGGER`, `RUN_NAME`, and `OUTPUT_ROOT` accept environment overrides.
Additional native config overrides are forwarded at the end of the command, for example:

```bash
RUN_NAME=rwkv7-g1k-1.5b-20260930-ctx25600-ultradata-opd-long \
  bash examples/train/rwkv/run_rwkv_on_policy_distill.sh \
  trainer.max_training_steps=500 \
  > /tmp/results_rwkv_opd_long.log 2>&1
```

### Optional NoRA policy

The launcher enables NoRA by default, using the same PEFT fork and FP32 normalization
as AntiDoom. Install the pinned fork with the command in "Antidoom FTPO with NoRA"
after preparing the RWKV environment, then warm the exact interpreter's native cache:

```bash
MAX_JOBS=2 bash examples/train/rwkv/run_rwkv_on_policy_distill.sh --prepare-flashrwkv2 \
  > /tmp/results_rwkv_opd_prepare_flashrwkv2.log 2>&1
```

The launcher keeps Ray workers in the prepared project environment and shares the
canonical `python3` identity with torchrun so worktrees reuse the native cache.
NoRA is selected through
`trainer.policy.model.lora.use_nora=true`; ordinary training defaults elsewhere remain
unchanged. To run full-parameter OPD, set `USE_NORA=false LORA_RANK=0` and choose its
learning rate explicitly with `LR`.

NoRA uses rank 256, alpha 256, zero dropout, and the same attention and output-head
targets as AntiDoom: `r_proj`, `k_proj`, `v_proj`, `o_proj`, and `lm_head`.
The pinned Transformers-RWKV passes the raw A/B parameters and `use_nora` into
FlashRWKV2's fused BF16 projection forward/backward operators. No Python adapter
forward is inserted around those fused projections.

With policy `fsdp_size=1`, every GPU keeps a complete frozen base and synchronizes
adapter gradients once per optimizer step. The teacher shards its parameters across
all eight GPUs to reduce host storage during colocation offload.
Weight sync computes the normalized update and adds it to each base projection in
FP32, then streams complete FP16 weights through the native transfer engine.
It leaves the live training parameters untouched and resets recurrent inference
caches through the existing weight-sync lifecycle. Resumable checkpoints retain
the adapter and optimizer; periodic HF exports contain complete merged weights.

Both models keep full-sequence recurrent forwards (`trainer.rwkv_chunk_size=0`).

`examples.train.rwkv.profile_opd` measures the maximum rank time and peak allocation
with full-length synthetic sequences, the native FSDP strategy, and the fused RWKV
operators. Run policy and teacher separately so their working memory does not overlap:

```bash
XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv" \
uv run --isolated --no-config --no-project .venv/bin/torchrun \
  --standalone --nproc-per-node=8 -m examples.train.rwkv.profile_opd \
  --role policy --model "$STUDENT_MODEL" --micro-batch-sizes 1 2 4 \
  > /tmp/results_rwkv_opd_profile_policy.log 2>&1
```

Use `--role ref --model "$TEACHER_MODEL"` for teacher scoring. These measurements
report component throughput; they do not include rollout, colocation offload, or
weight-transfer time.

### Eight-card sizing

Full-context component measurements on eight 96-GB RTX PRO 6000 GPUs use 25600-token
sequences, rank-256 NoRA with the attention/output-head targets above, and the pinned
FlashRWKV2 fused operators. Policy measurements include backward and AdamW; teacher
measurements run without gradients. Throughput is aggregated over all eight ranks;
memory is the maximum PyTorch allocation across ranks.

| Component | Per-GPU micro-batch | Tokens/s | Peak GiB |
| --- | ---: | ---: | ---: |
| NoRA student | 1 | 73,321 | 19.4 |
| NoRA student | 2 | 103,666 | 25.7 |
| NoRA student | 4 | 120,995 | 44.1 |
| NoRA student | 8 | 128,794 | 80.8 |
| Frozen teacher | 1 | 53,585 | 55.8 |
| Frozen teacher | 2 | 57,956 | 62.0 |
| Frozen teacher | 4 | 60,304 | 71.4 |
| Frozen teacher, FSDP 8 | 4 | 60,950 | 50.1 |
| Frozen teacher, FSDP 8 | 8 | 60,455 | 71.6 |

These measurements select training micro-batch 8, forward micro-batch 4, and teacher
FSDP size 8. Policy
profiling enables expandable activation segments after model initialization, matching
the production worker. The complete local corpus contains 85994 questions; the native
17408-token prompt filter retains 72321 (Math 32412, Code 23665, Long_Context 4372,
Knowledge 11872). The production run can point `TRAIN_FILE` at the resulting
`train-length17408.parquet` to reuse this filtering step.

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
`examples/train/rwkv/rollout.py` uses the same `gsm8k.utils.compute_strict_score` for its Math domain
and requests generated token IDs to verify the actual final EOS token. Other domains and non-strict
GSM8K rewards retain their existing scoring.
Use `environment.env_class=gsm8k` and `environment.skyrl_gym.gsm8k.strict_reward=true` to enable
strict rewards for dataset rows labelled `gsm8k` in training and evaluation, including the fully
asynchronous entry point. Rows labelled with other environments retain their own routing;
GSM8K rewards remain unchanged when strict mode is disabled. The shared parquet files do not
need to be rewritten. Historical answer-match scores are not directly comparable to strict rewards.
For the asynchronous pipeline, use the existing
`examples.train.fully_async.main_fully_async` entrypoint, set `trainer.fully_async.enabled=true` and
`generator.batched=false`; also pass `environment.skyrl_gym.gsm8k.strict_reward=true`.
The launcher sets `trainer.fully_async.clear_kv_cache_on_weight_sync=true`: each weight update
resets cached prefixes and running recurrent states before generation resumes. RWKV fully-async
training requires this setting so cached states are recomputed with the updated weights.
The default gate is `3e-3`; monitor
`policy/loss_metrics/flashreinforce/acceptance_rate` and the rollout/trainer logprob-difference
metrics before tuning it.

Before a fresh training run, verify the model's `bot + open_think` template and score a fresh
rollout batch with the native strict GSM8K environment. Check `environment/structural_format_valid`,
`environment/ended_eod`, `environment/truncated`, and `environment/strict_reward` alongside
the positive-reward count; an all-zero batch provides no centered reward signal for FlashREINFORCE.
Keep the training and evaluation sampling recipes above, batch size, optimizer, Score-Centering,
and admission gate fixed when comparing the corrected reward/cache baseline.

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
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
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

## Direct Nemotron CoT SFT

Use `prepare_sft.py` to train directly on the original Nemotron expert traces.
It preserves `reasoning_content` and the final `content` in the RWKV Bot
`<think>...</think>` format, supervises the response plus `✿` and EOS, and
assigns every trajectory of the same original problem to one split. The seed-0
95%/5% split contains 271,385 training and 14,131 validation trajectories in
the local Nemotron-SFT-Math-v4 snapshot. No sequence truncation is applied.

```bash
MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"
DATA_DIR="$HOME/data/rwkv-direct-sft"

uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.prepare_sft \
  --input "$HOME/Datasets/Nemotron-SFT-Math-v4" \
  --model "$MODEL_DIR" --output "$DATA_DIR" --num-workers 8 \
  > /tmp/results_direct_sft_prepare.log 2>&1

MODEL_DIR="$MODEL_DIR" DATA_DIR="$DATA_DIR" NUM_GPUS=8 \
  BATCH_SIZE=56 MICRO_BATCH_SIZE=7 NCCL_SHM_DISABLE=0 \
  bash examples/train/rwkv/run_rwkv_sft.sh fsdp_config.fsdp_size=1 \
  > /tmp/results_direct_sft.log 2>&1
```

The launcher trains one epoch with native FSDP and complete cross-chunk
gradients (`rwkv_chunk_size=2048`, `wkv_mode=fp32io16`). Similar-length batches
are shuffled using a checkpointable sampler; each source row is visited once.
AdamW uses peak learning rate `1e-6`, cosine decay to `1e-7`, and 5% warmup
based on the actual epoch step count. The decay floor is controlled by
`MIN_LR_RATE` (default `0.1`, a fraction of the peak learning rate); warmup
still starts at zero. The native FSDP scheduler is `cosine_with_min_lr` with
`optimizer_config.scheduler_specific_kwargs={min_lr_rate: 0.1}`. On resume,
the restored optimizer uses its saved learning rate for the first update;
the next scheduler advance applies the configured floor at the restored step.
Every 50 steps it saves resumable checkpoints, exports HF
weights and logs held-out SFT loss to W&B under `rwkv_direct_sft`. Periodic
evaluation uses a fixed 512-row panel selected by UUID hash from the validation
partition; the complete validation split remains in `DATA_DIR/validation`.
SFT loss measures teacher-forced fit, rather than free-generation math accuracy.
`NUM_GPUS`, `BATCH_SIZE`, `MICRO_BATCH_SIZE`, `CHUNK_SIZE`, `LEARNING_RATE`, `MIN_LR_RATE`, `RUN_NAME` and
`OUTPUT_ROOT` are environment overrides. Pass `resume_from=latest` to resume
from the same run directory. `--limit 128` prepares a separate small smoke dataset.

On eight PRO6000 GPUs, `fsdp_config.fsdp_size=1` uses native FSDP's replicated
data-parallel mesh, keeping model and Adam state on each GPU. This improves
throughput for the 1.5B model while complete 140k-token CoT fits with microbatch
7 and chunk 2048. Keep global batch 56 when continuing the seven-GPU run so
the sampler order, epoch length (4,847 steps), and scheduler stay consistent.
The checkpoint loader restores weights and Adam state across GPU counts using
PyTorch's distributed state-dict APIs. GPU-count changes alter rank-local RNG
streams and floating-point reduction order.

A same-data benchmark on 56 rows of 12k–14k-token traces, with unchanged
weights and learning rate zero, produced the following means over two steps
after excluding initialization:

| GPUs | FSDP size | Microbatch | NCCL SHM | Seconds/step | Tokens/s |
| --- | --- | --- | --- | --- | --- |
| 7 | 7 | 4 | disabled | 23.54 | 30,321 |
| 8 | 8 | 7 | enabled | 14.16 | 50,409 |
| 8 | 1 | 7 | enabled | 13.35 | 53,440 |

Every 500 training steps and at the final pass, evaluate the same HF
checkpoint with `rwkv-rs/lighteval-rwkv` on the full `aime24`, `aime25`,
`aimo_progress_prize_1` and `math_500` tasks. Use 16 generated responses per
question for AIME24, AIME25 and AIMO, and four for Math500. Report avg@16 and
avg@4 respectively: mean correctness across the responses, then across
questions. Configure the producer explicitly; its current default sampling
policy does not guarantee these response counts.

Use CoT with the `bot` template and `open_think`, a 24,576-token completion
budget, temperature 0.96, top-p 0.76, top-k 32, presence penalty 1.0,
frequency penalty 0.1 and penalty decay 0.988. Save per-rollout results,
question and completion counts, and truncation rates alongside aggregate
scores. Invoke the existing producer checkout with its dedicated environment;
SkyRL owns the final export and inference service. Set `LIGHTEVAL_PROJECT` to
the producer checkout and `WANDB_RUN_PATH` to `entity/project/run_id` to enable
periodic benchmarks. SFT loss evaluation and weight saving remain every
50 steps. The launcher checks `BENCHMARK_INTERVAL` (default 500) before
starting inference; other loss evaluations return no benchmark metrics.
The native `eval_command` hook waits for generation and scoring, then merges
the locally saved metrics with eval loss at the same training step. W&B
uploads run in a separate CPU process after scoring and do not delay the
next training step. A file lock serializes uploads to each evaluation run.
`EVAL_ONLY=1 EVAL_STEP=<step> BENCHMARK_INTERVAL=1` evaluates an existing
export independently.

The original model is evaluated once. The first round allocates four inference
replicas to each model; subsequent rounds use all eight replicas for SFT.
Each replica supports 1,024 concurrent sequences. The training workers remain
resident while their unused CUDA allocator memory is released. Inference uses
65% of each GPU and terminates before training continues. Scores are logged
to the training run and to evaluation curves keyed by the checkpoint step.
Each checkpoint has a separate output directory and W&B result artifact.
This workflow only uploads to W&B.

With RWKV and `fsdp_size=1`, save one complete model and Adam state on rank 0
and retain each rank's scheduler and RNG state. The eight-replica experiment
writes about 17.07 GiB per checkpoint instead of 136.57 GiB. Restore the full
state through PyTorch's distributed state-dict APIs. Older checkpoints retain
their per-rank loading path.

Each phase records background upload progress in `upload_status.json` and
`upload.log` alongside `eval_metrics.json`. Upload failures leave the saved
scores and completions intact and do not fail training. To retry an upload
without running inference, set `OUTPUT_ROOT`, `WANDB_RUN_PATH`,
`UPLOAD_ONLY=1 UPLOAD_PHASE=sft EVAL_STEP=<step>` and invoke the same launcher.
For the original model, use `UPLOAD_PHASE=base` and its evaluated step.

For the final AIME24, AIME25 and AIMO avg@256 evaluation, use
`EVAL_ONLY=1 EVAL_STEP=4847 EVAL_PHASES=sft EVAL_AVG_K=256`,
`EVAL_TASKS="aime24 aime25 aimo_progress_prize_1"` and
`EVAL_ROOT="$OUTPUT_ROOT/evaluation/lighteval_avg256"`. This generates
7,680, 7,680 and 2,560 completions respectively. The separate output root
preserves the periodic avg@16 and Math500 avg@4 results; scores and artifacts
are labeled avg@256 in the same W&B experiment. Upload-only retries must
use the same `EVAL_ROOT`.

## Antidoom FTPO with NoRA

`antidoom.py` adapts the repetition detector and final-token preference loss from
[Liquid's TRL notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/antidoom_with_trl.ipynb)
and [Antidoom](https://github.com/Liquid4All/antidoom) (Apache-2.0). It generates
completions with the selected RWKV checkpoint, extracts preferences at the first
repeated token, and trains with SkyRL's native `FSDPStrategy`.

The experiment combines **complete benchmarks selected by G1k truncation rate**
on [the RWKV scoreboard](https://eval.rwkv.rs/scoreboard) with untested sources
from [LiquidAI/antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0).
The benchmark selection uses active G1k 1.5B scores with truncation rate at least 50%:

| Benchmark | Recorded questions | Truncation rate | Response budget |
| --- | ---: | ---: | ---: |
| `aime25` | 30 | 90.00% | 24576 |
| `aimo_progress_prize_1` | 10 | 90.00% | 24576 |
| `gpqa:diamond` | 198 | 87.88% | 24576 |
| `aime24` | 30 | 83.33% | 24576 |
| `gpqa:main` | 448 | 82.14% | 24576 |
| `olympiad_bench` | 2673 | 78.15% | 24576 |
| `lcb:codegeneration` | 268 | 77.24% | 24576 |
| `ifbench_multiturn` | 1387 | 53.64% | 4096 |
| `ifbench_test` | 300 | 52.33% | 4096 |

The recorded total is 5,344 benchmark rows; GPQA main and diamond overlap.
These are question counts, not counts of loops or FTPO examples. The loader
exports every native task Doc and records its actual count, which can differ
from an older scoreboard dataset revision. `--param-size` and
`--min-truncation-rate` change the benchmark selection.

The mix supplement selects entire component sources with no recorded scoreboard
evaluation, considering scores across all models, including expired records.
GSM8K, MMLU, MATH and
[MetaMathQA's GSM8K/MATH derivatives](https://huggingface.co/datasets/meta-math/MetaMathQA)
are explicitly excluded. MathQA and CommonsenseQA are excluded by their existing
scoreboard evaluations. The current untested components are:

| Mix source ID | Original mix rows | Response budget |
| --- | ---: | ---: |
| `pubmedqa_artificial_train` | 50000 | 4096 |
| `apps_train` | 5000 | 24576 |
| `open_perfectblend_ultrainteract` | 49540 | 24576 |
| `open_perfectblend_ultrachat200k` | 49958 | 4096 |
| `open_perfectblend_evol_codealpaca` | 49981 | 24576 |
| `open_perfectblend_autoif` | 49995 | 4096 |
| `open_perfectblend_lmsys_arena` | 49206 | 4096 |
| `ifstruct_train_generated` | 20000 | 4096 |

These components contain 323,680 candidate prompts before deduplication.
The mix contains training prompts; adding it does not export the full upstream
benchmark test splits. `export-mix` keeps every benchmark row, then appends
unique mix prompts, deduplicating against the benchmarks and earlier mix rows.
It preserves component source, split, sample ID, license, and the pinned mix
revision in the output. Its manifest records exclusions and actual counts.
With the current benchmark export and pinned mix revision, 38,986 repeated
prompts are removed and 284,694 mix prompts are appended. The combined input
contains 290,038 rows, including all 5,344 original benchmark rows.

Export active score metadata from `scoreboard_production` on `rwkv-rs-server`
with a read-only transaction. The query reads models, tasks, sampling budgets,
and aggregate scores:

```bash
ssh rwkv-rs-server \
  'sudo -n -u postgres psql -X -q -A -t -v ON_ERROR_STOP=1 -d scoreboard_production' \
  > /tmp/antidoom_g1k_scores.jsonl 2> /tmp/results_antidoom_scores.log <<'SQL'
BEGIN READ ONLY;
SET LOCAL statement_timeout = '15s';
SELECT json_build_object(
  'model', row_to_json(m), 'score_id', s.id,
  'benchmark', b.name, 'field', b.field, 'framework', b.framework,
  'num_samples', b.num_samples, 'avg_k', b.avg_k,
  'cot_mode', s.cot_mode, 'score', s.score,
  'truncation_rate', s.truncation_rate, 'sampling', row_to_json(c))
FROM scores s
JOIN rwkv_models m ON m.id = s.model_id
JOIN benchmarks b ON b.id = s.benchmark_id
JOIN sampling_configs c ON c.id = s.sampling_config_id
WHERE lower(m.data_version) = 'g1k' AND NOT s.is_expired;
ROLLBACK;
SQL
```

`export-benchmarks` reads the producer's
[native RWKV task registry](https://github.com/rwkv-rs/lighteval-rwkv/tree/c5475e27a1e58184c9a6cdbd30572ec1ae601dd7)
using **this project's interpreter**. Keep that source checkout at the stated
revision. Install its dataset-loader dependencies in the SkyRL `.venv`, with
existing package versions constrained:

```bash
uv run --isolated --no-config --no-project uv pip freeze \
  --python .venv/bin/python --exclude-editable \
  > /tmp/antidoom_existing_constraints.txt 2> /tmp/results_antidoom_constraints.log
rg '^(antlr4-python3-runtime|latex2sympy2-extended)==' \
  /tmp/antidoom_existing_constraints.txt > /tmp/antidoom_math_constraints.txt
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --constraint /tmp/antidoom_existing_constraints.txt \
  'inspect-ai>=0.3.140' 'GitPython>=3.1.41' 'termcolor==2.3.0' \
  pytablewriter colorlog 'aenum==3.1.15' 'nltk==3.10.3' scikit-learn \
  sacrebleu 'rouge-score==0.1.2' sentencepiece pycountry langcodes jsonlines \
  aiolimiter diskcache more-itertools \
  > /tmp/results_antidoom_export_install.log 2>&1
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --no-deps \
  'lighteval @ git+https://github.com/rwkv-rs/lighteval-rwkv@c5475e27a1e58184c9a6cdbd30572ec1ae601dd7' \
  > /tmp/results_antidoom_native_export_install.log 2>&1
# Use the native registry's fixed math-parser versions while exporting.
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --no-deps \
  'latex2sympy2-extended==1.0.6' 'antlr4-python3-runtime==4.13.2' \
  > /tmp/results_antidoom_export_math_install.log 2>&1

DATA_ROOT="$HOME/skyrl-rwkv-runs/antidoom-g1k-whole-benchmarks"
LIGHTEVAL_PROJECT="$HOME/Projects/MachineLearning/lighteval-rwkv"
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.antidoom select-benchmarks \
  --scoreboard-records /tmp/antidoom_g1k_scores.jsonl \
  --param-size 1.5b --min-truncation-rate 0.5 --output "$DATA_ROOT/selection" \
  > /tmp/results_antidoom_selection.log 2>&1
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.antidoom export-benchmarks \
  --benchmarks "$DATA_ROOT/selection/benchmarks.json" \
  --lighteval-project "$LIGHTEVAL_PROJECT" --output "$DATA_ROOT/questions" \
  > /tmp/results_antidoom_questions.log 2>&1

# Restore the original SkyRL math-parser versions before rollout/training.
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --no-deps --requirements /tmp/antidoom_math_constraints.txt \
  > /tmp/results_antidoom_export_restore.log 2>&1
```

The exporter calls `task.get_docs(None)` for every selected task, preserving
native dataset splits, filters, question formatting, and source metadata.
It loads original datasets from Hugging Face and writes question-only JSONL.
No evaluation pipeline or response-cache reader runs. The raw dataset download
cache is independent of model-generated evaluation answers. GPQA access requires
the dataset's Hugging Face authorization.

Fetch the set of benchmarks ever evaluated, then append the untested mix
components. This step uses the standard SkyRL dependencies:

```bash
ssh rwkv-rs-server \
  'sudo -n -u postgres psql -X -q -A -t -v ON_ERROR_STOP=1 -d scoreboard_production' \
  > /tmp/antidoom_tested_benchmarks.jsonl 2> /tmp/results_antidoom_tested_benchmarks.log <<'SQL'
BEGIN READ ONLY;
SET LOCAL statement_timeout = '15s';
SELECT json_build_object('benchmark', b.name)
FROM benchmarks b
WHERE EXISTS (SELECT 1 FROM scores s WHERE s.benchmark_id = b.id)
ORDER BY b.name;
ROLLBACK;
SQL
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.antidoom export-mix \
  --benchmark-data "$DATA_ROOT/questions" \
  --tested-benchmarks /tmp/antidoom_tested_benchmarks.jsonl \
  --output "$DATA_ROOT/combined" \
  > /tmp/results_antidoom_combined_questions.log 2>&1
```

The mix revision defaults to `79ff2d479c5e59ea9973a66a7ca9ce8b703f9c7b`.
Every added row contains only its original prompt and source metadata. All
model responses are generated with the selected RWKV checkpoint in the next step.

Install [Joluck/NoRA](https://github.com/Joluck/NoRA)'s PEFT fork in the project's
existing RWKV `.venv`. Its fixed revision is
`4669c35ce0ffd027456272ee5050c0f58ac8d22a`. This replaces PEFT with
`0.19.2.dev0` without changing the other installed packages. The project lock
continues to pin the standard PEFT dependency. Run this example through the
explicit project interpreter so a project sync does not replace NoRA.

```bash
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --no-deps \
  'peft @ git+https://github.com/Joluck/NoRA@4669c35ce0ffd027456272ee5050c0f58ac8d22a#subdirectory=peft' \
  > /tmp/results_antidoom_install.log 2>&1

# Download the original Transformers checkpoint from the official RWKV repo.
uv run --isolated --no-config --no-project .venv/bin/hf download \
  rwkv-rs/rwkv7-g1-st --revision b672701cccf4bddcc66f65c8c9a468b776c9836a \
  --include 'rwkv7-g1k-13.3b-20260930-ctx25600/*' \
  --local-dir "$HOME/Weights/RWKV/hf" --max-workers 4 \
  > /tmp/results_antidoom_model_download.log 2>&1

MODEL_DIR="$HOME/Weights/RWKV/hf/rwkv7-g1k-13.3b-20260930-ctx25600"
OUTPUT_ROOT="$HOME/skyrl-rwkv-runs/$(basename "$MODEL_DIR")-antidoom-nora"

# On rwkv-sha-pro6000x8, reroll every question with the original G1k 13.3B model.
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.antidoom generate \
  --model "$MODEL_DIR" --input "$DATA_ROOT/combined/prompts.jsonl" \
  --output "$OUTPUT_ROOT/rollout" --limit 0 --max-model-len 32768 \
  --concurrency 4096 --max-num-seqs 256 --gpu-memory-utilization 0.9 \
  --shard-rollouts --timeout 7200 \
  > /tmp/results_antidoom_rollout.log 2>&1

# Use exact rollout IDs and the same checkpoint to choose alternative tokens.
uv run --isolated --no-config --no-project .venv/bin/torchrun \
  --standalone --nproc-per-node=8 \
  -m examples.train.rwkv.antidoom prepare \
  --model "$MODEL_DIR" --rollouts "$OUTPUT_ROOT/rollout" \
  --output "$OUTPUT_ROOT/data" \
  > /tmp/results_antidoom_prepare.log 2>&1

# Native FSDP2; replicated frozen weights, token-bounded microbatches, one epoch.
PYTHON_EXEC="$PWD/.venv/bin/python" \
XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv" \
TORCH_EXTENSIONS_DIR="$HOME/.cache/skyrl-rwkv/torch_extensions" \
uv run --isolated --no-config --no-project --env-file .env .venv/bin/torchrun \
  --standalone --nproc-per-node=8 -m examples.train.rwkv.antidoom train \
  --model "$MODEL_DIR" --data "$OUTPUT_ROOT/data" --output "$OUTPUT_ROOT/train" \
  --learning-rate 5e-7 --epochs 1 --fsdp-size 1 --micro-batch-size 8 --max-batch-tokens 32768 \
  --gradient-accumulation-steps 2 --eval-max-samples 1024 \
  > /tmp/results_antidoom_train.log 2>&1
```

Generation starts fresh TP1 vLLM engines with the explicitly selected
checkpoint and requires a new output directory. All questions are generated
with temperature 1, top-p 1, top-k -1, and disabled penalties. Source metadata
selects the checkpoint's `bot` + `open_think` template for mathematics, coding,
and GPQA, and `assistant` + `open_think` for IFBench. Each source keeps its
recorded response budget (24576 or 4096); `--max-tokens` can reduce the budget.
If the prompt consumes some of the context budget, the shortened generation
budget is recorded. A prompt exceeding the context limit stops the run.
Use distinct output roots for different checkpoints or fresh repeats.
Requests are continuously replenished as completions finish. `--shard-rollouts`
writes one trajectory file per engine, allowing `prepare` to run on all eight
GPUs without rereading every trajectory on every rank. Progress includes actual
generated tokens and throughput.
`--resume` continues a partially completed run from this same checkpoint,
skipping its saved prompt indices. This uses the current experiment's own new
rollouts. Resume accepts one completion per prompt.

For explicit alternative prompt sources, `generate --input` also accepts
OpenAI `messages`, ShareGPT `conversations`, or exact `prompt_ids`; `--dataset`
accepts an explicitly named Hugging Face prompt dataset. `--base-url` can target
an inference service already serving the same checkpoint.

FTPO contexts and rejected IDs are slices of the original token sequence;
decoded text is used only to locate repetition. Up to 20 non-whitespace
alternatives come from the rollout model's top 50 logits. The detector selects
actual repetition in the newly generated responses; truncation by itself
does not create a preference example. Contexts longer than `--max-length`
(32768 by default) are dropped as complete rows. A hash split assigns every
sibling completion and every GPQA variant of the same original question to
the same partition. Training
requires non-empty train and validation partitions; inspect `data/manifest.json`
and increase prompt coverage if too few loops were found.

For a text-only evaluation cache, `prepare --cached-text` accepts JSONL rows
with the server-rendered `prompt_text`, raw `response_text`, `repeat_hit`
(`start`, `end`, `period`, `repeats`, `snippet`), `source` metadata, and
`cached_text_provenance` containing `response_sha256` and the cache origin.
Both texts must round-trip through the training tokenizer. Supplied boundaries
must contain at least four consecutive exact repeats; long-period loops are
supported without regenerating responses. These are reconstructed token
sequences, recorded as `retokenized_evaluation_cache` in the data manifest.
The cached decoding parameters and model alias remain in the provenance.
FTPO alternatives are recomputed from the selected training checkpoint.

Data construction follows the full
[Antidoom reference](https://github.com/Liquid4All/antidoom/tree/bd6a126476e18554b0cacaea3fd9f258fdde1f97/src/antidoom),
including token-aware loop-boundary refinement, ascending-probability candidate
selection within top-50 and `--chosen-min-p 0.01`, rejected-token frequency
normalisation (strength 0.3), and chosen-surface deduplication/pruning (strength
0.5; reference count is the larger of 50 and the 95th percentile). RWKV-specific
filtering excludes EOS and the `✿` protocol delimiter from both preference
targets. Incomplete UTF-8 byte tokens retain their distinct token IDs.
The notebook's simplified from-scratch, highest-probability-first extraction omits these
reference pipeline steps.

By default `prepare` retains 70% of the training pool so rejected-token
normalisation can downsample frequent targets; `--max-train-examples` overrides
that limit. The held-out problem split and generation prompts are fixed before
training-row regularisation. Validation rows are not frequency-downsampled.
The manifest records the reference revision and construction parameters.
Rebuild existing FTPO data with `prepare`; `train` rejects preference targets
containing EOS or RWKV protocol tokens.

NoRA uses `use_nora=True`, rank 8, alpha equal to rank, zero dropout, and the
RWKV attention projections `r_proj`, `k_proj`, `v_proj`, `o_proj`, and the
`lm_head` output projection. `--target-modules` can select the adaptation scope.
BF16 training passes the original PEFT A/B parameters and scaling to fused
FlashRWKV2 projection operators, including NoRA column normalization and its
backward. WKV recurrence retains its existing implementation. ChannelMix
`key` and `value` can also be selected with `--target-modules`; the default
five targets above preserve this example's training recipe. Each sample's
hidden state is selected before the output-head projection, so a batch of B
samples produces B vocabulary rows. PEFT continues to manage disabling,
saving, loading, and merging adapters. Native FP16 inference absorbs Head
and any ChannelMix adapters before the full-model merge probe. Each FTPO
sample scores a single next-token position. The loss keeps the notebook's
margin 2, non-target logit MSE coefficient 0.4, target coefficient 0.05 and
target tolerance 0.5. Reference logits come from the frozen base with the
adapter disabled. Batched MSE terms are averaged across the masked token
positions, matching the reference even after chosen counts vary per example.
AdamW's default LR is `5e-7`, a conservative starting point for the normalized
NoRA update. The reference notebook's `1.5e-5` uses ordinary LoRA and is not a
scale-equivalent NoRA setting. Weight decay is 0.01, with a linear schedule
and gradient clipping 1.0. Set `--rank`, `--learning-rate`, `--epochs` and
`--gradient-accumulation-steps` explicitly for other experiments. `--lora-alpha`
can override NoRA's default alpha equal to rank. For a complete training cycle,
use `--max-steps 0` so the learning-rate schedule spans all requested epochs.
`--warmup-ratio` optionally reserves the initial fraction of optimizer steps
for linear warmup (default 0).

Training buckets FTPO contexts by length, right-pads each microbatch, and selects
the final original token position of each row. Trailing padding does not enter
the selected causal prefixes. `--max-batch-tokens` bounds padded tokens per GPU;
`--micro-batch-size` caps examples per forward. Length groups are shuffled as
whole optimizer-step batches, keeping all ranks at similar sequence lengths.
Gradients synchronize at the end of accumulation. With `--fsdp-size 1`, every
GPU retains the frozen model and only adapter gradients are synchronized. Use
`--profile --max-steps 3 --report-to none` to measure real step time, token
throughput, and the maximum allocation across all ranks before a production
run. Profiling visits the longest context groups first.

For `rwkv7-g1k-13.3b-20260930-ctx25600` on eight 96-GB RTX PRO 6000 GPUs,
the resource-benchmarked production configuration uses `--rank 256 --lora-alpha 256`,
`--learning-rate 2e-6 --warmup-ratio 0.05 --epochs 3 --max-steps 0`,
`--micro-batch-size 16 --max-batch-tokens 32768`, and
`--gradient-accumulation-steps 1 --fsdp-size 1`. Keep the output head and
all four attention projections enabled. On the prepared 748-example FTPO
training split this completes 93 optimizer steps, with peak allocation about
85 GiB per GPU and approximately 7.1k useful context tokens/s across eight GPUs.

Every 50 optimizer steps and at each epoch boundary, including the final step, training writes adapter weights,
per-rank optimizer/scheduler/RNG states, and held-out FTPO loss on a fixed
`--eval-max-samples` subset. The frozen base is identified by the exact model
path instead of being copied into every checkpoint. W&B receives training and
evaluation curves under `rwkv_antidoom`, plus training and merge metrics artifacts.
Training logs include cumulative examples, completed epoch fraction and the
learning rate actually used by each optimizer update. Preference data and
generation text remain in the experiment output directory.
Use `--report-to none` for a local smoke run. After training, the final NoRA
adapter is automatically absorbed with `merge_and_unload(safe_merge=True)`.
`train/merged` contains full HF weights, tokenizer, generation configurations
and `merge_metrics.json`. NoRA's column normalization is computed in FP32,
then the updates are absorbed in FP32 and the complete model is exported in
FP16 for vLLM inference. The merge checks the FP16 adapter and absorbed model's
next-token log probabilities and records their KL divergence. The output head
is absorbed before the native RWKV inference probe, which reads its weight
directly; a separate FP32 check verifies the head's adapter/merged projection.
This preserves
updates that would be lost when added directly to BF16 base weights.

For a held-out before/after doom-loop comparison, run `generate` on
`data/validation_prompts.jsonl` once with `MODEL_DIR` and once with
`$OUTPUT_ROOT/train/merged`, using distinct output directories and the same
seed and token budget. Set `--sampling eval` for both runs: temperature 0.96,
top-p 0.76, top-k 32, presence penalty 1.0, frequency penalty 0.1, and decay
0.988. `doom_loop_rate` in `summary.json` measures consecutive exact text-block
repetition only. Compare it with `truncated` and answer quality, and inspect
truncated completions for paraphrased reasoning loops, repeated draft/check
cycles and task drift. Zero exact repetitions does not establish zero semantic
repetition. Held-out FTPO loss measures preference fit; free generation and
completion review supply the before/after behavioral evaluation.

For paired answer-quality evaluation, use the same ordered held-out input file
and seed for both checkpoints. Each saved response records its actual seed
(`seed + prompt_index`); verify prompt IDs, seeds and response budgets agree
before comparing results. `score` accepts those responses and a JSONL containing
`prompt_ids`, the captured `prompt_text`, `source`, and native LightEval `doc`
metadata. GPQA gold letters must follow the choices in the captured prompt.
The scorer reads the producer's native mathematical comparator, GPQA extractor,
LiveCodeBench test harness and IFBench constraint checkers without modifying
the producer project. It reports strict correctness separately from native
GPQA missing-answer expectation, alongside extraction, truncation and exact-loop
rates. Long-period loop scoring defaults to period 8192 and sampling interval 16.

The native mathematical scorer uses ANTLR 4.13, while SkyRL's configuration
parser uses ANTLR 4.9. Keep the scorer's two compatibility packages in an import
directory inside the project's dedicated environment:

```bash
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --target .venv/antidoom_eval --no-deps \
  'antlr4-python3-runtime==4.13.2' 'latex2sympy2_extended==1.0.6' \
  > /tmp/results_antidoom_eval_dependencies.log 2>&1

PYTHONPATH="$PWD/.venv/antidoom_eval" \
uv run --isolated --no-config --no-project .venv/bin/python \
  -m examples.train.rwkv.antidoom score \
  --input "$OUTPUT_ROOT/evaluation_inputs.jsonl" \
  --rollouts "$OUTPUT_ROOT/eval_base" --output "$OUTPUT_ROOT/scores_base" \
  --lighteval-project "$HOME/Projects/MachineLearning/lighteval-rwkv" \
  > /tmp/results_antidoom_eval_score.log 2>&1
```

IFBench also needs its native `spacy`, `syllapy`, `emoji` and `langdetect`
dependencies, the compatible `en_core_web_sm` model, and NLTK resources.
Install them with `uv` in this same project environment before scoring.

Restore the locked PEFT package after the NoRA experiment:

```bash
uv run --isolated --no-config --no-project uv pip install \
  --python .venv/bin/python --no-deps 'peft==0.18.1' \
  > /tmp/results_antidoom_restore_peft.log 2>&1
```
