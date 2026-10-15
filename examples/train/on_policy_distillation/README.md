# On-Policy Distillation

This example trains models via On-Policy Distillation (following the
[Thinking Machines on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) blog post).
In on-policy distillation (OPD), a student trains on its own rollouts while a frozen teacher grades every
token: the advantage is the per-token reverse KL, `log π_teacher − log π_student`.

For RWKV7 G1k-13.3B to G1k-1.5B distillation using this same entry point, see the
[RWKV recipe](../rwkv/README.md#on-policy-distillation).

## Quickstart

```bash
# Qwen3-4B-Base student, DAPO-17k prompts, AIME24 eval. The job launches the teacher
# (Qwen3-32B, TP=4) on 4 GPUs of an 8-GPU node, next to the student's 4.
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh
```

## Teacher backends

`trainer.teacher.backend` picks where the teacher runs. Each backend reads only its own fields.

**`skyrl` (default).** The job launches the teacher's vLLM deployment on its own GPUs, on top of the
student's, and drives it like the student's engines. Set `max_model_len` to cover prompt + response + 1;
unset, vLLM uses the model's native context.

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.inference_engine.num_engines=1 \
  trainer.teacher.inference_engine.tensor_parallel_size=4 \
  trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=10241 \
  ...
```

**`vllm`.** One vLLM endpoint you run per teacher model: a single server, or a router in front of several.
It is scored through `/v1/completions` with `prompt_logprobs`, and its `max_model_len` must cover
prompt + response + 1.

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.teacher.backend=vllm \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.server_url=http://teacher-host:8000 \
  ...
```

## Configuration

Every OPD field is documented in `skyrl/train/opd/config.py`.

## Metrics

- `opd/reverse_kl`: approaches 0 as the student matches the teacher.
- `opd/reverse_kl_abs_max`: the largest per-token reverse KL in the batch.
- `opd/adv_rl_abs_mean`, `opd/adv_opd_abs_mean`: the scale of the reward term and the teacher term, for
  tuning `kl_coef` when `trainer.algorithm.opd.use_task_reward=true`.
- `opd/teacher_time_exposed`: seconds the step waited on the teacher after the last rollout came back.
- `opd/teacher_time_per_group_mean`: mean time to score one prompt's group.

## Caveats

- The student and the teacher must share a tokenizer: the teacher scores the student's token ids verbatim.
- With `use_task_reward=true`, a z-scored GRPO advantage (order 1 per token) is added to a log-probability
  difference (0.01–1 nats per token). `kl_coef` trades them off.
- A failed teacher request fails the batch after the client's retries.
- Step-wise trajectories and top-k distillation losses are not supported yet.
## Optional STOPD

[STOPD](https://github.com/YouAreSpecialToMe/STOPD) samples an independent token
budget for each training rollout, rounded up to a block boundary. Retained
response tokens receive inverse-probability weights. The weighted loss uses a
fixed denominator from the previous update's estimated full token count; it
does not divide by the current, randomly truncated token count. Evaluation
continues to generate complete responses.

STOPD is disabled by default. Enable it through the same OPD entry point:

```bash
bash examples/train/rwkv/run_rwkv_on_policy_distill.sh \
  trainer.algorithm.stopd.enabled=true \
  trainer.algorithm.stopd.block=128 \
  trainer.algorithm.stopd.warmup_steps=25
```

Warmup counts completed global updates, including updates restored from a
checkpoint. The supported configuration is synchronous, single-turn OPD with
one response per prompt and one optimizer update per rollout batch. Sampled
prefixes keep their actual tokens, logprobs, and action masks; no terminal token
is appended at a sampled cutoff. Additional advantage normalization, entropy
loss, KL loss, and overlong filtering must be disabled.

Native checkpoints retain model, optimizer, scheduler, RNG, and dataloader state.
With STOPD enabled they also contain `stopd_state.pt`, including the step,
lagged denominator, budget seed, and truncation configuration. Resume one with
`trainer.resume_mode=from_path trainer.resume_path=<global_step_directory>`.
When importing a Vanilla OPD checkpoint after warmup, set
`trainer.algorithm.stopd.initial_denominator` to the preceding update's full
response-token count (the sum of its dumped rollout `loss_mask` values).

To compare both modes from the same complete checkpoint and data position:

```bash
uv run --isolated --no-project .venv/bin/python3 \
  examples/train/rwkv/benchmark_stopd.py \
  --checkpoint <global_step_directory> --output <new_benchmark_directory>
```

This runs three updates per arm with the checkpoint's original training
configuration and independent, reproducible sampling seeds. It checks restored
optimizer/data state, retained rollout prefixes, finite updates, and the saved
STOPD denominator, and reports end-to-end and per-stage timings. The optional
`--continue-training` resumes the resulting STOPD checkpoint after these checks
and at least 5% measured end-to-end acceleration. Periodic evaluation and
checkpoint intervals then follow the original experiment configuration.
