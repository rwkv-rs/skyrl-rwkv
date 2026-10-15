# On-Policy Distillation
This folder contains scripts for running On-Policy Distillation on SkyRL.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6ecc9da-cb67-4935-b95a-193ba6c4843c" width="45%" />
  <img src="https://github.com/user-attachments/assets/255c46bd-f443-4f36-b8fc-9546c3a01cf1" width="45%" />
</p>

On-Policy Distillation is a technique recently highlighted by [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation), which combines the benefits of on-policy RL style training, with the dense reward signal of distillation. The main idea is to collect on-policy samples from a student model, use a teacher to grade each token from the student samples, update the student policy accordingly, and repeat. On-Policy Distillation has previously been shown to be a more compute efficient approach than RL for efficiently post-training models ([Agarwal et al](https://arxiv.org/abs/2306.13649), [Gu et al](https://arxiv.org/abs/2306.08543), [Qwen3 team](https://arxiv.org/abs/2505.09388)).

In `main_on_policy_distill.py` we provide a simple example for modifying SkyRL to implement On-Policy Distillation by replacing the ref model with a teacher model, and modifying the reward/advantage computation logic to use the reverse KL loss.
<img width="471" height="51" alt="image" src="https://github.com/user-attachments/assets/4d5a9649-832a-4ba9-86af-5de063f2773f" />

We provide a full writeup at https://novasky-ai.notion.site/on-policy-distillation.

For RWKV7 G1k-13.3B to G1k-1.5B distillation using this same entry point, see the
[RWKV recipe](../rwkv/README.md#on-policy-distillation).

## Quickstart
To get started, first set up the dataset from the DAPO example:

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
```

Then, just make sure to set the path to your desired teacher model, and you're ready to kick off training!

```bash
TEACHER_MODEL=<YOUR_MODEL_HERE>
bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh trainer.ref.mode.path=$TEACHER_MODEL
```

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

To compare both modes from the same complete Vanilla OPD checkpoint and data
position after warmup:

```bash
uv run --isolated --no-project .venv/bin/python3 \
  examples/train/rwkv/benchmark_stopd.py \
  --checkpoint <global_step_directory> --output <new_benchmark_directory>
```

This runs three updates per arm with the checkpoint's original training
configuration and independent, reproducible sampling seeds. It checks restored
optimizer/data state, actual generator-to-trainer tokens/logprobs/masks, IPW
advantages, finite updates, and the saved STOPD denominator, and reports
end-to-end and per-stage timings. The optional
`--continue-training` resumes the resulting STOPD checkpoint after these checks
and at least 5% measured end-to-end acceleration. Periodic evaluation and
checkpoint intervals then follow the original experiment configuration.

## Optional proximal teacher reward

[Trust Region Policy Distillation (TOP-D)](https://arxiv.org/abs/2607.04751),
Eq. (5), replaces the immediate log-ratio reward with
`log(alpha * p_teacher / p_old + 1 - alpha)`. Set
`trainer.algorithm.proximal_teacher_alpha=0.2` (or the RWKV launcher's
`PROXIMAL_TEACHER_ALPHA=0.2`) to enable this probability mixture. The default
`1.0` preserves the original OPD computation. The implementation uses
`logaddexp` for numerical stability and reuses the cached teacher and student
token logprobs, requiring no additional model forward pass or checkpoint state.

This option implements TOP-D's immediate reward smoothing. It composes with
STOPD's token IPW correction. The paper's full algorithm also uses future
returns, group normalization, and clipped updates over multiple mini-batches;
those require separate treatment of unobserved suffixes under truncation.
The paper's reported full-algorithm accuracy gains therefore do not describe
this combination. Measure its quality through the experiment's evaluations.

Use `--proximal-teacher-alpha 0.2` with `benchmark_stopd.py` to apply the same
reward transformation to both full and truncated rollout arms and the resumed
production run. The benchmark records the coefficient and verifies actual
transformed rewards and IPW advantages before continuing training.
