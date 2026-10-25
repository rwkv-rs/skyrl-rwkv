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
bash examples/train/rwkv/train/opd/run.sh \
  trainer.algorithm.stopd.enabled=true \
  trainer.algorithm.stopd.block=128 \
  trainer.algorithm.stopd.warmup_steps=25
```

Warmup counts completed global updates, including updates restored from a
checkpoint. Without TOP-D, the supported configuration is synchronous, single-turn OPD with
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
  examples/train/rwkv/train/opd/benchmark_stopd.py \
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

This option alone implements immediate reward smoothing and composes with
STOPD's rollout truncation. Enable the complete algorithm separately below.

Use `--proximal-teacher-alpha 0.2` with `benchmark_stopd.py` to apply the same
reward transformation to both full and truncated rollout arms and the resumed
production run. The benchmark records the coefficient and verifies actual
transformed rewards and IPW advantages before continuing training.

## Optional complete TOP-D

Enable `trainer.algorithm.topd.enabled=true` together with `no_op` advantages,
the native `regular` PPO loss, `prompt_mean` loss reduction, and at least two
responses per prompt. The example computes Eq. (8)'s immediate reward plus the
mean strictly future reward (zero future reward for the final token). It then
normalizes returns using the population mean and standard deviation over all
valid tokens of all responses to the same prompt. Padding is excluded. The
standard deviation has an additive `trainer.algorithm.topd.epsilon=1e-6`.

Eq. (7)'s loss averages token losses within each prompt group, then averages
across prompts in each mini-batch. Teacher logprobs, behavior-policy logprobs,
returns, and advantages are computed once per rollout batch and remain fixed
through native PPO mini-batches and `trainer.update_epochs_per_batch` epochs.
Only student logprobs are recomputed after optimizer updates. Evaluation,
NoRA, FlashRWKV2, and full-weight synchronization retain their native paths.
Complete TOP-D rollouts use independent request seeds for responses in each
group, including when STOPD is disabled.

The RWKV launcher exposes a complete recipe matching the paper's group size,
mini-batch size, clipping defaults, and proximal-teacher coefficient:

```bash
USE_TOPD=true USE_STOPD=true \
  bash examples/train/rwkv/train/opd/run.sh \
  trainer.resume_mode=from_path \
  trainer.resume_path=<latest_complete_checkpoint>
```

Launcher defaults are `TOPD_GROUP_SIZE=8`, `TOPD_MINI_BATCH_SIZE=32`,
`TOPD_ALPHA=0.2`, and one epoch over the collected batch. SkyRL batch sizes
count **prompts**: 512 prompts with group size 8 produce 4096 responses and
16 optimizer updates. Adjust the group size and mini-batch size for available
hardware. `UPDATE_EPOCHS_PER_BATCH` controls additional reuse. Keep the prompt
batch size unchanged when restoring an existing dataloader checkpoint.

**STOPD compatibility:** enabling both options selects the example's
`topd_stopd` custom policy loss. Responses to the same prompt share a random
STOPD budget, while token sampling seeds remain independent. Generation,
teacher scoring, and training stop at the observed prefix. Evaluation still
uses its full context budget. Simply applying token IPW to prefix-normalized
TOP-D advantages would bias the nonlinear group statistics and future returns.

The combination instead uses a finite randomized telescoping sum, an extension
of the two algorithms based on [Rhee & Glynn's randomization method](https://web.stanford.edu/~glynn/papers/2015/RheeG15.html).
Let `F_j` be the complete clipped TOP-D group objective at horizon
`min(j * block, max_len)`, recomputing Eq. (8), group moments, and the group's
token count at that horizon. If `J` is the observed horizon and
`q_j = (max_len - (j - 1) * block) / max_len`, the estimator is
`F_1 + sum_{j=2..J} (F_j - F_{j-1}) / q_j`.
All required terms are computable from the observed prefix. Its expectation
over stopping budgets is `F_M`, including EOS and a partial final block.
Warmup uses the ordinary full TOP-D objective.

Signed telescoping coefficients stay outside PPO clipping. Two fixed target
channels accumulate positive and negative advantages separately, preserving
the clipped objective without storing one tensor per horizon. Native
mini-batches and inner epochs reuse these targets and behavior logprobs.
Tests exhaustively compare expected losses and gradients with full TOP-D,
including off-policy ratios on both sides of the clipping interval. This
fixed-policy identity does not guarantee equal optimizer trajectories or
variance: randomized estimates can have higher variance, and BF16 kernels
can introduce sequence-shape-dependent rounding. Measure actual throughput
and gradient behavior when choosing a production configuration. With TOP-D
disabled, STOPD keeps its original token IPW and lagged normalization.

Native checkpoints restore the model, optimizer, scheduler, RNG, and data cursor;
the existing STOPD sidecar also restores its step and budget seed. TOP-D adds
no persistent statistics: each new batch recomputes group moments. An existing
OPD/STOPD checkpoint can enter TOP-D without resetting the student or data
position. TOP-D normalizes within each group objective and needs no legacy lagged
denominator when importing a checkpoint without STOPD state.
Epoch-end resumes finish the saved iterator before entering the next epoch,
preserving the next shuffle as well as mid-epoch data order.
`topd/observed_tokens`, `topd/observed_return_mean`, target coefficient statistics,
and `topd/optimizer_steps_per_batch` expose actual training work in the tracker.
Keep training and evaluation lengths separate; this recipe retains the RWKV
training response limit of 8192 and LightEval's total context budget of 25600.

For padded RWKV kernels, `trainer.algorithm.topd.balance_batch=true` sorts
whole prompt groups by length within each existing PPO mini-batch, then balances
equal-size microbatch units across data-parallel ranks. It preserves mini-batch
membership, group statistics, and optimizer update counts. Behavior logprobs and
training use the same reordered microbatches. The batch records the original row
mapping in `batch_permutation`; `perf/dense_tokens_*` and
`perf/policy_rank_max_tokens_*` report the change in padded work. This option
requires fixed, equal forward/train microbatches aligned with the prompt/DP grid.

`trainer.rwkv_response_only_head=true` uses the installed RWKV model's
`logits_to_keep` API to project only response prediction positions. The full
recurrent history and its gradients are retained. Valid response logprobs and
entropy keep their native positions; unused prompt/padding outputs are zero.
This applies to the non-chunked dense RWKV path; score centering retains its
full support layout. Other model paths are unchanged. Both performance options
are disabled by default. The RWKV launcher exposes them as
`TOPD_BALANCE_BATCH=true RWKV_RESPONSE_ONLY_HEAD=true`.

## RWKV evaluation and metrics

The RWKV launcher evaluates the synchronized student with the existing
`lighteval-rwkv` producer, following `run_rwkv_sft.sh`. Its task registry,
CoT prompts, boxed-answer extraction, and math verifier own the evaluation.
The default task remains AIME 2024 with 32 independent completions per question
(`avg@32`). All eight GPUs serve the producer through SkyRL's OpenAI router.
For asynchronous training, evaluation closes rollout admission, waits for
in-flight training requests, and offloads trainer models and optimizer states.
Temporary native vLLM servers load the current merged HF export on the learner
GPUs and join the existing rollout servers. Evaluation then releases those
temporary servers, restores trainer residency, and resumes admission; buffered
training rollouts retain their order and policy versions. With seven learner
GPUs and one rollout GPU, this gives eight independent evaluation engines.
Evaluation uses temperature 0.96, top-p 0.76, top-k 32,
presence penalty 1.0, frequency penalty 0.1, penalty decay 0.988, and the
`bot` / `open_think` prompt settings supplied by the producer.
The single-turn RWKV launcher uses the native router's `round_robin` policy,
so independent OpenAI requests without session headers reach all engines.

Set `LIGHTEVAL_PROJECT` to the existing producer checkout. The evaluation
subprocess uses this project's interpreter. Install the producer's pinned
math packages into a private directory inside this project's `.venv`:

```bash
uv run --isolated --no-project uv pip install \
  --python .venv/bin/python3 --target .venv/rwkv-eval --no-deps \
  'latex2sympy2_extended==1.0.6' 'antlr4-python3-runtime==4.13.2'
```

The evaluation subprocess prepends that directory and the producer's sources
to `PYTHONPATH`. Training retains its existing dependencies. The producer's
remaining runtime dependencies must be installed in this project's environment.

`trainer.algorithm.lighteval.context_length=25600` is a **total** token budget.
For each question, the adapter tokenizes the producer's actual messages with
the inference tokenizer and sets completion tokens to `25600 - prompt_tokens`.
This leaves the training rollout limit of 8192 and STOPD's truncation schedule
unchanged. Set `trainer.algorithm.lighteval.project=null` to use native OPD
evaluation instead. Non-RWKV OPD examples keep their native evaluator by default.

Each evaluation saves the producer's completions, cache, details, scores, and
producer commit under `<export_path>/lighteval/global_step_<N>/`. The native
trainer logs `eval/lighteval/aime24/avg@32`, completion counts, and truncation
rate to the experiment tracker at the checkpoint's actual global step.
`trainer.num_logger_eval_samples` controls how many raw completions and native
extracted answers are also uploaded to the W&B sample table (20 in the RWKV
launcher). Complete evaluation records remain in the producer output.

OPD replaces environment rewards with dense teacher/student log-ratio rewards;
environment task reward is not part of its optimization objective. Track
`opd/reward_mean`, `opd/reward_abs_mean`, and `opd/reverse_kl_mean` for
token-normalized distillation diagnostics. With STOPD alone, these use the same
inverse-probability weights and lagged denominator as the loss, excluding
padding. With TOP-D, these report means of the observed tokens; truncated
prefix diagnostics are not estimates of the full-response token mean.
The reverse-KL diagnostic is a signed Monte Carlo estimate and can
be negative on an individual batch. Use the producer's task scores to assess
capability, rather than interpreting environment rewards or sequence reward
sums as evaluation accuracy.
