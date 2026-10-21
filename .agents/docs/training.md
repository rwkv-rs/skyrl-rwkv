# Training

## Entrypoints

- **`skyrl/train/entrypoints/main_base.py`** — Primary training entrypoint. Handles inference server setup, training loop, weight sync.
- **`skyrl/train/entrypoints/main_generate.py`** — Generation-only entrypoint (no training).
- **`skyrl/train/main_value.py`** — Standalone offline critic pretraining via `torchrun` and native `FSDPStrategy`; argparse flags, no Ray/inference engines. Optional `--use-nora` freezes the RWKV backbone and trains existing fused NoRA adapters plus the full FP32 value head; `--rank` defaults to 8 and alpha defaults to rank. Rank sweeps double 8, 16, 32, etc. FSDP checkpoints retain adapters/head; HF exports and `best/` merge complete BF16 critic weights. Dataset/collator: `skyrl/train/dataset/value_dataset.py`. Exact-token input contract and RWKV commands: `examples/train/rwkv/README.md` (Offline value pretraining).
- **`examples/train/rwkv/antidoom.py`** — Select complete benchmarks by active G1k truncation rate, export all original native LightEval questions, append untested AntiDoom mix components with `export-mix` (excluding GSM8K/MMLU/MATH and MetaMathQA), and generate fresh rollouts with an explicit RWKV checkpoint. Extract FTPO data from new repetitions, then train NoRA in length-bucketed microbatches with native `FSDPStrategy` via `torchrun` and automatically merge the final adapter into full HF weights. See the RWKV README (Antidoom FTPO with NoRA) for selection/export commands and pinned dependencies.

RWKV offline value data collection uses `examples/train/rwkv/rollout.py --sampling eval --save-rollouts` to obtain correct and incorrect completions before balancing. Online SAO retains its separate unmodified rollout sampling. Build the exact-token train/validation dataset with `examples/train/rwkv/build_value_data.py --input <rollout-output> --output <new-data-directory>`; see the RWKV README for the complete command chain.

`rollout.py --samples-per-prompt` controls collection size (default 64). For online
UltraData rewards, use `prepare_opd.py --task-rewards --validation-fraction 0.05
--seed 42` and the same UUID split as value pretraining. The native `ultradata`
environment loads golden answers/test cases by source path and byte offset and
shares the offline scoring functions. Per-environment template metadata selects
Math `open_think` and other domains `fake_think`; use unbatched generation with
conversation multi-turn token preservation. Online FSDP critics accept
`trainer.critic.model.lora.use_nora=true`, training adapters and the full head.

Calibrate a critic against a fixed policy with the same `--sampling rollout`
distribution and generation budget used online. Preserve the observed outcome
frequencies. For tasks that count budget exhaustion as failure, pass
`build_value_data.py --truncation-as-terminal` to retain these trajectories with
explicit zero returns. Use `main_value.py --loss-reduction sequence_mean` to
match the online critic objective across accumulated microbatches, and
`--eval-before-train` to measure the initialization on the same held-out questions.
Validation reports token/sequence MSE and MSE skill relative to the optimal constant
prediction, so a low loss on mostly negative outcomes does not imply a useful critic.
It also reports first-action calibration and per-domain statistics when the dataset
contains `domain`, exposing biases that an aggregate score can hide.

Online SAO uses the native `gae_adaptive` estimator with `rollout_is` loss. It skips
observation positions using `response_mask & loss_mask`, uses action-length-adaptive
policy lambda (`gae_adaptive.alpha=1.5`, lambda one below `gae_adaptive.min_length=50`), and separate critic lambda
(`gae_adaptive.critic_lambd=1`). Optional `rollout_is.score_centering=true` reuses
FSDP sampler-support capture and centers the complete DIS-weighted score. Its raw
support capture requires temperature 1, top-p 1, top-k -1 and no penalties. See the
RWKV README's Online SAO section for the configuration and validation scope.

With separate policy/critic GPU groups, `trainer.train_critic_and_policy_parallel=true`
overlaps their updates while retaining the barrier before weight sync; colocation
is rejected. Set `trainer.critic.optimizer_config.lr=0` to retain critic inference
while skipping its backward and optimizer steps, including after checkpoint restore.
Offline value pretraining supports `--group-by-length` through the
Transformers distributed sampler. Its metrics separate training time/token throughput
from evaluation and checkpoint export time.

RWKV token microbatching (`trainer.max_tokens_per_microbatch`) groups similar
lengths under a padded-token budget, preserving complete streams and restoring
sample order. Offline benchmark generation can opt into CUDA graphs with
`antidoom.py generate --no-enforce-eager`; its default remains eager.
Full-sequence RWKV training right-pads internal recurrent inputs to a multiple
of 16, retaining the native Pretrain kernel instead of the ragged StateTune
path. Output positions and loss masks retain their original lengths.

SAO logs unscaled critic token/sequence MSE, prediction/target means, residual bias,
and explained variance when targets vary. These diagnose calibration separately
from the microbatch-scaled optimization loss. Budget-truncated UltraData answers
receive zero terminal reward; `generator.apply_overlong_filtering=true` excludes
them from actor and critic losses for a controlled ablation. Sampled EOS remains
an action even when its ID equals the padding ID; synthetic EOS stays masked.

Set `trainer.metric_for_best_model=eval/all/avg_score` to save full training state
and an HF export whenever evaluation improves. `trainer.greater_is_better` defaults
to true. `best_checkpoint.json` identifies the retained checkpoint; retention keeps
it in addition to the last `max_ckpts_to_keep` checkpoints, and resume restores
its score/step. If `export_path` equals `ckpt_path`, HF exports use each model's
`huggingface/` subdirectory so atomic export cannot overwrite optimizer/RNG state.

## Running Training

```bash
# Megatron GRPO on GSM8K
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron \
  trainer.policy.model.path=<model> \
  environment.env_class=gsm8k \
  ...

# Use --env-file for secrets
uv run --isolated --extra megatron --env-file .env.test -m skyrl.train.entrypoints.main_base ...
```

## Config

- Configurations are implemented as dataclass. CLI parsing is via OmegaConf.
- Pass overrides as `key=value` args on the command line. Unlike Hydra, we do not support `+` overrides for new keys
- Main config object: `SkyRLTrainConfig` in `skyrl/train/config/`.
- Document a new or changed field with an attribute docstring directly below it (the `"""..."""`
  style already used throughout `config.py`), not in a separate docs page. The
  `/docs/api-ref/skyrl/config` reference page is generated from these docstrings by
  `docs/generate-api-docs.py` via griffe, which reads attribute docstrings but *not*
  `field(metadata={"help": ...})`.
- Keep the first physical line of an attribute docstring a complete sentence: the generated
  reference's summary table shows only that first line, truncating at the newline.
- If a field's guidance is topic-specific (a backend, a training mode, a pitfall with a symptom),
  put the long-form explanation on the corresponding example/tutorial/troubleshooting page and keep
  the docstring itself focused on the field.

## Example Scripts

Located in `examples/train/<task>/`:
- `examples/train/gsm8k/` — GSM8K math training
- `examples/train/text_to_sql/` — SQL training
- `examples/train/rwkv/run_rwkv_on_policy_distill.sh` — RWKV7 G1k-13.3B to G1k-1.5B on-policy distillation on local UltraData with AIME 2024 evaluation, using the native OPD entrypoint and the project-local RWKV environment. `prepare_opd.py` streams all four UltraData domains to Parquet; `profile_opd.py` measures full-context FSDP/FlashRWKV2 component throughput on torchrun ranks. Setup and overrides: `examples/train/rwkv/README.md` (On-policy distillation).
- RWKV OPD defaults to the merged AntiDoom teacher and optional NoRA student training (`trainer.policy.model.lora.use_nora`). It reuses AntiDoom's pinned PEFT fork and FlashRWKV2 fused projections; FSDP weight sync and HF exports merge complete weights in FP32 before FP16 conversion. Ordinary LoRA keeps its adapter sync path.
- Optional STOPD lives in the native OPD example (`trainer.algorithm.stopd`, disabled by default). It uses independent per-request block budgets, inverse-probability token weights, and a lagged token-count denominator, with full-response evaluation. `stopd_state.pt` is saved with the native checkpoint; importing a Vanilla checkpoint after warmup requires the preceding full response-token count. `examples/train/rwkv/benchmark_stopd.py` compares both modes from one complete checkpoint and can continue its validated STOPD checkpoint.
- Each has a `run_*.sh` script with preconfigured overrides.
