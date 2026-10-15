# Training

## Entrypoints

- **`skyrl/train/entrypoints/main_base.py`** — Primary training entrypoint. Handles inference server setup, training loop, weight sync.
- **`skyrl/train/entrypoints/main_generate.py`** — Generation-only entrypoint (no training).
- **`skyrl/train/main_value.py`** — Standalone offline critic pretraining via `torchrun` and native `FSDPStrategy`; argparse flags, no Ray/inference engines. Dataset/collator: `skyrl/train/dataset/value_dataset.py`. Exact-token input contract and RWKV commands: `examples/train/rwkv/README.md` (Offline value pretraining).
- **`examples/train/rwkv/antidoom.py`** — Select complete benchmarks by active G1k truncation rate, export all original native LightEval questions, append untested AntiDoom mix components with `export-mix` (excluding GSM8K/MMLU/MATH and MetaMathQA), and generate fresh rollouts with an explicit RWKV checkpoint. Extract FTPO data from new repetitions, then train NoRA in length-bucketed microbatches with native `FSDPStrategy` via `torchrun` and automatically merge the final adapter into full HF weights. See the RWKV README (Antidoom FTPO with NoRA) for selection/export commands and pinned dependencies.

RWKV offline data collection uses `examples/train/rwkv/rollout.py --sampling rollout --save-rollouts`. Build the exact-token train/validation dataset with `examples/train/rwkv/build_value_data.py --input <rollout-output> --output <new-data-directory>`; see the RWKV README for the complete command chain.

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
- `examples/train/rwkv/run_rwkv_on_policy_distill.sh` — RWKV7 G1k-13.3B to G1k-1.5B on-policy distillation on DAPO-Math with AIME 2024 evaluation, using the native OPD entrypoint and the project-local RWKV environment. Setup and overrides: `examples/train/rwkv/README.md` (On-policy distillation).
- Each has a `run_*.sh` script with preconfigured overrides.
