# RWKV training recipes

The RWKV example is organized by responsibility. Each training method has one
`README.md`, `prepare.py`, and `run.sh`; shared rollout, data, evaluation, and
FlashRWKV2 preflight code lives in `train/common` or `eval`.

The five launchers are:

- `train/antidoom`
- `train/flashreinforce`
- `train/opd`
- `train/sao`
- `train/sft`

Use `--test_mode` for a bounded data prefix and five optimizer steps. Test mode
disables W&B and uses console logging. Normal runs force W&B on. The default
training settings target eight Pro6000 GPUs and use `wkv_mode=fp32io16`.

`eval/run.sh` starts eight one-GPU vLLM-RWKV workers behind one router and
reuses the stable `lighteval-rwkv` producer for GSM8K, MATH500, AIME24, AIME25,
and AIMO avg@k/pass@k evaluation.
