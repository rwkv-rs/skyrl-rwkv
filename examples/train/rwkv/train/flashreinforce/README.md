# RWKV FlashREINFORCE

This launcher uses native SkyRL FSDP, eight colocated TP1 vLLM-RWKV engines,
`wkv_mode=fp32io16`, and the production rollout/evaluation sampling policies.
Set `USE_NORA=true` to train adapters; the default is full-parameter training.
`--test_mode` prefixes both parquet splits and runs exactly five optimizer
steps.

```bash
bash examples/train/rwkv/train/flashreinforce/run.sh --test_mode
bash examples/train/rwkv/train/flashreinforce/run.sh
```
