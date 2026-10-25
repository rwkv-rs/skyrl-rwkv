# RWKV Antidoom / FTPO

`run.sh` launches the reference FTPO implementation with NoRA adapters on all
eight GPUs. It accepts `--test_mode`, which creates bounded JSONL prefixes and
sets `--max-steps 5`; the default uses the complete prepared preference set.

The eight-GPU default uses two examples per rank microbatch, eight gradient
accumulation steps, and a 32,768-token padded-token cap. That gives a
128-example global update while length bucketing keeps long responses from
forcing every microbatch to the worst-case length. `NUM_GPUS`, `FSDP_SIZE`,
`MICRO_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, and `MAX_BATCH_TOKENS` remain
available for a hardware sweep. Evaluation is capped at 1,024 validation
examples in production and 256 in test mode.

```bash
bash examples/train/rwkv/train/antidoom/run.sh --test_mode
bash examples/train/rwkv/train/antidoom/run.sh
```
