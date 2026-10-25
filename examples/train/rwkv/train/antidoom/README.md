# RWKV Antidoom / FTPO

`run.sh` launches the reference FTPO implementation with NoRA adapters on all
eight GPUs. It accepts `--test_mode`, which creates bounded JSONL prefixes and
sets `--max-steps 5`; the default uses the complete prepared preference set.

```bash
bash examples/train/rwkv/train/antidoom/run.sh --test_mode
bash examples/train/rwkv/train/antidoom/run.sh
```
