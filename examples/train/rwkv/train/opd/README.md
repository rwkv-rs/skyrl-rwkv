# RWKV on-policy distillation

The launcher keeps the student and teacher on the eight-card placement and
uses eight inference engines for rollout. `USE_NORA=true` is the memory-saving
default; set it to `false` for full-parameter student training. `--test_mode`
creates bounded parquet prefixes and limits the run to five steps.
