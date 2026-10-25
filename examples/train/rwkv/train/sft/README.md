# RWKV SFT

SFT consumes the pretokenized store produced by `prepare.py`, uses the native
length-grouped sampler, and evaluates each saved checkpoint through
`../eval/run.sh` on all eight GPUs. The default is full-parameter training;
`USE_NORA=true` enables adapter training. `--test_mode` creates a bounded
pretokenized store and runs five steps.
