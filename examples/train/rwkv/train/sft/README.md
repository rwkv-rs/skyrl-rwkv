# RWKV SFT

SFT consumes the pretokenized store produced by `prepare.py`, uses the native
length-grouped sampler, and evaluates each saved checkpoint through
`../eval/run.sh` on all eight GPUs. The default is full-parameter training;
`USE_NORA=true` enables adapter training. `--test_mode` creates a bounded
pretokenized store and runs five steps. Pass `--sst` to enable Subspace
Tuning on the full-parameter run. SST protects the leading singular
directions of the RWKV attention and MLP projections while preserving the
optimizer's ordinary moments and applied update scale. It is mutually
exclusive with NoRA; set `SST_RANK` (default `0.99`) and
`SST_REFRESH_INTERVAL` (default `0`, no refresh) to control the protected
fraction and basis refresh cadence.
