# RWKV evaluation

`run.sh` starts eight one-GPU vLLM-RWKV workers and a round-robin router, then
invokes the pinned `lighteval-rwkv` producer through `prepare.py`. Results are
written as flat W&B metrics plus detailed JSON rows.
