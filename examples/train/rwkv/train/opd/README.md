# RWKV on-policy distillation

The default fully async launcher keeps the student and teacher on a shared
four-GPU placement and uses four dedicated inference engines for rollout. This
overlaps teacher-conditioned training with generation; set `ASYNC_MODE=disabled`
for the eight-GPU colocated synchronous baseline or `ASYNC_MODE=zero_stale` for
the one-step-off-policy comparison. `USE_NORA=true` is the memory-saving
default; set it to `false` for full-parameter student training. `--test_mode`
creates bounded parquet prefixes and limits the run to five steps.

The throughput profile reserves six GPUs for the student/ref learner and two
for TP1 inference. Production updates use batch 512, a 32,768-token trainer
microbatch cap, and 16,384 inference batched tokens; request fan-out is bounded
at 64 workers per inference engine instead of multiplying by the full stale
window. Test mode uses batch 32, micro-batch 2, and 128 total generation
workers. `ASYNC_MODE=disabled` selects the measured all-eight-GPU colocated
baseline and lowers the test micro-batch to 1 for allocator stability.
