# RWKV Fork Upstream Weight-Sync Handoff

This document is for the maintainers of `transformers-rwkv` and `vllm-rwkv`.
The SkyRL branch has merged the latest `upstream/main` weight-sync architecture and
keeps RWKV support as a model-specific adapter.

## Required upstream alignment

1. Rebase the RWKV forks onto the exact upstream dependency versions used by SkyRL:
   - vLLM `0.30.0`
   - Transformers `5.16.1`
   - Torch `2.13.0` for the RWKV extra
2. Preserve the RWKV model type identifier `rwkv` and the RWKV7 stateful execution
   contract.
3. Preserve the existing RWKV vLLM entry points used by this repository:
   - `vllm.model_executor.models.rwkv`
   - `vllm.v1.worker.gpu.model_states.rwkv`
   - the dev RLHF API routes
4. Do not require SkyRL's removed legacy `layerwise_reload.py` API from vLLM.
   SkyRL now enters weight synchronization through vLLM's native weight-transfer
   engine lifecycle.

## Weight-sync contract

The receiver must accept checkpoint-format tensors through the native
`start_weight_update` / `receive_weights` / `finish_weight_update` lifecycle.
RWKV-specific requirements are:

- `*.mlp.value.weight` checkpoint tensors may require the runtime transpose used by
  the RWKV vLLM model.
- Runtime folded/canonical tensors must be refreshed after the full update, before
  the next generation request.
- CUDA graphs captured against the previous recurrent runtime state must be dropped
  after the update.
- Recurrent state/cache slots must not survive a policy weight update.
- The receiver must remain compatible with SkyRL's native NCCL, CUDA IPC, and delta
  transfer engines; it must not add model-specific behavior to other model types.

## Trainer-side contract

The trainer exports the effective BF16 values used by RWKV's mixed-precision forward
when the wire dtype is FP16. This avoids exporting the FP32 master value rounded
directly to FP16 when the actual forward consumed BF16.

The exporter must preserve:

- parameter names and source shapes expected by `transformers-rwkv`;
- deterministic iteration order across FSDP ranks;
- behavior logprob/token alignment;
- no sequence parallelism, microbatch packing, or `torch.compile` for RWKV.

## Optional FlashREINFORCE Score-Centering contract

SkyRL's Score-Centering path is opt-in (`flashreinforce.score_centering=false`
by default) and currently uses the FSDP worker with multi-turn trajectories. When
it is enabled, the inference engine must capture a separate float32 top-k logprob
side channel (`sample_support_top_k`, default 128) without changing rollout
sampling `top_k`. The side channel contains the sampler's raw logprobs, while the
support-ID channel remains available for support replay. Sampled-token behavior
logprobs and support rows must cover the same response tokens; synthetic EOS and
observation rows are zero/masked padding.

The trainer scores the recorded candidate IDs against the full current-policy
logits and separately uses the sampled-token logprob when vLLM's approximate
support omits that token. The existing FlashREINFORCE sequence-KL admission gate
and default loss are unchanged when Score-Centering is disabled. Megatron is not
currently enabled for this correction.

## Validation checklist

Run after rebasing either fork:

1. Model load and one-token recurrent forward.
2. Batched recurrent forward with right-padded sequences.
3. HF/trainer logprob parity on a fixed token batch.
4. vLLM generation with `wkv_mode=fp32io16`.
5. Native SkyRL NCCL and CUDA IPC weight-sync tests.
6. Delta weight-sync reload followed by generation.
7. Two consecutive weight updates with recurrent-state invalidation.
8. CUDA-graph generation after a weight update.
9. RWKV tokenizer/chat-template compatibility for all three supported prompt templates.

## Deliverables back to SkyRL

Please provide:

- the new fork commit SHAs;
- a short list of upstream conflicts and their resolutions;
- confirmation that the native weight-transfer lifecycle above is supported;
- the exact model/config/tokenizer revision used for parity tests.
