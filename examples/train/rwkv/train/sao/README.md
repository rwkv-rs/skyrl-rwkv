# RWKV SAO

The production configuration uses one rollout per prompt, adaptive GAE, two
policy GPUs, four critic GPUs, and two dedicated inference engines. Critic and
policy updates overlap generation and each other. Set `ASYNC_MODE=disabled` to
benchmark the eight-GPU colocated synchronous path. `POLICY_USE_NORA` and
`CRITIC_USE_NORA` select adapter training independently. `--test_mode` copies
only a bounded prefix of each parquet split and runs five steps.

The production split is two policy GPUs, four critic GPUs, and two TP1
inference engines. It uses a 128-prompt batch, 32,768-token trainer
microbatches, 16,384 inference batched tokens, and 64 generation workers per
inference engine. The critic keeps two update passes per rollout while policy
and critic updates overlap generation. Test mode uses batch 32, critic
mini-batches of 16, and one evaluation sample per prompt; set
`ASYNC_MODE=disabled` to run the stable colocated comparison with one shared
micro-batch.
