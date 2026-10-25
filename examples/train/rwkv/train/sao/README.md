# RWKV SAO

The production configuration uses one rollout per prompt, asynchronous value
updates, adaptive GAE, eight inference engines, and separate optional NoRA
settings for the policy and critic. Set `POLICY_USE_NORA` or
`CRITIC_USE_NORA` to select adapter training independently. `--test_mode`
copies only a bounded prefix of each parquet split and runs five steps.
