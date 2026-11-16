"""Plain-replica (DDP) training workers.

Counterparts of ``fsdp_worker.py`` for the :class:`DDPStrategy` backend:
every rank owns a complete model replica, gradients are averaged by DDP's
bucketed all-reduce, and optimizer/clip/weight-sync/export all run on plain
tensors (see ``ddp_strategy.py`` for the measured rationale).

LoRA/NoRA and the sharded_rdt / FP8 weight-sync paths are FSDP/Megatron
features and are rejected here; full-parameter training (``lora.rank=0``)
is the supported configuration.
"""

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import ray
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig

from skyrl.backends.skyrl_train.utils.profiler import build_profiler_from_policy_cfg
from skyrl.backends.skyrl_train.workers.ddp.ddp_strategy import DDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    get_llm_for_sequence_regression,
)
from skyrl.backends.skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    BaseBatchIterator,
    get_inference_weight_prefix,
)

if TYPE_CHECKING:
    from skyrl.train.config import InferenceEngineConfig


class DDPPolicyWorkerBase(PolicyWorkerBase):
    def _iter_microbatch_experiences(self, microbatch_iterator):
        if (
            os.environ.get("SKYRL_RWKV_PREFETCH", "0") != "1"
            or not getattr(self.model, "is_rwkv", False)
            or microbatch_iterator.data.device != torch.device("cpu")
            or not torch.cuda.is_available()
            or any(
                value is not None and not isinstance(value, torch.Tensor) for value in microbatch_iterator.data.values()
            )
        ):
            yield from super()._iter_microbatch_experiences(microbatch_iterator)
            return
        device = torch.cuda.current_device()
        compute_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
        batches = iter(microbatch_iterator)

        def stage():
            batch = next(batches, None)
            if batch is None:
                return None
            lengths = batch["attention_mask"].sum(-1).tolist()
            batch.pin_memory()
            with torch.cuda.stream(copy_stream):
                batch.to(device, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(copy_stream)
            experience = BaseBatchIterator.batch_to_experience(batch)
            experience.metadata = dict(batch.metadata or {})
            experience.metadata["rwkv_sequence_lengths"] = lengths
            return experience, ready

        pending = stage()
        while pending is not None:
            experience, ready = pending
            compute_stream.wait_event(ready)
            for value in vars(experience).values():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value.record_stream(compute_stream)
            pending = stage()
            yield experience
        copy_stream.synchronize()

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg: "InferenceEngineConfig"):
        if inference_engine_cfg.fp8_weight_sync_mode is not None:
            raise ValueError("Serialized FP8 weight sync currently requires trainer.strategy='megatron'.")
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy == "ddp"
        strategy = DDPStrategy(
            fsdp_config=self.cfg.policy.fsdp_config,
            # Inference-only workers skip the optimizer entirely.
            optimizer_config=None if self.cfg.policy.inference_only_init else self.cfg.policy.optimizer_config,
            model_config=self.cfg.policy.model,
            ddp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
            bf16=self.cfg.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = self.cfg.policy.model.lora.rank > 0

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        is_multimodal = hasattr(model_config, "vision_config") and model_config.vision_config is not None
        self._is_multimodal_lm_only = self.cfg.policy.language_model_only and is_multimodal

        # Full replica: load directly onto the local GPU (a few GB for the
        # 1B model); no meta-init is needed because nothing is sharded.
        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            # Under trainer.bf16 a full-parameter DDP model keeps fp32
            # master weights; the strategy exposes differentiable BF16 compute
            # views while DDP accumulates and reduces gradients in FP32.
            bf16=self.cfg.policy.inference_only_init or (self._is_lora and self.cfg.bf16),
            lora_rank=self.cfg.policy.model.lora.rank,
            lora_alpha=self.cfg.policy.model.lora.alpha,
            lora_dropout=self.cfg.policy.model.lora.dropout,
            lora_init_method=self.cfg.policy.model.lora.init_method,
            use_nora=self.cfg.policy.model.lora.use_nora,
            target_modules=self.cfg.policy.model.lora.target_modules,
            exclude_modules=self.cfg.policy.model.lora.exclude_modules,
            sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            use_torch_compile=self.cfg.policy.use_torch_compile,
            model_config_kwargs=self.cfg.policy.model_config_kwargs,
            meta_init=False,
            language_model_only=self.cfg.policy.language_model_only,
            logprobs_chunk_size=self.cfg.logprobs_chunk_size,
            rwkv_chunk_size=self.cfg.rwkv_chunk_size,
            rwkv_response_only_head=self.cfg.rwkv_response_only_head,
        )
        if (
            wrapped_model.is_rwkv
            and self.cfg.fully_async.enabled
            and not self.cfg.fully_async.clear_kv_cache_on_weight_sync
        ):
            raise ValueError("RWKV fully-async training requires fully_async.clear_kv_cache_on_weight_sync=True")
        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        if self.cfg.gradient_checkpointing:
            wrapped_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
            )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        if self.cfg.policy.inference_only_init:
            assert (
                self.optimizer is None and self.scheduler is None
            ), "inference_only_init should skip optimizer and scheduler construction"
        else:
            assert (
                self.optimizer is not None and self.scheduler is not None
            ), "DDP preparation should create optimizer and scheduler"

        # Enable expandable_segments after init so model weights stay in
        # IPC-compatible standard CUDA memory (same as the FSDP workers).
        self._set_expandable_segments(True)

        # Created only on profiled ranks.
        self.profiler = build_profiler_from_policy_cfg(self.cfg)

    @contextmanager
    def _microbatch_context(self, *, is_last: bool):
        # Skip DDP's gradient all-reduce for every micro-batch but the last;
        # the final micro-batch's backward does one bucketed all-reduce over
        # the whole accumulated gradient (same semantics as FSDP's
        # set_requires_gradient_sync(False) with sync_gradients_each_microbatch
        # off).
        module = self.model.model
        if not is_last and isinstance(module, DDP):
            with module.no_sync():
                yield
            return
        yield

    def _build_weight_source(self, dtype: "torch.dtype", backend: str) -> Any:
        """``WeightSource`` over the plain (unsharded) policy replica.

        ``FsdpWeightSource`` is reused on purpose: for non-DTensor
        parameters ``materialize_full_tensor`` is a pass-through, so
        iteration is a local ``state_dict()`` read with no collectives --
        exactly what a colocated engine wants per rank.
        """
        from skyrl.backends.skyrl_train.weight_sync.sources import FsdpWeightSource

        weight_prefix = get_inference_weight_prefix(self._is_multimodal_lm_only)
        if backend == "sharded_rdt":
            raise ValueError("DDP strategy does not support sharded_rdt weight sync (weights are unsharded).")
        if self.cfg.policy.model.lora.use_nora:
            raise ValueError("DDP strategy does not support NoRA weight sync yet.")
        module = self.model.model
        if isinstance(module, DDP):
            # DDP prefixes state_dict keys with module.; the inference engine expects plain HF names.
            module = module.module
        return FsdpWeightSource(module, dtype, weight_prefix)

    async def broadcast_to_inference_engines(
        self,
        inference_engine_client,
        inference_engine_cfg,
        model_id: Optional[str] = None,
    ):
        if inference_engine_client is None:
            inference_engine_client = self._weight_sync_inference_client
        if self._is_lora:
            raise NotImplementedError(
                "DDP strategy does not support LoRA/NoRA weight sync; train full-parameter "
                "(trainer.policy.model.lora.rank=0) or use trainer.strategy=fsdp."
            )
        await self._sync_weights_to_inference_engines(inference_engine_client, inference_engine_cfg)

    def _set_pad_token_id(self, pad_token_id):
        # NOTE: self.model -> HFModelWrapper; self.model.model -> AutoModelForCausalLM (DDP-wrapped)
        if not self.model.is_rwkv:
            module = self.model.model
            if isinstance(module, DDP):
                module = module.module
            module.config.pad_token_id = pad_token_id


class DDPCriticWorkerBase(CriticWorkerBase):
    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy == "ddp"
        strategy = DDPStrategy(
            fsdp_config=self.cfg.critic.fsdp_config,
            optimizer_config=self.cfg.critic.optimizer_config,
            model_config=self.cfg.critic.model,
            ddp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        critic = get_llm_for_sequence_regression(
            model_path,
            "critic",
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=False,
            lora_rank=self.cfg.critic.model.lora.rank,
            lora_alpha=self.cfg.critic.model.lora.alpha,
            lora_dropout=self.cfg.critic.model.lora.dropout,
            use_nora=self.cfg.critic.model.lora.use_nora,
            freeze_backbone=self.cfg.critic.freeze_backbone,
            target_modules=self.cfg.critic.model.lora.target_modules,
            exclude_modules=self.cfg.critic.model.lora.exclude_modules,
            value_head_prefix=self.cfg.algorithm.value_head_prefix,
            init_value_head=self.cfg.policy.model.path == self.cfg.critic.model.path,
            sequence_parallel_size=self.cfg.critic.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            model_config_kwargs=self.cfg.critic.model_config_kwargs,
            meta_init=False,
        )
        self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

        if self.cfg.gradient_checkpointing and not self.cfg.critic.freeze_backbone:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
            )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None
        if self.cfg.critic.resume_path is not None:
            self.load_checkpoint(
                self.cfg.critic.resume_path,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )

        self._set_expandable_segments(True)

    def _set_pad_token_id(self, pad_token_id):
        if not self.model.is_rwkv:
            self.model.config.pad_token_id = pad_token_id


class DDPRefWorkerBase(RefWorkerBase):
    def init_model(self, model_path):
        assert self.cfg.strategy == "ddp"
        strategy = DDPStrategy(
            fsdp_config=self.cfg.ref.fsdp_config,
            ddp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=self.cfg.bf16,
            sequence_parallel_size=self.cfg.ref.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            model_config_kwargs=self.cfg.ref.model_config_kwargs,
            meta_init=False,
            language_model_only=self.cfg.ref.language_model_only,
            logprobs_chunk_size=self.cfg.logprobs_chunk_size,
            rwkv_response_only_head=self.cfg.rwkv_response_only_head,
        )
        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

        self._set_expandable_segments(True)


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(DDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(DDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(DDPRefWorkerBase)
