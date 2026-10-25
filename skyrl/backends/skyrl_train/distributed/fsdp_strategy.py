import copy
import gc
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from loguru import logger
from packaging import version
from torch import distributed as dist
from torch import optim
from transformers.trainer import get_scheduler

from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    PrecisionType,
    apply_fsdp2,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_get_full_state_dict,
    fsdp2_load_full_state_dict,
    load_fsdp2_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp2_model_to_cpu,
    offload_fsdp_optimizer,
)
from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig
from skyrl.utils.tok import check_is_vlm, get_processor

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
else:
    CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy = None, None, None


_DEFAULT_SST_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # RWKV projections used by the native NoRA recipe.
    "r_proj",
    "mlp.key",
    "mlp.value",
)


class _SSTProjector:
    """Keep the free-to-free singular subspace for one dense weight matrix."""

    def __init__(self, weight: torch.Tensor, rank: float) -> None:
        self.rank = rank
        self.free_left: Optional[torch.Tensor] = None
        self.free_right: Optional[torch.Tensor] = None
        self.refresh(weight)

    def refresh(self, weight: torch.Tensor) -> None:
        """Recompute free singular bases from the current reference weight."""
        weight = weight.detach().float()
        if weight.ndim != 2:
            raise ValueError(f"SST only supports matrices, got shape {tuple(weight.shape)}")
        rows, cols = weight.shape
        rank = min(rows, cols)
        protected = min(int(self.rank * rank), rank)
        if protected == rank:
            self.free_left = weight.new_empty((rows, 0))
            self.free_right = weight.new_empty((cols, 0))
            return
        # full_matrices=False gives a compact basis and lets the protected
        # fraction stay cheap even for wide RWKV projections.
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        self.free_left = u[:, protected:].contiguous()
        self.free_right = vh[protected:].transpose(0, 1).contiguous()

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """Project an optimizer update onto the free-to-free block."""
        if self.free_left is None or self.free_right is None:
            return delta
        if self.free_left.shape[1] == 0 or self.free_right.shape[1] == 0:
            return torch.zeros_like(delta)
        left = self.free_left.to(device=delta.device, dtype=torch.float32)
        right = self.free_right.to(device=delta.device, dtype=torch.float32)
        delta_float = delta.float()
        projected = left @ (left.transpose(0, 1) @ delta_float @ right) @ right.transpose(0, 1)
        return projected.to(dtype=delta.dtype)


class FSDPStrategy(DistributedStrategy):
    """
    The strategy for training with FSDP.
    """

    def __init__(
        self,
        fsdp_config: FSDPConfig,
        optimizer_config: Optional[OptimizerConfig] = None,
        model_config: Optional[ModelConfig] = None,
        fsdp_strategy: str = "fsdp",
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        num_training_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert fsdp_strategy == "fsdp", f"Unsupported FSDP strategy: {fsdp_strategy}"
        self.fsdp_config = fsdp_config
        self.optimizer_config = optimizer_config
        self.model_config = model_config
        self.fsdp_strategy = fsdp_strategy
        self.max_norm = optimizer_config.max_grad_norm if optimizer_config is not None else 1.0
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.seed = seed
        self.device_mesh = None
        self.total_training_steps: Optional[int] = num_training_steps

        # When cpu_offload is off, manually offload weights/optimizer to cpu.
        self.manual_offload = not self.fsdp_config.cpu_offload
        if self.optimizer_config is not None:
            self.manual_offload_optimizer = self.optimizer_config.offload_after_step and self.manual_offload
        else:
            self.manual_offload_optimizer = False

        # LoRA related configs
        self.is_lora = self.model_config.lora.rank > 0 if self.model_config is not None else False
        self.sst = bool(getattr(optimizer_config, "sst", False)) if optimizer_config is not None else False
        if self.sst and self.is_lora:
            raise ValueError("SST requires full-parameter training and cannot be combined with LoRA or NoRA.")
        self.sst_rank = float(getattr(optimizer_config, "sst_rank", 0.99)) if optimizer_config is not None else 0.99
        self.sst_refresh_interval = (
            int(getattr(optimizer_config, "sst_refresh_interval", 0)) if optimizer_config is not None else 0
        )
        if self.sst and not 0 <= self.sst_rank < 1:
            raise ValueError(f"sst_rank must be in [0, 1), got {self.sst_rank}")
        if self.sst and self.sst_refresh_interval < 0:
            raise ValueError(f"sst_refresh_interval must be non-negative, got {self.sst_refresh_interval}")
        configured_targets = getattr(optimizer_config, "sst_target_modules", None) if optimizer_config else None
        self.sst_target_modules = tuple(configured_targets or _DEFAULT_SST_TARGET_MODULES)
        self._sst_projectors: dict[str, _SSTProjector] = {}
        self._sst_warned = False
        self._sst_step = 0

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self) -> None:
        self.set_seed(self.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        self.world_size = dist.get_world_size()

        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=self.fsdp_config.fsdp_size)

    def offload_to_cpu(self, model, optimizer, offload_optimizer=True, offload_model=True):
        """
        Offload model weights and optimizer to CPU memory.

        Only runs when cpu_offload is disabled in the FSDP config; otherwise FSDP2 handles offload natively.
        """
        if isinstance(model, HFModelWrapper):
            model = model.model

        if self.manual_offload:
            if offload_model:
                offload_fsdp2_model_to_cpu(model, empty_cache=True)

            if optimizer is not None and self.manual_offload_optimizer and offload_optimizer:
                offload_fsdp_optimizer(optimizer)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU."""
        if isinstance(model, HFModelWrapper):
            model = model.model

        if self.manual_offload:
            if backload_model:
                load_fsdp2_model_to_gpu(model)
            if optimizer is not None and self.manual_offload_optimizer and backload_optimizer:
                load_fsdp_optimizer(optimizer, torch.cuda.current_device())

        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        """Perform backward pass"""
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        grad_norm = None
        if isinstance(model, HFModelWrapper):
            model = model.model

        if self.max_norm > 0:
            # NOTE (sumanthrh): All `grad_norm`s returned here are the original grad norms before clipping.
            if FSDPModule is not None and isinstance(model, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

        # Skip update if gradient norm is not finite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                logger.warning(f"rank {rank} grad_norm is not finite: {grad_norm}")
            else:
                logger.warning(f"grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            return grad_norm

        before = {}
        if self.sst:
            before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if name in self._sst_projectors and parameter.ndim == 2
            }
        optimizer.step()
        if self.sst:
            self._sst_step += 1
            self._project_sst_updates(model, before)
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with FSDP"""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._fsdp_init_train_model(*arg))
            else:
                ret.append(self._fsdp_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _fsdp_init_model(self, model, is_train=True, is_wrapped=False):
        # Setup mixed precision
        mixed_precision_config = getattr(self.fsdp_config, "mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.param_dtype)
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.reduce_dtype)
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32

        assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
        # Recurrent RWKV chunks carry FP32 WKV state tensors between decoder
        # layers; casting forward inputs would round them to BF16 at every layer
        # boundary. Their hidden states already follow the BF16 compute params.
        rwkv_infctx = getattr(model, "is_rwkv", False) and getattr(model, "rwkv_chunk_size", 0) > 0
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=not rwkv_infctx
        )

        cpu_offload = CPUOffloadPolicy(pin_memory=True) if self.fsdp_config.cpu_offload else None

        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": self.fsdp_config.reshard_after_forward,
        }
        module = model.model if is_wrapped else model
        full_state = module.state_dict()

        # Move the entire module to meta before apply_fsdp2 so the sharded
        # DTensors are allocated directly on GPU at their final sharded size,
        # rather than first materializing the full model on each rank.
        # Reference: huggingface/accelerate fsdp2_prepare_model uses the
        # same pattern.
        #
        # Non-persistent buffers (e.g. RotaryEmbedding.inv_freq) are not in
        # state_dict, so they would be wiped by the meta swap. We snapshot
        # rank 0's values before the swap and restore them on rank 0 after,
        # so _sync_non_persistent_buffers (inside fsdp2_load_full_state_dict)
        # can broadcast real values to the other ranks.
        non_persistent_snapshot = {}
        if dist.get_rank() == 0:
            for sub_name, sub in module.named_modules():
                for bname in getattr(sub, "_non_persistent_buffers_set", set()):
                    buf = sub._buffers.get(bname)
                    if buf is not None and not buf.is_meta:
                        non_persistent_snapshot[(sub_name, bname)] = buf.detach().clone()

        module.to(torch.device("meta"))

        if dist.get_rank() == 0:
            for (sub_name, bname), buf in non_persistent_snapshot.items():
                sub = module.get_submodule(sub_name) if sub_name else module
                sub._buffers[bname] = buf

        apply_fsdp2(module, fsdp_kwargs, self.fsdp_config)
        fsdp2_load_full_state_dict(module, full_state, cpu_offload)
        return module

    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        """Initialize a model for training with FSDP"""
        is_wrapped = isinstance(model, HFModelWrapper)
        if self.sst:
            # SST needs the unsharded reference matrix to compute its bases.
            # Capture it before composable FSDP replaces parameters with shards.
            self._init_sst_projectors(model.model if is_wrapped else model)
        fsdp_module = self._fsdp_init_model(model, is_train=True, is_wrapped=is_wrapped)

        optim_config = self.optimizer_config
        if optim_config is not None:
            new_optimizer = optim.AdamW(
                fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.adam_betas,
                weight_decay=optim_config.weight_decay,
            )

            lr_scheduler = get_scheduler(
                optim_config.scheduler,
                new_optimizer,
                num_warmup_steps=optim_config.num_warmup_steps,
                num_training_steps=self.total_training_steps,
                scheduler_specific_kwargs=optim_config.scheduler_specific_kwargs,
            )
        else:
            new_optimizer = None
            lr_scheduler = None

        if is_wrapped:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model, new_optimizer, lr_scheduler

    def _init_sst_projectors(self, module: nn.Module) -> None:
        """Build SST bases for the configured dense projection parameters."""
        self._sst_projectors.clear()
        for name, parameter in module.named_parameters():
            matches_target = any(
                name == target or name.startswith(f"{target}.") or name.endswith(f".{target}") or f".{target}." in name
                for target in self.sst_target_modules
            )
            if parameter.ndim != 2 or not matches_target:
                continue
            if parameter.is_meta:
                continue
            try:
                self._sst_projectors[name] = _SSTProjector(parameter.detach(), self.sst_rank)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping SST basis for {}: {}", name, exc)
        if not self._sst_projectors:
            logger.warning("SST is enabled but no matching dense projection parameters were found.")

    def _project_sst_updates(self, model: nn.Module, before: dict[str, torch.Tensor]) -> None:
        """Replace constrained optimizer updates with their SST projections."""
        from torch.distributed.tensor import DTensor

        named_parameters = dict(model.named_parameters())
        for name, projector in self._sst_projectors.items():
            parameter = named_parameters.get(name)
            old = before.get(name)
            if parameter is None or old is None or parameter.ndim != 2 or tuple(parameter.shape) != tuple(old.shape):
                continue
            try:
                with torch.no_grad():
                    if isinstance(parameter, DTensor):
                        delta = parameter.to_local() - old.to_local()
                        projected = self._project_sst_dtensor_delta(parameter, projector, delta)
                        local = old.to_local() + projected
                        parameter.copy_(DTensor.from_local(local, parameter.device_mesh, parameter.placements))
                    else:
                        delta = parameter.detach() - old
                        parameter.copy_(old + projector.project(delta))
            except (RuntimeError, TypeError, ValueError) as exc:
                if not self._sst_warned:
                    logger.warning("SST could not project sharded parameters in this FSDP setup: {}", exc)
                    self._sst_warned = True
                continue
            if self.sst_refresh_interval and self._sst_step % self.sst_refresh_interval == 0:
                try:
                    reference = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter.detach()
                    projector.refresh(reference)
                except (RuntimeError, ValueError) as exc:
                    if not self._sst_warned:
                        logger.warning("SST basis refresh was skipped: {}", exc)
                        self._sst_warned = True

    @staticmethod
    def _project_sst_dtensor_delta(parameter, projector: _SSTProjector, delta: torch.Tensor) -> torch.Tensor:
        """Project a row- or column-sharded FSDP2 update without materializing the parameter."""
        from torch.distributed.tensor import DTensor, Replicate, Shard

        placements = parameter.placements
        shard = next((placement for placement in placements if isinstance(placement, Shard)), None)
        if shard is None:
            return projector.project(delta)

        left = projector.free_left.to(device=delta.device, dtype=torch.float32)
        right = projector.free_right.to(device=delta.device, dtype=torch.float32)
        delta_float = delta.float()
        if shard.dim == 0:
            # Rows are sharded: apply the input-side projector locally, gather
            # rows, then apply the output-side projector before redistributing.
            local = delta_float @ right @ right.transpose(0, 1)
            global_delta = DTensor.from_local(local, parameter.device_mesh, parameter.placements).full_tensor()
            global_projected = left @ (left.transpose(0, 1) @ global_delta)
        elif shard.dim == 1:
            # Columns are sharded: the output-side projection is local, then
            # the input-side projection is applied after gathering columns.
            local = left @ (left.transpose(0, 1) @ delta_float)
            global_delta = DTensor.from_local(local, parameter.device_mesh, parameter.placements).full_tensor()
            global_projected = global_delta @ right @ right.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported SST shard dimension {shard.dim}")
        replicated_placements = tuple(Replicate() for _ in parameter.placements)
        replicated = DTensor.from_local(global_projected, parameter.device_mesh, replicated_placements)
        return replicated.redistribute(parameter.device_mesh, placements).to_local().to(dtype=delta.dtype)

    def _fsdp_init_eval_model(self, model):
        """Initialize a model for evaluation with FSDP"""
        is_wrapped = isinstance(model, HFModelWrapper)
        fsdp_module = self._fsdp_init_model(model, is_train=False, is_wrapped=is_wrapped)

        if is_wrapped:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from HFModelWrapper. FSDP2 composable wrapping leaves the underlying module accessible directly."""
        if isinstance(model, HFModelWrapper):
            return self._unwrap_model(model.model)
        return model

    def _fix_fsdp_config(self, config):
        """Fix architecture names by removing FSDP prefix if present"""
        # Determine which config to save
        config_to_save = config

        # Fix architecture name by removing FSDP prefix if present
        if hasattr(config_to_save, "architectures") and config_to_save.architectures:
            # Create a copy of the config to avoid modifying the original
            config_to_save = copy.deepcopy(config_to_save)

            # Fix architecture names to remove FSDP prefix
            fixed_architectures = []
            for arch in config_to_save.architectures:
                fixed_arch = arch
                if arch.startswith("FSDP"):
                    # Remove "FSDP" prefix (for fsdp2)
                    fixed_arch = arch[len("FSDP") :]
                    self.print(f"[rank-0]: Fixed architecture name: {arch} -> {fixed_arch}")
                fixed_architectures.append(fixed_arch)

            config_to_save.architectures = fixed_architectures

        return config_to_save

    def _save_lora_adapters(self, model, ckpt_dir):
        """Save LoRA adapters in HuggingFace PEFT format"""
        from dataclasses import asdict

        from safetensors.torch import save_file

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            collect_lora_params,
            layered_summon_lora_params,
        )

        lora_save_path = os.path.join(ckpt_dir, "lora_adapter")
        peft_config = {}

        if self.is_rank_0():
            io.makedirs(lora_save_path, exist_ok=True)
            peft_config = asdict(model.peft_config.get("default", {}))
            if peft_config:
                if peft_config["task_type"] is not None:
                    peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])

        if model.peft_config["default"].modules_to_save:
            lora_params = collect_lora_params(model)
        else:
            lora_params = layered_summon_lora_params(model)

        if self.is_rank_0():
            save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
            with io.open_file(os.path.join(lora_save_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            self.print(f"[rank-0]: Saved LoRA adapter to: {lora_save_path}")

        dist.barrier()

    def save_checkpoint(
        self,
        model,
        ckpt_dir,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        """Save model checkpoint for FSDP"""
        import warnings

        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        # Extract the actual model for saving
        if isinstance(model, HFModelWrapper):
            save_model = model.model
        else:
            save_model = model

        # Define paths for saving individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        is_rwkv = (
            model.is_rwkv
            if isinstance(model, HFModelWrapper)
            else getattr(getattr(save_model, "config", None), "model_type", None) == "rwkv"
        )
        replicated_rwkv = is_rwkv and self.fsdp_config.fsdp_size == 1

        with io.local_work_dir(ckpt_dir) as work_dir:
            model_path = os.path.join(work_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
            optim_path = os.path.join(work_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
            extra_path = os.path.join(work_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # FSDP2 state_dict returns DTensors directly; no state_dict_type context needed.
                model_state_dict = save_model.state_dict()
                # Get and save optimizer state dict if optimizer is provided
                optimizer_state_dict = {}
                if optimizer is not None:
                    optimizer_state_dict = optimizer.state_dict()
                if not replicated_rwkv or rank == 0:
                    if replicated_rwkv:
                        from torch.distributed.tensor import DTensor
                        from torch.utils._pytree import tree_map

                        # fsdp_size=1: every local tensor is the complete parameter/state.
                        model_state_dict = tree_map(
                            lambda value: value.to_local() if isinstance(value, DTensor) else value,
                            model_state_dict,
                        )
                        optimizer_state_dict = tree_map(
                            lambda value: value.to_local() if isinstance(value, DTensor) else value,
                            optimizer_state_dict,
                        )
                    self.print(f"[rank-{rank}]: Saving model to {model_path}")
                    with io.open_file(model_path, "wb") as f:
                        torch.save(model_state_dict, f)
                    self.print(f"[rank-{rank}]: Saving optim to {optim_path}")
                    with io.open_file(optim_path, "wb") as f:
                        torch.save(optimizer_state_dict, f)

                # Get scheduler state dict if scheduler is provided
                lr_scheduler_state_dict = {}
                if scheduler is not None:
                    lr_scheduler_state_dict = scheduler.state_dict()

                # Create extra state dict with client state and any additional info
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "client_state": client_state,
                    "tag": tag,
                    "fsdp_strategy": self.fsdp_strategy,
                    "world_size": world_size,
                    "rank": rank,
                    "rng": self.get_rng_state(),  # Add RNG state for reproducibility
                }

                # Save extra state
                self.print(f"[rank-{rank}]: Saving extra_state to {extra_path}")
                with io.open_file(extra_path, "wb") as f:
                    torch.save(extra_state_dict, f)

                # Garbage collect temporary buffers from materializing the state dicts
                gc.collect()

            if self.is_rank_0():
                config_save_model = self._unwrap_model(model)
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(config_save_model.config, hf_dir, tokenizer)

                # Also save runtime FSDP config
                fsdp_config_path = os.path.join(work_dir, "fsdp_config.json")
                with io.open_file(fsdp_config_path, "w") as f:
                    json.dump(
                        {
                            "fsdp_strategy": self.fsdp_strategy,
                            "world_size": self.world_size,
                            "replicated_rwkv": replicated_rwkv,
                        },
                        f,
                        indent=4,
                    )

        # Save LoRA adapters if using LoRA
        if self.is_lora and hasattr(save_model, "peft_config"):
            self._save_lora_adapters(save_model, ckpt_dir)

        # Final barrier to ensure all operations complete
        dist.barrier()
        torch.cuda.synchronize()
        self.print(f"[rank-{rank}]: Checkpoint saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    ):
        """Load model checkpoint for FSDP"""
        import warnings

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract the actual model for loading
        load_model = model
        if isinstance(model, HFModelWrapper):
            load_model = model.model

        config_path = os.path.join(ckpt_dir, "fsdp_config.json")
        if io.exists(config_path):
            with io.open_file(config_path, "r") as f:
                saved_config = json.load(f)
                saved_world_size = saved_config["world_size"]
            if saved_world_size != self.world_size or saved_config.get("replicated_rwkv", False):
                return self._load_resharded_checkpoint(
                    load_model,
                    ckpt_dir,
                    saved_world_size,
                    optimizer,
                    scheduler,
                    load_module_strict,
                    load_optimizer_states,
                    load_lr_scheduler_states,
                    replicated_rwkv=saved_config.get("replicated_rwkv", False),
                )

        # Define paths for loading individual rank files
        rank = self.get_rank()
        world_size = self.world_size

        model_file = f"model_world_size_{world_size}_rank_{rank}.pt"
        optim_file = f"optim_world_size_{world_size}_rank_{rank}.pt"
        extra_file = f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        with io.local_read_files(ckpt_dir, [model_file, optim_file, extra_file]) as read_dir:
            model_path = os.path.join(read_dir, model_file)
            optim_path = os.path.join(read_dir, optim_file)
            extra_path = os.path.join(read_dir, extra_file)

            # Check if checkpoint files exist
            if not io.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            if not io.exists(extra_path):
                raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

            # Optimizer path is optional since we may not save optimizer states initially
            optim_exists = io.exists(optim_path)

            self.print(f"[rank-{rank}]: Loading model from {model_path}")
            self.print(f"[rank-{rank}]: Loading extra_state from {extra_path}")
            if optim_exists:
                self.print(f"[rank-{rank}]: Loading optim from {optim_path}")

            # Load state dictionaries from disk
            with io.open_file(model_path, "rb") as f:
                model_state_dict = torch.load(f, map_location="cpu", weights_only=False)
            with io.open_file(extra_path, "rb") as f:
                extra_state_dict = torch.load(f, map_location="cpu", weights_only=False)

            optimizer_state_dict = {}
            if optim_exists and load_optimizer_states:
                with io.open_file(optim_path, "rb") as f:
                    optimizer_state_dict = torch.load(f, map_location="cpu", weights_only=False)

        # Extract scheduler state from extra state
        lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FSDP2: load_state_dict accepts DTensors directly; no state_dict_type context needed.
            load_model.load_state_dict(model_state_dict, strict=load_module_strict)
            self.print(f"[rank-{rank}]: Successfully loaded model state dict")

            # Load optimizer state dict if optimizer object is provided and loading is requested
            if optimizer is not None and load_optimizer_states and optimizer_state_dict:
                optimizer.load_state_dict(optimizer_state_dict)
                self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

            # Load scheduler state dict if scheduler object is provided and loading is requested
            if scheduler is not None and load_lr_scheduler_states:
                scheduler.load_state_dict(lr_scheduler_state_dict)
                self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

        # Load RNG state for reproducibility
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # Wait for all ranks to finish loading
        dist.barrier()

        # Create states dict with extra information
        client_state = extra_state_dict.get("client_state", {})
        states = {
            "client_state": client_state,
            "tag": extra_state_dict.get("tag", tag),
            "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
            "world_size": extra_state_dict.get("world_size", world_size),
            "rank": extra_state_dict.get("rank", rank),
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states

    def _load_resharded_checkpoint(
        self,
        model,
        ckpt_dir,
        saved_world_size,
        optimizer,
        scheduler,
        load_module_strict,
        load_optimizer_states,
        load_lr_scheduler_states,
        replicated_rwkv=False,
    ):
        """Restore a replicated checkpoint or reshard a one-dimensional FSDP checkpoint.

        Read the full replica or reconstruct saved shards on rank 0, then use
        PyTorch's distributed state-dict APIs to distribute weights and Adam state.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
            set_optimizer_state_dict,
        )
        from torch.distributed.tensor import DTensor, Shard
        from torch.utils._pytree import tree_map

        def merge_shards(*values):
            first = values[0]
            if not isinstance(first, DTensor):
                return first
            if first.device_mesh.ndim != 1 or not isinstance(first.placements[0], Shard):
                raise ValueError("Changing GPU count requires a one-dimensional sharded FSDP checkpoint")
            dim = first.placements[0].dim
            return torch.cat([value.to_local() for value in values], dim=dim).narrow(dim, 0, first.shape[dim])

        def load_full_state(prefix):
            ranks = [0] if replicated_rwkv else range(saved_world_size)
            files = [f"{prefix}_world_size_{saved_world_size}_rank_{rank}.pt" for rank in ranks]
            with io.local_read_files(ckpt_dir, files) as read_dir:
                shards = [
                    torch.load(
                        os.path.join(read_dir, name),
                        map_location="cpu",
                        weights_only=False,
                    )
                    for name in files
                ]
            return shards[0] if replicated_rwkv else tree_map(merge_shards, *shards)

        rank = self.get_rank()
        self.print(f"Restoring FSDP checkpoint from {saved_world_size} to {self.world_size} ranks")
        full_model_state = load_full_state("model") if rank == 0 else {}
        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=load_module_strict)
        set_model_state_dict(model, full_model_state, options=options)
        del full_model_state

        if optimizer is not None and load_optimizer_states:
            full_optimizer_state = {}
            if rank == 0:
                full_optimizer_state = load_full_state("optim")
                # Native optimizer.state_dict uses integer IDs; DCP uses parameter names.
                names = {id(parameter): name for name, parameter in model.named_parameters()}
                id_to_name = {}
                for saved_group, group in zip(
                    full_optimizer_state["param_groups"],
                    optimizer.param_groups,
                    strict=True,
                ):
                    for parameter_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                        id_to_name[parameter_id] = names[id(parameter)]
                    saved_group["params"] = [id_to_name[parameter_id] for parameter_id in saved_group["params"]]
                full_optimizer_state["state"] = {
                    id_to_name[parameter_id]: state for parameter_id, state in full_optimizer_state["state"].items()
                }
            set_optimizer_state_dict(model, optimizer, full_optimizer_state, options=options)
            del full_optimizer_state

        # Rank-local random streams cannot be bitwise preserved when ranks change.
        extra_file = f"extra_state_world_size_{saved_world_size}_rank_{rank % saved_world_size}.pt"
        with io.local_read_files(ckpt_dir, [extra_file]) as read_dir:
            extra = torch.load(
                os.path.join(read_dir, extra_file),
                map_location="cpu",
                weights_only=False,
            )
        if scheduler is not None and load_lr_scheduler_states:
            scheduler.load_state_dict(extra["lr_scheduler"])
        self.load_rng_state(extra["rng"])
        dist.barrier()
        self.print(f"Checkpoint weights, optimizer and scheduler restored on {self.world_size} ranks")
        return ckpt_dir, {
            "client_state": extra.get("client_state", {}),
            "tag": extra.get("tag"),
            "fsdp_strategy": self.fsdp_strategy,
            "world_size": self.world_size,
            "rank": rank,
        }

    # TODO (erictang000): Test in multi-node setting
    def save_hf_model(
        self,
        model: Union[HFModelWrapper, nn.Module],
        output_dir: str,
        tokenizer=None,
        output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        """Save model in HuggingFace safetensors format using FSDP's full state dict gathering"""

        # Step 1: Create output directory (rank 0 only)
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
            self.print(f"[rank-0]: Created output directory: {output_dir}")

        # Step 2: Extract models - get both the model for saving metadata and the FSDP model for state dict
        model_to_save = self._unwrap_model(model)  # For saving config/metadata
        fsdp_model = model.model if isinstance(model, HFModelWrapper) else model  # For state dict collection

        # Validate that we have a proper HuggingFace model
        if not hasattr(model_to_save, "config") or not hasattr(model_to_save, "save_pretrained"):
            raise ValueError("Model must be a HuggingFace model with config and save_pretrained method")

        # Step 3: Collect full state dict on rank 0 only
        output_state_dict = fsdp2_get_full_state_dict(fsdp_model, cpu_offload=True, rank0_only=True)

        # Step 4: Save on rank 0 only
        if self.is_rank_0():
            if self.model_config is not None and self.model_config.lora.use_nora:
                from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
                    iter_merged_lora_weights,
                )

                output_state_dict = {
                    name: weight.to(output_dtype).contiguous()
                    for name, weight in iter_merged_lora_weights(model_to_save, output_state_dict)
                }
                model_to_save = model_to_save.get_base_model()
            with self._atomic_local_export_dir(output_dir) as work_dir:
                # Save the model in HuggingFace format using safetensors
                model_to_save.save_pretrained(work_dir, state_dict=output_state_dict, safe_serialization=True, **kwargs)

                # Fix and save the config
                config_to_save = self._fix_fsdp_config(model_to_save.config)
                if self.model_config is not None and self.model_config.lora.use_nora:
                    config_to_save = copy.deepcopy(config_to_save)
                    config_to_save.dtype = output_dtype
                config_to_save.save_pretrained(work_dir)

                # Save tokenizer if provided
                if tokenizer is not None:
                    tokenizer.save_pretrained(work_dir)

                # This export path bypasses save_hf_configs, so replicate its VLM handling:
                # AutoProcessor/vLLM can't reload a VLM without preprocessor_config.json.
                # Guarded so a resolution failure can't abort the save; no-op for text-only.
                try:
                    if check_is_vlm(model_to_save.config):
                        processor = get_processor(model_to_save.config.name_or_path)
                        processor.save_pretrained(work_dir)
                except Exception as e:
                    logger.warning(f"Could not save processor for '{model_to_save.config.name_or_path}'. Error: {e}")

            self.print(f"[rank-0]: Successfully saved model to {output_dir}")

        dist.barrier()

    @contextmanager
    def _atomic_local_export_dir(self, output_dir: str):
        """Rank-0 HF export dir that appears at ``output_dir`` only once complete.

        Atomic write to the local export path provides a consistent way to signal export
        finish. Ex: An asynchronous watcher can watch for ``config.json`` file to populate
        in the export directory.
        """
        if io.is_cloud_path(output_dir):
            with io.local_work_dir(output_dir) as work_dir:
                yield work_dir
            return

        parent = os.path.dirname(os.path.abspath(output_dir)) or "."
        io.makedirs(parent, exist_ok=True)
        staging_dir = tempfile.mkdtemp(prefix=".tmp-hf-export-", dir=parent)
        old_dir = None
        try:
            yield staging_dir
            # Swap via renames only (each atomic on the same filesystem) so a reader
            # never sees output_dir mid-delete: move any prior export aside, move the
            # new one in, then rmtree the old one afterwards.
            if os.path.lexists(output_dir):
                old_dir = tempfile.mktemp(prefix=".old-hf-export-", dir=parent)
                os.replace(output_dir, old_dir)
            os.replace(staging_dir, output_dir)
            staging_dir = None
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            if old_dir is not None:
                shutil.rmtree(old_dir, ignore_errors=True)
