"""Plain-replica (DDP) training strategy.

Motivation
----------
FSDP2 wraps every parameter in a ``DTensor``.  In this environment
(torch 2.13.0+cu130, RTX PRO 6000 Blackwell) the non-fused
``torch.optim.AdamW`` and ``torch.nn.utils.clip_grad_norm_`` take a slow
per-parameter path on DTensors: measured on one card, 300 tensors x 1M
params, plain tensors step in 0.105 s while the identical DTensor step
takes 1.835 s (17x); clip shows the same effect.  Scaled to the 1B RWKV
model this reproduces the ~11.4 s optimizer step seen in every run --
the work is on the GPU, so "move the optimizer to the GPU" cannot fix
it while the parameters stay DTensors (even a single-rank
``Replicate`` mesh still wraps parameters in DTensors, which is why
dp=1 and dp=8 show the same ~11-12 s).

For a 1B model that fits comfortably on one 96GB card, a full replica
per rank (classic DDP) is cheaper in every way that matters:

* optimizer + clip run on plain tensors -> fast foreach kernels;
* weight sync to the colocated vLLM engine is a local ``state_dict()``
  read per rank (no all-gather collectives);
* HF export is a local rank-0 save (no fsdp2 full-state gather);
* checkpoints are plain full replicas: resumable at any world size, and
  the ``replicated_rwkv`` marker lets the FSDP loader resume DDP
  checkpoints as well.

Cost: ~10GiB more static memory per card than FSDP sharding --
irrelevant on 96GB cards for a 1B model.
"""

import gc
import json
import os
import random
import re
import shutil
import tempfile
from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from loguru import logger
from torch import distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parametrize
from transformers.trainer import get_scheduler

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig
from skyrl.utils.tok import check_is_vlm, get_processor


class _RWKVComputeDtype(nn.Module):
    def forward(self, parameter):
        return parameter.to(torch.bfloat16)


def _save_rwkv_master_state(module, state_dict, prefix, local_metadata):
    for internal_name, public_name in module._rwkv_master_state_keys.items():
        internal_key = prefix + internal_name
        if internal_key in state_dict:
            state_dict[prefix + public_name] = state_dict.pop(internal_key)


def _load_rwkv_master_state(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    for internal_name, public_name in module._rwkv_master_state_keys.items():
        public_key = prefix + public_name
        if public_key in state_dict:
            state_dict[prefix + internal_name] = state_dict.pop(public_key)


def _prepare_rwkv_mixed_precision(module):
    """Expose BF16 compute views while keeping canonical FP32 optimizer parameters and checkpoints."""
    if hasattr(module, "_rwkv_master_parameters"):
        return module._rwkv_master_parameters
    master_parameters = list(module.parameters())
    state_keys = {}
    for module_name, submodule in list(module.named_modules()):
        for name, parameter in list(submodule.named_parameters(recurse=False)):
            if parameter.dtype != torch.float32:
                continue
            prefix = f"{module_name}." if module_name else ""
            state_keys[f"{prefix}parametrizations.{name}.original"] = prefix + name
            parametrize.register_parametrization(submodule, name, _RWKVComputeDtype(), unsafe=True)
    module._rwkv_master_parameters = master_parameters
    module._rwkv_master_state_keys = state_keys
    module.register_state_dict_post_hook(_save_rwkv_master_state)
    module.register_load_state_dict_pre_hook(_load_rwkv_master_state)
    return master_parameters


class DDPStrategy(DistributedStrategy):
    """Strategy that keeps a full model replica on every rank (classic DDP)."""

    def __init__(
        self,
        fsdp_config: Optional[FSDPConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        model_config: Optional[ModelConfig] = None,
        ddp_strategy: str = "ddp",
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        num_training_steps: Optional[int] = None,
        bf16: bool = True,
    ) -> None:
        super().__init__()
        assert ddp_strategy == "ddp", f"Unsupported DDP strategy: {ddp_strategy}"
        # Accepted for constructor symmetry with the FSDP workers; nothing
        # in it is used (there is no sharding).
        self.fsdp_config = fsdp_config
        self.optimizer_config = optimizer_config
        self.model_config = model_config
        self.bf16 = bf16
        # Recorded in checkpoints so loaders can identify the strategy.
        self.fsdp_strategy = ddp_strategy
        self.max_norm = optimizer_config.max_grad_norm if optimizer_config is not None else 1.0
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.seed = seed
        # The worker builds its own dp device mesh for dispatch/metrics
        # (Worker.init_worker_process_group); the strategy does not need one.
        self.device_mesh = None
        self.total_training_steps: Optional[int] = num_training_steps

        # No native FSDP-style offload exists; weights/optimizer are moved
        # manually around rollouts (same contract as FSDP's manual path).
        self.manual_offload = True
        self.manual_offload_optimizer = (
            bool(optimizer_config.offload_after_step) if optimizer_config is not None else False
        )
        self.is_lora = self.model_config.lora.rank > 0 if self.model_config is not None else False

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

        self.world_size = dist.get_world_size()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _unwrap_ddp(self, module: nn.Module) -> nn.Module:
        if isinstance(module, DDP):
            return module.module
        return module

    def _model_module(self, model) -> nn.Module:
        """The bare (non-DDP) module that owns the parameters."""
        module = model.model if isinstance(model, HFModelWrapper) else model
        return self._unwrap_ddp(module)

    def _offload_optimizer_to(self, optimizer, device) -> None:
        if optimizer is None or not optimizer.state:
            return
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

    # ------------------------------------------------------------------
    # offload / backload (colocated mode: free GPU memory for the engine)
    # ------------------------------------------------------------------
    def offload_to_cpu(self, model, optimizer, offload_optimizer=True, offload_model=True):
        """Offload the full replica (and, if configured, optimizer state) to CPU."""
        if self.manual_offload:
            if offload_model:
                with torch.no_grad():
                    self._model_module(model).to("cpu", non_blocking=True)
            if self.manual_offload_optimizer and offload_optimizer:
                with torch.no_grad():
                    self._offload_optimizer_to(optimizer, "cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, backload_optimizer=True, backload_model=True):
        """Reload the full replica back onto the local GPU."""
        if self.manual_offload:
            device = torch.cuda.current_device()
            if backload_model:
                with torch.no_grad():
                    self._model_module(model).to(device, non_blocking=True)
            if self.manual_offload_optimizer and backload_optimizer:
                with torch.no_grad():
                    self._offload_optimizer_to(optimizer, device)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # train step
    # ------------------------------------------------------------------
    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        # DDP buckets the gradient all-reduce inside this backward call.
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Clip (plain-tensor foreach path) and step the local replica optimizer."""
        module = self._model_module(model)
        grad_norm = None
        if self.max_norm > 0:
            # Plain tensors: clip_grad_norm_ uses the fused foreach path.
            # (On FSDP2 DTensors the same call takes the slow per-parameter
            # path -- see the module docstring.)
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=self.max_norm)

        # Skip update if gradient norm is not finite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            rank = dist.get_rank() if dist.is_initialized() else -1
            logger.warning(f"rank {rank} grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            return grad_norm

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    # ------------------------------------------------------------------
    # model / optimizer preparation
    # ------------------------------------------------------------------
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with DDP (no sharding)."""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._init_train_model(*arg))
            else:
                ret.append(self._init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _maybe_wrap_ddp(self, module: nn.Module) -> nn.Module:
        if self.world_size > 1 and dist.is_initialized():
            # broadcast_buffers=False: buffers are deterministically
            # initialized from the shared seed; avoid one extra collective
            # per forward.
            return DDP(module, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        return module

    def _init_train_model(self, model, optimizer, scheduler):
        """Initialize a model for training: full local replica + DDP + AdamW."""
        is_wrapped = isinstance(model, HFModelWrapper)
        module = model.model if is_wrapped else model
        # Full replica loads straight onto the local GPU (a 1B model is a
        # few GB per rank); no meta-init dance is needed because nothing
        # is sharded.
        module = module.to(torch.cuda.current_device())
        if is_wrapped and model.is_rwkv and self.bf16 and not self.is_lora:
            optimizer_parameters = _prepare_rwkv_mixed_precision(module)
        else:
            optimizer_parameters = list(module.parameters())
        wrapped = self._maybe_wrap_ddp(module)

        optim_config = self.optimizer_config
        if optim_config is not None:
            # Plain-tensor parameters: AdamW takes the fast foreach kernel
            # path (the DTensor path in this torch build is ~17x slower).
            new_optimizer = optim.AdamW(
                optimizer_parameters,
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
            model.model = wrapped
        else:
            model = wrapped

        return model, new_optimizer, lr_scheduler

    def _init_eval_model(self, model):
        """Inference-only replica: plain module (no gradient sync to save)."""
        is_wrapped = isinstance(model, HFModelWrapper)
        module = model.model if is_wrapped else model
        module = module.to(torch.cuda.current_device())
        if is_wrapped:
            model.model = module
        else:
            model = module
        return module

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
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
        """Save a full replica (rank 0) plus HF metadata next to it."""
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        save_model = self._model_module(model)  # plain inner module
        rank = self.get_rank()
        world_size = self.world_size

        with io.local_work_dir(ckpt_dir) as work_dir:
            if self.is_rank_0():
                model_path = os.path.join(work_dir, f"model_world_size_{world_size}_rank_0.pt")
                optim_path = os.path.join(work_dir, f"optim_world_size_{world_size}_rank_0.pt")
                extra_path = os.path.join(work_dir, f"extra_state_world_size_{world_size}_rank_0.pt")

                model_state_dict = save_model.state_dict()
                optimizer_state_dict = optimizer.state_dict() if optimizer is not None else {}
                self.print(f"[rank-0]: Saving DDP replica to {model_path}")
                with io.open_file(model_path, "wb") as f:
                    torch.save(model_state_dict, f)
                self.print(f"[rank-0]: Saving optim to {optim_path}")
                with io.open_file(optim_path, "wb") as f:
                    torch.save(optimizer_state_dict, f)

                lr_scheduler_state_dict = scheduler.state_dict() if scheduler is not None else {}
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "client_state": client_state,
                    "tag": tag,
                    "fsdp_strategy": self.fsdp_strategy,
                    "world_size": world_size,
                    "rank": 0,
                    "rng": self.get_rng_state(),
                }
                with io.open_file(extra_path, "wb") as f:
                    torch.save(extra_state_dict, f)

                gc.collect()

                # HF metadata next to the replica (rank 0), like FSDP does.
                if hasattr(save_model, "config"):
                    hf_dir = os.path.join(work_dir, "huggingface")
                    self.save_hf_configs(save_model.config, hf_dir, tokenizer)

                # Marker: this checkpoint is a full replica.  It makes the
                # FSDP loader take its replicated-resume path, so FSDP runs
                # can also pick DDP checkpoints back up (and vice versa for
                # fsdp_size=1 checkpoints).
                with io.open_file(os.path.join(work_dir, "fsdp_config.json"), "w") as f:
                    json.dump(
                        {
                            "fsdp_strategy": self.fsdp_strategy,
                            "world_size": world_size,
                            "replicated_rwkv": True,
                        },
                        f,
                        indent=4,
                    )

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
        """Load a full-replica checkpoint; works at any world size."""
        import warnings

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        if io.is_cloud_path(ckpt_dir):
            raise ValueError("DDP resume currently supports local checkpoint directories only.")

        load_model = self._model_module(model)
        rank = self.get_rank()
        world_size = self.world_size

        config_path = os.path.join(ckpt_dir, "fsdp_config.json")
        if io.exists(config_path):
            with io.open_file(config_path, "r") as f:
                saved_config = json.load(f)
            saved_world_size = saved_config.get("world_size", world_size)
            replicated = saved_config.get("replicated_rwkv", False)
            saved_strategy = saved_config.get("fsdp_strategy", "ddp")
            if saved_strategy == "fsdp" and not replicated and saved_world_size > 1:
                raise ValueError(
                    f"Checkpoint at {ckpt_dir} is a multi-rank sharded FSDP checkpoint; "
                    "the DDP strategy can only resume full-replica checkpoints "
                    "(DDP runs, or FSDP runs with fsdp_size=1)."
                )

        def _parse_model_file(name):
            m = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", name)
            if m is None:
                return None
            return int(m.group(1)), int(m.group(2))

        try:
            entries = [(_parse_model_file(name), name) for name in os.listdir(ckpt_dir)]
        except OSError as exc:
            raise FileNotFoundError(f"Cannot list checkpoint directory {ckpt_dir}: {exc}")
        candidates = [(ws_rank, name) for ws_rank, name in entries if ws_rank is not None]
        if not candidates:
            raise FileNotFoundError(f"No DDP model checkpoint found in {ckpt_dir}")

        # Prefer rank 0 of the same world size, then any rank 0, then anything.
        def _score(item):
            (ws, rk), _name = item
            return (rk != 0, ws != world_size, -ws)

        _, model_file = min(candidates, key=_score)
        extra_file = model_file.replace("model_world_size_", "extra_state_world_size_")
        optim_file = model_file.replace("model_world_size_", "optim_world_size_")

        states = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                with io.local_read_files(ckpt_dir, [model_file, optim_file, extra_file]) as read_dir:
                    model_path = os.path.join(read_dir, model_file)
                    extra_path = os.path.join(read_dir, extra_file)
                    if not io.exists(model_path):
                        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
                    if not io.exists(extra_path):
                        raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

                    with io.open_file(model_path, "rb") as f:
                        model_state_dict = torch.load(f, map_location="cpu", weights_only=False)
                    with io.open_file(extra_path, "rb") as f:
                        extra_state_dict = torch.load(f, map_location="cpu", weights_only=False)

                    optimizer_state_dict = {}
                    optim_path = os.path.join(read_dir, optim_file)
                    if load_optimizer_states and io.exists(optim_path):
                        with io.open_file(optim_path, "rb") as f:
                            optimizer_state_dict = torch.load(f, map_location="cpu", weights_only=False)

                lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    load_model.load_state_dict(model_state_dict, strict=load_module_strict)
                    self.print(f"[rank-{rank}]: Successfully loaded DDP model state dict")

                    if optimizer is not None and load_optimizer_states and optimizer_state_dict:
                        optimizer.load_state_dict(optimizer_state_dict)
                        self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

                    if scheduler is not None and load_lr_scheduler_states:
                        scheduler.load_state_dict(lr_scheduler_state_dict)
                        self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

                if "rng" in extra_state_dict:
                    self.load_rng_state(extra_state_dict["rng"])
                states = {
                    "client_state": extra_state_dict.get("client_state", {}),
                    "tag": extra_state_dict.get("tag", tag),
                    "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
                }
                del model_state_dict, extra_state_dict, optimizer_state_dict
                gc.collect()
            dist.barrier()

        if states is None:
            raise RuntimeError(f"Rank {rank} did not participate in checkpoint loading")

        states = {
            **states,
            "world_size": world_size,
            "rank": rank,
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states

    # ------------------------------------------------------------------
    # HF export
    # ------------------------------------------------------------------
    def save_hf_model(
        self,
        model: Union[HFModelWrapper, nn.Module],
        output_dir: str,
        tokenizer=None,
        output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        """Save a local full replica in HuggingFace safetensors format (rank 0)."""

        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
            self.print(f"[rank-0]: Created output directory: {output_dir}")

        model_to_save = self._unwrap_model(model)  # plain inner HF module

        if not hasattr(model_to_save, "config") or not hasattr(model_to_save, "save_pretrained"):
            raise ValueError("Model must be a HuggingFace model with config and save_pretrained methods")

        if self.is_rank_0():
            # Full replica: a local state_dict read, no gather collectives.
            output_state_dict = model_to_save.state_dict()
            with self._atomic_local_export_dir(output_dir) as work_dir:
                model_to_save.save_pretrained(work_dir, state_dict=output_state_dict, safe_serialization=True, **kwargs)

                # Plain config (no FSDP architecture prefix to fix).
                model_to_save.config.save_pretrained(work_dir)

                if tokenizer is not None:
                    tokenizer.save_pretrained(work_dir)

                # Replicate FSDPStrategy's VLM handling so AutoProcessor can
                # reload the export; guarded so it can never abort the save.
                try:
                    if check_is_vlm(model_to_save.config):
                        processor = get_processor(model_to_save.config.name_or_path)
                        processor.save_pretrained(work_dir)
                except Exception as e:
                    logger.warning(f"Could not save processor for '{model_to_save.config.name_or_path}'. Error: {e}")

            self.print(f"[rank-0]: Successfully saved model to {output_dir}")

        dist.barrier()

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap HFModelWrapper / DDP down to the plain inner HF module."""
        if isinstance(model, HFModelWrapper):
            return self._unwrap_model(model.model)
        if isinstance(model, DDP):
            return self._unwrap_model(model.module)
        return model

    @contextmanager
    def _atomic_local_export_dir(self, output_dir: str):
        """Rank-0 HF export dir that appears at ``output_dir`` only once complete.

        Same atomic staging as FSDPStrategy._atomic_local_export_dir so an
        external watcher can key off ``config.json`` populating.
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
            # Swap via renames only (each atomic on the same filesystem) so a
            # reader never sees output_dir mid-delete.
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
