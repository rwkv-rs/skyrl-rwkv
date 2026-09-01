"""Shared vLLM layerwise-reload lifecycle for SkyRL's vLLM worker-extension classes.

Provides `LayerwiseReloadWorkerMixin`, the start/finish bracket that
`new_inference_worker_wrap.NewInferenceWorkerWrap` uses to run vLLM's layerwise
reload once per weight sync rather than once per chunk.
"""

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def get_numel_loaded(weight_loader: Callable, args: inspect.BoundArguments) -> tuple[int, object]:
    """
    Determine how many elements would be loaded by a weight loader call.

    Args:
        weight_loader: used to load weights
        args: bound arguments to weight loader

    Returns:
        number of elements loaded by the weight loader, the return value of the
        weight loader
    """
    # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
    from vllm.model_executor.model_loader.reload.meta import CopyCounter

    with CopyCounter() as counter:
        return_value = weight_loader(*args.args, **args.kwargs)

    # A weight loader fills a single destination parameter, so the number of
    # loaded elements is at most that parameter's size. Some loaders copy into
    # the parameter more than once -- e.g. ``composed_weight_loader`` runs an
    # in-place post-load transform (``param.copy_(fn(param))``) on top of the
    # initial copy -- which would make CopyCounter report twice the parameter
    # size. Over-counting inflates the layer's loaded-element total and can
    # finalize the layer before every parameter is loaded, silently dropping
    # the trailing parameter(s) (e.g. Mamba ``mixer.D``). Cap the count at the
    # destination size to keep the per-layer accounting correct.
    numel = counter.copied_numel
    param = args.arguments.get("param", None)
    if isinstance(param, torch.Tensor):
        numel = min(numel, param.numel())
    return numel, return_value


def patch_numel_loaded():
    # vLLM's layerwise reload binds get_numel_loaded at import time
    # (`from .meta import get_numel_loaded`), so its call site at
    # layerwise.py uses the `layerwise` module's own binding. Rebind that
    # attribute to our patched version to substitute the symbol.
    from vllm.model_executor.model_loader.reload import layerwise as _layerwise
    from vllm.model_executor.model_loader.reload import meta as _meta

    _layerwise.get_numel_loaded = get_numel_loaded
    _meta.get_numel_loaded = get_numel_loaded


@torch.no_grad()
def load_rwkv_checkpoint_weights(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load RWKV checkpoint tensors into its already-processed vLLM model.

    vLLM transposes the channel-mix value projection after the initial model
    load.  Reloading that checkpoint tensor through the generic layerwise path
    therefore tries to copy a ``[hidden, intermediate]`` tensor into the live
    ``[intermediate, hidden]`` kernel storage.  Keep the live storage (and its
    CUDA-graph references) and apply that one runtime-layout transform here;
    all shape-preserving tensors continue through RWKV's native loader.
    """
    regular_weights = []
    loaded = set()
    for name, weight in weights:
        if name.endswith(".mlp.value.weight"):
            model.get_parameter(name).copy_(weight.T)
            loaded.add(name)
        else:
            regular_weights.append((name, weight))

    if regular_weights:
        loaded.update(model.load_weights(weights=regular_weights))
    return loaded


@torch.no_grad()
def refresh_rwkv_runtime_weights(model: torch.nn.Module, fold_embedding: Callable) -> None:
    """Refresh RWKV runtime-only tensors after checkpoint weights are loaded.

    This mirrors ``RwkvModel.process_weights_after_loading`` while copying into
    the existing parameter/buffer storage so captured CUDA graphs keep valid
    addresses.
    """
    rwkv_model = model.model
    embedding = rwkv_model.embed_tokens.weight
    embedding_norm = rwkv_model.embedding_norm
    for start in range(0, embedding.shape[0], 4096):
        end = min(start + 4096, embedding.shape[0])
        folded = fold_embedding(
            embedding[start:end].to(torch.bfloat16).contiguous(),
            embedding_norm.weight,
            embedding_norm.bias,
            eps=rwkv_model.config.layer_norm_epsilon,
        )
        embedding[start:end].copy_(folded)
    rwkv_model._embedding_norm_folded = True

    for layer_idx, layer in enumerate(
        rwkv_model.layers[rwkv_model.start_layer : rwkv_model.end_layer],
        start=rwkv_model.start_layer,
    ):
        attention = layer.linear_attn
        for runtime_name, checkpoint_name in (
            ("w1_canonical", "w1"),
            ("a1_canonical", "a1"),
            ("g1_canonical", "g1"),
            ("w2_canonical", "w2"),
            ("a2_canonical", "a2"),
            ("g2_canonical", "g2"),
        ):
            getattr(attention, runtime_name).copy_(getattr(attention, checkpoint_name).T)

        if layer_idx == 0:
            for name in (
                "v1_canonical",
                "v2_canonical",
                "layer_zero_v0",
                "layer_zero_v1_runtime",
                "layer_zero_v2_runtime",
            ):
                getattr(attention, name).zero_()
        else:
            attention.v1_canonical.copy_(attention.v1.T)
            attention.v2_canonical.copy_(attention.v2.T)


def finalize_rwkv_runtime_weights(model: torch.nn.Module) -> None:
    """Refresh RWKV derived weights with the authoritative vLLM kernel."""
    from vllm.model_executor.models.rwkv import _load_flashrwkv2

    flashrwkv2 = _load_flashrwkv2()
    refresh_rwkv_runtime_weights(
        model,
        flashrwkv2.infer_embedding_ln0_forward_varlen,
    )


_PATCHED_LAYERWISE_NUMEL_LOADED = False


def _empty_cuda_cache_rocm() -> None:
    """Release unused ROCm cached blocks after full-weight sync."""
    is_rocm = torch.version.hip is not None
    if not torch.cuda.is_available() or not is_rocm:
        return

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


class LayerwiseReloadWorkerMixin:
    """Bracket a multi-chunk weight sync with one vLLM layerwise-reload init/finalize.

    `skyrl_start_weight_update` initializes the layerwise reload once; each chunk then loads
    its weights raw; `skyrl_finish_weight_update` finalizes once over the whole weight set.
    A per-chunk `reload_weights` is the wrong approach: it re-finalizes on every call
    and restores layers absent from that chunk, corrupting a multi-chunk sync.
    """

    vllm_config: "VllmConfig"
    model_runner: "GPUModelRunner"
    model_config: "ModelConfig"
    device: torch.device

    # NOTE: named with a `skyrl_` prefix to avoid colliding with vLLM's own
    # Worker.start_weight_update / finish_weight_update (added in vllm-project/vllm
    # #39212, merge e3b65a5, shipped in vLLM 0.22.0+). vLLM injects the
    # worker-extension class as a *base* of Worker and asserts the extension
    # defines no attribute already present on Worker, so same-named methods abort
    # engine init. The skyrl_-prefixed variants keep SkyRL's IPC weight-sync path
    # (and the MoE set_current_vllm_config wrapping) intact alongside vLLM's native API.
    def skyrl_start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "skyrl_start_weight_update called while a weight update is "
                "already active. Call skyrl_finish_weight_update first."
            )
        if getattr(self, "_weight_update_active", False):
            raise RuntimeError("vLLM native weight update is already active. Call finish_weight_update first.")

        # Ensure the get_numel_loaded patch is in effect before layerwise
        # reload runs.
        global _PATCHED_LAYERWISE_NUMEL_LOADED
        if not _PATCHED_LAYERWISE_NUMEL_LOADED:
            # use patched version, based on https://github.com/vllm-project/vllm/pull/44814
            patch_numel_loaded()
            _PATCHED_LAYERWISE_NUMEL_LOADED = True

        model = self.model_runner.model
        self._skyrl_rwkv_checkpoint_update = is_checkpoint_format and model.config.model_type == "rwkv"
        if is_checkpoint_format and not self._skyrl_rwkv_checkpoint_update:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True
        # vLLM's native /update_weights endpoint checks these flags before
        # calling the configured WeightTransferEngine. Mirroring them lets
        # SkyRL keep its patched layerwise start/finish while using native
        # update_weights for transports such as checkpoint-delta.
        self._is_checkpoint_format = is_checkpoint_format
        self._weight_update_active = True

    def skyrl_finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_ipc calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_start_weight_update must be called before skyrl_finish_weight_update.")

        # The sharded_rdt engine defers its GPU post-processing (scatter/quant/
        # kernel-copy) to background threads during update, so drain it here —
        # before finalize, which needs every layer fully loaded + reset. No-op
        # for the ipc/nccl engines (they process synchronously per chunk).
        engine = getattr(self, "weight_transfer_engine", None)
        if engine is not None and getattr(engine, "defers_processing", False):
            drain_pending = getattr(engine, "drain_pending", None)
            if drain_pending is not None:
                drain_pending()

        if self._skyrl_is_checkpoint_format:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                model = self.model_runner.model
                if self._skyrl_rwkv_checkpoint_update:
                    finalize_rwkv_runtime_weights(model)
                else:
                    finalize_layerwise_reload(model, self.model_config)

        self._skyrl_weight_update_active = False
        self._skyrl_rwkv_checkpoint_update = False
        self._skyrl_is_checkpoint_format = True
        self._weight_update_active = False
        self._is_checkpoint_format = True
        _empty_cuda_cache_rocm()
