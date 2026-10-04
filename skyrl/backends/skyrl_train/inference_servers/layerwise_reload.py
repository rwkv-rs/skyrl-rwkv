"""RWKV checkpoint-to-runtime weight loading for vLLM receive engines."""

import logging
import os
from collections.abc import Callable, Iterable

import torch

logger = logging.getLogger(__name__)
_RWKV_DEBUG_CHECKSUM_NAMES = frozenset(
    {
        "model.embed_tokens.weight",
        "model.embedding_norm.weight",
        "model.embedding_norm.bias",
        "model.layers.0.linear_attn.w1",
        "model.layers.0.linear_attn.w2",
        "model.layers.1.linear_attn.v1",
        "model.layers.0.mlp.value.weight",
        "model.layers.23.mlp.key.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
)


def _rwkv_debug_checksum(tensor: torch.Tensor) -> str:
    sample = tensor.detach().reshape(-1)[:4096].float()
    return (
        f"sum={sample.sum().item():.9g},"
        f"absmax={sample.abs().max().item():.9g},"
        f"ptr={tensor.data_ptr()}"
    )


def _log_rwkv_debug_weight(model: torch.nn.Module, name: str, source: torch.Tensor) -> None:
    if (
        os.environ.get("SKYRL_RWKV_DEBUG_WEIGHT_CHECKSUMS") != "1"
        or name not in _RWKV_DEBUG_CHECKSUM_NAMES
    ):
        return
    target = model.get_parameter(name)
    expected = source.T if name.endswith(".mlp.value.weight") else source
    delta = (target.detach().float() - expected.detach().float()).reshape(-1)[:4096]
    logger.info(
        "RWKV weight checksum name=%s source=(%s) target=(%s) first4096_max_abs_delta=%.9g",
        name,
        _rwkv_debug_checksum(source),
        _rwkv_debug_checksum(target),
        delta.abs().max().item(),
    )



@torch.no_grad()
def load_rwkv_checkpoint_weights(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load RWKV checkpoint tensors into its already-processed vLLM model.

    vLLM transposes the channel-mix value projection after the initial model
    load.  Reloading that checkpoint tensor through the generic layerwise path
    therefore tries to copy a ``[hidden, intermediate]`` tensor into the live
    ``[intermediate, hidden]`` kernel storage.  Keep the live storage (and its
    CUDA-graph references) and apply that one runtime-layout transform here;
    all shape-preserving tensors continue through RWKV's native loader.
    """
    loaded = set()

    def iter_regular_weights():
        for name, weight in weights:
            if name.endswith(".mlp.value.weight"):
                target = model.get_parameter(name)
                if tuple(target.shape) == tuple(weight.T.shape):
                    target.copy_(weight.T)
                elif tuple(target.shape) == tuple(weight.shape):
                    target.copy_(weight)
                else:
                    raise ValueError(
                        f"RWKV value weight shape mismatch for {name!r}: "
                        f"target={tuple(target.shape)}, source={tuple(weight.shape)}"
                    )
                _log_rwkv_debug_weight(model, name, weight)
                loaded.add(name)
            else:
                yield name, weight
                _log_rwkv_debug_weight(model, name, weight)

    loaded.update(model.load_weights(weights=iter_regular_weights()))
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

    if os.environ.get("SKYRL_RWKV_DEBUG_WEIGHT_CHECKSUMS") == "1":
        for name in (
            "model.embed_tokens.weight",
            "model.layers.0.linear_attn.w1_canonical",
            "model.layers.0.linear_attn.w2_canonical",
            "model.layers.1.linear_attn.v1_canonical",
        ):
            tensor = (
                model.get_parameter(name)
                if name == "model.embed_tokens.weight"
                else model.get_buffer(name)
            )
            logger.info(
                "RWKV runtime checksum name=%s (%s)",
                name,
                _rwkv_debug_checksum(tensor),
            )


def finalize_rwkv_runtime_weights(model: torch.nn.Module) -> None:
    """Refresh RWKV derived weights with the authoritative vLLM kernel."""
    from vllm.model_executor.models.rwkv import _load_flashrwkv2

    flashrwkv2 = _load_flashrwkv2()
    refresh_rwkv_runtime_weights(
        model,
        flashrwkv2.infer_embedding_ln0_forward_varlen,
    )


def clear_rwkv_cudagraphs() -> None:
    """Drop graphs captured against the previous RWKV runtime weights.

    RWKV has non-parameter tensors (folded embeddings and canonical low-rank
    projections) which participate in the captured forward.  The weight sync
    refreshes those tensors in place, but a graph captured before the refresh
    can also retain graph-local/static intermediates derived from the old
    values.  Re-capture after every complete update so replay observes the
    newly synchronized runtime state.
    """
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    torch.accelerator.synchronize()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()


