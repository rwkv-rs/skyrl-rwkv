"""RWKV checkpoint-to-runtime weight loading for vLLM receive engines."""

from collections.abc import Callable

import torch


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


