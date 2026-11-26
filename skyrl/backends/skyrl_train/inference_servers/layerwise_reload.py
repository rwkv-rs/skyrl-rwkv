"""RWKV checkpoint-to-runtime weight loading for vLLM receive engines."""

from collections.abc import Iterable

import torch


@torch.no_grad()
def load_rwkv_checkpoint_weights(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load a checkpoint chunk with vLLM's RWKV tensor loaders."""
    loaded = model.load_weights(weights=weights)
    # SkyRL's receive engines stream into live storage without vLLM's reload
    # context. Record the names so vLLM can fold a newly loaded embedding once.
    model.record_checkpoint_weights_loaded(loaded)
    return loaded


def finalize_rwkv_runtime_weights(model: torch.nn.Module) -> None:
    """Run vLLM's RWKV post-load hooks after all checkpoint chunks arrive."""
    from vllm.model_executor.model_loader.weight_utils import finalize_model_weights

    finalize_model_weights(model)


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
