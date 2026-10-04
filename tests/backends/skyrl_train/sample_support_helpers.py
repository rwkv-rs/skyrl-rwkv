"""Shared builders for CSR sample-support fixtures."""

from collections.abc import Sequence

import numpy as np

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_LOGPROBS_DTYPE,
    CompactSampleSupport,
)


def csr(rows, logprobs=None) -> CompactSampleSupport:
    """CSR support from a ``[tokens, top_k]`` grid whose negative entries are absent.

    ``logprobs`` are the matching sampler scores; when given they join as the support's
    sampler-logprob member column.
    """
    rows = np.asarray(rows, dtype=SAMPLE_SUPPORT_DTYPE)
    return CompactSampleSupport.from_masked_rows(rows, rows >= 0, logprobs=logprobs)


def ragged_support(trajectories: PackedTensor | Sequence, logprobs: PackedTensor | Sequence | None = None):
    """Pack one ``[tokens, top_k]`` grid per trajectory; negative entries are absent.

    A ``PackedTensor`` of fixed-width rows supplies one grid per segment. ``logprobs`` are the
    matching sampler scores, which join as the support's sampler-logprob member column.
    """
    if isinstance(trajectories, PackedTensor):
        trajectories = [trajectories.segment(index) for index in range(len(trajectories))]
    if logprobs is None:
        return PackedRaggedTensor.from_rows([csr(rows) for rows in trajectories])
    if isinstance(logprobs, PackedTensor):
        logprobs = [logprobs.segment(index) for index in range(len(logprobs))]
    grids = list(zip(trajectories, logprobs, strict=True))
    return PackedRaggedTensor.from_rows([csr(rows, scores) for rows, scores in grids])


def dense(support: CompactSampleSupport, fill: int = -1) -> np.ndarray:
    """Expand CSR rows back into a ``[tokens, top_k]`` grid padded with ``fill``."""
    return _expand(support, support.token_ids, SAMPLE_SUPPORT_DTYPE, fill)


def dense_logprobs(support: CompactSampleSupport, fill: float = float("-inf")) -> np.ndarray:
    """Expand the sampler-logprob column back into a ``[tokens, top_k]`` grid."""
    logprobs = support.sample_support_logprobs
    if logprobs is None:
        raise ValueError("support carries no sampler-logprob column")
    return _expand(support, logprobs, SAMPLE_SUPPORT_LOGPROBS_DTYPE, fill)


def _expand(support: CompactSampleSupport, members: np.ndarray, dtype, fill) -> np.ndarray:
    grid = np.full((support.num_rows, support.top_k), fill, dtype=dtype)
    columns = np.arange(members.size) - np.repeat(support.offsets[:-1], support.row_lengths)
    grid[np.repeat(np.arange(support.num_rows), support.row_lengths), columns] = members
    return grid
