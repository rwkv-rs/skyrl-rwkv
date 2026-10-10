"""RWKV trajectory diagnostics on the native trajectory logger."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from skyrl.train.utils.trajectory_logging import TrajectoryLogger


def _assistant_spans(loss_mask: Optional[List[int]]) -> List[List[int]]:
    """Return contiguous assistant-token spans as ``[start, end)`` pairs."""
    if not loss_mask:
        return []
    spans: List[List[int]] = []
    start: Optional[int] = None
    for index, mask_value in enumerate(loss_mask):
        if mask_value == 1 and start is None:
            start = index
        elif mask_value != 1 and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(loss_mask)])
    return spans


@dataclasses.dataclass
class RWKVTrajectoryLogger(TrajectoryLogger):
    """Extend native rows with termination/mask diagnostics and background writes."""

    columns: Tuple[str, ...] = (
        "step",
        "idx",
        "reward",
        "num_turns",
        "assistant_spans",
        "assistant_token_count",
        "loss_mask",
        "stop_reason",
        "ended_with_eos",
        "truncated",
        "trajectory",
    )

    def log(
        self,
        *,
        tracker: Any,
        num_samples: int,
        prompts: List[Any],
        generator_output: Dict[str, Any],
        tokenizer: Any,
        global_step: Optional[int],
        num_turns_list: Optional[List[int]] = None,
        wandb_key: str,
        include_idx: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write RWKV diagnostics incrementally on the tracker's ordered writer."""
        response_ids = generator_output.get("response_ids") or []
        if num_samples <= 0 or tracker is None or tracker.backend != "wandb" or not response_ids:
            return
        loss_masks = generator_output.get("loss_masks") or []
        rewards = generator_output.get("rewards") or []
        stop_reasons = generator_output.get("stop_reasons")
        if num_turns_list is None:
            num_turns_list = [self.count_assistant_turns(m) for m in loss_masks]
        samples = self.build_samples(
            num_samples=num_samples,
            prompts=prompts,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            num_turns_list=num_turns_list,
            tokenizer=tokenizer,
            stop_reasons=stop_reasons,
            **kwargs,
        )
        if not samples:
            return
        # `global_step` may be None (eval-only context); the table API wants
        # a numeric step.
        step = 0 if global_step is None else global_step
        # ``build_samples`` emits a tuple matching ``columns`` without the leading
        # ``step`` column; drop ``idx`` when the caller doesn't want it logged.
        columns = list(self.columns)
        if not include_idx:
            columns = [c for c in columns if c != "idx"]
            samples = [sample[1:] for sample in samples]
        # Prepend `step` to each row so rows from different calls remain
        # distinguishable in the accumulating table.
        tracker.log_samples_to_table(
            key=wandb_key,
            columns=columns,
            samples=[(step, *sample) for sample in samples],
            step=step,
            incremental=True,
            asynchronous=True,
        )

    def build_samples(
        self,
        *,
        num_samples: int,
        prompts: List[Any],
        response_ids: List[List[int]],
        rewards: List[float],
        loss_masks: List[List[int]],
        num_turns_list: List[int],
        tokenizer: Any,
        stop_reasons: Optional[List[Optional[str]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """Build the per-row tuples to be written.

        Default shape: ``(idx, reward, num_turns, assistant_spans, assistant_token_count,
        loss_mask, stop_reason, ended_with_eos, truncated, trajectory)`` (matching
        :attr:`columns` after the ``step`` column is prepended by :meth:`log`).
        ``idx`` is the position in the input arrays, *not* a sequential row
        number, so it points back to the original sample.

        Indices are chosen by :meth:`select_sample_indices`, which by default
        anchors on the min- and max-reward samples and fills the rest at
        random. Override either method to customize selection or column shape;
        ``**kwargs`` are whatever extra values the caller passed to
        :meth:`log`.
        """
        total = min(
            len(response_ids),
            len(prompts),
            len(rewards),
            len(loss_masks),
            len(num_turns_list),
        )
        # Per-token rewards arrive as lists; collapse to scalars so the
        # min/max picks and the wandb column are both well-typed.
        scalar_rewards = [float(sum(r)) if isinstance(r, list) else float(r) for r in rewards[:total]]
        indices = self.select_sample_indices(num_samples=num_samples, rewards=scalar_rewards, total=total)
        stop_reasons = stop_reasons or [None] * total
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_token_id, int):
            eos_token_id = None
        samples = []
        for i in indices:
            mask = loss_masks[i]
            response = response_ids[i]
            spans = _assistant_spans(mask)
            samples.append(
                (
                    i,
                    scalar_rewards[i],
                    num_turns_list[i],
                    spans,
                    sum(end - start for start, end in spans),
                    list(mask),
                    stop_reasons[i],
                    bool(response and eos_token_id is not None and response[-1] == eos_token_id),
                    stop_reasons[i] in {"length", "max_tokens"},
                    self.format_trajectory(prompts[i], response, mask, tokenizer, **kwargs),
                )
            )
        return samples
