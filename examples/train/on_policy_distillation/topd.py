"""Token returns and group advantages from TOP-D, arXiv:2607.04751, Eqs. (7–8).

Uses SkyRL's contiguous prompt groups and right-aligned response tensors.
The STOPD combination uses a randomized telescoping sum of group objectives.
It only needs observed prefixes, including for future returns and group moments.
"""

from dataclasses import dataclass
import copy
import math

import torch

from skyrl.train.config import BaseConfig
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, pad_training_input_batch
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss, reduce_loss
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean, safe_exp_delta


@dataclass
class TOPDConfig(BaseConfig):
    enabled: bool = False
    """Enable future returns, token group normalization, and native PPO updates."""
    epsilon: float = 1e-6
    """Numerical stabilizer added to the population standard deviation."""
    balance_batch: bool = False
    """Group similar lengths and balance padded microbatch work within each PPO mini-batch."""

    def __post_init__(self):
        if not 0 < self.epsilon < float("inf"):
            raise ValueError("TOP-D epsilon must be finite and positive")


def balance_training_input(data, dp_size, micro_batch_size):
    """Preserve prompt groups and mini-batch membership while reducing padding.

    Prototype: RWKV SFT's LengthGroupedSampler and native MeshDispatch chunks.
    Sort whole prompt groups, form equal-size units aligned to both groups and
    microbatches, then greedily place costly units on the least-loaded DP rank.
    Every rank keeps the same number of rows and native collective calls.
    """
    groups = data.metadata["policy_prompt_boundaries"]
    boundaries = data.metadata["policy_mini_batch_boundaries"]
    group_size = groups[0][1] - groups[0][0]
    unit_size = math.lcm(group_size, micro_batch_size)
    real_rows = boundaries[-1][1]
    padding_sizes = [(-(end - start)) % (dp_size * unit_size) for start, end in boundaries]
    padding_size = sum(padding_sizes)
    if padding_size:
        # Reuse native zero-loss padding, but place complete microbatches on
        # each DP rank. A one-token input avoids computing copied long prompts.
        source = TrainingInputBatch(
            {key: value[:real_rows] if value is not None else None for key, value in data.items()}
        )
        source.metadata = dict(data.metadata)
        source.metadata["uids"] = data.metadata["uids"][:real_rows]
        data = pad_training_input_batch(source, padding_size)
        data["sequences"][real_rows:] = 0
        data["attention_mask"][real_rows:] = 0
        data["attention_mask"][real_rows:, -1] = 1
        data["response_mask"][real_rows:] = 0
    lengths = data["attention_mask"].sum(-1).tolist()
    order, balanced_boundaries = [], []
    padding_start = real_rows
    before, after, before_makespan, after_makespan = 0, 0, 0, 0

    def cost(indices):
        return sum(
            len(indices[i : i + micro_batch_size]) * max(lengths[j] for j in indices[i : i + micro_batch_size])
            for i in range(0, len(indices), micro_batch_size)
        )

    for (start, end), size in zip(boundaries, padding_sizes, strict=True):
        prompt_groups = [list(range(a, b)) for a, b in groups if start <= a < end]
        prompt_groups.sort(key=lambda group: max(lengths[i] for i in group))
        sorted_indices = [i for group in prompt_groups for i in group]
        sorted_indices.extend(range(padding_start, padding_start + size))
        padding_start += size
        units = [sorted_indices[i : i + unit_size] for i in range(0, len(sorted_indices), unit_size)]
        units.sort(key=cost, reverse=True)
        rank_rows = len(sorted_indices) // dp_size
        slots = rank_rows // unit_size
        assigned = [[] for _ in range(dp_size)]
        loads = [0] * dp_size
        for unit in units:
            rank = min((r for r in range(dp_size) if len(assigned[r]) < slots), key=lambda r: loads[r])
            assigned[rank].append(unit)
            loads[rank] += cost(unit)
        balanced_start = len(order)
        order.extend(i for rank in assigned for unit in rank for i in unit)
        balanced_boundaries.append((balanced_start, len(order)))
        original_rank_rows = math.ceil((end - start) / dp_size)
        original_costs = [
            cost(list(range(a, min(a + original_rank_rows, end)))) for a in range(start, end, original_rank_rows)
        ]
        before += sum(original_costs)
        after += sum(loads)
        before_makespan += max(original_costs)
        after_makespan += max(loads)
    indices = torch.tensor(order, dtype=torch.long, device=data["sequences"].device)
    result = TrainingInputBatch({key: value[indices] if value is not None else None for key, value in data.items()})
    result.metadata = dict(data.metadata)
    result.metadata["uids"] = [data.metadata["uids"][i] for i in order]
    result.metadata["batch_permutation"] = order
    result.metadata["policy_mini_batch_boundaries"] = balanced_boundaries
    group_lengths = {start: end - start for start, end in groups}
    result.metadata["policy_prompt_boundaries"] = [
        (row, row + group_lengths[index]) for row, index in enumerate(order) if index in group_lengths
    ]
    # Synthetic rows are interspersed; their zero loss/response masks exclude
    # them without treating them as additional prompts or trailing real rows.
    result.metadata["pad_size"] = 0
    return result, {
        "perf/dense_tokens_before_balance": before,
        "perf/dense_tokens_after_balance": after,
        "perf/policy_rank_max_tokens_before_balance": before_makespan,
        "perf/policy_rank_max_tokens_after_balance": after_makespan,
        "perf/dp_padding_sequences": padding_size,
    }


@torch.no_grad()
def compute_topd_advantage(token_level_rewards, loss_mask, prompt_boundaries, epsilon=1e-6):
    """Immediate reward plus the mean strictly future reward, then token z-score.

    A final token has zero future reward. Masked and synthetic padding tokens
    contribute neither to future returns nor to prompt-group moments. Moments
    pool tokens across all responses to each prompt, not sequence means.
    """
    mask = loss_mask.bool()
    rewards = token_level_rewards.float().masked_fill(~mask, 0)
    suffix = rewards.flip(-1).cumsum(-1).flip(-1) - rewards
    remaining = mask.flip(-1).long().cumsum(-1).flip(-1) - mask.long()
    returns = (rewards + suffix / remaining.clamp_min(1)).masked_fill(~mask, 0)
    advantages = torch.zeros_like(returns)
    counts = []
    for start, end in prompt_boundaries:
        group_mask = mask[start:end]
        values = returns[start:end][group_mask]
        count = values.numel()
        counts.append((start, end, count))
        if count:
            std, mean = torch.std_mean(values, correction=0)
            advantages[start:end] = ((returns[start:end] - mean) / (std + epsilon)).masked_fill(~group_mask, 0)
    return advantages, returns, counts


@torch.no_grad()
def compute_topd_stopd_advantage(rewards, loss_mask, prompt_boundaries, max_len, block, truncating, epsilon=1e-6):
    """Unbiased finite telescoping estimator of the complete TOP-D objective.

    Responses to one prompt share a STOPD budget. F_j is the clipped TOP-D
    objective with returns and group moments recomputed at horizon j * block.
    With q_j=P(B >= (j-1)*block+1), E[sum_{j<=J}(F_j-F_{j-1})/q_j]=F_M.
    Signed level coefficients must stay OUTSIDE PPO clipping. Two channels
    accumulate their positive/negative advantage contributions separately;
    this preserves the clipped objective without retaining every level.
    See Rhee & Glynn (2015), https://doi.org/10.1287/opre.2015.1404.
    """
    mask = loss_mask.bool()
    coefficients = torch.zeros((*rewards.shape, 2), dtype=torch.float64, device=rewards.device)
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    counts = []
    for start, end in prompt_boundaries:
        group_mask = mask[start:end]
        counts.append((start, end, int(group_mask.sum())))
        columns = group_mask.any(0).nonzero().flatten()
        if not len(columns):
            continue
        left, right = int(columns[0]), int(columns[-1]) + 1
        group_mask = group_mask[:, left:right]
        group_rewards = rewards[start:end, left:right]
        positions = group_mask.long().cumsum(-1)
        levels = (int(group_mask.sum(-1).max()) + block - 1) // block
        for level in range(1 if truncating else levels, levels + 1):
            prefix = group_mask & (positions <= level * block)
            advantage, prefix_returns, _ = compute_topd_advantage(group_rewards, prefix, [(0, end - start)], epsilon)
            inverse_q = max_len / (max_len - (level - 1) * block) if truncating else 1.0
            coefficient = inverse_q
            if level < levels:
                coefficient -= max_len / (max_len - level * block)
            scale = coefficient / prefix.sum().clamp_min(1).item()
            coefficients[start:end, left:right, 0] += advantage.double().clamp_min(0) * scale
            coefficients[start:end, left:right, 1] += advantage.double().clamp_max(0) * scale
        returns[start:end, left:right] = prefix_returns
    return coefficients.float(), returns, counts


def topd_stopd_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None):
    """PPO on signed telescoping levels, with normalization fixed by the trainer."""
    if getattr(config, "async_opd", False):
        old_log_probs = rollout_logprobs
    ratio = safe_exp_delta(log_probs - old_log_probs, clip=20.0, out_dtype=log_probs.dtype)
    clipped = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high)
    objective = torch.minimum(ratio, clipped) * advantages[..., 0] + torch.maximum(ratio, clipped) * advantages[..., 1]
    loss = reduce_loss(-objective, loss_mask)
    clip_ratio = masked_mean((ratio != clipped).float(), loss_mask).detach().item()
    return loss, {"clip_ratio": clip_ratio}


def async_topd_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None):
    """TOP-D's clipped surrogate anchored to the recorded behavior policy."""
    config = copy.copy(config)
    config.policy_loss_type = "regular"
    return ppo_policy_loss(log_probs, rollout_logprobs, advantages, config, loss_mask, rollout_logprobs)
