"""Token returns and group advantages from TOP-D, arXiv:2607.04751, Eqs. (7–8).

Uses SkyRL's contiguous prompt groups and right-aligned response tensors.
The STOPD combination uses a randomized telescoping sum of group objectives.
It only needs observed prefixes, including for future returns and group moments.
"""

from dataclasses import dataclass

import torch

from skyrl.train.config import BaseConfig
from skyrl.backends.skyrl_train.utils.ppo_utils import reduce_loss
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean, safe_exp_delta


@dataclass
class TOPDConfig(BaseConfig):
    enabled: bool = False
    """Enable future returns, token group normalization, and native PPO updates."""
    epsilon: float = 1e-6
    """Numerical stabilizer added to the population standard deviation."""

    def __post_init__(self):
        if not 0 < self.epsilon < float("inf"):
            raise ValueError("TOP-D epsilon must be finite and positive")


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
    ratio = safe_exp_delta(log_probs - old_log_probs, clip=20.0, out_dtype=log_probs.dtype)
    clipped = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high)
    objective = torch.minimum(ratio, clipped) * advantages[..., 0] + torch.maximum(ratio, clipped) * advantages[..., 1]
    loss = reduce_loss(-objective, loss_mask)
    clip_ratio = masked_mean((ratio != clipped).float(), loss_mask).detach().item()
    return loss, {"clip_ratio": clip_ratio}
