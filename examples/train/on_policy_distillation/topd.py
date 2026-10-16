"""Token returns and group advantages from TOP-D, arXiv:2607.04751, Eqs. (7–8).

Uses SkyRL's contiguous prompt groups and right-aligned response tensors.
STOPD may select loss prefixes only after these complete-response statistics
have been computed; otherwise suffix returns and group standard deviations
would depend on the random stopping time.
"""

from dataclasses import dataclass

import torch

from skyrl.train.config import BaseConfig


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


def mask_training_input(data, budgets):
    """Select prefix loss tokens while preserving the complete forward layout."""
    positions = data["response_mask"].long().cumsum(-1)
    limits = torch.tensor(budgets, device=positions.device).unsqueeze(-1)
    prefix_mask = positions <= limits
    data["loss_mask"] = data["loss_mask"] * prefix_mask
    data["advantages"] = data["advantages"] * prefix_mask
    return data
