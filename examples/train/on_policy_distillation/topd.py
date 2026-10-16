"""Token returns and group advantages from TOP-D, arXiv:2607.04751, Eqs. (7–8).

Uses SkyRL's contiguous prompt groups and right-aligned response tensors.
STOPD may select loss prefixes only after these complete-response statistics
have been computed; otherwise suffix returns and group standard deviations
would depend on the random stopping time.
"""

from dataclasses import dataclass

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import BaseConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors


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


def truncate_training_input(data, budgets, pad_token_id):
    """Repack complete trajectories as prefixes after computing fixed TOP-D targets.

    Reuses the native collator for token/mask layout and slices all response
    tensors at the same token positions. No EOS is inserted at a sampled cut.
    """
    lengths = data["response_mask"].sum(-1).long().tolist()
    kept = [min(length, budget) for length, budget in zip(lengths, budgets, strict=True)]
    prompts, responses = [], []
    for row, length, count in zip(range(len(lengths)), lengths, kept, strict=True):
        tokens = data["sequences"][row][data["attention_mask"][row].bool()].tolist()
        prompt_length = len(tokens) - length
        prompts.append(tokens[:prompt_length])
        responses.append(tokens[prompt_length : prompt_length + count])
    tensors = convert_prompts_responses_to_batch_tensors(
        pad_token_id, prompts, responses, [[0.0] * count for count in kept], [[1] * count for count in kept]
    )
    sequences, attention_mask, response_mask = tensors[:3]
    result = {}
    for key, value in data.items():
        if key in ("sequences", "attention_mask", "response_mask"):
            continue
        if value is None:
            result[key] = None
            continue
        # The supported single-turn text path has only response-aligned fields.
        target = value.new_zeros(response_mask.shape)
        for row, length, count in zip(range(len(lengths)), lengths, kept, strict=True):
            if count:
                target[row, -count:] = value[row, value.shape[1] - length : value.shape[1] - length + count]
        result[key] = target
    result.update(sequences=sequences, attention_mask=attention_mask, response_mask=response_mask)
    batch = TrainingInputBatch(result)
    batch.metadata = data.metadata.copy()
    batch.metadata["response_length"] = response_mask.shape[1]
    return batch
