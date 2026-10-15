"""Offline token-level Monte-Carlo value targets on exact rollout tokens.

Adapted from ReDiPPO's RewardDataCollator and PreTrainer.compute_loss (Apache-2.0):
https://github.com/cii030/ReDiPPO/tree/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining
Changes: explicit multi-turn action masks, native scalar critic outputs, exact
rollout token IDs, and explicit returns for truncated trajectories.
"""

import math

import torch
from loguru import logger
from torch.utils.data import Dataset

from skyrl.train.dataset.pretokenized import _load_as_hf_dataset


class ValueDataset(Dataset):
    """Unpadded rows: input_ids, full-sequence loss_mask, reward or returns.

    Completed trajectories with an undiscounted terminal reward use that outcome
    at every action. Truncated trajectories require explicit per-token returns
    (including any bootstrap). Prompt and tool-observation tokens remain context
    but are never loss targets. Over-length rows are dropped, not truncated.
    """

    def __init__(self, path: str, max_length: int):
        dataset = _load_as_hf_dataset(path)
        self.dataset = dataset.map(self._normalize_row)
        self.dataset = self.dataset.filter(lambda row: len(row["input_ids"]) <= max_length)
        logger.info(f"Value dataset: {len(self.dataset)}/{len(dataset)} trajectories within max_length={max_length}")
        if not len(self.dataset):
            raise ValueError("Value dataset has no trajectories within max_length.")

    @staticmethod
    def _normalize_row(row):
        input_ids = row["input_ids"]
        loss_mask = row["loss_mask"]
        if len(input_ids) < 2 or len(loss_mask) != len(input_ids):
            raise ValueError("input_ids and full-sequence loss_mask must have the same length >= 2.")
        if any(not isinstance(token, int) or token < 0 for token in input_ids):
            raise ValueError("input_ids must contain non-negative integer token IDs.")
        if any(value not in (0, 1) for value in loss_mask) or loss_mask[0] != 0 or not any(loss_mask):
            raise ValueError("loss_mask must be binary, contain actions, and start with a prompt token (0).")
        if row.get("attention_mask") is not None and row["attention_mask"] != [1] * len(input_ids):
            raise ValueError("Value dataset rows must be unpadded.")
        returns = row.get("returns")
        if returns is None:
            if row.get("truncated", False):
                raise ValueError("Truncated trajectories require explicit per-token returns with bootstrap.")
            reward = float(row["reward"])
            if not math.isfinite(reward):
                raise ValueError("reward must be finite.")
            returns = [reward] * len(input_ids)
        if len(returns) != len(input_ids) or any(not math.isfinite(float(value)) for value in returns):
            raise ValueError("returns must contain one finite value per input token.")
        return {"input_ids": input_ids, "loss_mask": loss_mask, "returns": returns}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return {key: row[key] for key in ("input_ids", "loss_mask", "returns")}


class RewardDataCollator:
    """Left pad context; shift labels and masks to action-before-token states.

    The native critic already emits values[:, :-1]. Labels/masks are therefore
    shifted once here, not again in the loss. The first response token is scored
    by the state after the final prompt token.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        sequence_length = max(len(row["input_ids"]) for row in features)
        shape = (len(features), sequence_length)
        input_ids = torch.full(shape, self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        response_mask = torch.zeros(shape, dtype=torch.bool)
        labels = torch.zeros(shape, dtype=torch.float32)
        for index, row in enumerate(features):
            length = len(row["input_ids"])
            input_ids[index, -length:] = torch.tensor(row["input_ids"])
            attention_mask[index, -length:] = 1
            response_mask[index, -length:] = torch.tensor(row["loss_mask"], dtype=torch.bool)
            labels[index, -length:] = torch.tensor(row["returns"], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_actions": sequence_length - 1,
            "labels": labels[:, 1:],
            "response_mask": response_mask[:, 1:],
        }


def value_loss(predictions, labels, response_mask):
    """FP32 masked squared-error sum and action count, for global normalization.

    Keep raw scalar outputs to match SkyRL's online MSE critic and arbitrary
    reward scales; ReDiPPO's probability-only sigmoid is not applied.
    """
    errors = (predictions.float() - labels.float()).square()
    return errors.masked_select(response_mask).sum(), response_mask.sum()
