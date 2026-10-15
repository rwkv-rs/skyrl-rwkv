"""STOPD budgets, IPW weights, and lagged token-count normalization.

Adapted from YouAreSpecialToMe/STOPD, src/stopd/{core,truncation}.py,
under Apache-2.0: https://github.com/YouAreSpecialToMe/STOPD.
SkyRL's trainer holds the complete rollout batch, so statistics are summed
on the driver before the native data-parallel and microbatch reductions.
"""

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Optional

from skyrl.train.config import BaseConfig


@dataclass
class STOPDConfig(BaseConfig):
    """Optional stochastic truncated on-policy distillation."""

    enabled: bool = False
    """Enable independent rollout truncation and IPW loss correction."""
    block: int = 128
    """Round each random token budget up to this block boundary."""
    warmup_steps: int = 25
    """Number of complete OPD updates before truncation, counted from global step zero."""
    initial_denominator: Optional[float] = None
    """Full response-token count of the preceding step when importing a Vanilla checkpoint."""
    seed: int = 0
    """Seed for independent rollout budgets and per-request sampling."""

    def __post_init__(self):
        if self.block < 1 or self.warmup_steps < 0:
            raise ValueError("STOPD requires block >= 1 and warmup_steps >= 0")
        if self.initial_denominator is not None and (
            not math.isfinite(self.initial_denominator) or self.initial_denominator <= 0
        ):
            raise ValueError("STOPD initial_denominator must be finite and positive")


def request_seed(seed, step, sample, rollout, purpose):
    """Independent deterministic 32-bit seeds for stops and token sampling."""
    value = ":".join(map(str, (seed, step, sample, rollout, purpose)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def token_weights(mask, max_len, block):
    """IPW weights at zero-based response positions, with padding excluded."""
    mask = mask.bool()
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    weights = max_len / (max_len - (positions // block) * block).float()
    return weights * mask


class STOPD:
    """State for the current step's fixed denominator and the next step's estimate."""

    def __init__(self, max_len, block=128, warmup_steps=25, initial_denominator=None, seed=0):
        if not 1 <= block <= max_len:
            raise ValueError("STOPD requires 1 <= block <= max response length")
        if warmup_steps == 0 and initial_denominator is None:
            raise ValueError("STOPD without warmup requires initial_denominator")
        self.max_len = max_len
        self.block = block
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.step = 0
        self.denominator = initial_denominator

    @property
    def truncating(self):
        return self.step >= self.warmup_steps

    def budgets(self, n, offset=0):
        if not self.truncating:
            return [self.max_len] * n
        return [
            min(
                (
                    random.Random(request_seed(self.seed, self.step + 1, offset + i, 0, "block")).randrange(
                        self.max_len
                    )
                    // self.block
                    + 1
                )
                * self.block,
                self.max_len,
            )
            for i in range(n)
        ]

    def weights(self, mask):
        return token_weights(mask, self.max_len, self.block) if self.truncating else mask.bool().float()

    def multiplier(self, mask, n_obs):
        """Cancel native token-mean reduction in favor of the fixed denominator."""
        weights = self.weights(mask)
        return weights * (float(n_obs) / self.denominator) if self.truncating else weights

    def end_step(self, mask):
        weights = self.weights(mask)
        estimate = weights.double().sum().item()
        stats = {
            "step": self.step + 1,
            "truncating": int(self.truncating),
            "tokens": int(mask.sum().item()),
            "n_hat": estimate,
            "max_weight": weights.max().item(),
        }
        if self.truncating:
            stats["denominator"] = self.denominator
            stats["n_hat_over_denominator"] = estimate / self.denominator
        if self.truncating or self.step + 1 == self.warmup_steps:
            self.denominator = estimate
        self.step += 1
        return stats

    def state_dict(self):
        return {
            "version": 1,
            "step": self.step,
            "denominator": self.denominator,
            "max_len": self.max_len,
            "block": self.block,
            "warmup_steps": self.warmup_steps,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        if state["version"] != 1:
            raise ValueError("Unsupported STOPD checkpoint version")
        for key in ("max_len", "block", "warmup_steps", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"STOPD checkpoint {key} does not match the current config")
        if state["step"] >= self.warmup_steps and (
            state["denominator"] is None or not math.isfinite(state["denominator"]) or state["denominator"] <= 0
        ):
            raise ValueError("STOPD checkpoint requires a finite positive denominator")
        self.step = state["step"]
        self.denominator = state["denominator"]
