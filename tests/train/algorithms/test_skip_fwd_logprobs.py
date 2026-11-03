"""
Tests for skipping the policy forward pass when the loss optimizes against rollout logprobs,
and for the worker-side per-mini-batch train/rollout logprob diff metric.

uv run --isolated --extra dev pytest tests/train/algorithms/test_skip_fwd_logprobs.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    off_policy_correction_enabled,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    LOSSES_WITHOUT_OLD_LOGPROBS,
    compute_policy_loss_cispo,
    dppo_policy_loss,
    rollout_is_policy_loss,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker_utils import (
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY,
    compute_minibatch_rollout_logprob_diff_metrics,
    reduce_metrics,
)
from skyrl.train.config import AlgorithmConfig, OffPolicyCorrectionConfig
from skyrl.train.utils.trainer_utils import finalize_minibatch_rollout_logprob_diff_std


@pytest.mark.parametrize("loss_name", ["regular", "dual_clip"])
@pytest.mark.parametrize("strict", [False, True])
def test_strict_ppo_skips_old_forward_only_with_recorded_rollout(loss_name, strict):
    from types import SimpleNamespace

    from skyrl.train.trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    algorithm = AlgorithmConfig(policy_loss_type=loss_name, github_strict_loss=strict, use_kl_loss=False)
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(algorithm=algorithm))
    assert trainer._skip_policy_forward({"rollout_logprobs": torch.zeros(1, 2)}) is strict
    assert trainer._skip_policy_forward({"rollout_logprobs": None}) is False
    algorithm.use_kl_in_reward = True
    assert trainer._skip_policy_forward({"rollout_logprobs": torch.zeros(1, 2)}) is False
    algorithm.use_kl_in_reward = False
    algorithm.off_policy_correction.tis_ratio_type = "token"
    assert trainer._skip_policy_forward({"rollout_logprobs": torch.zeros(1, 2)}) is False


def test_strict_anchor_does_not_report_fabricated_zero_forward_diff():
    from unittest.mock import MagicMock

    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = SkyRLTrainConfig()
    trainer.cfg.trainer.algorithm.github_strict_loss = True
    trainer.cfg.trainer.algorithm.use_kl_loss = False
    trainer.dispatch = MagicMock()
    trainer.ref_model = None
    trainer.all_metrics = {}
    trainer.all_timings = {}
    data = TrainingInputBatch(
        {
            "sequences": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "response_mask": torch.ones(2, 3),
            "loss_mask": torch.ones(2, 3),
            "rollout_logprobs": torch.full((2, 3), -1.0),
        }
    )
    data.metadata = {"response_length": 3}
    result = trainer.fwd_logprobs_values_reward(data)
    trainer.dispatch.forward.assert_not_called()
    assert torch.equal(result["action_log_probs"], result["rollout_logprobs"])
    assert not any("rollout_train_logprobs_abs_diff" in name for name in trainer.all_metrics)


def test_losses_without_old_logprobs():
    """rollout_is and dppo optimize against rollout logprobs and do not need old logprobs."""
    assert "rollout_is" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "dppo" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "regular" not in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "gspo" not in LOSSES_WITHOUT_OLD_LOGPROBS


@pytest.mark.parametrize("source,expected", [(None, True), ("rollout", True), ("trainer", False)])
def test_old_logprob_source_preserves_strict_numerical_conventions(source, expected):
    config = AlgorithmConfig(github_strict_loss=True, old_logprobs_source=source)
    assert config.uses_rollout_old_logprobs is expected
    assert config.github_strict_loss is True
    assert config.grpo_norm_by_std is True


def test_invalid_old_logprob_source_is_rejected():
    config = AlgorithmConfig(old_logprobs_source="current_minibatch")
    with pytest.raises(ValueError, match="old_logprobs_source"):
        _ = config.uses_rollout_old_logprobs


def test_recomputed_denominator_is_fixed_before_all_optimizer_steps():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = SkyRLTrainConfig()
    trainer.cfg.trainer.algorithm.github_strict_loss = True
    trainer.cfg.trainer.algorithm.old_logprobs_source = "trainer"
    trainer.cfg.trainer.algorithm.use_kl_loss = False
    trainer.dispatch = MagicMock()
    trainer.ref_model = None
    trainer.all_metrics = {}
    trainer.all_timings = {}
    events = []
    old_logprobs = torch.tensor([[-1.0, -1.5], [-2.0, -2.5], [-3.0, -3.5], [-4.0, -4.5]])

    def compute_old(*args, **kwargs):
        events.append("old_forward")
        return old_logprobs.clone()

    trainer._execute_forward_pass = compute_old
    data = TrainingInputBatch(
        {
            "sequences": torch.ones(4, 3, dtype=torch.long),
            "attention_mask": torch.ones(4, 3),
            "response_mask": torch.ones(4, 2),
            "loss_mask": torch.ones(4, 2),
            "rollout_logprobs": None,
            "advantages": torch.ones(4, 2),
        }
    )
    data.metadata = {"response_length": 2, "policy_mini_batch_boundaries": [(0, 2), (2, 4)]}
    prepared = trainer.fwd_logprobs_values_reward(data)
    assert torch.equal(prepared["action_log_probs"], old_logprobs)
    assert not prepared["action_log_probs"].requires_grad
    trainer.dispatch.stage_data.side_effect = lambda model, batch, boundaries: [
        batch[start:end] for start, end in boundaries
    ]

    def train_minibatch(model, batch, **worker_kwargs):
        assert worker_kwargs == {"return_per_token_outputs": False}
        start = 0 if events.count("optimizer") == 0 else 2
        assert torch.equal(batch["action_log_probs"], old_logprobs[start : start + 2])
        events.append("backward")
        return SimpleNamespace(metrics={"policy_loss": 0.0})

    def update(model):
        events.append("optimizer")
        return 1.0

    trainer.dispatch.forward_backward_from_staged.side_effect = train_minibatch
    trainer.dispatch.optim_step.side_effect = update
    trainer._execute_training_step("policy", prepared)
    assert events == ["old_forward", "backward", "optimizer", "backward", "optimizer"]


@pytest.mark.parametrize(
    "loss_name, loss_fn",
    [("rollout_is", rollout_is_policy_loss), ("dppo", dppo_policy_loss)],
)
def test_skip_path_losses_run_with_old_log_probs_none(loss_name, loss_fn):
    """Skip-path invariant: these losses must produce a finite loss with ``old_log_probs=None``
    when off-policy correction is disabled (rollout logprobs stand in for the old logprobs)."""
    assert loss_name in LOSSES_WITHOUT_OLD_LOGPROBS
    config = AlgorithmConfig(policy_loss_type=loss_name, off_policy_correction=OffPolicyCorrectionConfig())
    assert not off_policy_correction_enabled(config.off_policy_correction)

    log_probs = torch.tensor([[-1.0, -1.5, -0.8], [-0.7, -1.2, -2.0]])
    rollout_logprobs = torch.tensor([[-1.2, -1.0, -0.9], [-0.6, -1.5, -1.7]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    loss, _ = loss_fn(
        log_probs,
        None,  # old_log_probs: skipped forward pass
        advantages,
        config=config,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
    )
    assert torch.isfinite(loss)


def test_cispo_rollout_anchor_runs_with_old_log_probs_none():
    """Skip-path invariant for cispo with cispo_anchor='rollout': it anchors on rollout logprobs
    and never reads old logprobs, so it must produce a finite loss with ``old_log_probs=None``.
    This is what makes it safe for ``_skip_policy_forward`` to skip the old-logprobs forward."""
    from skyrl.train.config import CISPOConfig

    config = AlgorithmConfig(
        policy_loss_type="cispo",
        cispo=CISPOConfig(cispo_anchor="rollout"),
        off_policy_correction=OffPolicyCorrectionConfig(),
    )
    assert not off_policy_correction_enabled(config.off_policy_correction)

    log_probs = torch.tensor([[-1.0, -1.5, -0.8], [-0.7, -1.2, -2.0]])
    rollout_logprobs = torch.tensor([[-1.2, -1.0, -0.9], [-0.6, -1.5, -1.7]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    loss, _ = compute_policy_loss_cispo(
        log_probs,
        None,  # old_log_probs: skipped forward pass
        advantages,
        config=config,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
    )
    assert torch.isfinite(loss)


def test_off_policy_correction_enabled():
    """The helper detects every path in compute_off_policy_correction that reads old logprobs."""
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig())
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(tis_ratio_type="token"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(sequence_mask_metric="product"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_high=100.0))
    # token mask requires both bounds to be set
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(
        OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4, token_mask_is_threshold_high=100.0)
    )


def test_minibatch_metric_values_match_manual():
    """The worker metric reports the masked abs diff moments/extremes via the same masked_mean /
    `* loss_mask` pattern as the off-policy `is_ratio_*` metrics."""
    action_log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -3.0], [-0.5, -2.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    metrics = compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_logprobs, loss_mask)

    abs_diff = (action_log_probs - rollout_logprobs).abs()
    masked = abs_diff * loss_mask
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY] == masked_mean(abs_diff, loss_mask).item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY] == masked_mean(abs_diff.square(), loss_mask).item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY] == masked.max().item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY] == masked.min().item()


def test_minibatch_metric_emitted_unconditionally_when_rollout_present():
    """Metric is omitted only when rollout logprobs are absent (a DP-uniform condition). With
    rollout logprobs present but every token masked, it still emits the keys as 0.0 so the key set
    stays consistent across DP ranks."""
    alp = torch.tensor([[-1.0, -2.0]])
    rlp = torch.tensor([[-1.5, -1.0]])
    assert compute_minibatch_rollout_logprob_diff_metrics(alp, None, None) == {}

    fully_masked = torch.zeros_like(alp)
    masked_metrics = compute_minibatch_rollout_logprob_diff_metrics(alp, rlp, fully_masked)
    assert set(masked_metrics) == {
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY,
    }
    assert all(v == 0.0 for v in masked_metrics.values())


def test_finalize_std_noop_without_moments():
    """The std derivation is a no-op when the moment keys are absent (e.g. critic training)."""
    metrics = {"foo": 1.0}
    finalize_minibatch_rollout_logprob_diff_std(metrics)
    assert metrics == {"foo": 1.0}


def test_std_aggregates_correctly_across_microbatches_and_minibatches():
    """Reduce per-micro-batch moments over micro-batches then mini-batches, derive the std, and
    check it equals the exact pooled population std (exact because all sizes are equal)."""
    # Four equal-size micro-batches grouped into two mini-batches of two micro-batches each.
    diffs = [
        torch.tensor([0.1, 0.3, 0.5, 0.2]),
        torch.tensor([1.0, 0.8, 0.6, 0.4]),
        torch.tensor([0.2, 0.2, 0.9, 0.7]),
        torch.tensor([0.3, 0.5, 0.1, 0.6]),
    ]

    def moments(d):
        return {
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY: d.mean().item(),
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY: d.square().mean().item(),
        }

    keys = list(moments(diffs[0]).keys())
    # Worker-level reduction over micro-batches within each mini-batch.
    mb1 = reduce_metrics({k: [moments(diffs[0])[k], moments(diffs[1])[k]] for k in keys})
    mb2 = reduce_metrics({k: [moments(diffs[2])[k], moments(diffs[3])[k]] for k in keys})

    # Trainer-level reduction across mini-batches.
    all_metrics = {k: [mb1[k], mb2[k]] for k in keys}
    reduced = reduce_metrics(all_metrics)
    finalize_minibatch_rollout_logprob_diff_std(reduced)

    # All micro-batches are equal size, so the pooled mean/std are exact.
    pooled = torch.cat(diffs)
    assert MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY not in reduced
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY] - pooled.mean().item()) < 1e-6
    expected_std = pooled.std(unbiased=False).item()
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY] - expected_std) < 1e-6
