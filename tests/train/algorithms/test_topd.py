"""TOP-D equations, complete-group STOPD expectation, and native PPO reuse."""

import copy
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from examples.train.on_policy_distillation.main_on_policy_distill import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationTrainer,
    validate_stopd_cfg,
)
from examples.train.on_policy_distillation.stopd import STOPD
from examples.train.on_policy_distillation.topd import (
    compute_topd_advantage,
    mask_training_input,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.trainer import RayPPOTrainer


def config(**overrides):
    cfg = OnPolicyDistillationConfig.from_cli_overrides(
        {
            "trainer.algorithm.topd.enabled": True,
            "trainer.algorithm.advantage_estimator": "no_op",
            "trainer.algorithm.policy_loss_type": "regular",
            "trainer.algorithm.loss_reduction": "prompt_mean",
            "trainer.algorithm.use_kl_in_reward": True,
            "trainer.algorithm.use_kl_loss": False,
            "trainer.algorithm.proximal_teacher_alpha": 0.2,
            "trainer.algorithm.stopd.enabled": True,
            "trainer.algorithm.stopd.block": 2,
            "trainer.algorithm.stopd.warmup_steps": 0,
            "trainer.algorithm.stopd.initial_denominator": 11,
            "trainer.train_batch_size": 2,
            "trainer.policy_mini_batch_size": 1,
            "trainer.update_epochs_per_batch": 2,
            "generator.n_samples_per_prompt": 2,
            "generator.sampling_params.max_generate_length": 8,
        }
        | overrides
    )
    validate_stopd_cfg(cfg)
    return cfg


def batch():
    prompts = [[1, 2], [3], [4, 5, 6], [7]]
    responses = [[10, 11, 12, 0], [13, 14, 0], [15, 16, 17, 18, 19, 20, 0], [21, 0]]
    rewards = [[0.1, -0.2, 0.8, -0.1], [0.4, 0.5, -0.1], [0.7, -0.1, 0.3, -0.2, 0.4, 0.2, -0.1], [0.6, 0.2]]
    tensors = convert_prompts_responses_to_batch_tensors(
        0,
        prompts,
        responses,
        rewards,
        [[1] * len(r) for r in responses],
        [[-float(token) / 100 for token in r] for r in responses],
    )
    data = TrainingInputBatch(
        dict(
            zip(
                ["sequences", "attention_mask", "response_mask", "rewards", "loss_mask", "rollout_logprobs"],
                tensors[:6],
            )
        )
    )
    data["action_log_probs"] = data["rollout_logprobs"].clone()
    data["base_action_log_probs"] = data["action_log_probs"] + data["rewards"]
    data["values"] = None
    data.metadata = {
        "uids": ["a", "a", "b", "b"],
        "policy_prompt_boundaries": [(0, 2), (2, 4)],
        "policy_mini_batch_boundaries": [(0, 2), (2, 4)],
        "pad_size": 0,
        "avg_response_length": 4,
    }
    return data


def trainer(stopd=True, **overrides):
    obj = object.__new__(OnPolicyDistillationTrainer)
    obj.cfg = config(**overrides)
    obj.stopd = STOPD(8, 2, 0, 11, seed=42) if stopd else None
    obj.tokenizer = SimpleNamespace(pad_token_id=0)
    obj.all_metrics = {}
    return obj


def test_returns_and_group_moments_match_independent_scalar_equations():
    data = batch()
    advantages, returns, counts = compute_topd_advantage(
        data["rewards"], data["loss_mask"], data.metadata["policy_prompt_boundaries"]
    )
    expected = torch.zeros_like(returns)
    for i in range(4):
        values = data["rewards"][i][data["loss_mask"][i].bool()].tolist()
        for t in range(len(values)):
            future = sum(values[t + 1 :]) / (len(values) - t - 1) if t + 1 < len(values) else 0
            expected[i, -len(values) + t] = values[t] + future
    torch.testing.assert_close(returns, expected)
    assert counts == [(0, 2, 7), (2, 4, 9)]
    for start, end in data.metadata["policy_prompt_boundaries"]:
        mask = data["loss_mask"][start:end].bool()
        values = expected[start:end][mask]
        ref = (values - values.mean()) / (values.std(correction=0) + 1e-6)
        torch.testing.assert_close(advantages[start:end][mask], ref)
    assert (advantages[~data["loss_mask"].bool()] == 0).all()


def test_constant_single_token_and_padding_groups_are_finite():
    rewards = torch.tensor([[0.0, 3.0], [99.0, 3.0], [float("nan"), 99.0]])
    mask = torch.tensor([[0, 1], [0, 1], [0, 0]])
    advantages, returns, _ = compute_topd_advantage(rewards, mask, [(0, 2)])
    assert advantages.count_nonzero() == 0
    torch.testing.assert_close(returns, torch.tensor([[0.0, 3.0], [0.0, 3.0], [0.0, 0.0]]))


def test_prefix_loss_mask_preserves_full_forward_layout_and_fixed_targets():
    data = batch()
    data["advantages"], data["returns"], _ = compute_topd_advantage(
        data["rewards"], data["loss_mask"], [(0, 2), (2, 4)]
    )
    original = copy.deepcopy(data)
    result = mask_training_input(data, [2, 4, 4, 2])
    for i, kept in enumerate([2, 3, 4, 2]):
        length = int(data["response_mask"][i].sum())
        expected = torch.zeros_like(result["loss_mask"][i])
        expected[-length : result["loss_mask"].shape[1] - length + kept] = 1
        torch.testing.assert_close(result["loss_mask"][i], expected)
    for key in (
        "sequences",
        "attention_mask",
        "response_mask",
        "returns",
        "action_log_probs",
        "base_action_log_probs",
        "rollout_logprobs",
    ):
        torch.testing.assert_close(result[key], original[key], rtol=0, atol=0)
    assert (result["advantages"][~result["loss_mask"].bool()] == 0).all()


def test_stopd_expected_clipped_loss_and_gradient_equal_complete_topd():
    obj = trainer()
    full = batch()
    full["advantages"], full["returns"], counts = compute_topd_advantage(
        full["rewards"], full["loss_mask"], [(0, 2), (2, 4)]
    )
    full.metadata["topd_full_token_counts"] = counts
    delta = torch.linspace(-0.7, 0.7, full["advantages"].numel()).reshape_as(full["advantages"])
    theta = torch.tensor(1.0, requires_grad=True)
    reference = copy.deepcopy(full)
    obj._normalize_advantages(reference, [(0, 4)])
    loss, _ = ppo_policy_loss(
        full["action_log_probs"] + theta * delta,
        full["action_log_probs"],
        reference["advantages"],
        obj.cfg.trainer.algorithm,
        loss_mask=full["loss_mask"],
    )
    expected_grad = torch.autograd.grad(loss, theta)[0]
    losses, grads = [], []
    # Exhaustive independent block-budget distribution, including EOS before cut.
    for budgets in itertools.product([2, 4, 6, 8], repeat=4):
        sample = copy.deepcopy(full)
        sample["advantages"] *= obj.stopd.weights(full["loss_mask"])
        sample["delta"] = delta
        sample = mask_training_input(sample, budgets)
        obj._normalize_advantages(sample, [(0, 4)])
        current = sample["action_log_probs"] + theta * sample["delta"]
        value, _ = ppo_policy_loss(
            current,
            sample["action_log_probs"],
            sample["advantages"],
            obj.cfg.trainer.algorithm,
            loss_mask=sample["loss_mask"],
        )
        losses.append(value.detach())
        grads.append(torch.autograd.grad(value, theta)[0])
    torch.testing.assert_close(torch.stack(losses).mean(), loss.detach(), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(torch.stack(grads).mean(), expected_grad, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_inner_updates_reuse_fixed_targets_and_advance_stopd_once(monkeypatch, enabled):
    obj = trainer(stopd=enabled)
    data = obj.compute_advantages_and_returns(batch())
    old = data["action_log_probs"].clone()
    targets = data["returns"].clone()
    obj.global_step = 1
    obj.cfg.trainer.critic.model.path = None
    obj.all_timings = {}
    obj.dispatch = MagicMock()
    obj.dispatch.stage_data.side_effect = lambda model, data, boundaries: [data[a:b] for a, b in boundaries]
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    deltas, clip_ratios = [], []

    def forward_backward(model, chunk):
        delta = parameter * chunk["advantages"].sign()
        value, metrics = ppo_policy_loss(
            chunk["action_log_probs"] + delta,
            chunk["action_log_probs"],
            chunk["advantages"],
            obj.cfg.trainer.algorithm,
            loss_mask=chunk["loss_mask"],
        )
        value.backward()
        deltas.append(delta.detach().abs().max().item())
        clip_ratios.append(metrics["clip_ratio"])
        return SimpleNamespace(metrics={"policy_loss": value.item()})

    def optim_step(model):
        norm = parameter.grad.abs().item()
        optimizer.step()
        optimizer.zero_grad()
        return norm

    obj.dispatch.forward_backward_from_staged.side_effect = forward_backward
    obj.dispatch.optim_step.side_effect = optim_step
    OnPolicyDistillationTrainer.train_critic_and_policy(obj, data)
    assert obj.dispatch.optim_step.call_count == 4
    assert obj.dispatch.stage_data.call_count == 1
    assert deltas[0] == 0 and max(deltas[1:]) > 0
    assert max(clip_ratios[1:]) > 0
    assert parameter.item() != 0
    torch.testing.assert_close(data["action_log_probs"], old, rtol=0, atol=0)
    torch.testing.assert_close(data["returns"], targets, rtol=0, atol=0)
    if enabled:
        assert obj.stopd.step == 1
    assert obj.all_metrics["topd/optimizer_steps_per_batch"] == 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("generator.n_samples_per_prompt", 1),
        ("trainer.algorithm.policy_loss_type", "importance_sampling"),
        ("trainer.algorithm.loss_reduction", "token_mean"),
        ("generator.apply_overlong_filtering", True),
        ("trainer.algorithm.advantage_batch_normalize", True),
    ],
)
def test_invalid_topd_config_fails_at_boundary(key, value):
    with pytest.raises(ValueError, match="TOP-D"):
        config(**{key: value})


def test_plain_topd_uses_complete_data_and_native_prompt_reduction():
    obj = trainer(stopd=False)
    original = batch()
    result = obj.compute_advantages_and_returns(original)
    reference = copy.deepcopy(result)
    obj._normalize_advantages(result, [(0, 2), (2, 4)])
    RayPPOTrainer._normalize_advantages(obj, reference, [(0, 2), (2, 4)], [(0, 2), (2, 4)])
    torch.testing.assert_close(result["advantages"], reference["advantages"])


@pytest.mark.parametrize("completed", [2, 5])
@pytest.mark.parametrize("workers", [0, 2])
def test_resume_preserves_next_data_and_shuffle_at_epoch_boundary(tmp_path, monkeypatch, completed, workers):
    def loader():
        return StatefulDataLoader(
            list(range(11)),
            batch_size=2,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(123),
            num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
        )

    uninterrupted = loader()
    iterator = iter(uninterrupted)
    for _ in range(completed):
        next(iterator)
    state = uninterrupted.state_dict()
    torch.save(state, tmp_path / "data.pt")
    if completed == len(uninterrupted):
        assert next(iterator, None) is None
        expected = next(iter(uninterrupted))
    else:
        expected = next(iterator)
    obj = trainer()
    obj.stopd.step = completed
    torch.save(obj.stopd.state_dict(), tmp_path / "stopd_state.pt")
    obj.train_dataloader = loader()
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (completed, str(tmp_path)))
    obj.load_checkpoints()
    torch.testing.assert_close(next(iter(obj.train_dataloader)), expected, rtol=0, atol=0)
    assert obj.stopd.step == completed
