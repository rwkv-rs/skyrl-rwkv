"""TOP-D equations, complete-group STOPD expectation, and native PPO reuse."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from examples.train.on_policy_distillation.main_on_policy_distill import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationExp,
    OnPolicyDistillationTrainer,
    validate_stopd_cfg,
)
from examples.train.on_policy_distillation.stopd import STOPD
from examples.train.on_policy_distillation.topd import (
    balance_training_input,
    compute_topd_advantage,
    compute_topd_stopd_advantage,
    topd_stopd_policy_loss,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss
from skyrl.backends.skyrl_train.workers.worker_utils import (
    BaseBatchIterator,
    SampleBasedBatchIterator,
)
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.trainer import RayPPOTrainer


def config(**overrides):
    cfg = OnPolicyDistillationConfig.from_cli_overrides(
        {
            "trainer.algorithm.topd.enabled": True,
            "trainer.algorithm.advantage_estimator": "no_op",
            "trainer.algorithm.policy_loss_type": "topd_stopd",
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
    if not stopd:
        overrides = {
            "trainer.algorithm.stopd.enabled": False,
            "trainer.algorithm.policy_loss_type": "regular",
        } | overrides
    obj.cfg = config(**overrides)
    obj.stopd = STOPD(8, 2, 0, 11, seed=42, require_denominator=False) if stopd else None
    obj.tokenizer = SimpleNamespace(pad_token_id=0)
    obj.all_metrics = {}
    return obj


@pytest.mark.parametrize("stopd", [False, True])
def test_full_response_metrics_recover_mean_across_all_stop_budgets(stopd):
    obj = trainer(stopd=stopd)
    # Enumerate every equally likely block budget, with two responses per group.
    budgets = torch.tensor([2, 4, 6, 8]).repeat_interleave(2) if stopd else torch.full((8,), 8)
    mask = (torch.arange(8)[None, :] < budgets[:, None]).float()
    kl = torch.arange(1, 9).float().expand(8, -1)
    old = torch.full_like(kl, -1)
    data = TrainingInputBatch(
        {"action_log_probs": old, "base_action_log_probs": old - kl, "response_mask": mask, "loss_mask": mask}
    )
    data.metadata = {"pad_size": 0}
    result = obj.apply_reward_kl_penalty(data)
    expected_reward = torch.log(0.2 * torch.exp(-kl) + 0.8)
    torch.testing.assert_close(result["rewards"], expected_reward * mask)
    assert obj.all_metrics["opd/reverse_kl_mean"] == pytest.approx(3.5 if stopd else 4.5)
    assert obj.all_metrics["opd/full_reverse_kl_mean_estimate"] == pytest.approx(4.5)
    assert obj.all_metrics["opd/full_reward_mean_estimate"] == pytest.approx(expected_reward[0].mean().item())


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


@pytest.mark.parametrize("dp_size", [4, 7])
def test_balance_preserves_prompt_mini_batches_signed_targets_and_gradients(dp_size):
    lengths = [80, 79, 3, 2, 70, 68, 4, 3, 60, 59, 5, 4, 50, 49, 6, 5] * 8
    rows = len(lengths)
    tensors = convert_prompts_responses_to_batch_tensors(
        0,
        [[1, 2]] * rows,
        [[3 + i % 10] * length for i, length in enumerate(lengths)],
        [[(t % 7 - 3) * 0.1 + i * 0.01 for t in range(length)] for i, length in enumerate(lengths)],
        [[1] * length for length in lengths],
    )
    data = TrainingInputBatch(
        dict(zip(["sequences", "attention_mask", "response_mask", "rewards", "loss_mask"], tensors[:5]))
    )
    data["features"] = torch.linspace(-0.7, 0.7, data["rewards"].numel()).reshape_as(data["rewards"])
    data["rollout_logprobs"] = torch.full_like(data["rewards"], -50)
    groups = [(i, i + 2) for i in range(0, rows, 2)]
    data.metadata = {
        "uids": [str(i // 2) for i in range(rows)],
        "policy_prompt_boundaries": groups,
        "policy_mini_batch_boundaries": [(i, i + 32) for i in range(0, rows, 32)],
    }
    data = pad_training_input_batch(data, (-rows) % dp_size)
    original_uids = data.metadata["uids"].copy()
    balanced, metrics = balance_training_input(data, dp_size=dp_size, micro_batch_size=4)
    order = balanced.metadata["batch_permutation"]
    real_positions = [i for i, source in enumerate(order) if source < rows]
    real_order = [order[i] for i in real_positions]
    assert sorted(real_order) == list(range(rows))
    assert data.metadata["uids"] == original_uids
    assert metrics["perf/dense_tokens_after_balance"] < metrics["perf/dense_tokens_before_balance"]
    assert metrics["perf/policy_rank_max_tokens_after_balance"] <= metrics["perf/policy_rank_max_tokens_before_balance"]
    for (start, end), (a, b) in zip(
        data.metadata["policy_mini_batch_boundaries"], balanced.metadata["policy_mini_batch_boundaries"], strict=True
    ):
        assert set(i for i in order[a:b] if i < rows) == set(range(start, end))
        assert (b - a) % (dp_size * 4) == 0
    for start, end in balanced.metadata["policy_prompt_boundaries"]:
        assert len(set(balanced.metadata["uids"][start:end])) == 1
    for key, value in data.items():
        torch.testing.assert_close(balanced[key][real_positions], value[real_order], rtol=0, atol=0)
    padding = torch.tensor([i >= rows for i in order])
    assert not balanced["loss_mask"][padding].any()
    assert not balanced["response_mask"][padding].any()
    assert not balanced["rollout_logprobs"][padding].any()
    assert not balanced["rewards"][padding].any()
    if padding.any():
        dummy = torch.zeros_like(balanced["rollout_logprobs"][padding], dtype=torch.float16, requires_grad=True)
        algorithm = config().trainer.algorithm
        algorithm.async_opd = True
        loss, _ = topd_stopd_policy_loss(
            dummy,
            torch.zeros_like(dummy),
            torch.zeros((*dummy.shape, 2), dtype=dummy.dtype),
            algorithm,
            balanced["loss_mask"][padding],
            balanced["rollout_logprobs"][padding].half(),
        )
        assert loss.item() == 0
        assert not torch.autograd.grad(loss, dummy)[0].any()
    assert (balanced["attention_mask"][padding].sum(-1) == 1).all()
    assert balanced.metadata["pad_size"] == 0
    targets = []
    theta = torch.tensor(0.8, requires_grad=True)
    losses, grads = [], []
    for current in [data, balanced]:
        coefficients, _, _ = compute_topd_stopd_advantage(
            current["rewards"], current["loss_mask"], current.metadata["policy_prompt_boundaries"], 128, 16, True
        )
        targets.append(coefficients)
        loss, _ = topd_stopd_policy_loss(
            theta * current["features"],
            torch.zeros_like(current["features"]),
            coefficients / len(groups),
            config().trainer.algorithm,
            current["loss_mask"],
        )
        losses.append(loss)
        grads.append(torch.autograd.grad(loss, theta)[0])
    torch.testing.assert_close(targets[1][real_positions], targets[0][real_order], rtol=0, atol=0)
    assert not targets[1][padding].any()
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(grads[0], grads[1])

    # Retain the production schedule: four mini-batches, two epochs, one
    # global clip per Adam update. Padding is absent from every group divisor.
    updates = []
    for source in (data, balanced):
        obj = trainer()
        obj.cfg.trainer.algorithm.async_opd = True
        current = copy.deepcopy(source)
        coefficients, _, counts = compute_topd_stopd_advantage(
            current["rewards"], current["loss_mask"], current.metadata["policy_prompt_boundaries"], 128, 16, True
        )
        current["advantages"] = coefficients
        current.metadata["topd_token_counts"] = counts
        obj._normalize_advantages(current, current.metadata["policy_mini_batch_boundaries"])
        parameter = torch.nn.Parameter(torch.tensor(0.8))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        gradients = []
        for _ in range(2):
            for start, end in current.metadata["policy_mini_batch_boundaries"]:
                optimizer.zero_grad()
                features = current["features"][start:end]
                loss, _ = topd_stopd_policy_loss(
                    parameter * features,
                    torch.zeros_like(features),
                    current["advantages"][start:end],
                    obj.cfg.trainer.algorithm,
                    current["loss_mask"][start:end],
                    rollout_logprobs=torch.full_like(features, -0.4),
                )
                loss.backward()
                gradients.append(parameter.grad.clone())
                torch.nn.utils.clip_grad_norm_([parameter], 1.0)
                optimizer.step()
        assert optimizer.state[parameter]["step"] == 8
        updates.append((torch.stack(gradients), parameter.detach(), optimizer.state[parameter]["exp_avg"]))
    for actual, expected in zip(*updates, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_balance_rejects_mismatched_microbatch_grid():
    with pytest.raises(ValueError, match="balance_batch"):
        config(
            **{
                "trainer.algorithm.topd.balance_batch": True,
                "trainer.micro_train_batch_size_per_gpu": 4,
                "trainer.micro_forward_batch_size_per_gpu": 2,
            }
        )


@pytest.mark.parametrize("response_only", [False, True])
def test_native_scoring_preserves_response_mask_for_selected_rwkv_head(response_only):
    obj = trainer(**{"trainer.rwkv_response_only_head": response_only})
    obj.critic_model = None
    obj.ref_model = object()
    obj.dispatch = MagicMock()
    data = batch()
    data.metadata["response_length"] = data["response_mask"].shape[1]
    calls = []

    def forward(model, scored, key, mini_batch_boundaries):
        calls.append(model)
        assert ("response_mask" in scored) == response_only
        if response_only:
            torch.testing.assert_close(scored["response_mask"], data["response_mask"], rtol=0, atol=0)
        return torch.zeros_like(data["response_mask"])

    obj._execute_forward_pass = forward
    result = obj.fwd_logprobs_values_reward(data)
    assert calls == ["ref", "policy"]
    assert result["action_log_probs"].shape == result["base_action_log_probs"].shape == data["response_mask"].shape


def test_constant_single_token_and_padding_groups_are_finite():
    rewards = torch.tensor([[0.0, 3.0], [99.0, 3.0], [float("nan"), 99.0]])
    mask = torch.tensor([[0, 1], [0, 1], [0, 0]])
    advantages, returns, _ = compute_topd_advantage(rewards, mask, [(0, 2)])
    assert advantages.count_nonzero() == 0
    torch.testing.assert_close(returns, torch.tensor([[0.0, 3.0], [0.0, 3.0], [0.0, 0.0]]))


@pytest.mark.parametrize("max_len,block", [(8, 2), (8, 3), (7, 7)])
def test_stopd_expected_clipped_loss_and_gradient_equal_complete_topd(max_len, block):
    obj = trainer()
    full = batch()
    full["advantages"], full["returns"], counts = compute_topd_advantage(
        full["rewards"], full["loss_mask"], [(0, 2), (2, 4)]
    )
    full.metadata["topd_token_counts"] = counts
    delta = torch.linspace(-0.7, 0.7, full["advantages"].numel()).reshape_as(full["advantages"])
    theta = torch.tensor(1.0, requires_grad=True)
    reference = copy.deepcopy(full)
    obj.stopd = None
    obj._normalize_advantages(reference, [(0, 4)])
    plain_cfg = copy.deepcopy(obj.cfg.trainer.algorithm)
    plain_cfg.policy_loss_type = "regular"
    loss, _ = ppo_policy_loss(
        full["action_log_probs"] + theta * delta,
        full["action_log_probs"],
        reference["advantages"],
        plain_cfg,
        loss_mask=full["loss_mask"],
    )
    expected_grad = torch.autograd.grad(loss, theta)[0]
    losses, grads = [], []
    # Enumerate uniform integer anchors (including a partial last block).
    # All responses in each prompt share the resulting budget.
    for anchor in range(max_len):
        budget = min((anchor // block + 1) * block, max_len)
        mask = full["loss_mask"].bool() & (full["response_mask"].long().cumsum(-1) <= budget)
        observed = full["rewards"].masked_fill(~mask, float("nan"))
        coefficients, _, _ = compute_topd_stopd_advantage(observed, mask, [(0, 2), (2, 4)], max_len, block, True)
        value, _ = topd_stopd_policy_loss(
            full["action_log_probs"] + theta * delta,
            full["action_log_probs"],
            coefficients / 2,
            obj.cfg.trainer.algorithm,
            loss_mask=mask,
        )
        losses.append(value.detach())
        grads.append(torch.autograd.grad(value, theta)[0])
    torch.testing.assert_close(torch.stack(losses).mean(), loss.detach(), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(torch.stack(grads).mean(), expected_grad, atol=2e-7, rtol=1e-5)


def test_stopd_warmup_matches_full_topd_without_randomization():
    data = batch()
    mask = data["loss_mask"].bool()
    groups = [(0, 2), (2, 4)]
    advantage, returns, counts = compute_topd_advantage(data["rewards"], mask, groups)
    coefficients, actual_returns, actual_counts = compute_topd_stopd_advantage(
        data["rewards"], mask, groups, 8, 2, False
    )
    for start, end, count in counts:
        torch.testing.assert_close(coefficients[start:end, :, 0], advantage[start:end].clamp_min(0) / count)
        torch.testing.assert_close(coefficients[start:end, :, 1], advantage[start:end].clamp_max(0) / count)
    torch.testing.assert_close(actual_returns, returns)
    assert actual_counts == counts


@pytest.mark.parametrize("displacement", [0.0, 0.1, 0.7])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_prefix_objective_matches_signed_loss_gradient(displacement, asynchronous):
    from examples.train.rwkv.audit_topd import reference_loss_gradient

    obj = trainer()
    obj.cfg.trainer.algorithm.async_opd = asynchronous
    data = obj.apply_reward_kl_penalty(batch())
    data["rollout_logprobs"] = data["action_log_probs"] - 0.3
    data = obj.compute_advantages_and_returns(data)
    obj._normalize_advantages(data, [(0, 4)])
    delta = torch.linspace(-displacement, displacement, data["action_log_probs"].numel()).reshape_as(
        data["action_log_probs"]
    )
    current = (data["action_log_probs"] + delta).requires_grad_()
    reference_loss, reference_gradient = reference_loss_gradient(data, current, obj.cfg)
    actual_loss, _ = topd_stopd_policy_loss(
        current,
        data["action_log_probs"],
        data["advantages"],
        obj.cfg.trainer.algorithm,
        loss_mask=data["loss_mask"],
        rollout_logprobs=data["rollout_logprobs"],
    )
    actual_gradient = torch.autograd.grad(actual_loss, current)[0]
    torch.testing.assert_close(
        actual_loss.detach().double(), torch.tensor(reference_loss, dtype=torch.float64), atol=2e-7, rtol=1e-5
    )
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-7, rtol=1e-5)


def test_signed_targets_survive_native_microbatch_and_experience_conversion():
    obj = trainer()
    data = obj.compute_advantages_and_returns(batch())
    data.metadata["response_length"] = data["loss_mask"].shape[1]
    obj._normalize_advantages(data, [(0, 2), (2, 4)])
    chunks = list(SampleBasedBatchIterator(data, sample_batch_size=1))
    assert len(chunks) == 4
    for index, chunk in enumerate(chunks):
        experience = BaseBatchIterator.batch_to_experience(chunk)
        torch.testing.assert_close(experience.advantages, data["advantages"][index : index + 1])
        assert experience.advantages.shape[-1] == 2


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
        direction = chunk["advantages"].sum(-1) if enabled else chunk["advantages"]
        delta = parameter * direction.sign()
        loss_fn = topd_stopd_policy_loss if enabled else ppo_policy_loss
        value, metrics = loss_fn(
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


def test_topd_import_needs_no_legacy_lagged_denominator(tmp_path, monkeypatch):
    loader = StatefulDataLoader(
        list(range(8)), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(123)
    )
    next(iter(loader))
    torch.save(loader.state_dict(), tmp_path / "data.pt")
    obj = trainer()
    obj.train_dataloader = StatefulDataLoader(
        list(range(8)), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(123)
    )
    obj.stopd = STOPD(8, 2, 0, require_denominator=False)
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (1, str(tmp_path)))
    obj.load_checkpoints()
    assert obj.stopd.step == 1 and obj.stopd.denominator is None
    metrics = obj.stopd.end_step(torch.ones(2, 4))
    assert "denominator" not in metrics
    assert obj.stopd.step == 2 and obj.stopd.denominator > 0
    restored = STOPD(8, 2, 0, require_denominator=False)
    restored.load_state_dict(obj.stopd.state_dict())
    assert restored.state_dict() == obj.stopd.state_dict()


@pytest.mark.parametrize("stopd", [False, True])
def test_topd_uses_independent_request_seeds_with_or_without_stopd(monkeypatch, stopd):
    cfg = config(
        **{
            "trainer.algorithm.stopd.enabled": stopd,
            "trainer.algorithm.policy_loss_type": "topd_stopd" if stopd else "regular",
            "trainer.seed": 42,
            "trainer.algorithm.stopd.seed": 19,
        }
    )
    captured = {}

    def generator(**kwargs):
        captured.update(kwargs)
        return captured

    monkeypatch.setattr("examples.train.on_policy_distillation.main_on_policy_distill.STOPDGenerator", generator)
    result = object.__new__(OnPolicyDistillationExp).get_generator(cfg, None, None)
    assert result["complete_rollouts"] == (not stopd)
    assert result["grouped_budgets"] == stopd
    assert result["stopd_cfg"].seed == (19 if stopd else 42)
    assert cfg.trainer.algorithm.stopd.seed == 19


@pytest.mark.parametrize("shift", [0.0, 0.4, 1.3])
def test_async_opd_expected_gradient_is_current_reverse_kl(shift):
    from examples.train.on_policy_distillation.main_on_policy_distill import (
        async_opd_policy_loss,
    )

    # Enumerate all behavior-policy actions: this is an exact expectation,
    # including stale behavior, rather than a Monte Carlo tolerance test.
    logits = torch.tensor([[0.3, -0.9, 0.1 + shift], [-0.2, 0.7, -0.3]], requires_grad=True)
    teacher = torch.tensor([[0.7, -0.4, 0.2], [-0.8, 0.3, 0.4]]).log_softmax(-1).requires_grad_()
    behavior = torch.tensor([[-0.6, 0.2, 0.4], [0.8, 0.1, -0.9]]).log_softmax(-1)
    current = logits.log_softmax(-1)
    weights = torch.tensor([[0.5], [1.25]])  # Includes a fixed STOPD/IPW weight.
    actual, _ = async_opd_policy_loss(
        current,
        torch.zeros_like(current),
        behavior.exp() * weights,
        SimpleNamespace(),
        torch.ones_like(current),
        behavior,
        teacher,
    )
    expected = (current.exp() * (current - teacher.detach()) * weights).sum()
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=2e-6)
    assert torch.autograd.grad(actual, teacher, allow_unused=True)[0] is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_topd_stopd_streamed_gradients_and_adam_equal_whole_batch(asynchronous):
    obj = trainer()
    obj.cfg.trainer.algorithm.async_opd = asynchronous
    data = obj.apply_reward_kl_penalty(batch())
    data["rollout_logprobs"] -= 0.4
    data = obj.compute_advantages_and_returns(data)
    obj._normalize_advantages(data, [(0, 4)])
    torch.manual_seed(13)
    features = torch.randn(*data["action_log_probs"].shape, 5)
    initial = torch.randn(5) * 0.2
    results = []
    for streaming in [False, True]:
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            for start, end in [(2, 4), (0, 2)] if streaming else [(0, 4)]:
                current = data["action_log_probs"][start:end] + features[start:end] @ parameter
                loss, _ = topd_stopd_policy_loss(
                    current,
                    data["action_log_probs"][start:end],
                    data["advantages"][start:end],
                    obj.cfg.trainer.algorithm,
                    data["loss_mask"][start:end],
                    data["rollout_logprobs"][start:end],
                )
                loss.backward()
            gradient = parameter.grad.clone()
            torch.nn.utils.clip_grad_norm_([parameter], 0.05)
            optimizer.step()
        results.append(
            (
                gradient,
                parameter.detach(),
                optimizer.state[parameter]["exp_avg"],
                optimizer.state[parameter]["exp_avg_sq"],
            )
        )
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("mode", ["one_step_off", "zero_stale"])
@pytest.mark.parametrize("policy_gpus", [4, 6, 7])
def test_async_topd_stopd_configuration_preserves_logical_optimizer_schedule(mode, policy_gpus):
    from skyrl.train.utils.utils import validate_batch_sizes

    cfg = config(
        **{
            "trainer.fully_async.enabled": True,
            "trainer.fully_async.mode": mode,
            "trainer.fully_async.max_staleness_steps": int(mode == "one_step_off"),
            "trainer.algorithm.async_opd": mode != "zero_stale",
            "trainer.placement.policy_num_gpus_per_node": policy_gpus,
            "trainer.placement.ref_num_gpus_per_node": policy_gpus,
            "trainer.train_batch_size": 512,
            "trainer.policy_mini_batch_size": 128,
            "trainer.micro_train_batch_size_per_gpu": 4,
            "trainer.micro_forward_batch_size_per_gpu": 4,
            "trainer.algorithm.topd.balance_batch": True,
        }
    )
    validate_batch_sizes(cfg)
    assert cfg.trainer.update_epochs_per_batch * cfg.trainer.train_batch_size // cfg.trainer.policy_mini_batch_size == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", ["topd_stopd", "grpo"])
async def test_native_group_streaming_matches_complete_topd_stopd_optimizer(objective):
    from examples.train.on_policy_distillation.main_on_policy_distill import (
        FullyAsyncOnPolicyDistillationTrainer,
    )
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdaptiveKLController
    from skyrl.train.fully_async_trainer import (
        FullyAsyncRayPPOTrainer,
        GeneratedOutputGroup,
    )
    from skyrl.train.generators.utils import concatenate_generator_outputs

    cfg = config(
        **{
            "trainer.policy_mini_batch_size": 2,
            "trainer.update_epochs_per_batch": 1,
            "trainer.micro_train_batch_size_per_gpu": 1,
            "trainer.micro_forward_batch_size_per_gpu": 1,
            "trainer.fully_async.enabled": True,
            "trainer.fully_async.mode": "zero_stale",
            "trainer.fully_async.max_staleness_steps": 0,
            "trainer.fully_async.streaming_batch_size": 1,
        }
    )
    if objective == "grpo":
        cfg.trainer.algorithm.topd.enabled = False
        cfg.trainer.algorithm.stopd.enabled = False
        cfg.trainer.algorithm.advantage_estimator = "grpo"
        cfg.trainer.algorithm.policy_loss_type = "regular"
    loss_function = topd_stopd_policy_loss if objective == "topd_stopd" else ppo_policy_loss
    trainer_class = FullyAsyncOnPolicyDistillationTrainer if objective == "topd_stopd" else FullyAsyncRayPPOTrainer
    groups = []
    for i, uid in enumerate(["a", "b"]):
        responses = [[10 + i, 11, 12, 0], [13 + i, 0]]
        output = {
            "prompt_token_ids": [[1, 2], [1, 2]],
            "response_ids": responses,
            "rewards": [[0.0] * len(row) for row in responses],
            "loss_masks": [[1] * len(row) for row in responses],
            "rollout_logprobs": [[-0.6] * len(row) for row in responses],
            "rollout_metrics": {},
            "stop_reasons": ["stop", "stop"],
        }
        groups.append(GeneratedOutputGroup(output, uid, 1))
    results = []
    for streaming in [False, True]:
        obj = object.__new__(trainer_class)
        obj.cfg, obj.global_step, obj.mini_batch_size = cfg, 1, 2
        obj.stopd = STOPD(8, 2, 0, 11, require_denominator=False)
        obj.tokenizer = SimpleNamespace(pad_token_id=0)
        obj.all_metrics, obj.all_timings = {}, {}
        obj.reward_kl_controller = AdaptiveKLController(init_kl_coef=0.1, target=0.1, horizon=100)
        obj.postprocess_generator_output = lambda output, uids: (output, uids)
        parameter = torch.nn.Parameter(torch.tensor(0.13))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)

        def features(data):
            return data["sequences"][:, -data["loss_mask"].shape[1] :].float().sin()

        def score(data):
            data["action_log_probs"] = torch.full_like(data["loss_mask"], -0.5, dtype=torch.float32)
            data["base_action_log_probs"] = data["action_log_probs"] + features(data) * 0.2
            data["values"] = None
            return data

        def backward(model, data, **kwargs):
            current = data["action_log_probs"] + parameter * features(data)
            loss, _ = loss_function(
                current,
                data["action_log_probs"],
                data["advantages"],
                cfg.trainer.algorithm,
                data["loss_mask"],
                data["rollout_logprobs"],
            )
            loss.backward()
            return SimpleNamespace(metrics={"policy_loss": loss.item()})

        def step(model, gradient_scale=1.0):
            parameter.grad.mul_(gradient_scale)
            norm = torch.nn.utils.clip_grad_norm_([parameter], 0.05).item()
            optimizer.step()
            return norm

        obj.dispatch = SimpleNamespace(
            dp_size=lambda _: 1, get_lcm_dp_size=lambda: 1, forward_backward=backward, optim_step=step
        )
        obj.fwd_logprobs_values_reward = score
        full_output = concatenate_generator_outputs([g.generator_output for g in groups])
        full = obj.convert_to_training_input(full_output, ["a", "a", "b", "b"])
        if streaming:
            obj._begin_streaming_step()
            for group in reversed(groups):
                obj._stream_training_groups([group])
            await obj._run_training(full)
            if objective == "topd_stopd":
                assert obj.stopd.step == 1
        else:
            full = score(full)
            full = obj.apply_reward_kl_penalty(full)
            full = obj.compute_advantages_and_returns(full)
            obj._normalize_advantages(full, [(0, 4)], full.metadata["policy_prompt_boundaries"])
            backward("policy", full)
            step("policy")
        results.append(
            (parameter.detach().clone(), parameter.grad.clone(), optimizer.state[parameter]["exp_avg"].clone())
        )
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("displacement", [0.0, 0.7])
def test_async_topd_without_stopd_reuses_native_clipped_behavior_objective(displacement):
    from examples.train.on_policy_distillation.topd import async_topd_policy_loss

    cfg = config(
        **{
            "trainer.algorithm.stopd.enabled": False,
            "trainer.algorithm.async_opd": True,
            "trainer.algorithm.policy_loss_type": "async_topd",
            "trainer.fully_async.enabled": True,
            "trainer.fully_async.mode": "one_step_off",
            "trainer.fully_async.max_staleness_steps": 1,
        }
    )
    data = batch()
    advantage, _, _ = compute_topd_advantage(data["rewards"], data["loss_mask"], [(0, 2), (2, 4)])
    current = (data["action_log_probs"] + displacement).requires_grad_()
    behavior = data["rollout_logprobs"] - 0.3
    actual, _ = async_topd_policy_loss(
        current, data["action_log_probs"], advantage, cfg.trainer.algorithm, data["loss_mask"], behavior
    )
    regular = copy.copy(cfg.trainer.algorithm)
    regular.policy_loss_type = "regular"
    expected, _ = ppo_policy_loss(current, behavior, advantage, regular, data["loss_mask"], behavior)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, current, retain_graph=True)[0], torch.autograd.grad(expected, current)[0]
    )
    assert cfg.trainer.algorithm.policy_loss_type == "async_topd"
