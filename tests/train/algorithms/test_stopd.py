"""STOPD expectation, native loss reduction, generation, and resume contracts."""

import copy
import itertools
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput

from examples.train.on_policy_distillation.main_on_policy_distill import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationExp,
    OnPolicyDistillationTrainer,
    STOPDGenerator,
    compute_no_op_advantage,
    validate_stopd_cfg,
)
from examples.train.on_policy_distillation.stopd import (
    STOPD,
    request_seed,
    token_weights,
)
from examples.train.rwkv.benchmark_stopd import run_config, summarize
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    importance_sampling_loss,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import prepare_generator_input
from skyrl.train.trainer import RayPPOTrainer


def config(**overrides):
    defaults = {
        "trainer.algorithm.advantage_estimator": "no_op",
        "trainer.algorithm.policy_loss_type": "importance_sampling",
        "trainer.algorithm.use_kl_in_reward": True,
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.stopd.enabled": True,
        "trainer.algorithm.stopd.block": 2,
        "trainer.algorithm.stopd.warmup_steps": 0,
        "trainer.algorithm.stopd.initial_denominator": 11,
        "trainer.train_batch_size": 4,
        "trainer.policy_mini_batch_size": 4,
        "generator.n_samples_per_prompt": 1,
        "generator.sampling_params.max_generate_length": 8,
        "generator.max_input_length": 512,
    }
    cfg = OnPolicyDistillationConfig.from_cli_overrides(defaults | overrides)
    validate_stopd_cfg(cfg)
    return cfg


def test_expected_gradient_and_token_count_match_complete_rollouts():
    lengths = torch.tensor([3, 8])
    features = torch.arange(1, 17, dtype=torch.float64).reshape(2, 8) / 16
    full_mask = torch.arange(8)[None, :] < lengths[:, None]
    expected = (features * full_mask).sum() / lengths.sum()
    gradients, estimates = [], []
    for cuts in itertools.product([2, 4, 6, 8], repeat=2):
        mask = torch.arange(8)[None, :] < torch.minimum(lengths, torch.tensor(cuts))[:, None]
        stopd = STOPD(8, 2, 0, 11)
        parameter = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
        loss = (parameter * features * stopd.multiplier(mask, mask.sum())).sum() / mask.sum()
        gradients.append(torch.autograd.grad(loss, parameter)[0])
        estimates.append(stopd.end_step(mask)["n_hat"])
    torch.testing.assert_close(torch.stack(gradients).mean(), expected, rtol=1e-7, atol=1e-7)
    assert sum(estimates) / len(estimates) == pytest.approx(11, rel=1e-7)


@pytest.mark.parametrize("length", [1, 3, 8, 10])
def test_partial_last_block_is_unbiased(length):
    expected = torch.zeros(length)
    for anchor in range(10):
        cutoff = min((anchor // 4 + 1) * 4, 10)
        mask = torch.arange(length)[None, :] < min(length, cutoff)
        expected += token_weights(mask, 10, 4)[0] / 10
    torch.testing.assert_close(expected, torch.ones(length))


def test_padding_does_not_change_response_positions():
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]])
    weights = token_weights(mask, 8, 2)
    torch.testing.assert_close(weights[0, 2:], weights[1, :4])
    assert not weights[~mask.bool()].any()
    assert weights[0, 2].item() == 1
    assert weights[0, -1].item() == pytest.approx(8 / 6)


def test_warmup_lagged_denominator_and_checkpoint_roundtrip():
    stopd = STOPD(8, 2, warmup_steps=2, seed=91)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert stopd.budgets(32) == [8] * 32
    stopd.end_step(mask)
    assert stopd.denominator is None
    stopd.end_step(mask)
    assert stopd.denominator == 6
    assert len(set(stopd.budgets(32))) > 1
    denominator = stopd.denominator
    stats = stopd.end_step(mask)
    assert stats["denominator"] == denominator
    assert stopd.denominator == stats["n_hat"]
    resumed = STOPD(8, 2, warmup_steps=2, seed=91)
    resumed.load_state_dict(stopd.state_dict())
    assert resumed.budgets(32) == stopd.budgets(32)
    torch.testing.assert_close(resumed.multiplier(mask, 6), stopd.multiplier(mask, 6))
    assert resumed.end_step(mask) == stopd.end_step(mask)


@pytest.mark.parametrize("key,value", [("version", 0), ("seed", 99), ("block", 4), ("denominator", 0)])
def test_incompatible_checkpoint_is_rejected(key, value):
    stopd = STOPD(8, 2, 0, 11)
    state = stopd.state_dict() | {key: value}
    with pytest.raises(ValueError):
        stopd.load_state_dict(state)


@pytest.mark.parametrize("pad_size", [0, 1])
@pytest.mark.parametrize("dp_size,micro_batch", [(1, 4), (2, 1), (2, 2), (4, 1)])
def test_native_opd_gradient_is_invariant_to_dp_microbatches_and_padding(monkeypatch, pad_size, dp_size, micro_batch):
    # This tests the local reduction, independent of registry synchronization.
    monkeypatch.setattr(AdvantageEstimatorRegistry, "sync_with_actor", classmethod(lambda cls: None))
    monkeypatch.setitem(AdvantageEstimatorRegistry._functions, "no_op", compute_no_op_advantage)
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config()
    trainer.all_metrics = {}
    trainer.stopd = STOPD(8, 2, 0, 11)
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
    rewards = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 10
    data = TrainingInputBatch({"loss_mask": mask.float(), "response_mask": mask, "rewards": rewards, "values": None})
    data.metadata = {"uids": [str(i) for i in range(4)], "avg_response_length": 2.5, "pad_size": pad_size}
    data = trainer.compute_advantages_and_returns(data)
    data = trainer._normalize_advantages(data, [(0, 4)])
    parameter = torch.tensor(0.1, requires_grad=True)
    features = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 20
    losses = []
    per_rank = 4 // dp_size
    for rank in range(dp_size):
        for start in range(rank * per_rank, (rank + 1) * per_rank, micro_batch):
            end = min(start + micro_batch, (rank + 1) * per_rank)
            loss, _ = importance_sampling_loss(
                parameter * features[start:end],
                torch.zeros_like(features[start:end]),
                data["advantages"][start:end],
                trainer.cfg.trainer.algorithm,
                loss_mask=mask[start:end],
            )
            # Native workers multiply SUM losses by DP size before FSDP's mean.
            losses.append(loss * dp_size / dp_size)
    gradient = torch.autograd.grad(sum(losses), parameter)[0]
    real_mask = mask.bool().clone()
    if pad_size:
        real_mask[-pad_size:] = False
    reference = -(torch.exp(parameter * features) * rewards * token_weights(real_mask, 8, 2)).sum() / 11
    expected_gradient = torch.autograd.grad(reference, parameter)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert trainer.stopd.end_step(real_mask)["tokens"] == real_mask.sum().item()


@pytest.mark.parametrize(
    "field,value",
    [
        ("generator.max_turns", 2),
        ("generator.batched", True),
        ("generator.apply_overlong_filtering", True),
        ("trainer.update_epochs_per_batch", 2),
        ("trainer.algorithm.advantage_batch_normalize", True),
        ("trainer.algorithm.loss_reduction", "sequence_mean"),
    ],
)
def test_unsupported_training_configuration_is_rejected(field, value):
    with pytest.raises(ValueError, match="STOPD"):
        config(**{field: value})


def test_disabled_mode_uses_native_generator_and_masks(monkeypatch):
    cfg = config(**{"trainer.algorithm.stopd.enabled": False})
    assert not cfg.generator.preserve_truncated_action_mask
    sentinel = object()
    monkeypatch.setattr("skyrl.train.entrypoints.main_base.BasePPOExp.get_generator", lambda *args: sentinel)
    experiment = object.__new__(OnPolicyDistillationExp)
    assert experiment.get_generator(cfg, None, None) is sentinel


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("warmup_steps", [0, 25])
async def test_request_budgets_and_seeds_preserve_eval_and_inputs(monkeypatch, phase, warmup_steps):
    generator = object.__new__(STOPDGenerator)
    generator.stopd = STOPD(8, 2, warmup_steps, 11, seed=42)
    inputs, _ = prepare_generator_input(
        [
            {"prompt": [{"role": "user", "content": "question"}], "uid": str(i), "env_class": "aime", "env_extras": {}}
            for i in range(32)
        ],
        1,
        {"max_tokens": 8, "temperature": 1},
        "aime",
        phase,
        5,
    )
    original = copy.deepcopy(inputs)

    async def agent_loop(_self, _prompt, _env_class, env_extras, max_tokens, max_input_length, **kwargs):
        assert not any(key.startswith("_stopd") for key in env_extras)
        return {"budget": max_tokens, "sampling": kwargs["sampling_params"]}

    async def generate(self, batch, disable_tqdm=False):
        return [
            await self.agent_loop(
                batch["prompts"][i],
                batch["env_classes"][i],
                batch["env_extras"][i],
                8,
                512,
                sampling_params=batch["sampling_params"],
                training_phase=phase,
            )
            for i in range(32)
        ]

    monkeypatch.setattr(SkyRLGymGenerator, "generate", generate)
    monkeypatch.setattr(SkyRLGymGenerator, "agent_loop", agent_loop)
    result = await generator.generate(inputs)
    assert inputs == original
    if phase == "eval":
        assert all(row == {"budget": 8, "sampling": original["sampling_params"]} for row in result)
    else:
        assert len({row["sampling"]["seed"] for row in result}) == 32
        assert result[0]["sampling"]["seed"] == request_seed(42, 5, 0, 0, "rollout")
        assert all(row["sampling"]["max_tokens"] == row["budget"] for row in result)
        assert {row["budget"] for row in result} == ({8} if warmup_steps else {2, 4, 6, 8})


@pytest.mark.asyncio
async def test_forced_cut_retains_exact_sampled_tokens_logprobs_and_masks(monkeypatch):
    cfg = config()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.eos_token = "<eos>"
    tokenizer.apply_chat_template.return_value = [1, 2, 0]
    tokenizer.encode.return_value = []
    tokenizer.decode.return_value = "partial answer"
    engine = MagicMock()
    engine.finish_session = AsyncMock()

    async def generate(batch, model=None):
        budget = batch["sampling_params"]["max_tokens"]
        return {
            "responses": ["partial answer"],
            "response_ids": [[7] * budget],
            "response_logprobs": [[-0.5] * budget],
            "stop_reasons": ["length"],
        }

    engine.generate = AsyncMock(side_effect=generate)
    env = MagicMock()
    env.strict_reward = True
    env.init.side_effect = lambda messages: (messages, {})
    env.get_metrics.return_value = {}
    env.step.return_value = BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})
    monkeypatch.setattr("skyrl.train.generators.skyrl_gym_generator.skyrl_gym.make", lambda *args, **kwargs: env)
    generator = STOPDGenerator(
        cfg.generator, cfg.environment.skyrl_gym, engine, tokenizer, stopd_cfg=cfg.trainer.algorithm.stopd
    )
    inputs, _ = prepare_generator_input(
        [{"prompt": [{"role": "user", "content": "question"}], "uid": "row", "env_class": "aime", "env_extras": {}}],
        1,
        {"max_tokens": 8, "temperature": 1, "logprobs": 1},
        "aime",
        "train",
        1,
    )
    output = await generator.generate(inputs, disable_tqdm=True)
    budget = engine.generate.call_args.args[0]["sampling_params"]["max_tokens"]
    assert output["response_ids"] == [[7] * budget]
    assert output["rollout_logprobs"] == [[-0.5] * budget]
    assert output["loss_masks"] == [[1] * budget]
    assert output["stop_reasons"] == ["length"]


def test_resume_restores_dataloader_and_stopd_state(tmp_path, monkeypatch):
    state = STOPD(8, 2, 0, 11, seed=0)
    state.step = 50
    torch.save(state.state_dict(), tmp_path / "stopd_state.pt")
    torch.save({"position": 50}, tmp_path / "data.pt")
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (50, str(tmp_path)))
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.stopd = STOPD(8, 2, 0, 999)
    trainer.train_dataloader = MagicMock()
    assert trainer.load_checkpoints() == (50, str(tmp_path))
    trainer.train_dataloader.load_state_dict.assert_called_once_with({"position": 50})
    assert trainer.stopd.state_dict() == state.state_dict()


def test_import_vanilla_checkpoint_requires_its_preceding_token_count(tmp_path, monkeypatch):
    torch.save({"position": 50}, tmp_path / "data.pt")
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (50, str(tmp_path)))
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.stopd = STOPD(8, 2, 25)
    trainer.train_dataloader = MagicMock()
    with pytest.raises(ValueError, match="initial_denominator"):
        trainer.load_checkpoints()
    trainer.stopd = STOPD(8, 2, 25, 12345)
    trainer.load_checkpoints()
    assert trainer.stopd.step == 50
    assert trainer.stopd.denominator == 12345


def test_warmup_preserves_native_advantage_dtype(monkeypatch):
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.stopd = STOPD(8, 2, 25)
    data = {"advantages": torch.ones(2, 3, dtype=torch.bfloat16)}
    monkeypatch.setattr(RayPPOTrainer, "compute_advantages_and_returns", lambda self, batch: batch)
    assert trainer.compute_advantages_and_returns(data) is data
    assert data["advantages"].dtype == torch.bfloat16


def test_stopd_state_is_saved_before_native_checkpoint_publication(tmp_path, monkeypatch):
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config()
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    trainer.global_step = 51
    trainer.stopd = STOPD(8, 2, 25, 11)
    trainer.stopd.step = 51

    def save_native(self):
        saved = torch.load(tmp_path / "global_step_51/stopd_state.pt", weights_only=False)
        assert saved == self.stopd.state_dict()
        return "complete"

    monkeypatch.setattr(RayPPOTrainer, "save_checkpoints", save_native)
    assert trainer.save_checkpoints() == "complete"


def test_invalid_dataloader_resume_is_not_silently_ignored(tmp_path, monkeypatch):
    torch.save({"position": 50}, tmp_path / "data.pt")
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (50, str(tmp_path)))
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.stopd = STOPD(8, 2, 25, 11)
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.load_state_dict.side_effect = ValueError("invalid sampler state")
    with pytest.raises(ValueError, match="invalid sampler state"):
        trainer.load_checkpoints()


def test_benchmark_replays_complete_configuration_and_checkpoint(tmp_path, monkeypatch):
    original = asdict(config())
    original["trainer"]["policy"]["model"]["path"] = "rwkv7-g1k-1.5b-20260930-ctx25600"
    calls = []
    monkeypatch.setattr(
        "examples.train.rwkv.benchmark_stopd.subprocess.run", lambda *args, **kwargs: calls.append(args)
    )
    source = tmp_path / "global_step_50"
    for mode in ("vanilla", "stopd"):
        run_config(tmp_path, original, tmp_path / mode, mode, source, 50, 3, 2, 0, 1234)
        # Parse the exact root-object overrides given to the native CLI.
        replay = OnPolicyDistillationConfig.from_cli_overrides(calls[-1][0][2:])
        validate_stopd_cfg(replay)
        assert replay.trainer.resume_mode == "from_path"
        assert replay.trainer.resume_path == str(source)
        assert replay.trainer.max_training_steps == 53
        assert replay.trainer.policy.optimizer_config == config().trainer.policy.optimizer_config
        assert replay.trainer.algorithm.stopd.initial_denominator == 1234
        assert replay.trainer.algorithm.stopd.warmup_steps == (54 if mode == "vanilla" else 25)
    assert original["trainer"]["resume_path"] is None


def test_benchmark_requires_complete_finite_updates_and_native_state_restore(tmp_path):
    lines = ["Successfully loaded optimizer state", "Successfully loaded dataloader state"]
    for _ in range(3):
        for stage in ("step", "generate", "fwd_logprobs_values_reward", "policy_train", "sync_weights"):
            lines.append(f"Finished: '{stage}', time cost: 3.00s")
        lines.append(str({"policy_entropy": 0.5, "final_loss": 0.1, "grad_norm": 0.2}))
    log = tmp_path / "console.log"
    log.write_text("\n".join(lines))
    assert summarize(tmp_path, 3)["timings"]["step"]["mean"] == 3
    log.write_text("\n".join(lines[1:]))
    with pytest.raises(ValueError, match="restore complete training state"):
        summarize(tmp_path, 3)
