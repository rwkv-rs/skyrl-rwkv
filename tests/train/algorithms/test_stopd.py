"""STOPD expectation, native loss reduction, generation, and resume contracts."""

import copy
import itertools
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from examples.train.on_policy_distillation.main_on_policy_distill import (
    FullyAsyncOnPolicyDistillationTrainer,
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
from examples.train.rwkv.train.opd.benchmark_stopd import (
    run_config,
    summarize,
    validate_training_input,
)
from examples.train.rwkv.eval.evaluate import set_generation_budgets
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_vllm_sampling_params,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    importance_sampling_loss,
)
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import _AsyncStalenessManager
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import prepare_generator_input
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.async_utils import BackgroundFailure
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


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


@pytest.mark.parametrize("model_type", ["rwkv", "qwen3"])
@pytest.mark.parametrize("explicit_stop", [None, [], ["custom stop"]])
def test_rwkv_model_termination_defaults_preserve_sampling_overrides(monkeypatch, model_type, explicit_stop):
    cfg = config()
    cfg.generator.sampling_params.stop = copy.deepcopy(explicit_stop)
    cfg.generator.eval_sampling_params.stop = copy.deepcopy(explicit_stop)
    stops = ["✿", "\nUser:", "\n### User"]
    monkeypatch.setattr("skyrl.train.entrypoints.main_base.get_tokenizer", MagicMock())
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda *a, **k: SimpleNamespace(model_type=model_type)
    )
    generation_config = MagicMock(return_value=SimpleNamespace(stop_strings=stops))
    monkeypatch.setattr("transformers.GenerationConfig.from_pretrained", generation_config)
    for method in ("get_train_dataset", "get_eval_dataset", "get_colocate_pg"):
        monkeypatch.setattr(BasePPOExp, method, lambda self: None)

    BasePPOExp(cfg)

    expected = stops if model_type == "rwkv" and explicit_stop is None else explicit_stop
    assert cfg.generator.sampling_params.stop == expected
    assert cfg.generator.eval_sampling_params.stop == expected
    assert generation_config.call_count == (1 if model_type == "rwkv" else 0)
    for sampling_params in (cfg.generator.sampling_params, cfg.generator.eval_sampling_params):
        request = get_vllm_sampling_params(sampling_params)
        assert request["stop"] == expected
        assert request["include_stop_str_in_output"] is True


def test_lighteval_budgets_use_exact_producer_messages_and_rwkv_template():
    docs = [SimpleNamespace(query="short"), SimpleNamespace(query="long")]
    model = SimpleNamespace(
        prompt_manager=SimpleNamespace(prepare_prompt_api=lambda doc: [{"role": "user", "content": doc.query}]),
        config=SimpleNamespace(
            extra_body={"chat_template_kwargs": {"rwkv_prompt_template": "bot", "rwkv_generation_prompt": "open_think"}}
        ),
    )
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = [list(range(100)), list(range(2400))]
    strip_template = MagicMock(side_effect=lambda messages: messages)
    set_generation_budgets(docs, model, tokenizer, 25600, strip_template)
    assert [doc.generation_size for doc in docs] == [25500, 23200]
    tokenizer.apply_chat_template.assert_any_call(
        [{"role": "user", "content": "long"}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        rwkv_prompt_template="bot",
        rwkv_generation_prompt="open_think",
    )
    assert strip_template.call_count == 2


def test_lighteval_rejects_prompt_that_consumes_context():
    model = MagicMock()
    model.config.extra_body = {"chat_template_kwargs": {}}
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1] * 25600
    with pytest.raises(ValueError, match="prompt exceeds"):
        set_generation_budgets([SimpleNamespace()], model, tokenizer, 25600, lambda value: value)


@pytest.mark.asyncio
async def test_native_opd_eval_is_preserved_when_lighteval_disabled(monkeypatch):
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config()
    native = AsyncMock(return_value={"eval/all/avg_score": 0.5})
    monkeypatch.setattr(RayPPOTrainer, "eval", native)
    assert await trainer.eval() == {"eval/all/avg_score": 0.5}
    native.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [0, 1])
async def test_lighteval_uses_live_pool_without_changing_training_or_stopd(monkeypatch, tmp_path, returncode):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config(**{"trainer.algorithm.lighteval.project": "/producer"})
    trainer.cfg.trainer.export_path = str(tmp_path)
    trainer.cfg.trainer.num_logger_eval_samples = 1
    trainer.global_step = 100
    trainer.tracker = MagicMock()
    trainer.inference_engine_client = SimpleNamespace(
        proxy_url="http://localhost:8000", server_urls=["http://localhost:8001"], model_name="rwkv7-test"
    )
    before = asdict(trainer.cfg)
    output = tmp_path / "lighteval/global_step_100"

    async def run(*args, **kwargs):
        import json

        (output / "eval_metrics.json").write_text(json.dumps({"eval/lighteval/aime24/avg@32": 0.5}))
        completions = output / "completions.jsonl"
        completions.write_text(
            json.dumps(
                {
                    "task_name": "aime24|0",
                    "doc": {"choices": ["25"]},
                    "model_response": {
                        "rendered_prompt": "prompt",
                        "text": ["</think>\\boxed{25}"],
                        "text_post_processed": ["25"],
                        "finish_reasons": ["stop"],
                    },
                }
            )
            + "\n"
        )
        (output / "completed.json").write_text(json.dumps({"completions_path": str(completions)}))
        return SimpleNamespace(wait=AsyncMock(return_value=returncode))

    process = AsyncMock(side_effect=run)
    monkeypatch.setattr(
        "examples.train.on_policy_distillation.main_on_policy_distill.asyncio.create_subprocess_exec", process
    )
    scraper = MagicMock()
    if returncode:
        with pytest.raises(RuntimeError, match="producer.log"):
            await trainer.eval(vllm_metrics_scraper=scraper)
        trainer.tracker.log_samples_to_table.assert_not_called()
    else:
        assert await trainer.eval(vllm_metrics_scraper=scraper) == {"eval/lighteval/aime24/avg@32": 0.5}
        sample_log = trainer.tracker.log_samples_to_table.call_args
        assert sample_log.kwargs == {"step": 100, "incremental": True}
        assert sample_log.args[2][0][-2:] == ("25", "stop")
    scraper.resume.assert_called_once()
    scraper.pause.assert_called_once()
    args = process.call_args.args
    assert args[args.index("--url") + 1] == "http://localhost:8000"
    assert args[args.index("--context-length") + 1] == "25600"
    assert args[args.index("--num-samples") + 1] == "32"
    assert process.call_args.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert process.call_args.kwargs["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert asdict(trainer.cfg) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", [None, "startup", "evaluation"])
async def test_async_lighteval_lends_training_gpus_and_restores_them(monkeypatch, tmp_path, failure_stage):
    trainer = object.__new__(FullyAsyncOnPolicyDistillationTrainer)
    trainer.cfg = config(**{"trainer.algorithm.lighteval.project": "/producer"})
    trainer.cfg.trainer.export_path = str(tmp_path)
    trainer.global_step = 100
    trainer.tokenizer = object()
    trainer.inference_engine_client = SimpleNamespace(server_urls=["http://sampler"], model_name="rwkv7-test")
    trainer._staleness_manager = _AsyncStalenessManager(4, 4, 1)
    trainer._generation_failure = BackgroundFailure()
    trainer.dispatch = MagicMock()
    trainer.dispatch.get_worker_gpu_locations.return_value = [("node", gpu) for gpu in range(7)]
    events = []

    def export():
        assert trainer._staleness_manager._paused
        events.append("export")

    @contextmanager
    def offload():
        events.append("offload")
        try:
            yield
        finally:
            events.append("restore")

    trainer.save_models = export
    trainer.dispatch.offload_for_evaluation = offload
    pool = MagicMock()
    pool.get_server_urls.return_value = [f"http://learner{gpu}" for gpu in range(7)]

    def start():
        events.append("start")
        if failure_stage == "startup":
            raise RuntimeError("startup failed")

    pool.start.side_effect = start
    pool.shutdown.side_effect = lambda: events.append("shutdown")
    create_pool = MagicMock(return_value=pool)
    monkeypatch.setattr("examples.train.rwkv.eval.evaluate.create_eval_server_pool", create_pool)
    router = MagicMock()
    router.shutdown.side_effect = lambda: events.append("router_shutdown")
    client = SimpleNamespace(server_urls=["http://sampler"] + pool.get_server_urls())

    def build_client(cfg, tokenizer):
        assert cfg.generator.inference_engine.external_server_urls == client.server_urls
        assert cfg.trainer.policy.model.path == str(tmp_path / "global_step_100/policy")
        assert cfg.generator.inference_engine.served_model_name == "rwkv7-test"
        assert tokenizer is trainer.tokenizer
        return client, SimpleNamespace(router=router)

    monkeypatch.setattr("skyrl.backends.skyrl_train.inference_servers.setup.build_new_inference_client", build_client)

    async def evaluate(*args, inference_engine_client, **kwargs):
        assert trainer._staleness_manager._paused
        assert inference_engine_client is client
        assert events == ["export", "offload", "start"]
        events.append("evaluate")
        if failure_stage == "evaluation":
            raise RuntimeError("evaluation failed")
        return {"eval/lighteval/aime24/avg@32": 0.5}

    monkeypatch.setattr(OnPolicyDistillationTrainer, "eval", evaluate)
    before = asdict(trainer.cfg)
    if failure_stage:
        with pytest.raises(RuntimeError, match=failure_stage):
            await trainer.eval()
    else:
        assert await trainer.eval() == {"eval/lighteval/aime24/avg@32": 0.5, "eval/lighteval/num_engines": 8}
    assert events[-2:] == ["shutdown", "restore"]
    if failure_stage != "startup":
        assert events[-3] == "router_shutdown"
    assert not trainer._staleness_manager._paused
    assert asdict(trainer.cfg) == before
    assert create_pool.call_args.args[1] == [("node", gpu) for gpu in range(7)]


@pytest.mark.parametrize("alpha", [0.1, 0.2, 1.0])
def test_proximal_teacher_rewards_match_probability_mixture(alpha):
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config(**{"trainer.algorithm.proximal_teacher_alpha": alpha})
    trainer.stopd, trainer.all_metrics = None, {}
    student = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    teacher = torch.tensor([[0.05, 0.15, 0.8]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0]])
    data = TrainingInputBatch(
        {
            "loss_mask": mask,
            "response_mask": mask,
            "action_log_probs": student.log(),
            "base_action_log_probs": teacher.log(),
        }
    )
    data.metadata = {}
    actual = trainer.apply_reward_kl_penalty(data)["rewards"]
    expected = ((alpha * teacher + (1 - alpha) * student) / student).log() * mask
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    if alpha == 1:
        torch.testing.assert_close(actual, -(student.log() - teacher.log()) * mask, rtol=0, atol=0)


def test_proximal_teacher_extreme_ratios_are_finite_and_negative_reward_is_bounded():
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config(**{"trainer.algorithm.proximal_teacher_alpha": 0.2})
    trainer.stopd, trainer.all_metrics = None, {}
    data = TrainingInputBatch(
        {
            "loss_mask": torch.tensor([[1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]]),
            "action_log_probs": torch.tensor([[0.0, -1000.0, -4.0]]),
            "base_action_log_probs": torch.tensor([[-1000.0, 0.0, -8.0]]),
        }
    )
    data.metadata = {}
    actual = trainer.apply_reward_kl_penalty(data)["rewards"]
    assert torch.isfinite(actual).all()
    assert actual[0, 0] == pytest.approx(torch.tensor(0.8).log().item())
    assert actual[0, 1] == pytest.approx(1000 + torch.tensor(0.2).log().item())
    assert actual[0, 2] == 0


def test_opd_metrics_use_ipw_and_lagged_denominator_without_padding():
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config()
    trainer.stopd, trainer.all_metrics = STOPD(8, 2, 0, 11), {}
    mask = torch.ones(2, 4)
    data = TrainingInputBatch(
        {
            "loss_mask": mask,
            "response_mask": mask,
            "action_log_probs": torch.tensor([[-1.0, -2.0, -3.0, -4.0], [-100.0] * 4]),
            "base_action_log_probs": torch.zeros(2, 4),
        }
    )
    data.metadata = {"pad_size": 1}
    trainer.apply_reward_kl_penalty(data)
    expected = (1 + 2 + (3 + 4) * 8 / 6) / 11
    assert trainer.all_metrics["opd/reward_mean"] == pytest.approx(expected)
    assert trainer.all_metrics["opd/reward_abs_mean"] == pytest.approx(expected)
    assert trainer.all_metrics["opd/reverse_kl_mean"] == pytest.approx(-expected)
    assert trainer.stopd.denominator == 11


@pytest.mark.parametrize("alpha", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_proximal_teacher_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="proximal_teacher_alpha"):
        config(**{"trainer.algorithm.proximal_teacher_alpha": alpha})


def test_stopd_ipw_remains_unbiased_with_proximal_teacher_rewards():
    student = torch.linspace(0.1, 0.8, 8, dtype=torch.float64)
    teacher = student.flip(0)
    rewards = ((0.2 * teacher + 0.8 * student) / student).log()
    gradients = []
    for cutoff in [2, 4, 6, 8]:
        mask = torch.arange(8)[None, :] < cutoff
        stopd = STOPD(8, 2, 0, 8)
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss = -(parameter.exp() * rewards * stopd.weights(mask)).sum() / 8
        gradients.append(torch.autograd.grad(loss, parameter)[0])
    torch.testing.assert_close(torch.stack(gradients).mean(), -rewards.mean())


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
    boundaries = [(0, 4 - pad_size)]
    data.metadata = {
        "uids": [str(i) for i in range(4)],
        "avg_response_length": 2.5,
        "pad_size": pad_size,
        "policy_mini_batch_boundaries": boundaries,
    }
    data = trainer.compute_advantages_and_returns(data)
    data = trainer._normalize_advantages(data, boundaries)
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
@pytest.mark.parametrize("complete_rollouts", [False, True])
@pytest.mark.parametrize("grouped_budgets", [False, True])
async def test_request_budgets_and_seeds_preserve_eval_and_inputs(
    monkeypatch, phase, warmup_steps, complete_rollouts, grouped_budgets
):
    generator = object.__new__(STOPDGenerator)
    generator.complete_rollouts = complete_rollouts
    generator.async_generation = False
    generator.grouped_budgets = grouped_budgets
    generator.generator_cfg = SimpleNamespace(n_samples_per_prompt=2)
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
        assert {row["budget"] for row in result} == ({8} if warmup_steps or complete_rollouts else {2, 4, 6, 8})
        if grouped_budgets:
            assert all(result[i]["budget"] == result[i + 1]["budget"] for i in range(0, 32, 2))


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_length,budget,expected", [(3, 8, 8), (7, 8, 5), (7, 4, 4), (12, 8, None)])
async def test_stopd_context_cap_preserves_prompt_and_sampling(monkeypatch, prompt_length, budget, expected):
    generator = object.__new__(STOPDGenerator)
    generator.context_length = 12
    generator.generator_cfg = SimpleNamespace(chat_template_kwargs={"rwkv_prompt_template": "bot"})
    generator.tokenizer = MagicMock()
    generator.tokenizer.apply_chat_template.return_value = list(range(prompt_length))
    native = AsyncMock(return_value="result")
    monkeypatch.setattr(SkyRLGymGenerator, "agent_loop", native)
    prompt = [{"role": "user", "content": "question"}]
    call = generator.agent_loop(
        prompt, "aime", {"_stopd_budget": budget, "_stopd_seed": 42}, 8, 11, sampling_params={"temperature": 1}
    )
    if expected is None:
        with pytest.raises(ValueError, match="consumes"):
            await call
        native.assert_not_awaited()
    else:
        assert await call == "result"
        assert native.call_args.args[0] is prompt
        assert native.call_args.args[3] == expected
        assert native.call_args.kwargs["sampling_params"] == {"max_tokens": expected, "seed": 42, "temperature": 1}
        generator.tokenizer.apply_chat_template.assert_called_once_with(
            prompt, add_generation_prompt=True, tokenize=True, return_dict=False, rwkv_prompt_template="bot"
        )


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
    trainer.cfg = config()
    trainer.train_dataloader = MagicMock()
    assert trainer.load_checkpoints() == (50, str(tmp_path))
    trainer.train_dataloader.load_state_dict.assert_called_once_with({"position": 50})
    assert trainer.stopd.state_dict() == state.state_dict()


def test_import_vanilla_checkpoint_requires_its_preceding_token_count(tmp_path, monkeypatch):
    torch.save({"position": 50}, tmp_path / "data.pt")
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", lambda self: (50, str(tmp_path)))
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.stopd = STOPD(8, 2, 25)
    trainer.cfg = config()
    trainer.train_dataloader = MagicMock()
    with pytest.raises(ValueError, match="initial_denominator"):
        trainer.load_checkpoints()
    trainer.stopd = STOPD(8, 2, 25, 12345)
    trainer.load_checkpoints()
    assert trainer.stopd.step == 50
    assert trainer.stopd.denominator == 12345


def test_warmup_preserves_native_advantage_dtype(monkeypatch):
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config()
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
    trainer.cfg = config()
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.load_state_dict.side_effect = ValueError("invalid sampler state")
    with pytest.raises(ValueError, match="invalid sampler state"):
        trainer.load_checkpoints()


def test_benchmark_replays_complete_configuration_and_checkpoint(tmp_path, monkeypatch):
    original = asdict(config())
    original["trainer"]["policy"]["model"]["path"] = "rwkv7-g1k-1.5b-20260930-ctx25600"
    calls = []
    monkeypatch.setattr(
        "examples.train.rwkv.train.opd.benchmark_stopd.subprocess.run", lambda *args, **kwargs: calls.append(args)
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


@pytest.mark.parametrize(
    "backend_message",
    [
        "Successfully loaded optimizer state",
        "Checkpoint weights, optimizer and scheduler restored on 8 ranks",
    ],
)
def test_benchmark_requires_complete_finite_updates_and_native_state_restore(tmp_path, backend_message):
    lines = ["Successfully loaded dataloader state"]
    (tmp_path / "logs").mkdir()
    infra = tmp_path / "logs/infra-test.log"
    infra.write_text(backend_message)
    for _ in range(3):
        for stage in ("step", "generate", "fwd_logprobs_values_reward", "policy_train", "sync_weights"):
            lines.append(f"Finished: '{stage}', time cost: 3.00s")
        lines.append(str({"policy_entropy": 0.5, "final_loss": 0.1, "grad_norm": 0.2}))
    log = tmp_path / "console.log"
    log.write_text("\n".join(lines))
    assert summarize(tmp_path, 3)["timings"]["step"]["mean"] == 3
    infra.write_text("")
    with pytest.raises(ValueError, match="restore complete training state"):
        summarize(tmp_path, 3)


@pytest.mark.parametrize("alpha", [1.0, 0.2])
@pytest.mark.parametrize("corrupt", [None, "sequences", "rollout_logprobs", "loss_mask", "advantages"])
def test_actual_training_input_preserves_generated_prefix_and_ipw(tmp_path, corrupt, alpha):
    prompts, responses = [[10, 20], [11]], [[7, 8, 9], [5, 6, 7, 8]]
    logprobs = [[-0.1] * len(tokens) for tokens in responses]
    masks = [[1] * len(tokens) for tokens in responses]
    tensors = convert_prompts_responses_to_batch_tensors(0, prompts, responses, masks, masks, logprobs)
    data = TrainingInputBatch(
        dict(
            zip(
                ["sequences", "attention_mask", "response_mask", "rewards", "loss_mask", "rollout_logprobs"],
                tensors[:6],
                strict=True,
            )
        )
    )
    data.metadata = {"pad_size": 0, "policy_mini_batch_boundaries": [(0, 2)]}
    data["action_log_probs"] = data["rollout_logprobs"].clone()
    data["base_action_log_probs"] = data["action_log_probs"] + 0.1 * data["loss_mask"]
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = config(**{"trainer.algorithm.proximal_teacher_alpha": alpha})
    trainer.stopd, trainer.all_metrics = None, {}
    trainer.apply_reward_kl_penalty(data)
    data["returns"] = data["rewards"].clone()
    stopd = STOPD(8, 2, 0, 11)
    data["advantages"] = data["rewards"] * stopd.multiplier(data["loss_mask"], data["loss_mask"].sum())
    rows = [
        dict(prompt_token_ids=p, response_token_ids=r, rollout_logprobs=lp, loss_mask=m)
        for p, r, lp, m in zip(prompts, responses, logprobs, masks, strict=True)
    ]
    if corrupt:
        data[corrupt][0, -1] += 1
    data.pop("rewards")
    directory = tmp_path / "exports/dumped_data"
    directory.mkdir(parents=True)
    data.save(directory / "global_step_51_training_input.pkl")
    if corrupt:
        with pytest.raises((AssertionError, ValueError)):
            validate_training_input(tmp_path, rows, stopd, 51, alpha)
    else:
        assert validate_training_input(tmp_path, rows, stopd, 51, alpha) == {
            "rows": 2,
            "response_tokens": 7,
            "ipw_advantages_exact": True,
        }


@pytest.mark.asyncio
async def test_async_stopd_budgets_and_sampling_seeds_follow_uids(monkeypatch):
    generator = object.__new__(STOPDGenerator)
    generator.complete_rollouts = False
    generator.grouped_budgets = True
    generator.async_generation = True
    generator.generator_cfg = SimpleNamespace(n_samples_per_prompt=2)
    generator.stopd = STOPD(8192, 128, 0, 11, seed=42)

    async def capture(self, inputs, disable_tqdm=False):
        return inputs["env_extras"]

    monkeypatch.setattr(SkyRLGymGenerator, "generate", capture)
    prompts = [
        {"prompt": [{"role": "user", "content": "question"}], "uid": str(i), "env_class": "aime", "env_extras": {}}
        for i in range(24)
    ]

    def inputs(rows):
        return prepare_generator_input(rows, 2, {"max_tokens": 8192}, "aime", "train", 23)[0]

    together = await generator.generate(inputs(prompts))
    separate = []
    for row in prompts:
        separate.extend(await generator.generate(inputs([row])))
    assert together == separate
    budgets = [row["_stopd_budget"] for row in together]
    assert all(budgets[i] == budgets[i + 1] for i in range(0, len(budgets), 2))
    assert len(set(budgets)) > 8
    assert all(together[i] != together[i + 1] for i in range(0, len(together), 2))
