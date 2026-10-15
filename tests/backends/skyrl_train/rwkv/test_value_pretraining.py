import json
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.model_wrapper import (
    _get_critic_model,
    get_llm_for_sequence_regression,
)
from skyrl.train.dataset.value_dataset import (
    RewardDataCollator,
    ValueDataset,
    value_loss,
)
from skyrl.train.main_value import evaluate


def write_dataset(tmp_path, rows):
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_value_dataset_terminal_reward_and_tool_observation_masks(tmp_path):
    path = write_dataset(
        tmp_path,
        [{"input_ids": [1, 2, 3, 4, 5, 6], "loss_mask": [0, 0, 1, 0, 0, 1], "reward": -2.0}],
    )
    dataset = ValueDataset(path, max_length=6)
    batch = RewardDataCollator(0)([dataset[0]])
    assert batch["num_actions"] == 5
    assert batch["response_mask"].tolist() == [[False, True, False, False, True]]
    assert batch["labels"].tolist() == [[-2.0] * 5]
    predictions = torch.zeros((1, 5), requires_grad=True)
    error, count = value_loss(predictions, batch["labels"], batch["response_mask"])
    assert (error / count).item() == 4.0
    (error / count).backward()
    assert predictions.grad.tolist() == [[0, 2, 0, 0, 2]]


def test_value_dataset_drops_overlength_without_changing_terminal_targets(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            {"input_ids": [1, 2], "loss_mask": [0, 1], "reward": 1.0},
            {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "reward": 0.0},
        ],
    )
    dataset = ValueDataset(path, max_length=2)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [1, 2]
    assert dataset[0]["returns"] == [1.0, 1.0]


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"loss_mask": [1, 1, 1]}, "start with a prompt"),
        ({"loss_mask": [0, 1]}, "same length"),
        ({"loss_mask": [0, 2, 1]}, "binary"),
        ({"attention_mask": [0, 1, 1]}, "unpadded"),
        ({"truncated": True}, "bootstrap"),
        ({"reward": float("nan")}, "finite"),
        ({"returns": [0, 1]}, "per input token"),
    ],
)
def test_value_dataset_rejects_invalid_targets(tmp_path, fields, message):
    row = {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "reward": 1.0, **fields}
    with pytest.raises(ValueError, match=message):
        ValueDataset(write_dataset(tmp_path, [row]), max_length=8)


def test_value_dataset_mixes_completed_and_bootstrapped_rows(tmp_path):
    rows = [
        {"input_ids": [1, 2], "loss_mask": [0, 1], "reward": 1.0, "truncated": False},
        {"input_ids": [3, 4], "loss_mask": [0, 1], "returns": [0, 0.75], "truncated": True},
    ]
    dataset = ValueDataset(write_dataset(tmp_path, rows), max_length=8)
    batch = RewardDataCollator(0)([dataset[0], dataset[1]])
    torch.testing.assert_close(batch["labels"], torch.tensor([[1.0], [0.75]]))


def test_value_dataset_accepts_explicit_bootstrapped_returns(tmp_path):
    path = write_dataset(
        tmp_path,
        [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "truncated": True, "returns": [0, 0.7, 0.8]}],
    )
    batch = RewardDataCollator(0)([ValueDataset(path, max_length=8)[0]])
    torch.testing.assert_close(batch["labels"], torch.tensor([[0.7, 0.8]]))


class ToyRwkvPreTrainedModel(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        torch.nn.init.ones_(self.value_head.weight)


class ToyRwkvModel(torch.nn.Module):
    """Causal recurrence makes left-padding contamination visible in tests."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, config.hidden_size)
        self.calls = []
        self.autocast_states = []

    def forward(self, input_ids, use_cache):
        assert not use_cache
        self.calls.append(input_ids.detach().clone())
        self.autocast_states.append(torch.is_autocast_enabled(input_ids.device.type))
        return {"last_hidden_state": self.embed_tokens(input_ids).cumsum(1)}


def make_critic(**kwargs):
    config = SimpleNamespace(model_type="rwkv", hidden_size=4)
    return _get_critic_model(ToyRwkvPreTrainedModel, ToyRwkvModel, **kwargs)(config)


def test_rwkv_critic_ragged_padding_alignment_and_backward():
    model = make_critic()
    features = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "returns": [0, 1, 1]},
        {"input_ids": [4, 5, 6, 7, 8], "loss_mask": [0, 0, 1, 0, 1], "returns": [0, 0, 0, 0, 0]},
    ]
    batch = RewardDataCollator(0)(features)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        predictions, outputs = model(
            batch["input_ids"], batch["num_actions"], batch["attention_mask"], return_output=True
        )
    assert len(model.model.calls) == 1
    assert model.model.calls[0].tolist() == [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]
    assert model.model.autocast_states == [False]
    assert outputs["last_hidden_state"].shape == (2, 5, 4)
    for index, row in enumerate(features):
        ids = torch.tensor(row["input_ids"])
        hidden = model.model.embed_tokens(ids).cumsum(0)
        expected = model.value_head(hidden).squeeze(-1)[:-1]
        torch.testing.assert_close(predictions[index, -(len(ids) - 1) :], expected)
    error, count = value_loss(predictions, batch["labels"], batch["response_mask"])
    (error / count).backward()
    assert torch.isfinite(model.model.embed_tokens.weight.grad).all()
    assert torch.isfinite(model.value_head.weight.grad).all()
    assert not batch["response_mask"][0, :2].any()


@pytest.mark.parametrize("kwargs", [{"remove_microbatch_padding": True}, {"sequence_parallel_size": 2}])
def test_rwkv_critic_rejects_incompatible_packing(kwargs):
    with pytest.raises(ValueError, match="RWKV does not support"):
        make_critic(**kwargs)


def test_rwkv_critic_checkpoint_round_trip_uses_native_bf16(tmp_path):
    from transformers import RwkvConfig, RwkvForCausalLM

    config = RwkvConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=1,
        intermediate_size=256,
        decay_low_rank_dim=64,
        a_low_rank_dim=64,
        gate_low_rank_dim=64,
        v_low_rank_dim=64,
    )
    source = tmp_path / "source"
    RwkvForCausalLM(config).save_pretrained(source)
    critic = get_llm_for_sequence_regression(str(source), "critic", bf16=False, use_flash_attention_2=True)
    assert all(parameter.dtype == torch.bfloat16 for parameter in critic.parameters())
    assert critic.config._attn_implementation == "eager"
    assert critic.config.wkv_mode == "fp32io16"
    with torch.no_grad():
        critic.value_head.weight.fill_(0.125)
    checkpoint = tmp_path / "critic"
    critic.save_pretrained(checkpoint)
    restored = get_llm_for_sequence_regression(str(checkpoint), "critic")
    assert not restored.config.use_cache
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_value_loss_is_global_token_normalized_across_microbatches():
    parameter = torch.tensor(0.5, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True]])
    full_error, count = value_loss(parameter.expand_as(labels), labels, mask)
    full_gradient = torch.autograd.grad(full_error / count, parameter)[0]
    split_loss = sum(value_loss(parameter.expand_as(row), row, row_mask)[0] for row, row_mask in zip(labels, mask))
    split_gradient = torch.autograd.grad(split_loss / count, parameter)[0]
    torch.testing.assert_close(full_gradient, split_gradient)


def test_evaluate_distributed_tail_metrics_use_each_trajectory_once(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    reductions = []

    def reduce_stats(stats):
        if reductions:
            stats += reductions[0]
        else:
            reductions.append(stats.clone())

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_stats)
    model = make_critic()
    with torch.no_grad():
        model.value_head.weight.zero_()
    dataset = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "returns": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [0, 1], "returns": [0, 0]},
        {"input_ids": [6, 7], "loss_mask": [0, 1], "returns": [0, 0]},
    ]
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    evaluate(model, dataset, RewardDataCollator(0), batch_size=1, device="cpu")
    rank_zero_calls = len(model.model.calls)
    model.model.calls.clear()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    metrics = evaluate(model, dataset, RewardDataCollator(0), batch_size=1, device="cpu")
    assert len(model.model.calls) == rank_zero_calls == 2
    assert metrics["eval/token_mse"] == 0.5
    assert metrics["eval/sequence_mse"] == pytest.approx(1 / 3)
    assert metrics["eval/explained_variance"] == 0.0


def test_evaluate_excludes_dummy_tail_rows_and_preserves_training_mode(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _: None)
    model = make_critic()
    with torch.no_grad():
        model.value_head.weight.zero_()
    dataset = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "returns": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [0, 1], "returns": [0, 0]},
        {"input_ids": [6, 7], "loss_mask": [0, 1], "returns": [0, 0]},
    ]
    metrics = evaluate(model, dataset, RewardDataCollator(0), batch_size=2, device="cpu")
    assert metrics["eval/token_mse"] == 0.5
    assert metrics["eval/sequence_mse"] == pytest.approx(1 / 3)
    assert metrics["eval/explained_variance"] == 0.0
    assert model.training
