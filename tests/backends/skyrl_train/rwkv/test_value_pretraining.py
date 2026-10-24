import json
from types import SimpleNamespace

import pytest
import torch
from peft import PeftModel

from skyrl.backends.skyrl_train.workers.model_wrapper import (
    _get_critic_model,
    get_llm_for_sequence_regression,
)
from skyrl.train import main_value
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


def test_value_checkpoint_retention_keeps_recent_steps_and_best(tmp_path):
    for name in ["global_step_2", "global_step_50", "global_step_100", "global_step_150", "best"]:
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    main_value.cleanup_old_checkpoints(str(tmp_path), -1)
    assert len(list(tmp_path.iterdir())) == 5
    main_value.cleanup_old_checkpoints(str(tmp_path), 2)
    assert {path.name for path in tmp_path.iterdir()} == {"global_step_100", "global_step_150", "best"}
    assert (tmp_path / "best/model.safetensors").read_bytes() == b"checkpoint"


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

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        self.calls.append(input_ids.detach().clone())
        self.autocast_states.append(torch.is_autocast_enabled(input_ids.device.type))
        hidden_states = self.embed_tokens(input_ids).cumsum(1)
        if past_key_values is not None:
            hidden_states = hidden_states + past_key_values
        return {"last_hidden_state": hidden_states, "past_key_values": hidden_states[:, -1:]}


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
    assert model.model.calls[0].shape == (2, 16)
    assert model.model.calls[0][:, :5].tolist() == [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]
    assert not model.model.calls[0][:, 5:].any()
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


def test_probability_value_head_preserves_alignment_and_gradient():
    model = make_critic().to(torch.bfloat16)
    batch = RewardDataCollator(0)(
        [
            {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "returns": [0, 1, 1]},
            {"input_ids": [4, 5], "loss_mask": [0, 1], "returns": [0, 0]},
        ]
    )
    raw = main_value.critic_forward(model, batch, "cpu")[0].detach()
    model.config.value_head_calibration = {"scale": 0.25, "bias": -0.5}
    predictions, labels, mask = main_value.critic_forward(model, batch, "cpu")
    torch.testing.assert_close(predictions, torch.sigmoid(raw.float() * 0.25 - 0.5))
    assert predictions.dtype == torch.float32
    assert ((predictions >= 0) & (predictions <= 1)).all()
    error, count = value_loss(predictions, labels, mask, "sequence_mean")
    (error / count).backward()
    assert torch.isfinite(model.value_head.weight.grad).all()
    assert model.value_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [{"remove_microbatch_padding": True}, {"sequence_parallel_size": 2}])
def test_rwkv_critic_rejects_incompatible_packing(kwargs):
    with pytest.raises(ValueError, match="RWKV does not support"):
        make_critic(**kwargs)


@pytest.fixture
def rwkv_checkpoint(tmp_path):
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
    return source


def test_rwkv_critic_checkpoint_round_trip_uses_native_bf16(tmp_path, rwkv_checkpoint):
    source = rwkv_checkpoint
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


@pytest.mark.parametrize("rank", [8, 16])
@pytest.mark.parametrize("freeze_backbone", [False, True])
@pytest.mark.parametrize("target_modules", [["v_proj", "o_proj"], ["mlp.key", "mlp.value"]])
def test_rwkv_nora_critic_updates_and_exports_full_head(
    tmp_path, rwkv_checkpoint, monkeypatch, rank, freeze_backbone, target_modules
):
    from transformers import RwkvModel

    from skyrl.backends.skyrl_train.distributed import fsdp_strategy
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.train.config import FSDPConfig, ModelConfig, SkyRLLoraConfig

    def causal_forward(self, input_ids, use_cache=True, past_key_values=None):
        """Exercise real NoRA projections on CPU without the CUDA recurrence."""
        hidden = self.embed_tokens(input_ids).cumsum(1)
        for layer in self.layers:
            hidden = hidden + layer.linear_attn.o_proj(torch.tanh(layer.linear_attn.v_proj(hidden)))
            hidden = hidden + layer.mlp.value(torch.relu(layer.mlp.key(hidden)).square())
        return {"last_hidden_state": hidden, "past_key_values": None}

    monkeypatch.setattr(RwkvModel, "forward", causal_forward)
    model = get_llm_for_sequence_regression(
        str(rwkv_checkpoint),
        "critic",
        use_nora=True,
        lora_rank=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        freeze_backbone=freeze_backbone,
    )
    if freeze_backbone:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.001)
    head = model.get_base_model().value_head
    assert head.weight.dtype == torch.float32
    assert head.weight.requires_grad
    assert model.peft_config["default"].use_nora
    assert model.peft_config["default"].target_modules == set(target_modules)
    assert model.peft_config["default"].modules_to_save == ["value_head"]
    frozen_before = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
    trainable_before = {name: value.clone() for name, value in model.named_parameters() if value.requires_grad}
    assert all("lora_" in name or ".value_head.modules_to_save." in name for name in trainable_before)
    if target_modules == ["mlp.key", "mlp.value"]:
        assert all(".mlp." in name or ".value_head.modules_to_save." in name for name in trainable_before)
    assert all(module.training for module in model.modules())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01, weight_decay=0)
    batch = RewardDataCollator(0)([{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1], "returns": [0, 1, 1, 1]}])
    for _ in range(3):
        predictions, labels, mask = main_value.critic_forward(model, batch, "cpu")
        error, count = value_loss(predictions, labels, mask)
        (error / count).backward()
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
        if not freeze_backbone:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n
            )
        optimizer.step()
        optimizer.zero_grad()
    for name, value in model.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
        else:
            assert not torch.equal(value, trainable_before[name]), name
    expected = main_value.critic_forward(model.eval(), batch, "cpu")[0].detach()
    strategy = FSDPStrategy(
        FSDPConfig(),
        model_config=ModelConfig(lora=SkyRLLoraConfig(rank=rank, use_nora=True)),
    )
    monkeypatch.setattr(strategy, "is_rank_0", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_strategy, "fsdp2_get_full_state_dict", lambda model, **kwargs: model.state_dict())
    adapter_path = tmp_path / "checkpoint"
    strategy._save_lora_adapters(model, str(adapter_path))
    adapter = adapter_path / "lora_adapter"
    restored_base = get_llm_for_sequence_regression(str(rwkv_checkpoint), "critic")
    restored_base.value_head.float()
    restored = PeftModel.from_pretrained(restored_base, str(adapter)).eval()
    torch.testing.assert_close(main_value.critic_forward(restored, batch, "cpu")[0], expected, rtol=0, atol=0)
    before_export = {name: value.clone() for name, value in model.state_dict().items()}
    output = tmp_path / "merged"
    strategy.save_hf_model(model, str(output), output_dtype=torch.bfloat16)
    restored = get_llm_for_sequence_regression(str(output), "critic").eval()
    assert not any("lora_" in name or "modules_to_save" in name for name in restored.state_dict())
    assert all(value.dtype == torch.bfloat16 for value in restored.parameters())
    torch.testing.assert_close(restored.value_head.weight, head.weight.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(
        main_value.critic_forward(restored, batch, "cpu")[0].float(), expected, rtol=0.03, atol=0.01
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_export[name], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 8])
def test_rwkv_head_only_critic_keeps_backbone_and_adapters_fixed(rwkv_checkpoint, monkeypatch, rank):
    from transformers import RwkvModel

    def causal_forward(self, input_ids, use_cache=True, past_key_values=None):
        hidden = self.embed_tokens(input_ids).cumsum(1)
        for layer in self.layers:
            hidden = hidden + layer.linear_attn.o_proj(torch.tanh(layer.linear_attn.v_proj(hidden)))
        return {"last_hidden_state": hidden, "past_key_values": None}

    monkeypatch.setattr(RwkvModel, "forward", causal_forward)
    model = get_llm_for_sequence_regression(
        str(rwkv_checkpoint),
        "critic",
        use_nora=bool(rank),
        lora_rank=rank,
        lora_alpha=rank,
        target_modules=["v_proj", "o_proj"],
        freeze_backbone=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    critic = model.get_base_model() if rank else model
    backbone = getattr(critic, critic.base_model_prefix)
    head = critic.value_head
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    head_before = head.weight.detach().clone()
    assert head.weight.dtype == torch.float32 and head.weight.requires_grad
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all("value_head" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    input_gradients = []
    hook = head.register_forward_pre_hook(lambda module, args: input_gradients.append(args[0].requires_grad))
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.001)
    batch = RewardDataCollator(0)([{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "returns": [1, 1, 1, 1]}])
    predictions, labels, mask = main_value.critic_forward(model, batch, "cpu")
    error, count = value_loss(predictions, labels, mask)
    (error / count).backward()
    assert input_gradients == [False]
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())
    optimizer.step()
    hook.remove()
    assert not torch.equal(head.weight, head_before)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_value_pretraining_cli_nora_rank_sweep(monkeypatch, rank):
    calls = []
    monkeypatch.setattr(main_value, "train", calls.append)
    main_value.main(
        [
            "--model",
            "rwkv7-test",
            "--train-data",
            "train.jsonl",
            "--val-data",
            "validation.jsonl",
            "--output-dir",
            f"rank-{rank}",
            "--use-nora",
            "--rank",
            str(rank),
        ]
    )
    args = calls[0]
    assert args.use_nora and args.rank == rank
    assert args.lora_alpha == rank
    assert args.target_modules == ["mlp.key", "mlp.value"]
    assert not args.freeze_backbone


def test_value_pretraining_cli_preserves_native_initialization_and_calibration(monkeypatch):
    calls = []
    monkeypatch.setattr(main_value, "train", calls.append)
    main_value.main(
        [
            "--model",
            "rwkv7-test",
            "--train-data",
            "train.jsonl",
            "--val-data",
            "validation.jsonl",
            "--output-dir",
            "value",
            "--resume-path",
            "/value/global_step_96",
            "--freeze-backbone",
            "--model-config-kwargs",
            '{"value_head_calibration":{"scale":10.0,"bias":-4.0}}',
        ]
    )
    assert calls[0].resume_path == "/value/global_step_96"
    assert calls[0].freeze_backbone
    assert calls[0].model_config_kwargs == {"value_head_calibration": {"scale": 10.0, "bias": -4.0}}


def test_probability_value_config_round_trips_without_changing_head_weights(rwkv_checkpoint):
    raw = get_llm_for_sequence_regression(str(rwkv_checkpoint), "critic")
    source = rwkv_checkpoint.parent / "critic-source"
    raw.save_pretrained(source)
    calibrated = get_llm_for_sequence_regression(
        str(source), "critic", model_config_kwargs={"value_head_calibration": {"scale": 10.0, "bias": -4.0}}
    )
    assert raw.config.value_head_calibration is None
    torch.testing.assert_close(raw.value_head.weight, calibrated.value_head.weight, rtol=0, atol=0)
    output = rwkv_checkpoint.parent / "calibrated"
    calibrated.save_pretrained(output)
    restored = get_llm_for_sequence_regression(str(output), "critic")
    assert restored.config.value_head_calibration == {"scale": 10.0, "bias": -4.0}
    torch.testing.assert_close(restored.value_head.weight, raw.value_head.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "calibration", [{"scale": -1, "bias": 0}, {"scale": float("nan"), "bias": 0}, {"scale": 1, "bias": float("inf")}]
)
def test_probability_value_config_rejects_nonfinite_or_reversed_scores(rwkv_checkpoint, calibration):
    with pytest.raises(ValueError, match="value_head_calibration"):
        get_llm_for_sequence_regression(
            str(rwkv_checkpoint), "critic", model_config_kwargs={"value_head_calibration": calibration}
        )


@pytest.mark.parametrize("option,value", [("--rank", "0"), ("--lora-alpha", "-1"), ("--fsdp-size", "0")])
def test_value_pretraining_cli_rejects_invalid_adapter_options(option, value):
    with pytest.raises(SystemExit, match="2"):
        main_value.main(
            [
                "--model",
                "rwkv7-test",
                "--train-data",
                "train.jsonl",
                "--val-data",
                "validation.jsonl",
                "--output-dir",
                "value",
                "--use-nora",
                option,
                value,
            ]
        )


def test_rwkv_value_pretraining_preserves_native_config(tmp_path, monkeypatch):
    path = write_dataset(tmp_path, [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "reward": 1.0}])
    model = make_critic()
    model.config.pad_token_id = None
    loads = []

    def load_model(*args, **kwargs):
        loads.append(kwargs)
        return model

    def prepare(pair):
        assert pair[0] is model
        assert model.config.pad_token_id is None
        raise RuntimeError("model setup complete")

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda _: None)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(main_value, "get_tokenizer", lambda _: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(main_value, "get_llm_for_sequence_regression", load_model)
    monkeypatch.setattr(
        main_value,
        "FSDPStrategy",
        lambda *args, **kwargs: SimpleNamespace(setup_distributed=lambda: None, prepare=prepare),
    )
    with pytest.raises(RuntimeError, match="model setup complete"):
        main_value.main(
            ["--model", "rwkv7-test", "--train-data", path, "--val-data", path, "--output-dir", "value", "--use-nora"]
        )
    assert loads[0]["use_nora"]
    assert loads[0]["lora_rank"] == loads[0]["lora_alpha"] == 8


def test_value_loss_is_global_token_normalized_across_microbatches():
    parameter = torch.tensor(0.5, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True]])
    full_error, count = value_loss(parameter.expand_as(labels), labels, mask)
    full_gradient = torch.autograd.grad(full_error / count, parameter)[0]
    split_loss = sum(value_loss(parameter.expand_as(row), row, row_mask)[0] for row, row_mask in zip(labels, mask))
    split_gradient = torch.autograd.grad(split_loss / count, parameter)[0]
    torch.testing.assert_close(full_gradient, split_gradient)


@pytest.mark.parametrize("reduction", ["sequence_mean", "seq_mean_token_sum_norm"])
def test_sequence_value_loss_matches_online_critic_across_microbatches(reduction):
    from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_critic_loss
    from skyrl.train.config import AlgorithmConfig

    parameter = torch.tensor(0.3, requires_grad=True)
    labels = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[False, False, True], [True, True, True]])
    predictions = parameter.expand_as(labels)
    config = AlgorithmConfig(value_clip=None, critic_loss_reduction=reduction, max_seq_len=8)
    native_loss, _ = ppo_critic_loss(predictions, predictions.detach(), labels, config, mask)
    native_gradient = torch.autograd.grad(native_loss, parameter)[0]
    error, count = value_loss(predictions, labels, mask, reduction, max_seq_len=8)
    assert count == 2
    torch.testing.assert_close(0.5 * error / count, native_loss)
    split_error = sum(
        value_loss(parameter.expand_as(row), row, row_mask, reduction, max_seq_len=8)[0]
        for row, row_mask in zip(labels, mask)
    )
    split_gradient = torch.autograd.grad(0.5 * split_error / count, parameter)[0]
    torch.testing.assert_close(split_gradient, native_gradient)


def test_fixed_critic_normalization_removes_future_length_bias_at_shared_state():
    from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_critic_loss
    from skyrl.train.config import AlgorithmConfig

    # Two equally likely futures share only their initial state: success after
    # one action or failure after ten. Later states already predict their targets.
    value = torch.tensor(0.5, requires_grad=True)
    predictions = torch.cat([value.expand(2, 1), torch.zeros(2, 9)], dim=-1)
    labels = torch.zeros(2, 10)
    labels[0, 0] = 1
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 1:] = False
    fixed, _ = ppo_critic_loss(
        predictions,
        predictions.detach(),
        labels,
        AlgorithmConfig(value_clip=None, critic_loss_reduction="seq_mean_token_sum_norm", max_seq_len=10),
        mask,
    )
    sequence, _ = ppo_critic_loss(predictions, predictions.detach(), labels, AlgorithmConfig(value_clip=None), mask)
    assert torch.autograd.grad(fixed, value, retain_graph=True)[0].item() == 0
    assert torch.autograd.grad(sequence, value)[0].item() < 0


@pytest.mark.parametrize("budget", [None, 0, -1])
def test_fixed_value_loss_requires_explicit_positive_budget(monkeypatch, budget):
    monkeypatch.setattr(main_value, "train", lambda _: pytest.fail("invalid CLI reached training"))
    args = [
        "--model",
        "rwkv7-test",
        "--train-data",
        "train.jsonl",
        "--val-data",
        "validation.jsonl",
        "--output-dir",
        "value",
        "--loss-reduction",
        "seq_mean_token_sum_norm",
    ]
    if budget is not None:
        args += ["--max-seq-len", str(budget)]
    with pytest.raises(SystemExit):
        main_value.main(args)


def test_fixed_value_loss_cli_preserves_input_and_loss_budgets(monkeypatch):
    calls = []
    monkeypatch.setattr(main_value, "train", calls.append)
    main_value.main(
        [
            "--model",
            "rwkv7-test",
            "--train-data",
            "train.jsonl",
            "--val-data",
            "validation.jsonl",
            "--output-dir",
            "value",
            "--loss-reduction",
            "seq_mean_token_sum_norm",
            "--max-seq-len",
            "8192",
            "--max-length",
            "32768",
        ]
    )
    assert calls[0].max_length == 32768
    assert calls[0].max_seq_len == 8192


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
    assert metrics["eval/token_constant_mse"] == 0.25
    assert metrics["eval/token_mse_skill"] == -1.0
    assert metrics["eval/sequence_constant_mse"] == pytest.approx(2 / 9)
    assert metrics["eval/sequence_mse_skill"] == pytest.approx(-0.5)
    assert metrics["eval/sequence_target_mean"] == pytest.approx(1 / 3)
    assert metrics["eval/sequence_value_mean"] == 0.0


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


def test_evaluate_reports_domain_and_initial_action_calibration(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _: None)
    model = make_critic()
    with torch.no_grad():
        model.value_head.weight.zero_()
    dataset = [
        {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "returns": [0, 0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [0, 1], "returns": [0, 0]},
        {"input_ids": [6, 7], "loss_mask": [0, 1], "returns": [0, 0]},
    ]
    metrics = evaluate(
        model, dataset, RewardDataCollator(0), batch_size=2, device="cpu", domains=["Math", "Math", "Code"]
    )
    assert metrics["eval/sequences"] == 3
    assert metrics["eval/Math/sequences"] == 2
    assert metrics["eval/Code/sequences"] == 1
    assert metrics["eval/Math/sequence_mse_skill"] == -1
    assert metrics["eval/Math/initial_target_mean"] == 0.5
    assert metrics["eval/Math/initial_value_mean"] == 0
    assert metrics["eval/Math/initial_mse_skill"] == -1
    assert metrics["eval/initial_mse"] == pytest.approx(1 / 3)
    assert metrics["eval/Code/token_mse"] == 0
