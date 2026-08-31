from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_vllm_sampling_params,
)
from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    load_rwkv_checkpoint_weights,
    refresh_rwkv_runtime_weights,
)
from skyrl.backends.skyrl_train.utils import torch_utils as torch_utils_module
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl.backends.skyrl_train.workers import model_wrapper as model_wrapper_module
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import SamplingParams, SkyRLTrainConfig


@pytest.fixture(autouse=True)
def use_torch_logprobs(monkeypatch):
    monkeypatch.setattr(torch_utils_module, "FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)


class ToyCausalLM(torch.nn.Module):
    def __init__(self, model_type: str = "rwkv") -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.lm_head = torch.nn.Linear(8, 32, bias=False)
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), kwargs))
        return {"logits": self.lm_head(self.embed_tokens(input_ids))}


class AutocastAwareToyCausalLM(ToyCausalLM):
    def __init__(self, model_type: str) -> None:
        super().__init__(model_type=model_type)
        self.autocast_states = []
        self.training_states = []

    def forward(self, input_ids, **kwargs):
        self.autocast_states.append(torch.is_autocast_enabled(input_ids.device.type))
        self.training_states.append(self.training)
        return super().forward(input_ids, **kwargs)


class ToyRwkvAttention(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        for name in ("w1", "a1", "g1", "w2", "a2", "g2", "v1", "v2"):
            self.register_parameter(name, torch.nn.Parameter(torch.randn(2, 3)))
        for name in (
            "w1_canonical",
            "a1_canonical",
            "g1_canonical",
            "w2_canonical",
            "a2_canonical",
            "g2_canonical",
            "v1_canonical",
            "v2_canonical",
        ):
            self.register_buffer(name, torch.full((3, 2), torch.nan))
        self.register_buffer("layer_zero_v0", torch.full((2,), torch.nan))
        self.register_buffer("layer_zero_v1_runtime", torch.full((2, 3), torch.nan))
        self.register_buffer("layer_zero_v2_runtime", torch.full((3, 2), torch.nan))


class ToyRwkvLayer(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.linear_attn = ToyRwkvAttention(layer_idx)
        self.mlp = torch.nn.Module()
        self.mlp.value = torch.nn.Module()
        self.mlp.value.weight = torch.nn.Parameter(torch.empty(4, 2))


class ToyVllmRwkv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="rwkv")
        self.model = torch.nn.Module()
        self.model.config = SimpleNamespace(layer_norm_epsilon=1e-5)
        self.model.embed_tokens = torch.nn.Embedding(5, 2, dtype=torch.float16)
        self.model.embedding_norm = torch.nn.LayerNorm(2, dtype=torch.bfloat16)
        self.model.layers = torch.nn.ModuleList([ToyRwkvLayer(0), ToyRwkvLayer(1)])
        self.model.start_layer = 0
        self.model.end_layer = 2
        self.model._embedding_norm_folded = False

    def load_weights(self, weights):
        loaded = set()
        with torch.no_grad():
            for name, weight in weights:
                self.get_parameter(name).copy_(weight)
                loaded.add(name)
        return loaded


def expected_action_log_probs(model, sequences, attention_mask, num_actions):
    expected = []
    for sequence, mask in zip(sequences, attention_mask):
        valid_sequence = sequence[-int(mask.sum().item()) :].unsqueeze(0)
        logits = model(valid_sequence)["logits"]
        labels = torch.roll(valid_sequence, shifts=-1, dims=1)
        expected.append(logprobs_from_logits(logits, labels)[:, -num_actions - 1 : -1])
    return torch.cat(expected)


def test_rwkv_equal_length_sequences_share_one_forward_bucket():
    model = ToyCausalLM()
    wrapper = HFModelWrapper(model)
    sequences = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])

    actual = wrapper(sequences, num_actions=2, attention_mask=attention_mask)

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    model.calls.clear()
    expected = expected_action_log_probs(model, sequences, attention_mask, num_actions=2)
    assert torch.allclose(actual, expected)


def test_rwkv_ragged_batch_restores_padding_and_entropy_positions():
    model = ToyCausalLM()
    wrapper = HFModelWrapper(model)
    sequences = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8], [9, 10, 11, 12, 13]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])

    action_log_probs, output = wrapper(
        sequences,
        num_actions=2,
        attention_mask=attention_mask,
        compute_entropy=True,
        return_output=True,
    )

    assert [tuple(call[0].shape) for call in model.calls] == [(1, 3), (2, 5)]
    assert action_log_probs.shape == (3, 2)
    assert output["entropy"].shape == sequences.shape
    assert torch.equal(output["entropy"][1, :2], torch.zeros(2))


def test_rwkv_recurrent_forward_preserves_gradients():
    model = ToyCausalLM()
    wrapper = HFModelWrapper(model)
    sequences = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])

    wrapper(sequences, num_actions=2, attention_mask=attention_mask).sum().backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_rwkv_weight_reload_preserves_runtime_layout_and_derived_storage():
    model = ToyVllmRwkv()
    checkpoint_value = torch.arange(8, dtype=torch.float16).view(2, 4)
    checkpoint_embedding = torch.arange(10, dtype=torch.float16).view(5, 2)
    canonical = model.model.layers[1].linear_attn.w1_canonical
    canonical_data_ptr = canonical.data_ptr()

    loaded = load_rwkv_checkpoint_weights(
        model,
        [
            ("model.layers.0.mlp.value.weight", checkpoint_value),
            ("model.embed_tokens.weight", checkpoint_embedding),
        ],
    )
    refresh_rwkv_runtime_weights(
        model,
        lambda embedding, weight, bias, eps: embedding.to(torch.float16) + 1,
    )

    assert loaded == {
        "model.layers.0.mlp.value.weight",
        "model.embed_tokens.weight",
    }
    assert torch.equal(model.model.layers[0].mlp.value.weight, checkpoint_value.T)
    assert torch.equal(model.model.embed_tokens.weight, checkpoint_embedding + 1)
    assert model.model._embedding_norm_folded
    assert canonical.data_ptr() == canonical_data_ptr
    assert torch.equal(canonical, model.model.layers[1].linear_attn.w1.T)
    assert torch.count_nonzero(model.model.layers[0].linear_attn.v1_canonical) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"remove_microbatch_padding": True}, "remove_microbatch_padding=true"),
        ({"sequence_parallel_size": 2}, "sequence parallelism"),
        ({"use_torch_compile": True}, "torch.compile"),
    ],
)
def test_rwkv_rejects_incompatible_forward_modes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HFModelWrapper(ToyCausalLM(), **kwargs)


def test_non_rwkv_model_keeps_direct_forward_path():
    model = ToyCausalLM(model_type="llama")
    wrapper = HFModelWrapper(model)
    sequences = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

    wrapper(sequences, num_actions=2, attention_mask=attention_mask)

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], sequences)
    assert torch.equal(model.calls[0][1]["attention_mask"], attention_mask)


def test_rwkv_disables_outer_autocast_without_changing_direct_forward():
    sequences = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(sequences)
    rwkv_model = AutocastAwareToyCausalLM("rwkv")
    direct_model = AutocastAwareToyCausalLM("llama")
    rwkv_model.eval()
    direct_model.eval()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        HFModelWrapper(rwkv_model)(sequences, num_actions=2, attention_mask=attention_mask)
        HFModelWrapper(direct_model)(sequences, num_actions=2, attention_mask=attention_mask)

    assert rwkv_model.autocast_states == [False]
    assert direct_model.autocast_states == [True]
    assert rwkv_model.training_states == [True]
    assert direct_model.training_states == [False]
    assert not rwkv_model.training


def test_rwkv_policy_does_not_write_tokenizer_padding_into_model_config():
    rwkv_wrapper = HFModelWrapper(ToyCausalLM("rwkv"))
    direct_wrapper = HFModelWrapper(ToyCausalLM("llama"))
    rwkv_worker = object.__new__(FSDPPolicyWorkerBase)
    direct_worker = object.__new__(FSDPPolicyWorkerBase)
    rwkv_worker.model = rwkv_wrapper
    direct_worker.model = direct_wrapper

    rwkv_worker._set_pad_token_id(0)
    direct_worker._set_pad_token_id(0)

    assert not hasattr(rwkv_wrapper.model.config, "pad_token_id")
    assert direct_wrapper.model.config.pad_token_id == 0


def test_rwkv_string_load_does_not_force_attention_implementation(monkeypatch):
    config = SimpleNamespace(
        model_type="rwkv",
        vision_config=None,
        use_cache=True,
        to_dict=dict,
    )
    model = ToyCausalLM()
    model.config = config
    captured_kwargs = {}

    monkeypatch.setattr(
        model_wrapper_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: config,
    )

    def fake_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(
        model_wrapper_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    wrapper = HFModelWrapper("rwkv-checkpoint", bf16=False)

    assert wrapper.is_recurrent
    assert "attn_implementation" not in captured_kwargs
    assert not hasattr(config, "_attn_implementation")


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        (
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "penalty_decay": 1.0,
            },
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "penalty_decay": 1.0,
            },
        ),
        (
            {
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.1,
                "penalty_decay": 0.988,
            },
            {
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.1,
                "penalty_decay": 0.988,
            },
        ),
    ],
)
def test_rwkv_sampling_parameters_are_forwarded_to_vllm(sampling_params, expected):
    passthrough_keys = {"presence_penalty", "frequency_penalty", "penalty_decay"}
    additional_kwargs = {
        key: value for key, value in sampling_params.items() if key in passthrough_keys
    }
    config = SamplingParams(
        max_generate_length=128,
        **{key: value for key, value in sampling_params.items() if key not in passthrough_keys},
        additional_kwargs=additional_kwargs,
    )

    actual = get_vllm_sampling_params(config)

    assert {key: actual[key] for key in expected} == expected


def test_rwkv_recipe_kwargs_parse_through_typed_config():
    config = SkyRLTrainConfig.from_cli_overrides(
        [
            "generator.chat_template_kwargs={rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}",
            "generator.inference_engine.model_dtype=float16",
            "generator.sampling_params.additional_kwargs={presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}",
            "generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}",
        ]
    )

    assert config.generator.chat_template_kwargs == {
        "rwkv_prompt_template": "bot",
        "rwkv_generation_prompt": "open_think",
    }
    assert config.generator.inference_engine.model_dtype == "float16"
    assert config.generator.sampling_params.additional_kwargs == {
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "penalty_decay": 1.0,
    }
    assert config.generator.eval_sampling_params.additional_kwargs == {
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "penalty_decay": 0.988,
    }
