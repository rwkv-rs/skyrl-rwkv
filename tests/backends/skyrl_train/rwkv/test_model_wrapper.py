from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_vllm_sampling_params,
)
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import SamplingParams, SkyRLTrainConfig


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
            "generator.sampling_params.additional_kwargs={presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}",
            "generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}",
        ]
    )

    assert config.generator.chat_template_kwargs == {
        "rwkv_prompt_template": "bot",
        "rwkv_generation_prompt": "open_think",
    }
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
