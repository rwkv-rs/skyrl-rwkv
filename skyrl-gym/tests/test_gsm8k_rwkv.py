import skyrl_gym
import pytest
from omegaconf import DictConfig


def _make_env(ground_truth="42", strict_reward=True):
    return skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k", "strict_reward": strict_reward}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )


def test_strict_rwkv_reward_requires_thinking_and_eos():
    env = _make_env()
    response = "reasoning </think> \\boxed{42}"
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    result = env.step(response)
    assert result["reward"] == 1.0
    assert result["metadata"]["structural_format_valid"] is True


@pytest.mark.parametrize(
    "response, ended_eod, truncated",
    [
        ("reasoning \\boxed{42}", True, False),
        ("></think> \\boxed{42}", True, False),
        (">   </think> \\boxed{42}", True, False),
        ("reasoning </think> </think> \\boxed{42}", True, False),
        ("reasoning </think> The answer is 42", True, False),
        ("reasoning </think> $x = 42$", True, False),
        ("reasoning </think> \\boxed{}", True, False),
        ("reasoning </think> \\boxed{42", True, False),
        ("reasoning </think> \\boxed{42}", False, False),
        ("reasoning </think> \\boxed{42}", True, True),
    ],
)
def test_strict_rwkv_reward_rejects_invalid_termination(response, ended_eod, truncated):
    env = _make_env()
    env.set_generation_metadata(
        action=response,
        ended_eod=ended_eod,
        truncated=truncated,
        stop_reason="length" if truncated else "stop",
    )
    assert env.step(response)["reward"] == 0.0


@pytest.mark.parametrize(
    "answer, ground_truth",
    [
        ("42.0", "42"),
        (r"\frac{84}{2}", "42"),
        ("6^2+6", "42"),
        (r"\sqrt{1764}", "42"),
        (r"\frac{1}{2}", "0.5"),
        ("1,000", "1000"),
        ("$42$", "42"),
    ],
)
def test_strict_rwkv_reward_uses_mathematical_equivalence(answer, ground_truth):
    env = _make_env(ground_truth)
    response = f">reasoning </think> \\boxed{{{answer}}}"
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    result = env.step(response)
    assert result["reward"] == 1.0
    assert result["metadata"]["answer_parseable"] is True
    assert result["metadata"]["is_correct"] is True
    assert env.get_metrics() == result["metadata"]


@pytest.mark.parametrize(
    "response, expected",
    [
        (r"reasoning </think> #### 42", 1.0),
        (r"reasoning </think> \boxed{41} then \boxed{42}", 1.0),
        (r"reasoning </think> \boxed{42} then \boxed{41}", 0.0),
        (r"reasoning \boxed{42} </think> \boxed{41}", 0.0),
        (r"reasoning </think> \boxed{41} followed by $x=42$", 0.0),
    ],
)
def test_strict_rwkv_reward_only_verifies_the_strictly_extracted_answer(response, expected):
    env = _make_env()
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    assert env.step(response)["reward"] == expected


def test_strict_rwkv_reward_handles_math_parse_errors(monkeypatch):
    def fail_parse(*args, **kwargs):
        raise ValueError("invalid mathematical expression")

    monkeypatch.setattr("math_verify.parse", fail_parse)
    env = _make_env()
    response = r"reasoning </think> \boxed{42}"
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    result = env.step(response)
    assert result["reward"] == 0.0
    assert result["metadata"]["answer_parseable"] is False


@pytest.mark.parametrize("answer, expected", [("42", 1.0), ("42.0", 0.0)])
def test_nonstrict_rwkv_reward_keeps_string_matching(answer, expected):
    env = _make_env(strict_reward=False)
    assert env.step(f"\\boxed{{{answer}}}")["reward"] == expected
