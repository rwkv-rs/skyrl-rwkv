import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("The answer is #### 42", "42", 1.0),
        ("The final answer is \\boxed{42}", "42", 1.0),
        ("The answer is \\boxed{42}", "43", 0.0),
        ("The answer is #### 42", "43", 0.0),
        # answer is not in the expected format
        ("The answer is 42", "42", 0.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(output)
    assert step_output["reward"] == expected


def _make_strict_env():
    return skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k", "strict_reward": True}),
        extras={"reward_spec": {"method": "rule", "ground_truth": "42"}},
    )


def test_strict_reward_requires_complete_thinking_response():
    env = _make_strict_env()
    env.set_generation_metadata(
        action="reasoning </think> \\boxed{42}",
        ended_eod=True,
        truncated=False,
        stop_reason="stop",
    )

    step_output = env.step("reasoning </think> \\boxed{42}")

    assert step_output["reward"] == 1.0
    assert step_output["metadata"]["structural_format_valid"] is True
    assert step_output["metadata"]["ended_eod"] is True


def test_strict_reward_does_not_count_open_think_completion_as_reasoning():
    env = _make_strict_env()
    response = "></think> \\boxed{42}"
    env.set_generation_metadata(
        action=response,
        ended_eod=True,
        truncated=False,
        stop_reason="stop",
    )

    step_output = env.step(response)

    assert step_output["reward"] == 0.0
    assert step_output["metadata"]["thought_nonempty"] is False


@pytest.mark.parametrize(
    "response, ended_eod, truncated",
    [
        ("reasoning \\boxed{42}", True, False),
        ("reasoning </think> \\boxed{41}", True, False),
        ("reasoning </think> \\boxed{42}", False, False),
        ("reasoning </think> \\boxed{42}", True, True),
        ("reasoning </think> answer </think> \\boxed{42}", True, False),
    ],
)
def test_strict_reward_rejects_incomplete_or_invalid_response(response, ended_eod, truncated):
    env = _make_strict_env()
    env.set_generation_metadata(
        action=response,
        ended_eod=ended_eod,
        truncated=truncated,
        stop_reason="length" if truncated else "stop",
    )

    assert env.step(response)["reward"] == 0.0
