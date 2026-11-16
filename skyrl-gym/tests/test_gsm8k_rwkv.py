import skyrl_gym
import pytest
from omegaconf import DictConfig


def _make_env(ground_truth="42", strict_reward=True):
    return skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k", "strict_reward": strict_reward}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )


def test_generation_metadata_does_not_run_symbolic_scoring(monkeypatch):
    env = _make_env()
    calls = []

    def fail_score(*args, **kwargs):
        calls.append(args)
        raise AssertionError("metadata setup must not judge a response")

    monkeypatch.setattr(env, "_score_strict_once", fail_score)
    env.set_generation_metadata(
        action="reasoning </think> \\boxed{42}", ended_eod=False, truncated=True, stop_reason="length"
    )
    assert calls == []
    assert env._generation_metadata["truncated"] is True


def test_isolated_scoring_reuses_worker_and_preserves_rewards(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from skyrl_gym.envs.gsm8k import isolated_scoring

    monkeypatch.setenv("SKYRL_MATH_ARCHIVE", str(tmp_path))
    isolated_scoring._shutdown_workers()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for answer, expected in (("42", 1.0), ("41", 0.0)):
                reward, details = executor.submit(
                    isolated_scoring.score,
                    f"reasoning </think> \\boxed{{{answer}}}",
                    "42",
                    ended_eod=True,
                    truncated=False,
                ).result(timeout=30)
                assert reward == expected
                assert details["is_correct"] is bool(expected)
        events = [json.loads(line) for line in (tmp_path / "scoring.jsonl").read_text().splitlines()]
        scored = [event for event in events if event["event"] == "scored"]
        assert len(scored) == 2
        assert scored[0]["worker_pid"] == scored[1]["worker_pid"]
        assert [event["worker_requests"] for event in scored] == [1, 2]
    finally:
        isolated_scoring._shutdown_workers()


def test_isolated_scoring_timeout_is_immediate_zero_without_retry(tmp_path, monkeypatch):
    import json
    from skyrl_gym.envs.gsm8k import isolated_scoring

    monkeypatch.setenv("SKYRL_MATH_ARCHIVE", str(tmp_path))
    monkeypatch.setattr(isolated_scoring, "_SCORING_TIMEOUT_SECONDS", 0.1)
    isolated_scoring._shutdown_workers()
    monkeypatch.setattr(
        isolated_scoring,
        "run_memory_guarded",
        lambda command, request, rss_bytes: {
            "returncode": -9,
            "reason": "timeout",
            "stdout": "",
            "stderr": "",
            "seconds": 0.1,
            "peak_rss_bytes": 0,
            "worker_pid": 123,
            "worker_requests": 1,
        },
    )
    try:
        reward, details = isolated_scoring.score(
            "reasoning </think> \\boxed{42}",
            "42",
            ended_eod=True,
            truncated=False,
        )
        assert reward == 0.0
        assert details["scoring_timeout"] is True
        events = [json.loads(line) for line in (tmp_path / "scoring.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events if event["event"].startswith("scored")] == ["scored_timeout"]
        assert not any(event["event"] == "retry" for event in events)
    finally:
        isolated_scoring._shutdown_workers()


@pytest.mark.parametrize("base,exponent,expected", [(2, 10**80 - 1, 7), (3, 10**100, 42)])
def test_huge_integer_power_has_exact_modular_mismatch(base, exponent, expected):
    import sympy
    from skyrl_gym.envs.gsm8k.isolated_scoring import integer_modular_counterexample

    expression = sympy.Pow(base, sympy.Integer(exponent), evaluate=False)
    witness = integer_modular_counterexample(sympy.Integer(expected), expression)
    assert witness["kind"] == "exact_integer_modular_counterexample"
    assert witness["expected_residue"] == expected % witness["modulus"]
    assert witness["actual_residue"] == pow(base, exponent, witness["modulus"])
    assert witness["expected_residue"] != witness["actual_residue"]


def test_modular_certificate_never_claims_equality_or_rejects_unsupported_expressions():
    import sympy
    from skyrl_gym.envs.gsm8k.isolated_scoring import integer_modular_counterexample

    assert integer_modular_counterexample(sympy.Integer(8), sympy.Pow(2, 3, evaluate=False)) is None
    assert integer_modular_counterexample(sympy.Integer(8), sympy.Integer(8 + 101 * 103 * 107 * 109)) is None
    assert integer_modular_counterexample(sympy.Integer(1), sympy.Rational(1000001, 1000000)) is None
    assert integer_modular_counterexample(sympy.Integer(1), sympy.sin(sympy.Symbol("x"))) is None
    assert integer_modular_counterexample(sympy.Integer(1), sympy.Pow(0, 0, evaluate=False)) is None
    assert integer_modular_counterexample(sympy.Integer(1), sympy.Pow(2, -1, evaluate=False)) is None


def test_isolated_scoring_huge_power_does_not_expand_integer(tmp_path, monkeypatch):
    import json
    from skyrl_gym.envs.gsm8k import isolated_scoring

    monkeypatch.setenv("SKYRL_MATH_ARCHIVE", str(tmp_path))
    isolated_scoring._shutdown_workers()
    try:
        reward, details = isolated_scoring.score(
            r"reasoning </think> \boxed{2^{10^{80}-1}}",
            "7",
            ended_eod=True,
            truncated=False,
        )
        assert reward == 0.0 and details["answer_parseable"] and not details["is_correct"]
        event = json.loads((tmp_path / "scoring.jsonl").read_text().splitlines()[-1])
        assert event["event"] == "scored" and event["attempt"] == 1
        assert event["peak_rss_bytes"] < 256 * 1024**2
        result = json.loads(event["stdout"])
        assert result["certificates"][0]["kind"] == "exact_integer_modular_counterexample"
    finally:
        isolated_scoring._shutdown_workers()


def test_strict_rwkv_reward_requires_thinking_and_eos():
    env = _make_env()
    response = "reasoning </think> \\boxed{42}"
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    result = env.step(response)
    assert result["reward"] == 1.0
    assert result["metadata"]["structural_format_valid"] is True


def test_isolated_scoring_accepts_pipe_descriptors_above_select_limit(tmp_path, monkeypatch):
    import fcntl
    import os
    import resource
    from skyrl_gym.envs.gsm8k import isolated_scoring

    monkeypatch.setenv("SKYRL_MATH_ARCHIVE", str(tmp_path))
    isolated_scoring._shutdown_workers()
    original_popen = isolated_scoring.subprocess.Popen
    descriptors = []
    previous_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    if previous_limits[1] < 2049:
        pytest.skip("high-descriptor test requires a hard file limit above 2048")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(previous_limits[0], 2049), previous_limits[1]))

    def high_descriptor_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        descriptor = fcntl.fcntl(process.stdout.fileno(), fcntl.F_DUPFD_CLOEXEC, 2048)
        process.stdout.close()
        process.stdout = os.fdopen(descriptor, "rb", buffering=0)
        descriptors.append(descriptor)
        return process

    monkeypatch.setattr(isolated_scoring.subprocess, "Popen", high_descriptor_popen)
    try:
        reward, details = isolated_scoring.score(
            "reasoning </think> \\boxed{42}",
            "42",
            ended_eod=True,
            truncated=False,
        )
        assert descriptors and min(descriptors) >= 2048
        assert reward == 1.0 and details["is_correct"] is True
    finally:
        isolated_scoring._shutdown_workers()
        resource.setrlimit(resource.RLIMIT_NOFILE, previous_limits)


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


@pytest.mark.parametrize(
    "answer, ground_truth, expected",
    [
        ("(3, 4)", "(3,4)", 1.0),
        ("34", "(3,4)", 0.0),
        ("(4, 3)", "(3,4)", 0.0),
        ("(5, 21, 421)", "(5,21,421)", 1.0),
        ("521421", "(5,21,421)", 0.0),
        (r"\left(\frac{1}{2},\frac{3}{4}\right)", "(0.5,0.75)", 1.0),
        (r"\{2, 1\}", r"\{1,2\}", 1.0),
        ("12", r"\{1,2\}", 0.0),
        (r"100000000\pi", r"100,000,000\pi", 1.0),
        (r"100,000,000\pi", r"100000000\pi", 1.0),
        (r"\frac{1,000}{2}", "500", 1.0),
        (r"\frac{1}{1,000}", "0.001", 1.0),
        (r"\sqrt{1,000,000}", "1000", 1.0),
        ("(123,456)", "(123, 456)", 1.0),
        ("123456", "(123,456)", 0.0),
        (r"\{123,456\}", r"\{456,123\}", 1.0),
        ("123456", r"\{123,456\}", 0.0),
        (r"100{,}000{,}000\pi", r"100000000\pi", 1.0),
        (r"\{123{,}456\}", r"\{123,456\}", 1.0),
        (r"\frac{1{,}000}{2}", "500", 1.0),
    ],
)
def test_strict_rwkv_reward_preserves_mathematical_commas(answer, ground_truth, expected):
    env = _make_env(ground_truth)
    response = f">reasoning </think> \\boxed{{{answer}}}"
    env.set_generation_metadata(action=response, ended_eod=True, truncated=False, stop_reason="stop")
    result = env.step(response)
    assert result["metadata"]["extracted_answer"] == answer
    assert result["metadata"]["ground_truth_answer"] == ground_truth
    assert result["metadata"]["answer_parseable"] is True
    assert result["reward"] == expected


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


@pytest.mark.parametrize(
    "answer, ground_truth, expected", [("42", "42", 1.0), ("42.0", "42", 0.0), ("1,000", "1000", 1.0)]
)
def test_nonstrict_rwkv_reward_keeps_string_matching(answer, ground_truth, expected):
    env = _make_env(ground_truth, strict_reward=False)
    assert env.step(f"\\boxed{{{answer}}}")["reward"] == expected
