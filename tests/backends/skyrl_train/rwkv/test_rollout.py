import argparse
import asyncio
import importlib.util
import json
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[4] / "examples/train/rwkv/rollout.py"
spec = importlib.util.spec_from_file_location("rollout", SCRIPT)
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)


def test_rwkv_fake_think_answer_extraction():
    assert rollout.answer_text(">reason</think>\\boxed{4}") == r"\boxed{4}"
    assert rollout.answer_text(">plain answer") == "plain answer"


def test_text_match_uses_final_answer_and_option_mapping():
    assert rollout.text_score("Earlier I considered 4.\nFinal answer: 5", "5", "question")
    assert not rollout.text_score("Earlier answer: 4\nFinal answer: 5", "4", "question")
    query = "Which?\na: wrong\nb: The polynomial hierarchy collapses\nc: other"
    assert rollout.text_score("Final answer: b", "The polynomial hierarchy collapses", query)


def test_code_python_and_cpp_execution():
    fence = chr(96) * 3
    tests = {"inputs": [""], "outputs": ["3\n"]}
    assert rollout.run_code(">" + fence + "python\nprint(3)\n" + fence, tests) == (True, "ok")
    cpp = fence + 'cpp\n#include <iostream>\nint main(){std::cout << 3 << "\\n";}\n' + fence
    assert rollout.run_code(cpp, tests) == (True, "ok")


def test_long_context_selection_uses_token_budget(tmp_path: Path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": list(range(len(messages[0]["content"])))}

    domain = tmp_path / "Long_Context"
    domain.mkdir()
    path = domain / "part.jsonl"
    path.write_text(
        '{"uuid":"Long_Context_1","query":"123456","ground_truth":"5","domain":"Long_Context"}\n'
        '{"uuid":"Long_Context_2","query":"12","ground_truth":"2","domain":"Long_Context"}\n'
    )
    rows = list(rollout.rows(tmp_path, 4, Tokenizer(), 10, 5))
    assert [row["uuid"] for row in rows] == ["Long_Context_2"]


def test_write_output_keeps_question_ids_disjoint(tmp_path: Path):
    from collections import Counter

    groups = {
        "Math": {
            "Math_1": {
                "row": {"uuid": "Math_1", "query": "q", "ground_truth": "1"},
                "counts": Counter(correct=1, wrong=0, unanswered=0),
                "samples": {"correct": {"response": "1", "sample_index": 0, "reason": "math_verify"}},
            },
        },
        "Code": {
            "Code_1": {
                "row": {"uuid": "Code_1", "query": "q", "ground_truth": {"inputs": [""], "outputs": [""]}},
                "counts": Counter(correct=0, wrong=1, unanswered=0),
                "samples": {"wrong": {"response": "", "sample_index": 0, "reason": "wrong_output"}},
            },
        },
    }
    rollout.write_output(tmp_path, groups, 8192, 2)
    files = [tmp_path / f"{status}_samples.jsonl" for status in ("correct", "wrong", "unanswered")]
    ids = [line.split('"uuid": "')[1].split('"')[0] for file in files for line in file.read_text().splitlines()]
    assert len(ids) == len(set(ids))


def test_math_rollout_matches_strict_gsm8k_reward(tmp_path: Path, monkeypatch):
    from aiohttp import web

    import skyrl_gym

    valid = r">reasoning </think> \boxed{42}"
    cases = [
        (r">reasoning </think> \boxed{42.0}", [7, 0], "stop", "correct"),
        (r">reasoning </think> \boxed{\frac{84}{2}}", [7, 0], "stop", "correct"),
        (">reasoning </think> #### 42", [7, 0], "stop", "correct"),
        (r">reasoning </think> \boxed{43}", [7, 0], "stop", "wrong"),
        (r"\boxed{42}", [7, 0], "stop", "unanswered"),
        (r"> </think> \boxed{42}", [7, 0], "stop", "unanswered"),
        (">reasoning </think> $x = 42$", [7, 0], "stop", "unanswered"),
        (r">reasoning </think> \boxed{41} then \boxed{42}", [7, 0], "stop", "correct"),
        (valid, [7, 1], "stop", "unanswered"),
        (valid, [7, 0], "length", "unanswered"),
        (valid, None, "stop", "unanswered"),
        (valid, [], "stop", "unanswered"),
        (r">reasoning </think> </think> \boxed{42}", [7, 0], "stop", "unanswered"),
        (valid, [7, 0], "max_tokens", "unanswered"),
        (r">reasoning \boxed{42} </think> \boxed{43}", [7, 0], "stop", "wrong"),
        ("", [0], "stop", "unanswered"),
    ]
    for response, token_ids, finish, expected in cases:
        env = skyrl_gym.make(
            "gsm8k",
            env_config={"strict_reward": True},
            extras={"reward_spec": {"ground_truth": "42"}},
        )
        env.set_generation_metadata(
            action=response,
            ended_eod=bool(finish == "stop" and token_ids and token_ids[-1] == 0),
            truncated=finish in {"length", "max_tokens"},
            stop_reason=finish,
        )
        assert env.step(response)["reward"] == float(expected == "correct")

    tokenizer = SimpleNamespace(eos_token_id=0)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)),
    )
    for domain in ("Math", "Knowledge"):
        directory = tmp_path / domain
        directory.mkdir()
        row = {"uuid": f"{domain}_1", "query": domain, "domain": domain, "ground_truth": "42"}
        (directory / "part.jsonl").write_text(json.dumps(row) + "\n")

    choices = cycle(cases)
    requests = []

    async def complete(request):
        payload = await request.json()
        requests.append(payload)
        if payload["messages"][0]["content"] == "Math":
            response, token_ids, finish, _ = next(choices)
        else:
            response, token_ids, finish = "Final answer: 42", None, "stop"
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"content": response},
                        "token_ids": token_ids,
                        "finish_reason": finish,
                        "stop_reason": 0,
                    }
                ]
            }
        )

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        try:
            await rollout.run(
                argparse.Namespace(
                    model="test-rwkv",
                    data_root=tmp_path,
                    output=tmp_path / "output",
                    base_url=f"http://{host}:{port}",
                    limit=4,
                    concurrency=8,
                    max_model_len=16384,
                    max_tokens=8192,
                    timeout=10,
                )
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["Math"]["correct_rollouts"] == 16
    assert summary["Math"]["wrong_rollouts"] == 8
    assert summary["Math"]["unanswered_rollouts"] == 40
    assert summary["Knowledge"]["correct_rollouts"] == 64
    assert len(requests) == 128
    for payload in requests:
        is_math = payload["messages"][0]["content"] == "Math"
        assert payload["return_token_ids"] is is_math
        assert payload["chat_template_kwargs"]["rwkv_generation_prompt"] == ("open_think" if is_math else "fake_think")
