import asyncio
import json
import signal
import sys
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))
from examples.train.rwkv import rollout


class Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append((messages[0]["content"], kwargs))
        return {"input_ids": list(messages[0]["content"].encode())}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return bytes(token for token in ids if token != 0).decode()


@pytest.fixture
def tokenizer(monkeypatch):
    value = Tokenizer()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: value)),
    )
    return value


def dataset_row(root, domain, uuid, query, gold, part="part.jsonl"):
    directory = root / domain
    directory.mkdir(exist_ok=True)
    with (directory / part).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"uuid": uuid, "query": query, "ground_truth": gold, "domain": domain}, ensure_ascii=False)
            + "\n"
        )


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


def test_full_data_stream_preserves_oversized_rows_and_byte_offsets(tmp_path, tokenizer):
    dataset_row(tmp_path, "Long_Context", "Long_Context_1", "123456", "5")
    dataset_row(tmp_path, "Long_Context", "Long_Context_2", "二", "2")
    dataset_row(tmp_path, "Long_Context", "Long_Context_3", "3", "3", part="part2.jsonl")
    selected = list(rollout.rows(tmp_path, 0, tokenizer, 10, 5))
    assert [row["uuid"] for row in selected] == ["Long_Context_1", "Long_Context_2", "Long_Context_3"]
    assert [row["_prompt_too_long"] for row in selected] == [True, False, False]
    for row in selected:
        assert "ground_truth" not in row
        path, offset = row["_source"]
        with open(path, "rb") as handle:
            handle.seek(offset)
            assert json.loads(handle.readline())["uuid"] == row["uuid"]
    counts, reasons, _ = rollout.score_question(selected[0], [{"text": ""}] * 64, 0)
    assert counts == {"unanswered": 64}
    assert reasons == {"prompt_too_long": 64}


def test_explicit_limit_is_balanced_across_domains(tmp_path, tokenizer):
    for domain in rollout.DOMAINS:
        for index in range(3):
            dataset_row(tmp_path, domain, f"{domain}_{index}", str(index), "0")
    selected = list(rollout.rows(tmp_path, 6, tokenizer, 100, 5))
    assert Counter(row["domain"] for row in selected) == {"Math": 2, "Code": 2, "Long_Context": 1, "Knowledge": 1}
    completed = {f"{domain}_0" for domain in rollout.DOMAINS}
    remaining = list(rollout.rows(tmp_path, 6, tokenizer, 100, 5, completed=completed))
    assert [row["uuid"] for row in remaining] == ["Math_1", "Code_1"]
    assert len(tokenizer.prompts) == 8


def test_code_verdict_cache_counts_all_samples(tmp_path, tokenizer, monkeypatch):
    dataset_row(tmp_path, "Code", "Code_1", "print 3", {"inputs": [""], "outputs": ["3\n"]})
    row = next(rollout.rows(tmp_path, 0, tokenizer, 100, 5))
    calls = []

    def run_code(raw, gold):
        calls.append((raw, gold))
        return True, "ok"

    monkeypatch.setattr(rollout, "run_code", run_code)
    choices = [{"text": f">thought {index}</think> print(3)", "finish_reason": "stop"} for index in range(64)]
    counts, _, samples = rollout.score_question(row, choices, 0)
    assert counts == {"correct": 64}
    assert len(calls) == 1
    assert samples["correct"]["sample_index"] == 0
    assert rollout.score_question(row, choices, 0, collect_samples=False)[2] == {}


def test_write_output_keeps_question_ids_disjoint(tmp_path):
    groups = {
        domain: {
            "num_questions": 1,
            "counts": Counter(correct=64),
            "histogram": Counter({64: 1}),
            "reasons": Counter(ok=64),
        }
        for domain in ("Math", "Code")
    }
    candidates = {status: defaultdict(list) for status in rollout.STATUSES}
    candidates["correct"]["Math"] = [{"uuid": "Math_1", "response": "1"}]
    candidates["wrong"]["Code"] = [{"uuid": "Code_1", "response": "0"}]
    candidates["unanswered"]["Math"] = [{"uuid": "Math_1", "response": ""}]
    rollout.write_output(tmp_path, groups, candidates, {"limit": 0}, plot=True)
    files = [tmp_path / f"{status}_samples.jsonl" for status in rollout.STATUSES]
    ids = [json.loads(line)["uuid"] for file in files for line in file.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["Math"]["correct_count_histogram"] == {"64": 1}
    assert summary["Math"]["correct_counts_by_question_file"] == "correct_counts_by_question.jsonl"


def test_math_rollout_matches_strict_gsm8k_reward(tmp_path, tokenizer):
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
            "gsm8k", env_config={"strict_reward": True}, extras={"reward_spec": {"ground_truth": "42"}}
        )
        env.set_generation_metadata(
            action=response,
            ended_eod=bool(finish == "stop" and token_ids and token_ids[-1] == 0),
            truncated=finish in {"length", "max_tokens"},
            stop_reason=finish,
        )
        assert env.step(response)["reward"] == float(expected == "correct")
    for domain in ("Math", "Knowledge"):
        dataset_row(tmp_path, domain, f"{domain}_1", domain, "42")

    choices = cycle(cases)
    requests = []

    async def complete(request):
        payload = await request.json()
        requests.append(payload)
        if payload["prompt"] == list(b"Math"):
            response, token_ids, finish, _ = next(choices)
            ids = list(response.encode()) + [token_ids[-1]] if token_ids else token_ids
        else:
            ids, finish = list(b"Final answer: 42") + [0], "stop"
        return web.json_response({"choices": [{"text": "", "token_ids": ids, "finish_reason": finish}]})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        try:
            await rollout.run(
                rollout.parser.parse_args(
                    [
                        "--model",
                        "test-rwkv",
                        "--data-root",
                        str(tmp_path),
                        "--output",
                        str(tmp_path / "output"),
                        "--base-url",
                        f"http://{host}:{port}",
                        "--limit",
                        "4",
                        "--concurrency",
                        "8",
                        "--score-workers",
                        "1",
                        "--timeout",
                        "10",
                    ]
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
    assert all(payload["return_token_ids"] for payload in requests)
    assert {(query, kwargs["rwkv_generation_prompt"]) for query, kwargs in tokenizer.prompts} == {
        ("Math", "open_think"),
        ("Knowledge", "fake_think"),
    }


def test_full_pipeline_persists_before_generation_finishes_and_uses_all_engines(tmp_path, tokenizer):
    from aiohttp import web

    for index in range(4):
        dataset_row(tmp_path, "Knowledge", f"Knowledge_{index}", f"q{index}", "42", part=f"part{index // 2}.jsonl")
    output = tmp_path / "output"
    requests = Counter()

    async def run():
        persisted = asyncio.Event()

        async def observe():
            path = output / "correct_counts_by_question.jsonl"
            while not path.exists() or not path.stat().st_size:
                await asyncio.sleep(0.01)
            persisted.set()

        def complete(engine):
            async def handle(request):
                payload = await request.json()
                requests[engine] += 1
                if payload["prompt"] == list(b"q3"):
                    await asyncio.wait_for(persisted.wait(), timeout=10)
                await asyncio.sleep(0.01)
                return web.json_response(
                    {"choices": [{"token_ids": list(b"Final answer: 42") + [0], "finish_reason": "stop"}]}
                )

            return handle

        runners, urls = [], []
        for index in range(2):
            app = web.Application()
            app.router.add_post("/v1/completions", complete(index))
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            host, port = runner.addresses[0]
            urls.append(f"http://{host}:{port}")
            runners.append(runner)
        observer = asyncio.create_task(observe())
        try:
            args = rollout.parser.parse_args(
                [
                    "--model",
                    "test-rwkv",
                    "--data-root",
                    str(tmp_path),
                    "--output",
                    str(output),
                    "--base-url",
                    ",".join(urls),
                    "--max-num-seqs",
                    "1",
                    "--score-workers",
                    "1",
                    "--timeout",
                    "20",
                ]
            )
            await rollout.run(args)
            await observer
            dataset_row(tmp_path, "Knowledge", "Knowledge_4", "q4", "42", part="part2.jsonl")
            args.resume = True
            await rollout.run(args)
            args.max_tokens = 10
            with pytest.raises(ValueError, match="changed max_tokens"):
                await rollout.run(args)
        finally:
            observer.cancel()
            for runner in runners:
                await runner.cleanup()

    asyncio.run(run())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["limit"] == 0
    assert summary["config"]["concurrency"] == 4
    assert summary["config"]["status"] == "completed"
    assert summary["config"]["queues"]["inflight"] == [0, 0]
    assert summary["Knowledge"]["num_questions"] == 5
    assert summary["Knowledge"]["correct_rollouts"] == 320
    assert summary["Knowledge"]["correct_count_histogram"] == {"64": 5}
    assert summary["config"]["resumed_questions"] == 4
    assert summary["config"]["progress"]["scored_questions"] == 5
    assert sum(requests.values()) == 320 and all(requests[index] > 0 for index in range(2))
    assert len(tokenizer.prompts) == 5
    records = [json.loads(line) for line in (output / "correct_counts_by_question.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert {record["uuid"] for record in records} == {f"Knowledge_{index}" for index in range(5)}
    assert all(record["counts"] == {"correct": 64} for record in records)


def test_http_failure_stops_pipeline_and_keeps_external_server(tmp_path, tokenizer):
    from aiohttp import ClientSession, web

    dataset_row(tmp_path, "Knowledge", "Knowledge_1", "q", "42")

    async def run():
        app = web.Application()

        async def complete(request):
            return web.Response(status=503, text="engine unavailable")

        async def health(request):
            return web.Response(text="ok")

        app.router.add_post("/v1/completions", complete)
        app.router.add_get("/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0]
        url = f"http://{host}:{port}"
        try:
            with pytest.raises(ExceptionGroup) as error:
                await rollout.run(
                    rollout.parser.parse_args(
                        [
                            "--model",
                            "test-rwkv",
                            "--data-root",
                            str(tmp_path),
                            "--output",
                            str(tmp_path / "output"),
                            "--base-url",
                            url,
                            "--concurrency",
                            "2",
                            "--score-workers",
                            "1",
                        ]
                    )
                )
            assert any("HTTP 503" in str(exc) for exc in error.value.exceptions)
            async with ClientSession() as session, session.get(f"{url}/health") as response:
                assert await response.text() == "ok"
        finally:
            await runner.cleanup()

    asyncio.run(run())
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["config"]["status"] == "interrupted"
    assert summary["Knowledge"]["num_questions"] == 0


def test_partial_startup_failure_cleans_only_owned_process_groups(tmp_path, tokenizer, monkeypatch):
    processes, killed = [], []

    class Process:
        def __init__(self, command, *, env, stdout, stderr, start_new_session):
            self.pid = 90000 + len(processes)
            self.device = env["CUDA_VISIBLE_DEVICES"]
            if command[-1] == "--check":
                self.returncode = 0
            else:
                self.returncode = 1 if not processes else None
                processes.append(self)
            assert start_new_session

        async def wait(self):
            self.returncode = self.returncode or 0
            return self.returncode

    async def create_subprocess_exec(*command, **kwargs):
        return Process(command, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(rollout.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(RuntimeError, match="vLLM exited"):
        asyncio.run(rollout.run(rollout.parser.parse_args(["--output", str(tmp_path / "output")])))
    assert [process.device for process in processes] == ["2", "5"]
    assert killed == [(90001, signal.SIGTERM)]
    assert json.loads((tmp_path / "output/summary.json").read_text())["config"]["status"] == "interrupted"


@pytest.mark.parametrize("valid_eos", [True, False])
def test_owned_engine_acceptance_precedes_dataset_generation(tmp_path, tokenizer, monkeypatch, valid_eos):
    from aiohttp import web

    dataset_row(tmp_path, "Knowledge", "Knowledge_1", "q", "4")
    requests, killed = [], []
    process = SimpleNamespace(pid=91001, returncode=None)

    async def wait():
        process.returncode = 0
        return 0

    process.wait = wait
    monkeypatch.setattr(rollout.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    async def run():
        async def complete(request):
            payload = await request.json()
            requests.append(payload)
            text = r">reasoning </think> \boxed{4}"
            ids = list(text.encode()) + [0 if valid_eos else 1]
            return web.json_response({"choices": [{"token_ids": ids, "finish_reason": "stop"}]})

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0]

        async def start_servers(args, processes):
            processes.append(process)
            return [f"http://{host}:{port}"]

        monkeypatch.setattr(rollout, "start_servers", start_servers)
        args = rollout.parser.parse_args(
            [
                "--model",
                "test-rwkv",
                "--data-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
                "--max-num-seqs",
                "1",
                "--score-workers",
                "1",
            ]
        )
        try:
            if valid_eos:
                await rollout.run(args)
            else:
                with pytest.raises(RuntimeError, match="Engine acceptance failed"):
                    await rollout.run(args)
        finally:
            await runner.cleanup()

    asyncio.run(run())
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert killed == [(91001, signal.SIGTERM)]
    assert process.returncode == 0
    assert len(requests) == (65 if valid_eos else 1)
    assert summary["Knowledge"]["num_questions"] == int(valid_eos)
    if valid_eos:
        assert summary["config"]["engine_validation"] == "passed"
        assert summary["Knowledge"]["correct_rollouts"] == 64
    else:
        assert not (tmp_path / "output/correct_counts_by_question.jsonl").exists()
