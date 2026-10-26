import asyncio
import json
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))
from examples.train.rwkv.train.common import rollout
from examples.train.rwkv.train.sao import build_value_data


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


@pytest.mark.parametrize(
    "domain,answer,gold,finish",
    [
        ("Math", r">reasoning </think> \boxed{42}", "42", "stop"),
        ("Math", r">reasoning </think> \boxed{43}", "42", "stop"),
        ("Math", r">reasoning </think> \boxed{42}", "42", "length"),
        ("Knowledge", "Final answer: b", "second option", "stop"),
        ("Long_Context", "Final answer: 5", "5", "stop"),
        ("Long_Context", "Final answer: 5", "5", "length"),
        ("Code", "print(3)", {"inputs": [""], "outputs": ["3\n"]}, "stop"),
    ],
)
@pytest.mark.parametrize("threaded", [False, True])
def test_ultradata_native_rewards_match_rollout(
    tmp_path, tokenizer, monkeypatch, domain, answer, gold, finish, threaded
):
    from concurrent.futures import ThreadPoolExecutor

    import skyrl_gym
    from skyrl_gym.envs.ultradata import env as ultradata

    query = "Question?\na: first option\nb: second option"
    dataset_row(tmp_path, domain, f"{domain}_1", query, gold)
    row = next(rollout.rows(tmp_path, 0, tokenizer, 1000, 8))
    if domain == "Code":

        def scorer(raw, tests):
            return raw == "print(3)" and tests == gold, "ok"

        monkeypatch.setattr(rollout, "run_code", scorer)
        monkeypatch.setattr(ultradata, "run_code", scorer)
    counts, _, _ = rollout.score_question(row, [{"text": answer, "token_ids": [7, 0], "finish_reason": finish}], 0)
    path, offset = row["_source"]
    env = skyrl_gym.make(
        "ultradata",
        extras={"data_source": domain, "reward_spec": {"source_path": path, "source_offset": offset}},
    )
    _, metadata = env.init([{"role": "user", "content": query}])
    assert metadata["chat_template_kwargs"]["rwkv_generation_prompt"] == (
        "open_think" if domain == "Math" else "fake_think"
    )
    env.set_generation_metadata(
        action=answer, ended_eod=finish == "stop", truncated=finish == "length", stop_reason=finish
    )
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(env.step, answer).result()
    else:
        result = env.step(answer)
    assert result["reward"] == float(counts["correct"])


def test_prepare_ultradata_rl_keeps_lazy_rewards_and_question_split(tmp_path):
    import pyarrow.parquet as pq

    from examples.train.rwkv.train.opd.prepare import prepare

    source = tmp_path / "source"
    source.mkdir()
    for domain in rollout.DOMAINS:
        for index in range(8):
            gold = {"inputs": [""], "outputs": ["3\n"]} if domain == "Code" else "3"
            dataset_row(source, domain, f"{domain}_{index}", f"query {index}", gold)
    output = tmp_path / "rl" / "train.parquet"
    prepare(source, output, task_rewards=True, validation_fraction=0.5, seed=42)
    train = pq.read_table(output).to_pylist()
    validation = pq.read_table(output.with_name("validation.parquet")).to_pylist()
    assert len(train) + len(validation) == 32
    assert not {row["uuid"] for row in train} & {row["uuid"] for row in validation}
    for row in train + validation:
        assert row["env_class"] == "ultradata"
        spec = row["reward_spec"]
        with open(spec["source_path"], "rb") as stream:
            stream.seek(spec["source_offset"])
            assert json.loads(stream.readline())["uuid"] == row["uuid"]
    opd_output = tmp_path / "opd" / "train.parquet"
    prepare(source, opd_output)
    assert {row["env_class"] for row in pq.read_table(opd_output).to_pylist()} == {"aime"}


def test_filter_ultradata_removes_only_rwkv_pass_rate_one(tmp_path):
    from examples.train.rwkv.get_ultradata_rwkv import filter_dataset

    source = tmp_path / "source"
    counts_by_domain = {
        "Math": {"correct": 4},
        "Code": {"correct": 3, "wrong": 1},
        "Long_Context": {"wrong": 4},
        "Knowledge": {"correct": 2, "unanswered": 2},
    }
    source.mkdir()
    for domain in counts_by_domain:
        dataset_row(source, domain, f"{domain}_1", "question", "answer")

    rollout_output = tmp_path / "rollout"
    rollout_output.mkdir()
    with (rollout_output / "correct_counts_by_question.jsonl").open("w") as handle:
        for domain, counts in counts_by_domain.items():
            handle.write(json.dumps({"uuid": f"{domain}_1", "domain": domain, "counts": counts}) + "\n")

    output = tmp_path / "filtered"
    manifest = filter_dataset.filter_dataset(source, rollout_output, output, expected_samples=4)

    assert manifest["total_removed_pass_rate_1"] == 1
    assert manifest["total_kept_rows"] == 3
    assert not (output / "Math" / "part.jsonl").read_text()
    assert [
        json.loads((output / domain / "part.jsonl").read_text())["uuid"]
        for domain in ("Code", "Long_Context", "Knowledge")
    ] == ["Code_1", "Long_Context_1", "Knowledge_1"]


def test_prepare_source_normalizes_public_upstream_records(tmp_path, monkeypatch):
    from examples.train.rwkv.get_ultradata_rwkv import prepare_source

    fixtures = {
        "dapo": {"prompt": [{"role": "user", "content": "math"}], "reward_model": {"ground_truth": "4"}},
        "deepscaler": {"problem": "scale", "answer": "5"},
        "deepmath": {"question": "deep", "final_answer": "6"},
        "science": {"input": "science", "expected_answer": "A"},
        "hotpot": {
            "question": "who?",
            "answer": "Ada",
            "context": {"title": ["Paper"], "sentences": [["Ada worked here."]]},
        },
        "qasper": {
            "abstract": "Abstract",
            "full_text": [{"section_name": "Body", "paragraphs": ["Evidence"]}],
            "qas": [
                {
                    "question": "What?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "free_form_answer": "Yes",
                                "extractive_spans": [],
                                "yes_no": None,
                            }
                        }
                    ],
                }
            ],
        },
        "hardtests": {
            "question_content": "print",
            "public_test_cases": [{"input": "1\n", "output": "1\n"}],
        },
    }

    monkeypatch.setattr(prepare_source, "_iter_source_rows", lambda spec: [fixtures[spec.adapter]])
    output = tmp_path / "source"
    manifest = prepare_source.prepare(output, limit_per_source=1)

    assert manifest["stage"] == "pre_difficulty_calibration"
    assert manifest["total_rows"] == len(fixtures)
    assert {path.parent.name for path in output.glob("*/*.jsonl")} == set(prepare_source.DOMAINS)
    assert json.loads((output / "Code" / "hardtests.jsonl").read_text())["ground_truth"] == {
        "inputs": ["1\n"],
        "outputs": ["1\n"],
    }


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
                        "--save-rollouts",
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
    records = [
        json.loads(line)
        for path in (tmp_path / "output/rollouts").glob("*/*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 128
    math_records = [record for record in records if record["domain"] == "Math"]
    assert sum(record["reward"] for record in math_records) == 16
    assert sum(record["truncated"] for record in math_records) == 8
    assert any(len(record["response_ids"]) > 1 for record in math_records)
    assert all(record["prompt_ids"] == list(record["domain"].encode()) for record in records)
    assert {(query, kwargs["rwkv_generation_prompt"]) for query, kwargs in tokenizer.prompts} == {
        ("Math", "open_think"),
        ("Knowledge", "fake_think"),
    }


def test_math_rollout_preserves_coordinate_commas(tmp_path, tokenizer):
    dataset_row(tmp_path, "Math", "Math_00399", "Find the coordinates.", "(3,4)")
    row = next(rollout.rows(tmp_path, 0, tokenizer, 100, 5))
    choices = [
        {"text": f">reasoning </think> \\boxed{{{answer}}}", "token_ids": [0], "finish_reason": "stop"}
        for answer in ("(3, 4)", "34", "(4, 3)")
    ]
    counts, reasons, samples = rollout.score_question(row, choices, 0)
    assert counts == {"correct": 1, "wrong": 2}
    assert reasons == {"math_verify": 3}
    assert samples["correct"]["sample_index"] == 0
    assert samples["wrong"]["sample_index"] == 1


@pytest.mark.parametrize("samples_per_prompt", [1, 64])
def test_full_pipeline_persists_before_generation_finishes_and_uses_all_engines(
    tmp_path, tokenizer, samples_per_prompt
):
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
                assert all(payload[key] == value for key, value in rollout.ROLLOUT_SAMPLING_PARAMS.items())
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
                    "--save-rollouts",
                    "--samples-per-prompt",
                    str(samples_per_prompt),
                    "--sampling",
                    "rollout",
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
            args.samples_per_prompt = samples_per_prompt + 1
            with pytest.raises(ValueError, match="changed rollouts_per_question"):
                await rollout.run(args)
            args.samples_per_prompt = samples_per_prompt
            args.max_tokens = 10
            with pytest.raises(ValueError, match="changed max_tokens"):
                await rollout.run(args)
            args.max_tokens = 8192
            args.save_rollouts = False
            with pytest.raises(ValueError, match="changed save_rollouts"):
                await rollout.run(args)
            args.save_rollouts = True
            saved = json.loads((output / "summary.json").read_text())
            before = (output / "correct_counts_by_question.jsonl").read_bytes()
            for version in (None, 1):
                legacy = json.loads(json.dumps(saved))
                if version is None:
                    del legacy["config"]["scoring_version"]
                else:
                    legacy["config"]["scoring_version"] = version
                (output / "summary.json").write_text(json.dumps(legacy))
                with pytest.raises(ValueError, match="changed scoring_version"):
                    await rollout.run(args)
                assert (output / "correct_counts_by_question.jsonl").read_bytes() == before
            (output / "summary.json").write_text(json.dumps(saved))
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
    assert summary["Knowledge"]["correct_rollouts"] == 5 * samples_per_prompt
    assert summary["Knowledge"]["correct_count_histogram"] == {str(samples_per_prompt): 5}
    assert summary["config"]["resumed_questions"] == 4
    assert summary["config"]["progress"]["scored_questions"] == 5
    assert sum(requests.values()) == 5 * samples_per_prompt and all(requests[index] > 0 for index in range(2))
    assert len(tokenizer.prompts) == 5
    records = [json.loads(line) for line in (output / "correct_counts_by_question.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert {record["uuid"] for record in records} == {f"Knowledge_{index}" for index in range(5)}
    assert all(record["counts"] == {"correct": samples_per_prompt} for record in records)
    trajectories = [
        json.loads(line) for path in (output / "rollouts").glob("*/*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(trajectories) == 5 * samples_per_prompt
    assert len({(record["uuid"], record["sample_index"]) for record in trajectories}) == 5 * samples_per_prompt
    assert all(record["response_ids"] == list(b"Final answer: 42") + [0] for record in trajectories)
    assert all(record["reward"] == 1.0 and not record["truncated"] for record in trajectories)
    assert not list((output / "rollouts").glob("*/*.pending"))
    manifest = build_value_data.build_data(output, tmp_path / "value", validation_fraction=0.5, seed=0, max_length=64)
    assert sum(manifest["counts"][f"{split}_rollouts"] for split in ("train", "validation")) == 5 * samples_per_prompt


@pytest.fixture
def token_rollout_run(tmp_path):
    output = tmp_path / "token-run"
    shards = output / "rollouts/Knowledge"
    shards.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({"config": {"save_rollouts": True, "model": "test-rwkv"}}))
    for index in range(20):
        rows = [
            {"response_ids": [13, 0], "reward": 1.0, "truncated": False},
            {"response_ids": [14, 0], "response_mask": [1, 0], "reward": 0.0, "truncated": False},
            {"response_ids": [15, 16], "reward": 0.0, "truncated": True},
            {"response_ids": [], "reward": 0.0, "truncated": False},
            {"response_ids": [17] * 20, "reward": 0.0, "truncated": False},
            {"response_ids": [18, 0], "reward": 0.0, "truncated": True, "returns": [0, 0, 0.7, 0.8]},
        ]
        with (shards / f"question_{index}.jsonl").open("w") as handle:
            for sample_index, row in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "uuid": f"Knowledge_{index}",
                            "domain": "Knowledge",
                            "sample_index": sample_index,
                            "prompt_ids": [11, 12],
                            **row,
                        }
                    )
                    + "\n"
                )
    (shards / "unfinished.pending").write_text("not a completed trajectory")
    return output


def test_build_balanced_value_data_keeps_disjoint_questions_and_both_sources(tmp_path, token_rollout_run):
    other = tmp_path / "benchmark-run"
    shards = other / "rollouts/gsm8k"
    shards.mkdir(parents=True)
    (other / "summary.json").write_bytes((token_rollout_run / "summary.json").read_bytes())
    for index in range(20):
        with (shards / f"{index}.jsonl").open("w") as handle:
            for sample in range(4):
                handle.write(
                    json.dumps(
                        {
                            "uuid": f"problem_{index}",
                            "domain": "gsm8k",
                            "sample_index": sample,
                            "prompt_ids": [11, 12],
                            "response_ids": [13 + sample, 0],
                            "reward": float(sample == 0),
                            "truncated": False,
                        }
                    )
                    + "\n"
                )
    outputs = [tmp_path / name for name in ("balanced", "balanced-repeat")]
    for output in outputs:
        manifest = build_value_data.build_data(
            token_rollout_run,
            output,
            validation_fraction=0.5,
            max_length=8,
            additional_inputs=[other],
            balance_rewards=True,
        )
        assert len(manifest["additional_sources"]) == 1
        questions = {}
        for split in ("train", "validation"):
            rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
            assert {row["domain"] for row in rows} == {"Knowledge", "gsm8k"}
            identities = [(row["uuid"], row["sample_index"]) for row in rows]
            assert len(set(identities)) == len(identities)
            questions[split] = {row["uuid"] for row in rows}
            for domain in ("Knowledge", "gsm8k"):
                outcomes = Counter(row["reward"] for row in rows if row["domain"] == domain)
                assert outcomes[0] == outcomes[1] > 0
            assert manifest["counts"][f"{split}_rollouts"] == len(rows)
            assert manifest["counts"][f"{split}_action_tokens"] == sum(sum(row["loss_mask"]) for row in rows)
        assert questions["train"].isdisjoint(questions["validation"])
    for split in ("train", "validation"):
        assert (outputs[0] / f"{split}.jsonl").read_bytes() == (outputs[1] / f"{split}.jsonl").read_bytes()
    (other / "summary.json").write_text(json.dumps({"config": {"save_rollouts": True, "model": "other-checkpoint"}}))
    with pytest.raises(ValueError, match="same checkpoint"):
        build_value_data.build_data(token_rollout_run, tmp_path / "wrong-model", additional_inputs=[other])


def test_balance_drops_domains_without_both_outcomes_and_rejects_nonbinary(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"uuid": "one", "domain": "Math", "reward": 0, "loss_mask": [0, 1]}
    path.write_text(json.dumps(row) + "\n")
    counts, _ = build_value_data.balance_data(path, 42)
    assert counts["rollouts"] == counts["positive_rewards"] == 0
    assert path.read_bytes() == b""
    path.write_text(json.dumps({**row, "reward": 0.25}) + "\n")
    with pytest.raises(ValueError, match="binary"):
        build_value_data.balance_data(path, 42)


def test_native_benchmark_value_scoring_preserves_tokens_and_checks_provenance(tmp_path, monkeypatch):
    import hashlib

    from skyrl.utils import tok

    model = tmp_path / "model"
    model.mkdir()
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)
    monkeypatch.setenv("PYTHONPATH", "original-path")
    calls = []

    def extract(text, extractor, *, rendered_prompt):
        calls.append((text, extractor, rendered_prompt))
        return text

    monkeypatch.setitem(
        sys.modules,
        "src.rwkv_eval.answer_extract.free_response",
        SimpleNamespace(
            answer_extractor_for_benchmark=lambda name: "math",
            extract_answer=extract,
            verify_math_answer=lambda gold, answer: gold == answer,
        ),
    )
    monkeypatch.setattr(tok, "get_tokenizer", lambda model: SimpleNamespace(decode=lambda ids, **kwargs: "<think"))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"gold_answers": ["42"], "source": {"benchmark": "gsm8k", "problem_key": "family"}}) + "\n"
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    config = {
        "model": str(model),
        "input_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "samples_per_prompt": 2,
    }
    (raw / "generation.json").write_text(json.dumps(config))
    rows = [
        {
            "model": str(alias),
            "prompt_index": 0,
            "sample_index": sample,
            "prompt_ids": [7, 8],
            "response_ids": [9 + sample, 0],
            "text": text,
            "finish_reason": finish,
        }
        for sample, text, finish in [(0, "42", "stop"), (1, "41", "length")]
    ]
    (raw / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "scored"
    result = build_value_data.score_benchmarks(raw, prompts, output, tmp_path)
    assert result["counts"] == {"rollouts": 2, "positive_rewards": 1, "truncated": 1}
    scored = [json.loads(line) for line in (output / "rollouts/gsm8k/0.jsonl").read_text().splitlines()]
    assert [row["reward"] for row in scored] == [1.0, 0.0]
    assert [row["response_ids"] for row in scored] == [[9, 0], [10, 0]]
    assert all(row["uuid"] == "family" for row in scored)
    assert calls == [("42", "math", "<think"), ("41", "math", "<think")]
    prompts.write_text("changed\n")
    with pytest.raises(ValueError, match="match the input"):
        build_value_data.score_benchmarks(raw, prompts, tmp_path / "mismatched", tmp_path)


def test_build_value_data_can_keep_budget_truncations_as_terminal_failures(tmp_path, token_rollout_run):
    output = tmp_path / "bounded-value-data"
    manifest = build_value_data.build_data(
        token_rollout_run, output, validation_fraction=0.5, max_length=8, truncation_as_terminal=True
    )
    assert manifest["truncation_as_terminal"]
    assert manifest["counts"]["truncated_terminal_failures"] == 40
    rows = [
        json.loads(line)
        for split in ("train", "validation")
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 80
    assert sum(row["reward"] for row in rows) == 20
    for row in rows:
        if row["truncated"]:
            assert row["reward"] == 0
            assert row["returns"] == [0.0] * len(row["input_ids"])
            assert row["input_ids"][-2:] in ([15, 16], [18, 0])
            assert row["loss_mask"] == [0, 0, 1, 1]


def test_build_value_data_preserves_tokens_masks_outcomes_and_uuid_split(tmp_path, token_rollout_run):
    import torch

    from skyrl.train.dataset.value_dataset import (
        RewardDataCollator,
        ValueDataset,
        value_loss,
    )

    output = tmp_path / "value-data"
    manifest = build_value_data.build_data(token_rollout_run, output, validation_fraction=0.5, max_length=8)
    assert manifest["counts"]["read_rollouts"] == 120
    for reason in ("truncated", "empty", "overlength"):
        assert manifest["counts"][f"skipped_{reason}"] == 20
    splits = {
        split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation")
    }
    assert sum(len(rows) for rows in splits.values()) == 60
    assert {row["uuid"] for row in splits["train"]}.isdisjoint(row["uuid"] for row in splits["validation"])
    for split, rows in splits.items():
        assert manifest["counts"][f"{split}_questions"] == len({row["uuid"] for row in rows})
        for row in rows:
            assert row["input_ids"][:2] == [11, 12]
            assert row["loss_mask"][:2] == [0, 0]
            if row["sample_index"] == 0:
                assert row["input_ids"] == [11, 12, 13, 0] and row["reward"] == 1.0
            elif row["sample_index"] == 1:
                assert row["input_ids"] == [11, 12, 14, 0] and row["reward"] == 0.0
                assert row["loss_mask"] == [0, 0, 1, 0]
            else:
                assert row["returns"] == [0, 0, 0.7, 0.8]
        dataset = ValueDataset(str(output / f"{split}.jsonl"), max_length=8)
        batch = RewardDataCollator(0)([dataset[index] for index in range(len(dataset))])
        predictions = torch.zeros_like(batch["labels"], requires_grad=True)
        error, count = value_loss(predictions, batch["labels"], batch["response_mask"])
        (error / count).backward()
        assert torch.isfinite(predictions.grad).all()
    other = tmp_path / "value-data-repeat"
    build_value_data.build_data(token_rollout_run, other, validation_fraction=0.5, max_length=8)
    for name in ("train.jsonl", "validation.jsonl"):
        assert (output / name).read_bytes() == (other / name).read_bytes()
    cli_output = tmp_path / "value-data-cli"
    root = Path(__file__).parents[4]
    with (tmp_path / "build_cli.log").open("wb") as log:
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                sys.executable,
                str(root / "examples/train/rwkv/train/sao/build_value_data.py"),
                "--input",
                str(token_rollout_run),
                "--output",
                str(cli_output),
                "--validation-fraction",
                "0.5",
                "--max-length",
                "8",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=45,
        )
    for name in ("train.jsonl", "validation.jsonl"):
        assert (output / name).read_bytes() == (cli_output / name).read_bytes()
    with pytest.raises(FileExistsError):
        build_value_data.build_data(token_rollout_run, output, validation_fraction=0.5)


def test_build_value_data_rejects_empty_split_without_publishing(tmp_path, token_rollout_run):
    output = tmp_path / "empty-split"
    with pytest.raises(ValueError, match="empty train/validation"):
        build_value_data.build_data(token_rollout_run, output, validation_fraction=1e-20)
    assert not output.exists()
    assert not list(tmp_path.glob(".value-data-*"))


def test_trajectory_publication_is_atomic_and_can_replace_unmarked_question(tmp_path):
    row = {"uuid": "Knowledge_1", "domain": "Knowledge"}
    pending = rollout.start_trajectory(tmp_path, row, [11, 12])
    rollout.append_trajectory(pending, 1, [14, 0], "length")
    rollout.append_trajectory(pending, 0, [13, 0], "stop")
    path = rollout.trajectory_path(tmp_path, row)
    assert not path.exists()
    rollout.publish_trajectories(
        tmp_path, row, [{"status": "correct", "reason": "ok"}, {"status": "unanswered", "reason": "length"}]
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["reward"] for record in records] == [0.0, 1.0]
    assert [record["truncated"] for record in records] == [True, False]
    assert [record["sample_index"] for record in records] == [1, 0]
    pending = rollout.start_trajectory(tmp_path, row, [11, 12])
    rollout.append_trajectory(pending, 0, [15, 0], "stop")
    rollout.publish_trajectories(tmp_path, row, [{"status": "wrong", "reason": "answer_match"}])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["response_ids"] == [15, 0]
    assert not list(path.parent.glob("*.pending"))
    assert not list(path.parent.glob("*.tmp"))


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
@pytest.mark.parametrize("probe_text", [r">reasoning </think> \boxed{4}", r"\boxed{5}"])
def test_owned_engine_acceptance_precedes_dataset_generation(tmp_path, tokenizer, monkeypatch, valid_eos, probe_text):
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
            text = probe_text if len(requests) == 1 else r">reasoning </think> \boxed{4}"
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


@pytest.mark.parametrize("entrypoint", ["prepare", "check"])
@pytest.mark.parametrize("installed_commit", ["old-install", None])
def test_rwkv_native_artifact_preflight_rejects_unmatched_install_before_loading(
    tmp_path, monkeypatch, entrypoint, installed_commit
):
    from unittest.mock import MagicMock

    from examples.train.rwkv.train.common import prepare_flashrwkv2 as preparation

    (tmp_path / "pyproject.toml").write_text('[tool.uv.sources.vllm]\nrev = "project-pin"\n')
    monkeypatch.chdir(tmp_path)
    read_text = MagicMock(
        return_value=json.dumps({"vcs_info": {"commit_id": installed_commit}}) if installed_commit else None
    )
    distribution = MagicMock(return_value=SimpleNamespace(read_text=read_text))
    monkeypatch.setattr(preparation.importlib.metadata, "distribution", distribution)
    load = MagicMock()
    monkeypatch.setattr(preparation, "_load_flashrwkv2", load)

    with pytest.raises(SystemExit, match="vLLM RWKV revision mismatch") as error:
        getattr(preparation, entrypoint)()

    assert repr(installed_commit) in str(error.value)
    assert "project-pin" in str(error.value)
    distribution.assert_called_once_with("vllm")
    read_text.assert_called_once_with("direct_url.json")
    load.assert_not_called()


def test_rwkv_native_artifact_environment_records_verified_installed_revision(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from examples.train.rwkv.train.common import prepare_flashrwkv2 as preparation

    (tmp_path / "pyproject.toml").write_text('[tool.uv.sources.vllm]\nrev = "matched-install"\n')
    monkeypatch.chdir(tmp_path)
    installed_metadata = json.dumps({"vcs_info": {"commit_id": "matched-install"}})
    monkeypatch.setattr(
        preparation.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: installed_metadata),
    )
    monkeypatch.setattr(preparation.importlib.metadata, "version", lambda name: "test-version")
    library = tmp_path / "extension.so"
    library.write_bytes(b"mock native artifact")
    monkeypatch.setattr(preparation, "_load_flashrwkv2", MagicMock())
    monkeypatch.setattr(
        preparation,
        "load_extension",
        lambda: SimpleNamespace(library=str(library), target="test-target", status="cached"),
    )
    monkeypatch.setattr(preparation.subprocess, "check_output", lambda *args, **kwargs: "test-driver\n")

    preparation.prepare()

    environment = json.loads((tmp_path / "environment.json").read_text())
    assert environment["vllm_rwkv_commit"] == "matched-install"
    assert environment["build"] == "cached"
