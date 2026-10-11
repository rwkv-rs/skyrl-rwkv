"""Hermetic RWKV sampling-SFT data/API/sampler contract tests (no Ray or GPU)."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))
from examples.train.sampling_sft import boost, data
from examples.train.sampling_sft import generate_expert_traces as expert
from skyrl.train.utils import projection_sampling as sampling


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def question(uuid, domain="Math"):
    return {
        "uuid": uuid,
        "query": f"Question {uuid}",
        "ground_truth": "4",
        "domain": domain,
    }


def response(content="\\boxed{4}", reasoning="Compute two plus two.", finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 20}),
    )


def test_expert_trace_keeps_reasoning_final_answer_and_gold():
    row = {"uuid": "a", "prompt": "2+2?", "ground_truth": "4"}
    record = expert.expert_trace(row, response(), "teacher")
    assert record["solution"] == "Compute two plus two.\n\\boxed{4}"
    assert record["reasoning_content"] == "Compute two plus two."
    assert record["content"] == "\\boxed{4}"
    assert record["ground_truth"] == "4"
    assert record["usage"]["completion_tokens"] == 20
    assert expert.expert_trace(row, response(reasoning=None), "teacher")["solution"] == "\\boxed{4}"


@pytest.mark.parametrize(
    "result",
    [response(finish_reason="length"), response(content=""), response(content=None)],
)
def test_expert_rejects_truncated_or_empty_responses(result):
    with pytest.raises(ValueError, match="Incomplete expert"):
        expert.expert_trace({"uuid": "a"}, result, "teacher")


def test_ultradata_all_shards_domain_limit_and_disjoint_sharding(tmp_path):
    write_rows(tmp_path / "Math" / "a.jsonl", [question("a"), question("b")])
    write_rows(tmp_path / "Math" / "b.jsonl", [question("c")])
    write_rows(tmp_path / "Code" / "a.jsonl", [question("code", "Code")])
    argv = [
        "--dataset",
        str(tmp_path),
        "--output",
        "unused",
        "--domain",
        "Math",
        "--limit",
        "3",
    ]
    full = list(data.questions(expert.parse_args(argv)))
    assert [row["uuid"] for row in full] == ["a", "b", "c"]
    left = list(data.questions(expert.parse_args(argv + ["--num-shards", "2", "--shard-index", "0"])))
    right = list(data.questions(expert.parse_args(argv + ["--num-shards", "2", "--shard-index", "1"])))
    assert {r["uuid"] for r in left}.isdisjoint(r["uuid"] for r in right)
    assert {r["uuid"] for r in left + right} == {"a", "b", "c"}
    assert full[0]["prompt"] == "Question a"


def test_custom_question_key_stable_ids_and_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist([{"problem": "二加二?", "answer": "4"}]), path)
    args = expert.parse_args(
        [
            "--dataset",
            str(path),
            "--output",
            "unused",
            "--question-key",
            "problem",
            "--answer-key",
            "answer",
        ]
    )
    first = next(data.questions(args))
    assert first["prompt"] == "二加二?" and first["ground_truth"] == "4"
    assert first == next(data.questions(args))


def test_nemotron_expert_preserves_complete_reasoning_and_question_partition(tmp_path):
    rows = [
        {
            "uuid": uuid,
            "subset": "cot",
            "problem": "2+2?",
            "expected_answer": "4",
            "messages": [
                {"role": "user", "content": "2+2?\nUse boxed answer."},
                {
                    "role": "assistant",
                    "reasoning_content": "Full reasoning\n" * 10000,
                    "content": "\\boxed{4}",
                },
            ],
        }
        for uuid in ("trace-a", "trace-b")
    ]
    path = tmp_path / "nemotron.jsonl"
    write_rows(path, rows + [{"uuid": "tool-trace", "subset": "tir"}])
    traces = list(data.expert_rows(str(path), subset="cot"))
    assert len(traces) == 2
    assert traces[0]["solution"] == rows[0]["messages"][1]["reasoning_content"] + "\n\\boxed{4}"
    assert traces[0]["prompt"] == rows[0]["messages"][0]["content"]
    assert traces[0]["ground_truth"] == "4"
    assert traces[0]["question_id"] == traces[1]["question_id"]


def test_hf_input_streaming_configuration(monkeypatch):
    import datasets

    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([question("hf")])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    args = expert.parse_args(
        [
            "--dataset",
            "org/questions",
            "--dataset-config",
            "math",
            "--revision",
            "pinned",
            "--output",
            "unused",
        ]
    )
    assert next(data.questions(args))["uuid"] == "hf"
    assert calls == [
        (
            ("org/questions",),
            {"name": "math", "split": "train", "revision": "pinned", "streaming": True},
        )
    ]


def test_resume_repairs_only_partial_tail_and_refuses_setting_changes(tmp_path):
    path = tmp_path / "traces.jsonl"
    config = {"model": "teacher"}
    with data.open_output(path, config, False) as (handle, completed):
        assert not completed
        data.append_jsonl(handle, {"uuid": "a"})
    with path.open("ab") as handle:
        handle.write(b'{"uuid":"unfinished')
    with data.open_output(path, config, True) as (handle, completed):
        assert completed == {"a"}
        data.append_jsonl(handle, {"uuid": "b"})
    assert [row["uuid"] for row in data.load_rows(str(path))] == ["a", "b"]
    with pytest.raises(FileExistsError):
        with data.open_output(path, config, False):
            pass
    with pytest.raises(ValueError, match="configuration differs"):
        with data.open_output(path, {"model": "changed"}, True):
            pass


def test_api_concurrency_resume_and_secrets_not_persisted(tmp_path, monkeypatch):
    import openai

    source = tmp_path / "questions.jsonl"
    write_rows(source, [question(str(i)) for i in range(5)])
    output = tmp_path / "expert.jsonl"
    calls = []
    client_options = []
    active = peak = 0

    async def create(**kwargs):
        nonlocal active, peak
        calls.append(kwargs)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return response()

    class Client:
        def __init__(self, **kwargs):
            client_options.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setenv("DS_MODEL_NAME", "configured-teacher")
    monkeypatch.setenv("DS_MODEL_URL", "https://provider.invalid/v1/chat/completions")
    monkeypatch.setenv("DS_MODEL_API_KEY", "test-secret-key")
    argv = [
        "--dataset",
        str(source),
        "--output",
        str(output),
        "--concurrency",
        "2",
        "--limit",
        "3",
    ]
    asyncio.run(expert.generate(expert.parse_args(argv)))
    asyncio.run(expert.generate(expert.parse_args(argv + ["--limit", "5", "--resume"])))
    assert len(calls) == 5 and peak == 2
    assert len(list(data.load_rows(str(output)))) == 5
    assert all(call["model"] == "configured-teacher" for call in calls)
    assert all("max_tokens" not in call and "max_completion_tokens" not in call for call in calls)
    assert all("ground_truth" not in call["messages"][0]["content"] for call in calls)
    assert client_options[0]["base_url"] == "https://provider.invalid/v1"
    assert client_options[0]["api_key"] == "test-secret-key"
    assert "test-secret-key" not in output.read_text() + output.with_suffix(".jsonl.meta.json").read_text()


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        if text == "✿":
            return [10060]
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": [7, 8]}

    def decode(self, ids, *, skip_special_tokens):
        return "decoded"


def test_vllm_adapter_rwkv_template_context_and_unprivileged_scoring(monkeypatch):
    calls = []
    tokenizer = Tokenizer()

    def generate(prompt, params, **kwargs):
        calls.append((prompt, params))
        return [
            SimpleNamespace(
                prompt_logprobs=[
                    None,
                    {},
                    {3: SimpleNamespace(logprob=-2)},
                    {4: SimpleNamespace(logprob=-4)},
                ],
                outputs=[SimpleNamespace(token_ids=[4], finish_reason="length")],
            )
        ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        generate=generate,
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=10000)),
    )
    p = sampling.vLLMAutoregressiveSampler(
        llm,
        chat_template_kwargs={
            "rwkv_prompt_template": "bot",
            "rwkv_generation_prompt": "open_think",
        },
        sampling_params={
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
        },
    )
    assert sampling.get_logprobs_vLLM(p, "question", [3, 4]) == -3
    assert tokenizer.messages == [{"role": "user", "content": "question"}]
    assert tokenizer.kwargs["rwkv_generation_prompt"] == "open_think"
    assert calls[-1][0] == {"prompt_token_ids": [7, 8, 3, 4]}
    assert calls[-1][1]["temperature"] == 1
    assert calls[-1][1]["top_p"] == 1 and calls[-1][1]["top_k"] == -1
    assert "presence_penalty" not in calls[-1][1] and "penalty_decay" not in calls[-1][1]
    import random

    assert sampling.proposal_vLLM(p, "question", "expert info", [3], 1, random.Random(0)) == ([3, 4], False)
    assert tokenizer.messages == [
        {
            "role": "user",
            "content": "Expert solution:\nexpert info\nQuestion:\nquestion",
        }
    ]
    assert calls[-1][0] == {"prompt_token_ids": [7, 8, 3]}
    assert {key: calls[-1][1][key] for key in p.sampling_params} == p.sampling_params
    assert calls[-1][1]["stop_token_ids"] == [0, 10060]
    p.max_model_len = 2
    with pytest.raises(ValueError, match="Context needs"):
        sampling.get_logprobs_vLLM(p, "question", [3])


@pytest.mark.parametrize("rwkv_bot", [True, False])
@pytest.mark.parametrize("end_token", [10060, 0, 4])
def test_proposal_turn_separator_preserved_and_eos_appended(monkeypatch, rwkv_bot, end_token):
    import random

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    tokenizer = Tokenizer()
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        generate=lambda *args, **kwargs: [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[end_token],
                        finish_reason="stop" if end_token == 0 else "length",
                    )
                ]
            )
        ],
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=10000)),
    )
    p = sampling.vLLMAutoregressiveSampler(
        llm, chat_template_kwargs={"rwkv_prompt_template": "bot"} if rwkv_bot else {}
    )
    ids, finished = sampling.proposal_vLLM(p, "question", "expert info", [3], 1, random.Random(0))
    assert ids == ([3, 10060, 0] if rwkv_bot and end_token == 10060 else [3, end_token])
    assert finished == (end_token == 0 or (rwkv_bot and end_token == 10060))
    assert p.stop_token_ids == ([0, 10060] if rwkv_bot else [0])
    if rwkv_bot:
        assert tokenizer.messages == [
            {
                "role": "user",
                "content": "Expert solution:\nexpert info\nQuestion:\nquestion",
            }
        ]
    else:
        assert "expert info" in tokenizer.messages[0]["content"]
        assert "Partial response:" in tokenizer.messages[0]["content"]


def test_sampler_greedy_acceptance_and_old_block_rewrites(monkeypatch):
    p = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
    proposals = iter(
        [
            ([1, 1], False),
            ([2, 2], False),
            ([3, 3], False),
            ([2, 2, 1, 1], False),
            ([4, 4, 4, 4], False),
            ([5, 5, 5, 5], True),
        ]
    )
    scores = iter([-3.0, -2.0, -2.5, -2.5, -1.0, -1.5])
    prefixes = []

    def propose(p, q, solution, current, length, rng):
        prefixes.append(list(current))
        return next(proposals)

    monkeypatch.setattr(sampling, "proposal_vLLM", propose)
    monkeypatch.setattr(sampling, "get_logprobs_vLLM", lambda *args: next(scores))
    result = sampling.proj_samp_vLLM(p, "q", "s", max_new_tokens=4, block_num=2, mcmc_steps=2, seed=2)
    assert result["response_ids"] == [4, 4, 4, 4]
    assert result["acceptances"] == 2 and result["attempts"] == 4
    assert result["target_log_prob_cur"] == -1.0
    assert result["finish_reason"] == "length"  # Rejected completion must not change termination state.
    assert prefixes[3] == [2, 2]
    assert len(prefixes[4]) < 2  # Can edit an earlier block, not just the newest one.


def test_sampler_eos_ends_blocks_and_zero_step_rewrite(monkeypatch):
    p = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(sampling, "proposal_vLLM", lambda *args: ([3, 0], True))
    monkeypatch.setattr(sampling, "get_logprobs_vLLM", lambda *args: -1.0)
    result = sampling.proj_samp_vLLM(p, "q", "s", max_new_tokens=4, block_num=2, mcmc_steps=0)
    assert result["response_ids"] == [3, 0] and result["attempts"] == 0
    assert result["finish_reason"] == "stop"


def test_repetition_requires_three_adjacent_token_spans():
    span = list(range(32))
    assert not sampling.has_repetition(span * 2)
    assert sampling.has_repetition([999] + span * 3 + [10060, 0])
    assert not sampling.has_repetition(span + list(range(32, 64)) + span)


def test_repetitive_trajectory_is_skipped_before_scoring(monkeypatch):
    p = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(sampling, "proposal_vLLM", lambda *args: (list(range(16)) * 3, False))
    monkeypatch.setattr(
        sampling,
        "get_logprobs_vLLM",
        lambda *args: pytest.fail("Skipped trajectory must not be scored"),
    )
    result = sampling.proj_samp_vLLM(p, "q", "s", max_new_tokens=64, block_num=1, skip_repetition=True)
    assert result["finish_reason"] == "repetition" and result["attempts"] == 0


def test_batch_sampling_preserves_independent_rng_and_greedy_state(monkeypatch):
    p = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))

    def propose(p, question, solution, current, length, rng):
        single = isinstance(question, str)
        rows = [(current, length, rng)] if single else zip(current, length, rng, strict=True)
        results = []
        for prefix, count, generator in rows:
            gen = prefix + [generator.randrange(1, 20) for _ in range(count)]
            results.append((gen, False))
        return results[0] if single else results

    def score(p, question, current):
        return -sum(current) if isinstance(question, str) else [-sum(ids) for ids in current]

    monkeypatch.setattr(sampling, "proposal_vLLM", propose)
    monkeypatch.setattr(sampling, "get_logprobs_vLLM", score)
    settings = dict(max_new_tokens=12, block_num=3, mcmc_steps=3)
    separate = [sampling.proj_samp_vLLM(p, q, "s", seed=seed, **settings) for q, seed in zip(["a", "b"], [7, 9])]
    together = sampling.proj_samp_vLLM(p, ["a", "b"], ["s", "s"], seed=[7, 9], **settings)
    assert together == separate


def test_batched_vllm_proposals_keep_prefixes_and_terminal_ids(monkeypatch):
    import random

    calls = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))

    def generate(prompts, params, **kwargs):
        calls.append((prompts, params))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10060], finish_reason="stop")]),
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5], finish_reason="length")]),
        ]

    llm = SimpleNamespace(
        get_tokenizer=lambda: Tokenizer(),
        generate=generate,
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=10000)),
    )
    p = sampling.vLLMAutoregressiveSampler(llm, chat_template_kwargs={"rwkv_prompt_template": "bot"})
    results = sampling.proposal_vLLM(
        p,
        ["a", "b"],
        ["s1", "s2"],
        [[3], [4, 4]],
        [2, 3],
        [random.Random(7), random.Random(9)],
    )
    assert results == [([3, 10060, 0], True), ([4, 4, 5], False)]
    assert calls[0][0] == [
        {"prompt_token_ids": [7, 8, 3]},
        {"prompt_token_ids": [7, 8, 4, 4]},
    ]
    assert [params["max_tokens"] for params in calls[0][1]] == [2, 3]


@pytest.mark.parametrize("complete", [True, False])
def test_boost_native_sft_replay_mask_and_truncation(tmp_path, monkeypatch, complete):
    from skyrl.train.dataset.pretokenized import load_from_pretokenized

    p = SimpleNamespace(format_prompt=lambda q: [7, 8], tokenizer=Tokenizer())
    result = {
        "response_ids": [3, 10060, 0],
        "finish_reason": "stop" if complete else "length",
        "target_log_prob_cur": -1.0,
        "attempts": 10,
        "acceptances": 3,
    }
    monkeypatch.setattr(boost, "proj_samp_vLLM", lambda *args, **kwargs: result)
    args = boost.parse_args(["--input", "unused", "--output", "unused", "--model", "rwkv7-test"])
    record = boost.boost_record(p, {"uuid": "q", "prompt": "question", "solution": "expert"}, args)
    assert record["input_ids"] == ([7, 8, 3, 10060, 0] if complete else [])
    assert record["loss_mask"] == ([0, 0, 1, 1, 1] if complete else [])
    path = tmp_path / "train.jsonl"
    # Include a completed row so the native loader has trainable data in both cases.
    write_rows(
        path,
        [
            record,
            {
                **record,
                "uuid": "other",
                "input_ids": [7, 8, 3, 10060, 0],
                "loss_mask": [0, 0, 1, 1, 1],
            },
        ],
    )
    loaded = load_from_pretokenized(str(path))
    assert len(loaded) == (2 if complete else 1)
    assert loaded[0]["input_ids"] == [7, 8, 3, 10060, 0]
    assert loaded[0]["num_actions"] == 3
    assert loaded[0]["loss_mask"] == [1, 1, 1]
    assert loaded[0]["attention_mask"] == [1, 1, 1, 1, 1]
    from skyrl.train.dataset.collators import DefaultCollator

    batch = DefaultCollator(SimpleNamespace(pad_token_id=0), 1)([loaded[0]], batch_size=1)
    assert batch["sequences"].tolist() == [[7, 8, 3, 10060, 0]]
    assert batch["loss_mask"].tolist()[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert batch.metadata["response_length"] == 3


def test_question_partitions_stable_and_disjoint():
    ids = [f"q{i}" for i in range(100)]
    train = {uuid for uuid in ids if not data.validation_question(uuid, 0, 0.2)}
    validation = {uuid for uuid in ids if data.validation_question(uuid, 0, 0.2)}
    assert train and validation and not train.intersection(validation)
    assert train | validation == set(ids)
    assert validation == {uuid for uuid in ids if data.validation_question(uuid, 0, 0.2)}


@pytest.mark.parametrize(
    "argv",
    [
        ["--block-num", "3"],
        ["--mcmc-steps", "-1"],
        ["--validation-fraction", "1"],
        ["--repetition-penalty", "0"],
    ],
)
def test_boost_cli_rejects_invalid_sampling_settings(argv):
    with pytest.raises(SystemExit):
        boost.parse_args(["--input", "unused", "--output", "unused", "--model", "rwkv7-test", *argv])


def test_rwkv_boost_entrypoint_backend_options_resume_and_manifest(tmp_path, monkeypatch):
    import transformers

    source = tmp_path / "expert.jsonl"
    write_rows(source, [{"uuid": "a", "prompt": "question", "solution": "expert"}])
    output = tmp_path / "boosted.jsonl"
    options = []
    preflights = []
    shutdowns = []
    proposals = []
    tokenizer = Tokenizer()
    generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=0.96,
        top_p=0.76,
        top_k=32,
        presence_penalty=1.0,
        frequency_penalty=0.1,
        penalty_decay=0.988,
    )

    def llm(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=32768),
                engine_core=SimpleNamespace(shutdown=lambda: shutdowns.append(True)),
            ),
        )

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="rwkv"),
    )
    monkeypatch.setattr(
        transformers.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: generation_config,
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm))
    monkeypatch.setitem(
        sys.modules,
        "examples.train.rwkv.prepare_flashrwkv2",
        SimpleNamespace(check=lambda: preflights.append(True)),
    )

    def sample(p, *args, **kwargs):
        proposals.append(p.sampling_params)
        return {
            "response_ids": [3, 0],
            "finish_reason": "stop",
            "target_log_prob_cur": -1.0,
            "attempts": 10,
            "acceptances": 3,
        }

    monkeypatch.setattr(boost, "proj_samp_vLLM", sample)
    argv = [
        "--input",
        str(source),
        "--output",
        str(output),
        "--model",
        "rwkv7-test",
        "--partition",
        "all",
    ]
    boost.run(boost.parse_args(argv))
    boost.run(boost.parse_args(argv + ["--resume"]))
    assert len(list(data.load_rows(str(output)))) == 1
    assert len(preflights) == 2
    assert len(shutdowns) == 2
    assert options[0]["hf_overrides"] == {"wkv_mode": "fp32io16"}
    assert options[0]["dtype"] == "float16"
    assert options[0]["mamba_ssm_cache_dtype"] == "float32"
    assert options[0]["logprobs_mode"] == "raw_logprobs"
    assert options[0]["generation_config"] == "vllm"
    metadata = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
    assert metadata["repetition_penalty"] == 1.0
    assert (
        metadata["sampling_params"]
        == proposals[0]
        == {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
        }
    )
    assert tokenizer.kwargs["rwkv_prompt_template"] == "bot"
    with pytest.raises(ValueError, match="configuration differs"):
        boost.run(boost.parse_args(argv + ["--resume", "--seed", "1"]))
    generation_config.top_p = 0.8
    with pytest.raises(ValueError, match="configuration differs"):
        boost.run(boost.parse_args(argv + ["--resume"]))
    assert len(options) == 2  # Metadata mismatch fails before loading another GPU model.


def test_rwkv_sft_configuration_uses_native_pretokenized_fsdp():
    from skyrl.train.config.sft_config import (
        SFTConfig,
        build_skyrl_config_for_sft,
        validate_sft_cfg,
    )

    cfg = SFTConfig.from_cli_overrides(
        {
            "strategy": "fsdp",
            "model.path": "rwkv7-g1k-1.5b-20260930-ctx25600",
            "pretokenized_dataset_paths": ["train.jsonl"],
            "eval_pretokenized_dataset_paths": ["validation.jsonl"],
            "remove_microbatch_padding": False,
            "sequence_parallel_size": 1,
            "model_config_kwargs": {"wkv_mode": "fp32io16"},
            "rwkv_chunk_size": 1024,
            "placement.num_gpus_per_node": 1,
            "optimizer_config.scheduler": "cosine",
            "ckpt_interval": 50,
            "eval_interval": 50,
        }
    )
    validate_sft_cfg(cfg)
    native = build_skyrl_config_for_sft(cfg)
    assert native.trainer.policy.model_config_kwargs == {"wkv_mode": "fp32io16"}
    assert native.trainer.remove_microbatch_padding is False
    assert native.trainer.rwkv_chunk_size == 1024
    assert cfg.ckpt_interval == cfg.eval_interval == 50
    assert cfg.pretokenized_dataset_paths == ["train.jsonl"]
