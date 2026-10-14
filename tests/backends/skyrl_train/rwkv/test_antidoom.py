import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parents[4]))
from examples.train.rwkv import antidoom


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.mark.parametrize("source", [None, {"benchmark": "ifbench_test", "max_generated_tokens": 8}])
def test_generate_requests_original_ids_and_rollout_sampling(tmp_path, monkeypatch, source):
    import aiohttp

    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(json.dumps({"prompt_ids": [7, 9], "source": source}) + "\n")
    requests = []

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, json):
            requests.append((url, json))
            return self

        def raise_for_status(self):
            pass

        async def json(self):
            return {"choices": [{"token_ids": [65, 66], "finish_reason": "length"}]}

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args: SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(antidoom, "get_tokenizer", lambda *args, **kwargs: Tokenizer())
    output = tmp_path / "generation"
    asyncio.run(
        antidoom.generate(
            SimpleNamespace(
                output=output,
                model="rwkv7-test",
                input=input_path,
                sampling="rollout",
                base_url="http://example.test",
                concurrency=1,
                timeout=10,
                seed=42,
                limit=1,
                samples_per_prompt=1,
                max_tokens=32,
                max_model_len=100,
                prompt_template="bot",
                generation_prompt="open_think",
                shard_rollouts=False,
            )
        )
    )
    url, payload = requests[0]
    assert url == "http://example.test/v1/completions"
    assert payload["prompt"] == [7, 9]
    assert payload["return_token_ids"] is True
    assert payload["stop_token_ids"] == [0]
    assert payload["max_tokens"] == (8 if source else 32)
    assert {
        key: payload[key] for key in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty")
    } == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    row = next(antidoom.read_jsonl(output / "rollouts.jsonl"))
    assert row["prompt_ids"] == [7, 9]
    assert row["response_ids"] == [65, 66]
    assert row["source"] == source
    assert json.loads((output / "summary.json").read_text())["truncated"] == 1


def test_selects_whole_benchmarks_by_truncation(tmp_path):
    def score(name, rate, size="1.5b", mode="CoT", version="G1k"):
        return {
            "model": {"data_version": version, "param_size": size},
            "score_id": name,
            "benchmark": name,
            "field": "maths",
            "framework": "lighteval",
            "num_samples": 30,
            "cot_mode": mode,
            "truncation_rate": rate,
            "sampling": {"max_generated_tokens": 24576},
        }

    records = tmp_path / "scores.jsonl"
    rows = [
        score("aime25", 0.9),
        score("aime24", 0.5),
        score("math_500", 0.426),
        score("other-size", 0.99, size="7.2b"),
        score("NoCoT", 0.49, mode="NoCoT"),
        score("other-version", 0.99, version="G1j"),
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "selection"
    antidoom.select_benchmarks(
        SimpleNamespace(scoreboard_records=records, output=output, param_size="1.5b", min_truncation_rate=0.5)
    )
    manifest = json.loads((output / "benchmarks.json").read_text())
    assert manifest["scope"] == "complete_benchmarks"
    assert [row["name"] for row in manifest["benchmarks"]] == ["aime25", "aime24"]
    assert manifest["scoreboard_num_samples"] == 60
    assert (output / "benchmarks.toml").read_text() == 'maths = ["aime25", "aime24"]\n'


def test_exports_every_native_doc_and_keeps_gpqa_problem_identity(tmp_path, monkeypatch):
    native_calls = []

    class Task:
        def __init__(self, name, queries, question):
            self.name = name
            self.full_name = name + "|0"
            self.dataset_config = SimpleNamespace(version=1, hf_repo="Idavidrein/gpqa")
            self.evaluation_split = ["train"]
            self.rows = [{"Question": question} for _ in queries]
            self.docs = [
                SimpleNamespace(query=query, id=str(index), choices=["cached-gold"])
                for index, query in enumerate(queries)
            ]

        def get_docs(self, max_samples):
            native_calls.append((self.name, max_samples))
            return self.docs

    tasks = {
        name: Task(name, ["Q\nA) one\nB) two", "Other question"], "same original question")
        for name in ("gpqa:main", "gpqa:diamond")
    }

    class Registry:
        def __init__(self, **kwargs):
            assert kwargs["tasks"] == "gpqa:main,gpqa:diamond"

        def load_tasks(self):
            return tasks

    def load_datasets(selected, workers):
        assert workers == 1
        for task in selected.values():
            task.dataset = {"train": task.rows}

    monkeypatch.setitem(sys.modules, "src.rwkv_eval.main", SimpleNamespace(RWKVRegistry=Registry))
    monkeypatch.setitem(
        sys.modules,
        "lighteval.tasks.lighteval_task",
        SimpleNamespace(LightevalTask=SimpleNamespace(load_datasets=load_datasets)),
    )
    path = tmp_path / "benchmarks.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": name,
                        "field": "knowledge",
                        "cot_mode": "CoT",
                        "evidence": [{"max_generated_tokens": 24576}],
                    }
                    for name in tasks
                ]
            }
        )
    )
    output = tmp_path / "prompts"
    antidoom.export_benchmarks(SimpleNamespace(benchmarks=path, lighteval_project=tmp_path, output=output, seed=42))
    rows = list(antidoom.read_jsonl(output / "prompts.jsonl"))
    assert len(rows) == 4
    assert native_calls == [("gpqa:main", None), ("gpqa:diamond", None)]
    assert {row["source"]["doc_id"] for row in rows} == {"0", "1"}
    assert len({row["source"]["problem_key"] for row in rows}) == 1
    assert all(row["source"]["prompt_template"] == "bot" for row in rows)
    assert "cached-gold" not in (output / "prompts.jsonl").read_text()
    assert set(rows[0]) == {"query", "source"}


@pytest.mark.parametrize("pubmed_evaluated", [False, True])
def test_export_mix_filters_source_families_and_preserves_benchmarks(tmp_path, monkeypatch, pubmed_evaluated):
    import huggingface_hub

    base = tmp_path / "benchmarks"
    base.mkdir()
    base_rows = [{"query": "benchmark question", "source": {"benchmark": "aime25"}}] * 2
    (base / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in base_rows))
    (base / "manifest.json").write_text(json.dumps({"total_questions": 2, "exported_questions": {"aime25": 2}}))
    tested = tmp_path / "tested.jsonl"
    names = ["mathqa", "commonsenseqa"] + (["pubmedqa:artificial"] if pubmed_evaluated else [])
    tested.write_text("".join(json.dumps({"benchmark": name}) + "\n" for name in names))

    def sample(source_id, query, index=0):
        return {
            "id": f"antidoom:{source_id}:{index}",
            "conversations": [{"from": "human", "value": query}],
            "source": source_id + "-upstream",
            "source_config": None,
            "source_split": "train",
            "source_sample_id": str(index),
            "source_license": "mit",
            "source_license_url": "https://example.test/license",
            "heldout_policy": "training prompts",
        }

    mix = tmp_path / "mix.jsonl"
    mix_rows = [sample(source_id, source_id + " question") for source_id in antidoom.MIX_SOURCES]
    mix_rows += [sample("apps_train", "benchmark question", 1), sample("apps_train", " apps_train question ", 2)]
    for row in mix_rows:
        row.pop("source_config")
        if row["id"].startswith("antidoom:ifstruct_train_generated:"):
            row.pop("source_license_url")
    mix.write_text("".join(json.dumps(row) + "\n" for row in mix_rows))

    def download(dataset, filename, **kwargs):
        assert dataset == "LiquidAI/antidoom-mix-v1.0"
        assert filename == "train.jsonl"
        assert kwargs == {"repo_type": "dataset", "revision": "fixed-revision"}
        return mix

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    output = tmp_path / "combined"
    antidoom.export_mix(
        SimpleNamespace(benchmark_data=base, tested_benchmarks=tested, mix_revision="fixed-revision", output=output)
    )
    rows = list(antidoom.read_jsonl(output / "prompts.jsonl"))
    assert rows[:2] == base_rows
    appended = rows[2:]
    excluded = {
        "gsm8k_train",
        "math_lighteval_train",
        "mmlu_auxiliary_train",
        "math_qa_train",
        "commonsense_qa_train",
        "open_perfectblend_metamathqa",
    } | ({"pubmedqa_artificial_train"} if pubmed_evaluated else set())
    assert {row["source"]["source_id"] for row in appended} == set(antidoom.MIX_SOURCES) - excluded
    assert all(set(row) == {"query", "source"} for row in rows)
    assert len({row["source"]["problem_key"] for row in appended}) == len(appended)
    apps = next(row for row in appended if row["source"]["source_id"] == "apps_train")
    assert apps["source"]["hf_repo"] == "apps_train-upstream"
    assert apps["source"]["source_license"] == "mit"
    assert apps["source"]["max_generated_tokens"] == 24576
    assert apps["source"]["dataset_revision"] == "fixed-revision"
    assert apps["source"]["source_config"] is None
    ifstruct = next(row for row in appended if row["source"]["source_id"] == "ifstruct_train_generated")
    assert ifstruct["source"]["source_license_url"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["benchmark_questions"] == 2
    assert manifest["total_questions"] == len(rows)
    assert manifest["skipped_mix_rows"]["duplicate_prompt"] == 2
    assert manifest["excluded_mix_sources"]["open_perfectblend_metamathqa"] == "requested_exclusion"
    assert manifest["excluded_mix_sources"]["math_qa_train"] == "already_evaluated"


def test_repetition_and_original_token_boundary():
    pattern = "Let me reconsider this approach. "
    text = "Thinking. " + pattern * 5
    ids = list(text.encode())
    found, hit = antidoom.find_inner_repetition(text, sample_interval=1)
    assert found
    assert hit.repeats >= 4
    index = antidoom.find_rejected_token_index(Tokenizer(), ids, hit)
    assert len(bytes(ids[:index]).decode()) <= hit.repeat_start
    assert not antidoom.find_inner_repetition("First 2+2=4. Then 4*3=12. Answer: 12.")[0]


def test_utf8_boundary_chooses_first_token_of_character():
    ids = list("prefix思考思考".encode())
    hit = antidoom.RepeatHit(start=4, end=10, period=2, repeats=2, snippet="思考")
    assert antidoom.find_rejected_token_index(Tokenizer(), ids, hit) == 6


def test_pair_uses_exact_rollout_context_and_model_candidates():
    text = "Wait, let me reconsider this approach. " * 5
    row = {"prompt_ids": [17, 19], "response_ids": list(text.encode())}
    calls = []

    def forward(ids, **kwargs):
        calls.append(ids.tolist()[0])
        logits = torch.zeros(1, 1, 256)
        logits[0, 0, [65, 66, 32]] = torch.tensor([10.0, 9.0, 8.0])
        return SimpleNamespace(logits=logits)

    pair = antidoom.extract_ftpo_pair(row, Tokenizer(), forward, "cpu", 1000, top_k=3)
    index = pair["rejected_token_index"]
    assert calls == [row["prompt_ids"] + row["response_ids"][:index]]
    assert pair["prompt_ids"] == calls[0]
    assert pair["rejected_token_id"] == row["response_ids"][index]
    assert pair["chosen_ids"] == [65, 66]
    assert antidoom.extract_ftpo_pair(row, Tokenizer(), forward, "cpu", 1) is None
    assert len(calls) == 1


def test_ftpo_matches_notebook_loss_and_gradient_direction():
    logits = torch.tensor([[0.0, 0.5, -0.5, 1.0, 0.0]], requires_grad=True)
    reference = torch.zeros_like(logits)
    loss, metrics = antidoom.ftpo_loss(logits, reference, [1, 2], 0)
    deltas = torch.tensor([0.5, -0.5])
    expected_pref = (torch.nn.functional.softplus(2 - deltas) * ((2 - deltas) / 2).clamp(0, 1)).mean()
    expected = expected_pref + 0.4 * torch.tensor(0.5)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["pref_loss"], expected_pref)
    loss.backward()
    assert logits.grad[0, 0] > 0
    assert (logits.grad[0, [1, 2]] < 0).all()
    # The non-target logit is tethered toward the reference.
    assert logits.grad[0, 3] > 0


def test_batched_logits_keep_original_causal_contexts_and_gradients():
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    rows = [{"prompt_ids": [1, 2]}, {"prompt_ids": [3, 4, 5, 6]}, {"prompt_ids": [7, 8, 9]}]
    batched = antidoom.batch_logits(model, rows, "cpu")
    single = torch.cat([antidoom.last_logits(model, row["prompt_ids"], "cpu") for row in rows])
    torch.testing.assert_close(batched, single, atol=1e-6, rtol=1e-5)
    batched.square().mean().backward()
    batch_grad = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad()
    single.square().mean().backward()
    torch.testing.assert_close(batch_grad, model.model.embed_tokens.weight.grad, atol=1e-6, rtol=1e-5)


def test_microbatches_bound_padding_and_preserve_all_contexts():
    rows = [{"prompt_ids": [1] * length} for length in [9, 2, 5, 1, 4, 3]]
    batches = list(antidoom.microbatches(rows, 3, 12))
    assert [[len(row["prompt_ids"]) for row in batch] for batch in batches] == [[1, 2, 3], [4, 5], [9]]
    assert sum(len(batch) for batch in batches) == len(rows)
    assert all(len(batch) * max(len(row["prompt_ids"]) for row in batch) <= 12 for batch in batches)


def test_ftpo_target_tether_and_satisfied_margin():
    logits = torch.tensor([[0.0, 3.0, 0.0, 0.0]])
    loss, metrics = antidoom.ftpo_loss(logits, torch.zeros_like(logits), [1], 0)
    assert metrics["pref_loss"].item() == 0
    assert math.isclose(loss.item(), 0.05 * 2.5**2 / 2, rel_tol=1e-6)
    assert metrics["margin_win"].item() == 1


def test_ftpo_data_rejects_bad_ids_and_discards_overlength(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"prompt_ids": [1, 2], "chosen_ids": [3], "rejected_token_id": 4}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "prompt_ids": [1] * 10}) + "\n")
    assert antidoom.load_ftpo_data(path, 8, 4) == [row]
    path.write_text(json.dumps({**row, "chosen_ids": [4]}) + "\n")
    with pytest.raises(ValueError, match="distinct"):
        antidoom.load_ftpo_data(path, 8, 4)
    path.write_text(json.dumps({**row, "rejected_token_id": 8}) + "\n")
    with pytest.raises(ValueError, match="vocabulary"):
        antidoom.load_ftpo_data(path, 8, 4)


def test_prepare_groups_siblings_and_keeps_original_validation_prompts(tmp_path, monkeypatch):
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [
        {"prompt_ids": [key, variant], "response_ids": [20], "source": {"problem_key": str(key)}}
        for key in range(40)
        for variant in (1, 2)
        for _ in range(2)
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(antidoom, "get_tokenizer", lambda *args, **kwargs: None)
    model = SimpleNamespace(to=lambda _: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(antidoom, "load_model", lambda _: model)

    def extract(row, *args):
        return {
            "prompt_key": antidoom.prompt_key(row["prompt_ids"]),
            "problem_key": row["source"]["problem_key"],
            "source_prompt_ids": row["prompt_ids"],
            "prompt_ids": row["prompt_ids"] + [10],
            "chosen_ids": [11],
            "rejected_token_id": 12,
            "source": row["source"],
        }

    monkeypatch.setattr(antidoom, "extract_ftpo_pair", extract)
    output = tmp_path / "data"
    antidoom.prepare(
        SimpleNamespace(
            output=output,
            model="rwkv7-test",
            rollouts=rollouts,
            max_length=8192,
            top_k=50,
            num_chosen=20,
            seed=42,
            validation_fraction=0.25,
        )
    )
    train = list(antidoom.read_jsonl(output / "train.jsonl"))
    val = list(antidoom.read_jsonl(output / "validation.jsonl"))
    assert train and val
    assert len(train) + len(val) == len(rows)
    assert not {row["prompt_key"] for row in train} & {row["prompt_key"] for row in val}
    assert not {row["problem_key"] for row in train} & {row["problem_key"] for row in val}
    prompts = list(antidoom.read_jsonl(output / "validation_prompts.jsonl"))
    assert len(prompts) * 2 == len(val)
    assert {tuple(row["prompt_ids"]) for row in prompts} == {tuple(row["source_prompt_ids"]) for row in val}
    assert {row["source"]["problem_key"] for row in prompts} == {row["problem_key"] for row in val}


@pytest.mark.skipif("use_nora" not in LoraConfig.__dataclass_fields__, reason="requires Joluck/NoRA")
def test_nora_reference_and_merged_logits_after_optimizer_step():
    torch.manual_seed(7)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()
    prompt_ids = [1, 2, 3, 4]
    reference = antidoom.last_logits(base, prompt_ids, "cpu").detach()
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "v_proj"],
            use_nora=True,
        ),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    loss, _ = antidoom.row_loss(model, {"prompt_ids": prompt_ids, "chosen_ids": [7, 8], "rejected_token_id": 9}, "cpu")
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    optimizer.step()
    with torch.no_grad(), model.disable_adapter():
        torch.testing.assert_close(antidoom.last_logits(model, prompt_ids, "cpu"), reference)
    model.eval()
    adapted = antidoom.last_logits(model, prompt_ids, "cpu").detach()
    assert not torch.allclose(adapted, reference)
    merged = model.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(antidoom.last_logits(merged, prompt_ids, "cpu"), adapted, atol=1e-6, rtol=1e-5)
    assert not any("lora_" in name for name, _ in merged.named_parameters())
