"""RWKV Antidoom: own-model rollouts, FTPO, and merged NoRA weights.

Repetition detection and FTPO objective adapted from Liquid4All/antidoom
(Apache-2.0) and Liquid4All/cookbook/finetuning/notebooks/antidoom_with_trl.ipynb.
Changes: exact RWKV rollout IDs, native SkyRL FSDPStrategy, NoRA, and automatic
merge. Generation follows examples.train.rwkv.rollout's vLLM token API.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import random
import resource
import signal
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from skyrl.utils.tok import get_tokenizer

NORA_REVISION = "4669c35ce0ffd027456272ee5050c0f58ac8d22a"
TARGET_MODULES = ["r_proj", "k_proj", "v_proj", "o_proj"]
ANTIDOOM_DATASET = "LiquidAI/antidoom-mix-v1.0"
ANTIDOOM_REVISION = "79ff2d479c5e59ea9973a66a7ca9ce8b703f9c7b"
# Dataset-card source IDs mapped to scoreboard benchmark names and task fields.
MIX_SOURCES = {
    "gsm8k_train": ("gsm8k", "maths"),
    "math_lighteval_train": ("math", "maths"),
    "mmlu_auxiliary_train": ("mmlu", "knowledge"),
    "math_qa_train": ("mathqa", "maths"),
    "commonsense_qa_train": ("commonsenseqa", "reasoning"),
    "pubmedqa_artificial_train": ("pubmedqa", "knowledge"),
    "apps_train": ("apps", "coding"),
    "open_perfectblend_metamathqa": ("math", "maths"),
    "open_perfectblend_ultrainteract": ("ultrainteract", "reasoning"),
    "open_perfectblend_ultrachat200k": ("ultrachat200k", "dialogue"),
    "open_perfectblend_evol_codealpaca": ("evol_codealpaca", "coding"),
    "open_perfectblend_autoif": ("autoif", "instruction_following"),
    "open_perfectblend_lmsys_arena": ("lmsys_arena", "dialogue"),
    "ifstruct_train_generated": ("ifstruct", "instruction_following"),
}


@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self):
        return self.start + self.period


def _verify_repetition_at(text, start_pos, period, min_repeats, min_total_repeated):
    pattern = text[start_pos : start_pos + period]
    reps, pos = 0, start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos
    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period
    if reps >= min_repeats and reps * period >= min_total_repeated:
        return True, RepeatHit(start_pos, end_pos, period, reps, pattern[:100])
    return False, None


def find_inner_repetition(
    text,
    min_repeats=4,
    max_period=1024,
    min_period=1,
    min_total_repeated=60,
    sample_len=16,
    sample_interval=128,
):
    if len(text) < min_total_repeated:
        return False, None
    for sample_pos in range(0, len(text) - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]
        for other_pos in (
            text.find(fingerprint, sample_pos + sample_len),
            text.rfind(fingerprint, 0, sample_pos),
        ):
            period = abs(other_pos - sample_pos)
            if other_pos >= 0 and min_period <= period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    min(sample_pos, other_pos),
                    period,
                    min_repeats,
                    min_total_repeated,
                )
                if found:
                    return True, hit
    return False, None


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_key(prompt_ids):
    return hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()


def select_benchmarks(args):
    """Select whole tasks from current scoreboard score metadata."""
    benchmarks = {}
    for row in read_jsonl(args.scoreboard_records):
        model = row["model"]
        if model["data_version"].casefold() != "g1k" or model["param_size"] != args.param_size:
            continue
        if row["truncation_rate"] < args.min_truncation_rate:
            continue
        name = row["benchmark"]
        benchmark = benchmarks.setdefault(
            name,
            {
                "name": name,
                "field": row["field"],
                "framework": row["framework"],
                "num_samples": row["num_samples"],
                "cot_mode": row["cot_mode"],
                "evidence": [],
            },
        )
        benchmark["evidence"].append(
            {
                "model": model,
                "score_id": row["score_id"],
                "truncation_rate": row["truncation_rate"],
                "max_generated_tokens": row["sampling"]["max_generated_tokens"],
            }
        )
    if not benchmarks:
        raise ValueError("No whole benchmarks meet the G1k truncation-rate threshold.")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = sorted(
        benchmarks.values(), key=lambda row: (-max(s["truncation_rate"] for s in row["evidence"]), row["name"])
    )
    manifest = {
        "scoreboard": "https://eval.rwkv.rs/scoreboard",
        "selected_at": datetime.now(UTC).isoformat(),
        "data_version": "G1k",
        "param_size": args.param_size,
        "min_truncation_rate": args.min_truncation_rate,
        "scope": "complete_benchmarks",
        "benchmarks": selected,
        "scoreboard_num_samples": sum(row["num_samples"] for row in selected),
    }
    (args.output / "benchmarks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "benchmarks.toml").open("x") as handle:
        for field in sorted({row["field"] for row in selected}):
            handle.write(f'{field} = {json.dumps([row["name"] for row in selected if row["field"] == field])}\n')
    print(json.dumps(manifest), flush=True)


def export_benchmarks(args):
    """Load complete native LightEval task Docs, without an evaluation pipeline/cache."""
    # Read the producer's task/formatter implementation in the SkyRL interpreter.
    sys.path.insert(0, str(args.lighteval_project / "src"))
    sys.path.insert(0, str(args.lighteval_project))
    from src.rwkv_eval.main import RWKVRegistry
    from lighteval.tasks.lighteval_task import LightevalTask

    manifest = json.loads(args.benchmarks.read_text())
    specs = manifest["benchmarks"]
    random.seed(args.seed)
    tasks = RWKVRegistry(tasks=",".join(row["name"] for row in specs)).load_tasks()
    args.output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (args.output / "prompts.jsonl").open("x") as handle:
        for spec in specs:
            matched = [
                task for task in tasks.values() if task.name == spec["name"] or task.name.startswith(spec["name"] + ":")
            ]
            if not matched:
                raise ValueError(f'No native LightEval tasks match {spec["name"]}.')
            for task in matched:
                LightevalTask.load_datasets({task.full_name: task}, 1)
                for doc in task.get_docs(None):
                    question = doc.query
                    if spec["name"].startswith("gpqa:"):
                        question = task.dataset[task.evaluation_split[0]][int(doc.id)]["Question"]
                    key = hashlib.sha256(question.strip().encode()).hexdigest()
                    write_json(
                        handle,
                        {
                            "query": doc.query,
                            "source": {
                                "benchmark": spec["name"],
                                "task": task.full_name,
                                "doc_id": doc.id,
                                "field": spec["field"],
                                "problem_key": key,
                                "task_version": task.dataset_config.version,
                                "hf_repo": task.dataset_config.hf_repo,
                                "evaluation_splits": task.evaluation_split,
                                "prompt_template": (
                                    "bot"
                                    if spec["field"] in {"maths", "coding"} or spec["name"].startswith("gpqa:")
                                    else "assistant"
                                ),
                                "generation_prompt": "fake_think" if spec["cot_mode"] == "FakeCoT" else "open_think",
                                "max_generated_tokens": max(row["max_generated_tokens"] for row in spec["evidence"]),
                            },
                        },
                    )
                    counts[spec["name"]] += 1
                task.dataset = None
                task._docs = None
    manifest.update({"exported_questions": dict(counts), "total_questions": sum(counts.values()), "seed": args.seed})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(counts)), flush=True)


def export_mix(args):
    """Append untested AntiDoom source prompts to complete benchmark questions."""
    from huggingface_hub import hf_hub_download

    tested = {row["benchmark"].casefold().split(":")[0] for row in read_jsonl(args.tested_benchmarks)}
    excluded = {
        source: "requested_exclusion" if benchmark in {"gsm8k", "mmlu", "math"} else "already_evaluated"
        for source, (benchmark, _) in MIX_SOURCES.items()
        if benchmark in {"gsm8k", "mmlu", "math"} or benchmark in tested
    }
    mix_path = hf_hub_download(ANTIDOOM_DATASET, "train.jsonl", repo_type="dataset", revision=args.mix_revision)
    manifest = json.loads((args.benchmark_data / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    seen, counts, skipped = set(), Counter(), Counter()
    benchmark_count = 0
    with (args.output / "prompts.jsonl").open("x") as handle:
        for row in read_jsonl(args.benchmark_data / "prompts.jsonl"):
            seen.add(hashlib.sha256(row["query"].strip().encode()).hexdigest())
            write_json(handle, row)
            benchmark_count += 1
        for row in read_jsonl(mix_path):
            source_id = row["id"].split(":")[1]
            if source_id not in MIX_SOURCES:
                raise ValueError(f"Unknown AntiDoom mix source: {source_id}")
            if source_id in excluded:
                skipped[source_id] += 1
                continue
            turns = row["conversations"]
            if len(turns) != 1 or turns[0]["from"] != "human":
                raise ValueError(f'Expected a prompt-only AntiDoom row: {row["id"]}')
            query = turns[0]["value"]
            key = hashlib.sha256(query.strip().encode()).hexdigest()
            if key in seen:
                skipped["duplicate_prompt"] += 1
                continue
            seen.add(key)
            benchmark, field = MIX_SOURCES[source_id]
            high_difficulty = field == "coding" or benchmark == "ultrainteract"
            write_json(
                handle,
                {
                    "query": query,
                    "source": {
                        "benchmark": f"antidoom:{source_id}",
                        "source_id": source_id,
                        "field": field,
                        "problem_key": key,
                        "dataset": ANTIDOOM_DATASET,
                        "dataset_revision": args.mix_revision,
                        "doc_id": row["id"],
                        "hf_repo": row["source"],
                        "source_config": row.get("source_config"),
                        "source_split": row["source_split"],
                        "source_sample_id": row["source_sample_id"],
                        "source_license": row["source_license"],
                        "source_license_url": row.get("source_license_url"),
                        "heldout_policy": row["heldout_policy"],
                        "prompt_template": "bot" if high_difficulty else "assistant",
                        "generation_prompt": "open_think",
                        "max_generated_tokens": 24576 if high_difficulty else 4096,
                    },
                },
            )
            counts[source_id] += 1
    manifest.update(
        {
            "mix_dataset": ANTIDOOM_DATASET,
            "mix_revision": args.mix_revision,
            "tested_benchmarks": sorted(tested),
            "excluded_mix_sources": excluded,
            "benchmark_questions": benchmark_count,
            "mix_questions": dict(counts),
            "skipped_mix_rows": dict(skipped),
            "total_questions": benchmark_count + sum(counts.values()),
        }
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def find_rejected_token_index(tokenizer, generated_ids, hit):
    """Map the repetition boundary onto original IDs, including split UTF-8 tokens."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    char_pos = hit.repeat_start
    while char_pos < len(text) and text[char_pos].isspace():
        char_pos += 1
    before = text[:char_pos]
    low, high = 0, len(generated_ids)
    while low < high:
        index = (low + high) // 2
        prefix = tokenizer.decode(generated_ids[: index + 1], skip_special_tokens=True)
        # Prefixes grow monotonically. A partial UTF-8 character at this
        # boundary belongs to its first byte token, even before completion.
        if len(prefix) > char_pos and prefix.startswith(before):
            high = index
        else:
            low = index + 1
    return low if low < len(generated_ids) else None


def last_logits(model, prompt_ids, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    return model(ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()


def batch_logits(model, rows, device):
    """Right-pad causal contexts and select each original final position."""
    lengths = [len(row["prompt_ids"]) for row in rows]
    ids = torch.zeros(len(rows), max(lengths), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        ids[index, : lengths[index]] = torch.tensor(row["prompt_ids"], device=device)
    positions = torch.tensor(lengths, device=device) - 1
    logits = model(ids, use_cache=False, logits_to_keep=positions).logits
    indices = torch.arange(len(rows), device=device)
    # Each row's loss is causal: trailing pads are outside its selected prefix.
    return logits[indices, indices].float()


def microbatches(rows, batch_size, max_batch_tokens):
    """Bucket by length and bound the padded token count of each microbatch."""
    batch = []
    for row in sorted(rows, key=lambda row: len(row["prompt_ids"])):
        length = len(row["prompt_ids"])
        if batch and (len(batch) == batch_size or (len(batch) + 1) * length > max_batch_tokens):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def extract_ftpo_pair(row, tokenizer, model, device, max_length, top_k=50, num_chosen=20):
    generated_ids = row["response_ids"]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if row.get("cached_text_provenance"):
        hit = RepeatHit(**row["repeat_hit"])
    else:
        found, hit = find_inner_repetition(text)
        if not found:
            return None
    index = find_rejected_token_index(tokenizer, generated_ids, hit)
    if index is None:
        return None
    context_ids = row["prompt_ids"] + generated_ids[:index]
    if not context_ids or len(context_ids) > max_length:
        return None
    rejected_id = generated_ids[index]
    with torch.no_grad():
        logits = last_logits(model, context_ids, device)[0]
    chosen_ids = [
        token
        for token in logits.topk(min(top_k, logits.numel())).indices.tolist()
        if token != rejected_id and tokenizer.decode([token], skip_special_tokens=True).strip()
    ][:num_chosen]
    if not chosen_ids:
        return None
    return {
        "prompt_key": prompt_key(row["prompt_ids"]),
        "problem_key": (row.get("source") or {}).get("problem_key", prompt_key(row["prompt_ids"])),
        "source_prompt_ids": row["prompt_ids"],
        "prompt_ids": context_ids,
        "chosen_ids": chosen_ids,
        "rejected_token_id": rejected_id,
        "rejected_token_index": index,
        "repeat_hit": asdict(hit),
        "source": row.get("source"),
        **({"cached_text_provenance": row["cached_text_provenance"]} if row.get("cached_text_provenance") else {}),
    }


def encode_cached_response(row, tokenizer):
    """Reconstruct a text-only cache trajectory and verify its supplied exact loop."""
    prompt, text = row["prompt_text"], row["response_text"]
    hit = RepeatHit(**row["repeat_hit"])
    pattern = text[hit.start : hit.start + hit.period]
    if (
        hit.start < 0
        or hit.period < 1
        or hit.repeats < 4
        or not hit.start + 4 * hit.period <= hit.end <= len(text)
        or text[hit.start : hit.start + hit.repeats * hit.period] != pattern * hit.repeats
    ):
        raise ValueError("Cached FTPO boundary must identify at least four consecutive exact repeats.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(text, add_special_tokens=False)
    if (
        tokenizer.decode(prompt_ids, skip_special_tokens=True) != prompt
        or tokenizer.decode(response_ids, skip_special_tokens=True) != text
    ):
        raise ValueError("Cached prompt and response must round-trip through the training tokenizer.")
    provenance = row["cached_text_provenance"]
    if hashlib.sha256(text.encode()).hexdigest() != provenance["response_sha256"]:
        raise ValueError("Cached response SHA256 does not match the boundary provenance.")
    return {**row, "prompt_ids": prompt_ids, "response_ids": response_ids}


def ftpo_loss(
    logits,
    ref_logits,
    chosen_ids,
    rejected_id,
    lambda_mse=0.4,
    lambda_mse_target=0.05,
    tau_mse_target=0.5,
    clip_epsilon_logits=2.0,
):
    """Notebook FTPO at one next-token position; preserve weight gradients."""
    logits, ref_logits = logits.float(), ref_logits.float()
    chosen = torch.tensor(chosen_ids, device=logits.device)
    delta_tok = logits[0, chosen] - logits[0, rejected_id]
    weights = ((clip_epsilon_logits - delta_tok) / clip_epsilon_logits).clamp(0, 1)
    pref_loss = (F.softplus(clip_epsilon_logits - delta_tok) * weights).mean()
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask[0, chosen] = True
    target_mask[0, rejected_id] = True
    diff = logits - ref_logits
    mse_elem = diff.square()[~target_mask].mean()
    mse_target = (diff.abs()[target_mask] - tau_mse_target).clamp(min=0).square().mean()
    loss = pref_loss + lambda_mse * mse_elem + lambda_mse_target * mse_target
    return loss, {
        "loss": loss.detach(),
        "pref_loss": pref_loss.detach(),
        "mse_elem": mse_elem.detach(),
        "mse_target": mse_target.detach(),
        "chosen_win": (delta_tok > 0).float().mean().detach(),
        "margin_win": (delta_tok >= clip_epsilon_logits).float().mean().detach(),
    }


def load_model(model_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, wkv_mode="fp32io16")
    if model.config.model_type != "rwkv":
        raise ValueError("This example requires an RWKV model.")
    # RWKV's ordinary BF16 head is selected by training mode. Read-only
    # scoring uses no_grad; FP16 free generation is handled by vLLM.
    return model.train()


async def generate(args):
    import aiohttp
    from datasets import load_dataset
    from transformers import AutoConfig
    from examples.train.rwkv.rollout import ROLLOUT_SAMPLING_PARAMS, SAMPLING_PARAMS, start_servers

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < args.concurrency + 1024:
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.concurrency + 1024, hard_limit))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    tokenizer = get_tokenizer(args.model, trust_remote_code=False)
    eos_token_id = AutoConfig.from_pretrained(args.model).eos_token_id
    if args.input:
        samples = read_jsonl(args.input)
    else:
        samples = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
    sampling = ROLLOUT_SAMPLING_PARAMS if args.sampling == "rollout" else SAMPLING_PARAMS
    input_digest = None
    if args.input:
        with args.input.open("rb") as handle:
            input_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    run_config = {
        "model": str(Path(args.model).resolve()),
        "input_sha256": input_digest,
        "sampling": sampling,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "samples_per_prompt": args.samples_per_prompt,
        "prompt_template": args.prompt_template,
        "generation_prompt": args.generation_prompt,
    }
    manifest_path = args.output / "generation.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != run_config:
            raise ValueError("Resume requires the same checkpoint, input prompts, seed and decoding settings.")
    else:
        manifest_path.write_text(json.dumps(run_config, indent=2) + "\n")
    processes, counts = [], Counter()
    completed = set()
    if args.resume:
        for path in sorted(args.output.glob("rollouts*.jsonl")):
            for row in read_jsonl(path):
                if Path(row["model"]).resolve() != Path(args.model).resolve():
                    raise ValueError("Resume requires rollouts from the same checkpoint.")
                completed.add(row["prompt_index"])
                counts["rollouts"] += 1
                counts["generated_tokens"] += len(row["response_ids"])
                counts["doom_loops"] += int(row["repeat_hit"] is not None)
                counts["truncated"] += int(row["finish_reason"] == "length")
    tokens_before_resume = counts["generated_tokens"]
    started = time.monotonic()
    try:
        urls = (
            args.base_url.split(",")
            if args.base_url
            else await start_servers(
                args,
                processes,
                enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        with ExitStack() as stack:
            handles = [
                stack.enter_context(
                    (args.output / (f"rollouts_gpu_{index}.jsonl" if args.shard_rollouts else "rollouts.jsonl")).open(
                        "a" if args.resume else "x"
                    )
                )
                for index in range(len(urls) if args.shard_rollouts else 1)
            ]
            async with aiohttp.ClientSession(
                timeout=timeout, connector=aiohttp.TCPConnector(limit=args.concurrency, force_close=True)
            ) as session:

                async def request(index, sample):
                    handle = handles[index % len(handles)]
                    if "prompt_ids" in sample:
                        prompt_ids = sample["prompt_ids"]
                    else:
                        if "messages" in sample:
                            messages = sample["messages"]
                        elif "conversations" in sample:
                            roles = {"human": "user", "gpt": "assistant", "system": "system"}
                            messages = [
                                {"role": roles[turn["from"]], "content": turn["value"]}
                                for turn in sample["conversations"]
                            ]
                            if messages[-1]["role"] == "assistant":
                                messages = messages[:-1]
                        else:
                            messages = [{"role": "user", "content": sample["query"]}]
                        encoded = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            rwkv_prompt_template=(sample.get("source") or {}).get(
                                "prompt_template", args.prompt_template
                            ),
                            rwkv_generation_prompt=(sample.get("source") or {}).get(
                                "generation_prompt", args.generation_prompt
                            ),
                        )
                        prompt_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
                    if len(prompt_ids) >= args.max_model_len:
                        raise ValueError(
                            "A whole-benchmark prompt exceeds --max-model-len; increase the context budget."
                        )
                    budget = min(
                        args.max_tokens, (sample.get("source") or {}).get("max_generated_tokens", args.max_tokens)
                    )
                    max_tokens = min(budget, args.max_model_len - len(prompt_ids))
                    counts["shortened_budgets"] += int(max_tokens < budget)
                    async with semaphore:
                        for attempt in range(3):
                            try:
                                async with session.post(
                                    urls[index % len(urls)].rstrip("/") + "/v1/completions",
                                    json={
                                        "model": args.model,
                                        "prompt": prompt_ids,
                                        "max_tokens": max_tokens,
                                        "n": args.samples_per_prompt,
                                        "seed": args.seed + index,
                                        "stream": False,
                                        "return_token_ids": True,
                                        "stop_token_ids": [eos_token_id],
                                        **sampling,
                                    },
                                ) as response:
                                    response.raise_for_status()
                                    payload = await response.json()
                                break
                            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as exc:
                                if attempt == 2:
                                    raise
                                print(f"prompt_index={index} POST retry={attempt + 1}: {exc}", flush=True)
                                await asyncio.sleep(1)
                    for choice in payload["choices"]:
                        response_ids = choice["token_ids"]
                        text = tokenizer.decode(response_ids, skip_special_tokens=True)
                        found, hit = find_inner_repetition(text)
                        write_json(
                            handle,
                            {
                                "model": args.model,
                                "prompt_index": index,
                                "seed": args.seed + index,
                                "prompt_ids": prompt_ids,
                                "response_ids": response_ids,
                                "finish_reason": choice["finish_reason"],
                                "max_tokens": max_tokens,
                                "text": text,
                                "repeat_hit": asdict(hit) if hit else None,
                                "source": sample.get("source"),
                            },
                        )
                        handle.flush()
                        counts["rollouts"] += 1
                        counts["generated_tokens"] += len(response_ids)
                        counts["doom_loops"] += int(found)
                        counts["truncated"] += int(choice["finish_reason"] == "length")
                        if counts["rollouts"] % 128 == 0:
                            elapsed = time.monotonic() - started
                            progress = {
                                **counts,
                                "elapsed_seconds": elapsed,
                                "tokens_per_second": (counts["generated_tokens"] - tokens_before_resume) / elapsed,
                            }
                            (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                            print(json.dumps(progress), flush=True)

                tasks = set()
                for index, sample in enumerate(samples):
                    if args.limit and index >= args.limit:
                        break
                    if index in completed:
                        continue
                    if len(tasks) >= args.concurrency:
                        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            task.result()
                    tasks.add(asyncio.create_task(request(index, sample)))
                await asyncio.gather(*tasks)
        summary = {
            "model": args.model,
            "sampling": sampling,
            "seed": args.seed,
            "prompt_template": args.prompt_template,
            "generation_prompt": args.generation_prompt,
            "elapsed_seconds": time.monotonic() - started,
            **counts,
            "doom_loop_rate": counts["doom_loops"] / max(counts["rollouts"], 1),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        for process in processes:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


def prepare(args):
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(args.model, trust_remote_code=False)
    model = load_model(args.model).to("cuda")
    paths = sorted(args.rollouts.rglob("*.jsonl")) if args.rollouts.is_dir() else [args.rollouts]
    if world > 1 and len(paths) < world:
        raise ValueError("Distributed extraction requires at least one rollout shard per rank; use --shard-rollouts.")
    paths = paths[rank::world]
    suffix = f"_rank_{rank}" if world > 1 else ""
    counts, validation_prompts = Counter(), set()
    with (
        (args.output / f"train{suffix}.jsonl").open("x") as train,
        (args.output / f"validation{suffix}.jsonl").open("x") as val,
        (args.output / f"validation_prompts{suffix}.jsonl").open("x") as prompts,
    ):
        for path in paths:
            for row in read_jsonl(path):
                if args.cached_text:
                    row = encode_cached_response(row, tokenizer)
                if "model" in row and Path(row["model"]).resolve() != Path(args.model).resolve():
                    raise ValueError("FTPO extraction must use the checkpoint that generated the rollout.")
                pair = extract_ftpo_pair(row, tokenizer, model, "cuda", args.max_length, args.top_k, args.num_chosen)
                counts["rollouts"] += 1
                if pair is None:
                    continue
                # All sibling rollouts and variants of one problem share a partition.
                digest = hashlib.sha256(f'{args.seed}:{pair["problem_key"]}'.encode()).hexdigest()
                validation = int(digest[:16], 16) / 2**64 < args.validation_fraction
                write_json(val if validation else train, pair)
                counts["validation" if validation else "train"] += 1
                if validation and pair["prompt_key"] not in validation_prompts:
                    write_json(prompts, {"prompt_ids": pair["source_prompt_ids"], "source": pair["source"]})
                    validation_prompts.add(pair["prompt_key"])
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, dict(counts))
        if rank == 0:
            counts = sum((Counter(value) for value in gathered), Counter())
            for name in ("train", "validation", "validation_prompts"):
                with (args.output / f"{name}.jsonl").open("xb") as handle:
                    for index in range(world):
                        path = args.output / f"{name}_rank_{index}.jsonl"
                        with path.open("rb") as source:
                            shutil.copyfileobj(source, handle)
                        path.unlink()
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        manifest = {
            "model": args.model,
            "max_length": args.max_length,
            "seed": args.seed,
            "trajectory_format": "retokenized_evaluation_cache" if args.cached_text else "original_rollout_ids",
            **counts,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)


def score(args):
    """Score fresh paired generations using the producer's native benchmark metrics."""
    sys.path.insert(0, str(args.lighteval_project / "src"))
    sys.path.insert(0, str(args.lighteval_project))
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc
    from lighteval.tasks.tasks.ifbench.main import RWKVIFBench
    from lighteval.tasks.tasks.lcb.main import CodegenMetric
    from src.rwkv_eval.answer_extract.free_response import answer_extractor_for_benchmark, extract_answer
    from src.rwkv_eval.main import RWKVExactMatches

    inputs = list(read_jsonl(args.input))
    paths = sorted(args.rollouts.glob("rollouts*.jsonl"))
    responses = {}
    for path in paths:
        for row in read_jsonl(path):
            index = row["prompt_index"]
            if index in responses:
                raise ValueError("Paired evaluation requires one fresh response per question.")
            responses[index] = row
    if set(responses) != set(range(len(inputs))):
        raise ValueError("Evaluation responses do not cover the exact input question set.")
    args.output.mkdir(parents=True, exist_ok=False)
    totals, benchmarks = Counter(), {}
    with (args.output / "scores.jsonl").open("x") as handle:
        for index, sample in enumerate(inputs):
            row = responses[index]
            if row["prompt_ids"] != sample["prompt_ids"]:
                raise ValueError("Evaluation must retain the original held-out prompt IDs.")
            doc = Doc(**sample["doc"])
            name = sample["source"]["benchmark"]
            extractor = answer_extractor_for_benchmark(name)
            pred = extract_answer(row["text"], extractor, len(doc.choices), rendered_prompt=sample["prompt_text"])
            response = ModelResponse(
                text=[row["text"]], text_post_processed=[pred or ""], rendered_prompt=sample["prompt_text"]
            )
            if name.startswith("ifbench"):
                quality = RWKVIFBench().compute(doc=doc, model_response=response)
                native_score = quality["prompt_level_strict_acc"]
            elif extractor == "code":
                # CodegenMetric retains the native timeout and reliability guard.
                native_score = CodegenMetric().compute(doc=doc, model_response=response) if pred else 0.0
                quality = {"pass_at_1": float(native_score)}
            else:
                native_score = RWKVExactMatches(choice=extractor == "choice").compute(doc, response)
                quality = {"exact_match": float(native_score)}
            found, hit = find_inner_repetition(
                row["text"], max_period=args.max_period, sample_interval=args.sample_interval
            )
            values = {
                "questions": 1,
                "doom_loops": int(found),
                "truncated": int(row["finish_reason"] == "length"),
                "answer_extracted": int(bool(pred)),
                "strict_correct": int(native_score == 1),
                "native_score_sum": float(native_score),
                "generated_tokens": len(row["response_ids"]),
            }
            for key, value in quality.items():
                if isinstance(value, list):
                    values[key + "_passed"] = sum(value)
                    values[key + "_count"] = len(value)
                else:
                    values[key + "_sum"] = float(value)
            totals.update(values)
            benchmarks.setdefault(name, Counter()).update(values)
            write_json(
                handle,
                {
                    **row,
                    "parsed_answer": pred,
                    "gold_answers": doc.get_golds(),
                    "strict_correct": bool(native_score == 1),
                    "native_score": float(native_score),
                    "quality": quality,
                    "repeat_hit": asdict(hit) if hit else None,
                },
            )
            handle.flush()
            print(f"scored={index + 1}/{len(inputs)} benchmark={name} score={native_score}", flush=True)

    def summary(values):
        count = values["questions"]
        return {
            **values,
            "doom_loop_rate": values["doom_loops"] / count,
            "truncation_rate": values["truncated"] / count,
            "answer_extraction_rate": values["answer_extracted"] / count,
            "strict_correct_rate": values["strict_correct"] / count,
            "native_score": values["native_score_sum"] / count,
        }

    result = {
        "overall": summary(totals),
        "benchmarks": {name: summary(values) for name, values in benchmarks.items()},
        "detector": {"max_period": args.max_period, "sample_interval": args.sample_interval, "min_repeats": 4},
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def load_ftpo_data(path, vocab_size, max_length):
    rows = []
    for row in read_jsonl(path):
        prompt, chosen, rejected = row["prompt_ids"], row["chosen_ids"], row["rejected_token_id"]
        if not prompt or not chosen or rejected in chosen or len(set(chosen)) != len(chosen):
            raise ValueError("FTPO rows need a non-empty context and distinct chosen/rejected IDs.")
        if any(not isinstance(token, int) or not 0 <= token < vocab_size for token in prompt + chosen + [rejected]):
            raise ValueError("FTPO IDs must be integer tokens in the model vocabulary.")
        if len(prompt) <= max_length:
            rows.append(row)
    if not rows:
        raise ValueError(f"No FTPO rows within max_length={max_length}: {path}")
    return rows


def batch_loss(model, rows, device):
    logits = batch_logits(model, rows, device)
    # The frozen base is the reference; no second model or reference adapter.
    with torch.no_grad(), model.disable_adapter():
        ref_logits = batch_logits(model, rows, device)
    results = [
        ftpo_loss(logits[index : index + 1], ref_logits[index : index + 1], row["chosen_ids"], row["rejected_token_id"])
        for index, row in enumerate(rows)
    ]
    return torch.stack([loss for loss, _ in results]).mean(), {
        key: torch.stack([metrics[key] for _, metrics in results]).mean() for key in results[0][1]
    }


def row_loss(model, row, device):
    return batch_loss(model, [row], device)


@torch.no_grad()
def evaluate(model, rows, device, batch_size=1, max_batch_tokens=32768):
    import torch.distributed as dist

    stats = torch.zeros(2, device=device, dtype=torch.float64)
    rank, world = dist.get_rank(), dist.get_world_size()
    batches = list(microbatches(rows, batch_size, max_batch_tokens))
    for start in range(0, len(batches), world):
        index = start + rank
        batch = batches[index] if index < len(batches) else [rows[0]]
        loss, _ = batch_loss(model, batch, device)
        if index < len(batches):
            stats[0] += loss * len(batch)
            stats[1] += len(batch)
    dist.all_reduce(stats)
    return {"eval/loss": (stats[0] / stats[1]).item(), "eval/samples": stats[1].item()}


def merge_and_save(model, tokenizer, model_path, output, probe_ids, device):
    """Check the adapter/merged next-token distribution and export complete HF weights."""
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        # Normalize NoRA in FP32 before using the vanilla LoRA inference path.
        # Keep these FP32 factors for merging: BF16 base-weight rounding can
        # erase small updates, whereas rollout uses FP16 projections.
        factors = {}
        for layer in model.modules():
            if isinstance(layer, LoraLayer):
                weight_a = layer.lora_A["default"].weight.data.float()
                weight_b = layer.lora_B["default"].weight.data.float()
                if layer.use_nora.get("default") is True:
                    weight_a = weight_a / (weight_a.norm(dim=0, keepdim=True) + 1e-6)
                    layer.use_nora["default"] = False
                factors[layer] = (weight_a, weight_b)
                layer.lora_A["default"].weight.data = weight_a
        model.half().eval()
        before = last_logits(model, probe_ids, device)
        for layer, (weight_a, weight_b) in factors.items():
            layer.get_base_layer().weight.data = layer.get_base_layer().weight.data.float()
            layer.lora_A["default"].weight.data = weight_a
            layer.lora_B["default"].weight.data = weight_b
        merged = model.merge_and_unload(safe_merge=True).half().to(device).eval()
        after = last_logits(merged, probe_ids, device)
    max_error = (before - after).abs().max().item()
    before_logp, after_logp = F.log_softmax(before, -1), F.log_softmax(after, -1)
    torch.testing.assert_close(after_logp, before_logp, atol=0.15, rtol=0.01)
    kl = (before_logp.exp() * (before_logp - after_logp)).sum(-1).item()
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    for path in [*Path(model_path).glob("*generation_config.json"), *Path(model_path).glob("tokenizer_config.json")]:
        (output / path.name).write_bytes(path.read_bytes())
    (output / "merge_metrics.json").write_text(
        json.dumps(
            {
                "max_logit_error": max_error,
                "kl_reference_merged": kl,
                "merge_dtype": "float32",
                "export_dtype": "float16",
            }
        )
        + "\n"
    )
    return merged


def train(args):
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import save_file
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.backends.skyrl_train.distributed.fsdp_utils import layered_summon_lora_params
    from skyrl.train.config import FSDPConfig, OptimizerConfig

    if "use_nora" not in LoraConfig.__dataclass_fields__:
        raise RuntimeError(f"Install Joluck/NoRA at {NORA_REVISION} in the project .venv; see the RWKV README.")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    run = None
    try:
        tokenizer = get_tokenizer(args.model, trust_remote_code=False)
        manifest_path = args.data / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if Path(manifest["model"]).resolve() != Path(args.model).resolve():
                raise ValueError("FTPO training must use the checkpoint that generated the preferences.")
        model = load_model(args.model)
        train_rows = load_ftpo_data(args.data / "train.jsonl", model.config.vocab_size, args.max_length)
        val_rows = load_ftpo_data(args.data / "validation.jsonl", model.config.vocab_size, args.max_length)
        if {row["prompt_key"] for row in train_rows} & {row["prompt_key"] for row in val_rows}:
            raise ValueError("FTPO train/validation partitions contain the same source prompts.")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.rank,
                lora_alpha=args.rank,
                lora_dropout=0.0,
                target_modules=TARGET_MODULES,
                bias="none",
                use_nora=True,
            ),
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batches = list(microbatches(train_rows, args.micro_batch_size, args.max_batch_tokens))
        global_batch = dist.get_world_size() * args.gradient_accumulation_steps
        groups = [batches[start : start + global_batch] for start in range(0, len(batches), global_batch)]
        total_steps = len(groups) * args.epochs
        if args.max_steps:
            total_steps = min(total_steps, args.max_steps)
        strategy = FSDPStrategy(
            FSDPConfig(fsdp_size=args.fsdp_size, reshard_after_forward=False),
            OptimizerConfig(
                lr=args.learning_rate,
                weight_decay=0.01,
                scheduler="linear",
                num_warmup_steps=math.ceil(total_steps * args.warmup_ratio),
                offload_after_step=False,
            ),
            seed=args.seed,
            num_training_steps=total_steps,
        )
        strategy.setup_distributed()
        model, optimizer, scheduler = strategy.prepare((model, None, None))
        eval_order = torch.randperm(len(val_rows), generator=torch.Generator().manual_seed(args.seed)).tolist()
        eval_rows = [val_rows[index] for index in eval_order[: args.eval_max_samples]]
        args.output.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0 and args.report_to == "wandb":
            import wandb

            run = wandb.init(
                project="rwkv_antidoom",
                name=f"{Path(args.model).name}-antidoom-nora",
                config={
                    **vars(args),
                    "model_name": Path(args.model).name,
                    "nora_revision": NORA_REVISION,
                    "train_samples": len(train_rows),
                    "validation_samples": len(val_rows),
                    "total_steps": total_steps,
                    "trajectory_format": manifest.get("trajectory_format") if manifest_path.exists() else None,
                },
            )
            (args.output / "wandb.json").write_text(json.dumps({"id": run.id, "url": run.url}) + "\n")

        def save_and_evaluate(step, metrics):
            metrics.update(evaluate(model, eval_rows, device, args.micro_batch_size, args.max_batch_tokens))
            adapter_dir = args.output / "adapters" / f"step_{step}"
            adapter_state = layered_summon_lora_params(model)
            if dist.get_rank() == 0:
                adapter_dir.mkdir(parents=True, exist_ok=True)
                save_file(adapter_state, str(adapter_dir / "adapter_model.safetensors"))
                model.peft_config["default"].save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
            checkpoint_dir = args.output / "checkpoints" / f"step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "global_step": step,
                    "base_model": args.model,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": strategy.get_rng_state(),
                },
                checkpoint_dir / f"rank_{dist.get_rank()}.pt",
            )
            dist.barrier()

        step = 0
        model.train()
        if not args.profile:
            baseline = evaluate(model, eval_rows, device, args.micro_batch_size, args.max_batch_tokens)
            if dist.get_rank() == 0:
                print(f"step=0 {baseline}", flush=True)
                with (args.output / "metrics.jsonl").open("a") as handle:
                    write_json(handle, {"step": 0, **baseline})
                if run:
                    run.log(baseline, step=0)
        for epoch in range(args.epochs):
            order = (
                [len(groups) - 1]
                + sorted(
                    range(len(groups) - 1),
                    key=lambda index: max(len(batch) * len(batch[-1]["prompt_ids"]) for batch in groups[index]),
                    reverse=True,
                )
                if args.profile
                else torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            )
            for group_index in order:
                if step == total_steps:
                    break
                group = groups[group_index]
                count = sum(len(batch) for batch in group)
                stats = Counter()
                step_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                for micro in range(args.gradient_accumulation_steps):
                    index = micro * dist.get_world_size() + dist.get_rank()
                    valid = index < len(group)
                    rows = group[index] if valid else [train_rows[0]]
                    model.set_requires_gradient_sync(micro == args.gradient_accumulation_steps - 1)
                    loss, metrics = batch_loss(model, rows, device)
                    strategy.backward(
                        loss * (dist.get_world_size() * len(rows) / count if valid else 0), model, optimizer
                    )
                    if valid:
                        for key, value in metrics.items():
                            stats[key] += value.item() * len(rows)
                grad_norm = strategy.optimizer_step(optimizer, model, scheduler)
                values = torch.tensor([stats[key] for key in metrics], device=device, dtype=torch.float64)
                dist.all_reduce(values)
                peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.float64)
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                step += 1
                log = {f"train/{key}": value.item() / count for key, value in zip(metrics, values, strict=True)}
                log["train/lr"] = scheduler.get_last_lr()[0]
                if grad_norm is not None:
                    log["train/grad_norm"] = grad_norm.item()
                elapsed = time.monotonic() - step_started
                log.update(
                    {
                        "train/step_seconds": elapsed,
                        "train/samples": count,
                        "train/tokens_per_second": sum(len(row["prompt_ids"]) for batch in group for row in batch)
                        / elapsed,
                        "train/peak_memory_GiB": peak_memory.item() / 2**30,
                    }
                )
                if not args.profile and (step % args.save_steps == 0 or step == total_steps):
                    save_and_evaluate(step, log)
                if dist.get_rank() == 0:
                    print(f"step={step} {log}", flush=True)
                    with (args.output / "metrics.jsonl").open("a") as handle:
                        write_json(handle, {"step": step, **log})
                    if run:
                        run.log(log, step=step)
        if args.profile:
            return
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if dist.get_rank() == 0:
            model = PeftModel.from_pretrained(
                load_model(args.model), args.output / "adapters" / f"step_{step}", is_trainable=True
            )
            model.to(device)
            merge_and_save(model, tokenizer, args.model, args.output / "merged", val_rows[0]["prompt_ids"], device)
            if run:
                artifact = wandb.Artifact(f"{run.id}-ftpo", type="dataset")
                artifact.add_dir(str(args.data))
                artifact.add_file(str(args.output / "metrics.jsonl"))
                run.log_artifact(artifact)
        dist.barrier()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if run:
            run.finish()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select-benchmarks")
    select_parser.add_argument("--scoreboard-records", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--param-size", choices=("1.5b", "2.9b", "7.2b", "13.3b"), default="1.5b")
    select_parser.add_argument("--min-truncation-rate", type=float, default=0.5)
    export_parser = commands.add_parser("export-benchmarks")
    export_parser.add_argument("--benchmarks", type=Path, required=True)
    export_parser.add_argument("--lighteval-project", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--seed", type=int, default=42)
    mix_parser = commands.add_parser("export-mix")
    mix_parser.add_argument("--benchmark-data", type=Path, required=True)
    mix_parser.add_argument("--tested-benchmarks", type=Path, required=True)
    mix_parser.add_argument("--mix-revision", default=ANTIDOOM_REVISION)
    mix_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--input", type=Path, required=True, help="Held-out prompt IDs and native Doc metadata")
    score_parser.add_argument("--rollouts", type=Path, required=True)
    score_parser.add_argument("--lighteval-project", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--max-period", type=int, default=8192)
    score_parser.add_argument("--sample-interval", type=int, default=16)
    for name in ("generate", "prepare", "train"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True, help="Exact RWKV checkpoint to roll out and adapt")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=42)
        if name != "generate":
            command.add_argument("--max-length", type=int, default=32768)
    generate_parser = commands.choices["generate"]
    generate_parser.add_argument(
        "--input", type=Path, help="JSONL prompts: query, messages, conversations or prompt_ids"
    )
    generate_parser.add_argument("--dataset", help="Explicit optional HF prompt source")
    generate_parser.add_argument("--split", default="train")
    generate_parser.add_argument("--limit", type=int, default=0, help="0 uses all source prompts")
    generate_parser.add_argument("--samples-per-prompt", type=int, default=1)
    generate_parser.add_argument("--sampling", choices=("rollout", "eval"), default="rollout")
    generate_parser.add_argument("--prompt-template", default="bot")
    generate_parser.add_argument("--generation-prompt", default="open_think")
    generate_parser.add_argument("--base-url", help="Comma-separated vLLM endpoints; otherwise start local engines")
    generate_parser.add_argument("--concurrency", type=int, default=64)
    generate_parser.add_argument("--shard-rollouts", action="store_true", help="One JSONL shard per inference engine")
    generate_parser.add_argument(
        "--resume", action="store_true", help="Continue this checkpoint's own fresh rollout run"
    )
    generate_parser.add_argument("--max-tokens", type=int, default=24576)
    generate_parser.add_argument("--max-model-len", type=int, default=32768)
    generate_parser.add_argument("--max-num-seqs", type=int, default=128)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    generate_parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    generate_parser.add_argument("--port", type=int, default=19001)
    generate_parser.add_argument("--timeout", type=float, default=1800)
    generate_parser.add_argument("--startup-timeout", type=float, default=900)
    prepare_parser = commands.choices["prepare"]
    prepare_parser.add_argument("--rollouts", type=Path, required=True)
    prepare_parser.add_argument("--top-k", type=int, default=50)
    prepare_parser.add_argument("--num-chosen", type=int, default=20)
    prepare_parser.add_argument("--validation-fraction", type=float, default=0.1)
    prepare_parser.add_argument(
        "--cached-text",
        action="store_true",
        help="Retokenize cached prompt/response text and use its verified repeat_hit",
    )
    train_parser = commands.choices["train"]
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--rank", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.0)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train_parser.add_argument("--micro-batch-size", type=int, default=1)
    train_parser.add_argument("--max-batch-tokens", type=int, default=32768)
    train_parser.add_argument("--fsdp-size", type=int, default=1)
    train_parser.add_argument("--eval-max-samples", type=int, default=1024)
    train_parser.add_argument("--max-steps", type=int, default=0, help="0 trains all epochs")
    train_parser.add_argument("--profile", action="store_true", help="Measure training steps without exports/merge")
    train_parser.add_argument("--save-steps", type=int, default=50)
    train_parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    args = parser.parse_args()
    if args.command == "select-benchmarks":
        if not 0 <= args.min_truncation_rate <= 1:
            parser.error("--min-truncation-rate must be between 0 and 1")
        select_benchmarks(args)
    elif args.command == "export-benchmarks":
        export_benchmarks(args)
    elif args.command == "export-mix":
        export_mix(args)
    elif args.command == "generate":
        if not args.input and not args.dataset:
            parser.error("Provide the whole-benchmark --input export or an explicit --dataset")
        if args.resume and (args.samples_per_prompt != 1 or not args.input):
            parser.error("Rollout resume requires --input and --samples-per-prompt 1")
        if min(args.concurrency, args.samples_per_prompt, args.max_tokens, args.max_model_len) <= 0 or args.limit < 0:
            parser.error("Generation sizes must be positive; --limit must be non-negative")
        asyncio.run(generate(args))
    elif args.command == "prepare":
        if not 0 < args.validation_fraction < 1:
            parser.error("--validation-fraction must be between 0 and 1")
        if min(args.max_length, args.top_k, args.num_chosen) <= 0:
            parser.error("FTPO context and candidate sizes must be positive")
        prepare(args)
    elif args.command == "score":
        if min(args.max_period, args.sample_interval) <= 0:
            parser.error("Repetition period and sampling interval must be positive")
        score(args)
    else:
        if not 0 <= args.warmup_ratio < 1:
            parser.error("--warmup-ratio must be between 0 (inclusive) and 1 (exclusive)")
        if (
            min(
                args.max_length,
                args.epochs,
                args.gradient_accumulation_steps,
                args.save_steps,
                args.micro_batch_size,
                args.max_batch_tokens,
                args.eval_max_samples,
            )
            <= 0
            or args.max_steps < 0
        ):
            parser.error("Training sizes and intervals must be positive")
        if args.profile and not args.max_steps:
            parser.error("Set --max-steps for a profiling run")
        train(args)


if __name__ == "__main__":
    main()
