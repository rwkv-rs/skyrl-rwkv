#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import resource
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from skyrl_gym.envs.gsm8k.utils import compute_strict_score
from skyrl_gym.envs.ultradata.env import DOMAINS, answer_text, run_code, text_score

STATUSES = ("correct", "wrong", "unanswered")
ROLLOUTS_PER_QUESTION = 64
SCORING_VERSION = 2
ROLLOUT_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_decay": 1.0,
}
SAMPLING_PARAMS = {
    "temperature": 0.96,
    "top_p": 0.76,
    "top_k": 32,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.1,
    "penalty_decay": 0.988,
}


def rows(
    root: Path,
    limit: int,
    tokenizer,
    max_model_len: int,
    max_tokens: int,
    domain: str | None = None,
    completed=frozenset(),
):
    """Stream every shard; keep golden test cases on disk until grading."""
    for index, name in enumerate(DOMAINS):
        if domain is not None and domain != name:
            continue
        target = limit // len(DOMAINS) + (index < limit % len(DOMAINS)) if limit else None
        if target == 0:
            continue
        selected = 0
        for path in sorted((root / name).glob("*.jsonl")):
            with path.open("rb") as handle:
                offset = 0
                for line in handle:
                    row = orjson.loads(line)
                    source = (str(path), offset)
                    offset += len(line)
                    selected += 1
                    if row["uuid"] not in completed:
                        row.pop("ground_truth")
                        row["_source"] = source
                        encoded = tokenizer.apply_chat_template(
                            [{"role": "user", "content": row["query"]}],
                            tokenize=True,
                            add_generation_prompt=True,
                            rwkv_prompt_template="bot",
                            rwkv_generation_prompt="open_think" if name == "Math" else "fake_think",
                        )
                        row["_prompt_ids"] = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
                        row["prompt_tokens"] = len(row["_prompt_ids"])
                        row["_prompt_too_long"] = row["prompt_tokens"] + max_tokens > max_model_len
                        yield row
                    if target is not None and selected >= target:
                        break
            if target is not None and selected >= target:
                break


def trajectory_path(output: Path, row) -> Path:
    name = hashlib.sha256(row["uuid"].encode()).hexdigest()
    return output / "rollouts" / row["domain"] / f"{name}.jsonl"


def start_trajectory(output: Path, row, prompt_ids) -> Path:
    """Stage one question on disk so full response IDs do not accumulate in RAM."""
    path = trajectory_path(output, row).with_suffix(".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps({"prompt_ids": prompt_ids}, option=orjson.OPT_APPEND_NEWLINE))
    return path


def append_trajectory(path: Path, sample_index, response_ids, finish_reason):
    with path.open("ab") as handle:
        handle.write(
            orjson.dumps(
                {"sample_index": sample_index, "response_ids": response_ids, "finish_reason": finish_reason},
                option=orjson.OPT_APPEND_NEWLINE,
            )
        )


def publish_trajectories(output: Path, row, verdicts):
    """Publish a scored question atomically before marking it completed."""
    path = trajectory_path(output, row)
    pending = path.with_suffix(".pending")
    temporary = path.with_suffix(".tmp")
    with pending.open("rb") as source, temporary.open("wb") as target:
        prompt_ids = orjson.loads(source.readline())["prompt_ids"]
        for line in source:
            record = orjson.loads(line)
            verdict = verdicts[record["sample_index"]]
            target.write(
                orjson.dumps(
                    {
                        **record,
                        "uuid": row["uuid"],
                        "domain": row["domain"],
                        "prompt_ids": prompt_ids,
                        "reward": float(verdict["status"] == "correct"),
                        "truncated": record["finish_reason"] in {"length", "max_tokens"},
                        **verdict,
                    },
                    option=orjson.OPT_APPEND_NEWLINE,
                )
            )
    temporary.replace(path)
    pending.unlink()


def score_question(row, choices, eos_token_id, collect_samples=True, collect_verdicts=False):
    path, offset = row["_source"]
    with open(path, "rb") as handle:
        handle.seek(offset)
        original = orjson.loads(handle.readline())
    counts, reasons, samples, code_cache = Counter(), Counter(), {}, {}
    verdicts = []
    for index, choice in enumerate(choices):
        response = choice["text"]
        finish, response_ids = choice.get("finish_reason"), choice.get("token_ids") or []
        if row["_prompt_too_long"]:
            status, reason = "unanswered", "prompt_too_long"
        elif row["domain"] == "Math":
            truncated = finish in {"length", "max_tokens"}
            reward, details = compute_strict_score(
                response,
                original["ground_truth"],
                ended_eod=bool(not truncated and response_ids and response_ids[-1] == eos_token_id),
                truncated=truncated,
            )
            if details["truncated"]:
                status, reason = "unanswered", finish
            elif not response.strip():
                status, reason = "unanswered", "empty_response"
            elif not details["ended_eod"]:
                status, reason = "unanswered", "missing_eos"
            elif not details["structural_format_valid"]:
                status, reason = "unanswered", "invalid_format"
            elif not details["answer_parseable"]:
                status, reason = "unanswered", "parse_error"
            else:
                status, reason = ("correct" if reward else "wrong"), "math_verify"
        elif finish == "length":
            status, reason = "unanswered", "length"
        elif not response.strip():
            status, reason = "unanswered", "empty_response"
        elif row["domain"] == "Code":
            # Repeated final programs share a verdict, but all samples still count.
            key = answer_text(response)
            if key not in code_cache:
                code_cache[key] = run_code(response, original["ground_truth"])
            ok, reason = code_cache[key]
            status = "correct" if ok else "wrong" if reason == "wrong_output" else "unanswered"
        else:
            status, reason = (
                "correct" if text_score(response, str(original["ground_truth"]), row["query"]) else "wrong"
            ), "answer_match"
        counts[status] += 1
        reasons[reason] += 1
        if collect_verdicts:
            verdicts.append({"status": status, "reason": reason})
        if collect_samples:
            samples.setdefault(
                status,
                {
                    "domain": row["domain"],
                    "uuid": row["uuid"],
                    "query": row["query"],
                    "ground_truth": original["ground_truth"],
                    "response": response,
                    "status": status,
                    "sample_index": index,
                    "reason": reason,
                },
            )
    if collect_verdicts:
        return counts, reasons, samples, verdicts
    return counts, reasons, samples


def write_output(output: Path, groups, candidates, config, *, plot: bool = False):
    summary = {}
    for domain, question in groups.items():
        total = sum(question["counts"].values())
        summary[domain] = {
            "num_questions": question["num_questions"],
            "num_rollouts": total,
            **{f"{status}_rollouts": question["counts"][status] for status in STATUSES},
            "mean_pass_rate": question["counts"]["correct"] / total if total else 0.0,
            "correct_count_histogram": dict(question["histogram"]),
            "correct_counts_by_question_file": "correct_counts_by_question.jsonl",
            "reasons": dict(question["reasons"]),
        }
    summary["config"] = config
    temporary = output / "summary.json.tmp"
    temporary.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    temporary.replace(output / "summary.json")
    if not plot:
        return

    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(13, 9), layout="constrained")
    for ax, (domain, values) in zip(fig.subplots(len(groups), 1, squeeze=False).flat, groups.items()):
        hist = values["histogram"]
        ax.bar(range(65), [hist.get(count, 0) for count in range(65)])
        ax.set(
            title=domain,
            xlabel="Correct rollouts per question (out of 64)",
            ylabel="Questions",
            xticks=range(0, 65, 8),
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(output / "pass_rate_histogram.svg")
    used = set()
    for status, by_domain in candidates.items():
        values = []
        for index in range(max((len(v) for v in by_domain.values()), default=0)):
            for domain in sorted(by_domain):
                if index < len(by_domain[domain]) and by_domain[domain][index]["uuid"] not in used and len(values) < 20:
                    values.append(by_domain[domain][index])
                    used.add(values[-1]["uuid"])
        with (output / f"{status}_samples.jsonl").open("wb") as handle:
            for value in values:
                handle.write(orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE))


async def start_servers(
    args, processes, *, enforce_eager=False, gpu_memory_utilization=0.8, max_num_batched_tokens=8192
):
    import aiohttp
    import torch

    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("No visible CUDA GPUs; provide --base-url for external engines")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",")[:count] if visible is not None else [str(index) for index in range(count)]
    cache_root = Path.home() / ".cache/skyrl-rwkv"
    env = {
        **os.environ,
        "XDG_CACHE_HOME": str(cache_root),
        "TORCH_EXTENSIONS_DIR": str(cache_root / "torch_extensions"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # RWKV has no fixed positional embeddings; prefill remains chunked.
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    }
    with (args.output / "flashrwkv2_preflight.log").open("wb") as log:
        preflight = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("prepare_flashrwkv2.py")),
            "--check",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    processes.append(preflight)
    if await preflight.wait():
        raise RuntimeError(f"FlashRWKV2 cache preflight failed; inspect {args.output / 'flashrwkv2_preflight.log'}")
    processes.remove(preflight)
    urls = []
    for index, device in enumerate(devices):
        port = args.port + index
        log_path = args.output / f"vllm_gpu_{index}.log"
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "float16",
            "--generation-config",
            "vllm",
            "--skip-tokenizer-init",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--mamba-ssm-cache-dtype",
            "float32",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--disable-uvicorn-access-log",
        ]
        if enforce_eager:
            command.append("--enforce-eager")
        with log_path.open("wb") as log:
            processes.append(
                await asyncio.create_subprocess_exec(
                    *command,
                    env={**env, "CUDA_VISIBLE_DEVICES": device},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        urls.append(f"http://127.0.0.1:{port}")
        print(f"Starting GPU {device}: {urls[-1]} (pid={processes[-1].pid}, log={log_path})", flush=True)
    (args.output / "vllm.pids").write_text("\n".join(str(process.pid) for process in processes) + "\n")
    deadline = time.monotonic() + args.startup_timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        for url, process in zip(urls, processes, strict=True):
            while True:
                if process.returncode is not None:
                    raise RuntimeError(f"vLLM exited with {process.returncode}: {url}; inspect engine logs")
                try:
                    async with session.get(f"{url}/health") as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                except TimeoutError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"vLLM startup timed out: {url}")
                await asyncio.sleep(1)
    return urls


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer

    if args.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be positive")
    if args.resume and args.output is None:
        raise ValueError("--resume requires --output")
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("outputs") / f"ultradata_{Path(args.model).name}_{stamp}"
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (
        (args.output / "summary.json").exists() or (args.output / "correct_counts_by_question.jsonl").exists()
    ):
        raise FileExistsError(f"Results already exist: {args.output}; use a new --output directory")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    groups = {
        domain: {"num_questions": 0, "counts": Counter(), "reasons": Counter(), "histogram": Counter()}
        for domain in DOMAINS
    }
    candidates = {status: defaultdict(list) for status in STATUSES}
    sample_ids = set()
    progress = Counter()
    completed = set()
    elapsed_before_resume = 0.0
    processes, pool = [], None
    sampling_params = ROLLOUT_SAMPLING_PARAMS if args.sampling == "rollout" else SAMPLING_PARAMS
    config = {
        "model": args.model,
        "save_rollouts": args.save_rollouts,
        "limit": args.limit,
        "rollouts_per_question": args.samples_per_prompt,
        "scoring_version": SCORING_VERSION,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        **sampling_params,
        "status": "starting",
        "progress": progress,
    }
    if args.resume:
        saved = orjson.loads((args.output / "summary.json").read_bytes())["config"]
        for key in (
            "scoring_version",
            "model",
            "limit",
            "rollouts_per_question",
            "max_tokens",
            "max_model_len",
            *SAMPLING_PARAMS,
        ):
            if saved.get(key) != config[key]:
                raise ValueError(f"Cannot resume with changed {key}: {saved.get(key)!r} -> {config[key]!r}")
        if saved.get("save_rollouts", False) != args.save_rollouts:
            raise ValueError("Cannot resume with changed save_rollouts")
        with (args.output / "correct_counts_by_question.jsonl").open("rb") as handle:
            for line in handle:
                record = orjson.loads(line)
                completed.add(record["uuid"])
                question = groups[record["domain"]]
                question["num_questions"] += 1
                question["counts"].update(record["counts"])
                question["reasons"].update(record["reasons"])
                question["histogram"][record["counts"].get("correct", 0)] += 1
        for status in STATUSES:
            with (args.output / f"{status}_samples.jsonl").open("rb") as handle:
                for line in handle:
                    sample = orjson.loads(line)
                    candidates[status][sample["domain"]].append(sample)
                    sample_ids.add(sample["uuid"])
        progress.update(saved["progress"])
        progress["scored_questions"] = len(completed)
        elapsed_before_resume = saved["elapsed_seconds"]
        config["resumed_questions"] = len(completed)
    started = time.monotonic() - elapsed_before_resume
    (args.output / "rollout.pid").write_text(f"{os.getpid()}\n")
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        urls = (
            [url.strip().rstrip("/") for url in args.base_url.split(",") if url.strip()]
            if args.base_url
            else await start_servers(args, processes)
        )
        concurrency = args.concurrency or len(urls) * args.max_num_seqs * 2
        if not urls or concurrency < len(urls) or args.limit < 0:
            raise ValueError("Provide engines, concurrency >= engine count, and limit >= 0 (0 means all)")
        workers = [concurrency // len(urls) + (index < concurrency % len(urls)) for index in range(len(urls))]
        score_workers = args.score_workers or max(1, len(os.sched_getaffinity(0)) // 2 - 2 * len(urls))
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        tokenizer.model_max_length = args.max_model_len
        if processes:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": r"Compute 2 + 2. Put your final answer in \boxed{...}."}],
                tokenize=True,
                add_generation_prompt=True,
                rwkv_prompt_template="bot",
                rwkv_generation_prompt="open_think",
            )
            prompt_ids = prompt["input_ids"] if isinstance(prompt, Mapping) else prompt
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:

                async def validate(url):
                    async with session.post(
                        f"{url}/v1/completions",
                        json={
                            "model": args.model,
                            "prompt": prompt_ids,
                            "max_tokens": 512,
                            **SAMPLING_PARAMS,
                            "temperature": 0.0,
                            "return_token_ids": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        choice = (await response.json())["choices"][0]
                    ids = choice.get("token_ids") or []
                    decoded = tokenizer.decode(ids, skip_special_tokens=True)
                    reward, details = compute_strict_score(
                        decoded,
                        "4",
                        ended_eod=bool(ids and ids[-1] == tokenizer.eos_token_id),
                        truncated=choice.get("finish_reason") in {"length", "max_tokens"},
                    )
                    # Incorrect or malformed answers are valid policy samples.
                    # Check the token/EOS contract independently of answer quality.
                    if not ids or (choice.get("finish_reason") == "stop" and not details["ended_eod"]):
                        raise RuntimeError(f"Engine acceptance failed: {url}: {details}")
                    print(
                        f"Engine acceptance passed: {url}, tokens={len(ids)}, "
                        f"probe_reward={reward}, response={decoded!r}",
                        flush=True,
                    )

                await asyncio.gather(*(validate(url) for url in urls))
            config["engine_validation"] = "passed"
        config.update(
            endpoints=urls, concurrency=concurrency, max_num_seqs=args.max_num_seqs, score_workers=score_workers
        )
        print(json.dumps(config), flush=True)
        data_queue = asyncio.Queue(maxsize=2 * len(urls))
        queues = [asyncio.Queue(maxsize=2 * count) for count in workers]
        score_queue = asyncio.Queue(maxsize=2 * score_workers)
        active = [0] * len(urls)
        finished = asyncio.Event()
        pool = ProcessPoolExecutor(max_workers=score_workers, mp_context=multiprocessing.get_context("spawn"))

        async def produce(domain):
            source = rows(args.data_root, args.limit, tokenizer, args.max_model_len, args.max_tokens, domain, completed)
            while (row := await asyncio.to_thread(next, source, None)) is not None:
                await data_queue.put(row)
                progress["read_questions"] += 1

        async def dispatch():
            position = 0
            while (row := await data_queue.get()) is not None:
                prompt_ids = row.pop("_prompt_ids")
                if row["_prompt_too_long"]:
                    await score_queue.put((row, [{"text": ""}] * args.samples_per_prompt))
                    progress["oversized_questions"] += 1
                    continue
                body = orjson.dumps(
                    {
                        "model": args.model,
                        "prompt": prompt_ids,
                        "max_tokens": args.max_tokens,
                        **sampling_params,
                        "stream": False,
                        "return_token_ids": True,
                    }
                )
                question = {
                    "row": row,
                    "choices": [None] * args.samples_per_prompt,
                    "remaining": args.samples_per_prompt,
                    "body": body,
                }
                if args.save_rollouts:
                    question["trajectory_path"] = await asyncio.to_thread(
                        start_trajectory, args.output, row, prompt_ids
                    )
                    question["trajectory_lock"] = asyncio.Lock()
                # Keep the 64 copies together for prefix reuse; choose the least loaded engine.
                index = min(range(len(urls)), key=lambda i: (queues[i].qsize() + active[i], (i - position) % len(urls)))
                position = (index + 1) % len(urls)
                for sample_index in range(args.samples_per_prompt):
                    await queues[index].put((question, sample_index))
            for queue, count in zip(queues, workers, strict=True):
                for _ in range(count):
                    await queue.put(None)

        connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=max(workers), keepalive_timeout=2)
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as session:

            async def generate(index):
                while (item := await queues[index].get()) is not None:
                    question, sample_index = item
                    active[index] += 1
                    async with session.post(
                        f"{urls[index]}/v1/completions",
                        data=question["body"],
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        raw = await response.read()
                        if response.status >= 400:
                            raise RuntimeError(f"{urls[index]}: HTTP {response.status}: {raw[:500]!r}")
                        choice = orjson.loads(raw)["choices"][0]
                        del raw
                    active[index] -= 1
                    response_ids = choice.get("token_ids") or []
                    progress["generated_rollouts"] += 1
                    progress["generated_tokens"] += len(response_ids)
                    choice["text"] = await asyncio.to_thread(tokenizer.decode, response_ids, skip_special_tokens=True)
                    if args.save_rollouts:
                        async with question["trajectory_lock"]:
                            await asyncio.to_thread(
                                append_trajectory,
                                question["trajectory_path"],
                                sample_index,
                                response_ids,
                                choice.get("finish_reason"),
                            )
                    # Only the terminal token is needed in RAM for strict EOS validation.
                    choice["token_ids"] = response_ids[-1:]
                    del response_ids
                    question["choices"][sample_index] = choice
                    question["remaining"] -= 1
                    if not question["remaining"]:
                        await score_queue.put((question["row"], question["choices"]))

            with (args.output / "correct_counts_by_question.jsonl").open("ab" if args.resume else "xb") as counts_file:

                async def grade():
                    while (item := await score_queue.get()) is not None:
                        row, choices = item
                        collect_samples = row["uuid"] not in sample_ids and any(
                            len(candidates[status][row["domain"]]) < 20 for status in STATUSES
                        )
                        result = await loop.run_in_executor(
                            pool,
                            score_question,
                            row,
                            choices,
                            tokenizer.eos_token_id,
                            collect_samples,
                            args.save_rollouts,
                        )
                        counts, reasons, samples = result[:3]
                        if args.save_rollouts and not row["_prompt_too_long"]:
                            await asyncio.to_thread(publish_trajectories, args.output, row, result[3])
                        question = groups[row["domain"]]
                        question["num_questions"] += 1
                        question["counts"].update(counts)
                        question["reasons"].update(reasons)
                        question["histogram"][counts["correct"]] += 1
                        counts_file.write(
                            orjson.dumps(
                                {
                                    "uuid": row["uuid"],
                                    "domain": row["domain"],
                                    "prompt_tokens": row["prompt_tokens"],
                                    "counts": counts,
                                    "reasons": reasons,
                                },
                                option=orjson.OPT_APPEND_NEWLINE,
                            )
                        )
                        counts_file.flush()
                        progress["scored_questions"] += 1
                        for status in STATUSES:
                            if (
                                status in samples
                                and row["uuid"] not in sample_ids
                                and len(candidates[status][row["domain"]]) < 20
                            ):
                                candidates[status][row["domain"]].append(samples[status])
                                sample_ids.add(row["uuid"])

                async def monitor():
                    while not finished.is_set():
                        for process in processes:
                            if process.returncode is not None:
                                raise RuntimeError(f"vLLM pid {process.pid} exited with {process.returncode}")
                        config.update(status="running", elapsed_seconds=round(time.monotonic() - started, 1))
                        config["queues"] = {
                            "data": data_queue.qsize(),
                            "generation": [queue.qsize() for queue in queues],
                            "inflight": list(active),
                            "grading": score_queue.qsize(),
                        }
                        write_output(args.output, groups, candidates, config)
                        print(
                            json.dumps(
                                {
                                    "progress": progress,
                                    "queues": config["queues"],
                                    "elapsed_seconds": config["elapsed_seconds"],
                                }
                            ),
                            flush=True,
                        )
                        try:
                            await asyncio.wait_for(finished.wait(), timeout=30)
                        except TimeoutError:
                            pass

                async with asyncio.TaskGroup() as tasks:
                    producers = [tasks.create_task(produce(domain)) for domain in DOMAINS]
                    dispatcher = tasks.create_task(dispatch())
                    generators = [
                        tasks.create_task(generate(index)) for index, count in enumerate(workers) for _ in range(count)
                    ]
                    graders = [tasks.create_task(grade()) for _ in range(score_workers)]
                    tasks.create_task(monitor())
                    await asyncio.gather(*producers)
                    await data_queue.put(None)
                    await dispatcher
                    await asyncio.gather(*generators)
                    for _ in graders:
                        await score_queue.put(None)
                    await asyncio.gather(*graders)
                    finished.set()
        config["queues"] = {
            "data": data_queue.qsize(),
            "generation": [queue.qsize() for queue in queues],
            "inflight": list(active),
            "grading": score_queue.qsize(),
        }
        config["status"] = "completed"
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        for process in processes:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if config["status"] != "completed":
            config["status"] = "interrupted"
        config["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_output(args.output, groups, candidates, config, plot=True)


parser = argparse.ArgumentParser(description="Full UltraData evaluation on every visible GPU")
parser.add_argument("--data-root", type=Path, default=Path.home() / "data/UltraData-RL-2609/files")
parser.add_argument("--output", type=Path, help="Output directory (default: timestamped model-specific directory)")
parser.add_argument("--resume", action="store_true", help="Resume completed question counts from --output")
parser.add_argument(
    "--save-rollouts", action="store_true", help="Persist all scored token trajectories for value pretraining"
)
parser.add_argument("--samples-per-prompt", type=int, default=ROLLOUTS_PER_QUESTION)
parser.add_argument("--sampling", choices=("eval", "rollout"), default="eval", help="Sampling recipe (default: eval)")
parser.add_argument(
    "--base-url", help="Comma-separated external engines; otherwise launch one TP1 engine per visible GPU"
)
parser.add_argument("--model", default=str(Path.home() / "Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600"))
parser.add_argument("--port", type=int, default=19001)
parser.add_argument("--max-num-seqs", type=int, default=1024)
parser.add_argument("--concurrency", type=int, help="Total in-flight generations (default: 2 * engines * max-num-seqs)")
parser.add_argument(
    "--score-workers", type=int, help="Grading processes (default: physical-core budget minus engine CPUs)"
)
parser.add_argument(
    "--limit", type=int, default=0, help="Question limit balanced across domains; 0 runs every row in every shard"
)
parser.add_argument("--max-model-len", type=int, default=1048576)
parser.add_argument("--max-tokens", type=int, default=8192)
parser.add_argument("--timeout", type=float, default=1800)
parser.add_argument("--startup-timeout", type=float, default=900)

if __name__ == "__main__":
    asyncio.run(run(parser.parse_args()))
