#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from skyrl_gym.envs.gsm8k.utils import compute_strict_score

DOMAINS = ("Math", "Code", "Long_Context", "Knowledge")
STATUSES = ("correct", "wrong", "unanswered")
ROLLOUTS_PER_QUESTION = 64
SAMPLING_PARAMS = {
    "temperature": 0.96,
    "top_p": 0.76,
    "top_k": 32,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.1,
    "penalty_decay": 0.988,
}


def answer_text(raw: str) -> str:
    text = raw.split("</think>", 1)[1] if "</think>" in raw else raw
    text = text.strip()
    return text[1:].lstrip() if text.startswith(">") else text


def text_score(raw: str, gold: str, query: str) -> bool:
    text = answer_text(raw)
    value = None
    start = text.rfind("\\boxed")
    if start >= 0:
        left = text.find("{", start)
        if left >= 0:
            depth = 0
            for index in range(left, len(text)):
                depth += text[index] == "{"
                depth -= text[index] == "}"
                if depth == 0:
                    value = text[left + 1 : index]
                    break
    if value is None:
        markers = r"(?:final\s+answer|correct\s+answer|answer|答案|最终答案|正确答案)\s*(?:is|是|为|[:：=])"
        matches = list(re.finditer(markers + r"\s*(.+)", text, re.IGNORECASE))
        value = matches[-1].group(1) if matches else text.splitlines()[-1] if text else ""

    candidates = [value.strip()]
    options = {
        match.group(1).upper(): match.group(2).strip()
        for match in re.finditer(r"(?m)^\s*([A-Ja-j]|\d{1,2})\s*[-.):、]\s*(.+?)\s*$", query)
    }
    labels = list(
        re.finditer(
            r"(?:^\s*[\[(]?|(?:option|choice|statement|选项)\s*(?:is\s*)?)([A-J]|\d{1,2})(?=\s*[).:：]|\s*$|\s+(?:is|would|是|为))",
            value,
            re.IGNORECASE,
        )
    )
    if labels and (label := labels[-1].group(1).upper()) in options:
        candidates.append(options[label])

    target = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", str(gold)).casefold().replace("−", "-")
    target = " ".join(target.strip(" `*_\t\r\n.,;:。；：").split())
    return bool(target) and any(
        target
        == " ".join(
            re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", str(candidate))
            .casefold()
            .replace("−", "-")
            .strip(" `*_\t\r\n.,;:。；：")
            .split()
        )
        for candidate in candidates
    )


def run_code(raw: str, gold: dict[str, Any]) -> tuple[bool, str]:
    text = answer_text(raw)
    matches = list(re.finditer(r"\x60\x60\x60\s*([\w+.-]*)\s*\n(.*?)\n\x60\x60\x60", text, re.DOTALL))
    language = matches[0].group(1).casefold() if matches else "python"
    program = "\n".join(match.group(2) for match in matches) if matches else text
    if language in {"py", "python", "python3", ""}:
        command_prefix, suffix = ["/usr/bin/python3"], ".py"
    elif language in {"cpp", "c++", "cc", "cxx"}:
        command_prefix, suffix = [], ".cpp"
    else:
        return False, "unsupported_language"
    if not program.strip():
        return False, "empty_program"
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.chmod(0o755)
            source = path / ("solution" + suffix)
            source.write_text(program, encoding="utf-8")
            source.chmod(0o644)
            if suffix == ".cpp":
                executable = path / "solution"
                subprocess.run(
                    ["g++", "-O2", "-std=c++17", str(source), "-o", str(executable)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                command_prefix = [str(executable)]
            command = (
                ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix, str(source)]
                if suffix == ".py"
                else ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix]
            )
            env = {"PATH": "/usr/bin:/bin", "HOME": directory}
            for stdin, expected in zip(gold["inputs"], gold["outputs"], strict=True):
                result = subprocess.run(
                    command,
                    input=str(stdin),
                    text=True,
                    cwd=directory,
                    capture_output=True,
                    timeout=10,
                    env=env,
                    check=False,
                )
                if result.returncode == 124:
                    return False, "timeout"
                if result.returncode:
                    return False, "runtime_error"
                if result.stdout.rstrip() != str(expected).rstrip():
                    return False, "wrong_output"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError:
        return False, "compile_error"
    except (OSError, UnicodeError, KeyError):
        return False, "execution_error"


def rows(root: Path, limit: int, tokenizer, max_model_len: int, max_tokens: int, domain: str | None = None):
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
                    row.pop("ground_truth")
                    row["_source"] = (str(path), offset)
                    offset += len(line)
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
                    selected += 1
                    if target is not None and selected >= target:
                        break
            if target is not None and selected >= target:
                break


def score_question(row, choices, eos_token_id, collect_samples=True):
    path, offset = row["_source"]
    with open(path, "rb") as handle:
        handle.seek(offset)
        original = orjson.loads(handle.readline())
    counts, reasons, samples, code_cache = Counter(), Counter(), {}, {}
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


async def start_servers(args, processes):
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
            "8192",
            "--gpu-memory-utilization",
            "0.8",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--disable-uvicorn-access-log",
        ]
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

    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("outputs") / f"ultradata_{Path(args.model).name}_{stamp}"
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists() or (args.output / "correct_counts_by_question.jsonl").exists():
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
    processes, pool = [], None
    config = {
        "model": args.model,
        "limit": args.limit,
        "rollouts_per_question": ROLLOUTS_PER_QUESTION,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        **SAMPLING_PARAMS,
        "status": "starting",
        "progress": progress,
    }
    started = time.monotonic()
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
            source = rows(args.data_root, args.limit, tokenizer, args.max_model_len, args.max_tokens, domain)
            while (row := await asyncio.to_thread(next, source, None)) is not None:
                await data_queue.put(row)
                progress["read_questions"] += 1

        async def dispatch():
            position = 0
            while (row := await data_queue.get()) is not None:
                prompt_ids = row.pop("_prompt_ids")
                if row["_prompt_too_long"]:
                    await score_queue.put((row, [{"text": ""}] * ROLLOUTS_PER_QUESTION))
                    progress["oversized_questions"] += 1
                    continue
                body = orjson.dumps(
                    {
                        "model": args.model,
                        "prompt": prompt_ids,
                        "max_tokens": args.max_tokens,
                        **SAMPLING_PARAMS,
                        "stream": False,
                        "detokenize": False,
                        "return_token_ids": True,
                    }
                )
                question = {
                    "row": row,
                    "choices": [None] * ROLLOUTS_PER_QUESTION,
                    "remaining": ROLLOUTS_PER_QUESTION,
                    "body": body,
                }
                # Keep the 64 copies together for prefix reuse; choose the least loaded engine.
                index = min(range(len(urls)), key=lambda i: (queues[i].qsize() + active[i], (i - position) % len(urls)))
                position = (index + 1) % len(urls)
                for sample_index in range(ROLLOUTS_PER_QUESTION):
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
                    # Only the terminal token is needed for strict EOS validation.
                    choice["token_ids"] = response_ids[-1:]
                    del response_ids
                    question["choices"][sample_index] = choice
                    question["remaining"] -= 1
                    if not question["remaining"]:
                        await score_queue.put((question["row"], question["choices"]))

            with (args.output / "correct_counts_by_question.jsonl").open("xb") as counts_file:

                async def grade():
                    while (item := await score_queue.get()) is not None:
                        row, choices = item
                        collect_samples = row["uuid"] not in sample_ids and any(
                            len(candidates[status][row["domain"]]) < 20 for status in STATUSES
                        )
                        counts, reasons, samples = await loop.run_in_executor(
                            pool, score_question, row, choices, tokenizer.eos_token_id, collect_samples
                        )
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
