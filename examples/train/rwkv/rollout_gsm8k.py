#!/usr/bin/env python3
from __future__ import annotations
import argparse, asyncio, json, re, resource, subprocess, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
def answer_text(raw: str) -> str:
    text = raw.split("</think>", 1)[1] if "</think>" in raw else raw
    text = text.strip()
    return text[1:].lstrip() if text.startswith(">") else text
def math_score(raw: str, gold: str) -> tuple[str, str]:
    from math_verify import parse, verify
    prediction = answer_text(raw)
    try:
        expected = parse(f"$\\boxed{{{gold}}}$")
        candidates = [parse(prediction)]
        lines = [line for line in prediction.replace("＝", "=").splitlines() if "=" in line]
        if lines and "\\boxed" not in prediction and "answer" not in prediction.lower(): candidates.insert(0, parse(lines[-1]))
        ok = any(expected and candidate and verify(expected, candidate, strict=False) for candidate in candidates)
        return ("correct" if ok else "wrong" if any(candidates) else "unanswered", "math_verify")
    except Exception: return "unanswered", "parse_error"
def boxed(raw: str) -> str | None:
    start = raw.rfind("\\boxed")
    if start < 0: return None
    left = raw.find("{", start)
    if left < 0: return None
    depth = 0
    for index in range(left, len(raw)):
        depth += raw[index] == "{"
        depth -= raw[index] == "}"
        if depth == 0: return raw[left + 1:index]
    return None
def option_map(query: str) -> dict[str, str]:
    return {m.group(1).upper(): m.group(2).strip() for m in re.finditer(r"(?m)^\s*([A-Ja-j]|\d{1,2})\s*[-.):、]\s*(.+?)\s*$", query)}
def text_candidates(raw: str, query: str) -> list[str]:
    text = answer_text(raw)
    markers = r"(?:final\s+answer|correct\s+answer|answer|答案|最终答案|正确答案)\s*(?:is|是|为|[:：=])"
    matches = list(re.finditer(markers + r"\s*(.+)", text, re.I))
    value = boxed(text) or (matches[-1].group(1) if matches else text.splitlines()[-1] if text else "")
    candidates, options = [value.strip()], option_map(query)
    pattern = r"(?:^\s*[\[(]?|(?:option|choice|statement|选项)\s*(?:is\s*)?)([A-J]|\d{1,2})(?=\s*[).:：]|\s*$|\s+(?:is|would|是|为))"
    labels = list(re.finditer(pattern, value, re.I))
    if labels and (label := labels[-1].group(1).upper()) in options:
        candidates.append(options[label])
    return candidates
def normalize(value: str) -> str:
    value = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", str(value)).casefold().replace("−", "-")
    return " ".join(value.strip(" `*_\t\r\n.,;:。；：").split())
def text_score(raw: str, gold: str, query: str) -> bool:
    target = normalize(gold)
    return bool(target) and any(normalize(candidate) == target for candidate in text_candidates(raw, query))
def run_code(raw: str, gold: dict[str, Any]) -> tuple[bool, str]:
    text = answer_text(raw)
    matches = list(re.finditer(r"\x60\x60\x60\s*([\w+.-]*)\s*\n(.*?)\n\x60\x60\x60", text, re.S))
    language = matches[0].group(1).casefold() if matches else "python"
    program = "\n".join(m.group(2) for m in matches) if matches else text
    if language in {"py", "python", "python3", ""}:
        command_prefix, suffix = ["/usr/bin/python3"], ".py"
    elif language in {"cpp", "c++", "cc", "cxx"}:
        command_prefix, suffix = [], ".cpp"
    else:
        return False, "unsupported_language"
    if not program.strip(): return False, "empty_program"
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); path.chmod(0o755)
            source = path / ("solution" + suffix); source.write_text(program, encoding="utf-8"); source.chmod(0o644)
            if suffix == ".cpp":
                executable = path / "solution"
                subprocess.run(["g++", "-O2", "-std=c++17", str(source), "-o", str(executable)],
                               check=True, capture_output=True, timeout=30)
                command_prefix = [str(executable)]
            command = ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix, str(source)] if suffix == ".py" \
                else ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix]
            env = {"PATH": "/usr/bin:/bin", "HOME": directory}
            for stdin, expected in zip(gold["inputs"], gold["outputs"], strict=True):
                result = subprocess.run(command, input=str(stdin), text=True, cwd=directory,
                                        capture_output=True, timeout=10, env=env)
                if result.returncode == 124: return False, "timeout"
                if result.returncode:
                    return False, "runtime_error"
                if result.stdout.rstrip() != str(expected).rstrip():
                    return False, "wrong_output"
        return True, "ok"
    except subprocess.TimeoutExpired: return False, "timeout"
    except subprocess.CalledProcessError: return False, "compile_error"
    except (OSError, UnicodeError, KeyError): return False, "execution_error"
def score(domain: str, raw: str, gold: Any, query: str) -> tuple[str, str]:
    if not raw.strip(): return "unanswered", "empty_response"
    if domain == "Math": return math_score(raw, str(gold))
    if domain == "Code": ok, reason = run_code(raw, gold); return ("correct" if ok else "wrong" if reason == "wrong_output" else "unanswered", reason)
    return ("correct" if text_score(raw, str(gold), query) else "wrong", "answer_match")
def token_count(tokenizer, query: str) -> int:
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": query}], tokenize=True, add_generation_prompt=True, rwkv_prompt_template="bot", rwkv_generation_prompt="fake_think")
    return len(encoded["input_ids"] if "input_ids" in encoded else encoded)
def rows(root: Path, limit: int, tokenizer, max_model_len: int, max_tokens: int):
    configs = ("Math", "Code", "Long_Context", "Knowledge")
    for domain, target in ((domain, limit // 4 + (index < limit % 4)) for index, domain in enumerate(configs)):
        if not target: continue
        selected = 0
        for path in sorted((root / domain).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if domain == "Long_Context" and token_count(tokenizer, row["query"]) + max_tokens > max_model_len:
                        continue
                    yield row
                    selected += 1
                    if selected >= target:
                        break
            if selected >= target:
                break
async def one(session, url: str, model: str, row: dict[str, Any], max_tokens: int):
    body = {"model": model, "messages": [{"role": "user", "content": row["query"]}],
            "max_tokens": max_tokens, "temperature": 0.96, "top_p": 0.76, "top_k": 32,
            "presence_penalty": 1.0, "frequency_penalty": 0.1, "penalty_decay": 0.988,
            "chat_template_kwargs": {"rwkv_prompt_template": "bot", "rwkv_generation_prompt": "open_think" if row["domain"] == "Math" else "fake_think"},
            "stream": False}
    try:
        async with session.post(url, json=body) as response:
            if response.status >= 400:
                return "", f"http_{response.status}", None
            payload = await response.json()
        choice = payload["choices"][0]
        return choice["message"].get("content", ""), None, choice.get("finish_reason")
    except Exception as exc:
        return "", type(exc).__name__, None
def classify(item):
    (row, index, _), (response, error, finish) = item
    if error or finish == "length": return row, index, "unanswered", error or "length", response
    status, reason = score(row["domain"], response, row["ground_truth"], row["query"])
    return row, index, status, reason, response
async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, max(soft, args.concurrency + 128)), hard))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    selected = list(rows(args.data_root, args.limit, tokenizer, args.max_model_len, args.max_tokens))
    endpoints = [x.strip().rstrip("/") + "/v1/chat/completions" for x in args.base_url.split(",") if x.strip()]
    pending = [(row, index, endpoints[position % len(endpoints)]) for position, (row, index) in enumerate((item for item in ((row, i) for row in selected for i in range(64))))]
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=1024)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        results = await asyncio.gather(*(one(session, url, args.model, row, args.max_tokens) for row, _, url in pending))
    groups = defaultdict(dict)
    scored = await asyncio.gather(*(asyncio.to_thread(classify, item) for item in zip(pending, results, strict=True)))
    for row, index, status, reason, response in scored:
        question = groups[row["domain"]].setdefault(row["uuid"], {"row": row, "counts": Counter(), "reasons": Counter(), "samples": {}})
        question["counts"][status] += 1; question["reasons"][reason] += 1
        question["samples"].setdefault(status, {"response": response, "sample_index": index, "reason": reason})
    write_output(args.output, groups, args.max_tokens, args.limit)
def write_output(output: Path, groups, max_tokens: int, limit: int):
    output.mkdir(parents=True, exist_ok=True)
    summary, candidates = {}, {status: defaultdict(list) for status in ("correct", "wrong", "unanswered")}
    for domain, questions in groups.items():
        correct, wrong, unanswered = (sum(q["counts"][status] for q in questions.values())
                                      for status in ("correct", "wrong", "unanswered"))
        summary[domain] = {"num_questions": len(questions), "num_rollouts": correct + wrong + unanswered,
                           "correct_rollouts": correct, "wrong_rollouts": wrong, "unanswered_rollouts": unanswered,
                           "mean_pass_rate": correct / (correct + wrong + unanswered) if correct + wrong + unanswered else 0.0,
                           "correct_count_histogram": dict(Counter(q["counts"]["correct"] for q in questions.values())),
                           "correct_counts_by_question": {uuid: q["counts"]["correct"] for uuid, q in questions.items()},
                           "reasons": dict(sum((q.get("reasons", Counter()) for q in questions.values()), Counter()))}
        for uuid, question in questions.items():
            for status, sample in question["samples"].items():
                candidates[status][domain].append({"domain": domain, "uuid": uuid, "query": question["row"]["query"], "ground_truth": question["row"]["ground_truth"], "response": sample["response"],
                    "status": status, "sample_index": sample["sample_index"], "reason": sample["reason"]})
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    fig = Figure(figsize=(13, 9), layout="constrained")
    for ax, (domain, values) in zip(fig.subplots(len(summary), 1, squeeze=False).flat, summary.items()):
        hist = values["correct_count_histogram"]
        ax.bar(range(65), [hist.get(count, 0) for count in range(65)])
        ax.set(title=domain, xlabel="Correct rollouts per question (out of 64)", ylabel="Questions", xticks=range(0, 65, 8))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(output / "pass_rate_histogram.svg")
    summary["config"] = {"limit": limit, "rollouts_per_question": 64, "max_tokens": max_tokens, "temperature": 0.96, "top_p": 0.76, "top_k": 32, "presence_penalty": 1.0, "frequency_penalty": 0.1, "penalty_decay": 0.988}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    used = set()
    for status, by_domain in candidates.items():
        values = []
        for index in range(max((len(v) for v in by_domain.values()), default=0)):
            for domain in sorted(by_domain):
                if index < len(by_domain[domain]) and by_domain[domain][index]["uuid"] not in used and len(values) < 20:
                    values.append(by_domain[domain][index]); used.add(values[-1]["uuid"])
        (output / f"{status}_samples.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in values) + ("\n" if values else ""), encoding="utf-8")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ultradata_rwkv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:19001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=1800)
    asyncio.run(run(parser.parse_args()))
if __name__ == "__main__":
    main()
