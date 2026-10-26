"""Download and normalize the public pre-calibration UltraData sources.

The released UltraData-RL-2609 corpus has already passed the MiniCPM
difficulty screen.  This module reconstructs the verifier-compatible source
corpus from the public upstream datasets so RWKV can run its own screen.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DOMAINS = ("Math", "Code", "Long_Context", "Knowledge")
QASPER_TRAIN_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    config: str | None
    split: str
    domain: str
    adapter: str


SOURCE_SPECS = (
    SourceSpec("dapo_math_17k", "BytedTsinghua-SIA/DAPO-Math-17k", None, "train", "Math", "dapo"),
    SourceSpec(
        "deepscaler_preview",
        "agentica-org/DeepScaleR-Preview-Dataset",
        None,
        "train",
        "Math",
        "deepscaler",
    ),
    SourceSpec("deepmath_103k", "zwhe99/DeepMath-103K", None, "train", "Math", "deepmath"),
    SourceSpec("openscience_reasoning_2", "nvidia/OpenScienceReasoning-2", None, "train", "Knowledge", "science"),
    SourceSpec("hotpotqa", "hotpotqa/hotpot_qa", "distractor", "train", "Long_Context", "hotpot"),
    SourceSpec("qasper", "allenai/qasper", None, "train", "Long_Context", "qasper"),
    SourceSpec("hardtests", "sigcp/hardtests_problems", None, "train", "Code", "hardtests"),
)

# These releases contain model solutions but do not expose executable test
# cases.  They cannot be passed to the native code verifier without inventing
# a reward contract, so they are recorded in the manifest and skipped.
UNVERIFIABLE_CODE_SOURCES = (
    {"name": "opencode_reasoning", "repo": "nvidia/OpenCodeReasoning", "reason": "no public test cases"},
    {"name": "opencode_reasoning_2", "repo": "nvidia/OpenCodeReasoning-2", "reason": "no public test cases"},
)
UNAVAILABLE_SOURCES = (
    {
        "name": "ultradata_long_context_synthetic",
        "reason": "in-house synthetic data is not published as a public upstream dataset",
    },
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _code_text(value: Any) -> str:
    return str(value) if value is not None else ""


def _sequence_rows(value: Any) -> list[Mapping[str, Any]]:
    """Convert Datasets row-wise or viewer columnar sequences to rows."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    columns = {key: column for key, column in value.items() if isinstance(column, list)}
    length = max((len(column) for column in columns.values()), default=0)
    return [
        {key: column[index] if index < len(column) else None for key, column in columns.items()}
        for index in range(length)
    ]


def _prompt_content(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, Mapping) and message.get("content"):
                return _text(message["content"])
    return _text(prompt)


def _hotpot_context(row: Mapping[str, Any]) -> str:
    context = row.get("context") or {}
    titles = context.get("title", []) if isinstance(context, Mapping) else []
    sentences = context.get("sentences", []) if isinstance(context, Mapping) else []
    sections = []
    for title, paragraphs in zip(titles, sentences, strict=False):
        sections.append(f"## {_text(title)}\n" + "\n".join(_text(paragraph) for paragraph in paragraphs))
    return "\n\n".join(sections)


def _qasper_context(row: Mapping[str, Any]) -> str:
    sections = []
    for section in _sequence_rows(row.get("full_text")):
        if not isinstance(section, Mapping):
            continue
        title = _text(section.get("section_name"))
        paragraphs = "\n".join(_text(value) for value in section.get("paragraphs", []) or [])
        sections.append(f"## {title}\n{paragraphs}" if title else paragraphs)
    abstract = _text(row.get("abstract"))
    if abstract:
        sections.insert(0, f"## Abstract\n{abstract}")
    return "\n\n".join(section for section in sections if section)


def _qasper_answer(question: Mapping[str, Any]) -> str:
    for annotation in _sequence_rows(question.get("answers")):
        answer = annotation.get("answer", {}) if isinstance(annotation, Mapping) else {}
        if answer.get("unanswerable"):
            continue
        free_form = _text(answer.get("free_form_answer"))
        if free_form:
            return free_form
        spans = [_text(span) for span in answer.get("extractive_spans", []) or [] if _text(span)]
        if spans:
            return " ".join(spans)
        if answer.get("yes_no") is not None:
            return "yes" if answer["yes_no"] else "no"
    return ""


def normalize_dapo(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    prompt = _prompt_content(row)
    return (prompt, ground_truth) if prompt and _text(ground_truth) else None


def normalize_deepscaler(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    prompt, answer = _text(row.get("problem")), _text(row.get("answer"))
    return (prompt, answer) if prompt and answer else None


def normalize_deepmath(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    prompt, answer = _text(row.get("question")), _text(row.get("final_answer"))
    return (prompt, answer) if prompt and answer else None


def normalize_science(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    prompt, answer = _text(row.get("input")), _text(row.get("expected_answer"))
    return (prompt, answer) if prompt and answer else None


def normalize_hotpot(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    question, answer = _text(row.get("question")), _text(row.get("answer"))
    context = _hotpot_context(row)
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}" if context and question else ""
    return (prompt, answer) if prompt and answer else None


def normalize_qasper(row: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    context = _qasper_context(row)
    if not context:
        return
    for question in _sequence_rows(row.get("qas")):
        prompt, answer = _text(question.get("question")), _qasper_answer(question)
        if prompt and answer:
            yield (f"Paper context:\n{context}\n\nQuestion:\n{prompt}", answer)


def normalize_hardtests(row: Mapping[str, Any]) -> tuple[str, Any] | None:
    tests = row.get("public_test_cases") or []
    inputs = [_code_text(case.get("input")) for case in tests if isinstance(case, Mapping)]
    outputs = [_code_text(case.get("output")) for case in tests if isinstance(case, Mapping)]
    if not inputs or len(inputs) != len(outputs):
        return None
    prompt = _text(row.get("question_content"))
    ground_truth = {"inputs": inputs, "outputs": outputs}
    return (prompt, ground_truth) if prompt else None


ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "dapo": normalize_dapo,
    "deepscaler": normalize_deepscaler,
    "deepmath": normalize_deepmath,
    "science": normalize_science,
    "hotpot": normalize_hotpot,
    "qasper": normalize_qasper,
    "hardtests": normalize_hardtests,
}


def _iter_hf_rows(spec: SourceSpec) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(spec.repo, spec.config, split=spec.split, streaming=True)
    return dataset


def _iter_qasper_rows() -> Iterator[Mapping[str, Any]]:
    with urllib.request.urlopen(QASPER_TRAIN_URL, timeout=120) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
        try:
            member = next(member for member in archive.getmembers() if member.name.endswith("qasper-train-v0.3.json"))
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Qasper archive member is unreadable: {member.name}")
            records = json.load(handle)
        finally:
            archive.close()
    for paper_id, row in records.items():
        yield {"id": paper_id, **row}


def _iter_source_rows(spec: SourceSpec) -> Iterable[Mapping[str, Any]]:
    return _iter_qasper_rows() if spec.adapter == "qasper" else _iter_hf_rows(spec)


class CorpusWriter:
    def __init__(self, output: Path):
        self.output = output
        self.seen: set[str] = set()
        self.counts: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()

    def add(self, spec: SourceSpec, prompt: str, ground_truth: Any) -> bool:
        canonical = {
            "query": prompt,
            "ground_truth": ground_truth,
            "domain": spec.domain,
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if fingerprint in self.seen:
            self.skipped["duplicate"] += 1
            return False
        self.seen.add(fingerprint)
        row = {
            "uuid": f"{spec.domain}_{fingerprint[:24]}",
            "query": prompt,
            "ground_truth": ground_truth,
            "source": spec.name,
            "domain": spec.domain,
        }
        path = self.output / spec.domain / f"{spec.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.counts[spec.name] += 1
        return True


def prepare(output: Path, *, limit_per_source: int = 0, selected_domains: set[str] | None = None) -> dict[str, Any]:
    """Download and normalize all selected public source datasets."""

    if limit_per_source < 0:
        raise ValueError("limit_per_source must be non-negative")
    if output.exists():
        raise FileExistsError(f"Output source directory already exists: {output}")
    if selected_domains is None:
        selected_domains = set(DOMAINS)
    unknown = selected_domains - set(DOMAINS)
    if unknown:
        raise ValueError(f"Unknown domains: {sorted(unknown)}")

    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary source directory already exists: {temporary}")
    writer = CorpusWriter(temporary)
    sources: list[dict[str, Any]] = []
    try:
        for spec in SOURCE_SPECS:
            if spec.domain not in selected_domains:
                continue
            record = {"name": spec.name, "repo": spec.repo, "domain": spec.domain, "status": "completed"}
            read = written = skipped = 0
            duplicate_before = writer.skipped["duplicate"]
            try:
                for index, raw in enumerate(_iter_source_rows(spec)):
                    if limit_per_source and read >= limit_per_source:
                        break
                    read += 1
                    normalized = ADAPTERS[spec.adapter](raw)
                    values = normalized if spec.adapter == "qasper" else (normalized,)
                    for item in values:
                        if item is None:
                            skipped += 1
                            continue
                        prompt, ground_truth = item
                        written += int(writer.add(spec, prompt, ground_truth))
                record.update(
                    read=read,
                    written=written,
                    skipped=skipped + writer.skipped["duplicate"] - duplicate_before,
                )
            except Exception as error:
                record.update(status="error", error=repr(error), read=read, written=written, skipped=skipped)
                sources.append(record)
                raise
            sources.append(record)

        manifest = {
            "stage": "pre_difficulty_calibration",
            "source_description": "Public upstream sources before the MiniCPM pass-rate-1 screen",
            "domains": sorted(selected_domains),
            "limit_per_source": limit_per_source,
            "sources": sources,
            "unverifiable_sources": list(UNVERIFIABLE_CODE_SOURCES),
            "unavailable_sources": list(UNAVAILABLE_SOURCES),
            "counts": dict(writer.counts),
            "total_rows": sum(writer.counts.values()),
        }
        (temporary / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Canonical source corpus directory to create")
    parser.add_argument("--limit-per-source", type=int, default=0, help="Rows per source; 0 means all rows")
    parser.add_argument("--domain", action="append", choices=DOMAINS, help="Restrict preparation to a domain")
    args = parser.parse_args()
    if args.limit_per_source < 0:
        parser.error("--limit-per-source must be non-negative")
    manifest = prepare(
        args.output,
        limit_per_source=args.limit_per_source,
        selected_domains=set(args.domain) if args.domain else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
