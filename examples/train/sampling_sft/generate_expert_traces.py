"""Generate expert traces through the .env DS_MODEL_* OpenAI-compatible API.

Prototype: examples/train_integrations/verifiers/verifiers_generator.py.
Run from the repo root with uv run --env-file .env (see README.md).
"""

import argparse
import asyncio
import hashlib
import os

from examples.train.sampling_sft.data import append_jsonl, open_output, questions

INSTRUCTION = (
    "Solve the following task carefully. Explain your reasoning and give a complete final answer. "
    "Follow the output format requested by the task; for a mathematics problem, put the final answer in \\boxed{}."
)


def expert_trace(row, response, model):
    """Preserve reasoning_content separately, and use both parts as the expert solution."""
    choice = response.choices[0]
    content = choice.message.content
    if choice.finish_reason != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError(f"Incomplete expert response for {row['uuid']}: finish_reason={choice.finish_reason!r}")
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    solution = "\n".join(part for part in (reasoning, content) if part)
    return {
        **row,
        "solution": solution,
        "reasoning_content": reasoning,
        "content": content,
        "teacher_model": model,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage else None,
    }


async def generate(args):
    from openai import AsyncOpenAI

    model = os.environ["DS_MODEL_NAME"]
    # Accept either a base URL or the chat-completions URL at this user-input boundary.
    base_url = os.environ["DS_MODEL_URL"].rstrip("/").removesuffix("/chat/completions")
    config = {
        "version": 2,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "revision": args.revision,
        "question_key": args.question_key,
        "id_key": args.id_key,
        "answer_key": args.answer_key,
        "domain": args.domain,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "teacher_model": model,
        "endpoint_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
        "instruction": INSTRUCTION,
    }
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ["DS_MODEL_API_KEY"],
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    with open_output(args.output, config, args.resume) as (handle, completed):
        source = iter(questions(args))
        count = 0

        async def worker():
            nonlocal count
            for row in source:
                if row["uuid"] in completed:
                    continue
                # Reserve before awaiting so duplicate UUIDs cannot race between workers.
                completed.add(row["uuid"])
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": INSTRUCTION + "\n\n" + row["prompt"]}],
                )
                record = expert_trace(row, response, model)
                append_jsonl(handle, record)
                count += 1
                print(f"completed={count} uuid={row['uuid']} model={model}", flush=True)

        async with client:
            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Local JSONL/Parquet file/tree, or a Hugging Face dataset ID")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--revision", help="Hugging Face dataset revision")
    parser.add_argument("--question-key", default="query")
    parser.add_argument("--id-key", default="uuid")
    parser.add_argument("--answer-key", default="ground_truth")
    parser.add_argument("--domain", action="append", help="Repeat to include multiple UltraData domains")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0: all selected questions; API requests incur cost")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require limit >= 0, num-shards >= 1, and 0 <= shard-index < num-shards")
    if args.concurrency < 1 or args.timeout <= 0 or args.max_retries < 0:
        parser.error("Require positive concurrency, timeout and non-negative max-retries")
    return args


if __name__ == "__main__":
    asyncio.run(generate(parse_args()))
