"""Build model-specific, exact-token SFT data with projection-inspired sampling.

Prototypes: aakaran/finetuning-with-sampling boost_math.py and the native
SkyRL pretokenized SFT contract. No trainer or policy-loss changes are needed.
"""

import argparse
import functools
import hashlib
import os
import time
from pathlib import Path

from examples.train.sampling_sft.data import (
    append_jsonl,
    expert_rows,
    open_output,
    validation_question,
)
from skyrl.train.utils.projection_sampling import (
    proj_samp_vLLM,
    vLLMAutoregressiveSampler,
)


def boost_record(p, row, args):
    single = isinstance(row, dict)
    rows = [row] if single else row
    for item in rows:
        if not isinstance(item.get("solution"), str) or not item["solution"].strip():
            raise ValueError(f"Missing expert solution for {item['uuid']}")
    seeds = [
        int.from_bytes(hashlib.sha256(f"{args.seed}:{item['uuid']}".encode()).digest()[:4], "big") for item in rows
    ]
    start = time.monotonic()
    results = proj_samp_vLLM(
        p,
        rows[0]["prompt"] if single else [item["prompt"] for item in rows],
        rows[0]["solution"] if single else [item["solution"] for item in rows],
        mcmc_steps=args.mcmc_steps,
        max_new_tokens=args.max_new_tokens,
        block_num=args.block_num,
        seed=seeds[0] if single else seeds,
        skip_repetition=args.skip_repetition,
    )
    records = []
    for item, result in zip(rows, [results] if single else results, strict=True):
        prompt_ids = p.format_prompt(item["prompt"])
        response_ids = result["response_ids"]
        complete = result["finish_reason"] == "stop"
        records.append(
            {
                "uuid": item["uuid"],
                "domain": item.get("domain", ""),
                "prompt": item["prompt"],
                "solution": item["solution"],
                "ground_truth": item.get("ground_truth"),
                "question_id": item.get("question_id", item["uuid"]),
                "gen": p.tokenizer.decode(response_ids, skip_special_tokens=True),
                # Keep truncated records for audit/resume, but make them non-trainable.
                "input_ids": prompt_ids + response_ids if complete else [],
                "loss_mask": [0] * len(prompt_ids) + [1] * len(response_ids) if complete else [],
                **result,
                "runtime_sec": time.monotonic() - start,
                "sampling_batch_size": len(rows),
            }
        )
    return records[0] if single else records


def run(args):
    from transformers import AutoConfig, GenerationConfig

    model_config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    is_rwkv = model_config.model_type == "rwkv"
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.0 if is_rwkv and args.sampling_preset == "model" else 1.1
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    template = {}
    sampling_params = {
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.sampling_preset == "paper":
        sampling_params.update({"presence_penalty": 0.0, "frequency_penalty": 0.0})
    if is_rwkv:
        if args.sampling_preset == "model":
            generation_config = GenerationConfig.from_pretrained(args.model, revision=args.revision).to_dict()
            sampling_params.update(
                {
                    key: generation_config[key]
                    for key in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "presence_penalty",
                        "frequency_penalty",
                        "penalty_decay",
                    )
                }
            )
        cache = Path.home() / ".cache/skyrl-rwkv"
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache / "torch_extensions"))
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        template = {
            "rwkv_prompt_template": args.rwkv_prompt_template,
            "rwkv_generation_prompt": args.rwkv_generation_prompt,
        }
    config = {
        "version": 5 if is_rwkv else 1,
        "algorithm": "greedy_mean_logprob_suffix_search",
        "input": str(Path(args.input).resolve()),
        "model": args.model,
        "revision": args.revision,
        "model_type": args.model_type,
        "chat_template_kwargs": template,
        "max_new_tokens": args.max_new_tokens,
        "block_num": args.block_num,
        "mcmc_steps": args.mcmc_steps,
        "repetition_penalty": args.repetition_penalty,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "partition": args.partition,
        "validation_fraction": args.validation_fraction,
        "subset": args.subset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "skip_repetition": args.skip_repetition,
        "repetition_filter": "three_adjacent_spans_min16" if args.skip_repetition else None,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if is_rwkv:
        config["sampling_params"] = sampling_params
    if args.sampling_preset == "paper":
        config["sampling_preset"] = args.sampling_preset
    with open_output(
        args.output,
        config,
        args.resume,
        execution_keys=("batch_size", "max_num_batched_tokens", "enforce_eager", "gpu_memory_utilization"),
    ) as (handle, completed):
        # Validate resume metadata before allocating a GPU model.
        from vllm import LLM

        if is_rwkv:
            from examples.train.rwkv.prepare_flashrwkv2 import check

            check()
        llm = LLM(
            model=args.model,
            revision=args.revision,
            trust_remote_code=True,
            dtype="float16" if is_rwkv else "auto",
            hf_overrides={"wkv_mode": "fp32io16"} if is_rwkv else {},
            mamba_ssm_cache_dtype="float32" if is_rwkv else "auto",
            generation_config="vllm",
            logprobs_mode="raw_logprobs",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=max(32, args.batch_size),
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )
        try:
            p = vLLMAutoregressiveSampler(
                llm,
                model_type=args.model_type,
                chat_template_kwargs=template,
                repetition_penalty=args.repetition_penalty,
                sampling_params=sampling_params,
            )
            selected = written = truncated = repetitions = 0
            pending = []

            def write_batch(rows):
                nonlocal written, truncated, repetitions
                format_prompt = p.format_prompt
                if is_rwkv:
                    # Like native SFT's _parse_tools_str cache: reuse fixed text.
                    # RWKV appends response token IDs after formatting, so each
                    # batch only needs its question and expert-conditioned prefix.
                    p.format_prompt = functools.lru_cache(maxsize=2 * len(rows))(format_prompt)
                try:
                    result = boost_record(p, rows[0] if args.batch_size == 1 else rows, args)
                finally:
                    if is_rwkv:
                        p.format_prompt = format_prompt
                for record in [result] if args.batch_size == 1 else result:
                    append_jsonl(handle, record)
                    completed.add(record["uuid"])
                    written += 1
                    truncated += record["finish_reason"] == "length"
                    repetitions += record["finish_reason"] == "repetition"
                    print(
                        f"written={written} truncated={truncated} repetitions={repetitions} uuid={record['uuid']}",
                        flush=True,
                    )

            for index, row in enumerate(expert_rows(args.input, subset=args.subset)):
                if index % args.num_shards != args.shard_index:
                    continue
                validation = validation_question(
                    row.get("question_id", row["uuid"]),
                    args.seed,
                    args.validation_fraction,
                )
                if args.partition != "all" and validation != (args.partition == "validation"):
                    continue
                if args.limit and selected >= args.limit:
                    break
                selected += 1
                if row["uuid"] in completed:
                    continue
                pending.append(row)
                if len(pending) == args.batch_size:
                    write_batch(pending)
                    pending = []
            if pending:
                write_batch(pending)
        finally:
            llm.llm_engine.engine_core.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Expert JSONL file or directory (uuid/prompt/solution)",
    )
    parser.add_argument("--output", required=True, help="Exact-token JSONL for native pretokenized SFT")
    parser.add_argument(
        "--model",
        required=True,
        help="Exact learner checkpoint path or Hugging Face model ID",
    )
    parser.add_argument("--revision", help="Hugging Face model revision")
    parser.add_argument("--model-type", choices=["base", "chat"], default="chat")
    parser.add_argument(
        "--rwkv-prompt-template",
        choices=["bot", "assistant", "function_calling"],
        default="bot",
    )
    parser.add_argument(
        "--rwkv-generation-prompt",
        choices=["open_think", "fake_think"],
        default="open_think",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1856)
    parser.add_argument("--block-num", type=int, default=32, help="Block count, not tokens per block")
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=10,
        help="Greedy proposals per block; 0 gives the rewrite baseline",
    )
    parser.add_argument(
        "--sampling-preset",
        choices=["model", "paper"],
        default="model",
        help="model: RWKV generation_config defaults; paper: temperature 0.6, top-p 1, repetition penalty 1.1",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Default: 1.0 for RWKV model preset, otherwise 1.1",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Independent trajectories per vLLM proposal/scoring batch",
    )
    parser.add_argument(
        "--skip-repetition",
        action="store_true",
        help="Discard trajectories with three adjacent copies of any span of at least 16 tokens",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition", choices=["all", "train", "validation"], default="train")
    parser.add_argument("--subset", help="Source subset, e.g. cot for Nemotron-SFT-Math-v4")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0: all rows in this partition")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.block_num < 1 or args.max_new_tokens % args.block_num or args.mcmc_steps < 0:
        parser.error("Require positive max-new-tokens/block-num, exact divisibility and mcmc-steps >= 0")
    if not 0 < args.validation_fraction < 1 or args.limit < 0 or args.seed < 0:
        parser.error("Require 0 < validation-fraction < 1, limit >= 0 and seed >= 0")
    if args.max_model_len < 2 or args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Require max-model-len >= 2, tensor-parallel-size >= 1 and 0 < gpu-memory-utilization <= 1")
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        parser.error("Require repetition-penalty > 0")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    if args.batch_size < 1 or args.max_num_batched_tokens < 1:
        parser.error("Require batch-size and max-num-batched-tokens >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
