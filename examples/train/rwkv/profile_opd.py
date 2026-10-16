"""Profile RWKV OPD training kernels or fixed-work vLLM decoding."""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig, OptimizerConfig


def reload_inference_weights(model, path):
    """Use the native loader and the same derived-weight refresh as weight sync."""
    from safetensors import safe_open

    from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
        finalize_rwkv_runtime_weights,
        load_rwkv_checkpoint_weights,
    )

    def weights():
        for shard in sorted(Path(path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                names = handle.keys()
                for name in names:
                    yield name, handle.get_tensor(name)

    loaded = load_rwkv_checkpoint_weights(model, weights())
    finalize_rwkv_runtime_weights(model)
    return len(loaded)


def inference_graph_stats(worker, instrument=False):
    manager = worker.model_runner.cudagraph_manager
    if instrument:
        manager._opd_replays = 0
        original = manager.run_fullgraph

        def counted(*args, **kwargs):
            manager._opd_replays += 1
            return original(*args, **kwargs)

        manager.run_fullgraph = counted
    return {
        "captured_graphs": len(manager.graphs),
        "captured_token_counts": manager.captured_token_counts(),
        "replays": manager._opd_replays,
    }


def profile_inference(args):
    """Compare graph/eager reports on real prompts with identical decode work.

    Prototype: native vLLM LLM generation and RWKV weight-sync finalization.
    This is a standalone GPU process, separate from a running training job.
    """
    # Match run_engines_locally/vllm_v1_disable_multiproc in the OPD recipe.
    # The diagnostic callbacks then execute locally without pickle transport.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    prompts = []
    with args.rollouts.open() as handle:
        for line in handle:
            tokens = json.loads(line)["prompt_token_ids"]
            if len(tokens) <= 1024:
                prompts.append({"prompt_token_ids": tokens})
            if len(prompts) == max(args.micro_batch_sizes):
                break
    if len(prompts) < max(args.micro_batch_sizes):
        raise ValueError("The rollout file must contain enough prompts of at most 1024 tokens")
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.sequence_length,
        max_num_seqs=max(args.micro_batch_sizes),
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=args.enforce_eager,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE" if args.enforce_eager else "FULL_DECODE_ONLY"},
        enable_prefix_caching=True,
        enable_sleep_mode=True,
        logprobs_mode="raw_logprobs",
    )
    llm.collective_rpc(partial(inference_graph_stats, instrument=True))

    def generate(batch, length, temperature):
        llm.reset_prefix_cache()
        params = SamplingParams(
            temperature=temperature,
            top_p=1,
            top_k=-1,
            max_tokens=length,
            min_tokens=length,
            ignore_eos=True,
            logprobs=1,
            seed=42,
        )
        start = time.perf_counter()
        outputs = llm.generate(batch, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        records = []
        for output in outputs:
            completion = output.outputs[0]
            records.append(
                {
                    "tokens": completion.token_ids,
                    "logprobs": [
                        values[token].logprob
                        for token, values in zip(completion.token_ids, completion.logprobs, strict=True)
                    ],
                }
            )
        return elapsed, records

    result = {"model": args.model, "enforce_eager": args.enforce_eager, "correctness": {}, "benchmarks": []}
    check_prompts = prompts[: min(16, len(prompts))]
    _, result["correctness"]["initial"] = generate(check_prompts, 64, 0)
    llm.sleep(level=1)
    llm.wake_up()
    _, result["correctness"]["after_sleep"] = generate(check_prompts, 64, 0)
    if args.comparison_model:
        result["loaded_weights"] = llm.apply_model(partial(reload_inference_weights, path=args.comparison_model))
        _, result["correctness"]["changed_weights"] = generate(check_prompts, 64, 0)
        llm.sleep(level=1)
        llm.wake_up()
        llm.apply_model(partial(reload_inference_weights, path=args.model))
        _, result["correctness"]["restored_weights"] = generate(check_prompts, 64, 0)
    for size in args.micro_batch_sizes:
        generate(prompts[:size], 32, 1)
        times = [generate(prompts[:size], args.max_generate_length, 1)[0] for _ in range(args.steps)]
        result["benchmarks"].append(
            {
                "batch_size": size,
                "generated_tokens": size * args.max_generate_length,
                "seconds": times,
                "tokens_per_second": size * args.max_generate_length / (sum(times) / len(times)),
            }
        )
    result["graphs"] = llm.collective_rpc(inference_graph_stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "correctness"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", choices=["policy", "ref", "inference"], required=True)
    parser.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sequence-length", type=int, default=25600)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparison-model")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-generate-length", type=int, default=512)
    args = parser.parse_args()
    if args.role == "inference":
        if args.rollouts is None or args.output is None:
            parser.error("inference profiling requires --rollouts and --output")
        profile_inference(args)
        return
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    training = args.role == "policy"
    strategy = FSDPStrategy(
        FSDPConfig(fsdp_size=args.fsdp_size, reshard_after_forward=False, sync_gradients_each_microbatch=False),
        optimizer_config=OptimizerConfig(lr=2e-6, offload_after_step=False) if training else None,
        num_training_steps=10,
    )
    strategy.setup_distributed()
    wrapper = HFModelWrapper(
        args.model,
        bf16=True,
        lora_rank=args.rank if training else 0,
        lora_alpha=args.rank,
        use_nora=training,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
        model_config_kwargs={"wkv_mode": "fp32io16"},
        meta_init=dist.get_rank() != 0,
    )
    if training:
        wrapper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapper, optimizer, _ = strategy.prepare((wrapper, None, None))
        wrapper.train()
    else:
        wrapper = strategy.prepare(wrapper)
        wrapper.eval()
    # Match FSDPPolicyWorkerBase: keep model storage IPC-compatible and enable
    # expandable activation allocations only after model initialization.
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    for batch_size in args.micro_batch_sizes:
        ids = torch.randint(1, 60000, (batch_size, args.sequence_length), device="cuda")
        mask = torch.ones_like(ids)
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps + 1):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.set_grad_enabled(training):
                # Match PolicyWorkerBase._forward_backward_micro, including
                # autocast's FP32 log-softmax and the entropy logging pass.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=training):
                    logprobs, output = wrapper(
                        ids,
                        num_actions=8192,
                        attention_mask=mask,
                        return_output=True,
                        compute_entropy=training,
                        entropy_requires_grad=False,
                    )
                if training:
                    loss = -logprobs.float().mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - start, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if step:
                times.append(elapsed.item())
            del logprobs, output
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        seconds = sum(times) / len(times)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "role": args.role,
                        "micro_batch_size": batch_size,
                        "sequence_length": args.sequence_length,
                        "seconds": seconds,
                        "tokens_per_second": batch_size * args.sequence_length * dist.get_world_size() / seconds,
                        "peak_gib": peak.item() / 2**30,
                    }
                ),
                flush=True,
            )
        del ids, mask
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
