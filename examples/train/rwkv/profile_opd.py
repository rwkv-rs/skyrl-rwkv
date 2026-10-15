"""Profile full-sequence RWKV OPD policy or teacher kernels on all torchrun ranks."""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig, OptimizerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", choices=["policy", "ref"], required=True)
    parser.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sequence-length", type=int, default=25600)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    training = args.role == "policy"
    strategy = FSDPStrategy(
        FSDPConfig(fsdp_size=1, reshard_after_forward=False, sync_gradients_each_microbatch=False),
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
    else:
        wrapper = strategy.prepare(wrapper)
        wrapper.eval()
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
                logprobs = wrapper(ids, num_actions=8192, attention_mask=mask)
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
            del logprobs
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
