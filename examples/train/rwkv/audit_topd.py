"""Replay saved OPD rollouts through native FSDP and an independent TOP-D loss.

Run with torchrun on an idle training node. Model and Adam state are restored
from a complete checkpoint. Reports and diagnostic parameter snapshots use a
separate output directory. The native worker and benchmark_stopd provide the
execution paths.
"""

import argparse
import copy
import gc
import json
import math
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from examples.train.on_policy_distillation.main_on_policy_distill import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationTrainer,
)
from examples.train.on_policy_distillation.stopd import STOPD
from examples.train.on_policy_distillation.topd import topd_stopd_policy_loss
from examples.train.rwkv.prepare_flashrwkv2 import check
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.distributed.fsdp_utils import should_use_meta_init
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPRefWorkerBase
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.workers.worker import RefWorkerBase
from skyrl.train.config import MixedPrecisionConfig


def reference_loss_gradient(data, log_probs, config):
    """Differentiate the explicit sum of prefix PPO objectives in float64.

    This does not use the production returns, normalization, signed channels,
    or telescoping-coefficient implementation. Each observed level constructs
    its own scalar objective before the differences are weighted.
    """
    groups = data.metadata["policy_prompt_boundaries"]
    mask = data["loss_mask"].bool() & data["response_mask"].bool()
    result = torch.zeros_like(log_probs)
    total_loss = 0.0
    maximum = config.generator.sampling_params.max_generate_length
    block = config.trainer.algorithm.stopd.block
    alpha = config.trainer.algorithm.proximal_teacher_alpha
    epsilon = config.trainer.algorithm.topd.epsilon
    clip_low = config.trainer.algorithm.eps_clip_low
    clip_high = config.trainer.algorithm.eps_clip_high
    for start, end in groups:
        indices = [mask[i].nonzero().flatten() for i in range(start, end)]
        current = [
            log_probs[i, index].double().detach().requires_grad_() for i, index in zip(range(start, end), indices)
        ]
        old = [data["action_log_probs"][i, index].double() for i, index in zip(range(start, end), indices)]
        teacher = [data["base_action_log_probs"][i, index].double() for i, index in zip(range(start, end), indices)]
        rewards = [
            torch.logaddexp(t - o + math.log(alpha), torch.full_like(o, math.log1p(-alpha)))
            for t, o in zip(teacher, old)
        ]
        previous = torch.zeros((), dtype=torch.float64)
        objective = torch.zeros((), dtype=torch.float64)
        levels = math.ceil(max(map(len, current)) / block)
        for level in range(1, levels + 1):
            horizon = level * block
            values = []
            for reward in rewards:
                prefix = reward[:horizon]
                future_sum = prefix.flip(0).cumsum(0).flip(0) - prefix
                future_count = torch.arange(len(prefix) - 1, -1, -1).clamp_min(1)
                values.append(prefix + future_sum / future_count)
            pooled = torch.cat(values)
            mean = pooled.mean()
            standard_deviation = ((pooled - mean).square().mean()).sqrt()
            level_loss = torch.zeros((), dtype=torch.float64)
            for now, before, value in zip(current, old, values):
                advantage = (value - mean) / (standard_deviation + epsilon)
                ratio = (now[:horizon] - before[:horizon]).clamp(-20, 20).exp()
                bounded = ratio.clamp(1 - clip_low, 1 + clip_high)
                level_loss = level_loss - torch.minimum(ratio * advantage, bounded * advantage).sum() / len(pooled)
            survival = (maximum - (level - 1) * block) / maximum
            objective = objective + (level_loss - previous) / survival
            previous = level_loss
        objective = objective / len(groups)
        derivatives = torch.autograd.grad(objective, current)
        total_loss += objective.item()
        for i, index, derivative in zip(range(start, end), indices, derivatives):
            result[i, index] = derivative.float()
    return total_loss, result


def linear_reference_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, **kwargs):
    return -(log_probs * advantages * loss_mask).sum(), {}


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def compare_tensors(actual, expected):
    dot = aa = bb = error = maximum = 0.0
    for value, reference in zip(actual, expected, strict=True):
        a = value.detach().double().cpu()
        b = reference.detach().double().cpu()
        dot += (a * b).sum().item()
        aa += a.square().sum().item()
        bb += b.square().sum().item()
        error += (a - b).square().sum().item()
        maximum = max(maximum, (a - b).abs().max().item())
    return {
        "actual_norm": math.sqrt(aa),
        "reference_norm": math.sqrt(bb),
        "relative_error": math.sqrt(error / max(bb, 1e-30)),
        "cosine": dot / max(math.sqrt(aa * bb), 1e-30),
        "max_abs_error": maximum,
    }


def compare_learning_updates(trainer, worker, data, restore, forward, emit, rank, world):
    """Pair eight real Adam updates on complete trajectories and random prefixes.

    Both arms retain the same computation shapes, checkpoint, Adam state, old
    logprobs, prompts, and teacher scores. Only the loss's prefix observation
    changes. This isolates the estimator from generation and kernel rounding.
    The complete EOS trajectories are selected for a controlled replay, so
    these measurements are not a generalization benchmark.
    """
    original_stopd = trainer.stopd
    original_loss_type = trainer.cfg.trainer.algorithm.policy_loss_type
    plain_config = copy.deepcopy(trainer.cfg.trainer.algorithm)
    plain_config.policy_loss_type = "regular"
    full = copy.deepcopy(data)
    trainer.stopd = None
    trainer.compute_advantages_and_returns(full)
    trainer._normalize_advantages(full, [(0, len(full))])
    full_advantages = full["advantages"].clone()
    minibatch_rows = len(data) // 4
    boundaries = [(i, i + minibatch_rows) for i in range(0, len(data), minibatch_rows)]
    assert minibatch_rows % (world * trainer.cfg.trainer.micro_train_batch_size_per_gpu) == 0
    groups = data.metadata["policy_prompt_boundaries"]
    mask = data["loss_mask"].bool()
    positions = data["response_mask"].long().cumsum(-1)
    for name, seed in [("full_topd", None), ("stopd_seed0", 0), ("stopd_seed1", 1), ("stopd_seed2", 2)]:
        restore()
        observed = copy.deepcopy(data)
        observed.metadata["policy_mini_batch_boundaries"] = boundaries
        trainer.stopd = None if seed is None else original_stopd
        if seed is not None:
            rng = random.Random(seed)
            for start, end in groups:
                anchor = rng.randrange(original_stopd.max_len)
                budget = min((anchor // original_stopd.block + 1) * original_stopd.block, original_stopd.max_len)
                observed["loss_mask"][start:end] *= (positions[start:end] <= budget).to(observed["loss_mask"].dtype)
        trainer.compute_advantages_and_returns(observed)
        trainer._normalize_advantages(observed, boundaries)
        worker.policy_loss_fn = ppo_policy_loss if seed is None else topd_stopd_policy_loss
        worker.cfg.algorithm.policy_loss_type = "regular" if seed is None else "topd_stopd"
        update_metrics = []
        for epoch in range(2):
            for start, end in boundaries:
                per_rank = (end - start) // world
                shard = copy.deepcopy(observed[start + rank * per_rank : start + (rank + 1) * per_rank])
                result = worker.forward_backward(shard, return_per_token_outputs=False)
                norm = worker.optim_step()
                update_metrics.append({"epoch": epoch, "grad_norm": norm, "metrics": result.metrics})
        after = forward(worker, data)
        if rank == 0:
            loss, _ = ppo_policy_loss(
                after,
                data["action_log_probs"],
                full_advantages,
                plain_config,
                loss_mask=data["loss_mask"],
            )
            delta = after - data["action_log_probs"]
            ratio = delta.exp()
            old_kl = data["action_log_probs"] - data["base_action_log_probs"]
            # Fixed-prefix conditional-KL Monte Carlo surrogate, with a
            # ratio-1 control variate. This does not estimate new state visits.
            kl_change = (ratio * (after - data["base_action_log_probs"] - 1) + 1 - old_kl)[mask].mean()
            emit(
                name,
                {
                    "observed_tokens": int(observed["loss_mask"].sum()),
                    "complete_tokens": int(mask.sum()),
                    "full_topd_loss_after": loss.item(),
                    "conditional_kl_surrogate_change": kl_change.item(),
                    "linear_kl_change": (delta * old_kl)[mask].mean().item(),
                    "mean_abs_logprob_change": delta[mask].abs().mean().item(),
                    "updates": update_metrics,
                },
            )
    trainer.stopd = original_stopd
    trainer.cfg.trainer.algorithm.policy_loss_type = original_loss_type


def compare_conditional_kl(args, cfg, rank, world, emit):
    """Exact vocabulary KL on identical saved prefix states across checkpoints."""
    rows = []
    with args.rollouts.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == args.samples:
                break
    domains = sorted({r["env_extras"]["data_source"] for r in rows})
    local_rows = rows[rank::world]

    def logits(model, row):
        prompt, response = row["prompt_token_ids"], row["response_token_ids"]
        offsets = torch.linspace(0, len(response) - 1, min(16, len(response))).long()
        positions = (offsets + len(prompt) - 1).unsqueeze(0).to("cuda")
        ids = torch.tensor([prompt + response], device="cuda")
        # Match HFModelWrapper: CUDA autocast would promote RWKV LayerNorm
        # outputs to FP32 instead of retaining the FSDP compute dtype.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            output = model.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=positions
            )
        return output.logits[0].float().log_softmax(-1).cpu()

    ref_strategy = FSDPStrategy(cfg.trainer.ref.fsdp_config, seed=cfg.trainer.seed)
    ref_strategy.setup_distributed()
    teacher = HFModelWrapper(
        cfg.trainer.ref.model.path,
        bf16=True,
        model_config_kwargs={"wkv_mode": "fp32io16"},
        meta_init=should_use_meta_init(mesh=ref_strategy.device_mesh),
    )
    teacher = ref_strategy.prepare(teacher)
    teacher.model.eval()
    targets = [logits(teacher, row) for row in local_rows]
    emit("teacher_states", sum(len(t) for t in targets) * world)
    del teacher, ref_strategy
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    adapter = cfg.trainer.policy.model.lora
    strategy = FSDPStrategy(
        cfg.trainer.policy.fsdp_config, model_config=cfg.trainer.policy.model, seed=cfg.trainer.seed
    )
    strategy.setup_distributed()
    student = HFModelWrapper(
        cfg.trainer.policy.model.path,
        bf16=True,
        lora_rank=adapter.rank,
        lora_alpha=adapter.alpha,
        lora_dropout=0,
        use_nora=True,
        target_modules=adapter.target_modules,
        model_config_kwargs={"wkv_mode": "fp32io16"},
    )
    student = strategy.prepare(student)
    student.model.eval()
    proximal_targets = []
    alpha = cfg.trainer.algorithm.proximal_teacher_alpha
    baseline_adapters = {}
    comparisons = [(args.checkpoint, 1.0)] + [
        (checkpoint, scale) for checkpoint in args.comparison_checkpoints for scale in args.comparison_scales
    ]
    for index, (checkpoint, scale) in enumerate(comparisons):
        for module in student.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        strategy.load_checkpoint(
            student, str(checkpoint / "policy"), load_optimizer_states=False, load_lr_scheduler_states=False
        )
        with torch.no_grad():
            for name, parameter in student.named_parameters():
                if ".lora_A." not in name and ".lora_B." not in name:
                    continue
                value = local_tensor(parameter)
                if index == 0:
                    baseline_adapters[name] = value.detach().cpu().clone()
                elif scale != 1:
                    value.lerp_(baseline_adapters[name].to(value.device), 1 - scale)
        values = {domain: [0.0, 0.0, 0.0, 0.0] for domain in domains}
        per_sample = []
        for sample_index, (row, target) in enumerate(zip(local_rows, targets, strict=True)):
            logp = logits(student, row)
            if index == 0:
                proximal_targets.append(
                    target if alpha == 1 else torch.logaddexp(target + math.log(alpha), logp + math.log1p(-alpha))
                )
            probability = logp.exp()
            kl = (probability * (logp - target)).sum(-1)
            proximal_kl = (probability * (logp - proximal_targets[sample_index])).sum(-1)
            entropy = -(probability * logp).sum(-1)
            item = values[row["env_extras"]["data_source"]]
            item[0] += kl.double().sum().item()
            item[1] += entropy.double().sum().item()
            item[2] += len(kl)
            item[3] += proximal_kl.double().sum().item()
            per_sample.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "domain": row["env_extras"]["data_source"],
                    "reverse_kl": kl.double().mean().item(),
                    "proximal_kl": proximal_kl.double().mean().item(),
                    "states": len(kl),
                }
            )
        samples = [None] * world
        dist.all_gather_object(samples, per_sample)
        tensor = torch.tensor([values[d] for d in domains], dtype=torch.float64, device="cuda")
        dist.all_reduce(tensor)
        totals = tensor.sum(0).cpu()
        emit(
            f"checkpoint_{index}",
            {
                "path": str(checkpoint),
                "update_scale": scale,
                "states": int(totals[2]),
                "reverse_kl": (totals[0] / totals[2]).item(),
                "entropy": (totals[1] / totals[2]).item(),
                "proximal_kl": (totals[3] / totals[2]).item(),
                "per_sample": [sample for shard in samples for sample in shard],
                "domains": {
                    d: {
                        "states": int(v[2]),
                        "reverse_kl": (v[0] / v[2]).item(),
                        "entropy": (v[1] / v[2]).item(),
                        "proximal_kl": (v[3] / v[2]).item(),
                    }
                    for d, v in zip(domains, tensor.cpu())
                },
            },
        )


def compare_teacher_precision(args, cfg, rank, world, emit):
    """Separate teacher weight rounding from compute precision on fixed states."""
    from transformers import AutoModelForCausalLM

    from examples.train.rwkv.antidoom import load_model

    rows = []
    with args.rollouts.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == args.samples:
                break
    local_rows = rows[rank::world]
    adapter_config = json.loads((args.teacher_adapter / "adapter_config.json").read_text())
    targets = []
    variants = args.precision_variants
    for variant in variants:
        worker = None
        if variant == "native_fp16":
            cfg.trainer.ref.fsdp_config.mixed_precision = MixedPrecisionConfig(param_dtype="fp16")
            worker = object.__new__(FSDPRefWorkerBase)
            worker.cfg = cfg.trainer
            worker.sequence_parallel_size = cfg.trainer.ref.sequence_parallel_size
            worker.init_model(cfg.trainer.ref.model.path)
            model = worker.model.model
        elif variant == "unmerged_nora":
            model = load_model(adapter_config["base_model_name_or_path"], args.teacher_adapter)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.trainer.ref.model.path,
                dtype=torch.bfloat16 if variant == "merged_bf16" else torch.float16,
                wkv_mode="fp32io16",
            )
            if variant == "rounded_fp16":
                # Keep the FP16 execution path, changing only stored weight values.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.copy_(parameter.bfloat16().half())
        if worker is None:
            model.to("cuda").eval()
        measurements = []
        wrapper_errors = []
        for row_index, row in enumerate(local_rows):
            prompt, response = row["prompt_token_ids"], row["response_token_ids"]
            offsets = torch.linspace(0, len(response) - 1, min(64, len(response))).long()
            positions = (offsets + len(prompt) - 1).unsqueeze(0).cuda()
            ids = torch.tensor([prompt + response], device="cuda")
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                logp = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0]
                logp = logp.float().log_softmax(-1).cpu()
            if variant == "merged_fp16":
                targets.append(logp)
            target = targets[row_index]
            actions = torch.tensor(response)[offsets, None]
            action = logp.gather(-1, actions).squeeze(-1)
            expected = target.gather(-1, actions).squeeze(-1)
            if worker is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    native = worker.model(
                        ids,
                        num_actions=len(response),
                        attention_mask=torch.ones_like(ids),
                        response_mask=torch.ones((1, len(response)), device="cuda"),
                    )
                wrapper_errors.append((native[0].cpu()[offsets] - expected).abs().max().item())
            old = torch.tensor(row["rollout_logprobs"])[offsets]
            alpha = cfg.trainer.algorithm.proximal_teacher_alpha
            reward = torch.logaddexp(action - old + math.log(alpha), torch.full_like(action, math.log1p(-alpha)))
            expected_reward = torch.logaddexp(
                expected - old + math.log(alpha), torch.full_like(expected, math.log1p(-alpha))
            )
            measurements.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "domain": row["env_extras"]["data_source"],
                    "response_length": len(response),
                    "positions": offsets.tolist(),
                    "action_logprobs": action.tolist(),
                    "action_logprob_delta": (action - expected).tolist(),
                    "reward_delta": (reward - expected_reward).tolist(),
                    "kl": (target.exp() * (target - logp)).sum(-1).tolist(),
                    "top1_changed": (target.argmax(-1) != logp.argmax(-1)).tolist(),
                }
            )
        gathered = [None] * world
        dist.all_gather_object(gathered, measurements)
        flat = [sample for shard in gathered for sample in shard]
        delta = torch.tensor([v for item in flat for v in item["action_logprob_delta"]])
        reward_delta = torch.tensor([v for item in flat for v in item["reward_delta"]])
        emit(
            variant,
            {
                "states": delta.numel(),
                "mean_absolute_action_logprob_delta": delta.abs().mean().item(),
                "action_logprob_delta_quantiles": delta.abs().quantile(torch.tensor([0.5, 0.9, 0.99, 1.0])).tolist(),
                "mean_action_logprob_delta": delta.mean().item(),
                "mean_kl": sum(sum(item["kl"]) for item in flat) / delta.numel(),
                "top1_change_fraction": sum(sum(item["top1_changed"]) for item in flat) / delta.numel(),
                "mean_absolute_reward_delta": reward_delta.abs().mean().item(),
                "wrapper_max_action_logprob_error": max(wrapper_errors, default=0.0),
                "per_sample": flat,
            },
        )
        del model, worker
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--mode", choices=("gradient", "control", "score", "kl", "replay", "precision"), default="gradient"
    )
    parser.add_argument("--teacher-adapter", type=Path, help="Original AntiDoom adapter for the precision comparison")
    parser.add_argument(
        "--precision-variants",
        nargs="+",
        choices=("merged_fp16", "rounded_fp16", "merged_bf16", "unmerged_nora", "native_fp16"),
        default=["merged_fp16", "rounded_fp16", "merged_bf16", "unmerged_nora"],
    )
    parser.add_argument("--scored-input", type=Path, help="Saved native TOP-D input before advantage scaling")
    parser.add_argument("--comparison-checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument("--comparison-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[])
    args = parser.parse_args()
    if args.mode == "precision" and args.teacher_adapter is None:
        parser.error("precision requires --teacher-adapter")
    if args.mode == "precision" and args.precision_variants[0] != "merged_fp16":
        parser.error("precision comparisons must start with merged_fp16")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    check()
    cfg = OnPolicyDistillationConfig.from_cli_overrides(json.loads(args.config.read_text()))
    assert args.samples % (world * cfg.trainer.micro_train_batch_size_per_gpu) == 0
    assert cfg.generator.n_samples_per_prompt == 2
    cfg.trainer.train_batch_size = args.samples // 2
    cfg.trainer.policy_mini_batch_size = args.samples // 2
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    report = {"checkpoint": str(args.checkpoint), "rollouts": str(args.rollouts), "samples": args.samples}

    def emit(key, value):
        if rank == 0:
            report[key] = value
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value}), flush=True)

    if args.mode in ("kl", "precision"):
        compare = compare_conditional_kl if args.mode == "kl" else compare_teacher_precision
        compare(args, cfg, rank, world, emit)
        emit("complete", True)
        dist.barrier()
        dist.destroy_process_group()
        return

    rows = []
    pending = []
    with args.rollouts.open() as handle:
        for line in handle:
            pending.append(json.loads(line))
            if len(pending) != 2:
                continue
            if args.mode != "control" or all(r["ended_with_eos"] and not r["truncated"] for r in pending):
                rows.extend(pending)
            pending = []
            if len(rows) == args.samples:
                break
    assert len(rows) == args.samples
    trainer = object.__new__(OnPolicyDistillationTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.dispatch = SimpleNamespace(dp_size=lambda _: world, get_lcm_dp_size=lambda: world)
    trainer.all_metrics = {}
    trainer.stopd = STOPD(
        cfg.generator.sampling_params.max_generate_length,
        block=cfg.trainer.algorithm.stopd.block,
        warmup_steps=0,
        require_denominator=False,
    )
    generator_output = {
        "prompt_token_ids": [r["prompt_token_ids"] for r in rows],
        "response_ids": [r["response_token_ids"] for r in rows],
        "rewards": [[0.0] * len(r["response_token_ids"]) for r in rows],
        "loss_masks": [r["loss_mask"] for r in rows],
        "rollout_logprobs": [r["rollout_logprobs"] for r in rows],
    }
    uids = [r["env_extras"]["uuid"] for r in rows]
    assert all(uids[i] == uids[i + 1] for i in range(0, len(uids), 2))
    data = trainer.convert_to_training_input(generator_output, uids)
    if args.mode in ("score", "replay"):
        if args.scored_input is None:
            parser.error(f"{args.mode} requires --scored-input")
        with args.scored_input.open("rb") as handle:
            data = pickle.load(handle)
        assert len(data) == args.samples
    per_rank = args.samples // world
    emit(
        "batch",
        {
            "tokens": int(data["attention_mask"].sum()),
            "response_tokens": int(data["response_mask"].sum()),
            "balance": trainer.all_metrics,
        },
    )

    def forward(worker, batch):
        output = worker.forward(copy.deepcopy(batch[rank * per_rank : (rank + 1) * per_rank]))
        local = torch.tensor([r["logprobs"] for r in output.loss_fn_outputs], device="cuda")
        result = torch.empty((args.samples, local.shape[1]), dtype=local.dtype, device="cuda")
        dist.all_gather_into_tensor(result, local.contiguous())
        return result.cpu()

    if args.mode not in ("score", "replay"):
        ref_strategy = FSDPStrategy(cfg.trainer.ref.fsdp_config, seed=cfg.trainer.seed)
        ref_strategy.setup_distributed()
        reference = HFModelWrapper(
            cfg.trainer.ref.model.path,
            bf16=True,
            model_config_kwargs={"wkv_mode": "fp32io16"},
            rwkv_response_only_head=cfg.trainer.rwkv_response_only_head,
            meta_init=should_use_meta_init(mesh=ref_strategy.device_mesh),
        )
        reference = ref_strategy.prepare(reference)
        ref_worker = object.__new__(RefWorkerBase)
        ref_worker.cfg, ref_worker.model = cfg.trainer, reference
        data["base_action_log_probs"] = forward(ref_worker, data)
        emit("teacher_scored", True)
        del ref_worker, reference, ref_strategy
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    strategy = FSDPStrategy(
        cfg.trainer.policy.fsdp_config,
        cfg.trainer.policy.optimizer_config,
        cfg.trainer.policy.model,
        seed=cfg.trainer.seed,
        num_training_steps=2000,
    )
    strategy.setup_distributed()
    adapter = cfg.trainer.policy.model.lora
    model = HFModelWrapper(
        cfg.trainer.policy.model.path,
        bf16=True,
        lora_rank=adapter.rank,
        lora_alpha=adapter.alpha,
        lora_dropout=0,
        use_nora=True,
        target_modules=adapter.target_modules,
        model_config_kwargs={"wkv_mode": "fp32io16"},
        rwkv_response_only_head=cfg.trainer.rwkv_response_only_head,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model, optimizer, scheduler = strategy.prepare((model, None, None))

    def restore():
        # Scoring retains unsharded BF16 views. Restore the registered DTensor
        # parameters before DCP copies the saved model and Adam state.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        strategy.load_checkpoint(
            model,
            str(args.checkpoint / "policy"),
            optimizer,
            scheduler,
            load_optimizer_states=args.mode != "score",
            load_lr_scheduler_states=args.mode != "score",
        )
        optimizer.zero_grad(set_to_none=True)

    restore()
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker.cfg, worker.strategy, worker.model = cfg.trainer, strategy, model
    worker.optimizer, worker.scheduler = optimizer, scheduler
    worker.mesh_rank = SimpleNamespace(dp_size=world)
    worker.device_mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("dp", "sp"))
    worker.policy_loss_fn = topd_stopd_policy_loss
    worker._set_expandable_segments(True)
    if args.mode == "replay":
        if not args.learning_rates or any(not 0 < lr < float("inf") for lr in args.learning_rates):
            parser.error("replay requires positive finite --learning-rates")
        if data["advantages"].ndim == 2:
            trainer.stopd = None
            worker.policy_loss_fn = ppo_policy_loss
            worker.cfg.algorithm.policy_loss_type = "regular"
        boundaries = data.metadata["policy_mini_batch_boundaries"]
        for learning_rate in args.learning_rates:
            restore()
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = learning_rate
            scheduler.base_lrs = [learning_rate] * len(optimizer.param_groups)
            scheduler._last_lr = list(scheduler.base_lrs)
            observed = copy.deepcopy(data)
            trainer._normalize_advantages(observed, boundaries)
            updates = []
            for epoch in range(cfg.trainer.update_epochs_per_batch):
                for start, end in boundaries:
                    count = (end - start) // world
                    shard = copy.deepcopy(observed[start + rank * count : start + (rank + 1) * count])
                    metrics = worker.forward_backward(shard, return_per_token_outputs=False).metrics
                    norm = worker.optim_step()
                    updates.append({"epoch": epoch, "grad_norm": norm, "metrics": metrics})
            checkpoint = args.output / f"lr_{learning_rate:g}"
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            strategy.save_checkpoint(model, str(checkpoint / "policy"), rank, optimizer, scheduler)
            emit(f"lr_{learning_rate:g}", {"checkpoint": str(checkpoint), "updates": updates})
        emit("complete", True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if args.mode == "score":
        after = forward(worker, data)
        assert data["advantages"].ndim == 2, "Use the complete TOP-D arm's unscaled input"
        trainer.stopd = None
        trainer._normalize_advantages(data, [(0, len(data))])
        plain = copy.deepcopy(cfg.trainer.algorithm)
        plain.policy_loss_type = "regular"
        before_loss, _ = ppo_policy_loss(
            data["action_log_probs"], data["action_log_probs"], data["advantages"], plain, loss_mask=data["loss_mask"]
        )
        after_loss, _ = ppo_policy_loss(
            after, data["action_log_probs"], data["advantages"], plain, loss_mask=data["loss_mask"]
        )
        mask = data["loss_mask"].bool()
        delta = after - data["action_log_probs"]
        old_kl = data["action_log_probs"] - data["base_action_log_probs"]
        change = delta.exp() * (after - data["base_action_log_probs"] - 1) + 1 - old_kl
        permutation = data.metadata.get("batch_permutation", range(len(data)))
        domains = [rows[i]["env_extras"]["data_source"] for i in permutation]
        domain_metrics = {}
        for domain in sorted(set(domains)):
            selected = mask & torch.tensor([d == domain for d in domains]).unsqueeze(1)
            domain_metrics[domain] = {
                "tokens": int(selected.sum()),
                "conditional_kl_surrogate_change": change[selected].mean().item(),
                "mean_abs_logprob_change": delta[selected].abs().mean().item(),
            }
        emit(
            "fixed_complete_trajectory",
            {
                "before_loss": before_loss.item(),
                "after_loss": after_loss.item(),
                "improvement": before_loss.item() - after_loss.item(),
                "conditional_kl_surrogate_change": change[mask].mean().item(),
                "domains": domain_metrics,
            },
        )
        if rank == 0:
            torch.save(after, args.output / "after_logprobs.pt")
        emit("complete", True)
        dist.barrier()
        dist.destroy_process_group()
        return
    # With reshard_after_forward=False, a scoring forward exposes temporary
    # BF16 parameters on modules. The optimizer retains the FP32 leaf DTensors
    # receiving reduced gradients and updates; use those stable identities.
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    names = [parameter_names[id(p)] for p in parameters]
    data["action_log_probs"] = forward(worker, data)
    emit(
        "parameters_after_scoring",
        {
            "optimizer_parameters": len(parameters),
            "optimizer_leaves_absent_from_module_view": len(
                {id(p) for p in parameters} - {id(p) for p in model.parameters()}
            ),
        },
    )
    trainer.apply_reward_kl_penalty(data)
    if args.mode == "control":
        compare_learning_updates(trainer, worker, data, restore, forward, emit, rank, world)
        emit("complete", True)
        dist.barrier()
        dist.destroy_process_group()
        return
    trainer.compute_advantages_and_returns(data)
    trainer._normalize_advantages(data, data.metadata["policy_mini_batch_boundaries"])
    initial = [local_tensor(p).detach().cpu().clone() for p in parameters] if rank == 0 else None
    if rank == 0:
        torch.save(data, args.output / "scored.pt")
    emit("target_metrics", trainer.all_metrics)
    mask = data["loss_mask"].bool()
    response_ids = data["sequences"][:, -mask.shape[1] :]
    positions = data["response_mask"].long().cumsum(-1)
    coefficient = data["advantages"].sum(-1)
    regions = {"eos": mask & (response_ids == 0), "other": mask & (response_ids != 0)}
    regions.update(
        {
            f"tokens_{a}_{b}": mask & (positions > a) & (positions <= b)
            for a, b in ((0, 128), (128, 512), (512, 2048), (2048, 4096), (4096, 8192))
        }
    )
    emit(
        "token_regions",
        {
            name: {
                "tokens": int(selected.sum()),
                "reward_mean": data["rewards"][selected].mean().item(),
                "positive_gradient_coefficient_fraction": (coefficient[selected] > 0).float().mean().item(),
                "absolute_coefficient_sum": coefficient[selected].abs().sum().item(),
            }
            for name, selected in regions.items()
            if selected.any()
        },
    )

    if rank == 0:
        ref_loss, ref_derivative = reference_loss_gradient(data, data["action_log_probs"], cfg)
        current = data["action_log_probs"].detach().requires_grad_()
        loss, _ = topd_stopd_policy_loss(
            current, data["action_log_probs"], data["advantages"], cfg.trainer.algorithm, loss_mask=data["loss_mask"]
        )
        derivative = torch.autograd.grad(loss, current)[0]
        emit(
            "logprob_gradient",
            compare_tensors([derivative], [ref_derivative]) | {"native_loss": loss.item(), "reference_loss": ref_loss},
        )
    else:
        ref_derivative = torch.empty_like(data["action_log_probs"])
    reference_on_device = ref_derivative.to("cuda")
    dist.broadcast(reference_on_device, src=0)
    ref_derivative = reference_on_device.cpu()
    del reference_on_device

    def backward(batch, loss_fn):
        optimizer.zero_grad(set_to_none=True)
        worker.policy_loss_fn = loss_fn
        status = worker.forward_backward(
            copy.deepcopy(batch[rank * per_rank : (rank + 1) * per_rank]), return_per_token_outputs=False
        )
        missing = [name for name, p in zip(names, parameters) if p.grad is None]
        nonfinite = [
            name
            for name, p in zip(names, parameters)
            if p.grad is not None and not torch.isfinite(local_tensor(p.grad)).all()
        ]
        emit("gradient_presence", {"missing": missing, "nonfinite": nonfinite})
        assert not missing and not nonfinite, (missing, nonfinite)
        return status.metrics

    emit("native_backward", backward(data, topd_stopd_policy_loss))
    native_gradients = [local_tensor(p.grad).detach().cpu().clone() for p in parameters] if rank == 0 else None
    if rank == 0:
        norms = {name: gradient.double().norm().item() for name, gradient in zip(names, native_gradients)}
        emit("parameter_gradient_norms", norms)
    emit("native_grad_norm", worker.optim_step())
    native_parameters = [local_tensor(p).detach().cpu().clone() for p in parameters] if rank == 0 else None
    after = forward(worker, data)
    if rank == 0:
        native_objective, _ = topd_stopd_policy_loss(
            after, data["action_log_probs"], data["advantages"], cfg.trainer.algorithm, loss_mask=data["loss_mask"]
        )
        exact_after, _ = reference_loss_gradient(data, after, cfg)
        direction = compare_tensors(
            [p - before for p, before in zip(native_parameters, initial)], [-g for g in native_gradients]
        )
        emit(
            "native_update",
            {
                "delta_norm": direction["actual_norm"],
                "negative_gradient_cosine": direction["cosine"],
                "post_loss": native_objective.item(),
                "post_reference_loss": exact_after,
                "mean_abs_logprob_change": (after - data["action_log_probs"])[data["loss_mask"].bool()]
                .abs()
                .mean()
                .item(),
            },
        )
        torch.save(after, args.output / "native_after_logprobs.pt")

    restore()
    independent = copy.deepcopy(data)
    independent["advantages"] = -ref_derivative
    emit("reference_backward", backward(independent, linear_reference_loss))
    if rank == 0:
        emit("parameter_gradient", compare_tensors([local_tensor(p.grad) for p in parameters], native_gradients))
    emit("reference_grad_norm", worker.optim_step())
    if rank == 0:
        emit(
            "optimizer_update",
            compare_tensors(
                [local_tensor(p).detach().cpu() - before for p, before in zip(parameters, initial)],
                [p - before for p, before in zip(native_parameters, initial)],
            ),
        )
    emit("complete", True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
