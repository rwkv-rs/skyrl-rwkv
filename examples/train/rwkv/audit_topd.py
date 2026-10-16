"""Replay saved OPD rollouts through native FSDP and an independent TOP-D loss.

Run with torchrun on an idle training node. Model and Adam state are restored
from a complete checkpoint; diagnostic updates are never saved as training
checkpoints. The native worker and benchmark_stopd provide the execution paths.
"""

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
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
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.workers.worker import RefWorkerBase


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
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

    rows = []
    with args.rollouts.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
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
        strategy.load_checkpoint(model, str(args.checkpoint / "policy"), optimizer, scheduler)
        optimizer.zero_grad(set_to_none=True)

    restore()
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker.cfg, worker.strategy, worker.model = cfg.trainer, strategy, model
    worker.optimizer, worker.scheduler = optimizer, scheduler
    worker.mesh_rank = SimpleNamespace(dp_size=world)
    worker.device_mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("dp", "sp"))
    worker.policy_loss_fn = topd_stopd_policy_loss
    worker._set_expandable_segments(True)
    data["action_log_probs"] = forward(worker, data)
    trainer.apply_reward_kl_penalty(data)
    trainer.compute_advantages_and_returns(data)
    trainer._normalize_advantages(data, data.metadata["policy_mini_batch_boundaries"])
    parameters = [p for p in model.parameters() if p.requires_grad]
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
        assert all(p.grad is not None and torch.isfinite(local_tensor(p.grad)).all() for p in parameters)
        return status.metrics

    emit("native_backward", backward(data, topd_stopd_policy_loss))
    native_gradients = [local_tensor(p.grad).detach().cpu().clone() for p in parameters] if rank == 0 else None
    if rank == 0:
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
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
        emit(
            "native_update",
            compare_tensors(
                [p - before for p, before in zip(native_parameters, initial)], [-g for g in native_gradients]
            )
            | {
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
