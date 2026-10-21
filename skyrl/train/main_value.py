"""Offline critic pretraining with SkyRL's native critic and FSDP2 strategy.

Example: uv run --isolated --no-project .venv/bin/torchrun --standalone \
    --nproc-per-node=8 -m skyrl.train.main_value --model /path/to/rwkv7-g1k-1.5b-20260930-ctx25600 \
    --train-data train.jsonl --val-data validation.jsonl --output-dir checkpoints/value

Dataset/collation/alignment follows ReDiPPO's standard critic pretraining;
training, optimizer, and checkpoint handling reuse SkyRL FSDPStrategy.
"""

import argparse
import json
import math
import os
import shutil
import time
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    get_llm_for_sequence_regression,
)
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig, SkyRLLoraConfig
from skyrl.train.dataset.value_dataset import (
    RewardDataCollator,
    ValueDataset,
    value_loss,
)
from skyrl.utils.tok import get_tokenizer


def critic_forward(model, batch, device):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    predictions = model(batch["input_ids"], num_actions=batch["num_actions"], attention_mask=batch["attention_mask"])
    return predictions, batch["labels"], batch["response_mask"]


@torch.no_grad()
def evaluate(model, dataset, collator, batch_size, device, domains=None):
    """Exact held-out metrics; dummy tail rows keep FSDP collective counts equal."""
    model.eval()
    groups = [None, *sorted(set(domains))] if domains is not None else [None]
    stats = torch.zeros((len(groups), 15), dtype=torch.float64, device=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    for start in range(0, len(dataset), world_size * batch_size):
        indices = [start + rank * batch_size + index for index in range(batch_size)]
        batch = collator([dataset[index] if index < len(dataset) else dataset[0] for index in indices])
        for row, index in enumerate(indices):
            if index >= len(dataset):
                batch["response_mask"][row] = False
        predictions, labels, mask = critic_forward(model, batch, device)
        residual = (labels.float() - predictions.float()).double()
        for group_index, domain in enumerate(groups):
            group_mask = mask
            if domain is not None:
                selected = torch.tensor(
                    [index < len(dataset) and domains[index] == domain for index in indices], device=device
                )
                group_mask = mask & selected.unsqueeze(-1)
            targets = labels.double().masked_select(group_mask)
            errors = residual.square().masked_fill(~group_mask, 0)
            counts = group_mask.sum(-1)
            valid = counts > 0
            sequence_mse = errors.sum(-1) / counts.clamp_min(1)
            first_action = group_mask.long().argmax(-1, keepdim=True)
            initial_predictions = predictions.double().gather(-1, first_action).squeeze(-1).masked_select(valid)
            initial_targets = labels.double().gather(-1, first_action).squeeze(-1).masked_select(valid)
            stats[group_index] += torch.stack(
                [
                    errors.sum(),
                    group_mask.sum(),
                    targets.sum(),
                    targets.square().sum(),
                    residual.masked_select(group_mask).sum(),
                    sequence_mse.masked_select(valid).sum(),
                    valid.sum(),
                    predictions.double().masked_select(group_mask).sum(),
                    (labels.double().masked_fill(~group_mask, 0).sum(-1) / counts.clamp_min(1)).sum(),
                    (labels.double().square().masked_fill(~group_mask, 0).sum(-1) / counts.clamp_min(1)).sum(),
                    (predictions.double().masked_fill(~group_mask, 0).sum(-1) / counts.clamp_min(1)).sum(),
                    (initial_predictions - initial_targets).square().sum(),
                    initial_predictions.sum(),
                    initial_targets.sum(),
                    initial_targets.square().sum(),
                ]
            )
    dist.all_reduce(stats)
    metrics = {}
    for domain, values in zip(groups, stats.tolist()):
        (
            error,
            count,
            target_sum,
            target_sq,
            residual_sum,
            sequence_error,
            sequences,
            prediction_sum,
            sequence_target_sum,
            sequence_target_sq,
            sequence_prediction_sum,
            initial_error,
            initial_prediction_sum,
            initial_target_sum,
            initial_target_sq,
        ) = values
        target_var = target_sq / count - (target_sum / count) ** 2
        sequence_target_var = sequence_target_sq / sequences - (sequence_target_sum / sequences) ** 2
        initial_target_var = initial_target_sq / sequences - (initial_target_sum / sequences) ** 2
        residual_var = error / count - (residual_sum / count) ** 2
        prefix = "eval" if domain is None else f"eval/{domain}"
        metrics.update(
            {
                f"{prefix}/token_mse": error / count,
                f"{prefix}/sequence_mse": sequence_error / sequences,
                f"{prefix}/explained_variance": 1 - residual_var / target_var if target_var > 0 else float("nan"),
                f"{prefix}/target_mean": target_sum / count,
                f"{prefix}/value_mean": prediction_sum / count,
                f"{prefix}/token_constant_mse": target_var,
                f"{prefix}/token_mse_skill": 1 - error / count / target_var if target_var > 0 else float("nan"),
                f"{prefix}/sequence_constant_mse": sequence_target_var,
                f"{prefix}/sequence_mse_skill": (
                    1 - sequence_error / sequences / sequence_target_var if sequence_target_var > 0 else float("nan")
                ),
                f"{prefix}/sequence_target_mean": sequence_target_sum / sequences,
                f"{prefix}/sequence_value_mean": sequence_prediction_sum / sequences,
                f"{prefix}/initial_mse": initial_error / sequences,
                f"{prefix}/initial_mse_skill": (
                    1 - initial_error / sequences / initial_target_var if initial_target_var > 0 else float("nan")
                ),
                f"{prefix}/initial_target_mean": initial_target_sum / sequences,
                f"{prefix}/initial_value_mean": initial_prediction_sum / sequences,
                f"{prefix}/sequences": sequences,
            }
        )
    model.train()
    return metrics


def cleanup_old_checkpoints(checkpoint_base_path, max_checkpoints):
    """Retain recent value checkpoints, following the native trainer's retention policy."""
    if max_checkpoints < 0:
        return
    checkpoint_dirs = sorted(Path(checkpoint_base_path).glob("step_*"), key=lambda path: int(path.name[5:]))
    for path in checkpoint_dirs[:-max_checkpoints]:
        shutil.rmtree(path)
        logger.info(f"Cleaned up old checkpoint: {path}")


def train(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    run = None
    try:
        tokenizer = get_tokenizer(args.model)
        collator = RewardDataCollator(tokenizer.pad_token_id)
        train_dataset = ValueDataset(args.train_data, args.max_length)
        val_dataset = ValueDataset(args.val_data, args.max_length)
        val_domains = list(val_dataset.dataset["domain"]) if "domain" in val_dataset.dataset.column_names else None
        if args.group_by_length:
            sampler = DistributedLengthGroupedSampler(
                batch_size=args.batch_size * args.gradient_accumulation_steps * dist.get_world_size(),
                lengths=[len(ids) for ids in train_dataset.dataset["input_ids"]],
                seed=args.seed,
            )
        else:
            sampler = DistributedSampler(train_dataset, seed=args.seed)
        loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collator)
        steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
        lora_config = SkyRLLoraConfig(
            rank=args.rank if args.use_nora else 0,
            alpha=args.lora_alpha,
            dropout=0.0,
            use_nora=args.use_nora,
            target_modules=args.target_modules,
        )
        strategy = FSDPStrategy(
            FSDPConfig(fsdp_size=args.fsdp_size),
            optimizer_config=OptimizerConfig(lr=args.lr, offload_after_step=False),
            model_config=ModelConfig(path=args.model, lora=lora_config),
            seed=args.seed,
            micro_train_batch_size_per_gpu=args.batch_size,
            num_training_steps=steps_per_epoch * args.epochs,
        )
        strategy.setup_distributed()
        model = get_llm_for_sequence_regression(
            args.model,
            "critic",
            bf16=True,
            lora_rank=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            use_nora=lora_config.use_nora,
            model_config_kwargs=args.model_config_kwargs,
        )
        if model.config.model_type != "rwkv":
            model.config.pad_token_id = tokenizer.pad_token_id
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        logger.info(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model, optimizer, scheduler = strategy.prepare((model, None, None))
        if args.resume_path is not None:
            strategy.load_checkpoint(
                model,
                args.resume_path,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
        if dist.get_rank() == 0 and args.report_to == "wandb":
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    **vars(args),
                    "trainable_parameters": trainable_parameters,
                    "total_parameters": total_parameters,
                },
            )

        best_mse = math.inf

        def validate_and_save(step, metrics):
            nonlocal best_mse
            metrics.update(evaluate(model, val_dataset, collator, args.batch_size, device, domains=val_domains))
            checkpoint = os.path.join(args.output_dir, f"step_{step}")
            strategy.save_checkpoint(
                model,
                checkpoint,
                int(os.environ["LOCAL_RANK"]),
                optimizer=optimizer,
                scheduler=scheduler,
                client_state={"global_step": step, **metrics},
                tokenizer=tokenizer,
            )
            strategy.save_hf_model(
                model, os.path.join(checkpoint, "huggingface"), tokenizer=tokenizer, output_dtype=torch.bfloat16
            )
            if metrics["eval/sequence_mse"] < best_mse:
                best_mse = metrics["eval/sequence_mse"]
                strategy.save_hf_model(
                    model, os.path.join(args.output_dir, "best"), tokenizer=tokenizer, output_dtype=torch.bfloat16
                )
            if dist.get_rank() == 0:
                cleanup_old_checkpoints(args.output_dir, args.max_ckpts_to_keep)
            dist.barrier()

        step = 0
        if args.eval_before_train:
            metrics = {}
            validate_and_save(step, metrics)
            if dist.get_rank() == 0:
                logger.info(f"step={step} {metrics}")
                if run is not None:
                    run.log(metrics, step=step)
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            while batches := list(islice(iterator, args.gradient_accumulation_steps)):
                started = time.perf_counter()
                # FSDP averages rank gradients. Normalize over the entire
                # accumulation window, not each microbatch/rank.
                counts = torch.tensor(
                    [
                        sum(int(batch["response_mask"].sum()) for batch in batches),
                        sum(int(batch["attention_mask"].sum()) for batch in batches),
                        sum(batch["input_ids"].numel() for batch in batches),
                        sum(int(batch["response_mask"].any(-1).sum()) for batch in batches),
                    ],
                    device=device,
                )
                dist.all_reduce(counts)
                count = counts[3] if args.loss_reduction == "sequence_mean" else counts[0]
                error_sum = torch.zeros((), device=device)
                for batch in batches:
                    predictions, labels, mask = critic_forward(model, batch, device)
                    error, _ = value_loss(predictions, labels, mask, args.loss_reduction)
                    # The online PPO critic minimizes half the sequence-mean MSE.
                    loss_scale = 0.5 if args.loss_reduction == "sequence_mean" else 1.0
                    strategy.backward(error * loss_scale * dist.get_world_size() / count, model, optimizer)
                    error_sum += error.detach()
                grad_norm = strategy.optimizer_step(optimizer, model, scheduler)
                dist.all_reduce(error_sum)
                step += 1
                metrics = {
                    "train/mse": (error_sum / count).item(),
                    "train/epoch": epoch + 1,
                    "train/action_tokens": counts[0].item(),
                    "train/sequences": counts[3].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                seconds = time.perf_counter() - started
                metrics.update(
                    {
                        "timing/train_step": seconds,
                        "train/input_tokens": counts[1].item(),
                        "train/tokens_per_second": counts[1].item() / seconds,
                        "train/padding_fraction": 1 - counts[1].item() / counts[2].item(),
                    }
                )
                if grad_norm is not None:
                    metrics["train/grad_norm"] = grad_norm.detach().cpu().item()
                if step % args.save_steps == 0 or step == steps_per_epoch * args.epochs:
                    validation_started = time.perf_counter()
                    validate_and_save(step, metrics)
                    metrics["timing/eval_and_save"] = time.perf_counter() - validation_started
                if dist.get_rank() == 0:
                    logger.info(f"step={step} {metrics}")
                    if run is not None:
                        run.log(metrics, step=step)
    finally:
        if run is not None:
            run.finish()
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume-path", default=None, help="Initialize weights from a native value checkpoint")
    parser.add_argument("--model-config-kwargs", type=json.loads, default={})
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--loss-reduction", choices=("token_mean", "sequence_mean"), default="token_mean")
    parser.add_argument("--max-ckpts-to-keep", type=int, default=-1, help="Keep the last N checkpoints; -1 keeps all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--group-by-length", action="store_true", help="Group similar lengths across data-parallel ranks"
    )
    parser.add_argument("--use-nora", action="store_true", help="Train NoRA backbone adapters and the full value head")
    parser.add_argument("--rank", type=int, default=8, help="NoRA rank; sweep 8, 16, 32, 64, ...")
    parser.add_argument("--lora-alpha", type=int, default=None, help="NoRA alpha; defaults to rank")
    parser.add_argument("--target-modules", nargs="+", default=["r_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument(
        "--fsdp-size", type=int, default=-1, help="FSDP shard size; -1 uses all ranks, 1 replicates the base"
    )
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="rwkv-value-pretraining")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if not isinstance(args.model_config_kwargs, dict):
        parser.error("--model-config-kwargs must be a JSON object.")
    if args.max_ckpts_to_keep != -1 and args.max_ckpts_to_keep <= 0:
        parser.error("--max-ckpts-to-keep must be -1 or positive.")
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "save_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_length < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--max-length must be >= 2 and --lr must be finite and positive.")
    if args.rank <= 0 or (args.lora_alpha is not None and args.lora_alpha <= 0):
        parser.error("--rank and --lora-alpha must be positive.")
    if args.fsdp_size != -1 and args.fsdp_size <= 0:
        parser.error("--fsdp-size must be -1 or positive.")
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    train(args)


if __name__ == "__main__":
    main()
