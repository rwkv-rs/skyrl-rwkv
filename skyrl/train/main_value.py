"""Offline critic pretraining with SkyRL's native critic and FSDP2 strategy.

Example: uv run --isolated --no-project .venv/bin/torchrun --standalone \
    --nproc-per-node=8 -m skyrl.train.main_value --model /path/to/rwkv7-g1k-1.5b-20260930-ctx25600 \
    --train-data train.jsonl --val-data validation.jsonl --output-dir checkpoints/value

Dataset/collation/alignment follows ReDiPPO's standard critic pretraining;
training, optimizer, and checkpoint handling reuse SkyRL FSDPStrategy.
"""

import argparse
import math
import os
from itertools import islice

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    get_llm_for_sequence_regression,
)
from skyrl.train.config import FSDPConfig, OptimizerConfig
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
def evaluate(model, dataset, collator, batch_size, device):
    """Exact held-out metrics; dummy tail rows keep FSDP collective counts equal."""
    model.eval()
    stats = torch.zeros(8, dtype=torch.float64, device=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    for start in range(0, len(dataset), world_size * batch_size):
        indices = [start + rank * batch_size + index for index in range(batch_size)]
        batch = collator([dataset[index] if index < len(dataset) else dataset[0] for index in indices])
        for row, index in enumerate(indices):
            if index >= len(dataset):
                batch["response_mask"][row] = False
        predictions, labels, mask = critic_forward(model, batch, device)
        residual = (labels.float() - predictions.float()).double()
        targets = labels.double().masked_select(mask)
        errors = residual.square().masked_fill(~mask, 0)
        counts = mask.sum(-1)
        sequence_mse = errors.sum(-1) / counts.clamp_min(1)
        stats += torch.stack(
            [
                errors.sum(),
                mask.sum(),
                targets.sum(),
                targets.square().sum(),
                residual.masked_select(mask).sum(),
                sequence_mse.masked_select(counts > 0).sum(),
                (counts > 0).sum(),
                predictions.double().masked_select(mask).sum(),
            ]
        )
    dist.all_reduce(stats)
    error, count, target_sum, target_sq, residual_sum, sequence_error, sequences, prediction_sum = stats.tolist()
    target_var = target_sq / count - (target_sum / count) ** 2
    residual_var = error / count - (residual_sum / count) ** 2
    model.train()
    return {
        "eval/token_mse": error / count,
        "eval/sequence_mse": sequence_error / sequences,
        "eval/explained_variance": 1 - residual_var / target_var if target_var > 0 else float("nan"),
        "eval/target_mean": target_sum / count,
        "eval/value_mean": prediction_sum / count,
    }


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
        sampler = DistributedSampler(train_dataset, seed=args.seed)
        loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collator)
        steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
        strategy = FSDPStrategy(
            FSDPConfig(),
            optimizer_config=OptimizerConfig(lr=args.lr, offload_after_step=False),
            seed=args.seed,
            micro_train_batch_size_per_gpu=args.batch_size,
            num_training_steps=steps_per_epoch * args.epochs,
        )
        strategy.setup_distributed()
        model = get_llm_for_sequence_regression(args.model, "critic", bf16=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model, optimizer, scheduler = strategy.prepare((model, None, None))
        if dist.get_rank() == 0 and args.report_to == "wandb":
            import wandb

            run = wandb.init(project=args.wandb_project, config=vars(args))

        best_mse = math.inf

        def validate_and_save(step, metrics):
            nonlocal best_mse
            metrics.update(evaluate(model, val_dataset, collator, args.batch_size, device))
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
            strategy.save_hf_model(model, os.path.join(checkpoint, "huggingface"), tokenizer=tokenizer)
            if metrics["eval/sequence_mse"] < best_mse:
                best_mse = metrics["eval/sequence_mse"]
                strategy.save_hf_model(model, os.path.join(args.output_dir, "best"), tokenizer=tokenizer)

        step = 0
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            while batches := list(islice(iterator, args.gradient_accumulation_steps)):
                # FSDP averages rank gradients. Normalize over all action tokens
                # in the entire accumulation window, not each microbatch/rank.
                count = torch.tensor(sum(int(batch["response_mask"].sum()) for batch in batches), device=device)
                dist.all_reduce(count)
                error_sum = torch.zeros((), device=device)
                for batch in batches:
                    predictions, labels, mask = critic_forward(model, batch, device)
                    error, _ = value_loss(predictions, labels, mask)
                    strategy.backward(error * dist.get_world_size() / count, model, optimizer)
                    error_sum += error.detach()
                grad_norm = strategy.optimizer_step(optimizer, model, scheduler)
                dist.all_reduce(error_sum)
                step += 1
                metrics = {
                    "train/mse": (error_sum / count).item(),
                    "train/epoch": epoch + 1,
                    "train/action_tokens": count.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    metrics["train/grad_norm"] = grad_norm.detach().cpu().item()
                if step % args.save_steps == 0 or step == steps_per_epoch * args.epochs:
                    validate_and_save(step, metrics)
                if dist.get_rank() == 0:
                    logger.info(f"step={step} {metrics}")
                    if run is not None:
                        run.log(metrics, step=step)
    finally:
        if run is not None:
            run.finish()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="rwkv-value-pretraining")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "save_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_length < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--max-length must be >= 2 and --lr must be finite and positive.")
    train(args)


if __name__ == "__main__":
    main()
