"""Compare native Vanilla OPD and STOPD from one complete checkpoint.

Replays the checkpoint's configuration, model/optimizer state, and dataloader
position for both arms. Optionally wait for a running baseline to save, stop
that process tree, and continue production from the validated STOPD checkpoint.
"""

import argparse
import ast
import copy
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import time
from pathlib import Path

import psutil
import torch

from examples.train.on_policy_distillation.stopd import STOPD


def load_checkpoint(checkpoint):
    with (checkpoint / "trainer_state.pt").open("rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    step = state["global_step"]
    if checkpoint.name != f"global_step_{step}":
        raise ValueError("Checkpoint path and trainer step must agree")
    if not (checkpoint / "data.pt").is_file() or not list((checkpoint / "policy").glob("optim_world_size_*_rank_*.pt")):
        raise ValueError("A complete checkpoint with dataloader and optimizer state is required")
    return state


def read_rollouts(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def denominator_from_rollouts(rows):
    return sum(sum(row["loss_mask"]) for row in rows)


def stop_baseline(pid):
    process = psutil.Process(pid)
    if not any("run_rwkv_on_policy_distill.sh" in arg for arg in process.cmdline()):
        raise ValueError("baseline-pid must identify the current RWKV OPD launcher")
    children = process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    _, alive = psutil.wait_procs([process, *children], timeout=30)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def run_config(project, config, output, mode, checkpoint, start_step, steps, block, seed, denominator):
    cfg = copy.deepcopy(config)
    trainer = cfg["trainer"]
    trainer.update(
        resume_mode="from_path",
        resume_path=str(checkpoint),
        max_training_steps=start_step + steps,
        eval_before_train=False,
        eval_interval=-1,
        dump_train_results=True,
        dump_data_batch=True,
        num_logger_train_samples=0,
        logger="console",
        run_name=f"{Path(trainer['policy']['model']['path']).name}-ultradata-opd-nora-{mode}-from-step-{start_step}",
        log_path=str(output / "logs"),
        ckpt_path=str(output / "checkpoints"),
        export_path=str(output / "exports"),
    )
    trainer["algorithm"]["stopd"] = dict(
        enabled=True,
        block=block,
        seed=seed,
        initial_denominator=denominator,
        warmup_steps=start_step + steps + 1 if mode == "vanilla" else 25,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    command = ["bash", str(project / "examples/train/rwkv/run_rwkv_on_policy_distill.sh")]
    command.extend(f"{key}={json.dumps(value)}" for key, value in cfg.items())
    env = os.environ | {"RAY_ADDRESS": "local", "PYTHONPATH": f"{project}:{project / 'skyrl-gym'}"}
    print(f"Starting {mode}: checkpoint step {start_step}, {steps} updates", flush=True)
    with (output / "console.log").open("w") as handle:
        subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return cfg


def summarize(output, steps):
    text = re.sub(r"\x1b\[[0-9;]*m", "", (output / "console.log").read_text())
    timings = {}
    for stage in ("step", "generate", "fwd_logprobs_values_reward", "policy_train", "sync_weights"):
        values = list(map(float, re.findall(r"Finished: '" + stage + r"', time cost: ([0-9.]+)s", text)))[-steps:]
        if len(values) != steps:
            raise ValueError(f"Missing {stage} measurements in {output}")
        timings[stage] = {"values": values, "mean": statistics.mean(values), "median": statistics.median(values)}
    statuses = [ast.literal_eval(value) for value in re.findall(r"(\{'policy_entropy':[^\n]*\})", text)]
    if len(statuses) != steps or any(
        not math.isfinite(row[key]) for row in statuses for key in ("final_loss", "grad_norm")
    ):
        raise ValueError(f"Incomplete or non-finite training updates in {output}")
    # Native backend workers write to infrastructure logs; the trainer logs to
    # the console. Verbose infrastructure mode puts both on the console.
    restored = text + "\n".join(path.read_text() for path in (output / "logs").glob("infra-*.log"))
    optimizer_restored = any(
        message in restored
        for message in (
            "Successfully loaded optimizer state",
            "Checkpoint weights, optimizer and scheduler restored on ",
        )
    )
    if not optimizer_restored or "Successfully loaded dataloader state" not in text:
        raise ValueError(f"The benchmark did not restore complete training state in {output}")
    return {"timings": timings, "updates": statuses}


def validate_training_input(output, rows, stopd, step):
    with (output / "exports/dumped_data" / f"global_step_{step}_training_input.pkl").open("rb") as handle:
        data = pickle.load(handle)
    for index, row in enumerate(rows):
        tokens = data["sequences"][index][data["attention_mask"][index].bool()].tolist()
        if tokens != row["prompt_token_ids"] + row["response_token_ids"]:
            raise ValueError("Trainer sequence tokens differ from the generated prompt and response")
        mask = data["response_mask"][index].bool()
        if data["loss_mask"][index][mask].tolist() != row["loss_mask"]:
            raise ValueError("Trainer response loss mask differs from the generated prefix mask")
        torch.testing.assert_close(
            data["rollout_logprobs"][index][mask],
            torch.tensor(row["rollout_logprobs"], dtype=data["rollout_logprobs"].dtype),
            rtol=0,
            atol=0,
        )
    mask = data["response_mask"].bool() & data["loss_mask"].bool()
    pad_size = data.metadata.get("pad_size", 0)
    if pad_size:
        mask[-pad_size:] = False
    expected_rewards = -(data["action_log_probs"] - data["base_action_log_probs"]) * data["loss_mask"]
    torch.testing.assert_close(data["returns"], expected_rewards, rtol=0, atol=0)
    expected_advantages = expected_rewards
    if stopd.truncating:
        expected_advantages = expected_rewards * stopd.multiplier(mask, data["loss_mask"].sum().item())
    torch.testing.assert_close(data["advantages"], expected_advantages, rtol=0, atol=0)
    if not torch.isfinite(data["advantages"]).all():
        raise ValueError("Native OPD advantages must be finite")
    return {"rows": len(rows), "response_tokens": int(mask.sum()), "ipw_advantages_exact": True}


def validate_rollouts(vanilla, truncated, max_len, block, step, denominator, seed):
    full = read_rollouts(vanilla / "exports/dumped_rollouts" / f"global_step_{step}.jsonl")
    short = read_rollouts(truncated / "exports/dumped_rollouts" / f"global_step_{step}.jsonl")
    if [row["trajectory_id"] for row in full] != [row["trajectory_id"] for row in short]:
        raise ValueError("Both benchmark arms must resume the same dataloader position and row order")
    stopd = STOPD(max_len, block, 25, denominator, seed)
    stopd.step = step - 1
    budgets = stopd.budgets(len(short))
    for original, retained, budget in zip(full, short, budgets, strict=True):
        tokens = retained["response_token_ids"]
        if original["prompt_token_ids"] != retained["prompt_token_ids"]:
            raise ValueError("Benchmark prompt tokens differ")
        if len(tokens) != len(retained["rollout_logprobs"]) or retained["loss_mask"] != [1] * len(tokens):
            raise ValueError("Truncated tokens, logprobs, and loss masks must remain aligned")
        if len(tokens) > budget or (retained["truncated"] and len(tokens) != budget):
            raise ValueError("Truncated rollout did not end at its block budget")
    native = validate_training_input(truncated, short, stopd, step)
    vanilla_stopd = STOPD(max_len, block, step + 1, denominator, seed)
    vanilla_stopd.step = step - 1
    native_full = validate_training_input(vanilla, full, vanilla_stopd, step)
    return {
        "full_response_tokens": sum(len(row["response_token_ids"]) for row in full),
        "truncated_response_tokens": sum(len(row["response_token_ids"]) for row in short),
        "native_stopd_alignment": native,
        "native_vanilla_alignment": native_full,
    }


def validate_stopd_state(output, final_step):
    checkpoint = output / "checkpoints" / f"global_step_{final_step}"
    with (checkpoint / "stopd_state.pt").open("rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    if state["step"] != final_step:
        raise ValueError("STOPD checkpoint must retain the completed global step")
    rows = read_rollouts(output / "exports/dumped_rollouts" / f"global_step_{final_step}.jsonl")
    width = max(len(row["loss_mask"]) for row in rows)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for index, row in enumerate(rows):
        mask[index, : len(row["loss_mask"])] = torch.tensor(row["loss_mask"], dtype=torch.bool)
    stopd = STOPD(state["max_len"], state["block"], state["warmup_steps"], 1, state["seed"])
    stopd.load_state_dict(state)
    estimate = stopd.weights(mask).double().sum().item()
    if not math.isclose(state["denominator"], estimate, rel_tol=1e-7):
        raise ValueError("Saved lagged denominator must equal the last step's IPW token-count estimate")
    return checkpoint


def continue_training(project, original, checkpoint, output):
    cfg = copy.deepcopy(original)
    cfg["trainer"].update(
        resume_mode="from_path",
        resume_path=str(checkpoint),
        max_training_steps=None,
        eval_before_train=False,
        logger="wandb",
        run_name=output.name,
        log_path=str(output / "logs"),
        ckpt_path=str(output / "checkpoints"),
        export_path=str(output / "exports"),
    )
    with (checkpoint / "stopd_state.pt").open("rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    cfg["trainer"]["algorithm"]["stopd"] = dict(
        enabled=True,
        block=state["block"],
        seed=state["seed"],
        warmup_steps=state["warmup_steps"],
        initial_denominator=None,
    )
    output.mkdir(parents=True, exist_ok=False)
    command = ["bash", str(project / "examples/train/rwkv/run_rwkv_on_policy_distill.sh")]
    command.extend(f"{key}={json.dumps(value)}" for key, value in cfg.items())
    env = os.environ | {"RAY_ADDRESS": "local", "PYTHONPATH": f"{project}:{project / 'skyrl-gym'}"}
    with (output / "console.log").open("w") as handle:
        process = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (output / "launcher.pid").write_text(str(process.pid) + "\n")
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"pid": process.pid, "checkpoint": str(checkpoint), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--baseline-pid", type=int)
    parser.add_argument("--baseline-log", type=Path, help="Wait for checkpoint-step evaluation and logging to finish")
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.block < 1:
        parser.error("--steps and --block must be positive")
    project = Path(__file__).resolve().parents[3]
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    latest = args.checkpoint.parent / "latest_ckpt_global_step.txt"
    while not (args.checkpoint / "trainer_state.pt").exists() or (
        args.wait
        and (not latest.exists() or latest.read_text().strip() != args.checkpoint.name.removeprefix("global_step_"))
    ):
        if not args.wait:
            raise FileNotFoundError(args.checkpoint)
        if args.baseline_pid and not psutil.pid_exists(args.baseline_pid):
            raise RuntimeError("Baseline exited before saving the requested checkpoint")
        time.sleep(30)
    saved = load_checkpoint(args.checkpoint)
    original, step = saved["config"], saved["global_step"]
    if args.block > original["generator"]["sampling_params"]["max_generate_length"]:
        parser.error("--block must not exceed the checkpoint's max_generate_length")
    # The export's atomic rename happens after the complete checkpoint save.
    export_config = Path(original["trainer"]["export_path"]) / f"global_step_{step}/policy/config.json"
    while args.wait and not export_config.exists():
        if args.baseline_pid and not psutil.pid_exists(args.baseline_pid):
            raise RuntimeError("Baseline exited before completing the model export")
        time.sleep(5)
    rows = read_rollouts(Path(original["trainer"]["export_path"]) / "dumped_rollouts" / f"global_step_{step}.jsonl")
    denominator = denominator_from_rollouts(rows)
    if args.baseline_log:
        progress = re.compile(r"Training Batches Processed:[^\r\n]*\b" + str(step) + r"/\d+")
        while not progress.search(args.baseline_log.read_text()):
            if args.baseline_pid and not psutil.pid_exists(args.baseline_pid):
                raise RuntimeError("Baseline exited before completing checkpoint-step evaluation and logging")
            time.sleep(5)
    if args.baseline_pid:
        stop_baseline(args.baseline_pid)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"source_checkpoint": str(args.checkpoint), "source_step": step, "initial_denominator": denominator}
    for mode in ("vanilla", "stopd"):
        output = args.output / mode
        run_config(
            project, original, output, mode, args.checkpoint, step, args.steps, args.block, args.seed, denominator
        )
        report[mode] = summarize(output, args.steps)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["rollout_correctness"] = validate_rollouts(
        args.output / "vanilla",
        args.output / "stopd",
        original["generator"]["sampling_params"]["max_generate_length"],
        args.block,
        step + 1,
        denominator,
        args.seed,
    )
    next_checkpoint = validate_stopd_state(args.output / "stopd", step + args.steps)
    report["speedup"] = report["vanilla"]["timings"]["step"]["mean"] / report["stopd"]["timings"]["step"]["mean"]
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.continue_training:
        if report["speedup"] < 1.05:
            raise RuntimeError(
                "STOPD did not demonstrate at least 5% end-to-end acceleration; retain benchmark checkpoints"
            )
        run_name = (
            f"{Path(original['trainer']['policy']['model']['path']).name}-ultradata-opd-nora-stopd-from-step-{step}"
        )
        continuation = continue_training(project, original, next_checkpoint, args.output.parent / run_name)
        report["continuation"] = continuation
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(continuation), flush=True)


if __name__ == "__main__":
    main()
