import asyncio
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import ray
import torch

from examples.train.on_policy_distillation.stopd import STOPD, STOPDConfig, request_seed
from examples.train.on_policy_distillation.topd import (
    TOPDConfig,
    balance_training_input,
    compute_topd_advantage,
    compute_topd_stopd_advantage,
    topd_stopd_policy_loss,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
)
from skyrl.train.config import AlgorithmConfig, BaseConfig, SkyRLTrainConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.generators.base import TRAINING_PHASE_EVAL
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray


@dataclass
class RWKVEvalConfig(BaseConfig):
    project: str | None = None
    """Existing lighteval-rwkv checkout; None keeps native OPD evaluation."""
    task: str = "aime24"
    num_samples: int = 32
    context_length: int = 25600

    def __post_init__(self):
        if self.num_samples < 1 or self.context_length < 1:
            raise ValueError("RWKV evaluation requires positive num_samples and context_length")


@dataclass
class OnPolicyDistillationAlgorithmConfig(AlgorithmConfig):
    lighteval: RWKVEvalConfig = field(default_factory=RWKVEvalConfig)
    proximal_teacher_alpha: float = 1.0
    """TOP-D probability mixture weight for the teacher; 1.0 preserves native OPD."""
    stopd: STOPDConfig = field(default_factory=STOPDConfig)
    """Optional stochastic truncation, inverse-probability weights, and lagged normalization."""
    topd: TOPDConfig = field(default_factory=TOPDConfig)

    def __post_init__(self):
        if not isinstance(self.proximal_teacher_alpha, (int, float)) or not 0 < self.proximal_teacher_alpha <= 1:
            raise ValueError("proximal_teacher_alpha must be in (0, 1]")


OnPolicyDistillationConfig = make_config(algorithm_cls=OnPolicyDistillationAlgorithmConfig)


def validate_stopd_cfg(cfg):
    algorithm = cfg.trainer.algorithm
    generator = cfg.generator
    if algorithm.topd.enabled:
        if (
            algorithm.advantage_estimator != "no_op"
            or algorithm.policy_loss_type != ("topd_stopd" if algorithm.stopd.enabled else "regular")
            or algorithm.loss_reduction != "prompt_mean"
            or not algorithm.use_kl_in_reward
            or algorithm.use_kl_loss
            or algorithm.use_entropy_loss
            or algorithm.advantage_batch_normalize
            or generator.n_samples_per_prompt < 2
        ):
            raise ValueError(
                "TOP-D requires no_op, regular PPO (topd_stopd with STOPD), prompt_mean, dense rewards, "
                "and >= 2 responses per prompt"
            )
        if (
            generator.max_turns != 1
            or generator.batched
            or generator.step_wise_trajectories
            or not generator.use_conversation_multi_turn
            or generator.vision_language_generator
            or cfg.trainer.fully_async.enabled
            or algorithm.dynamic_sampling.type is not None
            or generator.apply_overlong_filtering
            or algorithm.off_policy_correction.tis_ratio_type is not None
            or algorithm.off_policy_correction.sequence_mask_metric is not None
            or algorithm.enable_sample_support_replay
            or generator.inference_engine.enable_return_routed_experts
        ):
            raise ValueError("TOP-D requires synchronous single-turn text rollouts without filtering or replay")
        generator.preserve_truncated_action_mask = True
        if algorithm.topd.balance_batch:
            trainer = cfg.trainer
            dp_size = trainer.placement.policy_num_nodes * trainer.placement.policy_num_gpus_per_node
            unit_size = math.lcm(generator.n_samples_per_prompt, trainer.micro_train_batch_size_per_gpu)
            if (
                trainer.strategy != "fsdp"
                or trainer.policy.sequence_parallel_size != 1
                or trainer.max_tokens_per_microbatch > 0
                or trainer.micro_forward_batch_size_per_gpu != trainer.micro_train_batch_size_per_gpu
                or trainer.train_batch_size * generator.n_samples_per_prompt % (dp_size * unit_size)
                or trainer.policy_mini_batch_size * generator.n_samples_per_prompt % (dp_size * unit_size)
            ):
                raise ValueError(
                    "TOP-D balance_batch requires FSDP without sequence parallelism, equal fixed forward/train "
                    "microbatches, and batch/mini-batch sizes divisible by the DP prompt/microbatch grid"
                )
    if not cfg.trainer.algorithm.stopd.enabled:
        return
    if generator.apply_overlong_filtering:
        raise ValueError("STOPD requires apply_overlong_filtering=false to retain sampled prefixes")
    generator.preserve_truncated_action_mask = True
    if not algorithm.topd.enabled and (
        algorithm.advantage_estimator != "no_op"
        or algorithm.policy_loss_type != "importance_sampling"
        or not algorithm.use_kl_in_reward
        or algorithm.use_kl_loss
        or algorithm.use_entropy_loss
        or algorithm.advantage_batch_normalize
        or algorithm.loss_reduction != "token_mean"
    ):
        raise ValueError("STOPD requires token-mean OPD with no_op advantages and importance_sampling loss")
    if (
        generator.max_turns != 1
        or generator.batched
        or generator.step_wise_trajectories
        or not generator.use_conversation_multi_turn
        or generator.vision_language_generator
        or cfg.trainer.fully_async.enabled
        or algorithm.dynamic_sampling.type is not None
        or (
            not algorithm.topd.enabled
            and (
                generator.n_samples_per_prompt != 1
                or cfg.trainer.update_epochs_per_batch != 1
                or cfg.trainer.policy_mini_batch_size != cfg.trainer.train_batch_size
            )
        )
    ):
        raise ValueError("STOPD requires synchronous single-turn rollouts and one optimizer step per rollout batch")
    if algorithm.stopd.block > generator.sampling_params.max_generate_length:
        raise ValueError("STOPD block must not exceed max_generate_length")


class STOPDGenerator(SkyRLGymGenerator):
    """Use native trajectory generation with independent per-request token budgets."""

    complete_rollouts = False
    grouped_budgets = False

    def __init__(self, *args, stopd_cfg, complete_rollouts=False, grouped_budgets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.complete_rollouts = complete_rollouts
        self.grouped_budgets = grouped_budgets
        self.stopd = STOPD(
            self.generator_cfg.sampling_params.max_generate_length,
            require_denominator=not (complete_rollouts or grouped_budgets),
            **{key: value for key, value in vars(stopd_cfg).items() if key != "enabled"},
        )

    async def generate(self, input_batch, disable_tqdm=False):
        metadata = input_batch["batch_metadata"]
        if metadata.training_phase == TRAINING_PHASE_EVAL:
            return await super().generate(input_batch, disable_tqdm=disable_tqdm)
        self.stopd.step = metadata.global_step - 1
        count = len(input_batch["prompts"])
        if self.complete_rollouts:
            budgets = [self.stopd.max_len] * count
        elif self.grouped_budgets:
            group_size = self.generator_cfg.n_samples_per_prompt
            budgets = [budget for budget in self.stopd.budgets(count // group_size) for _ in range(group_size)]
        else:
            budgets = self.stopd.budgets(count)
        input_batch = {**input_batch, "env_extras": copy.deepcopy(input_batch["env_extras"])}
        for index, (extras, budget) in enumerate(zip(input_batch["env_extras"], budgets, strict=True)):
            extras["_stopd_budget"] = budget
            extras["_stopd_seed"] = request_seed(self.stopd.seed, metadata.global_step, index, 0, "rollout")
        return await super().generate(input_batch, disable_tqdm=disable_tqdm)

    async def agent_loop(
        self,
        prompt,
        env_class,
        env_extras,
        max_tokens,
        max_input_length,
        sampling_params=None,
        trajectory_id=None,
        cache_salt=None,
        training_phase="train",
    ):
        if training_phase != TRAINING_PHASE_EVAL:
            max_tokens = env_extras.pop("_stopd_budget")
            sampling_params = {**sampling_params, "max_tokens": max_tokens, "seed": env_extras.pop("_stopd_seed")}
        return await super().agent_loop(
            prompt,
            env_class,
            env_extras,
            max_tokens,
            max_input_length,
            sampling_params=sampling_params,
            trajectory_id=trajectory_id,
            cache_salt=cache_salt,
            training_phase=training_phase,
        )


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """
    Custom trainer for On Policy Distillation.

    Overrides the apply_reward_kl_penalty method to set the rewards just to the kl penalty
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stopd_cfg = self.cfg.trainer.algorithm.stopd
        self.stopd = (
            STOPD(
                self.cfg.generator.sampling_params.max_generate_length,
                require_denominator=not self.cfg.trainer.algorithm.topd.enabled,
                **{key: value for key, value in vars(stopd_cfg).items() if key != "enabled"},
            )
            if stopd_cfg.enabled
            else None
        )

    def convert_to_training_input(self, generator_output, uids):
        data = super().convert_to_training_input(generator_output, uids)
        if self.cfg.trainer.algorithm.topd.enabled and self.cfg.trainer.algorithm.topd.balance_batch:
            data, metrics = balance_training_input(
                data, self.dispatch.dp_size("policy"), self.cfg.trainer.micro_train_batch_size_per_gpu
            )
            self.all_metrics.update(metrics)
        return data

    async def eval(self, vllm_metrics_scraper=None):
        config = self.cfg.trainer.algorithm.lighteval
        if config.project is None:
            return await super().eval(vllm_metrics_scraper=vllm_metrics_scraper)
        output = Path(self.cfg.trainer.export_path) / "lighteval" / f"global_step_{self.global_step}"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-config",
            "--no-project",
            sys.executable,
            "-m",
            "examples.train.rwkv.evaluate",
            "--project",
            config.project,
            "--url",
            self.inference_engine_client.proxy_url,
            "--model",
            self.inference_engine_client.model_name,
            "--tokenizer",
            self.cfg.trainer.policy.model.path,
            "--output",
            str(output),
            "--step",
            str(self.global_step),
            "--task",
            config.task,
            "--num-samples",
            str(config.num_samples),
            "--context-length",
            str(config.context_length),
            "--concurrency",
            str(
                self.cfg.generator.inference_engine.num_engines * self.cfg.generator.inference_engine.max_num_seqs + 32
            ),
        ]
        # Keep the producer's pinned math dependency private to the evaluation
        # subprocess, using this project's interpreter and virtual environment.
        dependencies = Path(sys.executable).parent.parent / "rwkv-eval"
        env = os.environ | {"PYTHONPATH": f"{dependencies}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.resume()
        try:
            with (output / "producer.log").open("w") as handle:
                process = await asyncio.create_subprocess_exec(*command, env=env, stdout=handle, stderr=handle)
                if await process.wait() != 0:
                    raise RuntimeError(f"lighteval-rwkv evaluation failed; see {output / 'producer.log'}")
        finally:
            if vllm_metrics_scraper is not None:
                vllm_metrics_scraper.pause()
        if self.cfg.trainer.num_logger_eval_samples > 0:
            completed = json.loads((output / "completed.json").read_text())
            samples = []
            with Path(completed["completions_path"]).open() as handle:
                for line in handle:
                    record = json.loads(line)
                    response = record["model_response"]
                    for text, answer, reason in zip(
                        response["text"], response["text_post_processed"], response["finish_reasons"], strict=True
                    ):
                        samples.append(
                            (
                                self.global_step,
                                record["task_name"],
                                response["rendered_prompt"],
                                json.dumps(record["doc"]["choices"]),
                                text,
                                answer,
                                reason,
                            )
                        )
                        if len(samples) == self.cfg.trainer.num_logger_eval_samples:
                            break
                    if len(samples) == self.cfg.trainer.num_logger_eval_samples:
                        break
            self.tracker.log_samples_to_table(
                "eval/lighteval/samples",
                ["step", "task", "prompt", "choices", "response", "answer_extracted", "finish_reason"],
                samples,
                step=self.global_step,
                incremental=True,
            )
        return json.loads((output / "eval_metrics.json").read_text())

    @staticmethod
    def _stopd_mask(data):
        mask = data["loss_mask"].bool() & data["response_mask"].bool()
        pad_size = data.metadata.get("pad_size", 0)
        if pad_size:
            mask[-pad_size:] = False
        return mask

    def compute_advantages_and_returns(self, data):
        if self.cfg.trainer.algorithm.topd.enabled:
            mask = self._stopd_mask(data)
            arguments = (data["rewards"], mask, data.metadata["policy_prompt_boundaries"])
            epsilon = self.cfg.trainer.algorithm.topd.epsilon
            if self.stopd is not None:
                advantages, returns, counts = compute_topd_stopd_advantage(
                    *arguments, self.stopd.max_len, self.stopd.block, self.stopd.truncating, epsilon
                )
            else:
                advantages, returns, counts = compute_topd_advantage(*arguments, epsilon)
            data["advantages"], data["returns"] = advantages, returns
            data.metadata["topd_token_counts"] = counts
            self.all_metrics.update(
                {
                    "topd/observed_tokens": int(mask.sum()),
                    "topd/observed_return_mean": returns[mask].mean().item(),
                    "topd/coefficient_abs_mean": advantages[mask].abs().mean().item(),
                    "topd/optimizer_steps_per_batch": len(data.metadata["policy_mini_batch_boundaries"])
                    * self.cfg.trainer.update_epochs_per_batch,
                }
            )
            if self.cfg.trainer.dump_data_batch:
                self.dump_data(data, file_name=f"global_step_{self.global_step}_topd_input")
            return data
        data = super().compute_advantages_and_returns(data)
        if self.stopd is not None and self.stopd.truncating:
            # Native token_mean divides by the complete mini-batch's loss mask
            # before DP sharding. Cancel that divisor in favor of C_k.
            mask = self._stopd_mask(data)
            start, end = data.metadata["policy_mini_batch_boundaries"][0]
            n_obs = data["loss_mask"][start:end].sum().item()
            data["advantages"] = data["advantages"] * self.stopd.multiplier(mask, n_obs)
        return data

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        if not self.cfg.trainer.algorithm.topd.enabled:
            return super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
        # Eq. (7): token means per group, then a mean over mini-batch prompts.
        # The telescoping coefficients already include each level's token mean.
        normalized = torch.zeros_like(data["advantages"])
        for start, end in mini_batch_boundaries:
            groups = [(a, b, n) for a, b, n in data.metadata["topd_token_counts"] if start <= a < end]
            for a, b, count in groups:
                denominator = len(groups) * (max(count, 1) if self.stopd is None else 1)
                normalized[a:b] = data["advantages"][a:b] / denominator
        data["advantages"] = normalized
        return data

    def train_critic_and_policy(self, data):
        status = super().train_critic_and_policy(data)
        if self.stopd is not None:
            metrics = self.stopd.end_step(self._stopd_mask(data))
            metrics["grouped_budgets"] = int(self.cfg.trainer.algorithm.topd.enabled)
            self.all_metrics.update({f"stopd/{key}": value for key, value in metrics.items()})
        return status

    def save_checkpoints(self):
        if self.stopd is not None:
            checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}")
            io.makedirs(checkpoint_path, exist_ok=True)
            with io.open_file(os.path.join(checkpoint_path, "stopd_state.pt"), "wb") as handle:
                torch.save(self.stopd.state_dict(), handle)
        # The native writer publishes latest_ckpt_global_step.txt last, after
        # the STOPD state and the complete model/optimizer/data state exist.
        return super().save_checkpoints()

    def load_checkpoints(self):
        global_step, checkpoint_path = super().load_checkpoints()
        if checkpoint_path is not None and (self.stopd is not None or self.cfg.trainer.algorithm.topd.enabled):
            # A STOPD continuation must restore the exact data position; propagate
            # a load failure rather than the native best-effort dataloader fallback.
            with io.open_file(os.path.join(checkpoint_path, "data.pt"), "rb") as handle:
                dataloader_state = torch.load(handle, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state)
            if "_iterator_finished" in dataloader_state:
                # Restore the iterator now without consuming a batch. The native
                # single-process sampler snapshot precedes the worker-seed RNG
                # draw; loading that sampler resets the draw. Multi-process
                # snapshots already include it in the sampler's generator state.
                self.train_dataloader.state_dict()
                if self.train_dataloader.num_workers == 0:
                    torch.empty((), dtype=torch.int64).random_(generator=self.train_dataloader.generator)
            if (
                "_iterator_finished" in dataloader_state
                and not dataloader_state["_iterator_finished"]
                and global_step > 0
                and global_step % len(self.train_dataloader) == 0
            ):
                # Native epoch-end saves happen just before StopIteration. Finish
                # that iterator so the trainer's next epoch gets the next shuffle,
                # rather than spending its first resumed epoch on an empty iterator.
                if next(iter(self.train_dataloader), None) is not None:
                    raise ValueError("Epoch-end checkpoint has unconsumed training batches")
        if self.stopd is not None and checkpoint_path is not None:
            state_path = os.path.join(checkpoint_path, "stopd_state.pt")
            if io.exists(state_path):
                with io.open_file(state_path, "rb") as handle:
                    self.stopd.load_state_dict(torch.load(handle, map_location="cpu", weights_only=False))
                if self.stopd.step != global_step:
                    raise ValueError("STOPD state and trainer checkpoint steps do not match")
            else:
                self.stopd.step = global_step
                if self.stopd.require_denominator and self.stopd.truncating and self.stopd.denominator is None:
                    raise ValueError("Importing a Vanilla OPD checkpoint requires stopd.initial_denominator")
        return global_step, checkpoint_path

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Computes the KL penalty and sets the rewards to the KL penalty"""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # set rewards to the KL penalty
        # note: tinker seems to use k1 or k2: https://github.com/thinking-machines-lab/tinker-cookbook/blob/3dd0463472dda5847efee80010b50514fa3068ef/tinker_cookbook/rl/metrics.py#L40
        rewards = -(action_log_probs - teacher_action_log_probs)
        alpha = self.cfg.trainer.algorithm.proximal_teacher_alpha
        if alpha < 1:
            # TOP-D Eq. (5): log(alpha * p_teacher / p_old + 1 - alpha).
            # The cached trainer logprobs are from the rollout policy before
            # this batch's update. Work in log space to avoid ratio overflow.
            # https://arxiv.org/abs/2607.04751
            rewards = torch.logaddexp(rewards + math.log(alpha), torch.full_like(rewards, math.log1p(-alpha)))
        rewards = rewards * loss_masks_all
        data["rewards"] = rewards
        mask = self._stopd_mask(data)
        if self.stopd is not None and self.stopd.truncating and not self.cfg.trainer.algorithm.topd.enabled:
            weights, denominator = self.stopd.weights(mask), self.stopd.denominator
        else:
            weights, denominator = mask, mask.sum().item()
        self.all_metrics.update(
            {
                "opd/reward_mean": (rewards * weights).sum().item() / denominator,
                "opd/reward_abs_mean": (rewards.abs() * weights).sum().item() / denominator,
                "opd/reverse_kl_mean": ((action_log_probs - teacher_action_log_probs) * weights).sum().item()
                / denominator,
            }
        )
        if self.cfg.trainer.algorithm.topd.enabled:
            # Self-normalized IPW estimates complete-response token means:
            # later tokens have lower STOPD observation probabilities.
            full_weights = self.stopd.weights(mask) if self.stopd is not None else mask
            full_count = full_weights.sum().item()
            self.all_metrics.update(
                {
                    "opd/full_reward_mean_estimate": (rewards * full_weights).sum().item() / full_count,
                    "opd/full_reverse_kl_mean_estimate": ((action_log_probs - teacher_action_log_probs) * full_weights)
                    .sum()
                    .item()
                    / full_count,
                }
            )
        return data


def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


class OnPolicyDistillationExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        algorithm = cfg.trainer.algorithm
        if not algorithm.stopd.enabled and not algorithm.topd.enabled:
            return super().get_generator(cfg, tokenizer, inference_engine_client)
        stopd_cfg = copy.deepcopy(algorithm.stopd)
        if not stopd_cfg.enabled:
            stopd_cfg.seed = cfg.trainer.seed
        return STOPDGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
            stopd_cfg=stopd_cfg,
            complete_rollouts=algorithm.topd.enabled and not algorithm.stopd.enabled,
            grouped_budgets=algorithm.topd.enabled and algorithm.stopd.enabled,
        )

    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = OnPolicyDistillationExp(cfg)
    exp.run()


def main() -> None:
    # Registry actors may import this module while unpickling the estimator.
    # Register from the entrypoint so that import never calls back into them.
    register_advantage_estimator("no_op")(compute_no_op_advantage)
    register_policy_loss("topd_stopd")(topd_stopd_policy_loss)
    cfg = OnPolicyDistillationConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)
    validate_stopd_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
