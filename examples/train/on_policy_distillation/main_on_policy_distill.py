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
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
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

    def __post_init__(self):
        if not isinstance(self.proximal_teacher_alpha, (int, float)) or not 0 < self.proximal_teacher_alpha <= 1:
            raise ValueError("proximal_teacher_alpha must be in (0, 1]")


OnPolicyDistillationConfig = make_config(algorithm_cls=OnPolicyDistillationAlgorithmConfig)


def validate_stopd_cfg(cfg):
    if not cfg.trainer.algorithm.stopd.enabled:
        return
    algorithm = cfg.trainer.algorithm
    generator = cfg.generator
    if generator.apply_overlong_filtering:
        raise ValueError("STOPD requires apply_overlong_filtering=false to retain sampled prefixes")
    generator.preserve_truncated_action_mask = True
    if (
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
        or generator.n_samples_per_prompt != 1
        or cfg.trainer.fully_async.enabled
        or algorithm.dynamic_sampling.type is not None
        or cfg.trainer.update_epochs_per_batch != 1
        or cfg.trainer.policy_mini_batch_size != cfg.trainer.train_batch_size
    ):
        raise ValueError("STOPD requires synchronous single-turn rollouts and one optimizer step per rollout batch")
    if algorithm.stopd.block > generator.sampling_params.max_generate_length:
        raise ValueError("STOPD block must not exceed max_generate_length")


class STOPDGenerator(SkyRLGymGenerator):
    """Use native trajectory generation with independent per-request token budgets."""

    def __init__(self, *args, stopd_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopd = STOPD(
            self.generator_cfg.sampling_params.max_generate_length,
            **{key: value for key, value in vars(stopd_cfg).items() if key != "enabled"},
        )

    async def generate(self, input_batch, disable_tqdm=False):
        metadata = input_batch["batch_metadata"]
        if metadata.training_phase == TRAINING_PHASE_EVAL:
            return await super().generate(input_batch, disable_tqdm=disable_tqdm)
        self.stopd.step = metadata.global_step - 1
        budgets = self.stopd.budgets(len(input_batch["prompts"]))
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
                **{key: value for key, value in vars(stopd_cfg).items() if key != "enabled"},
            )
            if stopd_cfg.enabled
            else None
        )

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
        data = super().compute_advantages_and_returns(data)
        if self.stopd is not None and self.stopd.truncating:
            # Native token_mean divides by the complete mini-batch's loss mask
            # before DP sharding. Cancel that divisor in favor of C_k.
            mask = self._stopd_mask(data)
            start, end = data.metadata["policy_mini_batch_boundaries"][0]
            n_obs = data["loss_mask"][start:end].sum().item()
            data["advantages"] = data["advantages"] * self.stopd.multiplier(mask, n_obs)
        return data

    def train_critic_and_policy(self, data):
        status = super().train_critic_and_policy(data)
        if self.stopd is not None:
            metrics = self.stopd.end_step(self._stopd_mask(data))
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
        if self.stopd is not None and checkpoint_path is not None:
            # A STOPD continuation must restore the exact data position; propagate
            # a load failure rather than the native best-effort dataloader fallback.
            with io.open_file(os.path.join(checkpoint_path, "data.pt"), "rb") as handle:
                self.train_dataloader.load_state_dict(torch.load(handle, map_location="cpu", weights_only=False))
            state_path = os.path.join(checkpoint_path, "stopd_state.pt")
            if io.exists(state_path):
                with io.open_file(state_path, "rb") as handle:
                    self.stopd.load_state_dict(torch.load(handle, map_location="cpu", weights_only=False))
                if self.stopd.step != global_step:
                    raise ValueError("STOPD state and trainer checkpoint steps do not match")
            else:
                self.stopd.step = global_step
                if self.stopd.truncating and self.stopd.denominator is None:
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
        if self.stopd is not None and self.stopd.truncating:
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
        return data


def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


class OnPolicyDistillationExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if not cfg.trainer.algorithm.stopd.enabled:
            return super().get_generator(cfg, tokenizer, inference_engine_client)
        return STOPDGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
            stopd_cfg=cfg.trainer.algorithm.stopd,
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
    cfg = OnPolicyDistillationConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)
    validate_stopd_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
