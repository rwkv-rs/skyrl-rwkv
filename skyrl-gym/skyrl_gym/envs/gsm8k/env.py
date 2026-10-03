from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        reward_spec = extras["reward_spec"]
        config_strict_reward = (
            env_config.get("strict_reward", False)
            if isinstance(env_config, dict)
            else getattr(env_config, "strict_reward", False)
        )
        self.strict_reward = bool(config_strict_reward or reward_spec.get("strict", False))
        self.ground_truth = reward_spec["ground_truth"]
        self._generation_metadata = {"ended_eod": False, "truncated": False, "stop_reason": None}
        self._strict_reward_details: Dict[str, Any] = {}

    def set_generation_metadata(
        self,
        *,
        action: str,
        ended_eod: bool,
        truncated: bool,
        stop_reason: str,
    ) -> None:
        """Provide generation termination metadata before scoring a response."""
        self._generation_metadata = {
            "ended_eod": bool(ended_eod),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
        }
        if self.strict_reward:
            _, self._strict_reward_details = utils.compute_strict_score(
                action,
                self.ground_truth,
                ended_eod=ended_eod,
                truncated=truncated,
            )
            self._strict_reward_details["stop_reason"] = stop_reason

    def _get_reward(self, action: str) -> float:
        if not self.strict_reward:
            return utils.compute_score(action, self.ground_truth)

        reward, self._strict_reward_details = utils.compute_strict_score(
            action,
            self.ground_truth,
            ended_eod=self._generation_metadata["ended_eod"],
            truncated=self._generation_metadata["truncated"],
        )
        self._strict_reward_details["stop_reason"] = self._generation_metadata["stop_reason"]
        return reward

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._strict_reward_details) if self.strict_reward else {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        metadata = dict(self._strict_reward_details) if self.strict_reward else {}
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
