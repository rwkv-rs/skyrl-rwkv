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
        self._strict_score_cache = None

    def _score_strict_once(self, action: str) -> float:
        metadata = self._generation_metadata
        key = (action, self.ground_truth, metadata["ended_eod"], metadata["truncated"])
        if self._strict_score_cache is None or self._strict_score_cache[0] != key:
            reward, details = utils.compute_strict_score(
                action, self.ground_truth, ended_eod=metadata["ended_eod"], truncated=metadata["truncated"]
            )
            self._strict_score_cache = (key, reward, dict(details))
        _, reward, details = self._strict_score_cache
        self._strict_reward_details = {**details, "stop_reason": metadata["stop_reason"], "scoring_skipped": False}
        return reward

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
            self._strict_reward_details = {**self._generation_metadata, "strict_reward": 0.0, "scoring_skipped": True}

    def _get_reward(self, action: str) -> float:
        if not self.strict_reward:
            return utils.compute_score(action, self.ground_truth)

        if self._generation_metadata["truncated"] or not self._generation_metadata["ended_eod"]:
            return 0.0
        return self._score_strict_once(action)

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._strict_reward_details) if self.strict_reward else {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        metadata = dict(self._strict_reward_details) if self.strict_reward else {}
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
