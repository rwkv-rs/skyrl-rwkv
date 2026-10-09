"""Tests for MoE config fields on MegatronConfig dataclass."""

from skyrl.train.config.config import MegatronConfig, build_nested_dataclass


class TestMegatronConfigMoEFields:
    """MoE and LoRA fields survive nested config parsing."""

    def test_moe_config_from_dict(self):
        """MoE fields should survive dict -> dataclass round-trip."""
        d = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": "none",
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
        }
        cfg = build_nested_dataclass(MegatronConfig, d)
        assert cfg.moe_token_dispatcher_type == "alltoall"
        assert cfg.moe_router_load_balancing_type == "none"
        assert cfg.moe_grouped_gemm is True
        assert cfg.moe_router_score_function == "sigmoid"
        assert cfg.moe_router_enable_expert_bias is True

    def test_lora_config_from_dict(self):
        d = {"lora_config": {"experts_shared_outer_loras": True}}
        cfg = build_nested_dataclass(MegatronConfig, d)
        assert cfg.lora_config.experts_shared_outer_loras is True
