"""CPU test that the PackedPerExpertLinear.sharded_state_dict backport installs idempotently."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.megatron

try:
    from megatron.bridge.peft import utils as bridge_peft_utils

    from skyrl.backends.skyrl_train.patches.megatron.patch_packed_per_expert_sharded_state_dict import (
        apply_packed_per_expert_sharded_state_dict_patch,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def test_patch_is_idempotent():
    cls = bridge_peft_utils.PackedPerExpertLinear
    apply_packed_per_expert_sharded_state_dict_patch()
    patched = cls.sharded_state_dict

    apply_packed_per_expert_sharded_state_dict_patch()
    assert cls.sharded_state_dict is patched
