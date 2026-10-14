import json
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.workers.worker import Worker
from skyrl.backends.skyrl_train_backend import (
    SkyRLTrainBackend,
    SkyRLTrainBackendOverrides,
)
from skyrl.train.config import FSDPConfig


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_load_checkpoint_restores_requested_training_state(tmp_path, load_optimizer):
    checkpoint_path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(checkpoint_path, "w"):
        pass

    backend = object.__new__(SkyRLTrainBackend)
    backend.config = SkyRLTrainBackendOverrides()
    backend._model_ids_to_role = {"model_test": "policy"}
    backend._dispatch = MagicMock()
    backend._inference_engines_initialized = False

    backend.load_checkpoint(str(checkpoint_path), "model_test", load_optimizer=load_optimizer)

    backend._dispatch.load_checkpoint.assert_called_once()
    call = backend._dispatch.load_checkpoint.call_args
    assert call.kwargs["load_optimizer_states"] is load_optimizer
    assert call.kwargs["load_lr_scheduler_states"] is load_optimizer


def test_weights_only_load_does_not_reset_live_optimizer():
    worker = object.__new__(Worker)
    worker.model = MagicMock()
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.strategy = MagicMock()
    worker.strategy.load_checkpoint.return_value = ("/checkpoint", {})
    live_optimizer = worker.optimizer
    live_scheduler = worker.scheduler

    Worker.load_checkpoint(
        worker,
        "/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    assert worker.optimizer is live_optimizer
    assert worker.scheduler is live_scheduler

    worker.strategy.load_checkpoint.assert_called_once_with(
        model=worker.model,
        optimizer=None,
        scheduler=None,
        ckpt_dir="/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )


@pytest.mark.parametrize("rwkv,fsdp_size,copies", [(True, 1, 1), (True, 2, 2), (False, 1, 2)])
def test_fsdp_checkpoint_saves_one_full_rwkv_replica(tmp_path, monkeypatch, rwkv, fsdp_size, copies):
    """Only fully replicated RWKV state is deduplicated; rank-local RNG remains separate."""
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    strategy = FSDPStrategy(FSDPConfig(fsdp_size=fsdp_size))
    strategy.world_size = 2
    strategy.is_rank_0 = lambda: strategy.get_rank() == 0
    strategy.save_hf_configs = MagicMock()
    strategy.get_rng_state = MagicMock(side_effect=[{"rank_rng": 0}, {"rank_rng": 1}])
    model = HFModelWrapper.__new__(HFModelWrapper)
    nn.Module.__init__(model)
    model.model = nn.Linear(2, 2)
    model.model.config = SimpleNamespace()
    model.is_rwkv = rwkv
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    for rank in range(2):
        strategy.get_rank = lambda rank=rank: rank
        strategy.save_checkpoint(model, str(tmp_path), rank, optimizer=optimizer, scheduler=scheduler)
    assert len(list(tmp_path.glob("model_world_size_*"))) == copies
    assert len(list(tmp_path.glob("optim_world_size_*"))) == copies
    config = json.loads((tmp_path / "fsdp_config.json").read_text())
    assert config["replicated_rwkv"] is (copies == 1)
    for rank in range(2):
        extra = torch.load(tmp_path / f"extra_state_world_size_2_rank_{rank}.pt", weights_only=False)
        assert extra["rng"] == {"rank_rng": rank}


def test_replicated_rwkv_checkpoint_uses_full_state_restore(tmp_path):
    """The replicated format is restored even when the GPU count stays unchanged."""
    (tmp_path / "fsdp_config.json").write_text(json.dumps({"world_size": 8, "replicated_rwkv": True}))
    strategy = FSDPStrategy(FSDPConfig(fsdp_size=1))
    strategy.world_size = 8
    strategy._load_resharded_checkpoint = MagicMock(return_value=(str(tmp_path), {}))
    model, optimizer, scheduler = MagicMock(), MagicMock(), MagicMock()
    strategy.load_checkpoint(model, str(tmp_path), optimizer=optimizer, scheduler=scheduler)
    strategy._load_resharded_checkpoint.assert_called_once_with(
        model, str(tmp_path), 8, optimizer, scheduler, True, True, True, replicated_rwkv=True
    )
