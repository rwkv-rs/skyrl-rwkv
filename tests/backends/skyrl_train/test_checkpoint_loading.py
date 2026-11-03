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
@pytest.mark.parametrize("wrapped", [True, False])
def test_fsdp_checkpoint_saves_one_full_rwkv_replica(tmp_path, monkeypatch, rwkv, fsdp_size, copies, wrapped):
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
    model.model.config = SimpleNamespace(model_type="rwkv" if rwkv else "other")
    model.is_rwkv = rwkv
    if not wrapped:
        model = model.model
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


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("saved_world_size", [3, 8])
def test_ddp_resume_serializes_reads_and_restores_adam(tmp_path, monkeypatch, rank, saved_world_size):
    from skyrl.backends.skyrl_train.workers.ddp import ddp_strategy as module

    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scheduler.step()
    torch.save(source.state_dict(), tmp_path / f"model_world_size_{saved_world_size}_rank_0.pt")
    torch.save(optimizer.state_dict(), tmp_path / f"optim_world_size_{saved_world_size}_rank_0.pt")
    torch.save(
        {"lr_scheduler": scheduler.state_dict(), "client_state": {"global_step": 1}},
        tmp_path / f"extra_state_world_size_{saved_world_size}_rank_0.pt",
    )
    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    strategy = object.__new__(module.DDPStrategy)
    strategy.world_size = 3
    strategy.fsdp_strategy = "ddp"
    strategy.get_rank = lambda: rank
    strategy.print = lambda *args: None
    strategy._model_module = lambda model: model
    events = []
    original_load = torch.load

    def load(*args, **kwargs):
        assert events.count("barrier") == rank
        events.append("load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", load)
    monkeypatch.setattr(module.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(module.gc, "collect", lambda: events.append("release"))
    _, state = strategy.load_checkpoint(
        restored, str(tmp_path), optimizer=restored_optimizer, scheduler=restored_scheduler
    )
    assert events == ["barrier"] * rank + ["load"] * 3 + ["release"] + ["barrier"] * (3 - rank)
    assert state["client_state"]["global_step"] == 1
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for expected, actual in zip(optimizer.state.values(), restored_optimizer.state.values(), strict=True):
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for model, current_optimizer in ((source, optimizer), (restored, restored_optimizer)):
        model(torch.ones(2, 3)).sum().backward()
        current_optimizer.step()
    for expected, actual in zip(source.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ddp_rwkv_keeps_fp32_masters_and_canonical_checkpoint_names():
    from skyrl.backends.skyrl_train.workers.ddp.ddp_strategy import (
        _prepare_rwkv_mixed_precision,
    )

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    original_names = set(model.state_dict())
    masters = _prepare_rwkv_mixed_precision(model)
    assert masters is _prepare_rwkv_mixed_precision(model)
    assert all(parameter.dtype == torch.float32 for parameter in masters)
    assert model[0].weight.dtype == torch.bfloat16
    optimizer = torch.optim.AdamW(masters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in masters]
    model(torch.ones(2, 3, dtype=torch.bfloat16)).float().sum().backward()
    assert all(parameter.grad.dtype == torch.float32 for parameter in masters)
    optimizer.step()
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, masters, strict=True))
    assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
    checkpoint = model.state_dict()
    assert set(checkpoint) == original_names
    restored = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    _prepare_rwkv_mixed_precision(restored)
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(
        restored(torch.ones(2, 3, dtype=torch.bfloat16)), model(torch.ones(2, 3, dtype=torch.bfloat16)), rtol=0, atol=0
    )
