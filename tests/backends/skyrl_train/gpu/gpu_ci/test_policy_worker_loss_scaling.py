"""GPU policy worker loss-scaling regression tests."""

from __future__ import annotations

import math
import socket
from types import SimpleNamespace
from typing import Callable, Literal

import pytest
import ray
import torch
from ray.util.placement_group import placement_group

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MICRO_BATCH_SIZES = (1, 2)
LOSS_FNS = ("cross_entropy", "dual_clip")
POLICY_CALLS: tuple[Literal["forward", "forward_backward"], ...] = ("forward", "forward_backward")
POLICY_WORKER_TYPES = [
    "fsdp",
    pytest.param("megatron", marks=pytest.mark.megatron),
]
MICROBATCH_INVARIANCE_REL_TOL = 1e-3
MICROBATCH_INVARIANCE_ABS_TOL = 1e-2


def _make_policy_cfg(
    policy_worker_type: str,
    loss_fn: str,
    *,
    micro_train_batch_size_per_gpu: int = 1,
    micro_forward_batch_size_per_gpu: int = 1,
) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = policy_worker_type
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.logger = "console"
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_forward_batch_size_per_gpu
    cfg.trainer.algorithm.policy_loss_type = loss_fn
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.temperature = 1.0
    cfg.generator.sampling_params.temperature = 1.0

    if policy_worker_type == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1

    validate_cfg(cfg)
    return cfg


def _make_loss_batch(loss_fn: str, *, batch_size: int = 4, seq_len: int = 10, num_actions: int = 4):
    batch = make_dummy_training_batch(batch_size=batch_size, seq_len=seq_len, num_actions=num_actions)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    batch["sequences"] = torch.randint(10, 100, (batch_size, seq_len), generator=generator, dtype=torch.long)
    batch["attention_mask"] = torch.ones((batch_size, seq_len), dtype=torch.long)
    batch["action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["base_action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["values"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["returns"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["response_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32)
    batch["rollout_logprobs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)

    if loss_fn == "cross_entropy":
        batch["advantages"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
        # Loss mask scaling replicates logic on `SFTTrainer` -> micro batch agnostic scaling
        batch["loss_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32) / (batch_size * num_actions)
    elif loss_fn == "dual_clip":
        batch["advantages"] = -torch.ones((batch_size, num_actions), dtype=torch.float32)
        batch["loss_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported loss_fn {loss_fn}")

    batch.metadata = {"response_length": num_actions}
    return batch


def _create_policy_placement_group(cfg: SkyRLTrainConfig):
    num_gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
    return raw_pg


def _cleanup_policy_actor_group(actor_group, raw_pg):
    for actor_info in actor_group.actor_infos:
        ray.kill(actor_info.handle, no_restart=True)
    ray.util.remove_placement_group(raw_pg)


def _run_policy_worker_calls(
    policy_worker_type: str,
    loss_fn: str,
    batch_factory: Callable[[], TrainingInputBatch],
    *,
    micro_batch_size: int,
) -> dict[str, WorkerOutput]:
    cfg = _make_policy_cfg(
        policy_worker_type,
        loss_fn,
        micro_train_batch_size_per_gpu=micro_batch_size,
        micro_forward_batch_size_per_gpu=micro_batch_size,
    )
    actor_group = None
    raw_pg = None
    try:
        raw_pg = _create_policy_placement_group(cfg)
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=raw_pg,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
            num_gpus_per_actor=1.0,
        )
        dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
        outputs = {}
        for call in POLICY_CALLS:
            batch = batch_factory()
            if call == "forward":
                outputs[call] = dispatch.forward("policy", batch, loss_fn=loss_fn)
            elif call == "forward_backward":
                outputs[call] = dispatch.forward_backward("policy", batch, loss_fn=loss_fn)
            else:
                raise ValueError(f"Unsupported call {call}")
        return outputs
    finally:
        if actor_group is not None and raw_pg is not None:
            _cleanup_policy_actor_group(actor_group, raw_pg)


def _primary_loss(output: WorkerOutput, loss_fn: str) -> float:
    metrics = output.metrics
    if loss_fn == "cross_entropy":
        assert "loss" in metrics
        return metrics["loss"]

    if "policy_loss" in metrics:
        if "final_loss" in metrics:
            assert math.isclose(metrics["final_loss"], metrics["policy_loss"], rel_tol=2e-4, abs_tol=2e-3)
        return metrics["policy_loss"]

    # FSDP forward(loss_fn="dual_clip") goes through _forward_micro_with_loss
    # and reports the selected policy loss under the generic "loss" key.
    assert "loss" in metrics
    return metrics["loss"]


def _assert_loss_output_shape(output: WorkerOutput, loss_fn: str, batch_size: int = 4, num_actions: int = 4):
    assert math.isfinite(_primary_loss(output, loss_fn))
    assert len(output.loss_fn_outputs) == batch_size
    for item in output.loss_fn_outputs:
        assert "logprobs" in item
        assert len(item["logprobs"]) == num_actions
        if loss_fn == "cross_entropy":
            assert "elementwise_loss" in item
            assert len(item["elementwise_loss"]) == num_actions


@pytest.mark.parametrize("policy_worker_type", POLICY_WORKER_TYPES, ids=["fsdp", "megatron"])
@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_policy_loss_is_microbatch_size_invariant(ray_init_fixture, policy_worker_type, loss_fn):
    losses_by_call = {call: {} for call in POLICY_CALLS}
    for micro_batch_size in MICRO_BATCH_SIZES:
        outputs_by_call = _run_policy_worker_calls(
            policy_worker_type,
            loss_fn,
            lambda: _make_loss_batch(loss_fn),
            micro_batch_size=micro_batch_size,
        )
        for call, output in outputs_by_call.items():
            _assert_loss_output_shape(output, loss_fn)
            losses_by_call[call][micro_batch_size] = _primary_loss(output, loss_fn)

    for call in POLICY_CALLS:
        assert losses_by_call[call][1] == pytest.approx(
            losses_by_call[call][2],
            rel=MICROBATCH_INVARIANCE_REL_TOL,
            abs=MICROBATCH_INVARIANCE_ABS_TOL,
        )


@pytest.mark.megatron
@pytest.mark.parametrize("loss_fn", LOSS_FNS)
@pytest.mark.parametrize("micro_batch_size", MICRO_BATCH_SIZES)
def test_policy_loss_fsdp_megatron_consistent(ray_init_fixture, micro_batch_size, loss_fn):
    losses_by_call = {call: {} for call in POLICY_CALLS}
    for policy_worker_type in ("fsdp", "megatron"):
        outputs_by_call = _run_policy_worker_calls(
            policy_worker_type,
            loss_fn,
            lambda: _make_loss_batch(loss_fn),
            micro_batch_size=micro_batch_size,
        )
        for call, output in outputs_by_call.items():
            _assert_loss_output_shape(output, loss_fn)
            losses_by_call[call][policy_worker_type] = _primary_loss(output, loss_fn)

    for call in POLICY_CALLS:
        assert losses_by_call[call]["fsdp"] == pytest.approx(
            losses_by_call[call]["megatron"],
            rel=2e-2,
            abs=2e-1,
        )


def _fsdp_accumulation_rank(rank, world_size, port):
    """Compare native FSDP gradients, clipping, and repeated AdamW updates."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
        create_device_mesh,
        fsdp2_clip_grad_norm_,
    )
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        generator = torch.Generator().manual_seed(1234 + rank)
        inputs = torch.randn(5, 8, generator=generator, device="cpu").cuda()
        targets = torch.randn(5, 4, generator=generator, device="cpu").cuda()
        for fsdp_size in (world_size, 1):
            mesh = create_device_mesh(world_size, fsdp_size)
            for dtype in (torch.float32, torch.bfloat16):
                reference = None
                for micro_size, sync_each, padding in (
                    (5, True, False),
                    (2, True, False),
                    (2, False, False),
                    (2, False, True),
                ):
                    torch.manual_seed(42)
                    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).cuda()
                    kwargs = {
                        "mesh": mesh,
                        "mp_policy": MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
                        "reshard_after_forward": False,
                    }
                    fully_shard(model[0], **kwargs)
                    fully_shard(model[2], **kwargs)
                    fully_shard(model, **kwargs)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
                    worker = SimpleNamespace(
                        cfg=SimpleNamespace(
                            policy=SimpleNamespace(
                                fsdp_config=SimpleNamespace(sync_gradients_each_microbatch=sync_each)
                            )
                        ),
                        model=SimpleNamespace(model=model),
                    )
                    results = []
                    for _ in range(2):
                        batches = [
                            (inputs[i : i + micro_size], targets[i : i + micro_size]) for i in range(0, 5, micro_size)
                        ]
                        if padding:
                            batches.append((inputs[:1], None))
                        loss_sum = torch.zeros((), device="cuda")
                        for index, (x, y) in enumerate(batches):
                            with FSDPPolicyWorkerBase._microbatch_context(worker, is_last=index == len(batches) - 1):
                                output = model(x).float()
                                loss = (
                                    (output - y).square().sum() / targets.numel() if y is not None else output.sum() * 0
                                )
                                loss.backward()
                                loss_sum += loss.detach()
                        torch.distributed.all_reduce(loss_sum)
                        loss_sum /= world_size
                        gradients = [p.grad.full_tensor().detach().cpu().clone() for p in model.parameters()]
                        norm = fsdp2_clip_grad_norm_(model.parameters(), 0.3).full_tensor().detach().cpu().clone()
                        optimizer.step()
                        weights = [p.full_tensor().detach().cpu().clone() for p in model.parameters()]
                        results.append([loss_sum.cpu(), norm, *gradients, *weights])
                        optimizer.zero_grad()
                    if reference is None:
                        reference = results
                    else:
                        for expected_step, actual_step in zip(reference, results, strict=True):
                            for expected, actual in zip(expected_step, actual_step, strict=True):
                                torch.testing.assert_close(
                                    actual,
                                    expected,
                                    rtol=5e-3 if dtype == torch.bfloat16 else 1e-5,
                                    atol=2e-4 if dtype == torch.bfloat16 else 1e-6,
                                )
                    del optimizer, model, worker
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def test_native_fsdp_accumulation_gradients_and_optimizer():
    """Unbalanced final/padding microbatches match full-batch FSDP and HSDP updates."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_fsdp_accumulation_rank, args=(2, port), nprocs=2, join=True)
