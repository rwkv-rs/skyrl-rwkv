"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
)
from skyrl.train.utils.async_utils import BackgroundFailure
from tests.train.util import example_dummy_config


@pytest.mark.asyncio
@pytest.mark.parametrize("save_at_epoch_end", [False, True])
@pytest.mark.parametrize("exhaust_early", [False, True])
async def test_train_epoch_end_saving(monkeypatch, save_at_epoch_end, exhaust_early):
    """Exercise the real loop: optional epoch saves, periodic saves, and final draining."""
    cfg = example_dummy_config()
    cfg.trainer.epochs = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.zero_variance_filter = exhaust_early
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.fully_async.sample_full_batch = exhaust_early
    if not save_at_epoch_end:
        cfg.trainer.fully_async.save_at_epoch_end = False
    cfg.trainer.eval_interval = 0
    cfg.trainer.ckpt_interval = cfg.trainer.hf_save_interval = 3
    cfg.trainer.update_ref_every_epoch = False
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    loader = StatefulDataLoader(list(range(4)), batch_size=1)
    monkeypatch.setattr("skyrl.train.fully_async_trainer.build_dataloader", lambda *args, **kwargs: loader)
    trainer = FullyAsyncRayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=None,
        inference_engine_client=None,
        generator=None,
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.save_weights_for_sampler = AsyncMock()
    trainer.dispatch.get_timing_metrics.return_value = {}
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "_run_generate_for_a_group_loop", AsyncMock())
    monkeypatch.setattr(trainer, "convert_generation_group_mini_batch_to_training_input", lambda *args: None)
    monkeypatch.setattr(trainer, "_run_training", AsyncMock(return_value={"loss": 0.0}))

    async def collect(*args):
        trained = trainer.async_train_dataloader.num_trained()
        exhausted = exhaust_early and trained == 2
        uids = [str(trained)] if exhausted else [str(trained), str(trained + 1)]
        groups = []
        for uid in uids:
            await trainer._staleness_manager.acquire_submission_slot()
            await trainer._staleness_manager.on_rollout_accepted()
            groups.append(
                GeneratedOutputGroup(generator_output={}, uid=uid, global_step_when_scheduled=trainer.global_step)
            )
        return groups, [], exhausted

    monkeypatch.setattr(trainer, "_collect_generation_mini_batch", collect)
    checkpoint_steps, export_steps = [], []
    monkeypatch.setattr(trainer, "save_checkpoints", lambda: checkpoint_steps.append(trainer.global_step))
    monkeypatch.setattr(trainer, "save_models", lambda: export_steps.append(trainer.global_step))
    await asyncio.wait_for(trainer.train(), timeout=10)
    expected = (
        ([2, 3, 3] if save_at_epoch_end else [3]) if exhaust_early else ([2, 3, 4, 5] if save_at_epoch_end else [3, 5])
    )
    assert checkpoint_steps == export_steps == expected
    assert trainer._run_training.await_count == (2 if exhaust_early else 4)
    trainer.dispatch.finalize_pending_saves.assert_called_once_with("policy")
    trainer.tracker.finish.assert_called_once()


def _make_async_dataloader(num_prompts: int, mini_batch_size: int) -> _AsyncDataloader:
    """Build an _AsyncDataloader over a trivial dataset of `num_prompts` single-prompt batches."""
    dataset = [[{"uid": str(i)}] for i in range(num_prompts)]
    # batch_size=1 (one prompt per draw) and identity collate so each batch is a list with one dict.
    loader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    return _AsyncDataloader(loader, mini_batch_size)


# --------------------------------------------------------------------------------------
# _AsyncStalenessManager
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluation_pause_drains_running_groups_and_retains_their_order():
    manager = _AsyncStalenessManager(4, 4, 1)
    assert await manager.acquire_submission_slot() == 0
    pause = manager.pause_generation(BackgroundFailure())
    enter = asyncio.create_task(pause.__aenter__())
    await asyncio.sleep(0)
    queued = asyncio.create_task(manager.acquire_submission_slot())
    await asyncio.sleep(0)
    assert not enter.done()
    assert not queued.done()
    await manager.on_rollout_accepted()
    await asyncio.wait_for(enter, timeout=2)
    assert manager._stat.accepted == 1
    assert manager._stat.running == 0
    assert not queued.done()
    await pause.__aexit__(None, None, None)
    assert await asyncio.wait_for(queued, timeout=2) == 1
    assert manager._stat.accepted == 1


@pytest.mark.asyncio
async def test_evaluation_pause_propagates_generator_failure_without_hanging():
    manager = _AsyncStalenessManager(4, 4, 1)
    await manager.acquire_submission_slot()
    failure = BackgroundFailure()
    pause = manager.pause_generation(failure)
    enter = asyncio.create_task(pause.__aenter__())
    await asyncio.sleep(0)
    failure.record(ValueError("generation failed"), "generator")
    with pytest.raises(ValueError, match="generation failed"):
        await asyncio.wait_for(enter, timeout=2)
    assert not manager._paused


@pytest.mark.asyncio
async def test_staleness_manager_filter_restores_capacity():
    """Dropping an accepted group via on_rollout_filtered must give producer capacity back.

    This is the deadlock regression: without reclassifying accepted -> filtered, dropped groups
    keep `accepted` climbing against a fixed staleness ceiling and starve the producers.
    """
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # consumer_capacity = (max_staleness_steps + current_global_step) * mini_batch = (1 + 1) * 2 = 4.
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()

    # At the staleness ceiling: accepted == 4 == ceiling, so no producer capacity remains.
    assert mgr._compute_capacity_unlocked() == 0

    await mgr.on_rollout_filtered()

    # Capacity restored by exactly one slot, and accounting is consistent.
    assert mgr._compute_capacity_unlocked() == 1
    assert mgr._stat.accepted == 3
    assert mgr._stat.filtered == 1
    assert mgr._stat.running == 0
    assert mgr._stat.submitted == 4


@pytest.mark.asyncio
async def test_staleness_manager_validate_epoch_end_with_filtered():
    """At epoch end, submitted == accepted + filtered and accepted == trained steps * mini_batch."""
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # Submit and finish 4 groups; drop 2 of them, train on the remaining 2 (one step).
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()
    for _ in range(2):
        await mgr.on_rollout_filtered()

    # One training step completed -> we are now working on global_step 2.
    await mgr.notify_capacity_change(2)
    await mgr.validate_state_at_epoch_end(global_step=2)  # must not raise


# --------------------------------------------------------------------------------------
# _AsyncDataloader
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_dataloader_filtered_uids_tracking():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)

    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}
    assert set(adl.get_consumed_uids_list()) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_async_dataloader_skips_filtered_uids():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])

    # Trained (0, 1) and filtered (2) are all skipped; only the rest are drawn.
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_restores_filtered():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    adl.load_state_from_checkpoint({"0", "1", "2"}, {"2"})

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_without_filtered_is_backward_compatible():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    # Old checkpoints have no filtered set; default treats everything consumed as trained.
    adl.load_state_from_checkpoint({"0", "1"})
    assert adl.num_trained() == 2
    assert adl.get_filtered_uids_list() == []


# --------------------------------------------------------------------------------------
# _drain_next_group
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_next_group_returns_buffered_items_then_exhaustion():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    buffer.put_nowait("b")

    # _drain_next_group uses no instance state, so a bare object stands in for `self`.
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    failure = BackgroundFailure()

    assert await drain(dummy, buffer, done, failure) == "a"
    assert await drain(dummy, buffer, done, failure) == "b"

    # Buffer empty and generators done -> exhausted.
    await FullyAsyncRayPPOTrainer._watch_generators_done([asyncio.create_task(asyncio.sleep(0))], done, failure)
    assert await drain(dummy, buffer, done, failure) is None


@pytest.mark.asyncio
async def test_drain_next_group_drains_remaining_before_exhaustion():
    """If generators finish while items remain, those items are returned before None."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    done.set()  # generators done, but a real item is still buffered

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    assert await drain(dummy, buffer, done, BackgroundFailure()) == "a"
    assert await drain(dummy, buffer, done, BackgroundFailure()) is None


@pytest.mark.asyncio
async def test_drain_next_group_blocks_until_item_arrives():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    async def delayed_put():
        await asyncio.sleep(0.05)
        buffer.put_nowait("x")

    producer = asyncio.create_task(delayed_put())
    assert await drain(dummy, buffer, done, BackgroundFailure()) == "x"
    await producer


@pytest.mark.asyncio
async def test_drain_next_group_raises_when_worker_fails_mid_drain():
    """A worker failure while other workers are still alive must raise, not block or read as exhaustion."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    failure = BackgroundFailure()
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    err = RuntimeError("generator crashed")
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.async_train_dataloader = SimpleNamespace(get_next_non_consumed_data=AsyncMock(side_effect=err))

    async def live_worker():
        await asyncio.sleep(3600)

    tasks = [
        asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer, failure)),
        asyncio.create_task(live_worker()),
    ]
    watcher = asyncio.create_task(FullyAsyncRayPPOTrainer._watch_generators_done(tasks, done, failure))
    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(drain(trainer, buffer, done, failure), timeout=5)
    assert exc_info.value is err
    assert err.__notes__ == ["raised in background generation worker"]
    assert not done.is_set()
    buffer.put_nowait("a")
    with pytest.raises(RuntimeError) as buffered_exc:
        await drain(trainer, buffer, done, failure)
    assert buffered_exc.value is err
    assert buffer.get_nowait() == "a"
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.wait_for(watcher, timeout=5)


# --------------------------------------------------------------------------------------
# _should_keep_group
# --------------------------------------------------------------------------------------


def _trainer_with_tol(tol: float):
    """A stand-in for `self` exposing just the cfg field _should_keep_group reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(algorithm=SimpleNamespace(zero_variance_filter_tol=tol)))
    )


def _group(rewards, loss_masks, uid="u"):
    return GeneratedOutputGroup(
        generator_output={"rewards": rewards, "loss_masks": loss_masks},
        uid=uid,
        global_step_when_scheduled=0,
    )


def test_should_keep_group():
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Zero-variance group -> drop.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0], [[1], [1]])) is False
    # Group with reward spread -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 0.0], [[1], [1]])) is True
    # Singleton -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0], [[1]])) is True
    # Masked trajectories are ignored: two equal live rewards + one masked -> still zero-variance.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0, 0.0], [[1], [1], [0]])) is False
    # Near-equal float rewards within tol -> drop.
    assert keep(_trainer_with_tol(1e-6), _group([0.6667, 0.66670001], [[1], [1]])) is False


def test_reprefix_metrics():
    """generate/X -> generate_<suffix>/X, preserving the leading namespace for tracker grouping."""
    reprefix = FullyAsyncRayPPOTrainer._reprefix_metrics
    out = reprefix(
        {"generate/avg_num_tokens": 10.0, "environment/score": 0.5, "bare": 1},
        "dropped",
    )
    assert out == {
        "generate_dropped/avg_num_tokens": 10.0,
        "environment_dropped/score": 0.5,
        "dropped/bare": 1,
    }


def test_should_keep_group_token_level_rewards():
    """Token-level rewards are collapsed to per-trajectory sequence rewards for the variance check."""
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Two trajectories, both summing to 1.0 -> zero variance -> drop.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [1.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is False
    )
    # One trajectory sums to 1.0, the other to 0.0 -> variance -> keep.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [0.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("epoch", [0, 1])
@pytest.mark.parametrize("num_rows", [24, 27])
async def test_import_synchronous_checkpoint_preserves_remaining_shuffle(workers, epoch, num_rows):
    import copy

    import torch

    rows = [{"uid": str(i)} for i in range(num_rows)]

    def loader(batch_size):
        return StatefulDataLoader(
            rows,
            batch_size=batch_size,
            shuffle=True,
            drop_last=batch_size != 1,
            num_workers=workers,
            generator=torch.Generator().manual_seed(42),
            collate_fn=list,
        )

    synchronous = loader(4)
    for _ in range(epoch):
        list(synchronous)
    iterator = iter(synchronous)
    prefix = [row["uid"] for _ in range(2) for row in next(iterator)]
    state = copy.deepcopy(synchronous.state_dict())
    remaining = [row["uid"] for batch in iterator for row in batch]
    following_epoch = [row["uid"] for batch in synchronous for row in batch]
    asynchronous = _AsyncDataloader(loader(1), 4, reshuffle_each_epoch=True)
    assert asynchronous.import_synchronous_state(state, 4) == set(prefix)
    restored = []
    while (batch := await asynchronous.get_next_non_consumed_data()) is not None:
        restored.append(batch[0]["uid"])
    assert restored == remaining
    # A subsequent async checkpoint must replay the same epoch permutation too.
    epoch_state = copy.deepcopy(asynchronous._train_dataloader_initial_state)
    second = _AsyncDataloader(loader(1), 4, reshuffle_each_epoch=True)
    second.load_state_from_checkpoint(set(prefix), epoch_start_state=epoch_state)
    replay = []
    while (batch := await second.get_next_non_consumed_data()) is not None:
        replay.append(batch[0]["uid"])
    assert replay == remaining
    await asynchronous.reset_at_epoch_end()
    restored = []
    while (batch := await asynchronous.get_next_non_consumed_data()) is not None:
        restored.append(batch[0]["uid"])
    assert restored == following_epoch


@pytest.mark.asyncio
async def test_fixed_batches_ignore_completion_order_and_stream_before_closure():
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 1
    trainer.mini_batch_size = 4
    trainer._pending_generation_batches = {}
    trainer.all_timings = {}
    trainer._phase_gauge = MagicMock()
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(fully_async=SimpleNamespace(streaming_batch_size=2)))
    trainer._begin_streaming_step = MagicMock()
    streamed = []
    trainer._stream_training_groups = lambda groups: streamed.extend(group.uid for group in groups)
    buffer = asyncio.Queue()
    for order in [4, 2, 0, 5, 3, 1]:
        buffer.put_nowait(
            GeneratedOutputGroup(
                {}, str(order), 1, generation_batch_step=order // 4 + 1, generation_submission_order=order
            )
        )
    groups, dropped, exhausted = await asyncio.wait_for(trainer._collect_fixed_batch(buffer, BackgroundFailure()), 2)
    assert [group.uid for group in groups] == ["0", "1", "2", "3"]
    assert streamed == ["2", "0", "3", "1"]
    assert [group.uid for group in trainer._pending_generation_batches[2]] == ["4", "5"]
    assert not dropped and not exhausted


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,staleness", [("one_step_off", 1), ("zero_stale", 0)])
async def test_fixed_submission_preserves_sampler_order_across_capacity_waits(monkeypatch, mode, staleness):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = example_dummy_config()
    trainer.cfg.trainer.fully_async.streaming_batch_size = 0
    trainer.async_mode = mode
    trainer.global_step = 1
    trainer.mini_batch_size = 4
    trainer._generation_submission_lock = asyncio.Lock()
    trainer._generation_completion_order = 0
    trainer._generation_consumption_order = 0
    trainer._pending_generation_batches = {}
    trainer.all_timings = {}
    trainer._phase_gauge = MagicMock()
    trainer.inference_engine_client = SimpleNamespace(weight_version=1)
    loader = StatefulDataLoader([{"uid": str(i)} for i in range(12)], batch_size=1, collate_fn=list)
    trainer.async_train_dataloader = _AsyncDataloader(loader, 4)
    manager = trainer._staleness_manager = _AsyncStalenessManager(3, 4, staleness)

    monkeypatch.setattr(
        "skyrl.train.fully_async_trainer.prepare_generator_input",
        lambda rows, *args: (
            {"batch_metadata": SimpleNamespace(global_step=1), "prompts": [rows[0]["uid"]]},
            [rows[0]["uid"]],
        ),
    )

    async def generate(inputs, disable_tqdm=True):
        await asyncio.sleep(0)
        return {}

    trainer.generator = SimpleNamespace(generate=generate)
    first_waiting, release_first = asyncio.Event(), asyncio.Event()
    reserve = manager.acquire_submission_slot

    async def delayed_first_reservation():
        if not first_waiting.is_set():
            first_waiting.set()
            await release_first.wait()
        return await reserve()

    monkeypatch.setattr(manager, "acquire_submission_slot", delayed_first_reservation)
    buffer, failure = asyncio.Queue(), BackgroundFailure()
    tasks = [asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer, failure)) for _ in range(3)]
    try:
        await asyncio.wait_for(first_waiting.wait(), 2)
        # Other producers can reach admission while the first sampler row waits.
        await asyncio.sleep(0)
        release_first.set()
        for step in range(1, 4):
            trainer.global_step = step
            groups, dropped, exhausted = await asyncio.wait_for(trainer._collect_fixed_batch(buffer, failure), 2)
            assert [group.uid for group in groups] == [str(i) for i in range((step - 1) * 4, step * 4)]
            assert not dropped and not exhausted
            await trainer.async_train_dataloader.mark_consumed_uids(group.uid for group in groups)
            await manager.notify_capacity_change(step + 1)
        # Exhaustion must not reserve a slot or wait for another policy update.
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert manager._stat.submitted == manager._stat.accepted == 12
        assert manager._stat.running == 0
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_weight_publication_barrier_waits_for_entire_admitted_batch():
    manager = _AsyncStalenessManager(4, 2, 1)
    assert [await manager.acquire_submission_slot() for _ in range(4)] == list(range(4))
    for _ in range(3):
        await manager.on_rollout_accepted()
    barrier = asyncio.create_task(manager.wait_for_accepted(4))
    await asyncio.sleep(0)
    assert not barrier.done()
    await manager.on_rollout_accepted()
    await asyncio.wait_for(barrier, 1)
    assert manager._compute_capacity_unlocked() == 0
    await manager.notify_capacity_change(2)
    assert manager._compute_capacity_unlocked() == 2
