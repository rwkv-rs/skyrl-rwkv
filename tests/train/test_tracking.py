"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident
from unittest.mock import MagicMock, patch

import pytest
import wandb

from skyrl.train.utils.rwkv_trajectory_logging import RWKVTrajectoryLogger
from skyrl.train.utils.tracking import Tracking


def test_wandb_init_receives_tags():
    """Tags passed to Tracking are forwarded to wandb.init."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
            tags=["foo", "bar"],
        )

        wandb_mock.init.assert_called_once()
        kwargs = wandb_mock.init.call_args.kwargs
        assert kwargs["tags"] == ["foo", "bar"]
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "exp"


def test_wandb_init_tags_default_none():
    """When tags are not provided, wandb.init receives tags=None."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
        )

        wandb_mock.init.assert_called_once()
        assert wandb_mock.init.call_args.kwargs["tags"] is None


def test_vllm_history_metrics_disable_automatic_summaries_once():
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        tracker = Tracking("proj", "exp", backend="wandb", config={})
        tracker.log({"vllm/train/generation_throughput_tok_s": 50, "train/loss": 1}, step=1)
        tracker.log({"vllm/train/generation_throughput_tok_s": 25}, step=2)
        wandb_mock.define_metric.assert_called_once_with("vllm/train/generation_throughput_tok_s", summary="none")
        assert wandb_mock.log.call_count == 2
        assert wandb_mock.log.call_args.kwargs["data"] == {"vllm/train/generation_throughput_tok_s": 25}
        tracker.finish()


def test_finish_removes_regular_sdk_summary_but_preserves_other_metrics():
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        summary = {"vllm/train/rate": 25, "train/loss": 1}
        wandb_mock.run.summary = summary
        tracker = Tracking("proj", "exp", backend="wandb", config={})
        tracker.log({"vllm/train/rate": 25}, step=1)
        tracker.update_summary({"vllm_correct_aggregate/train/rate": 30})
        tracker.finish()
        assert summary == {"train/loss": 1, "vllm_correct_aggregate/train/rate": 30, "run_status": "success"}
        wandb_mock.Api.assert_not_called()
        wandb_mock.finish.assert_called_once()


@pytest.fixture
def tracker():
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
        tracking = Tracking("proj", "exp", backend="wandb", config={})
        try:
            yield tracking
        finally:
            tracking.finish()


def test_incremental_table_adds_only_new_rows(tracker):
    with patch.object(wandb.Table, "add_data", autospec=True, side_effect=wandb.Table.add_data) as add_data:
        for step in range(1, 101):
            tracker.log_samples_to_table("train", ["step", "mask"], [(step, [1] * 1024)], step, incremental=True)
        table = tracker._sample_tables["train"]
        assert table.log_mode == "INCREMENTAL"
        assert len(table.data) == 100
        assert add_data.call_count == 100
        assert all(call.args[0]["train"] is table for call in wandb.log.call_args_list)


def test_default_table_keeps_logged_snapshots_unchanged(tracker):
    tracker.log_samples_to_table("train", ["step"], [(1,)], 1)
    first = wandb.log.call_args.args[0]["train"]
    tracker.log_samples_to_table("train", ["step"], [(2,)], 2)
    second = wandb.log.call_args.args[0]["train"]
    assert first is not second
    assert first.data == [[1]]
    assert second.data == [[1], [2]]
    assert tracker._log_executor is None


def test_background_tables_and_metrics_preserve_step_order_and_finish(tracker):
    started = Event()
    release = Event()
    writes = []
    caller_thread = get_ident()

    def write(data, step, commit=None):
        assert get_ident() != caller_thread
        if not writes:
            started.set()
            assert release.wait(10)
        writes.append((next(iter(data)), step, commit))

    wandb.log.side_effect = write
    wandb.finish.side_effect = lambda **kwargs: writes.append(("finish", None, None))
    try:
        tracker.log_samples_to_table("train", ["step"], [(1,)], 1, incremental=True, asynchronous=True)
        assert started.wait(5)
        # These calls return while the first table write is still blocked.
        tracker.log({"reward": 1.0}, 1, commit=True)
        tracker.log_samples_to_table("train", ["step"], [(2,)], 2, incremental=True, asynchronous=True)
        tracker.log({"reward": 2.0}, 2, commit=True)
        assert writes == []
        release.set()
        tracker.finish()
        assert writes == [
            ("train", 1, None),
            ("reward", 1, True),
            ("train", 2, None),
            ("reward", 2, True),
            ("finish", None, None),
        ]
    finally:
        release.set()


def test_background_queue_applies_backpressure(tracker):
    started = Event()
    release = Event()
    backpressure = Event()
    tracker._MAX_PENDING_LOGS = 2

    def write(*args, **kwargs):
        started.set()
        assert release.wait(10)

    wandb.log.side_effect = write
    try:
        tracker.log_samples_to_table("train", ["step"], [(1,)], 1, incremental=True, asynchronous=True)
        assert started.wait(5)
        tracker.log({"reward": 1.0}, 1)
        first = tracker._log_futures[0]
        result = first.result

        def wait_for_write():
            backpressure.set()
            return result()

        with patch.object(first, "result", side_effect=wait_for_write), ThreadPoolExecutor(max_workers=1) as executor:
            try:
                pending = executor.submit(tracker.log, {"reward": 2.0}, 2)
                assert backpressure.wait(5)
                assert len(tracker._log_futures) < tracker._MAX_PENDING_LOGS
                assert not pending.done()
            finally:
                release.set()
            pending.result(timeout=5)
        tracker.finish()
        assert wandb.log.call_count == 3
    finally:
        release.set()


@pytest.mark.parametrize("consumer", ["log", "finish"])
def test_background_writer_errors_are_visible(tracker, consumer):
    wandb.log.side_effect = RuntimeError("table log failed")
    tracker.log_samples_to_table("train", ["step"], [(1,)], 1, incremental=True, asynchronous=True)
    assert isinstance(tracker._log_futures[0].exception(timeout=5), RuntimeError)
    with pytest.raises(RuntimeError, match="table log failed"):
        if consumer == "log":
            tracker.log({"reward": 1.0}, 1)
        else:
            tracker.finish()
    tracker.finish()
    wandb.finish.assert_called()


def test_non_wandb_sample_logging_does_not_start_background_writer():
    tracker = Tracking("proj", "exp", backend="console")
    tracker.log_samples_to_table("train", ["step"], [(1,)], 1, incremental=True, asynchronous=True)
    assert tracker._log_executor is None
    assert not hasattr(tracker, "_sample_tables")


def test_rwkv_trajectory_logging_opts_into_incremental_background_writer():
    tracker = MagicMock(backend="wandb")
    tokenizer = MagicMock(eos_token_id=0)
    tokenizer.decode.return_value = "answer"
    RWKVTrajectoryLogger().log(
        tracker=tracker,
        num_samples=1,
        prompts=["question"],
        generator_output={"response_ids": [[42, 0]], "rewards": [1.0], "loss_masks": [[1, 1]]},
        tokenizer=tokenizer,
        global_step=3,
        wandb_key="trajectories/train",
    )
    kwargs = tracker.log_samples_to_table.call_args.kwargs
    assert kwargs["incremental"] is True
    assert kwargs["asynchronous"] is True
    assert kwargs["samples"][0][0] == 3
    assert kwargs["columns"] == list(RWKVTrajectoryLogger().columns)


def test_offline_wandb_serializes_only_incremental_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    tracker = Tracking("tracking-tests", "incremental", backend="wandb", config={})
    run_dir = Path(wandb.run.dir)
    try:
        for step in (1, 2):
            tracker.log_samples_to_table(
                "train", ["step", "mask"], [(step, [1] * 1024)], step, incremental=True, asynchronous=True
            )
            tracker.log({"reward": float(step)}, step, commit=True)
    finally:
        tracker.finish()
    tables = [json.loads(path.read_text()) for path in run_dir.glob("media/table/*.table.json")]
    assert len(tables) == 2
    assert all(len(table["data"]) == 1 for table in tables)
    assert sorted(table["data"][0][0] for table in tables) == [1, 2]
