"""
Tests for worker utility functions.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/test_worker_utils.py
"""

from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import (
    all_reduce_metrics,
    get_microbatch_iterator,
    reduce_metrics,
)


@pytest.mark.parametrize(
    "length,expected_sizes",
    [
        (1024, [4, 4]),
        (16368, [4, 4]),
        (16383, [3, 3, 2]),
        (16496, [3, 3, 2]),
        (17407, [3, 3, 2]),
    ],
)
def test_rwkv_microbatches_include_prompt_and_chunk_padding(length, expected_sizes):
    sequences = torch.arange(8).reshape(8, 1).expand(8, length).clone()
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": torch.ones_like(sequences)})
    iterator = get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)
    chunks = list(iterator)
    assert [len(chunk) for chunk in chunks] == expected_sizes
    assert all(len(chunk) * ((length + 15) // 16 * 16) <= 65535 for chunk in chunks)
    combined = iterator.reorder_and_combine_batches(chunks)
    torch.testing.assert_close(combined["sequences"], sequences)


def test_non_rwkv_sample_batching_keeps_existing_shape():
    sequences = torch.ones(8, 17407, dtype=torch.long)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": sequences})
    iterator = get_microbatch_iterator(batch, 4, -1)
    assert [len(chunk) for chunk in iterator] == [4, 4]


def test_rwkv_token_budget_accounts_for_alignment_and_restores_order():
    lengths = [16383, 16368, 16496, 17407, 1000, 3, 17001, 200]
    sequences = torch.arange(8).reshape(8, 1).expand(8, max(lengths)).clone()
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": mask})
    iterator = get_microbatch_iterator(batch, 4, 100000, use_padded_token_count=True)
    chunks = list(iterator)
    for chunk in chunks:
        longest = int(chunk["attention_mask"].sum(-1).max())
        assert len(chunk) * ((longest + 15) // 16 * 16) <= 65535
    combined = iterator.reorder_and_combine_batches(chunks)
    torch.testing.assert_close(combined["sequences"], sequences)


def test_rwkv_rejects_single_sequence_beyond_kernel_rows():
    sequences = torch.ones(1, 65536, dtype=torch.long)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": sequences})
    with pytest.raises(ValueError, match="CUDA kernel row limit"):
        get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)


def test_rwkv_tail_microbatch_keeps_global_token_mean_gradient():
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        apply_loss_reduction_to_advantages_minibatch,
        ppo_policy_loss,
    )
    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig().trainer.algorithm
    cfg.github_strict_loss = True
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(4).reshape(4, 1).expand(4, 16496).clone(),
            "attention_mask": torch.ones(4, 16496, dtype=torch.long),
            "loss_mask": torch.ones(4, 2),
            "advantages": torch.tensor([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]]),
        }
    )
    batch["advantages"] = apply_loss_reduction_to_advantages_minibatch(
        batch["advantages"],
        batch["loss_mask"],
        "token_mean",
        4,
        16496,
    )
    iterator = get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)
    for chunk in iterator:
        scores = (chunk["sequences"][:, :1].float() + 1).expand(-1, 2)
        loss, _ = ppo_policy_loss(
            parameter * scores, torch.zeros_like(scores), chunk["advantages"], cfg, chunk["loss_mask"]
        )
        loss.backward()
    reference = torch.nn.Parameter(torch.tensor(0.0))
    scores = (batch["sequences"][:, :1].float() + 1).expand(-1, 2)
    loss, _ = ppo_policy_loss(
        reference * scores, torch.zeros_like(scores), batch["advantages"], cfg, batch["loss_mask"]
    )
    loss.backward()
    torch.testing.assert_close(parameter.grad, reference.grad)
    assert parameter.grad.abs() > 0


def test_rwkv_dynamic_microbatch_mixes_four_short_and_three_long_without_reordering_outputs():
    lengths = [17407, 1000, 17000, 16368, 800, 16496, 1200, 16600, 16900, 16800]
    sequences = torch.arange(len(lengths)).reshape(-1, 1).expand(-1, max(lengths)).clone()
    mask = torch.arange(max(lengths)).unsqueeze(0) >= (max(lengths) - torch.tensor(lengths)).unsqueeze(1)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": mask})
    iterator = get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)
    chunks = list(iterator)
    assert [len(chunk) for chunk in chunks] == [4, 3, 3]
    for chunk in chunks:
        longest = int(chunk["attention_mask"].sum(-1).max())
        assert len(chunk) * ((longest + 15) // 16 * 16) <= 65535
    torch.testing.assert_close(iterator.reorder_and_combine_batches(chunks)["sequences"], sequences)
    items = [[{"sample": int(value)} for value in chunk["sequences"][:, 0]] for chunk in chunks]
    assert [item["sample"] for item in iterator.reorder_and_combine_items(items)] == list(range(len(lengths)))


@pytest.mark.parametrize("token_budget", [-1, 65535, 100000])
def test_rwkv_dynamic_microbatch_never_exceeds_configured_sample_cap(token_budget):
    sequences = torch.ones(13, 100, dtype=torch.long)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": sequences})
    iterator = get_microbatch_iterator(batch, 4, token_budget, use_padded_token_count=True)
    assert [len(chunk) for chunk in iterator] == [4, 4, 4, 1]


def test_rwkv_dynamic_padding_adds_no_loss_and_is_removed_from_outputs(monkeypatch):
    from skyrl.backends.skyrl_train.workers.worker_utils import TokenBasedBatchIterator

    monkeypatch.setattr(TokenBasedBatchIterator, "_sync_num_microbatches", lambda self: len(self._microbatches) + 1)
    sequences = torch.arange(4).reshape(4, 1).expand(4, 100).clone()
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": torch.ones_like(sequences)})
    batch.metadata = {"response_length": 99}
    iterator = get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)
    chunks = list(iterator)
    assert [len(chunk) for chunk in chunks] == [4, 1]
    assert chunks[-1]["loss_mask"].count_nonzero() == 0
    assert chunks[-1].metadata["is_padding_batch"]
    actual = iterator.reorder_and_combine_batches(chunks)
    torch.testing.assert_close(actual["sequences"], sequences)


def test_rwkv_dynamic_mixed_sizes_keep_token_mean_gradient():
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        apply_loss_reduction_to_advantages_minibatch,
        ppo_policy_loss,
    )
    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig().trainer.algorithm
    cfg.github_strict_loss = True
    lengths = [1000, 17407, 1500, 17407, 2000, 17407, 3000]
    sequences = torch.arange(7).reshape(-1, 1).expand(-1, max(lengths)).clone()
    attention = torch.arange(max(lengths)).unsqueeze(0) >= (max(lengths) - torch.tensor(lengths)).unsqueeze(1)
    masks = torch.tensor([[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 1]]).float()
    advantages = torch.tensor([1, -1, 2, -2, 3, -3, 4]).float().unsqueeze(1).expand(-1, 2)
    normalized = apply_loss_reduction_to_advantages_minibatch(advantages, masks, "token_mean", 4, max(lengths))
    batch = TrainingInputBatch(
        {"sequences": sequences, "attention_mask": attention, "advantages": normalized, "loss_mask": masks}
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    iterator = get_microbatch_iterator(batch, 4, -1, use_padded_token_count=True)
    assert [len(chunk) for chunk in iterator] == [4, 3]
    for chunk in iterator:
        scores = (chunk["sequences"][:, :1].float() + 1).expand(-1, 2)
        loss, _ = ppo_policy_loss(
            parameter * scores, torch.zeros_like(scores), chunk["advantages"], cfg, chunk["loss_mask"]
        )
        loss.backward()
    reference = torch.nn.Parameter(torch.tensor(0.0))
    scores = (sequences[:, :1].float() + 1).expand(-1, 2)
    loss, _ = ppo_policy_loss(reference * scores, torch.zeros_like(scores), normalized, cfg, masks)
    loss.backward()
    torch.testing.assert_close(parameter.grad, reference.grad)


def test_rwkv_critic_dynamic_microbatch_uses_sample_weights(monkeypatch):
    from types import SimpleNamespace

    from skyrl.backends.skyrl_train.workers.worker import CriticWorkerBase

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.workers.worker.all_reduce_metrics", lambda metrics, *args, **kwargs: metrics
    )
    worker = CriticWorkerBase.__new__(CriticWorkerBase)
    worker.cfg = SimpleNamespace(micro_train_batch_size_per_gpu=4, max_tokens_per_microbatch=-1)
    worker.model = SimpleNamespace(is_rwkv=True)
    worker.strategy = None
    worker.device_mesh = SimpleNamespace(get_group=lambda name: None)
    worker._micro_batches_accumulated = 0
    weights = []

    def forward_backward(experience, microbatch_weight=None):
        weights.append(microbatch_weight)
        return {"critic_loss": microbatch_weight}

    worker._forward_backward_micro = forward_backward
    lengths = [1000] * 4 + [17407] * 3
    sequences = torch.ones(7, max(lengths), dtype=torch.long)
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": mask})
    batch.metadata = {"response_length": 2}
    worker.forward_backward(batch)
    assert weights == pytest.approx([4 / 7, 3 / 7])
    assert worker._micro_batches_accumulated == 0


class TestReduceMetrics:
    def test_reduce_metrics_max_suffix(self):
        """Keys ending in _max should use max reduction."""
        metrics = {"is_ratio_max": [1.0, 5.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["is_ratio_max"] == 5.0

    def test_reduce_metrics_min_suffix(self):
        """Keys ending in _min should use min reduction."""
        metrics = {"is_ratio_min": [1.0, 5.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["is_ratio_min"] == 1.0

    def test_reduce_metrics_mean_default(self):
        """Keys without _max/_min/_loss suffix should use mean reduction."""
        metrics = {"entropy": [1.0, 2.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["entropy"] == 2.0

    def test_reduce_metrics_loss_default_mean(self):
        """_loss keys default to mean when sum_loss_metrics=False."""
        metrics = {"policy_loss": [1.0, 2.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["policy_loss"] == 2.0

    def test_reduce_metrics_sum_loss_metrics(self):
        """_loss keys are summed when sum_loss_metrics=True."""
        metrics = {"policy_loss": [1.0, 2.0, 3.0]}
        result = reduce_metrics(metrics, sum_loss_metrics=True)
        assert result["policy_loss"] == 6.0

    def test_reduce_metrics_mixed(self):
        """Test mixed metric types are reduced correctly."""
        metrics = {
            "is_ratio_max": [1.0, 10.0],
            "is_ratio_min": [0.5, 2.0],
            "policy_loss": [1.0, 3.0],
            "entropy": [1.0, 3.0],
        }
        result = reduce_metrics(metrics, sum_loss_metrics=True)
        assert result["is_ratio_max"] == 10.0
        assert result["is_ratio_min"] == 0.5
        assert result["policy_loss"] == 4.0  # sum
        assert result["entropy"] == 2.0  # mean

    def test_reduce_metrics_mtp_loss_is_averaged_not_summed(self):
        """mtp_loss / draft_loss are per-token MEANS, not pre-scaled sums. They must be averaged
        across microbatches even when sum_loss_metrics=True, or a true ~0.5 reads as ~N*0.5."""
        metrics = {
            "policy_loss": [1.0, 3.0],  # pre-scaled -> sum
            "mtp_loss": [0.5, 0.5, 0.5, 0.5],  # mean -> stays 0.5, NOT 2.0
            "draft_loss": [0.6, 0.4],  # mean -> 0.5
        }
        result = reduce_metrics(metrics, sum_loss_metrics=True)
        assert result["policy_loss"] == 4.0  # still summed
        assert result["mtp_loss"] == 0.5  # averaged, not 2.0
        assert result["draft_loss"] == 0.5

    def test_reduce_metrics_single_value(self):
        """Test reduction with single value lists."""
        metrics = {
            "is_ratio_max": [5.0],
            "is_ratio_min": [0.5],
            "policy_loss": [1.5],
        }
        result = reduce_metrics(metrics)
        assert result["is_ratio_max"] == 5.0
        assert result["is_ratio_min"] == 0.5
        assert result["policy_loss"] == 1.5

    def test_reduce_metrics_empty_raises(self):
        """Test that empty list raises assertion error."""
        metrics = {"policy_loss": []}
        with pytest.raises(AssertionError, match="No metrics for key"):
            reduce_metrics(metrics)


class TestAllReduceMetrics:
    @pytest.mark.parametrize("sum_loss_metrics", [True, False])
    def test_all_reduce_metrics_separates_by_suffix(self, sum_loss_metrics):
        """Verify metrics are correctly separated by suffix and reduced with correct ops."""
        strategy = MagicMock()

        # Mock all_reduce to return the input dict unchanged but track calls
        def mock_all_reduce(d, op, group=None):
            return {k: v for k, v in d.items()}

        strategy.all_reduce.side_effect = mock_all_reduce

        metrics = {
            "is_ratio_max": 10.0,
            "is_ratio_min": 0.1,
            "policy_loss": 1.5,
            "entropy": 0.5,
        }

        _ = all_reduce_metrics(metrics, strategy, sum_loss_metrics=sum_loss_metrics)

        # Verify all_reduce was called 4 times
        assert strategy.all_reduce.call_count == 4

        # Check that the correct ops were used
        calls = strategy.all_reduce.call_args_list

        # Find which call used which op
        ops_and_keys = []
        for call in calls:
            args, kwargs = call
            data_dict = args[0]
            op = kwargs.get("op") if kwargs else args[1]
            ops_and_keys.append((op, set(data_dict.keys())))

        # Verify mean metrics (entropy)
        mean_call = [c for c in ops_and_keys if c[0] == "mean"][0]
        if sum_loss_metrics:
            assert mean_call[1] == {"entropy"}
        else:
            assert mean_call[1] == {"entropy", "policy_loss"}

        # Verify sum metrics (explicit sum_keys)
        sum_call = [c for c in ops_and_keys if c[0] == "sum"][0]
        if sum_loss_metrics:
            assert sum_call[1] == {"policy_loss"}
        else:
            assert sum_call[1] == set()

        # Verify min metrics
        min_call = [c for c in ops_and_keys if c[0] == "min"][0]
        assert min_call[1] == {"is_ratio_min"}

        # Verify max metrics
        max_call = [c for c in ops_and_keys if c[0] == "max"][0]
        assert max_call[1] == {"is_ratio_max"}

    def test_all_reduce_metrics_average_loss_metrics(self):
        """Verify _loss keys are averaged when sum_loss_metrics=False."""
        strategy = MagicMock()

        # Mock all_reduce to return the input dict unchanged but track calls
        def mock_all_reduce(d, op, group=None):
            return {k: v for k, v in d.items()}

        strategy.all_reduce.side_effect = mock_all_reduce

        metrics = {"critic_loss": 1.5, "entropy": 0.5}
        _ = all_reduce_metrics(metrics, strategy, sum_loss_metrics=False)

        # Both should be mean-reduced (critic_loss is NOT summed without the flag)
        ops_and_keys = []
        for args, kwargs in strategy.all_reduce.call_args_list:
            data_dict = args[0]
            op = kwargs["op"]
            if data_dict:
                ops_and_keys.append((op, set(data_dict.keys())))

        mean_call = [c for c in ops_and_keys if c[0] == "mean"][0]
        assert mean_call[1] == {"critic_loss", "entropy"}

    def test_all_reduce_metrics_returns_merged_results(self):
        """Verify results from all reductions are merged correctly."""
        strategy = MagicMock()

        # Mock all_reduce to modify values based on op
        def mock_all_reduce(d, op, group=None):
            if op == "mean":
                return {k: v * 2 for k, v in d.items()}  # Double for mean
            elif op == "sum":
                return {k: v * 4 for k, v in d.items()}  # Quadruple for sum
            elif op == "min":
                return {k: v / 2 for k, v in d.items()}  # Halve for min
            elif op == "max":
                return {k: v * 3 for k, v in d.items()}  # Triple for max
            return d

        strategy.all_reduce.side_effect = mock_all_reduce

        metrics = {
            "is_ratio_max": 10.0,
            "is_ratio_min": 0.1,
            "policy_loss": 1.5,
            "entropy": 0.5,
        }

        result = all_reduce_metrics(metrics, strategy, sum_loss_metrics=True)

        # Check all keys are present
        assert "is_ratio_max" in result
        assert "is_ratio_min" in result
        assert "policy_loss" in result
        assert "entropy" in result

        # Check values were transformed correctly
        assert result["is_ratio_max"] == 30.0  # 10.0 * 3 (max op)
        assert result["is_ratio_min"] == 0.05  # 0.1 / 2 (min op)
        assert result["policy_loss"] == 6.0  # sum op
        assert result["entropy"] == 1.0  # 0.5 * 2 (mean op)
