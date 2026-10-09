"""Tests for ``weight_sync/base.py``.

``LoraLoadRequest`` carries the two fields an adapter load needs. It is not a
weight *transfer* -- it names a directory on disk -- so it has no tensor
metadata (``names`` / ``dtypes`` / ``shapes``) of its own.
"""

from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest


class TestLoraLoadRequest:
    def test_json_roundtrip(self):
        """The request crosses to the inference engine as JSON."""
        request = LoraLoadRequest(lora_path="/p", lora_name="tenant-a")
        assert request.to_json_dict() == {"lora_path": "/p", "lora_name": "tenant-a"}
        assert LoraLoadRequest.from_json_dict(request.to_json_dict()) == request
