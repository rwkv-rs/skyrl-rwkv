"""Record this process's ``GPUModelRunner`` for receive-side sync hooks.

vLLM constructs ``WeightTransferEngine`` instances with the model, not the
worker or runner. SkyRL's receive-side finish hooks need the runner to repair
state that lives outside plain model weights, such as serialized-FP8 KV scale
mirrors. Record the process-local runner after ``GPUModelRunner.load_model`` so
engine hooks can find it later in the same worker process.
"""

from __future__ import annotations

import weakref
from importlib import import_module
from typing import Any

_PATCHED = False
_CURRENT_MODEL_RUNNER: weakref.ReferenceType[Any] | None = None


def apply_model_runner_registry_patch() -> None:
    """Wrap the current runner and the legacy runner on vLLM versions that ship it."""
    global _PATCHED
    if _PATCHED:
        return

    runner_classes = []
    for module_name in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            runner = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
        else:
            runner_classes.append(runner.GPUModelRunner)

    for runner_cls in runner_classes:
        _record_runner_on_load(runner_cls)
    _PATCHED = True


def _record_runner_on_load(runner_cls: type[Any]) -> None:
    original = runner_cls.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        global _CURRENT_MODEL_RUNNER
        _CURRENT_MODEL_RUNNER = weakref.ref(self)
        return result

    load_model.__wrapped__ = original
    runner_cls.load_model = load_model


def current_model_runner() -> Any | None:
    """Return this worker process's ``GPUModelRunner``, if it has loaded."""
    if _CURRENT_MODEL_RUNNER is None:
        return None
    return _CURRENT_MODEL_RUNNER()
