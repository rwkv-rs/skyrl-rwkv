import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from torch.utils.cpp_extension import get_default_build_root

import torch
from vllm.model_executor.models.rwkv import _load_flashrwkv2
from flashrwkv2 import compile as c
from flashrwkv2.compile import load_extension


def prepare() -> None:
    _load_flashrwkv2()
    result = load_extension()
    assert Path(result.library).is_file()
    commit = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["uv"]["sources"]["vllm"]["rev"]
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]
    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "flashrwkv2": importlib.metadata.version("FlashRWKV2"),
        "vllm_rwkv_commit": commit,
        "compute_capability": result.target,
        "artifact": result.library,
        "build": result.status,
    }
    Path(result.library).with_name("environment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"FlashRWKV2 cache: {Path(result.library).parent}")
    print(f"FlashRWKV2 artifact: {result.library}")
    print(f"FlashRWKV2 build: {result.status}")


def check() -> None:
    _load_flashrwkv2()
    payload = c._cache_payload(torch, c.source_root(), c._capability(torch), c._toolchain(torch))
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = Path(get_default_build_root()) / f"_flashrwkv2_{key[:16]}" / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            f"FlashRWKV2 cache is missing: {manifest}; run --prepare-flashrwkv2 after uv sync --extra rwkv"
        )
    record = json.loads(manifest.read_text())
    artifact = Path(record["library"])
    if record["cache_key"] != key or not artifact.is_file():
        raise SystemExit(f"FlashRWKV2 cache is incomplete: {manifest}; run --prepare-flashrwkv2")
    result = c.load_extension()
    if result.status != "cached":
        raise SystemExit(f"FlashRWKV2 unexpectedly compiled during preflight: {result.status}")
    print(f"FlashRWKV2 cache: {manifest.parent}")
    print(f"FlashRWKV2 artifact: {artifact}")
    print("FlashRWKV2 build: cached / no compilation")


if sys.argv[1:] == ["--check"]:
    check()
else:
    prepare()
