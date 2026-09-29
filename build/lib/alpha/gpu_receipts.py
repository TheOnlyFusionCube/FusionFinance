"""Hash-chained GPU run manifests with reproducibility and memory evidence."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        stat = os.fstat(descriptor)
        if stat.st_size > 10 * 1024 * 1024:
            raise RuntimeError(f"untracked receipt file exceeds 10 MiB: {relative}")
        hasher = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            hasher.update(chunk)
        return {"path": relative, "size": stat.st_size, "sha256": hasher.hexdigest()}
    finally:
        os.close(descriptor)


def _git_output(repo_root: Path, arguments: list[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git receipt command failed: {arguments[0]}") from exc


def git_state(repo_root: str | Path = ".") -> tuple[str, str]:
    """Hash scoped source changes without reading env files or agent configuration."""
    root = Path(repo_root).resolve()
    scopes = ["alpha", "demo", "scripts", "docs", "tests"]
    commit = _git_output(root, ["rev-parse", "HEAD"]).decode().strip() or "unknown"
    tracked_diff = _git_output(
        root, ["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--", *scopes]
    )
    untracked = _git_output(
        root, ["ls-files", "-z", "--others", "--exclude-standard", "--", *scopes]
    ).split(b"\0")
    allowed_suffixes = {".py", ".sh", ".md"}
    records = []
    for encoded_relative in sorted(item for item in untracked if item):
        relative = encoded_relative.decode("utf-8")
        path = root / relative
        if path.suffix not in allowed_suffixes or ".env" in path.name or not path.is_file():
            continue
        if path.is_symlink() or root not in path.resolve().parents:
            raise RuntimeError(f"unsafe untracked receipt path: {relative}")
        records.append(_file_record(path, relative))
    dirty = canonical_hash({
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked": records,
    })
    return commit, dirty


def _device_metadata() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"torch": None, "error": type(exc).__name__}
    metadata: dict[str, Any] = {
        "torch": str(torch.__version__),
        "rocm_hip": str(torch.version.hip) if torch.version.hip else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.update({
            "device_name": str(props.name),
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
        })
    return metadata


@contextmanager
def gpu_memory_stage(device: str = "cuda:0"):
    """Capture per-stage peak allocated/reserved VRAM in bytes."""
    receipt: dict[str, int] = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    try:
        import torch
    except (ImportError, OSError):
        torch = None
    enabled = bool(torch is not None and device.startswith("cuda") and torch.cuda.is_available())
    if enabled:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield receipt
    finally:
        if enabled:
            torch.cuda.synchronize()
            receipt.update({
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            })


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    feature_hash: str,
    universe_hash: str,
    fold_hash: str,
    seed: int,
    autocast_dtype: str,
    stages: Mapping[str, Mapping[str, Any]],
    previous_receipt_hash: str | None = None,
    repo_root: str | Path = ".",
    git_commit: str | None = None,
    dirty_hash: str | None = None,
    allow_test_git_state: bool = False,
) -> dict[str, Any]:
    if (git_commit is None) != (dirty_hash is None):
        raise ValueError("git_commit and dirty_hash must be supplied together")
    if git_commit is not None and not allow_test_git_state:
        raise ValueError("caller-supplied git state is test-only")
    if git_commit is None:
        git_commit, dirty_hash = git_state(repo_root)
    hexadecimal = set("0123456789abcdef")
    if (len(git_commit) not in {40, 64} or any(char not in hexadecimal for char in git_commit) or
            len(dirty_hash) != 64 or any(char not in hexadecimal for char in dirty_hash)):
        raise ValueError("invalid git receipt digest")
    digests = [*input_hashes.values(), feature_hash, universe_hash, fold_hash]
    if any(len(value) != 64 or any(char not in hexadecimal for char in value) for value in digests):
        raise ValueError("input/feature/universe/fold hashes must be lowercase SHA-256")
    if previous_receipt_hash is not None and (
        len(previous_receipt_hash) != 64 or
        any(char not in "0123456789abcdef" for char in previous_receipt_hash)
    ):
        raise ValueError("previous_receipt_hash must be lowercase SHA-256")
    payload = {
        "schema_version": 1,
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit,
        "dirty_hash": dirty_hash,
        "config_hash": canonical_hash(dict(config)),
        "input_hashes": dict(sorted(input_hashes.items())),
        "feature_hash": feature_hash,
        "universe_hash": universe_hash,
        "fold_hash": fold_hash,
        "device": _device_metadata(),
        "seed": int(seed),
        "autocast_dtype": str(autocast_dtype),
        "stages": {name: dict(values) for name, values in sorted(stages.items())},
        "previous_receipt_hash": previous_receipt_hash,
    }
    return {**payload, "receipt_hash": canonical_hash(payload)}


def verify_run_manifest(manifest: Mapping[str, Any], *,
                        expected_previous_hash: str | None = None) -> bool:
    payload = dict(manifest)
    receipt_hash = payload.pop("receipt_hash", None)
    if receipt_hash != canonical_hash(payload):
        return False
    return payload.get("previous_receipt_hash") == expected_previous_hash


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(manifest), sort_keys=True, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
