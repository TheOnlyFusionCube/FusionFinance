#!/usr/bin/env python3
"""Build a deterministic, allowlisted, secret-scanned submission archive."""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation/FusionFinance_GitHub_Submission.zip"
ROOT_FILES = (
    ".dockerignore",
    ".gitignore",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "requirements.txt",
    "vercel.json",
)
INCLUDE_DIRECTORIES = (
    ".github",
    "alpha",
    "configs",
    "demo",
    "docs",
    "evidence",
    "presentation",
    "results",
    "scripts",
    "tests",
)
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "frames",
    "scratchpad",
    "secrets",
    "video-work",
}
SKIP_SUFFIXES = {
    ".aif",
    ".aiff",
    ".csv",
    ".m4a",
    ".mp3",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyo",
    ".wav",
    ".zip",
}
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PUBLIC_EXTENSIONLESS_TEXT_PATHS = {
    "alpha/filing_alpha/NOTICE",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PRIVATE_FILENAMES = {
    "voice.wav",
    "presenter-reference.jpg",
    "presenter-reference.jpeg",
    "presenter-reference.png",
}
LOCAL_ONLY_PATHS = {
    "evidence/market/barebone_window_ohlcv.json",
    "evidence/narrative/barebone_window_events.jsonl",
}
PUBLIC_BINARY_PATHS = {
    "docs/assets/dashboard.png",
    "presentation/FusionFinance_Demo.mp4",
    "presentation/FusionFinance_Submission.pdf",
}
def _max_file_bytes(relative: Path) -> int:
    """Return the public size budget, or zero for any unapproved file shape."""
    relative_text = relative.as_posix()
    if relative_text in PUBLIC_BINARY_PATHS:
        if relative_text == "presentation/FusionFinance_Demo.mp4":
            return 250 * 1024 * 1024
        return 25 * 1024 * 1024
    if _is_text_path(relative):
        return 25 * 1024 * 1024
    return 0


def _is_text_path(relative: Path) -> bool:
    relative_text = relative.as_posix()
    return (
        (len(relative.parts) == 1 and relative_text in ROOT_FILES)
        or relative_text in PUBLIC_EXTENSIONLESS_TEXT_PATHS
        or relative.suffix.lower() in TEXT_SUFFIXES
    )


def _is_utf8_text(path: Path) -> bool:
    try:
        payload = path.read_bytes()
        payload.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return b"\x00" not in payload


def _is_allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or path.is_symlink() or not path.is_file():
        return False
    if any(part in SKIP_PARTS for part in relative.parts):
        return False
    if path.name.lower() in PRIVATE_FILENAMES:
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if path.name.startswith(".env") or path.name == ".DS_Store":
        return False
    if relative.as_posix() in LOCAL_ONLY_PATHS:
        return False
    maximum_bytes = _max_file_bytes(relative)
    if maximum_bytes == 0 or path.stat().st_size > maximum_bytes:
        return False
    if relative.as_posix() in PUBLIC_BINARY_PATHS:
        return True
    return _is_utf8_text(path)


def collect_files(root: Path = ROOT) -> tuple[Path, ...]:
    if root != ROOT:
        raise ValueError("archive collection is bound to the repository root")
    candidates = [ROOT / name for name in ROOT_FILES]
    for directory in INCLUDE_DIRECTORIES:
        base = ROOT / directory
        if base.is_dir():
            candidates.extend(base.rglob("*"))
    return tuple(sorted({path for path in candidates if _is_allowed(path)}))


def _scan_for_secrets(files: tuple[Path, ...]) -> None:
    findings: list[str] = []
    for path in files:
        if not _is_text_path(path.relative_to(ROOT)):
            continue
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            findings.append(path.relative_to(ROOT).as_posix())
    if findings:
        raise ValueError("probable secret(s): " + ", ".join(findings))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_archive(output: Path = OUTPUT) -> tuple[int, str]:
    files = collect_files()
    _scan_for_secrets(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            hashes.append(f"{hashlib.sha256(payload).hexdigest()}  {relative}")
            archive.writestr(_zip_info(f"fusionfinance/{relative}"), payload)
        manifest = ("\n".join(hashes) + "\n").encode("utf-8")
        archive.writestr(_zip_info("fusionfinance/SHA256SUMS.txt"), manifest)
    return len(files), hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> int:
    count, digest = build_archive()
    print(f"wrote {OUTPUT} ({count} files, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
