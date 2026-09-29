"""Content-addressed record/replay cache for LLM calls.

LLM calls are the expensive, non-deterministic part of an LLM-first fund. This
wrapper keys every response by SHA-256 of (model, stage, canonical request), so
a recorded run can be replayed byte-for-byte: re-running a backtest costs
nothing, and anyone holding the cache can reproduce every decision. In
``replay`` mode a cache miss fails closed rather than silently calling a model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

from alpha.agents.models import AnalystRequest
from alpha.fund.ideas import IdeaRequest

CacheMode = Literal["record", "replay", "passthrough"]


class CacheMiss(RuntimeError):
    """Replay mode received a request that was never recorded."""


@dataclass
class RecordReplayProvider:
    inner: object
    directory: Path
    mode: CacheMode = "record"
    hits: int = 0
    misses: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("record", "replay", "passthrough"):
            raise ValueError("mode must be record, replay, or passthrough")
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def timeout_seconds(self) -> float | None:
        return getattr(self.inner, "timeout_seconds", None)

    def analyze(self, request: AnalystRequest) -> str:
        return self._call("analyze", request.model_dump_json(), lambda: self.inner.analyze(request))

    def originate(self, request: IdeaRequest) -> str:
        return self._call(
            "originate", request.model_dump_json(), lambda: self.inner.originate(request)
        )

    def key(self, stage: str, canonical_request: str) -> str:
        payload = json.dumps(
            {"model": self.model_id, "stage": stage, "request": canonical_request},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _call(self, stage: str, canonical_request: str, compute) -> str:
        key = self.key(stage, canonical_request)
        path = self.directory / f"{key}.json"
        if self.mode != "passthrough" and path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.mode == "replay":
            raise CacheMiss(f"no recorded {stage} response for {key[:12]}")
        response = compute()
        with self._lock:
            self.misses += 1
        if self.mode == "record" and isinstance(response, str):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"model": self.model_id, "stage": stage, "response": response}),
                encoding="utf-8",
            )
            temporary.replace(path)
        return response


__all__ = ["CacheMiss", "RecordReplayProvider"]
