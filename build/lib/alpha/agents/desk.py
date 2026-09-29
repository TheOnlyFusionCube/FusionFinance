"""Concurrent execution boundary for independent analyst roles."""
from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import Future, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from queue import Full, Queue
from threading import Lock, Thread

from pydantic import ValidationError

from alpha.agents.models import (
    ANALYST_ROLES,
    AnalystReport,
    AnalystRequest,
    AnalystRole,
    SealedSourceSnapshot,
    TradeProposal,
)
from alpha.agents.providers import AnalystProvider


class AnalystOutputError(RuntimeError):
    """A role failed or returned data outside the strict report schema."""


_MAX_OUTPUT_CHARACTERS = 100_000


@dataclass(frozen=True, slots=True)
class _AnalystTask:
    future: Future[AnalystReport]
    provider: AnalystProvider
    request: AnalystRequest
    role: AnalystRole


class _BoundedDaemonPool:
    """Fixed daemon workers with a bounded fail-closed queue."""

    def __init__(self, worker_count: int) -> None:
        self._worker_count = worker_count
        self._queue: Queue[_AnalystTask] = Queue(maxsize=worker_count * 4)
        self._start_lock = Lock()
        self._started = False

    def submit(
        self,
        provider: AnalystProvider,
        request: AnalystRequest,
        role: AnalystRole,
    ) -> Future[AnalystReport]:
        self._ensure_started()
        future: Future[AnalystReport] = Future()
        try:
            self._queue.put_nowait(
                _AnalystTask(
                    future=future,
                    provider=provider,
                    request=request,
                    role=role,
                )
            )
        except Full:
            future.set_exception(AnalystOutputError("analyst worker pool is saturated"))
        return future

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            for index in range(self._worker_count):
                Thread(
                    target=self._worker,
                    name=f"fusion-analyst-{index + 1}",
                    daemon=True,
                ).start()
            self._started = True

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if not task.future.set_running_or_notify_cancel():
                    continue
                try:
                    result = _invoke_and_validate(
                        task.provider,
                        task.request,
                        task.role,
                    )
                except AnalystOutputError as exc:
                    task.future.set_exception(exc)
                except BaseException:
                    task.future.set_exception(
                        AnalystOutputError(f"{task.role} analyst provider failed")
                    )
                else:
                    task.future.set_result(result)
            finally:
                self._queue.task_done()


_EXECUTOR = _BoundedDaemonPool(worker_count=len(ANALYST_ROLES))


def validate_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.01 <= float(value) <= 300.0
    ):
        raise ValueError("analyst timeout must be finite and in [0.01, 300] seconds")
    return float(value)


@dataclass(frozen=True, slots=True)
class AgentDesk:
    provider: AnalystProvider
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        validate_timeout(self.timeout_seconds)
        provider_timeout = getattr(self.provider, "timeout_seconds", None)
        if provider_timeout is not None and validate_timeout(provider_timeout) > float(
            self.timeout_seconds
        ):
            raise ValueError("provider timeout cannot exceed analyst timeout")

    def run(
        self,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
    ) -> tuple[AnalystReport, ...]:
        snapshot.verify_seal()
        _verify_point_in_time(proposal, snapshot)
        futures: dict[AnalystRole, Future[AnalystReport]] = {}
        try:
            futures = {
                role: _EXECUTOR.submit(
                    self.provider,
                    AnalystRequest(role=role, proposal=proposal, snapshot=snapshot),
                    role,
                )
                for role in ANALYST_ROLES
            }
            _done, pending = wait(
                futures.values(),
                timeout=self.timeout_seconds,
            )
            if pending:
                raise AnalystOutputError("analyst desk timed out")
            return tuple(self._read(role, futures[role]) for role in ANALYST_ROLES)
        finally:
            for future in futures.values():
                future.cancel()

    def _read(
        self,
        role: AnalystRole,
        future: Future[AnalystReport],
    ) -> AnalystReport:
        try:
            return future.result()
        except AnalystOutputError:
            raise
        except Exception:
            raise AnalystOutputError(f"{role} analyst provider failed") from None


def _invoke_and_validate(
    provider: AnalystProvider,
    request: AnalystRequest,
    role: AnalystRole,
) -> AnalystReport:
    try:
        raw = provider.analyze(request)
    except Exception:
        raise AnalystOutputError(f"{role} analyst provider failed") from None
    try:
        if not isinstance(raw, str) or len(raw) > _MAX_OUTPUT_CHARACTERS:
            raise ValueError("analyst response exceeds the output limit")
        decoded = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(decoded, dict):
            raise ValueError("analyst response must be one JSON object")
        report = AnalystReport.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
        raise AnalystOutputError(f"{role} analyst output is invalid") from None
    if report.role != role:
        raise AnalystOutputError(f"{role} analyst returned a mismatched role")
    return report


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    counts = Counter(key for key, _value in pairs)
    duplicate = next((key for key, count in counts.items() if count > 1), None)
    if duplicate is not None:
        raise ValueError(f"duplicate JSON key: {duplicate}")
    return dict(pairs)


def _verify_point_in_time(
    proposal: TradeProposal,
    snapshot: SealedSourceSnapshot,
) -> None:
    try:
        as_of = datetime.fromisoformat(proposal.as_of.replace("Z", "+00:00"))
        if as_of > datetime.now(timezone.utc):
            raise AnalystOutputError("point-in-time proposal cannot be future dated")
        future_sources = tuple(
            document.document_id
            for document in snapshot.documents
            if datetime.fromisoformat(
                document.available_at.replace("Z", "+00:00")
            )
            > as_of
        )
    except AnalystOutputError:
        raise
    except (TypeError, ValueError):
        raise AnalystOutputError("point-in-time timestamps are invalid") from None
    if future_sources:
        raise AnalystOutputError("point-in-time snapshot contains future sources")


__all__ = ["AgentDesk", "AnalystOutputError", "validate_timeout"]
