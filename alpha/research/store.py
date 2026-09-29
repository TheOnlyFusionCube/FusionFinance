"""Append-only point-in-time research store.

The store deduplicates records by content hash, persists them as one JSONL file
per record kind, and answers every query *as of* a timestamp: records whose
``available_at`` is later than the query time are invisible. A view's lineage
hash identifies exactly which records an analysis saw.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import TypeAdapter

from alpha.research.records import RECORD_KINDS, ResearchRecord, parse_timestamp

_ADAPTER = TypeAdapter(ResearchRecord)
T = TypeVar("T")


@dataclass
class ResearchStore:
    _records: dict[str, object] = field(default_factory=dict)
    _by_key: dict[tuple[str, str], list] = field(default_factory=lambda: defaultdict(list))
    _sorted: set[tuple[str, str]] = field(default_factory=set)
    _times: dict[tuple[str, str], list[datetime]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, records: Iterable[object]) -> int:
        added = 0
        for record in records:
            record = _ADAPTER.validate_python(
                record.model_dump() if hasattr(record, "model_dump") else record
            )
            identifier = record.record_id
            if identifier in self._records:
                continue
            self._records[identifier] = record
            tickers = {record.ticker, *getattr(record, "mentioned_tickers", ())}
            for ticker in tickers:
                key = (record.kind, ticker)
                self._by_key[key].append(record)
                self._sorted.discard(key)
            added += 1
        return added

    def tickers(self, kind: str | None = None) -> tuple[str, ...]:
        return tuple(sorted({t for (k, t) in self._by_key if kind is None or k == kind}))

    def view(self, as_of: str | datetime) -> "StoreView":
        cutoff = parse_timestamp(as_of) if isinstance(as_of, str) else as_of
        return StoreView(store=self, cutoff=cutoff)

    def _series(self, kind: str, ticker: str) -> tuple[list, list[datetime]]:
        key = (kind, ticker)
        if key not in self._by_key:
            return [], []
        if key not in self._sorted:
            self._by_key[key].sort(key=lambda r: (parse_timestamp(r.available_at), r.record_id))
            self._times[key] = [parse_timestamp(r.available_at) for r in self._by_key[key]]
            self._sorted.add(key)
        return self._by_key[key], self._times[key]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[str]] = defaultdict(list)
        for identifier in sorted(self._records):
            record = self._records[identifier]
            grouped[record.kind].append(record.model_dump_json())
        for kind in RECORD_KINDS:
            path = directory / f"{kind}.jsonl"
            if grouped.get(kind):
                path.write_text("\n".join(grouped[kind]) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "ResearchStore":
        store = cls()
        for kind in RECORD_KINDS:
            path = Path(directory) / f"{kind}.jsonl"
            if path.exists():
                store.add(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        return store


@dataclass(frozen=True)
class StoreView:
    store: ResearchStore
    cutoff: datetime
    cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def as_of(self) -> str:
        return self.cutoff.isoformat().replace("+00:00", "Z")

    def records(self, kind: str, ticker: str, *, since: datetime | None = None) -> list:
        records, times = self.store._series(kind, ticker)
        end = bisect_right(times, self.cutoff)
        start = 0 if since is None else bisect_left(times, since, 0, end)
        return records[start:end]

    def tickers(self, kind: str) -> tuple[str, ...]:
        return tuple(
            ticker for ticker in self.store.tickers(kind) if self.records(kind, ticker)
        )

    def lineage(self, ticker: str, kinds: Iterable[str] = RECORD_KINDS) -> str:
        identifiers = sorted(
            record.record_id for kind in kinds for record in self.records(kind, ticker)
        )
        return hashlib.sha256(
            json.dumps({"as_of": self.as_of, "ticker": ticker, "records": identifiers}).encode()
        ).hexdigest()


__all__ = ["ResearchStore", "StoreView"]
