"""Append-only, hash-chained decision ledger.

Every fund decision is written as one canonical JSON record whose hash covers
the previous record's hash. Editing, dropping, or reordering any entry breaks
the chain. The chain proves internal consistency, not signer identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

GENESIS = "0" * 64


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _link(previous: str, payload: dict) -> str:
    return hashlib.sha256(previous.encode() + _canonical(payload)).hexdigest()


@dataclass
class DecisionLedger:
    entries: list[dict] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.entries[-1]["record_hash"] if self.entries else GENESIS

    def append(self, payload: dict) -> str:
        if "record_hash" in payload or "previous_hash" in payload:
            raise ValueError("payload may not carry chain fields")
        previous = self.head
        record_hash = _link(previous, payload)
        self.entries.append({**payload, "previous_hash": previous, "record_hash": record_hash})
        return record_hash

    def verify(self) -> None:
        previous = GENESIS
        for position, entry in enumerate(self.entries):
            payload = {
                key: value for key, value in entry.items()
                if key not in {"previous_hash", "record_hash"}
            }
            if entry.get("previous_hash") != previous:
                raise ValueError(f"ledger entry {position} does not follow its predecessor")
            if entry.get("record_hash") != _link(previous, payload):
                raise ValueError(f"ledger entry {position} content does not match its hash")
            previous = entry["record_hash"]

    def write_jsonl(self, path: Path) -> None:
        self.verify()
        path.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in self.entries),
            encoding="utf-8",
        )

    @classmethod
    def read_jsonl(cls, path: Path) -> "DecisionLedger":
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ledger = cls(entries=entries)
        ledger.verify()
        return ledger


__all__ = ["DecisionLedger", "GENESIS"]
