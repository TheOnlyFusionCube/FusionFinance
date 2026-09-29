"""Vendor-agnostic file-drop collector.

News, analyst ratings, social mentions, Congressional disclosures, 13F
holdings, earnings, and transcripts usually come from licensed vendors. Rather
than hard-coding one vendor, the pipeline accepts JSON Lines files whose rows
validate as research records (one ``kind`` per row). Anything that fails
validation is reported, never silently coerced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from alpha.research.records import ResearchRecord

_ADAPTER = TypeAdapter(ResearchRecord)


@dataclass
class JsonlVendorCollector:
    directory: Path
    rejected: list[tuple[str, int, str]] = field(default_factory=list)

    def collect(self) -> list:
        records = []
        for path in sorted(Path(self.directory).glob("*.jsonl")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(_ADAPTER.validate_python(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    self.rejected.append((path.name, number, str(exc).splitlines()[0]))
        return records


__all__ = ["JsonlVendorCollector"]
