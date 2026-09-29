"""Pretraining-leakage guard for LLM decisions.

A backtest that asks a pretrained LLM about dates inside its training window is
not a test: the model may simply remember what happened. This guard marks every
decision as ``clean`` or ``contaminated`` and fails closed: a model with an
unknown knowledge cutoff is contaminated for every historical real-market date.
Contaminated decisions can still be run for debugging, but they are excluded
from reliability learning, meta-label training, and claim-bearing metrics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Mapping

# Models that were never pretrained on market history.
_NO_PRETRAINING_PREFIXES = ("fusionfinance-offline-",)


def _parse_day(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


@dataclass(frozen=True)
class KnowledgeCutoffGuard:
    """Map provider model IDs to the last date their training data could cover."""

    cutoffs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = {model: _parse_day(day).isoformat() for model, day in dict(self.cutoffs).items()}
        object.__setattr__(self, "cutoffs", MappingProxyType(parsed))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "KnowledgeCutoffGuard":
        """Read ``FUSION_LLM_KNOWLEDGE_CUTOFFS`` as ``model=YYYY-MM-DD,...``."""
        values = os.environ if environ is None else environ
        raw = values.get("FUSION_LLM_KNOWLEDGE_CUTOFFS", "")
        cutoffs: dict[str, str] = {}
        for item in filter(None, (part.strip() for part in raw.split(","))):
            model, separator, day = item.partition("=")
            if not separator or not model.strip() or not day.strip():
                raise ValueError("FUSION_LLM_KNOWLEDGE_CUTOFFS entries must be model=YYYY-MM-DD")
            cutoffs[model.strip()] = day.strip()
        return cls(cutoffs=cutoffs)

    def is_contaminated(self, model_id: str, as_of: str, *, synthetic_world: bool) -> bool:
        if synthetic_world or model_id.startswith(_NO_PRETRAINING_PREFIXES):
            return False
        cutoff = self.cutoffs.get(model_id)
        if cutoff is None:
            return True
        return _parse_day(as_of) <= date.fromisoformat(cutoff)


__all__ = ["KnowledgeCutoffGuard"]
