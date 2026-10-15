"""Canonical SHA-256 seals for a controlled run.

These hashes detect a changed config or tape. They are not signatures and
they are not hardware attestation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from demo.contracts import ExperimentConfig, MarketSession


def config_sha256(config: ExperimentConfig) -> str:
    """Hash the frozen experiment config, independent of field order."""

    return _sha256(config.model_dump(mode="json"))


def tape_sha256(sessions: Sequence[MarketSession]) -> str:
    """Hash the executed sessions in order, including each bar."""

    payload = [
        {
            "session": session.session.isoformat(),
            "bars": [
                {"ticker": bar.ticker, "open": bar.open, "close": bar.close}
                for bar in session.bars
            ],
        }
        for session in sessions
    ]
    return _sha256(payload)


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
