"""Fail-closed shell for a future Barebone-window experiment.

This module loads ``configs/barebone-comparison-v1.json`` and refuses to
treat the fair-race fixture as that experiment. It does not build a tape,
score a book, or emit a ledger. ``comparable_performance_claim`` stays false.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from demo.contracts import ExperimentConfig


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


BAREBONE_EXPERIMENT_ID = "barebone-comparison-v1"
FAIR_RACE_EXPERIMENT_ID = "fusionfinance-fair-race-v1"
FAIR_RACE_TAPE_HASH = (
    "3a2a378ce29b387b84f5345a7953028d478507f204b9f7be6eee609e0dd20c05"
)
FAIR_RACE_OHLCV = "evidence/market/locked_ohlcv.json"
BAREBONE_OHLCV = "evidence/market/barebone_window_ohlcv.json"
BAREBONE_PROVENANCE = "evidence/market/barebone_window_provenance.json"
BAREBONE_LICENSE_NOTE = "not redistributed; local bind only"
BAREBONE_WINDOW = ("2025-01-02", "2026-01-12")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPERIMENT_KEYS = (
    "schema_version",
    "experiment_id",
    "start_date",
    "end_date",
    "starting_capital",
    "universe",
    "benchmark_ticker",
    "rebalance_frequency_sessions",
    "execution_lag_sessions",
    "transaction_cost_bps",
    "slippage_bps",
    "max_gross_leverage",
    "max_position_weight",
    "annualization_sessions",
    "annual_risk_free_rate",
)
_TOP_LEVEL_KEYS = frozenset(
    _EXPERIMENT_KEYS
    + (
        "comparable_performance_claim",
        "secondary_benchmark_ticker",
        "evidence",
    )
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class BareboneEvidence(_FrozenModel):
    """Path of the Barebone-window extract. The file is not in this scaffold."""

    ohlcv: str = Field(min_length=1)
    tape_sha256: str | None = None

    @field_validator("ohlcv")
    @classmethod
    def _relative_ohlcv(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in parts
            or normalized != "/".join(parts)
        ):
            raise ValueError("evidence path must stay inside the repository")
        if normalized == FAIR_RACE_OHLCV or normalized.endswith("/locked_ohlcv.json"):
            raise ValueError("fair-race OHLCV is not the barebone-comparison tape")
        return normalized

    @field_validator("tape_sha256")
    @classmethod
    def _locked_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("tape_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone-comparison tape")
        return value


class BareboneComparisonConfig(_FrozenModel):
    """Locked contract shell. It is not a performance result."""

    experiment: ExperimentConfig
    secondary_benchmark_ticker: str | None
    comparable_performance_claim: bool = Field(default=False)
    evidence: BareboneEvidence

    @field_validator("comparable_performance_claim")
    @classmethod
    def _claim_is_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("comparable_performance_claim must be false")
        return False

    @field_validator("secondary_benchmark_ticker")
    @classmethod
    def _secondary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("secondary_benchmark_ticker must not be blank")
        return normalized

    @model_validator(mode="after")
    def _locked_shell(self) -> Self:
        experiment = self.experiment
        if experiment.experiment_id != BAREBONE_EXPERIMENT_ID:
            raise ValueError("experiment_id must be barebone-comparison-v1")
        window = (
            experiment.start_date.isoformat(),
            experiment.end_date.isoformat(),
        )
        if window != BAREBONE_WINDOW:
            raise ValueError(
                "barebone-comparison-v1 window is 2025-01-02 through 2026-01-12"
            )
        if experiment.benchmark_ticker != "SPY":
            raise ValueError("barebone-comparison-v1 benchmark is SPY")
        if experiment.max_position_weight != 0.1 or experiment.max_gross_leverage != 1.0:
            raise ValueError(
                "barebone-comparison-v1 keeps max_position_weight 0.10 "
                "and max_gross_leverage 1.0"
            )
        if (
            experiment.transaction_cost_bps != 5.0
            or experiment.slippage_bps != 2.0
            or experiment.rebalance_frequency_sessions != 10
            or experiment.execution_lag_sessions != 1
            or experiment.starting_capital != 10_000.0
            or experiment.annualization_sessions != 252
            or experiment.annual_risk_free_rate != 0.0
            or experiment.schema_version != "1.0"
        ):
            raise ValueError("barebone-comparison-v1 keeps the controlled-path locks")
        secondary = self.secondary_benchmark_ticker
        if secondary is not None and (
            secondary == experiment.benchmark_ticker or secondary in experiment.universe
        ):
            raise ValueError("secondary benchmark must stay outside the tradable book")
        return self

    def as_experiment_config(self) -> ExperimentConfig:
        """Shared kernel config. Evidence and the secondary benchmark stay outside it."""

        return self.experiment


def load_barebone_comparison_config(
    path: str | Path | None = None,
) -> BareboneComparisonConfig:
    """Load the scaffold. A missing tape is allowed here and rejected later."""

    config_path = (
        Path(path)
        if path is not None
        else _repo_root() / "configs" / "barebone-comparison-v1.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    return validate_barebone_payload(payload)


def validate_barebone_payload(payload: Mapping[str, object]) -> BareboneComparisonConfig:
    """Validate one Barebone comparison document without reading a tape."""

    if not isinstance(payload, Mapping):
        raise ValueError("barebone comparison config must be a JSON object")
    unknown = set(payload).difference(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            "unknown barebone comparison fields: " + ", ".join(sorted(unknown))
        )
    if payload.get("experiment_id") != BAREBONE_EXPERIMENT_ID:
        raise ValueError("experiment_id must be barebone-comparison-v1")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    if "secondary_benchmark_ticker" not in payload:
        raise ValueError("secondary_benchmark_ticker is required")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence path is required")
    if payload.get("max_position_weight") != 0.1 or payload.get("max_gross_leverage") != 1.0:
        raise ValueError(
            "barebone-comparison-v1 keeps max_position_weight 0.10 "
            "and max_gross_leverage 1.0"
        )
    start = _iso_date(payload.get("start_date"))
    end = _iso_date(payload.get("end_date"))
    if (start, end) != BAREBONE_WINDOW:
        raise ValueError(
            "barebone-comparison-v1 window is 2025-01-02 through 2026-01-12"
        )
    experiment = ExperimentConfig.model_validate(
        {key: payload[key] for key in _EXPERIMENT_KEYS}
    )
    return BareboneComparisonConfig(
        experiment=experiment,
        secondary_benchmark_ticker=payload.get("secondary_benchmark_ticker"),
        comparable_performance_claim=False,
        evidence=BareboneEvidence.model_validate(dict(evidence)),
    )


def require_barebone_evidence(
    config: BareboneComparisonConfig, *, root: Path | None = None
) -> Path:
    """Return the evidence file only when it exists and matches a locked hash.

    A missing file, a null hash, a mismatched hash, or the fair-race extract
    raises. This function does not parse prices.
    """

    base = _repo_root() if root is None else root
    relative = config.evidence.ohlcv
    path = base / relative
    if not path.is_file():
        raise FileNotFoundError(f"barebone-comparison evidence is missing: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    locked = config.evidence.tape_sha256
    if locked is None:
        raise ValueError("barebone-comparison-v1 refuses an unlocked evidence tape")
    if digest != locked:
        raise ValueError(
            "barebone-comparison evidence hash does not match the locked tape_sha256"
        )
    if digest == _fair_race_ohlcv_sha256() or locked == FAIR_RACE_TAPE_HASH:
        raise ValueError("fair-race OHLCV is not the barebone-comparison tape")
    return path


def refuse_barebone_performance_claim(
    config: BareboneComparisonConfig, claim: bool
) -> None:
    """Keep this experiment incomparable until a later locked tape exists.

    A true claim is refused while ``tape_sha256`` is null. It is still refused
    when a hash is present, because this scaffold does not authorize a claim.
    """

    if claim is False:
        return
    if config.evidence.tape_sha256 is None:
        raise ValueError(
            "barebone-comparison-v1 refuses comparable_performance_claim "
            "without a locked tape_sha256"
        )
    raise ValueError("barebone-comparison-v1 keeps comparable_performance_claim false")


def refuse_fair_race_as_barebone_comparison(document: Mapping[str, object]) -> None:
    """Refuse the fair-race fixture, its tape hash, and its OHLCV extract."""

    if not isinstance(document, Mapping):
        raise ValueError("comparison document must be a mapping")
    if any(flag is True for flag in _claim_flags(document)):
        raise ValueError("comparable_performance_claim must be false")
    text = "\n".join(_strings(document))
    if (
        FAIR_RACE_EXPERIMENT_ID in text
        or FAIR_RACE_TAPE_HASH in text
        or "locked_ohlcv.json" in text
    ):
        raise ValueError(
            "fusionfinance-fair-race-v1 results are not comparable "
            "for barebone-comparison-v1"
        )


def _iso_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _fair_race_ohlcv_sha256() -> str | None:
    path = _repo_root() / FAIR_RACE_OHLCV
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _claim_flags(value: object):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "comparable_performance_claim":
                yield item
            yield from _claim_flags(item)
    elif isinstance(value, list):
        for item in value:
            yield from _claim_flags(item)
