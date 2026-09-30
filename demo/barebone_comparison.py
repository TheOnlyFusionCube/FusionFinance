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
from demo.pure_ml import (
    MOMENTUM_LOOKBACK_SESSIONS,
    SCOREBOOK_MOMENTUM,
    SCOREBOOK_RIDGE,
)


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
BAREBONE_CORE_UNIVERSE = (
    "AAPL",
    "AMZN",
    "AVGO",
    "BRK-B",
    "COST",
    "GOOGL",
    "HD",
    "JPM",
    "LLY",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "ORCL",
    "TSLA",
    "WMT",
)
BAREBONE_SUPPLEMENTAL_UNIVERSE = (
    "AMD",
    "CRM",
    "ADBE",
    "PEP",
    "KO",
    "XOM",
    "CVX",
    "UNH",
    "V",
    "MA",
    "BAC",
    "WFC",
    "DIS",
    "CMCSA",
    "INTC",
    "QCOM",
    "TXN",
    "AMAT",
    "NOW",
    "ISRG",
    "BKNG",
    "TMO",
    "ABT",
    "MRK",
    "ACN",
    "IBM",
    "GE",
    "CAT",
    "HON",
    "LOW",
    "NKE",
    "UPS",
    "BA",
    "PFE",
)
BAREBONE_UNIVERSE = BAREBONE_CORE_UNIVERSE + BAREBONE_SUPPLEMENTAL_UNIVERSE
NARRATIVE_EVENTS = "evidence/narrative/barebone_window_events.jsonl"
NARRATIVE_PROVENANCE = "evidence/narrative/barebone_window_narrative_provenance.json"
NARRATIVE_MAP = "evidence/narrative/barebone_ticker_map.json"
NARRATIVE_SCORES = "evidence/narrative/barebone_window_scores.jsonl"
NARRATIVE_POLARITY_PROVENANCE = "evidence/narrative/barebone_window_polarity_provenance.json"
NARRATIVE_LEXICON = "evidence/narrative/barebone_polarity_lexicon.json"
LOCKED_NARRATIVE_SHA256 = "860cb1d3a86fd4a3353a876d421618d69e76228c65e44b6dac7e28c820b9d2a9"
LOCKED_TAPE_SHA256 = "c29f4810a8433e0de286da46409dbe95c17c1fa09d25e7809bb5f9e73ad8a205"
LOCKED_POLARITY_SHA256 = "c8b72ba38d4bb110d83c5b548a8a19389b33259f7d3e62612988a3242e8b7cae"
LOCKED_ATTENTION_SHA256 = "0e4d38235f19138f41244368e44b2cba9b8f78acd4ce9b713ac8d17dc0bd8227"
LOCKED_EDGAR_SHA256 = "b40cf901de329d13605336943307d656dbe09785dd04601744d0e24025d0945e"
ATTENTION_SCORES = "evidence/narrative/barebone_window_attention.jsonl"
ATTENTION_PROVENANCE = "evidence/narrative/barebone_window_attention_provenance.json"
HYBRID_PROVENANCE = "evidence/narrative/barebone_hybrid_provenance.json"
EDGAR_EVENTS = "evidence/narrative/barebone_window_edgar.jsonl"
EDGAR_PROVENANCE = "evidence/narrative/barebone_window_edgar_provenance.json"
EDGAR_CIK_MAP = "evidence/narrative/barebone_edgar_cik_map.json"
FORM4_EVENTS = "evidence/narrative/barebone_window_form4.jsonl"
FORM4_PROVENANCE = "evidence/narrative/barebone_window_form4_provenance.json"
LOCKED_FORM4_SHA256 = "4f46fd974486d189300727e474cc856caf0927e783f3d912f39c9487b42c929d"
SHORT_INTEREST_EVENTS = "evidence/market/barebone_window_short_interest.jsonl"
SHORT_INTEREST_PROVENANCE = "evidence/market/barebone_window_short_interest_provenance.json"
SHORT_INTEREST_SYMBOL_MAP = "evidence/market/barebone_short_interest_symbol_map.json"
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
        "scorebook",
        "momentum_lookback_sessions",
        "evidence",
    )
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class BareboneEvidence(_FrozenModel):
    """Paths of the Barebone-window extracts. The price and narrative files stay local."""

    ohlcv: str = Field(min_length=1)
    tape_sha256: str | None = None
    narrative_events: str = NARRATIVE_EVENTS
    narrative_sha256: str | None = None
    narrative_scores: str = NARRATIVE_SCORES
    polarity_sha256: str | None = None
    attention_scores: str = ATTENTION_SCORES
    attention_sha256: str | None = None
    edgar_events: str = EDGAR_EVENTS
    edgar_sha256: str | None = None
    edgar_form4_events: str = FORM4_EVENTS
    edgar_form4_sha256: str | None = None
    short_interest_events: str = SHORT_INTEREST_EVENTS
    short_interest_sha256: str | None = None

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

    @field_validator("narrative_events")
    @classmethod
    def _narrative_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != NARRATIVE_EVENTS:
            raise ValueError("narrative events path is the gitignored local bind")
        return normalized

    @field_validator("narrative_sha256")
    @classmethod
    def _narrative_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("narrative_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone narrative tape")
        return value

    @field_validator("narrative_scores")
    @classmethod
    def _scores_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != NARRATIVE_SCORES:
            raise ValueError("narrative scores path is the gitignored local bind")
        return normalized

    @field_validator("polarity_sha256")
    @classmethod
    def _polarity_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("polarity_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone polarity tape")
        return value

    @field_validator("attention_scores")
    @classmethod
    def _attention_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != ATTENTION_SCORES:
            raise ValueError("attention scores path is the gitignored local bind")
        return normalized

    @field_validator("attention_sha256")
    @classmethod
    def _attention_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("attention_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone attention tape")
        return value

    @field_validator("edgar_events")
    @classmethod
    def _edgar_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != EDGAR_EVENTS:
            raise ValueError("edgar events path is the gitignored local bind")
        return normalized

    @field_validator("edgar_sha256")
    @classmethod
    def _edgar_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("edgar_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone edgar tape")
        return value

    @field_validator("edgar_form4_events")
    @classmethod
    def _form4_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != FORM4_EVENTS:
            raise ValueError("form 4 events path is the gitignored local bind")
        return normalized

    @field_validator("edgar_form4_sha256")
    @classmethod
    def _form4_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("edgar_form4_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone form 4 tape")
        return value

    @field_validator("short_interest_events")
    @classmethod
    def _short_interest_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized != SHORT_INTEREST_EVENTS:
            raise ValueError("short interest events path is the gitignored local bind")
        return normalized

    @field_validator("short_interest_sha256")
    @classmethod
    def _short_interest_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("short_interest_sha256 must be a lowercase sha256 hex digest or null")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone short interest tape")
        return value


class BareboneComparisonConfig(_FrozenModel):
    """Locked contract shell. It is not a performance result."""

    experiment: ExperimentConfig
    secondary_benchmark_ticker: str | None
    scorebook: str
    momentum_lookback_sessions: int | None = None
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
        if experiment.universe != BAREBONE_UNIVERSE:
            raise ValueError(
                "barebone-comparison-v1 universe is the locked 16 names "
                "plus the supplemental liquid list"
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
        if self.scorebook not in {SCOREBOOK_RIDGE, SCOREBOOK_MOMENTUM}:
            raise ValueError("scorebook must be ridge or momentum")
        if self.scorebook == SCOREBOOK_MOMENTUM:
            if self.momentum_lookback_sessions != MOMENTUM_LOOKBACK_SESSIONS:
                raise ValueError("momentum lookback is locked at 63 sessions")
        elif self.momentum_lookback_sessions is not None:
            raise ValueError("ridge scorebook does not take a momentum lookback")
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
    if "scorebook" not in payload:
        raise ValueError("scorebook is required")
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
        scorebook=str(payload.get("scorebook")),
        momentum_lookback_sessions=payload.get("momentum_lookback_sessions"),
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
