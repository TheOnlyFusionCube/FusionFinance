"""Typed, point-in-time research records.

Every record carries two clocks:

``event_at``      when the underlying thing happened (a trade, a quarter end);
``available_at``  when the fund could first have known it (a filing's EDGAR
                  acceptance, a Congressional disclosure, an article's publish
                  time, a 13F filing date).

Analysis only ever filters on ``available_at``. That is the difference between
a research terminal and a backtestable research pipeline: an insider trade
dated Monday but filed Wednesday does not exist on Tuesday, and a Congressional
trade disclosed forty days later does not exist for forty days.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from functools import cached_property
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: str) -> str:
    return parse_timestamp(value).isoformat().replace("+00:00", "Z")


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: str
    ticker: str
    available_at: str
    event_at: str
    source: str = Field(min_length=1, max_length=200)

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if _TICKER.fullmatch(value) is None:
            raise ValueError("ticker must be an uppercase market symbol")
        return value

    @field_validator("available_at", "event_at")
    @classmethod
    def _aware(cls, value: str) -> str:
        return _iso(value)

    @model_validator(mode="after")
    def _causal(self):
        if parse_timestamp(self.available_at) < parse_timestamp(self.event_at):
            raise ValueError("a record cannot be available before its event")
        return self

    @cached_property
    def record_id(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


class PriceBar(_Record):
    kind: Literal["price_bar"] = "price_bar"
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _range(self) -> "PriceBar":
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("bar must satisfy low <= open/close <= high")
        return self


class NewsArticle(_Record):
    kind: Literal["news"] = "news"
    headline: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=20_000)
    region: str = "US"
    mentioned_tickers: tuple[str, ...] = ()


class AnalystAction(_Record):
    kind: Literal["analyst_action"] = "analyst_action"
    firm: str = Field(min_length=1)
    rating: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    price_target: float | None = Field(default=None, gt=0)
    prior_rating: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] | None = None
    prior_price_target: float | None = Field(default=None, gt=0)


class SocialMentions(_Record):
    """Aggregated mentions for one platform over one window ending at ``event_at``."""

    kind: Literal["social"] = "social"
    platform: Literal["reddit", "x", "stocktwits", "other"]
    mentions: int = Field(ge=0)
    unique_accounts: int = Field(ge=0)
    bullish: int = Field(ge=0)
    bearish: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts(self) -> "SocialMentions":
        if self.unique_accounts > self.mentions or self.bullish + self.bearish > self.mentions:
            raise ValueError("social counts are inconsistent")
        return self


InsiderCode = Literal["P", "S", "A", "M", "F", "G", "D", "C", "X", "J", "W", "OTHER"]


class InsiderTransaction(_Record):
    """One SEC Form 4 non-derivative line. ``available_at`` is EDGAR acceptance."""

    kind: Literal["insider"] = "insider"
    insider: str = Field(min_length=1)
    role: str = ""
    code: InsiderCode
    acquired: bool
    shares: float = Field(ge=0)
    price: float | None = Field(default=None, ge=0)
    shares_after: float | None = Field(default=None, ge=0)
    rule_10b5_1: bool = False
    accession: str = ""

    @property
    def value(self) -> float:
        return self.shares * (self.price or 0.0)

    @property
    def discretionary_purchase(self) -> bool:
        """Open-market buy with own cash: code P, acquired, not a 10b5-1 plan."""
        return self.code == "P" and self.acquired and not self.rule_10b5_1

    @property
    def discretionary_sale(self) -> bool:
        return self.code == "S" and not self.acquired and not self.rule_10b5_1


class CongressTrade(_Record):
    """STOCK Act disclosure. ``available_at`` is the disclosure, not the trade."""

    kind: Literal["congress"] = "congress"
    member: str = Field(min_length=1)
    chamber: Literal["house", "senate"]
    party: str = ""
    committees: tuple[str, ...] = ()
    direction: Literal["purchase", "sale"]
    amount_low: float = Field(ge=0)
    amount_high: float = Field(ge=0)
    committee_oversees_issuer: bool = False

    @property
    def disclosure_lag_days(self) -> float:
        return (parse_timestamp(self.available_at) - parse_timestamp(self.event_at)).total_seconds() / 86_400


class InstitutionalPosition(_Record):
    """13F line. ``event_at`` is quarter end, ``available_at`` the filing."""

    kind: Literal["institutional"] = "institutional"
    filer: str = Field(min_length=1)
    shares: float = Field(ge=0)
    value: float = Field(ge=0)
    prior_shares: float | None = Field(default=None, ge=0)

    @property
    def is_new_position(self) -> bool:
        return (self.prior_shares is None or self.prior_shares == 0) and self.shares > 0

    @property
    def is_exit(self) -> bool:
        return bool(self.prior_shares) and self.shares == 0


class EarningsReport(_Record):
    kind: Literal["earnings"] = "earnings"
    fiscal_period: str
    timing: Literal["pre_market", "post_market", "intraday", "unknown"] = "unknown"
    eps_actual: float
    eps_estimate: float | None = None
    revenue_actual: float = Field(ge=0)
    revenue_estimate: float | None = Field(default=None, ge=0)
    guidance_mid: float | None = None
    prior_guidance_mid: float | None = None
    gross_margin: float | None = None
    prior_gross_margin: float | None = None


class EarningsTranscript(_Record):
    kind: Literal["transcript"] = "transcript"
    fiscal_period: str
    text: str = Field(min_length=1, max_length=400_000)


class Fundamentals(_Record):
    """Annualised statement values from a filing. ``available_at`` is the filing."""

    kind: Literal["fundamentals"] = "fundamentals"
    period_end: date
    revenue: float | None = None
    net_income: float | None = None
    operating_cash_flow: float | None = None
    capex: float | None = None
    shares_outstanding: float | None = Field(default=None, gt=0)
    total_debt: float | None = Field(default=None, ge=0)
    cash: float | None = Field(default=None, ge=0)
    equity: float | None = None
    ebitda: float | None = None
    gross_profit: float | None = None
    total_assets: float | None = Field(default=None, gt=0)
    current_assets: float | None = Field(default=None, ge=0)
    current_liabilities: float | None = Field(default=None, ge=0)
    dividends_per_share: float | None = Field(default=None, ge=0)
    eps: float | None = None
    revenue_growth_3y: float | None = None
    sector: str = ""

    @property
    def free_cash_flow(self) -> float | None:
        if self.operating_cash_flow is None or self.capex is None:
            return None
        return self.operating_cash_flow - abs(self.capex)


class Relationship(_Record):
    """A directed business link used for second-order news impact."""

    kind: Literal["relationship"] = "relationship"
    counterparty: str
    relation: Literal["supplier", "customer", "competitor", "partner"]

    @field_validator("counterparty")
    @classmethod
    def _counterparty(cls, value: str) -> str:
        value = value.strip().upper()
        if _TICKER.fullmatch(value) is None:
            raise ValueError("counterparty must be a ticker")
        return value


ResearchRecord = Annotated[
    Union[
        PriceBar, NewsArticle, AnalystAction, SocialMentions, InsiderTransaction,
        CongressTrade, InstitutionalPosition, EarningsReport, EarningsTranscript,
        Fundamentals, Relationship,
    ],
    Field(discriminator="kind"),
]

RECORD_KINDS = (
    "price_bar", "news", "analyst_action", "social", "insider", "congress",
    "institutional", "earnings", "transcript", "fundamentals", "relationship",
)

__all__ = [
    "AnalystAction", "CongressTrade", "EarningsReport", "EarningsTranscript",
    "Fundamentals", "InsiderTransaction", "InstitutionalPosition", "NewsArticle",
    "PriceBar", "RECORD_KINDS", "Relationship", "ResearchRecord", "SocialMentions",
    "parse_timestamp",
]
