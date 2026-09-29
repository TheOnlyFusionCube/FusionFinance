"""Synthetic point-in-time market and document world for offline research.

The world is fictional by construction, so no pretrained LLM can have seen its
future. That is the one setting in which an LLM-first backtest is free of
pretraining leakage. It plants two kinds of document events:

* ``genuine``: an after-close disclosure whose sign is only knowable from the
  text. Prices do not move on the day; residual drift follows. A text reader
  can trade it; a price/volume model cannot know its sign in advance.
* ``hype``: a promotional narrative published after a quiet, low-volume price
  run-up. The text reads as bullish (or bearish) as a genuine event, but the
  move is already in the price and reverses. A text reader is fooled; a
  price/volume model can recognise the overextension.

Neither channel sees the whole picture. The world exists to test whether an
ML verifier can catch the LLM's blind spot without discarding its edge. It is
a mechanism test, not evidence of real-market alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.agents.models import NumericValue, SourceDocument

BENCHMARK = "MKT"

_GENUINE_POSITIVE = (
    "Quarterly revenue growth accelerated and margins improved.",
    "Management raised full-year guidance after demand was strong.",
    "Orders improved across regions and cash conversion was strong.",
)
_GENUINE_NEGATIVE = (
    "Quarterly revenue declined and margins deteriorated.",
    "Management cut full-year guidance after demand was weak.",
    "The company missed order expectations and cash conversion declined.",
)
_HYPE_POSITIVE = (
    "Promoters say growth accelerated and the story is strong.",
    "Commentators claim guidance will be raised on strong demand.",
    "Influencer notes say margins improved and momentum is strong.",
)
_HYPE_NEGATIVE = (
    "Short sellers claim demand is weak and revenue declined.",
    "Commentators allege margins deteriorated and guidance will be cut.",
    "Bearish notes say the company missed targets and growth is weak.",
)
_NEUTRAL = {
    "market": "Trading was orderly with no unusual order flow noted.",
    "news": "The company made no new announcements this session.",
    "fundamentals": "No new financial statements were filed this session.",
    "risk": "No new risk disclosures were filed this session.",
}


@dataclass(frozen=True)
class WorldConfig:
    n_names: int = 30
    n_sessions: int = 640
    seed: int = 7
    start: str = "2023-01-02"
    event_rate: float = 0.02
    hype_share: float = 0.4
    text_noise: float = 0.1
    idio_vol: tuple[float, float] = (0.008, 0.014)
    genuine_drift_days: int = 12
    genuine_drift_bps: float = 300.0
    hype_runup_days: int = 5
    hype_runup_bps: float = 600.0
    hype_reversal_days: int = 8
    hype_reversal_bps: float = 450.0


@dataclass(frozen=True)
class DocumentEvent:
    ticker: str
    session_index: int
    kind: str          # "genuine" | "hype"
    sign: int          # +1 | -1 true sign of the planted narrative
    text_sign: int     # sign the text conveys (noise can flip genuine text)
    strength: int


@dataclass
class SyntheticWorld:
    config: WorldConfig
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    opens: pd.DataFrame
    closes: pd.DataFrame
    volume: pd.DataFrame
    events: tuple[DocumentEvent, ...]
    _by_ticker: dict[str, list[DocumentEvent]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for event in self.events:
            self._by_ticker.setdefault(event.ticker, []).append(event)

    def as_of(self, index: int) -> str:
        return f"{self.dates[index].date().isoformat()}T21:00:00Z"

    def snapshot_documents(
        self, ticker: str, index: int, *, lookback: int = 5
    ) -> tuple[SourceDocument, ...]:
        """Documents available by the close of session ``index`` (never later)."""
        recent = [
            event for event in self._by_ticker.get(ticker, ())
            if index - lookback < event.session_index <= index
        ]
        day = self.dates[index].date().isoformat()
        if not recent:
            return tuple(
                SourceDocument(
                    document_id=f"{ticker}.{role}.{day}",
                    available_at=f"{day}T20:30:00Z",
                    text=text,
                    roles=(role,),
                )
                for role, text in _NEUTRAL.items()
            )
        event = recent[-1]
        published = self.dates[event.session_index].date().isoformat()
        rng = np.random.default_rng(
            (self.config.seed, event.session_index, self.tickers.index(ticker))
        )
        if event.kind == "genuine":
            pool = _GENUINE_POSITIVE if event.text_sign > 0 else _GENUINE_NEGATIVE
        else:
            pool = _HYPE_POSITIVE if event.text_sign > 0 else _HYPE_NEGATIVE
        headline, support = rng.choice(len(pool), size=2, replace=False)
        growth = float(event.text_sign * (4 + 3 * event.strength))
        stamp = f"{published}T20:30:00Z"
        return (
            SourceDocument(
                document_id=f"{ticker}.news.{published}",
                available_at=stamp,
                text=pool[int(headline)],
                roles=("news",),
            ),
            SourceDocument(
                document_id=f"{ticker}.fundamentals.{published}",
                available_at=stamp,
                text=pool[int(support)],
                roles=("fundamentals",),
                numeric_values=(NumericValue(key="growth_pct", value=growth),),
            ),
            SourceDocument(
                document_id=f"{ticker}.market.{published}",
                available_at=stamp,
                text=(
                    "Commentary on the tape was strong and buyers improved."
                    if event.text_sign > 0
                    else "Commentary on the tape was weak and bids declined."
                ),
                roles=("market",),
            ),
            SourceDocument(
                document_id=f"{ticker}.risk.{published}",
                available_at=stamp,
                text=_NEUTRAL["risk"],
                roles=("risk",),
            ),
        )


def build_world(config: WorldConfig = WorldConfig()) -> SyntheticWorld:
    rng = np.random.default_rng(config.seed)
    n, t = config.n_names, config.n_sessions
    tickers = tuple(f"S{index:02d}" for index in range(n))
    dates = pd.bdate_range(config.start, periods=t)
    market = rng.normal(0.0003, 0.009, size=t)
    beta = rng.uniform(0.8, 1.2, size=n)
    idio_vol = rng.uniform(*config.idio_vol, size=n)
    residual = rng.normal(0.0, 1.0, size=(t, n)) * idio_vol
    volume_multiplier = np.exp(rng.normal(0.0, 0.2, size=(t, n)))

    busy_until = np.full(n, -1)
    events: list[DocumentEvent] = []
    first = 260  # leave a full year of quiet history for 252-session features
    for day in range(first, t - 1):
        for name in range(n):
            if day <= busy_until[name] or rng.random() >= config.event_rate:
                continue
            sign = 1 if rng.random() < 0.5 else -1
            strength = int(rng.integers(1, 3))
            if rng.random() < config.hype_share:
                start = day - config.hype_runup_days + 1
                residual[start:day + 1, name] += (
                    sign * config.hype_runup_bps * 1e-4 / config.hype_runup_days
                )
                volume_multiplier[start:day + 1, name] *= 0.8
                end = min(t, day + 1 + config.hype_reversal_days)
                residual[day + 1:end, name] -= (
                    sign * config.hype_reversal_bps * 1e-4 / config.hype_reversal_days
                )
                events.append(DocumentEvent(tickers[name], day, "hype", sign, sign, strength))
                busy_until[name] = end
            else:
                end = min(t, day + 1 + config.genuine_drift_days)
                residual[day + 1:end, name] += (
                    sign * strength * config.genuine_drift_bps * 1e-4
                    / config.genuine_drift_days
                )
                volume_multiplier[day + 1:day + 3, name] *= 2.5
                text_sign = -sign if rng.random() < config.text_noise else sign
                events.append(
                    DocumentEvent(tickers[name], day, "genuine", sign, text_sign, strength)
                )
                busy_until[name] = end

    returns = market[:, None] * beta[None, :] + residual
    closes = 50.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))
    gaps = rng.normal(0.0, 0.002, size=(t, n))
    opens = np.vstack([closes[:1], closes[:-1]]) * (1.0 + gaps)
    volume = 1_000_000.0 * volume_multiplier
    bench = 100.0 * np.exp(np.cumsum(np.log1p(returns.mean(axis=1))))
    bench_open = np.concatenate([[bench[0]], bench[:-1]])

    close_frame = pd.DataFrame(closes, index=dates, columns=tickers)
    close_frame[BENCHMARK] = bench
    open_frame = pd.DataFrame(opens, index=dates, columns=tickers)
    open_frame[BENCHMARK] = bench_open
    return SyntheticWorld(
        config=config,
        dates=dates,
        tickers=tickers,
        opens=open_frame,
        closes=close_frame,
        volume=pd.DataFrame(volume, index=dates, columns=tickers),
        events=tuple(events),
    )


__all__ = ["BENCHMARK", "DocumentEvent", "SyntheticWorld", "WorldConfig", "build_world"]
