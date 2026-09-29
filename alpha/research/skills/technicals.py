"""Technical skill: confirmed support/resistance zones, trend, momentum, levels.

1. Reversal points are detected by an explicit rule (a bar is a pivot high if it
   is the highest of ``k`` bars on each side). A pivot is only usable once its
   right-hand ``k`` bars exist, so no level is drawn with future data.
2. Pivots are clustered into zones, and a zone is shown only if it appears
   independently on at least two resolutions (daily and weekly).
3. Zones are widened by an ATR buffer so stops respect normal noise.
4. Trend must clear a strictness bar (price, fast and slow averages aligned and
   the slow average rising or falling) before a directional bias is reported.
5. Momentum (RSI) adapts entries: oversold entries sit nearer the price.
6. A 0-100 confidence summarises agreement; below forty the call is "wait".
   At all-time highs, with no resistance overhead, targets fall back to an
   ATR projection instead of returning nothing.
"""
from __future__ import annotations

import pandas as pd

from alpha.research.cards import SignalCard, clip, empty_card
from alpha.research.market import atr, bars_frame, rsi
from alpha.research.store import StoreView


def pivots(frame: pd.DataFrame, k: int) -> tuple[list[float], list[float]]:
    highs, lows = [], []
    high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
    for index in range(k, len(frame) - k):
        window_high = high[index - k:index + k + 1]
        window_low = low[index - k:index + k + 1]
        if high[index] == window_high.max():
            highs.append(float(high[index]))
        if low[index] == window_low.min():
            lows.append(float(low[index]))
    return highs, lows


def cluster(levels: list[float], tolerance: float) -> list[tuple[float, int]]:
    zones: list[list[float]] = []
    for level in sorted(levels):
        if zones and level - zones[-1][-1] <= tolerance:
            zones[-1].append(level)
        else:
            zones.append([level])
    return [(sum(zone) / len(zone), len(zone)) for zone in zones]


def _weekly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def analyze_technicals(view: StoreView, ticker: str, *, lookback: int = 260) -> SignalCard:
    frame = bars_frame(view, ticker).iloc[-lookback:]
    if len(frame) < 60:
        return empty_card("technicals", ticker, view.as_of, "Not enough price history.", "market")
    price = float(frame["close"].iloc[-1])
    band = atr(frame)
    tolerance = max(band * 0.75, price * 0.005)
    daily_high, daily_low = pivots(frame, 5)
    weekly = _weekly(frame)
    weekly_high, weekly_low = pivots(weekly, 2) if len(weekly) >= 10 else ([], [])
    daily_zones = cluster(daily_high + daily_low, tolerance)
    weekly_zones = cluster(weekly_high + weekly_low, tolerance)
    confirmed = sorted(
        level for level, _count in daily_zones
        if any(abs(level - other) <= tolerance for other, _ in weekly_zones)
    )
    supports = [level for level in confirmed if level < price - 0.25 * band]
    resistances = [level for level in confirmed if level > price + 0.25 * band]

    close = frame["close"]
    fast = close.rolling(50).mean()
    slow = close.rolling(min(200, len(close) - 1)).mean()
    slope = float(slow.iloc[-1] - slow.iloc[-21]) if len(slow.dropna()) > 21 else 0.0
    if price > fast.iloc[-1] > slow.iloc[-1] and slope > 0:
        trend = "uptrend"
    elif price < fast.iloc[-1] < slow.iloc[-1] and slope < 0:
        trend = "downtrend"
    else:
        trend = "range"
    momentum = rsi(close)
    condition = "oversold" if momentum < 30 else "overheated" if momentum > 70 else "neutral"
    year_high = float(frame["high"].max())
    year_low = float(frame["low"].min())
    position = (price - year_low) / (year_high - year_low) if year_high > year_low else 0.5

    facts = {
        "atr": round(band, 4), "rsi": round(momentum, 2),
        "range_position_pct": round(position * 100, 2), "confirmed_zones": float(len(confirmed)),
    }
    lines = [
        f"Trend reads {trend} with momentum {condition} (RSI {momentum:.0f}); "
        f"price sits at {position:.0%} of its one-year range.",
    ]
    flags = []
    direction = 1 if trend == "uptrend" else -1 if trend == "downtrend" else 0
    entry = stop = target = None
    if direction > 0:
        anchor = supports[-1] if supports else price - 2 * band
        entry = price - 0.25 * band if condition == "oversold" else anchor + band * (1.0 if condition == "overheated" else 0.5)
        entry = min(entry, price)
        stop = anchor - band
        target = resistances[0] if resistances else price + 3 * band
        if not resistances:
            flags.append("ALL_TIME_HIGH_PROJECTION")
    elif direction < 0:
        anchor = resistances[0] if resistances else price + 2 * band
        entry = max(price + 0.25 * band if condition == "overheated" else anchor - band * 0.5, price)
        stop = anchor + band
        target = supports[-1] if supports else price - 3 * band
    agreement = sum((
        25 if direction else 0,
        25 if (direction > 0 and supports) or (direction < 0 and resistances) else 0,
        20 if (direction > 0 and condition != "overheated") or (direction < 0 and condition != "oversold") else 0,
        min(30, 10 * len(confirmed)),
    ))
    facts["technical_confidence"] = float(agreement)
    if entry is not None:
        facts.update({"entry": round(entry, 4), "stop": round(stop, 4), "target": round(target, 4)})
        lines.append(f"Plan: entry {entry:.2f}, stop {stop:.2f}, target {target:.2f}.")
    if supports:
        lines.append(f"Nearest confirmed support zone {supports[-1] - band:.2f} to {supports[-1] + band:.2f}.")
    if resistances:
        lines.append(f"Nearest confirmed resistance zone {resistances[0] - band:.2f} to {resistances[0] + band:.2f}.")
    if agreement < 40:
        flags.append("WAIT_FOR_SETUP")
        lines.append("Signals are mixed; wait for a better setup.")
    stretched = (price / float(close.iloc[-6]) - 1.0) / max(band / price, 1e-9) if len(close) > 6 else 0.0
    facts["five_day_move_atr"] = round(stretched, 3)
    if abs(stretched) > 4:
        flags.append("STRETCHED_MOVE")
        lines.append(f"The last five sessions moved {stretched:+.1f} ATRs, an overextended move.")
    score = clip(direction * agreement / 100.0 - 0.3 * clip(stretched / 6.0))
    return SignalCard(
        skill="technicals", ticker=ticker, as_of=view.as_of, score=score,
        confidence=clip(agreement / 100.0, 0.0, 1.0),
        headline=f"{trend.capitalize()}, {condition} momentum, confidence {agreement}/100.",
        lines=tuple(lines), facts=facts, flags=tuple(flags), role="market",
    )


__all__ = ["analyze_technicals", "cluster", "pivots"]
