"""Pure-Python technical indicators — no pandas/numpy dependency."""

from __future__ import annotations
import math


def ema(closes: list[float], period: int) -> list[float]:
    """Exponential moving average.

    Seeds the EMA from the first value (not an SMA seed) so that the EMA
    converges naturally and early-period lag is preserved — this ensures MACD
    line properly leads the signal line for trending series.

    Returns a list of length ``max(0, len(closes) - period + 1)``.
    Returns ``[]`` if ``len(closes) < period``.
    """
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    # Seed from first value; walk the full series; then trim the warm-up prefix
    # so the output length matches the SMA-seeded convention (starts at index period-1).
    result: list[float] = [closes[0]]
    for price in closes[1:]:
        result.append(price * k + result[-1] * (1 - k))
    return result[period - 1 :]


def rsi(closes: list[float], period: int = 14) -> float:
    """Wilder's RSI. Returns value 0-100. Returns 50.0 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    # Wilder smoothing over remaining candles
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[float, float]:
    """MACD line and signal line. Returns (0.0, 0.0) if insufficient data."""
    if len(closes) < slow + signal_period:
        return 0.0, 0.0
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    # align: ema_slow is shorter by (slow - fast) elements
    offset = slow - fast
    macd_line = [f - s for f, s in zip(ema_fast[offset:], ema_slow)]
    if len(macd_line) < signal_period:
        return 0.0, 0.0
    signal_line = ema(macd_line, signal_period)
    if not signal_line:
        return 0.0, 0.0
    return macd_line[-1], signal_line[-1]


def bollinger(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Bollinger Bands (upper, mid, lower). Returns (close, close, close) if insufficient."""
    if len(closes) < period:
        last = closes[-1] if closes else 0.0
        return last, last, last
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return mid + num_std * std, mid, mid - num_std * std


def vwap(klines: list[dict]) -> float:
    """Session VWAP from klines with open/high/low/close/volume keys."""
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for k in klines:
        typical = (k["high"] + k["low"] + k["close"]) / 3
        vol = k.get("volume", 0) or 0
        cum_tp_vol += typical * vol
        cum_vol += vol
    if cum_vol == 0:
        return klines[-1]["close"] if klines else 0.0
    return cum_tp_vol / cum_vol


def atr(klines: list[dict], period: int = 14) -> float:
    """Average True Range. Returns 0.0 if insufficient data."""
    if len(klines) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(klines)):
        high = klines[i]["high"]
        low = klines[i]["low"]
        prev_close = klines[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    # Wilder smoothing
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def support_resistance(
    klines: list[dict], lookback: int = 30, proximity_pct: float = 0.005
) -> tuple[list[float], list[float]]:
    """Fractal-based support and resistance levels.

    Returns (supports, resistances) sorted descending and ascending respectively,
    closest first. Uses 5-bar fractal pivots over the lookback window.
    """
    window = klines[-lookback:] if len(klines) >= lookback else klines
    if len(window) < 5:
        return [], []

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(2, len(window) - 2):
        h = window[i]["high"]
        l = window[i]["low"]
        # fractal high: higher than 2 bars on each side
        if (
            h > window[i - 1]["high"]
            and h > window[i - 2]["high"]
            and h > window[i + 1]["high"]
            and h > window[i + 2]["high"]
        ):
            pivot_highs.append(h)
        # fractal low
        if (
            l < window[i - 1]["low"]
            and l < window[i - 2]["low"]
            and l < window[i + 1]["low"]
            and l < window[i + 2]["low"]
        ):
            pivot_lows.append(l)

    current = window[-1]["close"]

    # Cluster nearby levels
    def _cluster(levels: list[float]) -> list[float]:
        if not levels:
            return []
        levels = sorted(set(levels))
        merged: list[float] = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - merged[-1]) / (merged[-1] or 1) < proximity_pct:
                merged[-1] = (merged[-1] + lvl) / 2  # average into cluster
            else:
                merged.append(lvl)
        return merged

    all_highs = _cluster(pivot_highs)
    all_lows = _cluster(pivot_lows)

    resistances = sorted([r for r in all_highs if r > current])
    supports = sorted([s for s in all_lows if s < current], reverse=True)

    return supports, resistances
