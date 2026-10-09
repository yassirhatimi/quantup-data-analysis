"""Price levels for intraday trade planning: ATR, floor-trader pivots, swings."""

from __future__ import annotations

import pandas as pd


def atr(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average True Range from a daily OHLC frame (High/Low/Close).

    ATR is the standard yardstick for stops and targets: a stop tighter than
    ~1 ATR is inside the market's normal daily noise.
    """
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Classic floor-trader pivots from the previous completed day.

    Widely watched intraday reference levels: P (pivot), R1/R2 resistance,
    S1/S2 support.
    """
    p = (prev_high + prev_low + prev_close) / 3
    return {
        "R2": p + (prev_high - prev_low),
        "R1": 2 * p - prev_low,
        "P": p,
        "S1": 2 * p - prev_high,
        "S2": p - (prev_high - prev_low),
    }


def swing_levels(ohlc: pd.DataFrame, window: int = 20) -> dict[str, float]:
    """Recent swing extremes: highest high / lowest low of the last `window` days."""
    return {
        f"high_{window}d": float(ohlc["High"].iloc[-window:].max()),
        f"low_{window}d": float(ohlc["Low"].iloc[-window:].min()),
    }
