"""Signal construction. Signals return *positions* (0/1 or weights) aligned
to the price index; the backtest engine shifts them to avoid lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wrap_like(values: np.ndarray, ref: pd.Series | pd.DataFrame):
    if isinstance(ref, pd.DataFrame):
        return pd.DataFrame(values, index=ref.index, columns=ref.columns)
    return pd.Series(values, index=ref.index, name=ref.name)


def sma(prices, window: int = 20):
    """Simple moving average."""
    return prices.rolling(window).mean()


def ema(prices, span: int = 20):
    """Exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()


def sma_crossover(prices, fast: int = 50, slow: int = 200):
    """Trend-following: long (1) while the fast SMA is above the slow SMA."""
    signal = (sma(prices, fast) > sma(prices, slow)).astype(float)
    signal[sma(prices, slow).isna()] = 0.0  # warm-up period: stay flat
    return signal


def momentum(prices, lookback: int = 252, skip: int = 21):
    """Total return over `lookback` periods, skipping the most recent `skip`
    (the classic 12-1 momentum uses lookback=252, skip=21 on daily bars)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def tsmom_signal(prices, lookback: int = 252, skip: int = 21):
    """Time-series momentum: long (1) when trailing momentum is positive."""
    mom = momentum(prices, lookback, skip)
    return (mom > 0).astype(float).where(mom.notna(), 0.0)


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21, top_n: int = 2) -> pd.DataFrame:
    """Relative momentum: equal weight the `top_n` assets by trailing momentum,
    hold nothing during the warm-up period."""
    mom = momentum(prices, lookback, skip)
    ranks = mom.rank(axis=1, ascending=False)
    weights = (ranks <= top_n).astype(float).div(top_n)
    return weights.where(mom.notna(), 0.0)


def rsi(prices, window: int = 14):
    """Wilder's Relative Strength Index in [0, 100]. <30 oversold, >70 overbought."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def zscore(prices, window: int = 20):
    """How many rolling standard deviations the price sits from its rolling mean."""
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / std


def mean_reversion_signal(prices, window: int = 20, entry_z: float = -1.0, exit_z: float = 0.0):
    """Long (1) when the z-score drops below `entry_z`, flat (0) once it
    recovers above `exit_z`; holds in between."""
    z = zscore(prices, window)
    raw = np.where(z < entry_z, 1.0, np.where(z > exit_z, 0.0, np.nan))
    return _wrap_like(raw, z).ffill().fillna(0.0)
