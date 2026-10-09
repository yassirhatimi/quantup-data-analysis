"""Return computations. Input: prices or returns as Series/DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Period-over-period percentage returns."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Log returns — additive across time, convenient for statistics."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compounded growth path of the return stream (starts near 0)."""
    return (1 + returns.fillna(0)).cumprod() - 1


def annualized_return(returns: pd.Series | pd.DataFrame, periods_per_year: int = TRADING_DAYS):
    """Geometric (compound) annual growth rate of a return stream."""
    compounded = (1 + returns).prod()
    n = returns.notna().sum() if isinstance(returns, pd.Series) else len(returns)
    return compounded ** (periods_per_year / n) - 1


def rebase(prices: pd.Series | pd.DataFrame, base: float = 100.0):
    """Index prices to a common starting value — the right way to compare
    assets of different price levels on ONE axis (never use a dual axis)."""
    return prices / prices.iloc[0] * base
