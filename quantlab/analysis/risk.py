"""Risk and performance statistics on return streams."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS):
    """Standard deviation of returns, scaled to one year."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Annualized excess return per unit of volatility."""
    excess = returns - risk_free_rate / periods_per_year
    vol = returns.std()
    return excess.mean() / vol * np.sqrt(periods_per_year)


def sortino_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Like Sharpe but penalizes only downside volatility."""
    excess = returns - risk_free_rate / periods_per_year
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean())
    return excess.mean() / downside * np.sqrt(periods_per_year)


def drawdown_series(returns):
    """Drawdown path: percentage below the running equity peak (<= 0)."""
    equity = (1 + returns.fillna(0)).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns):
    """Worst peak-to-trough loss (negative number)."""
    return drawdown_series(returns).min()


def historical_var(returns, level: float = 0.95):
    """Historical Value-at-Risk: loss exceeded on (1-level) of periods (positive)."""
    return -returns.quantile(1 - level)


def historical_cvar(returns, level: float = 0.95):
    """Expected shortfall: average loss on the worst (1-level) of periods (positive)."""
    cutoff = returns.quantile(1 - level)
    if isinstance(returns, pd.DataFrame):
        return pd.Series({c: -returns[c][returns[c] <= cutoff[c]].mean() for c in returns}, name="cvar")
    return -returns[returns <= cutoff].mean()


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Sensitivity of an asset to a market benchmark (cov/var)."""
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    cov = aligned.cov()
    return cov.iloc[0, 1] / cov.iloc[1, 1]


def rolling_volatility(returns, window: int = 21, periods_per_year: int = TRADING_DAYS):
    """Rolling annualized volatility (default window ~1 trading month)."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 90) -> pd.Series:
    """Rolling correlation between two return streams."""
    return a.rolling(window).corr(b)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of return columns."""
    return returns.corr()


def summary(returns, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """One table of the headline stats. Columns = assets, rows = metrics."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name or "asset")
    stats = {}
    for col in returns.columns:
        r = returns[col].dropna()
        n = len(r)
        ann_ret = (1 + r).prod() ** (periods_per_year / n) - 1 if n else np.nan
        stats[col] = {
            "ann_return": ann_ret,
            "ann_volatility": annualized_volatility(r, periods_per_year),
            "sharpe": sharpe_ratio(r, risk_free_rate, periods_per_year),
            "sortino": sortino_ratio(r, risk_free_rate, periods_per_year),
            "max_drawdown": max_drawdown(r),
            "var_95": historical_var(r),
            "cvar_95": historical_cvar(r),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
            "best_period": r.max(),
            "worst_period": r.min(),
            "n_periods": n,
        }
    return pd.DataFrame(stats)
