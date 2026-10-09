"""A small vectorized backtest engine.

Model: `positions` are portfolio weights known at the close of bar t; the
engine holds them over bar t+1 (shift by one — no lookahead) and charges
`cost_bps` on traded notional whenever weights change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.analysis import risk

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series          # portfolio value, starts at initial_capital
    returns: pd.Series         # net portfolio returns per bar
    positions: pd.DataFrame    # weights actually held (already shifted)
    turnover: pd.Series        # traded notional per bar (|Δweight| summed)
    cost_bps: float
    periods_per_year: int = TRADING_DAYS
    _stats_cache: pd.Series | None = field(default=None, repr=False)

    def drawdown(self) -> pd.Series:
        """Drawdown path of the strategy (<= 0)."""
        return self.equity / self.equity.cummax() - 1

    def stats(self) -> pd.Series:
        """Headline performance statistics."""
        if self._stats_cache is not None:
            return self._stats_cache
        r = self.returns
        n = len(r)
        ppy = self.periods_per_year
        years = n / ppy
        total_return = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        max_dd = self.drawdown().min()
        active = r[self.positions.abs().sum(axis=1) > 0]
        self._stats_cache = pd.Series(
            {
                "start": r.index[0],
                "end": r.index[-1],
                "total_return": total_return,
                "cagr": cagr,
                "ann_volatility": risk.annualized_volatility(r, ppy),
                "sharpe": risk.sharpe_ratio(r, periods_per_year=ppy),
                "sortino": risk.sortino_ratio(r, periods_per_year=ppy),
                "max_drawdown": max_dd,
                "calmar": cagr / abs(max_dd) if max_dd else np.nan,
                "hit_rate": (active > 0).mean() if len(active) else np.nan,
                "exposure": self.positions.abs().sum(axis=1).mean(),
                "ann_turnover": self.turnover.sum() / years if years > 0 else np.nan,
                "n_rebalances": int((self.turnover > 1e-12).sum()),
            },
            name=self.name,
        )
        return self._stats_cache

    def plot(self, benchmark: "BacktestResult | None" = None, ax=None):
        """Equity curve + drawdown panel. Pass another result to compare."""
        import matplotlib.pyplot as plt

        from quantlab import plotting

        fig, axes = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
        results = [self] + ([benchmark] if benchmark is not None else [])
        for i, res in enumerate(results):
            axes[0].plot(res.equity, color=plotting.PALETTE[i], label=res.name)
            axes[1].plot(res.drawdown(), color=plotting.PALETTE[i], linewidth=1.4)
        axes[0].set_title("Equity curve")
        if len(results) > 1:
            axes[0].legend()
        axes[1].set_title("Drawdown")
        axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        fig.tight_layout()
        return fig


def _as_frame(x: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    return x.to_frame(x.name or name) if isinstance(x, pd.Series) else x


def run_backtest(
    prices: pd.Series | pd.DataFrame,
    positions: pd.Series | pd.DataFrame,
    cost_bps: float | pd.Series = 5.0,
    financing_rate_annual: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: int = TRADING_DAYS,
    name: str = "strategy",
) -> BacktestResult:
    """Backtest weights against prices.

    Parameters
    ----------
    prices : close prices (Series or DataFrame, one column per asset).
    positions : target weights on the same index/columns. 1 = 100% long that
        asset. A multi-asset DataFrame is a portfolio (weights sum per row).
    cost_bps : one-way transaction cost in basis points of traded notional.
        Scalar, or a Series indexed by asset for per-instrument spreads
        (e.g. CFD half-spreads from `instruments.one_way_spread_bps`).
    financing_rate_annual : annualized holding cost charged on gross exposure
        every bar — models CFD overnight financing/swap (roughly benchmark
        rate + broker markup; check your broker's swap rates). 0 for ETFs.
    """
    prices = _as_frame(prices, "asset")
    positions = _as_frame(positions, "asset")
    positions = positions.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    if np.isscalar(cost_bps):
        cost_per_asset = pd.Series(float(cost_bps), index=prices.columns)
    else:
        cost_per_asset = pd.Series(cost_bps).reindex(prices.columns).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    held = positions.shift(1).fillna(0.0)              # held over the NEXT bar
    traded = held.diff().abs().fillna(0.0)
    turnover = traded.sum(axis=1)
    costs = traded.mul(cost_per_asset / 1e4, axis=1).sum(axis=1)
    financing = held.abs().sum(axis=1) * (financing_rate_annual / periods_per_year)
    portfolio_returns = (held * asset_returns).sum(axis=1) - costs - financing
    equity = initial_capital * (1 + portfolio_returns).cumprod()

    return BacktestResult(
        name=name,
        equity=equity,
        returns=portfolio_returns,
        positions=held,
        turnover=turnover,
        cost_bps=float(cost_per_asset.mean()),
        periods_per_year=periods_per_year,
    )


def equal_weight_benchmark(
    prices: pd.Series | pd.DataFrame,
    cost_bps: float | pd.Series = 0.0,
    financing_rate_annual: float = 0.0,
    name: str = "equal-weight buy & hold",
) -> BacktestResult:
    """Daily-rebalanced equal-weight portfolio of the same assets — the
    natural benchmark for strategies built on that universe."""
    prices = _as_frame(prices, "asset")
    weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    return run_backtest(
        prices, weights, cost_bps=cost_bps,
        financing_rate_annual=financing_rate_annual, name=name,
    )
