"""QuantLab quick start — runs with NO API keys.

Pulls an ETF universe from Yahoo Finance, prints a risk table, backtests a
trend strategy against an equal-weight benchmark and saves the equity chart.

Run from the project root:  python examples/quickstart.py
"""

import matplotlib

matplotlib.use("Agg")  # headless: save the chart instead of opening a window

from quantlab import plotting
from quantlab.analysis import returns, risk, signals
from quantlab.backtest import equal_weight_benchmark, run_backtest
from quantlab.data import yahoo

plotting.apply_style()

# 1. Data — daily adjusted closes
prices = yahoo.get_prices(["SPY", "QQQ", "TLT", "GLD"], start="2015-01-01")
print(f"\nLoaded {prices.shape[0]:,} days x {prices.shape[1]} assets "
      f"({prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d})\n")

# 2. Risk snapshot
rets = returns.simple_returns(prices)
print("Risk summary (daily data, annualized):")
print(risk.summary(rets).round(3).to_string(), "\n")

# 3. Signal -> backtest (engine lags positions one bar: no lookahead)
weights = signals.sma_crossover(prices, fast=50, slow=200) / prices.shape[1]
strategy = run_backtest(prices, weights, cost_bps=5, name="Trend (50/200 SMA)")
benchmark = equal_weight_benchmark(prices)

print("Backtest vs benchmark:")
import pandas as pd

print(pd.concat([strategy.stats(), benchmark.stats()], axis=1).round(3).to_string(), "\n")

# 4. Chart
fig = strategy.plot(benchmark=benchmark)
out = "quickstart_equity.png"
fig.savefig(out)
print(f"Saved chart -> {out}")
