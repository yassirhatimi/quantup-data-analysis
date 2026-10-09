"""Vectorized backtesting."""

from quantlab.backtest.engine import BacktestResult, equal_weight_benchmark, run_backtest

__all__ = ["run_backtest", "equal_weight_benchmark", "BacktestResult"]
