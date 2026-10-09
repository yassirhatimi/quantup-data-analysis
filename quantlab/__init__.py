"""QuantLab — a quantitative research toolkit.

Data clients (one module per source), return/risk analytics, signal
construction and a vectorized backtester.

Data sources
------------
- quantlab.data.yahoo       Historical prices (Yahoo Finance, no key)
- quantlab.data.finnhub     Real-time quotes, fundamentals, news (free key)
- quantlab.data.twelvedata  Real-time & intraday prices (free key)
- quantlab.data.fred        Economic data (free key)
- quantlab.data.news        News search via NewsAPI (free key)
- quantlab.data.polymarket  Prediction markets (no key)
- quantlab.data.binance     Crypto market data (no key)

Analytics
---------
- quantlab.analysis.returns   Simple/log/cumulative returns, rebasing
- quantlab.analysis.risk      Vol, Sharpe, Sortino, drawdowns, VaR/CVaR, beta
- quantlab.analysis.signals   SMA/EMA crossovers, momentum, RSI, z-scores
- quantlab.backtest.engine    Vectorized backtests with transaction costs
- quantlab.plotting           Matplotlib style + validated palette
"""

from quantlab import config

__version__ = "0.1.0"
__all__ = ["config", "__version__"]
