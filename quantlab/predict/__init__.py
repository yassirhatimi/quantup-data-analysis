"""Real-time market probability prediction for XAUUSD, EURUSD and GBPUSD.

A live engine that answers one question per asset per horizon: what are the
odds the market goes UP, goes DOWN, or stays in a RANGE?

    from quantlab.predict import engine
    result = engine.predict("XAUUSD")
    print(result.horizons["1h"].up_pct, result.confidence_label)

Design in one paragraph
-----------------------
RANGE only means something once you say how big a move counts, so the band is
defined explicitly: |return| <= band_k * sigma_baseline(h), where the baseline
comes from slow ATR(14) while the *predictive* sigma comes from fast EWMA
volatility. Vol expanding against its own baseline therefore shrinks RANGE by
construction. Returns are standardised Student-t (tail index from realised
kurtosis), variance-time scaled by the hour-of-day profile so "next 15 minutes"
in the New York overlap is not the same bet as 15 minutes in the Asian lull.
Direction enters *only* as drift, mu = kappa * score * sigma, with kappa capped
low and shrunk further by model confidence — which is what stops the engine
printing 87% because five indicators happened to agree.

Modules
-------
- config           weights, horizons, thresholds (JSON-overridable, backtestable)
- assets           per-asset spec: symbols, spreads, macro drivers, Polymarket rules
- market_data      live price + multi-timeframe bars, call budget, staleness
- technical        indicator suite, support/resistance, market structure
- macro_data       FRED drivers and the real release calendar
- news_data        headlines scored by a transparent keyword lexicon (a proxy)
- polymarket_data  relevant prediction markets, normalised, change-tracked
- regime           market-regime and trading-session classification
- engine           components -> weighted score -> probabilities + explanation
- quality          per-source data-quality accounting
- store            SQLite prediction history and outcome scoring
- backtest         historical replay, calibration, weight optimisation
- render           HTML dashboard components and text output
"""

from __future__ import annotations

__all__ = [
    "assets",
    "backtest",
    "config",
    "engine",
    "macro_data",
    "market_data",
    "news_data",
    "polymarket_data",
    "quality",
    "regime",
    "render",
    "store",
    "technical",
]
