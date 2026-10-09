"""Finnhub — real-time quotes, company fundamentals and news.

Free API key: https://finnhub.io/register (60 calls/min on the free tier).
Set FINNHUB_API_KEY in your environment or `.env`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantlab.config import get_key
from quantlab.data.base import get_json

BASE_URL = "https://finnhub.io/api/v1"

#: Commonly used keys inside the /stock/metric payload (there are ~100 more).
KEY_METRICS = [
    "marketCapitalization",
    "peTTM",
    "pb",
    "psTTM",
    "epsTTM",
    "roeTTM",
    "roaTTM",
    "grossMarginTTM",
    "operatingMarginTTM",
    "netProfitMarginTTM",
    "revenueGrowthTTMYoy",
    "epsGrowthTTMYoy",
    "currentRatioQuarterly",
    "totalDebt/totalEquityQuarterly",
    "dividendYieldIndicatedAnnual",
    "beta",
    "52WeekHigh",
    "52WeekLow",
]


def _get(path: str, **params):
    params["token"] = get_key("FINNHUB_API_KEY")
    return get_json(f"{BASE_URL}{path}", params=params)


def get_quote(symbol: str) -> dict:
    """Real-time quote. Fields: price, open, high, low, prev_close, change_pct."""
    q = _get("/quote", symbol=symbol)
    ts = pd.to_datetime(q["t"], unit="s", utc=True) if q.get("t") else None
    return {
        "symbol": symbol,
        "price": q.get("c"),
        "open": q.get("o"),
        "high": q.get("h"),
        "low": q.get("l"),
        "prev_close": q.get("pc"),
        "change": q.get("d"),
        "change_pct": q.get("dp"),
        "timestamp": ts,
    }


def get_quotes(symbols: Sequence[str]) -> pd.DataFrame:
    """Real-time quotes for several symbols (one API call each)."""
    df = pd.DataFrame([get_quote(s) for s in symbols]).set_index("symbol")
    return df


def get_company_profile(symbol: str) -> dict:
    """Name, exchange, industry, market cap, IPO date, logo, ..."""
    return _get("/stock/profile2", symbol=symbol)


def get_basic_financials(symbol: str) -> pd.Series:
    """All fundamental metrics for a symbol as a Series (P/E, ROE, margins...).

    Filter with `.reindex(KEY_METRICS)` for a readable summary.
    """
    data = _get("/stock/metric", symbol=symbol, metric="all")
    return pd.Series(data.get("metric", {}), name=symbol).sort_index()


def get_fundamentals(symbols: Sequence[str], metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Fundamentals comparison table: one column per symbol."""
    metrics = list(metrics) if metrics is not None else KEY_METRICS
    table = pd.concat([get_basic_financials(s) for s in symbols], axis=1)
    return table.reindex(metrics)


def get_company_news(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Company news between two YYYY-MM-DD dates, newest first."""
    articles = _get("/company-news", symbol=symbol, **{"from": start, "to": end})
    if not articles:
        return pd.DataFrame(columns=["datetime", "headline", "source", "summary", "url"])
    df = pd.DataFrame(articles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    cols = [c for c in ["datetime", "headline", "source", "summary", "url"] if c in df.columns]
    return df[cols].sort_values("datetime", ascending=False).reset_index(drop=True)


def get_market_news(category: str = "general") -> pd.DataFrame:
    """Latest market-wide news. Categories: general, forex, crypto, merger."""
    articles = _get("/news", category=category)
    df = pd.DataFrame(articles)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    cols = [c for c in ["datetime", "headline", "source", "summary", "url"] if c in df.columns]
    return df[cols].sort_values("datetime", ascending=False).reset_index(drop=True)


def search_symbols(query: str) -> pd.DataFrame:
    """Look up tickers by free-text query (company name, ISIN, ...)."""
    data = _get("/search", q=query)
    return pd.DataFrame(data.get("result", []))
