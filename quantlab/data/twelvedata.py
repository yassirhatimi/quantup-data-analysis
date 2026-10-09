"""Twelve Data — real-time prices and intraday time series.

Free API key: https://twelvedata.com (free tier: 8 calls/min, 800/day).
Set TWELVE_DATA_API_KEY in your environment or `.env`.
"""

from __future__ import annotations

import pandas as pd

from quantlab.config import get_key
from quantlab.data.base import DataSourceError, get_json

BASE_URL = "https://api.twelvedata.com"


def _get(path: str, **params):
    params["apikey"] = get_key("TWELVE_DATA_API_KEY")
    data = get_json(f"{BASE_URL}{path}", params=params)
    if isinstance(data, dict) and data.get("status") == "error":
        raise DataSourceError(f"Twelve Data error: {data.get('message')}")
    return data


def get_price(symbol: str) -> float:
    """Latest price for a symbol (stocks, FX like 'EUR/USD', crypto 'BTC/USD')."""
    return float(_get("/price", symbol=symbol)["price"])


def get_quote(symbol: str) -> dict:
    """Rich real-time quote: OHLC, previous close, change, 52-week range."""
    return _get("/quote", symbol=symbol)


def get_time_series(
    symbol: str,
    interval: str = "5min",
    outputsize: int = 390,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Intraday/daily OHLCV bars, oldest first, indexed by timestamp.

    interval: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month.
    """
    params: dict = {"symbol": symbol, "interval": interval, "outputsize": outputsize}
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    data = _get("/time_series", **params)
    values = data.get("values", [])
    if not values:
        raise DataSourceError(f"Twelve Data returned no bars for {symbol!r}")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
