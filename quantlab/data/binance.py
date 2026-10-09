"""Binance — crypto market data. No API key needed for public market data.

Uses `data-api.binance.vision` (Binance's official public market-data host)
and falls back to `api.binance.com` if unreachable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd

from quantlab.data.base import DataSourceError, get_json

BASE_URLS = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3",
]

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _get(path: str, **params):
    last_error: Exception | None = None
    for base in BASE_URLS:
        try:
            return get_json(f"{base}{path}", params=params, retries=2)
        except DataSourceError as exc:
            last_error = exc
    raise last_error


def _to_ms(value) -> int | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _parse_klines(raw: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS).drop(columns="ignore")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"):
        df[col] = pd.to_numeric(df[col])
    return df.set_index("open_time")


def get_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """OHLCV candles indexed by open time (UTC).

    interval: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M. If `start` is given the
    function paginates past the 1000-bar API limit automatically.
    """
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    if start_ms is None:
        raw = _get("/klines", symbol=symbol, interval=interval, limit=min(limit, 1000))
        return _parse_klines(raw)

    frames, cursor = [], start_ms
    for _ in range(60):  # safety cap: 60k bars
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": cursor}
        if end_ms:
            params["endTime"] = end_ms
        raw = _get("/klines", **params)
        if not raw:
            break
        frames.append(_parse_klines(raw))
        if len(raw) < 1000:
            break
        cursor = raw[-1][0] + 1  # next ms after last open_time
    if not frames:
        raise DataSourceError(f"Binance returned no klines for {symbol!r}")
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")].sort_index()


def get_close_prices(
    symbols: Sequence[str],
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Close prices for several symbols aligned in one DataFrame."""
    closes = {
        s: get_klines(s, interval=interval, start=start, end=end, limit=limit)["close"]
        for s in symbols
    }
    return pd.DataFrame(closes)


def get_price(symbol: str | None = None):
    """Latest price for one symbol (float) or all symbols (Series)."""
    if symbol:
        return float(_get("/ticker/price", symbol=symbol)["price"])
    data = _get("/ticker/price")
    ser = pd.Series({d["symbol"]: float(d["price"]) for d in data}, name="price")
    return ser.sort_index()


def get_ticker_24h(symbol: str) -> dict:
    """24h rolling stats: last price, change %, high/low, base & quote volume."""
    t = _get("/ticker/24hr", symbol=symbol)
    return {
        "symbol": t["symbol"],
        "last_price": float(t["lastPrice"]),
        "change_pct": float(t["priceChangePercent"]),
        "high": float(t["highPrice"]),
        "low": float(t["lowPrice"]),
        "volume": float(t["volume"]),
        "quote_volume": float(t["quoteVolume"]),
        "trades": int(t["count"]),
    }


def get_tickers_24h(symbols: Sequence[str]) -> pd.DataFrame:
    """24h stats for several symbols as a DataFrame."""
    return pd.DataFrame([get_ticker_24h(s) for s in symbols]).set_index("symbol")


def get_order_book(symbol: str, limit: int = 100) -> dict[str, pd.DataFrame]:
    """Order-book snapshot: {'bids': ..., 'asks': ...} price/qty DataFrames."""
    data = _get("/depth", symbol=symbol, limit=limit)
    book = {}
    for side in ("bids", "asks"):
        df = pd.DataFrame(data[side], columns=["price", "qty"], dtype=float)
        book[side] = df
    return book
