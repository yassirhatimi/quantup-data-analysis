"""Historical prices from Yahoo Finance (via `yfinance`). No API key needed.

Best for daily/weekly/monthly history on stocks, ETFs, indices, FX and
commodities. Intraday history is limited (e.g. 1m bars only ~7 days back).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf


def get_prices(
    tickers: str | Sequence[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, indexed by date.

    Parameters
    ----------
    tickers : "AAPL" or ["AAPL", "MSFT", ...]
    start, end : "YYYY-MM-DD" strings (end is exclusive). Defaults to max history.
    interval : "1d", "1wk", "1mo" (or intraday like "1h" for recent data).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        raise ValueError(f"Yahoo Finance returned no data for {list(tickers)}")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    failed = [c for c in close.columns if close[c].isna().all()]
    if failed:
        print(f"Warning: no data for {failed} (bad ticker or delisted); dropping.")
        close = close.drop(columns=failed)
    return close.dropna(how="all").sort_index()


def get_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Full OHLCV history for a single ticker (Open/High/Low/Close/Volume)."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker!r}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.2.x single ticker
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    return df.sort_index()


def get_info(ticker: str) -> dict:
    """Company/instrument metadata (sector, market cap, ...). Best effort —
    this scrapes Yahoo and occasionally returns partial data."""
    return dict(yf.Ticker(ticker).info)


def get_dividends(ticker: str) -> pd.Series:
    """Dividend history as a Series indexed by ex-date."""
    return yf.Ticker(ticker).dividends
