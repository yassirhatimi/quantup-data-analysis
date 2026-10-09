"""CFD instrument registry: US equity-index CFDs and spot gold.

What a CFD actually is: your broker's over-the-counter derivative that tracks
an underlying — US100 tracks the Nasdaq 100 (via its future), XAUUSD tracks
spot gold. Brokers add a spread and charge overnight financing (swap) on
positions held past the daily rollover.

For research we therefore proxy CFD prices with the underlying **continuous
front-month futures** from Yahoo Finance (no key, long history, ~24h sessions
like the CFDs themselves):

    US100 -> NQ=F   US500 -> ES=F   US30 -> YM=F   XAUUSD -> GC=F

Live-ish quotes come from Twelve Data spot (XAU/USD) and Finnhub ETF proxies
(QQQ/SPY/DIA) when keys are set, with a delayed Yahoo fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd

from quantlab.config import has_key
from quantlab.data import yahoo


@dataclass(frozen=True)
class Instrument:
    code: str                    # CFD-style code, e.g. "US100"
    name: str
    yahoo_future: str            # continuous front-month future (data proxy)
    yahoo_index: str | None      # cash index (RTH only, no volume)
    etf_proxy: str               # liquid ETF for free real-time-ish quotes
    twelvedata: str | None       # Twelve Data symbol when the free tier has it
    typical_spread_points: float  # typical retail CFD spread — CHECK YOUR BROKER
    session: str


INSTRUMENTS: dict[str, Instrument] = {
    "US100": Instrument(
        "US100", "Nasdaq 100 CFD", "NQ=F", "^NDX", "QQQ", None,
        2.0, "~23h/day Sun–Fri (CME Globex)",
    ),
    "US500": Instrument(
        "US500", "S&P 500 CFD", "ES=F", "^GSPC", "SPY", None,
        0.6, "~23h/day Sun–Fri (CME Globex)",
    ),
    "US30": Instrument(
        "US30", "Dow Jones 30 CFD", "YM=F", "^DJI", "DIA", None,
        2.5, "~23h/day Sun–Fri (CME Globex)",
    ),
    "XAUUSD": Instrument(
        "XAUUSD", "Gold spot CFD", "GC=F", None, "GLD", "XAU/USD",
        0.35, "~23h/day Sun–Fri",
    ),
    "EURUSD": Instrument(
        "EURUSD", "Euro / US dollar", "EURUSD=X", None, "FXE", "EUR/USD",
        0.0001, "24h/day Sun–Fri",
    ),
    "GBPUSD": Instrument(
        "GBPUSD", "British pound / US dollar", "GBPUSD=X", None, "FXB", "GBP/USD",
        0.00015, "24h/day Sun–Fri",
    ),
    "USOIL": Instrument(
        "USOIL", "WTI crude oil CFD", "CL=F", None, "USO", None,
        0.03, "~23h/day Sun–Fri (NYMEX)",
    ),
}

# Keep existing notebook/backtest portfolios stable when adding dashboard markets.
DEFAULT_CODES = ["US100", "US500", "US30", "XAUUSD"]


def registry_table() -> pd.DataFrame:
    """The instrument registry as a DataFrame (for display)."""
    return pd.DataFrame(
        {
            code: {
                "name": inst.name,
                "data proxy (Yahoo)": inst.yahoo_future,
                "cash index": inst.yahoo_index or "—",
                "ETF proxy": inst.etf_proxy,
                "typical spread (pts)": inst.typical_spread_points,
                "session": inst.session,
            }
            for code, inst in INSTRUMENTS.items()
        }
    ).T


def get_daily_prices(
    codes: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily closes of the underlying futures, columns named by CFD code."""
    codes = list(codes or DEFAULT_CODES)
    symbols = {INSTRUMENTS[c].yahoo_future: c for c in codes}
    prices = yahoo.get_prices(list(symbols), start=start, end=end)
    return prices.rename(columns=symbols)[codes]


def get_intraday(code: str, interval: str = "1h", days: int = 60) -> pd.DataFrame:
    """Intraday OHLCV of the underlying future via Yahoo.

    Yahoo limits: "1h" reaches ~730 days back, "15m"/"5m" ~60 days, "1m" ~7.
    """
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return yahoo.get_ohlcv(INSTRUMENTS[code].yahoo_future, start=start, interval=interval)


def get_live_quotes(codes: Sequence[str] | None = None) -> pd.DataFrame:
    """Best-available live quote per instrument.

    Preference order: Twelve Data spot (XAU/USD) -> Finnhub ETF proxy ->
    latest Yahoo daily close (delayed). The `source` column says which one
    you got; ETF proxies track the index but are not the CFD price itself.
    """
    rows = []
    for code in codes or DEFAULT_CODES:
        inst = INSTRUMENTS[code]
        row = None
        if inst.twelvedata and has_key("TWELVE_DATA_API_KEY"):
            try:
                from quantlab.data import twelvedata

                q = twelvedata.get_quote(inst.twelvedata)
                row = {
                    "price": float(q["close"]),
                    "change_pct": float(q.get("percent_change", float("nan"))),
                    "source": f"Twelve Data {inst.twelvedata} (spot)",
                }
            except Exception:
                row = None
        if row is None and has_key("FINNHUB_API_KEY"):
            try:
                from quantlab.data import finnhub

                q = finnhub.get_quote(inst.etf_proxy)
                row = {
                    "price": q["price"],
                    "change_pct": q["change_pct"],
                    "source": f"Finnhub {inst.etf_proxy} (ETF proxy)",
                }
            except Exception:
                row = None
        if row is None:
            week_ago = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            px = yahoo.get_prices(inst.yahoo_future, start=week_ago)[inst.yahoo_future].dropna()
            row = {
                "price": float(px.iloc[-1]),
                "change_pct": float(px.pct_change().iloc[-1] * 100),
                "source": f"Yahoo {inst.yahoo_future} (delayed close)",
            }
        rows.append({"code": code, "instrument": inst.name, **row})
    return pd.DataFrame(rows).set_index("code")


def one_way_spread_bps(codes: Sequence[str] | None = None, prices: pd.Series | None = None) -> pd.Series:
    """Cost of crossing the spread once, in bps of price, per instrument.

    Uses `typical_spread_points / 2` against the latest price (pass a
    `prices` Series indexed by code to reuse data you already have).
    """
    codes = list(codes or DEFAULT_CODES)
    if prices is None:
        last = get_daily_prices(codes).ffill().iloc[-1]
    else:
        last = prices
    return pd.Series(
        {c: INSTRUMENTS[c].typical_spread_points / 2 / last[c] * 1e4 for c in codes},
        name="one_way_spread_bps",
    )
