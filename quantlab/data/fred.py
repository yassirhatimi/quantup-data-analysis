"""FRED — economic data from the St. Louis Fed.

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in your environment or `.env`.

Popular series ids
------------------
GDP        Nominal GDP (quarterly)          CPIAUCSL   CPI index (monthly)
UNRATE     Unemployment rate                FEDFUNDS   Fed funds effective rate
DGS10      10y Treasury yield (daily)       DGS2       2y Treasury yield (daily)
T10Y2Y     10y-2y spread (daily)            VIXCLS     VIX close (daily)
DEXUSEU    USD/EUR exchange rate            M2SL       M2 money stock
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantlab.config import get_key
from quantlab.data.base import get_json

BASE_URL = "https://api.stlouisfed.org/fred"


def _get(path: str, **params):
    params.update({"api_key": get_key("FRED_API_KEY"), "file_type": "json"})
    return get_json(f"{BASE_URL}{path}", params=params)


def get_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """One FRED series as a float Series indexed by date ('.' becomes NaN)."""
    params: dict = {"series_id": series_id}
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    data = _get("/series/observations", **params)
    obs = data.get("observations", [])
    if not obs:
        raise ValueError(f"FRED returned no observations for {series_id!r}")
    df = pd.DataFrame(obs)
    values = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
    return pd.Series(values.values, index=pd.to_datetime(df["date"]), name=series_id)


def get_series_frame(series_ids: Iterable[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Several series aligned into one DataFrame (outer join on dates)."""
    return pd.concat([get_series(s, start, end) for s in series_ids], axis=1).sort_index()


def get_series_info(series_id: str) -> dict:
    """Metadata: title, units, frequency, seasonal adjustment, last update."""
    data = _get("/series", series_id=series_id)
    seriess = data.get("seriess", [])
    return seriess[0] if seriess else {}


def search(text: str, limit: int = 20) -> pd.DataFrame:
    """Full-text search over series, most popular first."""
    data = _get(
        "/series/search",
        search_text=text,
        limit=limit,
        order_by="popularity",
        sort_order="desc",
    )
    df = pd.DataFrame(data.get("seriess", []))
    cols = [c for c in ["id", "title", "frequency_short", "units_short", "popularity", "observation_start", "observation_end"] if c in df.columns]
    return df[cols] if not df.empty else df
