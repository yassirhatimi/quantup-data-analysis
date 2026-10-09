"""Polymarket — prediction-market prices. No API key needed for market data.

Prices live in [0, 1] and read as the market-implied probability of the
outcome. Two public APIs are used:

- Gamma API (market/event metadata): https://gamma-api.polymarket.com
- CLOB API (price history):          https://clob.polymarket.com
"""

from __future__ import annotations

import json

import pandas as pd

from quantlab.data.base import DataSourceError, get_json

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


def _parse_json_field(value):
    """Gamma encodes list fields (outcomes, prices, token ids) as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def get_markets(
    limit: int = 50,
    active: bool = True,
    closed: bool = False,
    order: str = "volume24hr",
    ascending: bool = False,
) -> pd.DataFrame:
    """Markets sorted by `order` (volume24hr, liquidity, endDate...).

    Columns include the market question, outcome names/prices, the implied
    probability of the first outcome (usually "Yes") and CLOB token ids for
    price-history lookups.
    """
    params = {
        "limit": limit,
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "order": order,
        "ascending": str(ascending).lower(),
    }
    raw = get_json(f"{GAMMA_URL}/markets", params=params)
    if not raw:
        raise DataSourceError("Polymarket Gamma API returned no markets")
    rows = []
    for m in raw:
        outcomes = _parse_json_field(m.get("outcomes")) or []
        prices = [float(p) for p in (_parse_json_field(m.get("outcomePrices")) or [])]
        token_ids = _parse_json_field(m.get("clobTokenIds")) or []
        rows.append(
            {
                "question": m.get("question"),
                "outcomes": outcomes,
                "outcome_prices": prices,
                "prob_first_outcome": prices[0] if prices else None,
                "volume_24h": float(m["volume24hr"]) if m.get("volume24hr") is not None else None,
                "liquidity": float(m["liquidityNum"]) if m.get("liquidityNum") is not None else None,
                "end_date": pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"),
                "clob_token_ids": token_ids,
                "slug": m.get("slug"),
            }
        )
    return pd.DataFrame(rows)


def get_events(limit: int = 25, order: str = "volume24hr") -> pd.DataFrame:
    """Events (groups of related markets, e.g. an election with many candidates)."""
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": order,
        "ascending": "false",
    }
    raw = get_json(f"{GAMMA_URL}/events", params=params)
    rows = [
        {
            "title": e.get("title"),
            "n_markets": len(e.get("markets", []) or []),
            "volume_24h": float(e["volume24hr"]) if e.get("volume24hr") is not None else None,
            "liquidity": float(e["liquidity"]) if e.get("liquidity") is not None else None,
            "end_date": pd.to_datetime(e.get("endDate"), utc=True, errors="coerce"),
            "slug": e.get("slug"),
        }
        for e in raw or []
    ]
    return pd.DataFrame(rows)


def get_price_history(
    clob_token_id: str,
    interval: str = "1w",
    fidelity: int = 60,
) -> pd.Series:
    """Price (= implied probability) history for one outcome token.

    Parameters
    ----------
    clob_token_id : one entry of a market's `clob_token_ids` (first = "Yes").
    interval : lookback window — 1h, 6h, 1d, 1w, 1m, max.
    fidelity : bar resolution in minutes.
    """
    params = {"market": clob_token_id, "interval": interval, "fidelity": fidelity}
    data = get_json(f"{CLOB_URL}/prices-history", params=params)
    history = data.get("history", [])
    if not history:
        raise DataSourceError(f"No price history for token {clob_token_id!r}")
    df = pd.DataFrame(history)
    return pd.Series(
        df["p"].astype(float).values,
        index=pd.to_datetime(df["t"], unit="s", utc=True),
        name="probability",
    )
