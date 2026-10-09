"""NewsAPI — broad news search across ~150k outlets.

Free developer key: https://newsapi.org/register (100 requests/day; articles
older than one month require a paid plan).
Set NEWSAPI_API_KEY in your environment or `.env`.

For company-specific financial news, `quantlab.data.finnhub.get_company_news`
is usually the better source.
"""

from __future__ import annotations

import pandas as pd

from quantlab.config import get_key
from quantlab.data.base import DataSourceError, get_json

BASE_URL = "https://newsapi.org/v2"


def _get(path: str, **params):
    params["apiKey"] = get_key("NEWSAPI_API_KEY")
    data = get_json(f"{BASE_URL}{path}", params=params)
    if data.get("status") != "ok":
        raise DataSourceError(f"NewsAPI error: {data.get('code')} {data.get('message')}")
    return data


def _to_frame(articles: list[dict]) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame(columns=["published_at", "source", "title", "description", "url"])
    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    df["source"] = df["source"].apply(lambda s: s.get("name") if isinstance(s, dict) else s)
    return (
        df[["published_at", "source", "title", "description", "url"]]
        .sort_values("published_at", ascending=False)
        .reset_index(drop=True)
    )


def search_news(
    query: str,
    start: str | None = None,
    end: str | None = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 50,
) -> pd.DataFrame:
    """Search all articles for a query ('Tesla', '"interest rates" AND Fed'...).

    sort_by: publishedAt | relevancy | popularity.
    """
    params: dict = {"q": query, "language": language, "sortBy": sort_by, "pageSize": page_size}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    return _to_frame(_get("/everything", **params).get("articles", []))


def top_headlines(category: str = "business", country: str = "us", page_size: int = 50) -> pd.DataFrame:
    """Current top headlines. Categories: business, technology, general, science..."""
    data = _get("/top-headlines", category=category, country=country, pageSize=page_size)
    return _to_frame(data.get("articles", []))
