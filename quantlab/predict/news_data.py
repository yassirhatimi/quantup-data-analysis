"""Headline scoring — a transparent keyword lexicon, labelled as the proxy it is.

This is **not** sentiment analysis and the output never claims to be. It counts
occurrences of per-asset bullish and bearish phrases across recent headlines
and turns the imbalance into a weak prior. Two guards do most of the work:

- **A financial-context filter.** An article must mention something from the
  asset's context list, or "gold medal" and "European Pro League" flood the
  sample. The count of what was dropped is reported, not hidden.
- **Word-boundary matching**, so "war" does not fire on "toward" and "cut" does
  not fire on "haircut".

Because the lexicon is crude, the score is deliberately compressed: even a
uniformly one-sided news flow cannot drive this channel past about 0.6, and
its default weight in the blend is 10%.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import has_key
from quantlab.predict import assets as asset_spec
from quantlab.predict.component import ComponentScore, clamp, missing

_TTL = 900.0
_cache: dict[str, tuple[float, object]] = {}

#: Headlines older than this contribute nothing — stale news is not news.
MAX_AGE_HOURS = 48
#: Cap on the per-article tilt so one hysterical headline cannot swing the run.
MAX_ARTICLE_TILT = 3


def clear_cache() -> None:
    _cache.clear()


def _cached(key: str, ttl: float, fn):
    hit = _cache.get(key)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


@dataclass
class Headline:
    when: pd.Timestamp
    source: str
    title: str
    tilt: int
    matched: list[str]
    url: str = ""


def _hits(text: str, words) -> list[str]:
    found = []
    for w in words:
        if re.search(rf"\b{re.escape(w)}", text):
            found.append(w)
    return found


def _fetch_newsapi(a: asset_spec.Asset, now: pd.Timestamp) -> pd.DataFrame:
    from quantlab.data import news
    start = (now - pd.Timedelta(hours=MAX_AGE_HOURS)).strftime("%Y-%m-%d")
    return news.search_news(a.news_query, start=start, sort_by="relevancy",
                            page_size=100)


def _fetch_finnhub(now: pd.Timestamp) -> pd.DataFrame:
    from quantlab.data import finnhub
    df = finnhub.get_market_news("forex")
    if df.empty:
        return df
    return df.rename(columns={"datetime": "published_at", "headline": "title",
                              "summary": "description"})


def score_news(code: str, now: pd.Timestamp | None = None) -> ComponentScore:
    a = asset_spec.get(code)
    now = now or pd.Timestamp.now(tz="UTC")

    if not (has_key("NEWSAPI_API_KEY") or has_key("FINNHUB_API_KEY")):
        return missing("news", "no NEWSAPI_API_KEY or FINNHUB_API_KEY configured")

    frames: list[pd.DataFrame] = []
    problems: list[str] = []

    if has_key("NEWSAPI_API_KEY"):
        try:
            frames.append(_cached(f"newsapi:{code}", _TTL,
                                  lambda: _fetch_newsapi(a, now)))
        except Exception as exc:
            problems.append(f"NewsAPI {type(exc).__name__}: {str(exc)[:70]}")
    if has_key("FINNHUB_API_KEY"):
        try:
            frames.append(_cached("finnhub:forex", _TTL, lambda: _fetch_finnhub(now)))
        except Exception as exc:
            problems.append(f"Finnhub {type(exc).__name__}: {str(exc)[:70]}")

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return missing("news", "; ".join(problems) or "no articles returned")

    df = pd.concat(frames, ignore_index=True)
    if "published_at" not in df.columns:
        return missing("news", "news payload had no timestamps")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df[df["published_at"] >= now - pd.Timedelta(hours=MAX_AGE_HOURS)]
    if df.empty:
        return missing("news", f"no articles in the last {MAX_AGE_HOURS}h", "stale")

    title = df["title"].fillna("")
    desc = df["description"].fillna("") if "description" in df.columns else ""
    text = (title + " " + desc).str.lower()

    keep = text.apply(lambda t: any(k in t for k in a.news_context))
    dropped = int((~keep).sum())
    df, text = df[keep], text[keep]
    if df.empty:
        return missing("news",
                       f"none of {dropped} articles carried financial context")

    headlines: list[Headline] = []
    for (_, row), body in zip(df.iterrows(), text):
        bull = _hits(body, a.news_bullish)
        bear = _hits(body, a.news_bearish)
        tilt = int(np.clip(len(bull) - len(bear), -MAX_ARTICLE_TILT, MAX_ARTICLE_TILT))
        headlines.append(Headline(
            row["published_at"], str(row.get("source", "?")), str(row.get("title", "")),
            tilt, bull if tilt > 0 else bear, str(row.get("url", ""))))

    # Recency weighting: a headline from 40 hours ago is barely news.
    ages = np.array([(now - h.when).total_seconds() / 3600 for h in headlines])
    weights = np.exp(-ages / 12.0)
    tilts = np.array([h.tilt for h in headlines], dtype=float)
    if weights.sum() <= 0:
        return missing("news", "all articles fell outside the recency window", "stale")

    mean_tilt = float(np.sum(tilts * weights) / np.sum(weights))
    # 1.5 scaling then a hard clip at 0.6: the lexicon is not good enough to be
    # allowed a full-strength vote.
    score = clamp(mean_tilt * 1.5, -0.6, 0.6)

    newest = max(h.when for h in headlines)
    age_s = (now - newest).total_seconds()
    status = "live" if age_s < 3 * 3600 else ("delayed" if age_s < 12 * 3600 else "stale")

    ranked = sorted(headlines, key=lambda h: (-abs(h.tilt), -h.when.timestamp()))
    ev = [f"`{h.tilt:+d}` {h.source}: {h.title[:110]}"
          + (f" [{', '.join(h.matched[:3])}]" if h.matched else "")
          for h in ranked[:4] if h.tilt != 0]
    if not ev:
        ev = [f"{len(headlines)} finance-filtered headlines, none matched the "
              f"bullish or bearish lexicon — reading as neutral"]

    note = f"keyword proxy over {len(headlines)} articles"
    if dropped:
        note += f" ({dropped} dropped for no financial context)"
    if problems:
        note += "; " + "; ".join(problems)

    return ComponentScore("news", score, evidence=ev, note=note, status=status,
                          detail={
                              "n_articles": len(headlines), "n_dropped": dropped,
                              "mean_tilt": mean_tilt,
                              "newest": str(newest), "age_hours": age_s / 3600,
                              "headlines": [{"when": str(h.when), "source": h.source,
                                             "title": h.title, "tilt": h.tilt,
                                             "matched": h.matched, "url": h.url}
                                            for h in ranked[:10]],
                          })
