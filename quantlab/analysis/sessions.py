"""Trading-session analytics for ~24h instruments (index & gold CFDs).

Sessions are defined in UTC (approximate, fixed year-round):

    Asia      00:00–07:00   (Tokyo/Hong Kong/Singapore)
    London    07:00–13:00   (European cash hours until the US takes over)
    New York  13:00–21:00   (US cash open ~13:30 UTC in summer, 14:30 winter)

Feed these functions hourly bars (e.g. `instruments.get_intraday(code, "1h")`);
any timezone-aware index is converted to UTC internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS: dict[str, tuple[int, int]] = {
    "Asia": (0, 7),
    "London": (7, 13),
    "New York": (13, 21),
}


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def label_sessions(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add `session` and `date` (UTC) columns to an hourly OHLC frame."""
    df = _to_utc(hourly)
    hours = df.index.hour
    session = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (start, end) in SESSIONS.items():
        session[(hours >= start) & (hours < end)] = name
    df["session"] = session
    df["date"] = df.index.date
    return df.dropna(subset=["session"])


def session_ohlc(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, session): open/high/low/close of that session."""
    df = label_sessions(hourly)
    grouped = df.groupby(["date", "session"])
    out = pd.DataFrame(
        {
            "open": grouped["Open"].first(),
            "high": grouped["High"].max(),
            "low": grouped["Low"].min(),
            "close": grouped["Close"].last(),
        }
    )
    out["range"] = out["high"] - out["low"]
    out["ret"] = out["close"] / out["open"] - 1
    return out.reset_index()


def session_stats(hourly: pd.DataFrame) -> pd.DataFrame:
    """Historical per-session tendencies over the supplied window.

    Returns one row per session:
      avg_range / median_range : session high-low in price points
      range_share              : share of the summed daily session ranges
      day_match_rate           : how often the session moved in the same
                                 direction as the full day (which sessions
                                 "make" the day vs chop around)
    """
    so = session_ohlc(hourly)
    day = so.groupby("date").agg(day_open=("open", "first"), day_close=("close", "last"))
    day["day_ret"] = day["day_close"] / day["day_open"] - 1
    so = so.merge(day[["day_ret"]], on="date")
    so = so[so["day_ret"] != 0]
    so["match"] = np.sign(so["ret"]) == np.sign(so["day_ret"])

    stats = so.groupby("session").agg(
        avg_range=("range", "mean"),
        median_range=("range", "median"),
        day_match_rate=("match", "mean"),
        n_days=("date", "nunique"),
    )
    stats["range_share"] = stats["avg_range"] / stats["avg_range"].sum()
    return stats.reindex(list(SESSIONS)).round(
        {"avg_range": 2, "median_range": 2, "day_match_rate": 3, "range_share": 3}
    )


def todays_sessions(hourly: pd.DataFrame) -> pd.DataFrame:
    """Today's (UTC) per-session OHLC so far — e.g. the Asia range London
    traders break out of. Sessions not yet traded are simply absent."""
    so = session_ohlc(hourly)
    today = pd.Timestamp.now(tz="UTC").date()
    out = so[so["date"] == today].set_index("session")
    return out.reindex([s for s in SESSIONS if s in out.index])


def most_active_hours(hourly: pd.DataFrame, top: int = 4) -> pd.Series:
    """Average absolute hourly move (%) by UTC hour, top `top` hours —
    where the volatility (and the opportunity) actually is."""
    df = _to_utc(hourly)
    move = (df["Close"] / df["Open"] - 1).abs() * 100
    by_hour = move.groupby(df.index.hour).mean().round(3)
    by_hour.index.name = "utc_hour"
    return by_hour.sort_values(ascending=False).head(top)
