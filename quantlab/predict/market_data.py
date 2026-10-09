"""Live price and multi-timeframe bars, with the feed's known defects fixed.

Two defects, both real and both measured rather than assumed:

1. **The Twelve Data clock.** XAU/USD and the majors come back stamped in the
   feed's own local time, running roughly ten hours ahead of UTC. The offset is
   recovered by rounding (last bar - now) to whole hours, which is robust to
   the feed drifting by minutes and to daylight-saving jumps.

2. **The synthetic weekend.** Twelve Data returns a gapless 24/7 FX series and
   fills the Friday 21:00 -> Sunday 21:00 closure with invented bars. They are
   not flat, so `High == Low` will not catch them; they carry about a third of
   weekday amplitude. Left in, they manufacture a session range for Monday
   morning and flatten the hour-of-day volatility profile. `drop_closed`
   removes them by calendar, never by price.

Everything is cached with a per-source TTL and every Twelve Data call is
counted against a persisted daily budget, because the free tier allows 800
calls a day and a live loop will happily eat that before lunch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import has_key
from quantlab.data import twelvedata, yahoo
from quantlab.predict import assets as asset_spec
from quantlab.predict import config
from quantlab.predict.quality import SourceStatus, age_status

REPO = Path(__file__).resolve().parents[2]
BUDGET_PATH = REPO / "data" / "predict" / "td_budget.json"

#: FX and metals: continuous from Sunday 21:00 UTC to Friday 21:00 UTC.
WEEK_OPEN_HOUR = 21   # Sunday
WEEK_CLOSE_HOUR = 21  # Friday

INTERVALS = {"m15": "15m", "h1": "1h", "d1": "1d"}
_TTL = {"price": 45, "m15": 240, "h1": 900, "d1": 3600}

_cache: dict[tuple, tuple[float, object]] = {}


# --------------------------------------------------------------------------
# call budget
# --------------------------------------------------------------------------

class CallBudget:
    """Persisted per-UTC-day counter for a rate-limited API.

    Persisted rather than in-memory because restarting the live loop must not
    hand it a fresh 800 calls it does not have.
    """

    def __init__(self, path: Path = BUDGET_PATH, limit: int | None = None):
        self.path = path
        self.limit = limit if limit is not None else config.TWELVEDATA_DAILY_BUDGET
        self._day, self._used = self._read()

    def _read(self) -> tuple[str, int]:
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("day") == today:
                return today, int(data.get("used", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return today, 0

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"day": self._day, "used": self._used}),
                                 encoding="utf-8")
        except OSError:
            pass  # a budget file we cannot write is not worth failing a run over

    @property
    def used(self) -> int:
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        if today != self._day:
            self._day, self._used = today, 0
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def can_spend(self, n: int = 1) -> bool:
        return self.remaining >= n

    def spend(self, n: int = 1) -> None:
        self._used = self.used + n
        self._write()


BUDGET = CallBudget()


# --------------------------------------------------------------------------
# calendar helpers
# --------------------------------------------------------------------------

def market_open(ts: pd.Timestamp) -> bool:
    """True when spot FX / metals are trading (Sun 21:00 -> Fri 21:00 UTC)."""
    dow, hour = ts.dayofweek, ts.hour + ts.minute / 60
    if dow == 5:                                   # Saturday
        return False
    if dow == 6:                                   # Sunday
        return hour >= WEEK_OPEN_HOUR
    if dow == 4:                                   # Friday
        return hour < WEEK_CLOSE_HOUR
    return True


def next_open(ts: pd.Timestamp) -> pd.Timestamp:
    """The next moment the market is open at or after `ts`."""
    if market_open(ts):
        return ts
    probe = ts.floor("h")
    for _ in range(24 * 4):
        probe += pd.Timedelta(hours=1)
        if market_open(probe):
            # Step back to the exact boundary where it opened.
            boundary = probe.normalize() + pd.Timedelta(hours=WEEK_OPEN_HOUR)
            return boundary if boundary <= probe and market_open(boundary) else probe
    return ts


def last_close(ts: pd.Timestamp) -> pd.Timestamp:
    """The most recent Friday 21:00 UTC weekly close at or before `ts`."""
    probe = ts.normalize() + pd.Timedelta(hours=WEEK_CLOSE_HOUR)
    while probe > ts or probe.dayofweek != 4:
        probe -= pd.Timedelta(days=1)
        probe = probe.normalize() + pd.Timedelta(hours=WEEK_CLOSE_HOUR)
    return probe


def drop_closed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bars inside the weekend closure. Index must already be UTC."""
    if df is None or df.empty:
        return df
    i = df.index
    closed = (((i.dayofweek == 4) & (i.hour >= WEEK_CLOSE_HOUR))
              | (i.dayofweek == 5)
              | ((i.dayofweek == 6) & (i.hour < WEEK_OPEN_HOUR)))
    return df[~closed]


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def correct_feed_clock(df: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Shift a Twelve Data frame from the feed's local clock onto UTC.

    Returns the corrected frame and the offset in hours (0.0 when the feed was
    already aligned), so the caller can report the correction rather than hide
    it.
    """
    if df is None or df.empty:
        return df, 0.0
    offset = round((df.index[-1] - now) / pd.Timedelta(hours=1))
    if offset == 0:
        return df, 0.0
    out = df.copy()
    out.index = out.index - pd.Timedelta(hours=offset)
    return out, float(offset)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _cached(key: tuple, ttl: float, fn):
    hit = _cache.get(key)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def clear_cache() -> None:
    _cache.clear()


def _yahoo_bars(symbol: str, interval: str, days: int, tries: int = 3) -> pd.DataFrame:
    """Yahoo bars, retried.

    Yahoo intermittently answers a perfectly good request with "possibly
    delisted; no price data found" under throttling. A live loop has to ride
    through that rather than drop an asset for the next hour — the same retry
    the Gold Watch and FX Watch jobs already use.
    """
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    last: Exception | None = None
    for attempt in range(tries):
        try:
            df = to_utc(yahoo.get_ohlcv(symbol, start=start, interval=interval))
            keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
            out = df[keep].dropna(subset=["Close"])
            if not out.empty:
                return out
            last = ValueError(f"Yahoo returned an empty frame for {symbol} {interval}")
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last = exc
        if attempt < tries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise last if last else ValueError(f"no data for {symbol}")


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """Everything price-related the engine needs for one asset, one run."""
    code: str
    now: pd.Timestamp
    price: float = float("nan")
    price_source: str = ""
    price_as_of: pd.Timestamp | None = None
    price_age_s: float | None = None
    price_status: str = "unavailable"
    is_open: bool = True
    prev_close: float = float("nan")     # previous completed daily close
    change: float = float("nan")
    change_pct: float = float("nan")
    basis: float = 0.0                   # futures minus spot (gold only)
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    bar_sources: dict[str, str] = field(default_factory=dict)
    volume_available: bool = False
    feed_offset_hours: float = 0.0
    statuses: list[SourceStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return np.isfinite(self.price) and "d1" in self.bars and len(self.bars["d1"]) > 60

    def m15(self) -> pd.DataFrame | None:
        return self.bars.get("m15")

    def h1(self) -> pd.DataFrame | None:
        return self.bars.get("h1")

    def d1(self) -> pd.DataFrame | None:
        return self.bars.get("d1")


def get_snapshot(code: str, now: pd.Timestamp | None = None,
                 use_twelvedata: bool = True) -> MarketSnapshot:
    """Live price plus 15m / 1h / daily bars for one asset.

    Never raises for a source failure: each leg records a SourceStatus and the
    snapshot reports what it managed to get. A snapshot with `ok == False` must
    not be turned into a prediction.
    """
    a = asset_spec.get(code)
    now = now or pd.Timestamp.now(tz="UTC")
    snap = MarketSnapshot(code=a.code, now=now, is_open=market_open(now))

    # ---- daily bars (the backbone: ATR, EWMA vol, kurtosis, trend) ----------
    try:
        d1 = _cached((a.yahoo_daily, "d1"), _TTL["d1"],
                     lambda: _yahoo_bars(a.yahoo_daily, "1d", 1100))
        snap.bars["d1"] = d1
        snap.bar_sources["d1"] = f"Yahoo {a.yahoo_daily} daily"
        snap.statuses.append(SourceStatus(
            "Daily bars", "live", f"Yahoo {a.yahoo_daily}, {len(d1)} bars",
            as_of=d1.index[-1]))
    except Exception as exc:
        snap.statuses.append(SourceStatus(
            "Daily bars", "unavailable", f"Yahoo {a.yahoo_daily}",
            error=f"{type(exc).__name__}: {exc}"[:120]))

    # ---- hourly bars (hour-of-day variance profile, session structure) ------
    try:
        h1 = _cached((a.yahoo_intraday, "h1"), _TTL["h1"],
                     lambda: _yahoo_bars(a.yahoo_intraday, "1h", 700))
        h1 = drop_closed(h1)
        snap.bars["h1"] = h1
        snap.bar_sources["h1"] = f"Yahoo {a.yahoo_intraday} 1h"
        snap.statuses.append(SourceStatus(
            "Hourly bars", "live", f"Yahoo {a.yahoo_intraday}, {len(h1)} bars",
            as_of=h1.index[-1]))
    except Exception as exc:
        snap.statuses.append(SourceStatus(
            "Hourly bars", "unavailable", f"Yahoo {a.yahoo_intraday}",
            error=f"{type(exc).__name__}: {exc}"[:120]))

    # ---- 15m bars: Twelve Data spot preferred, Yahoo as the fallback -------
    m15, m15_src = None, ""
    if use_twelvedata and has_key("TWELVE_DATA_API_KEY") and BUDGET.can_spend(1):
        try:
            def _td_bars():
                BUDGET.spend(1)
                raw = twelvedata.get_time_series(a.twelvedata, interval="15min",
                                                 outputsize=450)
                return raw.rename(columns=str.title)
            raw = _cached((a.twelvedata, "td15"), _TTL["m15"], _td_bars)
            raw = to_utc(raw)
            raw, offset = correct_feed_clock(raw, now)
            snap.feed_offset_hours = offset
            m15 = drop_closed(raw)
            m15_src = f"Twelve Data {a.twelvedata} 15m (spot)"
        except Exception as exc:
            snap.statuses.append(SourceStatus(
                "Twelve Data 15m", "unavailable", a.twelvedata,
                error=f"{type(exc).__name__}: {exc}"[:120]))

    if m15 is None or m15.empty:
        try:
            fallback = _cached((a.yahoo_intraday, "m15"), _TTL["m15"],
                               lambda: _yahoo_bars(a.yahoo_intraday, "15m", 55))
            m15 = drop_closed(fallback)
            m15_src = f"Yahoo {a.yahoo_intraday} 15m (fallback)"
        except Exception as exc:
            snap.statuses.append(SourceStatus(
                "15m bars", "unavailable", "Twelve Data and Yahoo both failed",
                error=f"{type(exc).__name__}: {exc}"[:120]))

    if m15 is not None and not m15.empty:
        snap.bars["m15"] = m15
        snap.bar_sources["m15"] = m15_src
        age = (now - m15.index[-1]).total_seconds()
        snap.statuses.append(SourceStatus(
            "15m bars", "live" if snap.is_open and age < 1800 else "delayed",
            f"{m15_src}, {len(m15)} bars", as_of=m15.index[-1], age_seconds=age))

    # ---- live price --------------------------------------------------------
    # How far behind the *bar* stream is, which is the only observable measure
    # of feed health available: the /price endpoint returns no timestamp.
    bar_lag_s = None
    if m15 is not None and not m15.empty:
        bar_lag_s = max(0.0, (now - m15.index[-1]).total_seconds())

    price, source, as_of, from_quote = float("nan"), "", None, False
    if use_twelvedata and has_key("TWELVE_DATA_API_KEY") and BUDGET.can_spend(1):
        try:
            def _td_price():
                BUDGET.spend(1)
                return float(twelvedata.get_price(a.twelvedata))
            price = _cached((a.twelvedata, "price"), _TTL["price"], _td_price)
            source = f"Twelve Data {a.twelvedata} spot quote"
            as_of, from_quote = now, True
        except Exception as exc:
            snap.statuses.append(SourceStatus(
                "Twelve Data price", "unavailable", a.twelvedata,
                error=f"{type(exc).__name__}: {exc}"[:120]))

    if not np.isfinite(price):
        for key in ("m15", "h1", "d1"):
            frame = snap.bars.get(key)
            if frame is not None and not frame.empty:
                price = float(frame["Close"].iloc[-1])
                source = f"{snap.bar_sources[key]} last close"
                as_of = frame.index[-1]
                break

    snap.price, snap.price_source, snap.price_as_of = price, source, as_of
    if as_of is not None:
        snap.price_age_s = max(0.0, (now - as_of).total_seconds())

    if not np.isfinite(price):
        snap.price_status = "unavailable"
    elif not snap.is_open:
        snap.price_status = "closed"
    elif from_quote:
        # A real-time quote is only as trustworthy as the feed behind it. Rate
        # it by how far the bar stream has fallen behind rather than by the
        # bar's own age, which would mark a fresh tick "stale" simply because
        # the current 15-minute candle opened fourteen minutes ago.
        snap.price_status = ("live" if (bar_lag_s or 0) <= 2 * 900
                             else "delayed" if (bar_lag_s or 0) <= 4 * 900
                             else "stale")
    else:
        snap.price_status = age_status(snap.price_age_s, config.PRICE_LIVE_SECONDS,
                                       config.PRICE_STALE_SECONDS)

    detail = source or "no source"
    if from_quote:
        detail += (" (endpoint carries no timestamp; rated by bar-feed lag "
                   f"{(bar_lag_s or 0) / 60:.0f} min)")
    if snap.feed_offset_hours:
        detail += f" · feed clock corrected {snap.feed_offset_hours:+.0f}h"
    if not snap.is_open:
        detail += " · market closed (weekend)"
    snap.statuses.append(SourceStatus(
        "Market price", snap.price_status, detail,
        as_of=as_of, age_seconds=snap.price_age_s))

    # ---- previous close, change, basis, volume -----------------------------
    d1 = snap.bars.get("d1")
    if d1 is not None and len(d1) >= 2:
        last_bar_is_today = d1.index[-1].date() >= now.date()
        done = d1.iloc[-2] if last_bar_is_today else d1.iloc[-1]
        raw_prev_close = float(done["Close"])
        if a.uses_basis and np.isfinite(price):
            # GC=F trades at a premium to spot; shift the futures history onto
            # the spot the prediction is actually about.
            snap.basis = float(d1["Close"].iloc[-1]) - price
        snap.prev_close = raw_prev_close - snap.basis
        if np.isfinite(price):
            snap.change = price - snap.prev_close
            snap.change_pct = snap.change / snap.prev_close * 100

        vol = d1.get("Volume")
        snap.volume_available = bool(
            a.has_volume and vol is not None and float(vol.tail(20).sum()) > 0)

    return snap


def spot_frame(snap: MarketSnapshot, key: str) -> pd.DataFrame | None:
    """Bars in spot terms — the basis removed for a futures-backed asset."""
    frame = snap.bars.get(key)
    if frame is None or frame.empty:
        return None
    if not snap.basis:
        return frame
    out = frame.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] - snap.basis
    return out
