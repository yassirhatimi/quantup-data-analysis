"""Macro drivers and the economic calendar, both from FRED.

Two jobs:

1. **Where the macro backdrop is**, from the drivers listed per asset in
   `assets.py`. Each series is transformed (level / year-on-year / change /
   spread), its three-month move is z-scored against its own history, and the
   asset's stated sign turns that into a push. Market-channel drivers (yields,
   the dollar, VIX) are repriced continuously and carry live information;
   release-channel drivers (CPI, payrolls) are already in the price the moment
   they print, so they set the regime rather than time it — and are weighted
   accordingly.

2. **What is scheduled next**, from FRED's release-dates API, so the calendar
   is real rather than a guess. Two honest caveats live in the code:

   - FRED publishes release *dates*, not times. The clock time attached here is
     the standard published release time for that report (08:30 America/New_York
     for CPI/PPI/payrolls, 10:00 for UMich), converted properly through the
     America/New_York timezone. It is labelled as the standard schedule.
   - **FOMC is deliberately absent.** FRED's release 101 returns an entry for
     every single calendar day, which carries no schedule information at all.
     Rather than invent a date, the engine reports FOMC as untracked and reads
     any date you pin into `events.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.config import get_key, has_key
from quantlab.data import fred
from quantlab.data.base import get_json
from quantlab.predict import assets as asset_spec
from quantlab.predict.component import ComponentScore, clamp, missing, squash

NY = ZoneInfo("America/New_York")
PINNED_PATH = Path(__file__).resolve().parent / "events.json"

_SERIES_TTL = 3600.0
_CAL_TTL = 3600.0
_cache: dict[str, tuple[float, object]] = {}


# --------------------------------------------------------------------------
# release calendar definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Release:
    release_id: int
    name: str
    impact: str            # HIGH | medium
    et_time: dtime
    affects: tuple[str, ...]


RELEASES: tuple[Release, ...] = (
    Release(10, "US CPI", "HIGH", dtime(8, 30), ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(50, "US Employment Situation (NFP)", "HIGH", dtime(8, 30),
            ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(54, "US Personal Income & Outlays (core PCE)", "HIGH", dtime(8, 30),
            ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(46, "US PPI", "medium", dtime(8, 30), ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(53, "US GDP", "medium", dtime(8, 30), ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(9, "US retail sales", "medium", dtime(8, 30), ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(192, "US JOLTS job openings", "medium", dtime(10, 0),
            ("XAUUSD", "EURUSD", "GBPUSD")),
    Release(91, "UMich consumer sentiment", "medium", dtime(10, 0),
            ("XAUUSD", "EURUSD", "GBPUSD")),
)

#: Reported to the user every run so an absent FOMC is never mistaken for
#: "no FOMC scheduled".
UNTRACKED_EVENTS = {
    "FOMC rate decision": "FRED release 101 returns every calendar day, so it "
                          "carries no schedule — pin the date in events.json",
    "ECB rate decision": "not carried as a dated FRED release",
    "Bank of England rate decision": "not carried as a dated FRED release",
    "ISM PMIs": "proprietary to ISM; FRED discontinued the NAPM series",
}

CALENDAR_DAYS = 35


@dataclass
class Event:
    when: pd.Timestamp
    name: str
    impact: str
    source: str
    affects: tuple[str, ...]

    def in_hours(self, now: pd.Timestamp) -> float:
        return (self.when - now).total_seconds() / 3600

    def countdown(self, now: pd.Timestamp) -> str:
        delta = self.when - now
        secs = delta.total_seconds()
        if secs < 0:
            return "in progress / just released"
        days, rem = divmod(int(secs), 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        if days:
            return f"in {days}d {hours}h"
        if hours:
            return f"in {hours}h {minutes:02d}m"
        return f"in {minutes}m"

    def to_dict(self, now: pd.Timestamp | None = None) -> dict:
        out = {"when": str(self.when), "name": self.name, "impact": self.impact,
               "source": self.source}
        if now is not None:
            out["countdown"] = self.countdown(now)
            out["hours_away"] = round(self.in_hours(now), 2)
        return out


# --------------------------------------------------------------------------
# series access
# --------------------------------------------------------------------------

def _cached(key: str, ttl: float, fn):
    hit = _cache.get(key)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def clear_cache() -> None:
    _cache.clear()


def read_series(series_id: str, start: str = "2005-01-01") -> pd.Series:
    return _cached(f"series:{series_id}", _SERIES_TTL,
                   lambda: fred.get_series(series_id, start=start).dropna())


@dataclass
class Reading:
    driver: asset_spec.MacroDriver
    value: float
    as_of: pd.Timestamp
    change: float
    z: float
    stale: bool
    age_days: int

    @property
    def push(self) -> float:
        return self.driver.sign * self.z


def _spacing_days(index: pd.DatetimeIndex) -> float:
    """Observation spacing inferred from the index itself.

    FRED frequencies are irregular — holidays break `infer_freq` — so the
    median calendar gap over the recent tail is the robust answer.
    """
    if len(index) < 4:
        return 30.0
    gaps = np.diff(index[-40:].values).astype("timedelta64[D]").astype(float)
    return float(max(np.median(gaps), 1.0))


def _transform(driver: asset_spec.MacroDriver) -> pd.Series:
    s = read_series(driver.series)
    if driver.transform == "spread":
        other = read_series(driver.against)
        idx = s.index.union(other.index)
        s = (s.reindex(idx).ffill() - other.reindex(idx).ffill()).dropna()
        return s
    if driver.transform == "yoy":
        gap = _spacing_days(s.index)
        n = max(1, round(365 / gap))
        return (s / s.shift(n) - 1).dropna() * 100
    if driver.transform == "chg":
        return s.diff().dropna()
    return s


def read_driver(driver: asset_spec.MacroDriver, now: pd.Timestamp) -> Reading | None:
    """One driver's latest reading, its three-month move and that move's z-score."""
    series = _transform(driver)
    if series.empty:
        return None

    as_of = pd.Timestamp(series.index[-1])
    if as_of.tz is None:
        as_of = as_of.tz_localize("UTC")
    age_days = int((now - as_of).days)
    if age_days > driver.max_age_days:
        return Reading(driver, float(series.iloc[-1]), as_of, float("nan"),
                       float("nan"), True, age_days)

    gap = _spacing_days(series.index)
    n = max(1, round(90 / gap))
    changes = series.diff(n).dropna()
    if len(changes) < 8:
        return Reading(driver, float(series.iloc[-1]), as_of, float("nan"),
                       float("nan"), False, age_days)
    sd = float(changes.tail(400).std())
    change = float(changes.iloc[-1])
    z = change / sd if sd > 0 else 0.0
    return Reading(driver, float(series.iloc[-1]), as_of, change,
                   float(np.clip(z, -4, 4)), False, age_days)


def _score_driver_set(code: str, drivers, name: str, market_weight: float = 2.0
                      ) -> ComponentScore:
    readings: list[Reading] = []
    failures: list[str] = []
    stale: list[str] = []

    if not has_key("FRED_API_KEY"):
        return missing(name, "FRED_API_KEY not set")

    now = pd.Timestamp.now(tz="UTC")
    for driver in drivers:
        try:
            r = read_driver(driver, now)
        except Exception as exc:
            failures.append(f"{driver.series} ({type(exc).__name__})")
            continue
        if r is None:
            failures.append(f"{driver.series} (empty)")
        elif r.stale:
            stale.append(f"{driver.label} last updated {r.as_of:%b %Y}")
        elif np.isfinite(r.z):
            readings.append(r)

    if not readings:
        note = "no FRED driver returned a usable reading"
        if failures:
            note += "; failed: " + ", ".join(failures[:4])
        if stale:
            note += "; too stale: " + "; ".join(stale[:3])
        return missing(name, note)

    weights = np.array([market_weight if r.driver.channel == "market" else 1.0
                        for r in readings])
    pushes = np.array([r.push for r in readings])
    raw = float(np.sum(weights * pushes) / np.sum(weights))
    score = squash(raw, 1.0)

    ranked = sorted(readings, key=lambda r: -abs(r.push))
    ev = [f"{r.driver.label} {r.value:,.2f} "
          f"(3m change {r.change:+,.2f}, z {r.z:+.1f}) → "
          f"{'bullish' if r.push > 0 else 'bearish'} {abs(r.push):.2f}"
          for r in ranked[:4]]
    note = ""
    if stale:
        note = "too stale to use: " + "; ".join(stale[:3])
    if failures:
        note = (note + "; " if note else "") + "unavailable: " + ", ".join(failures[:4])

    return ComponentScore(
        name, score, evidence=ev, note=note,
        status="live" if not (stale or failures) else "delayed",
        detail={
            "n_drivers": len(readings), "n_expected": len(tuple(drivers)),
            "raw_push": raw,
            "readings": [{"series": r.driver.series, "label": r.driver.label,
                          "value": r.value, "as_of": str(r.as_of.date()),
                          "change_3m": r.change, "z": r.z, "sign": r.driver.sign,
                          "channel": r.driver.channel, "push": r.push,
                          "why": r.driver.why}
                         for r in ranked],
        })


def score_macro(code: str) -> ComponentScore:
    """Growth, inflation, the dollar and risk appetite."""
    return _score_driver_set(code, asset_spec.get(code).macro_drivers, "macro")


def score_rates(code: str) -> ComponentScore:
    """The rate complex: yields, real yields, the curve and the policy gap.

    Kept separate from `macro` because interest-rate expectations deserve their
    own weight — for a major, the rate differential *is* the exchange rate's
    fundamental, and for gold the real yield is its entire opportunity cost.
    """
    return _score_driver_set(code, asset_spec.get(code).rate_drivers, "rates",
                             market_weight=1.5)


# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------

def _release_dates(release_id: int, now: pd.Timestamp, days: int) -> list[pd.Timestamp]:
    data = get_json(
        "https://api.stlouisfed.org/fred/release/dates",
        params={
            "api_key": get_key("FRED_API_KEY"), "file_type": "json",
            "release_id": release_id,
            "realtime_start": now.strftime("%Y-%m-%d"),
            "realtime_end": (now + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            # Required: a release that has not published yet has no data, so
            # without this flag a *future* date is invisible. The cost is that
            # releases with no attached series come back one-per-day, which
            # `_plausible` throws out.
            "include_release_dates_with_no_data": "true",
            "sort_order": "asc",
        }, timeout=45)
    return [pd.Timestamp(r["date"]) for r in data.get("release_dates", [])]


def _plausible(dates: list[pd.Timestamp], days: int) -> bool:
    """A real report fires a handful of times a month, not on half the days."""
    return 0 < len(dates) <= max(4, days // 4)


def _pinned_events(now: pd.Timestamp) -> list[Event]:
    """Manually pinned dates from events.json — FOMC lives here."""
    if not PINNED_PATH.exists():
        return []
    try:
        data = json.loads(PINNED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[Event] = []
    for row in data.get("events", []):
        try:
            when = pd.Timestamp(row["when_utc"], tz="UTC")
        except (KeyError, ValueError):
            continue
        if when < now - pd.Timedelta(hours=6):
            continue
        out.append(Event(when, row.get("name", "pinned event"),
                         row.get("impact", "HIGH"), "pinned in events.json",
                         tuple(row.get("affects", asset_spec.DEFAULT_CODES))))
    return out


def calendar(now: pd.Timestamp | None = None, days: int = CALENDAR_DAYS
             ) -> tuple[list[Event], list[str]]:
    """Scheduled releases over the next `days`, plus anything that failed."""
    now = now or pd.Timestamp.now(tz="UTC")

    def fetch():
        events: list[Event] = []
        problems: list[str] = []
        if not has_key("FRED_API_KEY"):
            return [], ["FRED_API_KEY not set — no economic calendar"]
        for rel in RELEASES:
            try:
                dates = _release_dates(rel.release_id, now, days)
            except Exception as exc:
                problems.append(f"{rel.name}: {type(exc).__name__}")
                continue
            if not _plausible(dates, days):
                if dates:
                    problems.append(
                        f"{rel.name}: FRED returned {len(dates)} dates in {days} "
                        f"days — not a real schedule, discarded")
                continue
            for d in dates:
                # FRED gives the date; the clock time is the report's standard
                # publication time, resolved through the New York timezone.
                local = pd.Timestamp.combine(d.date(), rel.et_time).tz_localize(NY)
                events.append(Event(local.tz_convert("UTC"), rel.name, rel.impact,
                                    "FRED release calendar (standard release time)",
                                    rel.affects))
        return events, problems

    events, problems = _cached(f"calendar:{days}", _CAL_TTL, fetch)
    events = sorted(list(events) + _pinned_events(now), key=lambda e: e.when)
    return [e for e in events if e.when >= now - pd.Timedelta(hours=2)], list(problems)


def events_for(code: str, events: list[Event]) -> list[Event]:
    return [e for e in events if code in e.affects]


def event_risk(events: list[Event], now: pd.Timestamp, until: pd.Timestamp) -> dict:
    """Highest-impact scheduled event inside a horizon window."""
    inside = [e for e in events if now <= e.when <= until]
    high = [e for e in inside if e.impact == "HIGH"]
    level = "HIGH" if high else ("medium" if inside else "none")
    return {"level": level, "events": inside,
            "names": [e.name for e in inside],
            "nearest": inside[0] if inside else None}
