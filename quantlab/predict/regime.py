"""Market-regime and trading-session classification.

The regime is decided *before* the probabilities, because it changes how they
are computed. Trend indicators in a dead range are noise; fading a level in a
strong trend is how accounts die. `config.REGIME_TILT` encodes that as weight
multipliers, and this module decides which set applies.

Two independent readings are produced and then combined:

- **trend** — direction and strength, from moving-average alignment plus ADX.
  Alignment says which way, ADX says whether "which way" is a real question.
- **volatility state** — fast realised vol against its own ATR baseline, with
  an imminent high-impact release able to force the high-volatility state on
  its own. An FOMC statement does not care what the last twenty candles did.

Sessions follow the same UTC convention as the rest of the project (Asia
00-07, London 07-13, New York 13-21) with the 13:00-16:00 London/New York
overlap called out, because that window carries the most volatility of the day
for all three of these assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.predict import market_data, technical

SESSIONS: dict[str, tuple[int, int]] = {
    "Asia": (0, 7),
    "London": (7, 13),
    "New York": (13, 21),
}
OVERLAP = (13, 16)

TREND_REGIMES = ("strong_bullish", "weak_bullish", "range",
                 "weak_bearish", "strong_bearish")

REGIME_LABEL = {
    "strong_bullish": "Strong bullish trend",
    "weak_bullish": "Weak bullish trend",
    "range": "Range",
    "weak_bearish": "Weak bearish trend",
    "strong_bearish": "Strong bearish trend",
    "high_volatility": "High volatility",
    "low_volatility": "Low-volatility market",
}
#: Used instead of the generic label when a scheduled release, rather than
#: realised volatility, is what forced the high-volatility state.
EVENT_VOL_LABEL = "High-volatility event regime"

ADX_TRENDING = 25.0
ADX_WEAK = 18.0
VOL_HIGH = 1.35
VOL_LOW = 0.75


@dataclass
class SessionState:
    name: str                    # Asia | London | New York | Closed
    is_open: bool
    is_overlap: bool
    start: pd.Timestamp
    end: pd.Timestamp
    next_name: str
    next_start: pd.Timestamp
    next_end: pd.Timestamp

    @property
    def label(self) -> str:
        if not self.is_open:
            return "Weekend / closed"
        if self.is_overlap:
            return "London / New York overlap"
        return f"{self.name} session"


@dataclass
class RegimeState:
    trend: str
    vol_state: str               # high_volatility | low_volatility | normal
    label: str
    adx: float
    slope_atr: float
    vol_ratio: float
    session: SessionState
    tilt_keys: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.trend in ("strong_bullish", "strong_bearish")

    @property
    def direction(self) -> int:
        if "bullish" in self.trend:
            return 1
        if "bearish" in self.trend:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {
            "trend": self.trend, "vol_state": self.vol_state, "label": self.label,
            "adx": None if not np.isfinite(self.adx) else round(self.adx, 1),
            "slope_atr": None if not np.isfinite(self.slope_atr) else round(self.slope_atr, 3),
            "vol_ratio": None if not np.isfinite(self.vol_ratio) else round(self.vol_ratio, 3),
            "session": self.session.label,
            "session_is_open": self.session.is_open,
            "next_session": self.session.next_name,
            "next_session_start": str(self.session.next_start),
            "next_session_end": str(self.session.next_end),
            "reasons": self.reasons,
        }


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def _session_of(ts: pd.Timestamp) -> str | None:
    hour = ts.hour
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return name
    return None


def _session_bounds(ts: pd.Timestamp, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_h, end_h = SESSIONS[name]
    day = ts.normalize()
    return day + pd.Timedelta(hours=start_h), day + pd.Timedelta(hours=end_h)


def session_state(now: pd.Timestamp) -> SessionState:
    """Which session we are in, and when the next one runs.

    The "next session" is the next distinct block that trades, skipping the
    weekend. From Friday's New York close that is Monday's Asia session, not
    Saturday.
    """
    is_open = market_data.market_open(now)
    current = _session_of(now) if is_open else None

    probe = now
    for _ in range(24 * 5):
        probe = (probe + pd.Timedelta(hours=1)).floor("h")
        nxt = _session_of(probe)
        if nxt is None or not market_data.market_open(probe):
            continue
        if current is not None and nxt == current and probe.date() == now.date():
            continue
        next_start, next_end = _session_bounds(probe, nxt)
        if next_start < now:
            next_start = probe
        break
    else:  # pragma: no cover - only reachable if the calendar logic breaks
        nxt = "Asia"
        next_start, next_end = _session_bounds(now + pd.Timedelta(days=1), "Asia")

    if current is not None:
        start, end = _session_bounds(now, current)
    else:
        start, end = now, next_start

    return SessionState(
        name=current or "Closed", is_open=is_open,
        is_overlap=bool(is_open and OVERLAP[0] <= now.hour < OVERLAP[1]),
        start=start, end=end,
        next_name=nxt, next_start=next_start, next_end=next_end)


# --------------------------------------------------------------------------
# regime
# --------------------------------------------------------------------------

def classify(ts: technical.TechnicalState, now: pd.Timestamp,
             event_level: str = "none") -> RegimeState:
    """Trend regime plus volatility state, with the reasons kept."""
    session = session_state(now)
    reasons: list[str] = []

    adx = ts.adx14 if np.isfinite(ts.adx14) else float("nan")
    slope_atr = float("nan")
    if np.isfinite(ts.ema20) and np.isfinite(ts.ema50) and np.isfinite(ts.atr14) and ts.atr14 > 0:
        slope_atr = (ts.ema20 - ts.ema50) / ts.atr14

    up = ts.price > ts.ema20 > ts.ema50 if np.isfinite(ts.ema50) else False
    down = ts.price < ts.ema20 < ts.ema50 if np.isfinite(ts.ema50) else False
    separated = np.isfinite(slope_atr) and abs(slope_atr) >= 0.35
    trending = np.isfinite(adx) and adx >= ADX_TRENDING
    weakly = np.isfinite(adx) and adx >= ADX_WEAK

    if up and trending and separated:
        trend = "strong_bullish"
    elif down and trending and separated:
        trend = "strong_bearish"
    elif up and weakly:
        trend = "weak_bullish"
    elif down and weakly:
        trend = "weak_bearish"
    elif up:
        trend = "weak_bullish"
    elif down:
        trend = "weak_bearish"
    else:
        trend = "range"

    if np.isfinite(adx):
        reasons.append(f"ADX(14) {adx:.0f} "
                       f"({'trending' if trending else 'weak' if weakly else 'no trend'})")
    if np.isfinite(slope_atr):
        reasons.append(f"EMA20-EMA50 gap {slope_atr:+.2f} ATR")
    reasons.append(f"price {'above' if up else 'below' if down else 'tangled with'} "
                   f"the EMA stack")

    vol_state, event_driven = "normal", False
    if event_level == "HIGH":
        vol_state, event_driven = "high_volatility", True
        reasons.append("high-impact release scheduled inside the day — event regime")
    elif np.isfinite(ts.vol_ratio) and ts.vol_ratio >= VOL_HIGH:
        vol_state = "high_volatility"
        reasons.append(f"realised vol {ts.vol_ratio:.2f}x its ATR baseline")
    elif np.isfinite(ts.vol_ratio) and ts.vol_ratio <= VOL_LOW:
        vol_state = "low_volatility"
        reasons.append(f"realised vol only {ts.vol_ratio:.2f}x its ATR baseline")

    tilt_keys = [trend] + ([vol_state] if vol_state != "normal" else [])
    label = REGIME_LABEL[trend]
    if vol_state != "normal":
        suffix = EVENT_VOL_LABEL if event_driven else REGIME_LABEL[vol_state]
        label += f" · {suffix.lower()}"

    return RegimeState(trend=trend, vol_state=vol_state, label=label, adx=adx,
                       slope_atr=slope_atr, vol_ratio=ts.vol_ratio,
                       session=session, tilt_keys=tilt_keys, reasons=reasons)


# --------------------------------------------------------------------------
# historical analogue
# --------------------------------------------------------------------------

def _regime_series(frame: pd.DataFrame) -> pd.Series:
    """Point-in-time regime label for every bar of a frame.

    Uses the same rules as `classify`, computed on the frame's own timeframe,
    so "similar conditions" means the same thing looking back as it does now.
    """
    close = frame["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    try:
        adx_s = technical.adx(frame)
    except Exception:
        adx_s = pd.Series(np.nan, index=frame.index)
    atr_s = frame["High"].rolling(14).max() - frame["Low"].rolling(14).min()
    atr_s = atr_s.replace(0, np.nan)

    gap = (ema20 - ema50).abs() / atr_s
    up = (close > ema20) & (ema20 > ema50)
    down = (close < ema20) & (ema20 < ema50)
    strong = (adx_s >= ADX_TRENDING) & (gap >= 0.12)
    weak = adx_s >= ADX_WEAK

    out = pd.Series("range", index=frame.index, dtype=object)
    out[up & weak] = "weak_bullish"
    out[down & weak] = "weak_bearish"
    out[up & strong] = "strong_bullish"
    out[down & strong] = "strong_bearish"
    out[up & ~weak & ~strong] = "weak_bullish"
    out[down & ~weak & ~strong] = "weak_bearish"
    return out


@dataclass
class Analogue:
    up_rate: float
    base_rate: float
    n: int
    mean_move: float
    horizon_bars: int
    description: str
    available: bool = True


def analogue(frame: pd.DataFrame, regime: str, horizon_bars: int,
             session_hours: tuple[int, int] | None = None,
             min_samples: int = 40) -> Analogue:
    """How this asset behaved next, historically, in the same regime.

    Optionally also conditioned on the time of day, so "similar conditions"
    can include "at this point in the session". Returns `available=False`
    rather than a number when the sample is too thin to mean anything — a
    conditional up-rate off twelve observations is a coin flip with a decimal
    point.
    """
    empty = Analogue(0.5, 0.5, 0, 0.0, horizon_bars, "insufficient history", False)
    if frame is None or len(frame) < 300 or horizon_bars < 1:
        return empty

    close = frame["Close"]
    forward = close.shift(-horizon_bars) / close - 1
    labels = _regime_series(frame)

    mask = (labels == regime) & forward.notna()
    if session_hours is not None:
        hours = frame.index.hour
        start, end = session_hours
        mask &= (hours >= start) & (hours < end)

    sample = forward[mask]
    base = forward.dropna()
    if len(sample) < min_samples:
        return Analogue(0.5, float((base > 0).mean()) if len(base) else 0.5,
                        int(len(sample)), 0.0, horizon_bars,
                        f"only {len(sample)} historical matches — too few to use",
                        False)

    up_rate = float((sample > 0).mean())
    base_rate = float((base > 0).mean())
    desc = (f"in {regime.replace('_', ' ')} conditions"
            + (f" during {start:02d}-{end:02d} UTC" if session_hours else "")
            + f", price was higher {horizon_bars} bar(s) later "
              f"{up_rate:.0%} of the time (n={len(sample)}, base rate {base_rate:.0%})")
    return Analogue(up_rate, base_rate, int(len(sample)), float(sample.mean()),
                    horizon_bars, desc, True)
