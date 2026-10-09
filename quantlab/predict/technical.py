"""Technical analysis: indicators, volatility, support/resistance, structure.

This module answers "what does the chart say", and splits that answer into the
four evidence channels the engine weights separately:

    technical      the indicator stack — RSI, moving averages, MACD, ADX
    price_action   where price actually is and what it just did
    volatility     the vol regime, fast realised vol against its slow baseline
    structure      the levels around price, and which side has more room

It also produces the raw material the probability model needs downstream: the
EWMA daily sigma, the ATR-implied baseline sigma, the Student-t tail index from
realised kurtosis, and the hour-of-day variance profile that makes "the next 15
minutes" mean something different at the New York open than at 03:00 UTC.

Every indicator is computed from the bars in the snapshot. Nothing is imputed:
where an input is missing (FX has no tick volume) the reading is simply absent
and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.analysis.levels import atr as wilder_atr
from quantlab.analysis.levels import pivot_points
from quantlab.predict import assets as asset_spec
from quantlab.predict import config, market_data
from quantlab.predict.component import ComponentScore, blend, clamp, missing, squash

#: Parkinson's constant: for a driftless diffusion the RMS log high-low range
#: is 1.665 sigma, which is how an ATR becomes a comparable daily sigma.
PARKINSON = 1.6651


# --------------------------------------------------------------------------
# indicator primitives
# --------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def adx(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's ADX — trend *strength* without direction. Above ~25 is a trend,
    below ~20 is a range, and that distinction changes which signals matter."""
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=ohlc.index).ewm(
        alpha=1 / window, min_periods=window, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=ohlc.index).ewm(
        alpha=1 / window, min_periods=window, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def ewma_sigma(close: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics daily volatility — the fast, reactive vol estimate."""
    ret = np.log(close / close.shift(1)).dropna()
    if len(ret) < 20:
        return float("nan")
    return float(np.sqrt(ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1]))


def tail_index(close: pd.Series) -> float:
    """Student-t degrees of freedom from realised excess kurtosis.

    Excess kurtosis of a t is 6/(nu-4), so nu = 4 + 6/k. Near-normal samples
    (k <= 0.2) get nu = NU_MAX, which is a Gaussian in all but name.
    """
    ret = np.log(close / close.shift(1)).dropna()
    if len(ret) < 100:
        return config.NU_MAX
    k = float(ret.kurtosis())
    if k <= 0.2:
        return config.NU_MAX
    return float(np.clip(4 + 6 / k, config.NU_MIN, config.NU_MAX))


def hour_profile(hourly: pd.DataFrame) -> pd.Series:
    """Share of a day's variance falling in each UTC hour (sums to 1).

    Gold and the majors are 24-hour markets with a heartbeat: the London open
    and the New York overlap carry several times the variance of the Asian
    lull. Treating every hour as equal would make a 15-minute forecast at 03:00
    look exactly as risky as one at 13:30, which is plainly false.
    """
    if hourly is None or len(hourly) < 200:
        return pd.Series(1 / 24, index=range(24))
    h = hourly.copy()
    r2 = np.log(h["Close"] / h["Close"].shift(1)) ** 2
    prof = r2.groupby(h.index.hour).mean()
    prof = prof.reindex(range(24)).fillna(prof.median())
    total = prof.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1 / 24, index=range(24))
    return prof / total


def variance_fraction(profile: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                      respect_closure: bool = True) -> float:
    """Fraction of one day's variance contained in [start, end).

    Walks the window in 15-minute steps and accumulates the hourly profile,
    skipping any step when the market is shut. That is what makes a horizon
    spanning the weekend correctly carry almost no variance, instead of
    pretending Saturday trades.
    """
    if end <= start:
        return 0.0
    total = 0.0
    step = pd.Timedelta(minutes=15)
    t = start
    guard = 0
    while t < end and guard < 40_000:
        if not respect_closure or market_data.market_open(t):
            total += float(profile.get(t.hour, 1 / 24)) / 4.0
        t += step
        guard += 1
    return total


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

@dataclass
class TechnicalState:
    code: str
    price: float
    prev_close: float = float("nan")
    change: float = float("nan")
    change_pct: float = float("nan")

    rsi14_d: float = float("nan")
    rsi14_h1: float = float("nan")
    rsi14_m15: float = float("nan")
    sma20: float = float("nan")
    sma50: float = float("nan")
    sma200: float = float("nan")
    ema20: float = float("nan")
    ema50: float = float("nan")
    macd_hist: float = float("nan")
    macd_hist_prev: float = float("nan")
    adx14: float = float("nan")

    atr14: float = float("nan")
    atr_pct: float = float("nan")
    sigma_ewma: float = float("nan")     # fast daily sigma (log return)
    sigma_atr: float = float("nan")      # slow daily sigma implied by ATR
    vol_ratio: float = float("nan")      # fast / slow
    vol_of_vol: float = float("nan")
    nu: float = config.NU_MAX
    profile: pd.Series = field(default_factory=lambda: pd.Series(1 / 24, index=range(24)))

    mom_1d: float = float("nan")
    mom_5d: float = float("nan")
    mom_20d: float = float("nan")
    thrust_2h: float = float("nan")

    day_open: float = float("nan")
    day_high: float = float("nan")
    day_low: float = float("nan")
    pos_in_day_range: float = float("nan")
    prev_high: float = float("nan")
    prev_low: float = float("nan")
    swing_high_20d: float = float("nan")
    swing_low_20d: float = float("nan")
    higher_highs: bool = False
    lower_lows: bool = False

    levels: dict[str, float] = field(default_factory=dict)
    nearest_support: tuple[str, float] | None = None
    nearest_resistance: tuple[str, float] | None = None
    room_up: float = float("nan")
    room_down: float = float("nan")

    volume_z: float | None = None
    volume_note: str = ""
    bars_note: str = ""

    @property
    def trend_label(self) -> str:
        if not np.isfinite(self.ema20) or not np.isfinite(self.ema50):
            return "unknown"
        if self.price > self.ema20 > self.ema50:
            return "bullish"
        if self.price < self.ema20 < self.ema50:
            return "bearish"
        return "mixed"

    @property
    def vol_label(self) -> str:
        if not np.isfinite(self.vol_ratio):
            return "unknown"
        if self.vol_ratio >= 1.35:
            return "high"
        if self.vol_ratio <= 0.75:
            return "low"
        return "medium"


def compute(snap: market_data.MarketSnapshot) -> TechnicalState:
    """Every indicator the engine uses, from one market snapshot."""
    a = asset_spec.get(snap.code)
    d1 = market_data.spot_frame(snap, "d1")
    h1 = market_data.spot_frame(snap, "h1")
    m15 = snap.bars.get("m15")          # already spot when it came from TD
    price = snap.price

    ts = TechnicalState(code=snap.code, price=price, prev_close=snap.prev_close,
                        change=snap.change, change_pct=snap.change_pct)

    if d1 is None or len(d1) < 60:
        ts.bars_note = "insufficient daily history"
        return ts

    close = d1["Close"]
    ts.rsi14_d = float(rsi(close).iloc[-1])
    ts.sma20 = float(close.rolling(20).mean().iloc[-1])
    ts.sma50 = float(close.rolling(50).mean().iloc[-1])
    ts.sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    ts.ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ts.ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    _, _, hist = macd(close)
    ts.macd_hist = float(hist.iloc[-1])
    ts.macd_hist_prev = float(hist.iloc[-2]) if len(hist) > 2 else float("nan")
    try:
        ts.adx14 = float(adx(d1).iloc[-1])
    except Exception:
        ts.adx14 = float("nan")

    ts.atr14 = float(wilder_atr(d1).iloc[-1])
    ts.atr_pct = ts.atr14 / price if price else float("nan")
    ts.sigma_ewma = ewma_sigma(close)
    ts.sigma_atr = ts.atr_pct / PARKINSON if np.isfinite(ts.atr_pct) else float("nan")
    if np.isfinite(ts.sigma_ewma) and np.isfinite(ts.sigma_atr) and ts.sigma_atr > 0:
        ts.vol_ratio = ts.sigma_ewma / ts.sigma_atr
    ret = np.log(close / close.shift(1)).dropna()
    if len(ret) > 60:
        realised = ret.rolling(20).std()
        ts.vol_of_vol = float(realised.pct_change().tail(60).std())
    ts.nu = tail_index(close)
    ts.profile = hour_profile(h1)

    ts.mom_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 2 else float("nan")
    ts.mom_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 6 else float("nan")
    ts.mom_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else float("nan")

    # Previous completed day, and the levels every desk marks off it.
    last_is_today = d1.index[-1].date() >= snap.now.date()
    done = d1.iloc[-2] if (last_is_today and len(d1) > 2) else d1.iloc[-1]
    ts.prev_high, ts.prev_low = float(done["High"]), float(done["Low"])
    piv = pivot_points(ts.prev_high, ts.prev_low, float(done["Close"]))
    ts.swing_high_20d = float(d1["High"].iloc[-20:].max())
    ts.swing_low_20d = float(d1["Low"].iloc[-20:].min())

    highs, lows = d1["High"].iloc[-10:], d1["Low"].iloc[-10:]
    ts.higher_highs = bool(highs.iloc[-3:].max() >= highs.max() and
                           lows.iloc[-3:].min() > lows.iloc[:5].min())
    ts.lower_lows = bool(lows.iloc[-3:].min() <= lows.min() and
                         highs.iloc[-3:].max() < highs.iloc[:5].max())

    # ---- intraday ----------------------------------------------------------
    session_start = _session_day_start(snap.now)
    if m15 is not None and len(m15) > 4:
        today = m15[m15.index >= session_start]
        if len(today) >= 2:
            ts.day_open = float(today["Open"].iloc[0])
            ts.day_high = float(today["High"].max())
            ts.day_low = float(today["Low"].min())
            span = ts.day_high - ts.day_low
            ts.pos_in_day_range = float((price - ts.day_low) / span) if span > 0 else 0.5
        n = min(8, len(m15) - 1)
        ts.thrust_2h = float(np.log(m15["Close"].iloc[-1] / m15["Close"].iloc[-1 - n]))
        ts.rsi14_m15 = float(rsi(m15["Close"]).iloc[-1])
    if h1 is not None and len(h1) > 30:
        ts.rsi14_h1 = float(rsi(h1["Close"]).iloc[-1])

    # ---- volume ------------------------------------------------------------
    if snap.volume_available and "Volume" in d1.columns:
        vol = d1["Volume"].replace(0, np.nan).dropna()
        if len(vol) > 30:
            mean, sd = vol.tail(60).mean(), vol.tail(60).std()
            if sd > 0:
                ts.volume_z = float((vol.iloc[-1] - mean) / sd)
                ts.volume_note = f"daily volume z {ts.volume_z:+.1f} vs 60d"
    else:
        ts.volume_note = ("no tick or contract volume behind spot FX bars"
                          if not a.has_volume else "volume series empty this run")

    # ---- levels ------------------------------------------------------------
    step = a.round_step
    base = np.floor(price / step) * step
    rounds = {f"round {asset_spec.fmt_price(a.code, x)}": float(x)
              for x in (base - step, base, base + step, base + 2 * step)}
    raw = {
        "R2": piv["R2"], "R1": piv["R1"], "pivot": piv["P"],
        "S1": piv["S1"], "S2": piv["S2"],
        "prev day high": ts.prev_high, "prev day low": ts.prev_low,
        "prev close": float(done["Close"]),
        "20d high": ts.swing_high_20d, "20d low": ts.swing_low_20d,
        "session high": ts.day_high, "session low": ts.day_low,
        "SMA20": ts.sma20, "SMA50": ts.sma50,
        **rounds,
    }
    ts.levels = {k: float(v) for k, v in raw.items()
                 if v is not None and np.isfinite(v)}

    above = sorted(((k, v) for k, v in ts.levels.items() if v > price), key=lambda kv: kv[1])
    below = sorted(((k, v) for k, v in ts.levels.items() if v < price),
                   key=lambda kv: -kv[1])
    ts.nearest_resistance = above[0] if above else None
    ts.nearest_support = below[0] if below else None
    if ts.nearest_resistance:
        ts.room_up = ts.nearest_resistance[1] - price
    if ts.nearest_support:
        ts.room_down = price - ts.nearest_support[1]

    return ts


def _level_text(code: str, name: str, price: float) -> str:
    """"R1 4,350.00", but just "round 4,350.00" — never the number twice.

    Tests for the formatted price rather than for any digit: "R1", "S2" and
    "20d high" all contain digits without containing their price.
    """
    text = asset_spec.fmt_price(code, price)
    return name if text in name else f"{name} {text}"


def _session_day_start(now: pd.Timestamp) -> pd.Timestamp:
    """Start of the current trading day (21:00 UTC rollover, as FX quotes it)."""
    start = now.normalize() - pd.Timedelta(hours=3)   # previous 21:00 UTC
    if now.hour >= 21:
        start = now.normalize() + pd.Timedelta(hours=21)
    return start


# --------------------------------------------------------------------------
# component scores
# --------------------------------------------------------------------------

def score_technical(ts: TechnicalState) -> ComponentScore:
    """The indicator stack: moving-average alignment, RSI, MACD, ADX-weighted."""
    if not np.isfinite(ts.rsi14_d):
        return missing("technical", "not enough daily history for indicators")

    parts: list[tuple[float, float]] = []
    ev: list[str] = []

    # Moving-average alignment across three timeframes of the same idea.
    stack = 0.0
    if np.isfinite(ts.sma200):
        stack += 1 if ts.price > ts.sma200 else -1
    stack += 1 if ts.price > ts.sma50 else -1
    stack += 1 if ts.price > ts.sma20 else -1
    stack += 1 if ts.ema20 > ts.ema50 else -1
    denom = 4 if np.isfinite(ts.sma200) else 3
    ma_score = stack / denom
    parts.append((ma_score, 1.4))
    ev.append(f"price {'above' if ts.price > ts.sma50 else 'below'} SMA50, "
              f"EMA20 {'>' if ts.ema20 > ts.ema50 else '<'} EMA50 "
              f"({ma_score:+.2f} MA alignment)")

    # RSI, blended across daily / hourly / 15m so one timeframe cannot dominate.
    rsi_parts = [(v, w) for v, w in
                 ((ts.rsi14_d, 1.0), (ts.rsi14_h1, 0.7), (ts.rsi14_m15, 0.4))
                 if np.isfinite(v)]
    if rsi_parts:
        rsi_blend = sum(v * w for v, w in rsi_parts) / sum(w for _, w in rsi_parts)
        # 50 is neutral; 20 points of RSI is a full unit of evidence.
        rsi_score = clamp((rsi_blend - 50) / 20)
        parts.append((rsi_score, 1.0))
        ev.append(f"RSI(14) {ts.rsi14_d:.0f} daily"
                  + (f" / {ts.rsi14_h1:.0f} 1h" if np.isfinite(ts.rsi14_h1) else "")
                  + (f" / {ts.rsi14_m15:.0f} 15m" if np.isfinite(ts.rsi14_m15) else ""))

    # MACD histogram: sign is the state, change is the news.
    if np.isfinite(ts.macd_hist) and np.isfinite(ts.atr14) and ts.atr14 > 0:
        mh = squash(ts.macd_hist / (0.30 * ts.atr14), 1.0)
        parts.append((mh, 0.8))
        turning = ""
        if np.isfinite(ts.macd_hist_prev):
            widening = abs(ts.macd_hist) > abs(ts.macd_hist_prev)
            turning = " and widening" if widening else " but narrowing"
        ev.append(f"MACD histogram {ts.macd_hist:+.2f}{turning}")

    score = blend(parts)

    # ADX scales conviction rather than direction: a clean signal in a
    # trendless tape is still a trendless tape.
    if np.isfinite(ts.adx14):
        strength = float(np.clip(ts.adx14 / 30.0, 0.45, 1.25))
        score = clamp(score * strength)
        ev.append(f"ADX(14) {ts.adx14:.0f} — "
                  f"{'trending' if ts.adx14 >= 25 else 'no strong trend'}")

    return ComponentScore("technical", score, evidence=ev, detail={
        "rsi_d": ts.rsi14_d, "rsi_h1": ts.rsi14_h1, "rsi_m15": ts.rsi14_m15,
        "sma20": ts.sma20, "sma50": ts.sma50, "sma200": ts.sma200,
        "ema20": ts.ema20, "ema50": ts.ema50,
        "macd_hist": ts.macd_hist, "adx": ts.adx14, "ma_alignment": ma_score,
    })


def score_price_action(ts: TechnicalState) -> ComponentScore:
    """Where price sits and what it just did — momentum, range position, thrust."""
    parts: list[tuple[float, float]] = []
    ev: list[str] = []

    if np.isfinite(ts.pos_in_day_range):
        pos = clamp((ts.pos_in_day_range - 0.5) * 2)
        parts.append((pos, 1.2))
        ev.append(f"holding {ts.pos_in_day_range:.0%} of today's "
                  f"{asset_spec.fmt_distance(ts.code, ts.day_high - ts.day_low, signed=False)} range")

    if np.isfinite(ts.thrust_2h) and np.isfinite(ts.sigma_ewma) and ts.sigma_ewma > 0:
        # Scale the last two hours by the vol the model expects over two hours.
        expected = ts.sigma_ewma * np.sqrt(2 / 24)
        parts.append((squash(ts.thrust_2h / expected, 1.2), 1.3))
        ev.append(f"last 2h move {ts.thrust_2h * 100:+.2f}% "
                  f"({ts.thrust_2h / expected:+.1f}x the 2h expectation)")

    for value, span, weight in ((ts.mom_1d, "1d", 0.8), (ts.mom_5d, "5d", 1.0),
                                (ts.mom_20d, "20d", 0.7)):
        if np.isfinite(value) and np.isfinite(ts.sigma_ewma) and ts.sigma_ewma > 0:
            days = {"1d": 1, "5d": 5, "20d": 20}[span]
            parts.append((squash(value / (ts.sigma_ewma * np.sqrt(days)), 1.5), weight))
    if np.isfinite(ts.mom_5d):
        ev.append(f"momentum {ts.mom_1d * 100:+.2f}% 1d · {ts.mom_5d * 100:+.2f}% 5d"
                  + (f" · {ts.mom_20d * 100:+.2f}% 20d" if np.isfinite(ts.mom_20d) else ""))

    if ts.higher_highs:
        parts.append((0.5, 0.6))
        ev.append("sequence of higher highs and higher lows over 10 sessions")
    elif ts.lower_lows:
        parts.append((-0.5, 0.6))
        ev.append("sequence of lower lows and lower highs over 10 sessions")

    if ts.volume_z is not None and np.isfinite(ts.mom_1d):
        # Volume confirms whatever the last day did; it has no sign of its own.
        conf = clamp(np.sign(ts.mom_1d) * np.clip(ts.volume_z / 2.0, -1, 1))
        parts.append((conf, 0.5))
        ev.append(f"{ts.volume_note} — "
                  f"{'confirming' if conf * np.sign(ts.mom_1d) > 0 else 'not confirming'} the move")

    if not parts:
        return missing("price_action", "no intraday bars to read price action from")

    return ComponentScore("price_action", blend(parts), evidence=ev, detail={
        "pos_in_day_range": ts.pos_in_day_range, "thrust_2h": ts.thrust_2h,
        "mom_1d": ts.mom_1d, "mom_5d": ts.mom_5d, "mom_20d": ts.mom_20d,
        "day_high": ts.day_high, "day_low": ts.day_low, "day_open": ts.day_open,
        "volume_z": ts.volume_z,
    })


def score_volatility(ts: TechnicalState) -> ComponentScore:
    """The vol regime.

    Volatility is mostly a *width* input, not a direction one — it reaches the
    probabilities through the predictive sigma and the RANGE band, which is
    where it belongs. The only direction it carries here is continuation:
    an expansion of realised vol tends to run with the prevailing move rather
    than against it, and that effect is deliberately small.
    """
    if not np.isfinite(ts.vol_ratio):
        return missing("volatility", "not enough history for a volatility estimate")

    ev = [f"EWMA vol {ts.sigma_ewma * 100:.2f}%/day vs ATR baseline "
          f"{ts.sigma_atr * 100:.2f}%/day (ratio {ts.vol_ratio:.2f}, {ts.vol_label})",
          f"ATR(14) {asset_spec.fmt_distance(ts.code, ts.atr14, signed=False)}"]

    expansion = clamp((ts.vol_ratio - 1.0) / 0.5)
    direction = 0.0
    if np.isfinite(ts.mom_1d):
        direction = clamp(np.sign(ts.mom_1d) * abs(expansion) * 0.5)
        if abs(expansion) > 0.2:
            ev.append(f"vol {'expanding' if expansion > 0 else 'contracting'} — "
                      f"{'supports continuation of' if expansion > 0 else 'takes energy out of'} "
                      f"the {'up' if ts.mom_1d > 0 else 'down'} move")

    return ComponentScore("volatility", direction, evidence=ev, detail={
        "sigma_ewma": ts.sigma_ewma, "sigma_atr": ts.sigma_atr,
        "vol_ratio": ts.vol_ratio, "atr14": ts.atr14, "vol_label": ts.vol_label,
        "nu": ts.nu, "vol_of_vol": ts.vol_of_vol,
    })


def score_structure(ts: TechnicalState) -> ComponentScore:
    """Support and resistance: which side has room, and is price at an edge.

    Read as a *reversion* signal — price pinned under resistance with open air
    below is bearish over the next few bars, whatever the trend thinks. The
    regime tilt in config.py is what stops this fighting a strong trend, by
    cutting the weight of this channel when ADX says the tape is trending.
    """
    if not ts.levels or not (np.isfinite(ts.atr14) and ts.atr14 > 0):
        return missing("structure", "no level structure available")

    parts: list[tuple[float, float]] = []
    ev: list[str] = []

    if np.isfinite(ts.room_up) and np.isfinite(ts.room_down):
        # Asymmetric room: more space above than below is bullish.
        total = ts.room_up + ts.room_down
        if total > 0:
            asym = clamp((ts.room_down - ts.room_up) / total)
            parts.append((asym, 1.0))
        res_name, res_px = ts.nearest_resistance
        sup_name, sup_px = ts.nearest_support
        ev.append(f"nearest resistance {_level_text(ts.code, res_name, res_px)} "
                  f"({asset_spec.fmt_distance(ts.code, ts.room_up)}), support "
                  f"{_level_text(ts.code, sup_name, sup_px)} "
                  f"({asset_spec.fmt_distance(ts.code, -ts.room_down)})")

        # Pinned against a level within a quarter ATR is the reversion case.
        pin = 0.25 * ts.atr14
        if ts.room_up < pin:
            parts.append((-0.6, 1.2))
            ev.append(f"price is pinned under {res_name} — inside 0.25 ATR of it")
        elif ts.room_down < pin:
            parts.append((0.6, 1.2))
            ev.append(f"price is sitting on {sup_name} — inside 0.25 ATR of it")

    if np.isfinite(ts.day_high) and np.isfinite(ts.day_low):
        if ts.price > ts.prev_high:
            parts.append((0.45, 0.8))
            ev.append(f"trading above yesterday's high "
                      f"{asset_spec.fmt_price(ts.code, ts.prev_high)} (breakout)")
        elif ts.price < ts.prev_low:
            parts.append((-0.45, 0.8))
            ev.append(f"trading below yesterday's low "
                      f"{asset_spec.fmt_price(ts.code, ts.prev_low)} (breakdown)")

    if not parts:
        return missing("structure", "price is not near any tracked level")

    return ComponentScore("structure", blend(parts), evidence=ev, detail={
        "nearest_resistance": list(ts.nearest_resistance) if ts.nearest_resistance else None,
        "nearest_support": list(ts.nearest_support) if ts.nearest_support else None,
        "room_up": ts.room_up, "room_down": ts.room_down,
        "levels": ts.levels,
    })
