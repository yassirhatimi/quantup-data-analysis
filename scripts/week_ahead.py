"""Week Ahead — a five-session forward analysis for XAUUSD, EURUSD and GBPUSD.

Run it on the weekend. One run produces, per asset:

  1. the week that just closed (OHLC, range vs ATR, where Friday settled),
  2. a fat-tailed **weekly** distribution for the coming Mon-Fri, built from a
     variance-ratio-corrected daily vol rather than naive sqrt(5) scaling,
  3. Monte-Carlo path statistics — P(touch) for every level, expected travel,
     the drawdown a long has to sit through even in weeks that finish up,
  4. an empirical analogue study: what the *next* five sessions did, historically,
     from states that resemble today's,
  5. a calibration check on the cone itself, so the intervals are earned,
  6. the cross-asset correlation, because these three are one dollar trade.

Why the model looks like this. Naive sqrt(5) scaling of a daily sigma is the
usual way to get a weekly number and it is usually wrong: daily returns are not
independent. The variance ratio VR(5) = Var(5d) / (5 x Var(1d)) measures the
violation directly — below 1 the asset mean-reverts inside the week and sqrt(5)
overstates the cone; above 1 it trends and sqrt(5) understates it. We measure
VR on overlapping five-day windows and scale by sqrt(VR).

The cone is **driftless** — median = spot — matching the rest of this project.
A week-ahead directional call from technicals is worth a few basis points of
edge and would be swamped by the width of the distribution; pretending
otherwise would make the intervals dishonest. Directional evidence is reported
separately as a tilt, and the analogue study is the only thing allowed to argue
for a drift, because it is measured rather than asserted.

Usage:
    python scripts/week_ahead.py                 # all three, report + charts
    python scripts/week_ahead.py XAUUSD          # one asset
    python scripts/week_ahead.py --no-chart      # skip the PNGs
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantlab.analysis.levels import atr, pivot_points
from quantlab.predict import assets as asset_spec
from quantlab.predict import macro_data, market_data

OUT_DIR = REPO / "dashboard" / "week_ahead"
HISTORY_DIR = OUT_DIR / "history"

SESSIONS = 5              # Mon-Fri
QS = [0.05, 0.25, 0.50, 0.75, 0.95]
N_PATHS = 40_000
RNG = np.random.default_rng(20260823)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def fmt(code: str, v: float) -> str:
    """Price formatted the way the asset is quoted."""
    if not np.isfinite(v):
        return "n/a"
    return f"{v:,.2f}" if code == "XAUUSD" else f"{v:.5f}"


def unit(code: str) -> str:
    return "$/oz" if code == "XAUUSD" else "pips"


def n_word() -> str:
    """SESSIONS spelled out. The prose quotes the horizon in several places and
    a midweek rebuild overrides SESSIONS, so nothing may hardcode "five"."""
    return {1: "one", 2: "two", 3: "three", 4: "four",
            5: "five"}.get(SESSIONS, str(SESSIONS))


def to_unit(code: str, v: float) -> float:
    """Convert a price difference into the asset's quoting unit."""
    return v if code == "XAUUSD" else v * 10_000


def drop_weekend(d: pd.DataFrame) -> pd.DataFrame:
    """Yahoo stamps a thin Sunday bar on spot FX; it is closure padding, not a
    session, and it corrupts both the 'week just closed' OHLC and the vol
    estimate. Daily bars are Mon-Fri or they are not sessions."""
    return d[d.index.weekday < 5]


def week_bounds(now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The Monday..Friday of the week that is about to start."""
    d = now.normalize()
    monday = d + pd.Timedelta(days=(7 - d.weekday()) % 7)
    if monday == d and now.weekday() < 5:      # already inside a trading week
        monday = d - pd.Timedelta(days=d.weekday())
    return monday, monday + pd.Timedelta(days=4)


def student_t_scaled(nu: float, size, rng) -> np.ndarray:
    """Student-t standardised to unit variance (finite only for nu > 2)."""
    nu = max(float(nu), 2.5)
    return rng.standard_t(nu, size=size) / np.sqrt(nu / (nu - 2.0))


def t_quantile(p: float, nu: float) -> float:
    """Unit-variance Student-t quantile, bisection on the CDF (no scipy here)."""
    nu = max(float(nu), 2.5)
    scale = np.sqrt(nu / (nu - 2.0))
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / scale


def _t_cdf(x: float, nu: float) -> float:
    """Student-t CDF via the regularised incomplete beta, series-free."""
    xx = nu / (nu + x * x)
    ib = 0.5 * _betainc(0.5 * nu, 0.5, xx)
    return ib if x <= 0 else 1.0 - ib


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _lgamma(a + b) - _lgamma(a) - _lgamma(b)
    front = np.exp(np.log(x) * a + np.log1p(-x) * b + lbeta) / a
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def _lgamma(z: float) -> float:
    from math import lgamma
    return lgamma(z)


# --------------------------------------------------------------------------
# the weekly volatility model
# --------------------------------------------------------------------------

@dataclass
class VolModel:
    sigma_week: float          # fractional, one standard deviation over 5 sessions
    sigma_day_ewma: float      # fractional daily EWMA
    sigma_day_eff: float       # daily sigma implied by sigma_week / sqrt(5)
    vr5: float                 # variance ratio at 5 days
    nu: float                  # Student-t tail index
    atr14: float               # price units
    vol_ratio: float           # EWMA vol vs its own 1y median — the regime read
    note: str = ""


def build_vol(code: str, d1: pd.DataFrame) -> VolModel:
    close = d1["Close"].astype(float)
    r = np.log(close).diff().dropna()

    # Fast EWMA daily vol (lambda 0.94, the RiskMetrics default) — what the
    # market is doing now, not what it averaged over the year.
    sig_d = float(r.ewm(alpha=1 - 0.94).std().iloc[-1])

    # Variance ratio at 5 days on overlapping windows, measured over 3 years.
    r5 = np.log(close).diff(SESSIONS).dropna()
    win = r.tail(750)
    win5 = r5.tail(750)
    vr = float(win5.var() / (SESSIONS * win.var())) if win.var() > 0 else 1.0
    vr = float(np.clip(vr, 0.55, 1.60))       # keep a bad estimate from running away

    sigma_week = sig_d * np.sqrt(SESSIONS) * np.sqrt(vr)

    # Tail index from the kurtosis of *weekly* returns — the horizon we forecast.
    wk = np.log(close).diff(SESSIONS).dropna().tail(750)
    z = (wk - wk.mean()) / wk.std()
    kurt = float(((z ** 4).mean()))
    excess = max(kurt - 3.0, 0.05)
    nu = float(np.clip(6.0 / excess + 4.0, 3.5, 30.0))

    a14 = float(atr(d1, 14).iloc[-1])
    med = float(r.ewm(alpha=1 - 0.94).std().tail(252).median())
    ratio = sig_d / med if med > 0 else 1.0

    return VolModel(sigma_week=sigma_week, sigma_day_ewma=sig_d,
                    sigma_day_eff=sigma_week / np.sqrt(SESSIONS), vr5=vr,
                    nu=nu, atr14=a14, vol_ratio=ratio)


# --------------------------------------------------------------------------
# Monte-Carlo paths
# --------------------------------------------------------------------------

@dataclass
class Paths:
    close: np.ndarray          # terminal price per path
    high: np.ndarray           # running max per path
    low: np.ndarray            # running min per path
    daily: np.ndarray          # (n, 5) price path


def simulate(spot: float, vm: VolModel, n: int = N_PATHS) -> Paths:
    """Five daily steps of standardised-t innovations at the effective sigma.

    Innovations are iid at sigma_day_eff, which reproduces sigma_week exactly
    by construction; the serial dependence the variance ratio measured has
    already been folded into that number. Path statistics (touch, drawdown)
    therefore carry the right terminal width but a slightly conventional shape
    inside the week — stated here rather than hidden.
    """
    shocks = student_t_scaled(vm.nu, (n, SESSIONS), RNG) * vm.sigma_day_eff
    logpath = np.cumsum(shocks, axis=1)
    prices = spot * np.exp(logpath)
    return Paths(close=prices[:, -1], high=prices.max(axis=1),
                 low=prices.min(axis=1), daily=prices)


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def build_levels(code: str, d1: pd.DataFrame, spot: float,
                 basis: float, last_week: dict) -> list[dict]:
    """The grid a desk actually carries into a new week."""
    close = d1["Close"].astype(float)
    lv: dict[str, float] = {}

    piv = pivot_points(last_week["high"], last_week["low"], last_week["close"])
    for k in ("S2", "S1", "P", "R1", "R2"):
        lv[f"weekly {k}"] = piv[k] - basis

    lv["last week high"] = last_week["high"] - basis
    lv["last week low"] = last_week["low"] - basis
    lv["last week close"] = last_week["close"] - basis

    for w, tag in ((20, "20d"), (60, "60d")):
        lv[f"{tag} high"] = float(d1["High"].tail(w).max()) - basis
        lv[f"{tag} low"] = float(d1["Low"].tail(w).min()) - basis
    lv["52w high"] = float(d1["High"].tail(252).max()) - basis
    lv["52w low"] = float(d1["Low"].tail(252).min()) - basis

    for w in (20, 50, 200):
        lv[f"SMA{w}"] = float(close.rolling(w).mean().iloc[-1]) - basis

    # Round numbers either side, on the increment traders actually watch.
    step = 50.0 if code == "XAUUSD" else 0.0100
    base = np.floor(spot / step) * step
    for k in (-2, -1, 0, 1, 2, 3):
        p = base + k * step
        if p > 0:
            lv[f"round {fmt(code, p)}"] = p

    out = []
    for name, price in lv.items():
        if not np.isfinite(price) or price <= 0:
            continue
        out.append({"name": name, "price": float(price)})
    # Merge levels that sit on top of each other into one cluster line.
    tol = 0.0012 * spot
    out.sort(key=lambda r: -r["price"])
    merged: list[dict] = []
    for row in out:
        if merged and abs(merged[-1]["price"] - row["price"]) <= tol:
            merged[-1]["name"] += " + " + row["name"]
            merged[-1]["price"] = 0.5 * (merged[-1]["price"] + row["price"])
        else:
            merged.append(dict(row))
    return merged


# --------------------------------------------------------------------------
# empirical path statistics — travel and heat
# --------------------------------------------------------------------------

def path_stats(d1: pd.DataFrame, vm: VolModel, spot: float, code: str) -> dict:
    """Weekly travel and adverse excursion, measured rather than simulated.

    The Monte-Carlo walks daily *closes*, so its running max and min miss every
    intraday extreme: it reported a median heat of about one dollar on gold,
    which is not a number anyone can place a stop against. The honest source is
    history — real highs and lows over real five-session windows.

    Every historical window is divided by the weekly sigma prevailing at its
    start, which turns it into a pure shape statistic, and then multiplied by
    today's sigma. That is what makes a 2024 quiet-market week comparable to
    this one, and it is why these numbers scale with the current vol regime
    instead of averaging it away.
    """
    d = drop_weekend(d1)
    C = d["Close"].astype(float)
    r = np.log(C).diff()
    sig_d = r.ewm(alpha=1 - 0.94).std()
    sig_w = sig_d * np.sqrt(SESSIONS) * np.sqrt(vm.vr5)      # sigma at each date

    hi = d["High"].astype(float).rolling(SESSIONS).max().shift(-SESSIONS)
    lo = d["Low"].astype(float).rolling(SESSIONS).min().shift(-SESSIONS)
    fwd = C.shift(-SESSIONS)

    df = pd.DataFrame({"c": C, "hi": hi, "lo": lo, "fwd": fwd, "sw": sig_w}).dropna()
    df = df[df["sw"] > 0]
    if len(df) < 100:
        return {}

    rng_z = ((df["hi"] - df["lo"]) / df["c"]) / df["sw"]      # travel, in sigmas
    up = df["fwd"] > df["c"]
    mae_l = ((df["c"] - df["lo"]) / df["c"] / df["sw"])[up]   # heat on a long
    mae_s = ((df["hi"] - df["c"]) / df["c"] / df["sw"])[~up]  # heat on a short

    S = vm.sigma_week                                          # today's sigma
    def px(z):                                                 # sigmas -> units
        return to_unit(code, spot * z * S)

    return {
        "n": int(len(df)),
        "range_med": px(float(rng_z.median())),
        "range_p25": px(float(rng_z.quantile(0.25))),
        "range_p75": px(float(rng_z.quantile(0.75))),
        "mae_long_med": px(float(mae_l.median())),
        "mae_long_p75": px(float(mae_l.quantile(0.75))),
        "mae_long_p90": px(float(mae_l.quantile(0.90))),
        "mae_short_med": px(float(mae_s.median())),
        "mae_short_p75": px(float(mae_s.quantile(0.75))),
        "n_up": int(up.sum()), "n_dn": int((~up).sum()),
    }


# --------------------------------------------------------------------------
# empirical analogue study
# --------------------------------------------------------------------------

def analogues(d1: pd.DataFrame, vm: VolModel) -> dict:
    """What the next five sessions did, historically, from a state like today's.

    Matching is on three standardised state variables — 20-day momentum, RSI(14)
    and the vol ratio — by Euclidean distance in z-space, taking the closest 10%
    of history. Distance matching rather than hard bands keeps the sample from
    collapsing to nothing when the present is unusual, which it currently is.
    """
    close = d1["Close"].astype(float)
    mom20 = close.pct_change(20)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    r = np.log(close).diff()
    sig = r.ewm(alpha=1 - 0.94).std()
    ratio = sig / sig.rolling(252).median()

    fwd = close.shift(-SESSIONS) / close - 1.0

    feat = pd.DataFrame({"mom": mom20, "rsi": rsi, "vol": ratio}).dropna()
    feat = feat.join(fwd.rename("fwd"))
    hist = feat.iloc[:-SESSIONS].dropna()
    if len(hist) < 120:
        return {"n": 0}

    z = (hist[["mom", "rsi", "vol"]] - hist[["mom", "rsi", "vol"]].mean()) / \
        hist[["mom", "rsi", "vol"]].std()
    now = (feat[["mom", "rsi", "vol"]].iloc[-1] -
           hist[["mom", "rsi", "vol"]].mean()) / hist[["mom", "rsi", "vol"]].std()
    dist = np.sqrt(((z - now) ** 2).sum(axis=1))
    k = max(int(0.10 * len(hist)), 40)
    pick = dist.nsmallest(k).index
    sample = hist.loc[pick, "fwd"] * 100

    return {
        "n": int(len(sample)),
        "p_up": float((sample > 0).mean() * 100),
        "median": float(sample.median()),
        "mean": float(sample.mean()),
        "q05": float(sample.quantile(0.05)),
        "q25": float(sample.quantile(0.25)),
        "q75": float(sample.quantile(0.75)),
        "q95": float(sample.quantile(0.95)),
        "state": {"mom20": float(feat["mom"].iloc[-1] * 100),
                  "rsi": float(feat["rsi"].iloc[-1]),
                  "vol": float(feat["vol"].iloc[-1])},
        "unconditional_p_up": float((hist["fwd"] > 0).mean() * 100),
        "unconditional_median": float(hist["fwd"].median() * 100),
    }


# --------------------------------------------------------------------------
# calibration of the cone itself
# --------------------------------------------------------------------------

def calibrate(d1: pd.DataFrame, lookback: int = 500) -> dict:
    """Walk history, build the same weekly cone, count how often it contained.

    A 90% interval that contains 78% of outcomes is not a 90% interval, and the
    only way to know is to check. Rebuilt out-of-sample at every step: the vol
    at time t uses only data up to t.
    """
    close = d1["Close"].astype(float)
    r = np.log(close).diff()
    sig = r.ewm(alpha=1 - 0.94).std()

    r5 = np.log(close).diff(SESSIONS)
    hits50 = hits90 = n = 0
    zs: list[float] = []
    nu_last = 8.0
    for i in range(max(len(close) - lookback, 300), len(close) - SESSIONS):
        s = float(sig.iloc[i])
        if not np.isfinite(s) or s <= 0:
            continue
        win = r.iloc[max(0, i - 750):i + 1]
        win5 = r5.iloc[max(0, i - 750):i + 1].dropna()
        if len(win5) < 100 or win.var() <= 0:
            continue
        vr = float(np.clip(win5.var() / (SESSIONS * win.var()), 0.55, 1.60))
        sw = s * np.sqrt(SESSIONS) * np.sqrt(vr)
        wk = win5.tail(750)
        z = (wk - wk.mean()) / wk.std()
        nu = float(np.clip(6.0 / max(float((z ** 4).mean()) - 3.0, 0.05) + 4.0, 3.5, 30.0))
        nu_last = nu
        actual = float(np.log(close.iloc[i + SESSIONS] / close.iloc[i]))
        for lvl, q in ((50, 0.75), (90, 0.95)):
            band = t_quantile(q, nu) * sw
            if abs(actual) <= band:
                if lvl == 50:
                    hits50 += 1
                else:
                    hits90 += 1
        zs.append(actual / sw)
        n += 1
    if n == 0:
        return {"n": 0, "scale": 1.0}

    # How wide should sigma have been? Compare the realised 90th percentile of
    # |z| with what the t-distribution promised. A ratio above 1 means the cone
    # was too narrow and the intervals were overselling their confidence.
    az = np.abs(np.array(zs))
    theo = t_quantile(0.95, nu_last)
    scale = float(np.clip(np.quantile(az, 0.90) / theo, 0.80, 1.35)) if theo > 0 else 1.0
    return {"n": n, "cover50": 100 * hits50 / n, "cover90": 100 * hits90 / n,
            "scale": scale}


# --------------------------------------------------------------------------
# per-asset assembly
# --------------------------------------------------------------------------

@dataclass
class AssetWeek:
    code: str
    name: str
    spot: float
    basis: float
    last_week: dict
    vol: VolModel
    cone: dict
    levels: list[dict]
    paths_stats: dict
    analog: dict
    calib: dict
    trend: dict
    daily_week: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def analyse(code: str, now: pd.Timestamp) -> AssetWeek | None:
    a = asset_spec.get(code)
    snap = market_data.get_snapshot(code, now=now)
    if not snap.ok:
        print(f"  {code}: snapshot not usable, skipping")
        return None

    d1 = snap.d1().copy()
    if d1.index.tz is not None:
        d1.index = d1.index.tz_localize(None)
    d1 = drop_weekend(d1)
    spot = float(snap.price)
    basis = float(snap.basis or 0.0)

    # ---- the week that just closed -------------------------------------
    last_bar = d1.index[-1]
    mon = last_bar - pd.Timedelta(days=int(last_bar.weekday()))
    wk = d1.loc[d1.index >= mon]
    if len(wk) < 3:                      # thin week — fall back to last 5 bars
        wk = d1.tail(SESSIONS)
    prev_close = float(d1["Close"].loc[d1.index < wk.index[0]].iloc[-1])
    last_week = {
        "start": wk.index[0], "end": wk.index[-1],
        "open": float(wk["Open"].iloc[0]), "high": float(wk["High"].max()),
        "low": float(wk["Low"].min()), "close": float(wk["Close"].iloc[-1]),
        "prev_close": prev_close,
        "pct": 100 * (float(wk["Close"].iloc[-1]) / prev_close - 1),
    }
    rng = last_week["high"] - last_week["low"]
    last_week["range"] = rng
    last_week["close_in_range"] = (100 * (last_week["close"] - last_week["low"]) / rng
                                   if rng > 0 else 50.0)

    daily_week = [{"date": ix.strftime("%a %d %b"),
                   "pct": 100 * (float(row["Close"]) / float(
                       d1["Close"].iloc[d1.index.get_loc(ix) - 1]) - 1),
                   "close": float(row["Close"]) - basis}
                  for ix, row in wk.iterrows()]

    # ---- model ---------------------------------------------------------
    vm = build_vol(code, d1)
    last_week["range_atr"] = rng / vm.atr14 if vm.atr14 > 0 else float("nan")

    # Widen (or tighten) the cone to whatever history says it should have been.
    calib = calibrate(d1)
    scale = float(calib.get("scale", 1.0))
    vm.sigma_week *= scale
    vm.sigma_day_eff = vm.sigma_week / np.sqrt(SESSIONS)
    vm.note = (f"sigma scaled x{scale:.2f} by the calibration check"
               if abs(scale - 1.0) > 0.02 else "no calibration adjustment needed")

    paths = simulate(spot, vm)
    cone = {f"q{int(q * 100):02d}": float(np.quantile(paths.close, q)) for q in QS}
    cone["sigma_week_pct"] = vm.sigma_week * 100
    cone["p_up"] = float((paths.close > spot).mean() * 100)

    levels = build_levels(code, d1, spot, basis, last_week)
    for row in levels:
        p = row["price"]
        row["dist"] = to_unit(code, p - spot)
        row["p_close_above"] = float((paths.close > p).mean() * 100)
        row["p_touch"] = float((paths.high >= p).mean() * 100) if p > spot \
            else float((paths.low <= p).mean() * 100)

    stats = {
        "p_range_1atr": float((np.abs(paths.close - spot) <= vm.atr14).mean() * 100),
        "p_up_1pct": float((paths.close / spot - 1 > 0.01).mean() * 100),
        "p_dn_1pct": float((paths.close / spot - 1 < -0.01).mean() * 100),
        "p_up_2pct": float((paths.close / spot - 1 > 0.02).mean() * 100),
        "p_dn_2pct": float((paths.close / spot - 1 < -0.02).mean() * 100),
    }
    stats.update(path_stats(d1, vm, spot, code))

    close = d1["Close"].astype(float)
    ema20 = float(close.ewm(span=20).mean().iloc[-1]) - basis
    ema50 = float(close.ewm(span=50).mean().iloc[-1]) - basis
    sma200 = float(close.rolling(200).mean().iloc[-1]) - basis
    delta = close.diff()
    up_ = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn_ = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi_d = float((100 - 100 / (1 + up_ / dn_.replace(0, np.nan))).iloc[-1])
    trend = {
        "ema20": ema20, "ema50": ema50, "sma200": sma200, "rsi": rsi_d,
        "above20": spot > ema20, "above50": spot > ema50, "above200": spot > sma200,
        "mom5": 100 * (float(close.iloc[-1]) / float(close.iloc[-6]) - 1),
        "mom20": 100 * (float(close.iloc[-1]) / float(close.iloc[-21]) - 1),
        "mom60": 100 * (float(close.iloc[-1]) / float(close.iloc[-61]) - 1),
        "ext20": 100 * (spot / ema20 - 1),
        "regime": ("bullish" if spot > ema20 > ema50 else
                   "bearish" if spot < ema20 < ema50 else "mixed"),
    }

    notes = []
    if basis:
        notes.append(f"{a.yahoo_daily} futures levels shifted onto spot by a "
                     f"{basis:+.2f} basis.")
    if snap.feed_offset_hours:
        notes.append(f"Twelve Data bar timestamps corrected by "
                     f"{snap.feed_offset_hours:+.0f}h onto UTC.")

    return AssetWeek(code=code, name=a.name, spot=spot, basis=basis,
                     last_week=last_week, vol=vm, cone=cone, levels=levels,
                     paths_stats=stats, analog=analogues(d1, vm),
                     calib=calib, trend=trend, daily_week=daily_week,
                     notes=notes)


# --------------------------------------------------------------------------
# cross-asset
# --------------------------------------------------------------------------

def cross_asset(results: dict[str, AssetWeek], now: pd.Timestamp) -> dict:
    series = {}
    for code in results:
        snap = market_data.get_snapshot(code, now=now)
        d = snap.d1()["Close"].astype(float).copy()
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        series[code] = np.log(d).diff(SESSIONS)
    df = pd.DataFrame(series).dropna()
    corr_1y = df.tail(252).corr()
    corr_3m = df.tail(63).corr()
    return {"corr_1y": corr_1y.round(2).to_dict(),
            "corr_3m": corr_3m.round(2).to_dict(),
            "n_1y": int(len(df.tail(252)))}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render(results: dict[str, AssetWeek], cross: dict, events: list,
           now: pd.Timestamp, mon: pd.Timestamp, fri: pd.Timestamp) -> str:
    L: list[str] = []
    add = L.append

    add(f"# Week Ahead — {mon:%d %b} to {fri:%d %b %Y}")
    add("")
    add(f"*Built {now:%Y-%m-%d %H:%M} UTC — XAUUSD, EURUSD, GBPUSD. "
        + (f"Horizon: the {n_word()} sessions from the Sunday reopen to Friday's "
           "close.*" if SESSIONS == 5 else
           f"Horizon: the {n_word()} sessions left in the week, "
           f"{mon:%a %d %b} open to Friday's close — rebuilt midweek.*"))
    add("")

    # ---- calendar ------------------------------------------------------
    add("## The week's calendar")
    add("")
    wk_ev = [e for e in events if mon <= e.when.tz_localize(None) <= fri + pd.Timedelta(days=1)]
    if wk_ev:
        add("| when (UTC) | event | impact |")
        add("|---|---|---|")
        for e in wk_ev:
            add(f"| {e.when.tz_localize(None):%a %d %b %H:%M} | {e.name} | "
                f"**{e.impact}** |")
    else:
        add("No tracked releases land inside the week.")
    add("")
    nxt = [e for e in events if e.when.tz_localize(None) > fri + pd.Timedelta(days=1)]
    if nxt:
        add(f"First event after the week: {nxt[0].name} — "
            f"{nxt[0].when.tz_localize(None):%a %d %b %H:%M} UTC.")
        add("")
    add("Central-bank decisions (FOMC, ECB, BoE) are **not tracked** — FRED "
        "carries no usable forward schedule for them and this project refuses to "
        "guess dates. Pin them in `quantlab/predict/events.json` if any fall in "
        "the week.")
    add("")

    # ---- scoreboard ----------------------------------------------------
    add("## Scoreboard — where the three stand")
    add("")
    add("| asset | spot | last week | week range /ATR | closed in range | regime | RSI(14)d | 20d mom | vol vs normal |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in results.values():
        lw = r.last_week
        add(f"| **{r.code}** | {fmt(r.code, r.spot)} | {lw['pct']:+.2f}% | "
            f"{lw['range_atr']:.1f}x | {lw['close_in_range']:.0f}% | "
            f"{r.trend['regime']} | {r.trend['rsi']:.0f} | "
            f"{r.trend['mom20']:+.2f}% | {r.vol.vol_ratio:.2f}x |")
    add("")

    # ---- per asset -----------------------------------------------------
    for r in results.values():
        add("---")
        add("")
        add(f"## {r.code} — {r.name}")
        add("")
        lw = r.last_week
        add(f"**Spot {fmt(r.code, r.spot)}** | last week "
            f"{fmt(r.code, lw['open'] - r.basis)} → {fmt(r.code, lw['close'] - r.basis)} "
            f"({lw['pct']:+.2f}%), high {fmt(r.code, lw['high'] - r.basis)} / "
            f"low {fmt(r.code, lw['low'] - r.basis)} | "
            f"ATR(14) {fmt(r.code, r.vol.atr14)} | "
            f"weekly sigma {r.vol.sigma_week * 100:.2f}%")
        add("")

        add("### The week just closed" if SESSIONS == 5
            else f"### The last {n_word()} sessions")
        add("")
        add("| day | close | change |")
        add("|---|---|---|")
        for d in r.daily_week:
            add(f"| {d['date']} | {fmt(r.code, d['close'])} | {d['pct']:+.2f}% |")
        add("")
        # Midweek the last bar is not a Friday, so name it rather than assume.
        last_day = r.daily_week[-1]["date"].split()[0] if r.daily_week else "Friday"
        add(f"Range was **{lw['range_atr']:.1f}x** a normal daily ATR and "
            f"{'Friday' if SESSIONS == 5 else last_day} "
            f"settled **{lw['close_in_range']:.0f}%** of the way up it "
            f"({'strong close' if lw['close_in_range'] > 70 else 'weak close' if lw['close_in_range'] < 30 else 'mid-range close'}).")
        add("")

        add("### The coming week's distribution")
        add("")
        c = r.cone
        add(f"Driftless fat-tailed cone, Student-t nu={r.vol.nu:.1f}, "
            f"weekly sigma {c['sigma_week_pct']:.2f}% "
            f"(daily EWMA {r.vol.sigma_day_ewma * 100:.2f}% x sqrt{SESSIONS} x "
            f"sqrt(VR {r.vol.vr5:.2f})):")
        add("")
        add("| quantile | Friday close | vs spot |")
        add("|---|---|---|")
        for q, lab in ((5, "5% — bear tail"), (25, "25%"), (50, "median"),
                       (75, "75%"), (95, "95% — bull tail")):
            v = c[f"q{q:02d}"]
            add(f"| {lab} | {fmt(r.code, v)} | "
                f"{to_unit(r.code, v - r.spot):+,.1f} {unit(r.code)} "
                f"({100 * (v / r.spot - 1):+.2f}%) |")
        add("")
        s = r.paths_stats
        add(f"- P(|move| <= 1 ATR by Friday): **{s['p_range_1atr']:.0f}%** · "
            f"P(> +1%): {s['p_up_1pct']:.0f}% · P(< -1%): {s['p_dn_1pct']:.0f}% · "
            f"P(> +2%): {s['p_up_2pct']:.0f}% · P(< -2%): {s['p_dn_2pct']:.0f}%")
        if s.get("range_med"):
            add(f"- Expected travel high-to-low, from {s['n']} real {n_word()}-session "
                f"windows rescaled to today's vol: **{s['range_med']:,.0f} "
                f"{unit(r.code)}** median "
                f"({s['range_p25']:,.0f} – {s['range_p75']:,.0f} interquartile)")
        add("")

        if s.get("mae_long_med"):
            add("**Heat — what you sit through even when you are right.** "
                "Measured on real intraday highs and lows, not simulated closes:")
            add("")
            add("| | median | 75th pct | 90th pct |")
            add("|---|---|---|---|")
            add(f"| Long, in weeks that **finish up** ({s['n_up']} cases) | "
                f"{s['mae_long_med']:,.0f} | {s['mae_long_p75']:,.0f} | "
                f"{s['mae_long_p90']:,.0f} |")
            add(f"| Short, in weeks that **finish down** ({s['n_dn']} cases) | "
                f"{s['mae_short_med']:,.0f} | {s['mae_short_p75']:,.0f} | — |")
            add("")
            add(f"*In {unit(r.code)}. A stop inside the 75th-percentile column is "
                f"taken out in more than a quarter of the weeks that ultimately go "
                f"your way.*")
            add("")

        cal = r.calib
        if cal.get("n"):
            add(f"**Calibration** — rebuilding this same cone at every one of the "
                f"last {cal['n']} sessions and checking what actually happened "
                f"{n_word()} sessions later: the 50% band contained **{cal['cover50']:.0f}%** "
                f"(target 50) and the 90% band contained **{cal['cover90']:.0f}%** "
                f"(target 90). {r.vol.note.capitalize()}; the cone above is the "
                f"adjusted one.")
            add("")

        add("### Levels for the week")
        add("")
        add(f"| level | price | dist ({unit(r.code)}) | P(touch this week) | P(close above Fri) |")
        add("|---|---|---|---|---|")
        for row in r.levels:
            mark = " ←" if abs(row["price"] - r.spot) < 0.0015 * r.spot else ""
            add(f"| {row['name']}{mark} | {fmt(r.code, row['price'])} | "
                f"{row['dist']:+,.1f} | {row['p_touch']:.0f}% | "
                f"{row['p_close_above']:.0f}% |")
        add("")

        an = r.analog
        if an.get("n"):
            st = an["state"]
            add("### What history did from a state like this")
            add("")
            add(f"Today's state: 20-day momentum **{st['mom20']:+.1f}%**, "
                f"RSI(14) daily **{st['rsi']:.0f}**, EWMA vol "
                f"**{st['vol']:.2f}x** its 1-year normal. Taking the "
                f"**{an['n']}** closest historical matches on those three "
                f"(standardised distance, ~3 years of daily history) and reading "
                f"what the *next {n_word()} sessions* did:")
            add("")
            add("| | analogues | all history |")
            add("|---|---|---|")
            add(f"| P(up over the week) | **{an['p_up']:.0f}%** | "
                f"{an['unconditional_p_up']:.0f}% |")
            add(f"| median {SESSIONS}-day return | **{an['median']:+.2f}%** | "
                f"{an['unconditional_median']:+.2f}% |")
            add(f"| mean | {an['mean']:+.2f}% | — |")
            add(f"| 5th / 95th pct | {an['q05']:+.2f}% / {an['q95']:+.2f}% | — |")
            add(f"| 25th / 75th pct | {an['q25']:+.2f}% / {an['q75']:+.2f}% | — |")
            add("")
            edge = an["p_up"] - an["unconditional_p_up"]
            add(f"Edge vs the unconditional base rate: **{edge:+.0f} points**. "
                + (f"Analogue samples overlap heavily ({n_word()}-day windows sampled "
                   "daily), so the effective sample is far smaller than the count "
                   "suggests — treat this as a tilt, not a signal."))
            add("")

        t = r.trend
        add("### Trend and positioning context")
        add("")
        add(f"- Regime **{t['regime']}** — spot {'above' if t['above20'] else 'below'} "
            f"EMA20 ({fmt(r.code, t['ema20'])}), "
            f"{'above' if t['above50'] else 'below'} EMA50 ({fmt(r.code, t['ema50'])}), "
            f"{'above' if t['above200'] else 'below'} SMA200 ({fmt(r.code, t['sma200'])})")
        add(f"- Momentum: {t['mom5']:+.2f}% 5d · {t['mom20']:+.2f}% 20d · "
            f"{t['mom60']:+.2f}% 60d")
        add(f"- Stretch above EMA20: **{t['ext20']:+.2f}%** · RSI(14) daily "
            f"**{t['rsi']:.0f}**")
        add(f"- Variance ratio VR({SESSIONS}) = **{r.vol.vr5:.2f}** — "
            + (f"returns trend inside the week, so a {n_word()}-day move tends to "
               "extend rather than fade" if r.vol.vr5 > 1.05 else
               f"returns mean-revert inside the week, so {n_word()}-day moves tend "
               "to give some back" if r.vol.vr5 < 0.95 else
               "close to a random walk inside the week"))
        add("")
        for n in r.notes:
            add(f"> {n}")
        if r.notes:
            add("")

    # ---- cross asset ---------------------------------------------------
    add("---")
    add("")
    add("## These are not three independent bets")
    add("")
    add(f"Weekly log-return correlation, last {cross['n_1y']} sessions (1y) and "
        f"last 63 (3m):")
    add("")
    codes = list(results)
    add("| | " + " | ".join(codes) + " |")
    add("|---|" + "---|" * len(codes))
    for a_ in codes:
        cells = []
        for b_ in codes:
            v1 = cross["corr_1y"][a_][b_]
            v3 = cross["corr_3m"][a_][b_]
            cells.append("—" if a_ == b_ else f"{v1:+.2f} / {v3:+.2f}")
        add(f"| **{a_}** | " + " | ".join(cells) + " |")
    add("")
    add("*(1y / 3m)*")
    add("")

    return "\n".join(L)


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def chart(r: AssetWeek, mon: pd.Timestamp, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from quantlab.plotting import BASELINE, GRID, MUTED, PALETTE, apply_style
    apply_style()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    vm, spot = r.vol, r.spot
    days = np.arange(0, SESSIONS + 1)
    labels = ["now"] + [(mon + pd.Timedelta(days=i)).strftime("%a") for i in range(SESSIONS)]

    bands = [(0.05, 0.95, 0.16), (0.25, 0.75, 0.30)]
    for lo, hi, alpha in bands:
        loq = [spot] + [spot * np.exp(t_quantile(lo, vm.nu) * vm.sigma_day_eff * np.sqrt(d))
                        for d in days[1:]]
        hiq = [spot] + [spot * np.exp(t_quantile(hi, vm.nu) * vm.sigma_day_eff * np.sqrt(d))
                        for d in days[1:]]
        ax.fill_between(days, loq, hiq, color=PALETTE[0], alpha=alpha, lw=0,
                        label=f"{round((hi - lo) * 100)}% band")
    ax.plot(days, [spot] * len(days), color=BASELINE, lw=1.6, ls="--",
            label=f"spot {fmt(r.code, spot)}")

    shown = 0
    for row in r.levels:
        if 12 < row["p_touch"] < 92 and shown < 7:
            ax.axhline(row["price"], color=MUTED, lw=0.8, alpha=0.55)
            ax.text(SESSIONS + 0.06, row["price"],
                    f"{row['name'].split(' + ')[0]}  {row['p_touch']:.0f}%",
                    fontsize=7.5, color=MUTED, va="center")
            shown += 1

    ax.set_xticks(days); ax.set_xticklabels(labels)
    ax.set_xlim(0, SESSIONS + 1.15)
    ax.set_title(f"{r.code} — week ahead cone, {mon:%d %b} to "
                 f"{(mon + pd.Timedelta(days=SESSIONS - 1)):%d %b %Y}")
    ax.set_ylabel(f"price ({unit(r.code) if r.code == 'XAUUSD' else 'rate'})")
    ax.grid(color=GRID, lw=0.6, alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    codes = [c.upper() for c in argv if not c.startswith("-")] or \
        list(asset_spec.DEFAULT_CODES)
    want_chart = "--no-chart" not in argv

    now = pd.Timestamp.now(tz="UTC")
    mon, fri = week_bounds(now.tz_localize(None))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Week Ahead — {mon:%Y-%m-%d} to {fri:%Y-%m-%d}  "
          f"(built {now:%Y-%m-%d %H:%M} UTC)")

    results: dict[str, AssetWeek] = {}
    for code in codes:
        print(f"  {code} ...", flush=True)
        r = analyse(code, now)
        if r:
            results[code] = r

    if not results:
        print("nothing to report")
        return 1

    events, _ = macro_data.calendar(days=21)
    cross = cross_asset(results, now)
    md = render(results, cross, events, now, mon, fri)

    out = OUT_DIR / "latest.md"
    out.write_text(md, encoding="utf-8")
    stamp = f"{mon:%Y-%m-%d}"
    (HISTORY_DIR / f"week_ahead_{stamp}.md").write_text(md, encoding="utf-8")

    if want_chart:
        for code, r in results.items():
            p = OUT_DIR / f"latest_{code}.png"
            try:
                chart(r, mon, p)
                (HISTORY_DIR / f"week_ahead_{stamp}_{code}.png").write_bytes(p.read_bytes())
            except Exception as exc:
                print(f"  chart {code} failed: {type(exc).__name__}: {exc}")

    state = {code: {"spot": r.spot, "sigma_week": r.vol.sigma_week,
                    "vr5": r.vol.vr5, "nu": r.vol.nu,
                    "cone": r.cone, "analog": r.analog, "calib": r.calib}
             for code, r in results.items()}
    (OUT_DIR / "state.json").write_text(json.dumps(state, indent=2, default=str),
                                        encoding="utf-8")

    print(f"\nReport -> {out}")
    for code, r in results.items():
        c = r.cone
        print(f"  {code}: spot {fmt(code, r.spot)}  sigma_wk "
              f"{r.vol.sigma_week * 100:.2f}%  cone 5-95 "
              f"{fmt(code, c['q05'])} .. {fmt(code, c['q95'])}  "
              f"analogue P(up) {r.analog.get('p_up', float('nan')):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
