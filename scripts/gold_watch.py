"""Gold Watch — XAUUSD analysis that reruns twice a day: London, then New York.

One run = one update:
  1. pull fresh data (Twelve Data XAU/USD 15m spot + Yahoo GC=F daily/hourly),
  2. rebuild the fat-tailed rest-of-day probability model from notebook 14,
  3. recompute bullish/bearish probabilities, support/resistance clusters,
  4. generate trade setups (entry / stop / TP1 / TP2 / R:R / confidence / why),
  5. diff everything against the previous run and flag significant changes,
  6. write dashboard/gold_watch/latest.md (+ fan chart PNG), archive to
     history/, and save the run state as JSON for the next comparison.

Scheduling: register_gold_watch_task.ps1 fires gold_watch.bat twice a day —
once in the London session (~08:10 UTC), once in New York (~14:10 UTC); the
guard below only lets runs through Mon-Fri between the London open and the NY
close (~07:00-21:00 UTC).  `--force` overrides the guard for manual runs,
`--no-chart` skips the PNG.

The probability model is deliberately driftless (a map, not a forecast):
directional evidence comes separately from the trend-regime tilt and intraday
momentum, and the two are never mixed into the cone.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantlab.analysis.levels import atr, pivot_points
from quantlab.data import twelvedata, yahoo

OUT_DIR = REPO / "dashboard" / "gold_watch"
HISTORY_DIR = OUT_DIR / "history"
STATE_DIR = OUT_DIR / "state"

SPREAD = 0.35          # typical retail XAUUSD CFD spread, $/oz round trip
SESSION_CLOSE_UTC = 21  # NY close — the horizon every probability points at
GUARD_START_UTC = 6.75  # 06:45, a touch before London open
GUARD_END_UTC = 21.25   # 21:15, a touch after NY close
QS = [0.05, 0.25, 0.50, 0.75, 0.95]

# What counts as a "significant change" between two runs.
BULL_SHIFT_PTS = 8.0    # composite bullish % moved this many points
VOL_SHIFT_REL = 0.10    # EWMA daily sigma moved 10% relative
PRICE_SHIFT_ATR = 0.5   # spot moved half an ATR between updates


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _retry(fn, tries: int = 3, wait: int = 15):
    """Yahoo intermittently returns 'no data' under throttling; an unattended
    job has to ride through that rather than skip an update."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def fetch_data(now: pd.Timestamp) -> dict:
    """Spot 15m bars (feed clock corrected to UTC), live price, GC=F frames."""
    try:
        spot_15m = twelvedata.get_time_series("XAU/USD", interval="15min", outputsize=450)
        spot_15m = spot_15m.rename(columns=str.title)
        live_px = float(twelvedata.get_price("XAU/USD"))
        in_spot_terms = True
    except Exception as e:  # rate limit, network, missing key -> futures terms
        print(f"Twelve Data unavailable ({e}); falling back to GC=F futures terms.")
        spot_15m, live_px, in_spot_terms = None, float("nan"), False

    gc_daily = _retry(lambda: yahoo.get_ohlcv(
        "GC=F", start=(now - pd.Timedelta(days=1100)).strftime("%Y-%m-%d")))
    gc_hourly = _retry(lambda: yahoo.get_ohlcv(
        "GC=F", start=(now - pd.Timedelta(days=60)).strftime("%Y-%m-%d"), interval="1h"))
    gc_last = float(gc_daily["Close"].iloc[-1])
    if not in_spot_terms:
        live_px = gc_last
    basis = gc_last - live_px

    if spot_15m is not None:
        if spot_15m.index.tz is None:
            spot_15m.index = spot_15m.index.tz_localize("UTC")
        # Twelve Data XAU/USD stamps bars in the feed's local clock, not UTC.
        feed_offset = round((spot_15m.index[-1] - now) / pd.Timedelta(hours=1)) * pd.Timedelta(hours=1)
        if feed_offset != pd.Timedelta(0):
            spot_15m.index = spot_15m.index - feed_offset

    return {
        "spot_15m": spot_15m, "live_px": live_px, "in_spot_terms": in_spot_terms,
        "gc_daily": gc_daily, "gc_hourly": gc_hourly, "basis": basis,
    }


# --------------------------------------------------------------------------
# volatility + probability machinery (notebook 14 model)
# --------------------------------------------------------------------------

def build_model(data: dict, now: pd.Timestamp) -> dict:
    gc_daily, gc_hourly = data["gc_daily"], data["gc_hourly"]
    ret = np.log(gc_daily["Close"] / gc_daily["Close"].shift(1)).dropna()

    sigma_day = float(np.sqrt(ret.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean().iloc[-1]))
    atr14 = float(atr(gc_daily).iloc[-1])

    # Student-t index by method of moments (excess kurtosis of t = 6/(nu-4)).
    k_ex = float(ret.kurtosis())
    nu = float(np.clip(4 + 6 / k_ex, 4.5, 30)) if k_ex > 0.2 else 30.0
    rng = np.random.default_rng(42)  # fixed seed: run-to-run prob deltas are real, not MC noise
    Z = rng.standard_t(nu, 1_000_000)
    Z /= np.sqrt(nu / (nu - 2))

    # Hour-of-day variance profile -> share of daily variance still ahead.
    h = gc_hourly.copy()
    h["r2"] = np.log(h["Close"] / h["Close"].shift(1)) ** 2
    prof = h.groupby(h.index.tz_convert("UTC").hour)["r2"].mean()
    prof = prof / prof.sum()

    session_end = now.normalize() + pd.Timedelta(hours=SESSION_CLOSE_UTC)
    if now >= session_end:
        session_end += pd.Timedelta(days=1)
    session_start = session_end - pd.Timedelta(hours=23)
    hours_left = [ts.hour for ts in pd.date_range(now.ceil("h"), session_end, freq="h", inclusive="left")]
    frac_of_hour = (now.ceil("h") - now) / pd.Timedelta(hours=1) * prof.get(now.hour, 0.0)
    f_remain = float(np.clip(prof.reindex(hours_left).fillna(0).sum() + frac_of_hour, 0.02, 1.0))
    sigma_remain = sigma_day * np.sqrt(f_remain)

    zq = {q: float(np.quantile(Z, q)) for q in QS}

    def p_above(level: float, ref: float, sig: float = sigma_remain) -> float:
        return float((Z > np.log(level / ref) / sig).mean())

    def p_touch(level: float, ref: float, sig: float = sigma_remain) -> float:
        pa = p_above(level, ref, sig)
        return float(min(1.0, 2 * (pa if level >= ref else 1 - pa)))

    return {
        "sigma_day": sigma_day, "sigma_remain": sigma_remain, "f_remain": f_remain,
        "nu": nu, "atr14": atr14, "zq": zq, "p_above": p_above, "p_touch": p_touch,
        "session_start": session_start, "session_end": session_end,
    }


# --------------------------------------------------------------------------
# levels, regime, probabilities
# --------------------------------------------------------------------------

def build_levels(data: dict, model: dict, now: pd.Timestamp) -> dict:
    gc_daily, spot_15m = data["gc_daily"], data["spot_15m"]
    live_px, basis = data["live_px"], data["basis"]

    gc_done = gc_daily.iloc[-2] if gc_daily.index[-1].date() >= now.date() else gc_daily.iloc[-1]
    piv = {k: v - basis for k, v in pivot_points(
        float(gc_done["High"]), float(gc_done["Low"]), float(gc_done["Close"])).items()}

    today_bars = spot_15m[spot_15m.index >= model["session_start"]] if spot_15m is not None else None
    have_today = today_bars is not None and len(today_bars) > 0
    today_hi = float(today_bars["High"].max()) if have_today else float("nan")
    today_lo = float(today_bars["Low"].min()) if have_today else float("nan")
    today_open = float(today_bars["Open"].iloc[0]) if have_today else float(gc_done["Close"]) - basis

    step = 25 if live_px < 3000 else 50
    base = np.floor(live_px / step) * step
    rounds = {f"round {int(x)}": float(x) for x in (base - step, base, base + step, base + 2 * step)}

    levels = {
        "R2": piv["R2"], "R1": piv["R1"], "pivot P": piv["P"], "S1": piv["S1"], "S2": piv["S2"],
        "prev close": float(gc_done["Close"]) - basis,
        "prev high": float(gc_done["High"]) - basis, "prev low": float(gc_done["Low"]) - basis,
        "today high": today_hi, "today low": today_lo,
        "20d high": float(gc_daily["High"].iloc[-20:].max()) - basis,
        "20d low": float(gc_daily["Low"].iloc[-20:].min()) - basis,
        **rounds,
    }
    levels = {k: v for k, v in levels.items() if np.isfinite(v)}
    return {"levels": levels, "today_hi": today_hi, "today_lo": today_lo,
            "today_open": today_open, "prev_close": float(gc_done["Close"]) - basis,
            "today_bars": today_bars if have_today else None}


def build_regime(data: dict) -> dict:
    gc_daily, live_px, basis = data["gc_daily"], data["live_px"], data["basis"]
    close = gc_daily["Close"]
    ema20, ema50 = close.ewm(span=20).mean(), close.ewm(span=50).mean()
    mom5 = float(close.iloc[-1] / close.iloc[-6] - 1)
    px_ref = live_px + basis
    bull_now = px_ref > ema20.iloc[-1] > ema50.iloc[-1]
    bear_now = px_ref < ema20.iloc[-1] < ema50.iloc[-1]
    regime = "bullish" if bull_now else ("bearish" if bear_now else "mixed")

    state = ((close < ema20) & (ema20 < ema50)) if regime == "bearish" else \
            ((close > ema20) & (ema20 > ema50)) if regime == "bullish" else \
            pd.Series(True, index=close.index)
    nxt = close.pct_change().shift(-1)[state].dropna()
    return {
        "regime": regime, "ema20": float(ema20.iloc[-1] - basis), "ema50": float(ema50.iloc[-1] - basis),
        "mom5": mom5, "tilt_n": int(len(nxt)), "tilt_up": float((nxt > 0).mean()),
        "tilt_median": float(nxt.median()), "tilt_mean": float(nxt.mean()),
    }


def build_probabilities(data: dict, model: dict, lv: dict, rg: dict) -> dict:
    """Composite bullish/bearish % from three independent reads.

    45% rest-of-day model P(close >= prev close), 30% historical regime tilt,
    25% intraday momentum (position in today's range + last-2h thrust scaled
    by the vol the model expects over 2h). Weights are judgment, not fit —
    they favor the probabilistic model and keep momentum from whipsawing the
    composite every update.
    """
    live_px = data["live_px"]
    p_close_up = model["p_above"](lv["prev_close"], live_px)
    p_above_open = model["p_above"](lv["today_open"], live_px)

    bars = lv["today_bars"]
    if bars is not None and len(bars) >= 2:
        rng_hi, rng_lo = lv["today_hi"], lv["today_lo"]
        pos_in_range = (live_px - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5
        r2h = float(np.log(bars["Close"].iloc[-1] / bars["Close"].iloc[max(0, len(bars) - 9)]))
        thrust = 0.5 + 0.5 * np.tanh(r2h / (model["sigma_day"] * np.sqrt(2 / 24)))
        momentum = 0.5 * pos_in_range + 0.5 * thrust
    else:
        momentum = 0.5

    bull = 0.45 * p_close_up + 0.30 * rg["tilt_up"] + 0.25 * momentum
    return {
        "bull_pct": float(bull * 100), "bear_pct": float((1 - bull) * 100),
        "p_close_up": float(p_close_up * 100), "p_above_open": float(p_above_open * 100),
        "tilt_up_pct": float(rg["tilt_up"] * 100), "momentum_score": float(momentum * 100),
    }


# --------------------------------------------------------------------------
# trade setups
# --------------------------------------------------------------------------

def clustered(levels: dict[str, float], tol: float) -> list[dict]:
    groups: list[list[tuple[str, float]]] = []
    for name, px in sorted(levels.items(), key=lambda kv: kv[1]):
        if groups and px - groups[-1][-1][1] <= tol:
            groups[-1].append((name, px))
        else:
            groups.append([(name, px)])
    return [{"lo": g[0][1], "hi": g[-1][1], "mid": (g[0][1] + g[-1][1]) / 2,
             "n": len(g), "names": " + ".join(n for n, _ in g)} for g in groups]


def _next_away(side: list[dict], frm: float, min_dist: float) -> dict | None:
    for c in side:
        edge = c["hi"] if c["hi"] < frm else c["lo"]
        if abs(frm - edge) >= min_dist:
            return c
    return None


def _confidence(side_aligned: str, cluster_n: int, rr1: float, fill: float,
                ev: float, stop_dist: float, atr14: float) -> int:
    """0-100, transparent weighted sum:

    regime alignment 30 / 15 / 5  (with / mixed / against the daily trend)
    confluence       7 per level in the entry cluster, max 21
    R:R to TP1       min(RR,3)/3 * 15
    fill odds        P(entry fills today) * 14
    EV after spread  clip(EV, 0, 2)/2 * 10
    stop placement   min(stop / 0.25 ATR, 1) * 10  (stops inside noise score low)
    """
    score = {"with": 30, "mixed": 15, "against": 5}[side_aligned]
    score += 7 * min(cluster_n, 3)
    score += min(rr1, 3) / 3 * 15
    score += fill * 14
    score += float(np.clip(ev, 0, 2)) / 2 * 10
    score += min(stop_dist / (0.25 * atr14), 1) * 10
    return int(round(score))


def build_plans(data: dict, model: dict, lv: dict, rg: dict) -> list[dict]:
    live_px, atr14 = data["live_px"], model["atr14"]
    p_touch = model["p_touch"]
    clusters = clustered(lv["levels"], tol=0.15 * atr14)
    res = [c for c in clusters if c["lo"] > live_px]
    sup = [c for c in clusters if c["hi"] < live_px][::-1]

    def make(name: str, side: str, entry: float, stop: float, entry_cluster: dict,
             tp1: float, tp2: float | None, why: str) -> dict | None:
        if not (min(stop, tp1) < entry < max(stop, tp1)):
            return None
        risk = abs(entry - stop)
        rew1 = abs(tp1 - entry)
        rr1 = rew1 / risk
        rr2 = abs(tp2 - entry) / risk if tp2 is not None else None
        # Barrier odds with the regime tilt folded in as drift (gambler's ruin
        # for drifted BM). Driftless this collapses to risk/(risk+reward) and
        # every plan's gross EV is exactly zero — the tilt is the only edge.
        u = abs(np.log(tp1 / entry))
        dn = abs(np.log(stop / entry))
        mu = rg["tilt_mean"] * model["f_remain"] * (1 if side == "long" else -1)
        x = 2 * mu / model["sigma_remain"] ** 2
        if abs(x * (u + dn)) < 1e-6:
            p_win = dn / (u + dn)
        else:
            p_win = float((np.exp(x * dn) - 1) / (np.exp(x * dn) - np.exp(-x * u)))
        fill = 1.0 if abs(entry - live_px) < 0.5 else p_touch(entry, live_px)
        ev = p_win * rew1 - (1 - p_win) * risk - SPREAD
        aligned = "mixed" if rg["regime"] == "mixed" else (
            "with" if (side == "long") == (rg["regime"] == "bullish") else "against")
        conf = _confidence(aligned, entry_cluster["n"], rr1, fill, ev, risk, atr14)
        return {
            "plan": name, "side": side, "entry": round(entry, 1), "stop": round(stop, 1),
            "tp1": round(tp1, 1), "tp2": round(tp2, 1) if tp2 is not None else None,
            "rr1": round(rr1, 2), "rr2": round(rr2, 2) if rr2 is not None else None,
            "p_win_pct": round(p_win * 100), "fill_pct": round(fill * 100),
            "ev": round(ev, 2), "confidence": conf, "why": why,
        }

    plans: list[dict | None] = []
    if rg["regime"] != "bullish" and res and sup:
        fade, brk = res[0], sup[0]
        tgt1 = _next_away(sup, fade["mid"], 0.5 * atr14)
        if tgt1:
            beyond = [c for c in sup if c["hi"] < tgt1["lo"]]
            tgt2 = _next_away(beyond, tgt1["lo"], 0.3 * atr14)
            plans.append(make(
                "fade the bounce", "short", fade["mid"], fade["hi"] + 0.2 * atr14, fade,
                tgt1["hi"], tgt2["hi"] if tgt2 else None,
                f"Sell a pullback into {fade['names']} ({fade['lo']:,.1f}-{fade['hi']:,.1f}), "
                f"stop 0.2 ATR beyond the cluster. Daily regime is {rg['regime']} "
                f"(price vs EMA20/50), historical next-day tilt {rg['tilt_median']*100:+.2f}% median; "
                f"TP1 is the {tgt1['names']} cluster."))
            plans.append(make(
                "short at market", "short", live_px, fade["hi"] + 0.2 * atr14, fade,
                tgt1["hi"], tgt2["hi"] if tgt2 else None,
                f"Immediate short with the stop above {fade['names']}; pays a worse entry than "
                f"the fade for a guaranteed fill in a {rg['regime']} regime."))
        tgt2b = _next_away(sup[1:], brk["lo"], 0.5 * atr14)
        if tgt2b:
            entry = brk["lo"] - 0.05 * atr14
            beyond = [c for c in sup if c["hi"] < tgt2b["lo"]]
            tgt3 = _next_away(beyond, tgt2b["lo"], 0.3 * atr14)
            plans.append(make(
                "breakdown", "short", entry, max(brk["hi"] + 0.1 * atr14, entry + 0.3 * atr14), brk,
                tgt2b["hi"], tgt3["hi"] if tgt3 else None,
                f"Stop-entry through support {brk['names']} ({brk['lo']:,.1f}); momentum trade — "
                f"only triggers if the level actually breaks, targets {tgt2b['names']}."))
    if rg["regime"] != "bearish" and res and sup:
        fade, brk = sup[0], res[0]
        tgt1 = _next_away(res, fade["mid"], 0.5 * atr14)
        if tgt1:
            beyond = [c for c in res if c["lo"] > tgt1["hi"]]
            tgt2 = _next_away(beyond, tgt1["hi"], 0.3 * atr14)
            plans.append(make(
                "buy the dip", "long", fade["mid"], fade["lo"] - 0.2 * atr14, fade,
                tgt1["lo"], tgt2["lo"] if tgt2 else None,
                f"Buy a dip into {fade['names']} ({fade['lo']:,.1f}-{fade['hi']:,.1f}), "
                f"stop 0.2 ATR below the cluster. Daily regime is {rg['regime']}, historical "
                f"next-day tilt {rg['tilt_median']*100:+.2f}% median; TP1 is {tgt1['names']}."))
            plans.append(make(
                "long at market", "long", live_px, fade["lo"] - 0.2 * atr14, fade,
                tgt1["lo"], tgt2["lo"] if tgt2 else None,
                f"Immediate long with the stop below {fade['names']}; guaranteed fill, "
                f"worse entry than the dip-buy."))
        tgt2b = _next_away(res[1:], brk["hi"], 0.5 * atr14)
        if tgt2b:
            entry = brk["hi"] + 0.05 * atr14
            beyond = [c for c in res if c["lo"] > tgt2b["hi"]]
            tgt3 = _next_away(beyond, tgt2b["hi"], 0.3 * atr14)
            plans.append(make(
                "breakout", "long", entry, min(brk["lo"] - 0.1 * atr14, entry - 0.3 * atr14), brk,
                tgt2b["lo"], tgt3["lo"] if tgt3 else None,
                f"Stop-entry through resistance {brk['names']} ({brk['hi']:,.1f}); only triggers "
                f"on the break, targets {tgt2b['names']}."))

    out = sorted([p for p in plans if p], key=lambda p: -p["confidence"])
    return out


# --------------------------------------------------------------------------
# comparison with the previous run
# --------------------------------------------------------------------------

def load_prev_state() -> dict | None:
    f = STATE_DIR / "latest.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare_runs(prev: dict | None, cur: dict) -> list[str]:
    """Human-readable highlights of what actually changed since last update."""
    if prev is None:
        return ["First run — nothing to compare against yet."]
    notes: list[str] = []
    age_h = (pd.Timestamp(cur["run_ts"]) - pd.Timestamp(prev["run_ts"])) / pd.Timedelta(hours=1)
    stale = " (previous run is from an earlier day)" if age_h > 20 else ""

    dpx = cur["live_px"] - prev["live_px"]
    if abs(dpx) >= PRICE_SHIFT_ATR * cur["atr14"]:
        notes.append(f"**Big move:** spot {dpx:+,.1f} $/oz since last update{stale} — "
                     f"{abs(dpx) / cur['atr14']:.1f}x ATR(14).")

    lo, hi = sorted((prev["live_px"], cur["live_px"]))
    crossed = [n for n, v in cur["levels"].items()
               if lo < v < hi and n not in ("today high", "today low")]
    if crossed:
        notes.append("**Structure:** traded through " + ", ".join(sorted(crossed)) + ".")

    if prev.get("regime") != cur["regime"]:
        notes.append(f"**Regime flip:** {prev.get('regime')} -> {cur['regime']} "
                     f"(price vs EMA20/EMA50 realignment).")

    dbull = cur["bull_pct"] - prev["bull_pct"]
    if abs(dbull) >= BULL_SHIFT_PTS:
        notes.append(f"**Probability shift:** composite bullish {prev['bull_pct']:.0f}% -> "
                     f"{cur['bull_pct']:.0f}% ({dbull:+.0f} pts).")

    dvol = cur["sigma_day"] / prev["sigma_day"] - 1 if prev.get("sigma_day") else 0
    if abs(dvol) >= VOL_SHIFT_REL:
        notes.append(f"**Volatility:** EWMA daily sigma {dvol:+.0%} vs last update "
                     f"({prev['sigma_day']*100:.2f}% -> {cur['sigma_day']*100:.2f}%).")

    dmom = cur["mom5"] - prev.get("mom5", cur["mom5"])
    if abs(dmom) >= 0.005:
        notes.append(f"**Momentum:** 5-day momentum {prev['mom5']*100:+.2f}% -> "
                     f"{cur['mom5']*100:+.2f}%.")

    if np.isfinite(cur.get("today_hi", float("nan"))) and prev.get("today_hi") is not None:
        if age_h < 20 and np.isfinite(prev["today_hi"]):
            if cur["today_hi"] > prev["today_hi"] + 0.5:
                notes.append(f"**New session high** {cur['today_hi']:,.1f} "
                             f"(was {prev['today_hi']:,.1f}).")
            if cur["today_lo"] < prev["today_lo"] - 0.5:
                notes.append(f"**New session low** {cur['today_lo']:,.1f} "
                             f"(was {prev['today_lo']:,.1f}).")

    prev_best = (prev.get("plans") or [{}])[0]
    cur_best = (cur.get("plans") or [{}])[0]
    if prev_best.get("side") and cur_best.get("side") and prev_best["side"] != cur_best["side"]:
        notes.append(f"**Best setup flipped sides:** {prev_best['side']} "
                     f"({prev_best.get('plan')}) -> {cur_best['side']} ({cur_best.get('plan')}).")

    return notes or [f"No significant changes since the last update ({age_h:.1f}h ago) — "
                     f"same regime, probabilities within {BULL_SHIFT_PTS:.0f} pts, "
                     f"price inside the prior structure."]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """Minimal markdown table (avoids the tabulate dependency)."""
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k, "—") if r.get(k) is not None else "—")
                              for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def session_name(now: pd.Timestamp) -> str:
    h = now.hour + now.minute / 60
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 21:
        return "New York"
    return "off-session"


def render_report(cur: dict, prev: dict | None, notes: list[str],
                  lv_rows: list[dict], chart_name: str | None) -> str:
    now = pd.Timestamp(cur["run_ts"])
    bull, bear = cur["bull_pct"], cur["bear_pct"]
    bias = "BULLISH" if bull >= 55 else ("BEARISH" if bull <= 45 else "NEUTRAL")

    md = [
        f"# Gold Watch — XAUUSD — {now:%Y-%m-%d %H:%M} UTC ({session_name(now)} session)",
        "",
        f"**Spot {cur['live_px']:,.2f}** | bias **{bias}** — bullish {bull:.0f}% / "
        f"bearish {bear:.0f}% | regime **{cur['regime']}** | "
        f"EWMA vol {cur['sigma_day']*100:.2f}%/day (${cur['live_px']*cur['sigma_day']:,.0f}) | "
        f"ATR(14) ${cur['atr14']:,.1f} | {cur['f_remain']*100:.0f}% of today's variance still ahead",
        "",
        "## Changes since last update",
        "",
        *[f"- {n}" for n in notes],
    ]
    if prev:
        md += ["", _md_table([
            {"m": "spot", "prev": f"{prev['live_px']:,.2f}", "now": f"{cur['live_px']:,.2f}",
             "d": f"{cur['live_px']-prev['live_px']:+,.2f}"},
            {"m": "bullish %", "prev": f"{prev['bull_pct']:.0f}", "now": f"{bull:.0f}",
             "d": f"{bull-prev['bull_pct']:+.0f}"},
            {"m": "regime", "prev": prev["regime"], "now": cur["regime"],
             "d": "flip" if prev["regime"] != cur["regime"] else "—"},
            {"m": "EWMA sigma %/day", "prev": f"{prev['sigma_day']*100:.2f}",
             "now": f"{cur['sigma_day']*100:.2f}",
             "d": f"{(cur['sigma_day']/prev['sigma_day']-1)*100:+.0f}%"},
            {"m": "5d momentum %", "prev": f"{prev['mom5']*100:+.2f}",
             "now": f"{cur['mom5']*100:+.2f}", "d": f"{(cur['mom5']-prev['mom5'])*100:+.2f}"},
        ], [("m", "metric"), ("prev", f"prev ({pd.Timestamp(prev['run_ts']):%H:%M} UTC)"),
            ("now", "now"), ("d", "delta")])]

    md += [
        "",
        "## Probabilities",
        "",
        f"- **Composite: bullish {bull:.0f}% / bearish {bear:.0f}%** "
        f"(45% rest-of-day model, 30% regime tilt, 25% intraday momentum)",
        f"- P(close >= prev close {cur['prev_close']:,.1f}) = {cur['p_close_up']:.0f}% "
        f"— the fat-tailed cone from spot, driftless",
        f"- Historical {cur['regime']}-regime next-day P(up) = {cur['tilt_up_pct']:.0f}% "
        f"(n={cur['tilt_n']} days, median {cur['tilt_median']*100:+.2f}%)",
        f"- Intraday momentum score {cur['momentum_score']:.0f}/100 "
        f"(position in today's range + last-2h thrust)",
        f"- Today's close cone (21:00 UTC): 5% {cur['cone'][0]:,.0f} / 25% {cur['cone'][1]:,.0f} / "
        f"median {cur['cone'][2]:,.0f} / 75% {cur['cone'][3]:,.0f} / 95% {cur['cone'][4]:,.0f}",
        "",
        "## Key levels",
        "",
        _md_table(lv_rows, [("level", "level"), ("price", "price"), ("dist", "dist $"),
                            ("p_close", "P(close above) %"), ("p_touch", "P(touch today) %")]),
        "",
        "## Trade setups (best first)",
        "",
    ]
    if not cur["plans"]:
        md.append("_No setup passes the filters right now (no clean cluster on the "
                  "regime-aligned side). Standing aside is the trade._")
    for p in cur["plans"]:
        tp2 = f" / TP2 {p['tp2']:,.1f} (R:R {p['rr2']})" if p["tp2"] is not None else ""
        md += [
            f"### {p['plan'].upper()} — {p['side']} @ {p['entry']:,.1f} — confidence {p['confidence']}/100",
            "",
            f"- Stop {p['stop']:,.1f} | TP1 {p['tp1']:,.1f} (R:R {p['rr1']}){tp2}",
            f"- P(win to TP1) {p['p_win_pct']}% (barrier odds incl. regime tilt) | "
            f"P(entry fills today) {p['fill_pct']}% | model EV {p['ev']:+.2f} $/oz after spread",
            f"- {p['why']}",
            "",
        ]
    if chart_name:
        md += ["## Chart", "", f"![probability fan]({chart_name})", ""]
    md += [
        "---",
        f"_Model: driftless Student-t (nu={cur['nu']:.1f}) at EWMA vol, variance-time scaled "
        f"to the 21:00 UTC close; levels from GC=F pivots/prior day shifted to spot by the "
        f"{cur['basis']:+,.2f} basis. P(win) is the barrier rule with the regime tilt as drift — "
        f"with zero tilt it collapses to risk/(risk+reward) and EV is minus the spread; the tilt "
        f"and level structure are the edge. Data: {cur['data_source']}. Not financial advice; "
        f"size so a stop-out costs "
        f"<=1% of the account._",
    ]
    return "\n".join(md)


def make_chart(data: dict, model: dict, lv: dict, path: Path, now: pd.Timestamp) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from quantlab.plotting import BASELINE, INK, MUTED, SECONDARY, SEQUENTIAL_STEPS, apply_style

        apply_style()
        live_px, spot_15m = data["live_px"], data["spot_15m"]
        sigma_remain, zq = model["sigma_remain"], model["zq"]

        fig, ax = plt.subplots(figsize=(10, 5.2))
        hist = (spot_15m if spot_15m is not None else data["gc_hourly"])["Close"]
        hist = hist[hist.index >= now.normalize() - pd.tseries.offsets.BDay(2)]
        ax.plot(hist.index, hist.values, color=SEQUENTIAL_STEPS[7], lw=1.6)

        fan_t = pd.date_range(now, model["session_end"], freq="15min")
        fracs = np.linspace(0.02, 1.0, len(fan_t))
        bands = {q: np.array([live_px * float(np.exp(zq[q] * sigma_remain * np.sqrt(f)))
                              for f in fracs]) for q in QS}
        ax.fill_between(fan_t, bands[0.05], bands[0.95], color=SEQUENTIAL_STEPS[1], alpha=0.75, lw=0)
        ax.fill_between(fan_t, bands[0.25], bands[0.75], color=SEQUENTIAL_STEPS[4], alpha=0.75, lw=0)
        ax.plot(fan_t, bands[0.50], color=SEQUENTIAL_STEPS[8], lw=1.2, ls=(0, (4, 3)))
        for q, lbl in [(0.95, "95%"), (0.75, "75%"), (0.50, "median"), (0.25, "25%"), (0.05, "5%")]:
            ax.annotate(f" {lbl} {bands[q][-1]:,.0f}", xy=(fan_t[-1], bands[q][-1]),
                        fontsize=8.5, color=SECONDARY, va="center")

        lo, hi = min(hist.min(), bands[0.05][-1]), max(hist.max(), bands[0.95][-1])
        for name in ("R2", "R1", "pivot P", "S1", "S2"):
            lvl = lv["levels"].get(name)
            if lvl and lo - 2 < lvl < hi + 2:
                ax.axhline(lvl, color=BASELINE, lw=0.9, ls=(0, (2, 3)))
                ax.annotate(f"{name} {lvl:,.0f} ", xy=(hist.index[0], lvl), fontsize=8.5,
                            color=MUTED, va="bottom", ha="left")
        ax.scatter([hist.index[-1]], [live_px], s=28, color=INK, zorder=5)
        ax.annotate(f"  {live_px:,.1f}", xy=(hist.index[-1], live_px), color=INK,
                    fontsize=9.5, fontweight="bold", va="bottom")
        ax.set_title(f"XAUUSD — probability fan to the close ({model['session_end']:%d %b} 21:00 UTC)",
                     pad=24)
        ax.text(0.0, 1.02, f"Gold Watch run {now:%Y-%m-%d %H:%M} UTC — driftless Student-t fan",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
        ax.margins(x=0.10)
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"chart skipped ({e})")
        return False


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def in_session(now: pd.Timestamp) -> bool:
    h = now.hour + now.minute / 60
    return now.dayofweek < 5 and GUARD_START_UTC <= h <= GUARD_END_UTC


def main(argv: list[str]) -> int:
    force = "--force" in argv
    no_chart = "--no-chart" in argv
    now = pd.Timestamp.now(tz="UTC")

    if not force and not in_session(now):
        print(f"{now:%Y-%m-%d %H:%M} UTC is outside London/NY sessions (Mon-Fri "
              f"~07:00-21:00 UTC) — skipped. Use --force to run anyway.")
        return 0

    for d in (OUT_DIR, HISTORY_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Gold Watch run {now:%Y-%m-%d %H:%M} UTC ({session_name(now)})")
    data = fetch_data(now)
    model = build_model(data, now)
    lv = build_levels(data, model, now)
    rg = build_regime(data)
    probs = build_probabilities(data, model, lv, rg)
    plans = build_plans(data, model, lv, rg)

    live_px = data["live_px"]
    lv_rows = [
        {"level": n, "price": f"{v:,.1f}", "dist": f"{v - live_px:+,.1f}",
         "p_close": f"{model['p_above'](v, live_px)*100:.0f}",
         "p_touch": f"{model['p_touch'](v, live_px)*100:.0f}"}
        for n, v in sorted(lv["levels"].items(), key=lambda kv: -kv[1])
    ]
    cone = [round(live_px * float(np.exp(model["zq"][q] * model["sigma_remain"])), 1) for q in QS]

    cur = {
        "run_ts": str(now), "live_px": float(live_px), "basis": float(data["basis"]),
        "data_source": "Twelve Data XAU/USD spot + Yahoo GC=F" if data["in_spot_terms"]
                       else "Yahoo GC=F only (Twelve Data unavailable this run)",
        "sigma_day": model["sigma_day"], "sigma_remain": model["sigma_remain"],
        "f_remain": model["f_remain"], "nu": model["nu"], "atr14": model["atr14"],
        "regime": rg["regime"], "ema20": rg["ema20"], "ema50": rg["ema50"], "mom5": rg["mom5"],
        "tilt_n": rg["tilt_n"], "tilt_up": rg["tilt_up"], "tilt_median": rg["tilt_median"],
        **probs,
        "prev_close": lv["prev_close"], "today_hi": lv["today_hi"], "today_lo": lv["today_lo"],
        "levels": {k: float(v) for k, v in lv["levels"].items()},
        "cone": cone, "plans": plans,
    }

    prev = load_prev_state()
    notes = compare_runs(prev, cur)

    stamp = f"{now:%Y-%m-%d_%H%M}"
    chart_ok = False
    if not no_chart:
        chart_ok = make_chart(data, model, lv, OUT_DIR / "latest.png", now)
    report = render_report(cur, prev, notes, lv_rows, "latest.png" if chart_ok else None)

    (OUT_DIR / "latest.md").write_text(report, encoding="utf-8")
    hist_md = HISTORY_DIR / f"gold_watch_{stamp}.md"
    hist_md.write_text(report.replace("(latest.png)", f"(gold_watch_{stamp}.png)"), encoding="utf-8")
    if chart_ok:
        shutil.copyfile(OUT_DIR / "latest.png", HISTORY_DIR / f"gold_watch_{stamp}.png")

    (STATE_DIR / "latest.json").write_text(
        json.dumps(cur, indent=1, default=float), encoding="utf-8")
    (STATE_DIR / f"run_{stamp}.json").write_text(
        json.dumps(cur, indent=1, default=float), encoding="utf-8")

    # Keep a month of history/state so the folder never balloons.
    cutoff = (now - pd.Timedelta(days=31)).timestamp()
    for d in (HISTORY_DIR, STATE_DIR):
        for f in d.iterdir():
            if f.name != "latest.json" and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

    print(f"\nSpot {live_px:,.2f} | bias bullish {cur['bull_pct']:.0f}% / "
          f"bearish {cur['bear_pct']:.0f}% | regime {cur['regime']}")
    for n in notes:
        print(f"  - {n.replace('**', '')}")
    if plans:
        best = plans[0]
        tp2 = f" TP2 {best['tp2']}" if best["tp2"] is not None else ""
        print(f"Best setup: {best['plan']} ({best['side']}) entry {best['entry']} "
              f"stop {best['stop']} TP1 {best['tp1']}{tp2} R:R {best['rr1']} "
              f"confidence {best['confidence']}/100")
    print(f"Report -> {OUT_DIR / 'latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
