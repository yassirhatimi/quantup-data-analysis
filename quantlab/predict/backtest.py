"""Historical replay: is a predicted 60% actually a 60%?

**Read this before reading the numbers.** The backtest replays the four
*chart-derived* components — technical, price action, volatility, structure —
plus the regime classification and the historical analogue, point-in-time on
historical bars. It does **not** replay macro, news or Polymarket, because no
point-in-time archive of them exists: FRED revises series in place, the free
NewsAPI window is one month, and Polymarket's relevant markets did not exist a
year ago. Backfilling them with today's values would leak the future into every
row and produce a beautiful, worthless equity curve.

So what is measured here is the replayable core, and the SQLite store measures
the full engine live from today forward (`store.calibration_table`). Both are
reported; neither is dressed up as the other.

Point-in-time discipline everywhere else:

- every indicator at time *t* is computed from bars strictly up to *t*;
- the volatility, tail index and hour-of-day profile use only prior data;
- the band that defines RANGE is the band as it stood at *t*, never recomputed
  with hindsight;
- a chronological split fits analogue rates and baseline frequencies on the
  first 70%, purges labels crossing the split and evaluates the last 30%.

    from quantlab.predict import backtest
    report = backtest.run("XAUUSD", horizon="4h")
    print(report.summary_text())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.predict import (assets as asset_spec, config, engine,
                              market_data, regime as regime_mod, technical)
from quantlab.predict.component import ComponentScore, clamp
from quantlab.predict.store import CLASSES, brier, log_loss, realized_class

#: Components the replay can honestly reproduce from historical bars.
REPLAYABLE = ("technical", "price_action", "volatility", "structure")


# --------------------------------------------------------------------------
# point-in-time feature construction
# --------------------------------------------------------------------------

@dataclass
class Sample:
    ts: pd.Timestamp
    price: float
    scores: dict[str, float]
    regime: str
    vol_state: str
    session: str
    hour: int
    sigma: float          # predictive sigma over the horizon
    band: float           # log-return band that defines RANGE
    nu: float
    vol_ratio: float
    forward_return: float
    actual: str
    event_window: bool = False
    outcome_ts: pd.Timestamp | None = None


def _vectorised_features(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Every indicator as a full series, so the replay is one pass not N.

    Computing these bar-by-bar through `technical.compute` would be correct but
    would take hours over 10k bars; every column here is a rolling/expanding
    construct that only ever looks backwards, which gives the same answer.
    """
    a = asset_spec.get(code)
    close, high, low = frame["Close"], frame["High"], frame["Low"]
    out = pd.DataFrame(index=frame.index)

    out["close"] = close
    out["rsi"] = technical.rsi(close)
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    _, _, hist = technical.macd(close)
    out["macd_hist"] = hist
    out["adx"] = technical.adx(frame)

    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    ret = np.log(close / close.shift(1))
    out["ret"] = ret
    out["sigma_fast"] = np.sqrt(ret.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean())
    out["sigma_slow"] = (out["atr"] / close) / technical.PARKINSON
    out["vol_ratio"] = out["sigma_fast"] / out["sigma_slow"]
    # Tail index as it was knowable at each bar: a trailing window, never the
    # kurtosis of the whole sample (which would be hindsight).
    out["nu"] = (4 + 6 / ret.rolling(750, min_periods=250).kurt().clip(lower=0.2)).clip(
        config.NU_MIN, config.NU_MAX).fillna(config.NU_MAX)

    out["mom_1"] = close.pct_change(1)
    out["mom_5"] = close.pct_change(5)
    out["mom_20"] = close.pct_change(20)
    span = high.rolling(24).max() - low.rolling(24).min()
    out["pos_in_range"] = ((close - low.rolling(24).min()) / span.replace(0, np.nan))
    out["thrust"] = np.log(close / close.shift(2))

    out["prev_high"] = high.rolling(24).max().shift(1)
    out["prev_low"] = low.rolling(24).min().shift(1)
    out["swing_high"] = high.rolling(120).max().shift(1)
    out["swing_low"] = low.rolling(120).min().shift(1)
    out["regime"] = regime_mod._regime_series(frame)
    out["hour"] = frame.index.hour
    out["spread_floor"] = a.spread / close
    return out


def _score_row(row: pd.Series) -> dict[str, float]:
    """The replayable components, from one row of the feature frame.

    Mirrors the live scoring functions in `technical.py`; kept deliberately
    close to them so the replay is measuring the same model, not a cousin.
    """
    price = row["close"]
    scores: dict[str, float] = {}

    stack = 0.0
    denom = 0
    for ref in ("sma200", "sma50", "sma20"):
        if np.isfinite(row[ref]):
            stack += 1 if price > row[ref] else -1
            denom += 1
    if np.isfinite(row["ema20"]) and np.isfinite(row["ema50"]):
        stack += 1 if row["ema20"] > row["ema50"] else -1
        denom += 1
    ma = stack / denom if denom else 0.0
    rsi_score = clamp((row["rsi"] - 50) / 20) if np.isfinite(row["rsi"]) else 0.0
    macd_score = 0.0
    if np.isfinite(row["macd_hist"]) and np.isfinite(row["atr"]) and row["atr"] > 0:
        macd_score = float(np.tanh(row["macd_hist"] / (0.30 * row["atr"])))
    tech = (1.4 * ma + 1.0 * rsi_score + 0.8 * macd_score) / 3.2
    if np.isfinite(row["adx"]):
        tech *= float(np.clip(row["adx"] / 30.0, 0.45, 1.25))
    scores["technical"] = clamp(tech)

    parts, weights = [], []
    if np.isfinite(row["pos_in_range"]):
        parts.append(clamp((row["pos_in_range"] - 0.5) * 2)); weights.append(1.2)
    if np.isfinite(row["thrust"]) and np.isfinite(row["sigma_fast"]) and row["sigma_fast"] > 0:
        parts.append(float(np.tanh(row["thrust"] / (row["sigma_fast"] * 1.2)))); weights.append(1.3)
    for key, days, w in (("mom_1", 1, 0.8), ("mom_5", 5, 1.0), ("mom_20", 20, 0.7)):
        if np.isfinite(row[key]) and np.isfinite(row["sigma_fast"]) and row["sigma_fast"] > 0:
            parts.append(float(np.tanh(row[key] / (row["sigma_fast"] * np.sqrt(days) * 1.5))))
            weights.append(w)
    scores["price_action"] = (clamp(np.average(parts, weights=weights))
                              if parts else 0.0)

    vol = 0.0
    if np.isfinite(row["vol_ratio"]) and np.isfinite(row["mom_1"]):
        expansion = clamp((row["vol_ratio"] - 1.0) / 0.5)
        vol = clamp(np.sign(row["mom_1"]) * abs(expansion) * 0.5)
    scores["volatility"] = vol

    struct_parts = []
    if np.isfinite(row["prev_high"]) and np.isfinite(row["prev_low"]):
        room_up = row["prev_high"] - price
        room_down = price - row["prev_low"]
        total = room_up + room_down
        if total > 0:
            struct_parts.append(clamp((room_down - room_up) / total))
        if np.isfinite(row["atr"]) and row["atr"] > 0:
            pin = 0.25 * row["atr"]
            if 0 < room_up < pin:
                struct_parts.append(-0.6)
            elif 0 < room_down < pin:
                struct_parts.append(0.6)
        if price > row["prev_high"]:
            struct_parts.append(0.45)
        elif price < row["prev_low"]:
            struct_parts.append(-0.45)
    scores["structure"] = clamp(np.mean(struct_parts)) if struct_parts else 0.0
    return scores


def build_samples(code: str, timeframe: str = "h1", horizon_bars: int = 4,
                  max_samples: int = 6000, frame: pd.DataFrame | None = None
                  ) -> list[Sample]:
    """Replay complete bars indexed by availability time.

    Supplied frames must use completion timestamps. Windows with missing bars
    or closures are excluded rather than relabelled as a longer horizon.
    """
    if horizon_bars < 1 or max_samples < 1:
        raise ValueError("horizon_bars and max_samples must be positive")
    a = asset_spec.get(code)
    if frame is None:
        interval, days = {"m15": ("15m", 55), "h1": ("1h", 700),
                          "d1": ("1d", 1100)}[timeframe]
        symbol = a.yahoo_daily if timeframe == "d1" else a.yahoo_intraday
        frame = market_data.drop_closed(market_data._yahoo_bars(symbol, interval, days))
        duration = pd.Timedelta(hours={"m15": 0.25, "h1": 1, "d1": 24}[timeframe])
        frame = frame.copy()
        frame.index = frame.index + duration
        frame = frame.loc[frame.index <= pd.Timestamp.now(tz="UTC")]
    if frame is None or len(frame) < 400:
        return []

    feats = _vectorised_features(frame, code)
    bar_hours = {"m15": 0.25, "h1": 1.0, "d1": 24.0}[timeframe]
    horizon_hours = bar_hours * horizon_bars

    close = feats["close"]
    forward = np.log(close.shift(-horizon_bars) / close)

    # Both vol estimates are per-*bar*; the profile is a share of a *day*, so
    # both have to be lifted to daily terms before being scaled by the variance
    # fraction. Scaling only one of them (easy to do, hard to see) would put a
    # systematic bias straight into the vol ratio and therefore into RANGE.
    to_daily = np.sqrt(24.0 / bar_hours)
    sigma_daily_fast = feats["sigma_fast"].values * to_daily
    sigma_daily_slow = feats["sigma_slow"].values * to_daily
    spread_floor = feats["spread_floor"].values

    samples: list[Sample] = []
    rows = feats.iloc[200:]
    stride = max(1, len(rows) // max_samples)
    for i in range(0, len(rows), stride):
        idx = rows.index[i]
        pos = feats.index.get_loc(idx)
        fwd = forward.iloc[pos]
        row = feats.iloc[pos]
        if not np.isfinite(fwd) or not np.isfinite(row["sigma_slow"]) \
                or not np.isfinite(row["rsi"]) or not np.isfinite(sigma_daily_fast[pos]):
            continue
        # Fit the intraday profile only on information available at this bar.
        # Daily bars cannot identify an intraday activity profile.
        profile = (pd.Series(1 / 24, index=range(24)) if timeframe == "d1"
                   else technical.hour_profile(frame.iloc[:pos + 1]))
        outcome_ts = idx + pd.Timedelta(hours=horizon_hours)
        if frame.index[pos + horizon_bars] != outcome_ts:
            continue
        frac = technical.variance_fraction(profile, idx, outcome_ts)
        if frac <= 0:
            continue
        sigma_h = float(sigma_daily_fast[pos] * np.sqrt(frac))
        if sigma_h <= 0:
            continue
        band = float(max(config.BAND_K * sigma_daily_slow[pos] * np.sqrt(frac),
                         config.BAND_MIN_SPREADS * spread_floor[pos]))
        samples.append(Sample(
            ts=idx, price=float(row["close"]), scores=_score_row(row),
            regime=str(row["regime"]),
            vol_state=("high_volatility" if row["vol_ratio"] >= regime_mod.VOL_HIGH
                       else "low_volatility" if row["vol_ratio"] <= regime_mod.VOL_LOW
                       else "normal"),
            session=_session_name(int(row["hour"])), hour=int(row["hour"]),
            sigma=sigma_h, band=band, nu=float(row["nu"]),
            vol_ratio=float(row["vol_ratio"]) if np.isfinite(row["vol_ratio"]) else 1.0,
            forward_return=float(fwd), actual=realized_class(float(fwd), band),
            outcome_ts=outcome_ts))
    return samples


def _session_name(hour: int) -> str:
    for name, (start, end) in regime_mod.SESSIONS.items():
        if start <= hour < end:
            return name
    return "Closed"


# --------------------------------------------------------------------------
# turning samples into probabilities
# --------------------------------------------------------------------------

def predict_sample(s: Sample, weights: dict[str, float], kappa: float,
                   horizon_key: str = "1h",
                   analogues: dict | None = None) -> dict[str, float]:
    """The engine's probability step, applied to one replayed sample."""
    live = {k: weights.get(k, 0.0) for k in REPLAYABLE}
    total = sum(live.values())
    if total <= 0:
        score = 0.0
    else:
        score = clamp(sum(s.scores[k] * w for k, w in live.items()) / total)

    if analogues:
        analog = analogues.get(s.regime)
        if analog is not None:
            score = ((1 - config.ANALOG_WEIGHT) * score
                     + config.ANALOG_WEIGHT * clamp((analog[0] - analog[1]) * 4))

    # Availability is 4/8 of the model in replay, so confidence is capped the
    # same way the live engine would cap it with those four sources missing.
    availability = sum(config.normalised_base_weights()[k] for k in REPLAYABLE)
    agreement = _agreement_of(s.scores, live)
    conf, _ = engine.compute_confidence(availability, agreement, s.vol_ratio,
                                        "none", "live", None)
    drift = kappa * (conf / 100.0) * config.KAPPA_HORIZON.get(horizon_key, 1.0) * score * s.sigma
    up, down, rng = engine._probabilities(s.band, s.sigma, drift, s.nu)
    return {"UP": up, "DOWN": down, "RANGE": rng, "confidence": conf, "score": score}


def _agreement_of(scores: dict[str, float], weights: dict[str, float]) -> float:
    pairs = [(scores[k], w) for k, w in weights.items() if w > 0]
    if len(pairs) < 2:
        return 0.5
    total = sum(w for _, w in pairs)
    mean = sum(s * w for s, w in pairs) / total
    var = sum(w * (s - mean) ** 2 for s, w in pairs) / total
    pos = sum(w for s, w in pairs if s > 0.02)
    neg = sum(w for s, w in pairs if s < -0.02)
    minority = min(pos, neg) / total if (pos + neg) > 0 else 0.0
    return float(np.clip(1 - np.sqrt(var) / 0.60, 0, 1) * (1 - minority))


def _analogue_map(samples: list[Sample]) -> dict:
    """In-sample base rates per regime, for the analogue term. Train-set only."""
    base = float(np.mean([s.forward_return > 0 for s in samples])) if samples else 0.5
    out: dict = {}
    for regime in set(s.regime for s in samples):
        subset = [s for s in samples if s.regime == regime]
        if len(subset) >= 40:
            out[regime] = (float(np.mean([s.forward_return > 0 for s in subset])), base)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

@dataclass
class Report:
    code: str
    timeframe: str
    horizon_bars: int
    horizon_label: str
    n: int
    period: tuple[pd.Timestamp, pd.Timestamp]
    metrics: dict
    baseline: dict
    calibration: pd.DataFrame
    by_regime: pd.DataFrame
    by_session: pd.DataFrame
    class_report: pd.DataFrame
    trading: dict
    weights: dict
    kappa: float
    note: str = ""
    extras: dict = field(default_factory=dict)

    def summary_text(self) -> str:
        m, b = self.metrics, self.baseline
        lines = [
            f"{self.code} · {self.horizon_label} · {self.n:,} samples "
            f"({self.period[0]:%Y-%m-%d} → {self.period[1]:%Y-%m-%d})",
            f"  Brier      {m['brier']:.4f}   (climatology {b['brier']:.4f}, "
            f"skill {m['brier_skill']:+.1%})",
            f"  Log loss   {m['log_loss']:.4f}   (climatology {b['log_loss']:.4f})",
            f"  Accuracy   {m['accuracy']:.1%}    (climatology {b['accuracy']:.1%})",
            f"  Directional accuracy {m['directional_accuracy']:.1%} "
            f"on {m['directional_n']:,} non-RANGE calls",
            f"  Calibration error (mean |observed - forecast|) {m['calibration_error']:.3f}",
        ]
        if self.trading:
            t = self.trading
            lines.append(f"  Trading sim: {t['n_trades']:,} trades · win rate "
                         f"{t['win_rate']:.1%} · total {t['total_return']:+.2%} · "
                         f"max drawdown {t['max_drawdown']:.2%}")
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


def _metrics(samples: list[Sample], probs: list[dict]) -> dict:
    briers = [brier({c: p[c] for c in CLASSES}, s.actual) for s, p in zip(samples, probs)]
    losses = [log_loss({c: p[c] for c in CLASSES}, s.actual) for s, p in zip(samples, probs)]
    predicted = [max(CLASSES, key=lambda c: p[c]) for p in probs]
    actual = [s.actual for s in samples]
    correct = [p == a for p, a in zip(predicted, actual)]

    directional = [(p, a) for p, a in zip(predicted, actual) if p in ("UP", "DOWN")]
    dir_correct = [p == a for p, a in directional]

    return {
        "brier": float(np.mean(briers)), "log_loss": float(np.mean(losses)),
        "accuracy": float(np.mean(correct)),
        "directional_accuracy": float(np.mean(dir_correct)) if dir_correct else float("nan"),
        "directional_n": len(directional),
        "mean_up": float(np.mean([p["UP"] for p in probs])),
        "mean_confidence": float(np.mean([p["confidence"] for p in probs])),
    }


def _baseline(samples: list[Sample], training: list[Sample] | None = None) -> dict:
    """Evaluate class frequencies fitted on the training window."""
    fitted = samples if training is None else training
    freq = {c: float(np.mean([s.actual == c for s in fitted])) for c in CLASSES}
    briers = [brier(freq, s.actual) for s in samples]
    losses = [log_loss(freq, s.actual) for s in samples]
    top = max(CLASSES, key=lambda c: freq[c])
    return {"brier": float(np.mean(briers)), "log_loss": float(np.mean(losses)),
            "accuracy": float(np.mean([s.actual == top for s in samples])),
            "frequencies": freq}


def _calibration(samples: list[Sample], probs: list[dict], bins: int = 10) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        for s, p in zip(samples, probs):
            rows.append({"class": cls, "p": p[cls], "hit": float(s.actual == cls)})
    long = pd.DataFrame(rows)
    edges = np.linspace(0, 1, bins + 1)
    long["bucket"] = pd.cut(long["p"], edges, include_lowest=True)
    out = long.groupby("bucket", observed=True).agg(
        n=("p", "size"), mean_forecast=("p", "mean"),
        observed=("hit", "mean")).reset_index()
    out["gap"] = out["observed"] - out["mean_forecast"]
    return out


def _class_report(samples: list[Sample], probs: list[dict]) -> pd.DataFrame:
    predicted = [max(CLASSES, key=lambda c: p[c]) for p in probs]
    actual = [s.actual for s in samples]
    rows = []
    for cls in CLASSES:
        tp = sum(1 for p, a in zip(predicted, actual) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(predicted, actual) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(predicted, actual) if p != cls and a == cls)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and np.isfinite(precision) and np.isfinite(recall)
              else float("nan"))
        rows.append({"class": cls, "predicted": tp + fp, "actual": tp + fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def _grouped(samples: list[Sample], probs: list[dict], key,
             training: list[Sample]) -> pd.DataFrame:
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        buckets.setdefault(key(s), []).append(i)
    rows = []
    for name, idx in sorted(buckets.items()):
        if len(idx) < 25:
            continue
        sub_s = [samples[i] for i in idx]
        sub_p = [probs[i] for i in idx]
        m = _metrics(sub_s, sub_p)
        b = _baseline(sub_s, training)
        rows.append({"group": name, "n": len(idx), "brier": m["brier"],
                     "baseline_brier": b["brier"],
                     "skill": 1 - m["brier"] / b["brier"] if b["brier"] else np.nan,
                     "accuracy": m["accuracy"],
                     "directional_accuracy": m["directional_accuracy"]})
    return pd.DataFrame(rows)


def _trading_sim(samples: list[Sample], probs: list[dict], code: str,
                 min_confidence: float = 0.0, edge: float = 0.05,
                 step: int = 1) -> dict:
    """Take the argmax when it is directional and clears an edge threshold.

    Deliberately crude — one unit per signal, no compounding, costed at the
    asset's round-trip spread. It exists to produce the win rate and maximum
    drawdown the brief asks for, not to be a strategy.

    `step` thins the sample so trades do not overlap. Without it a 4-hour
    horizon evaluated every hour books each move four times, which flatters
    the return and hides the true drawdown behind averaging.
    """
    a = asset_spec.get(code)
    pnl: list[float] = []
    for s, p in list(zip(samples, probs))[::max(1, step)]:
        if p["confidence"] < min_confidence:
            continue
        side = 1 if p["UP"] - p["DOWN"] > edge else (-1 if p["DOWN"] - p["UP"] > edge else 0)
        if side == 0:
            continue
        cost = a.spread / s.price
        pnl.append(side * s.forward_return - cost)
    if not pnl:
        return {}
    arr = np.array(pnl)
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    return {"n_trades": len(arr), "win_rate": float((arr > 0).mean()),
            "total_return": float(arr.sum()), "mean_return": float(arr.mean()),
            "max_drawdown": float((equity - peak).min()),
            "profit_factor": float(arr[arr > 0].sum() / -arr[arr < 0].sum())
            if (arr < 0).any() else float("inf")}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

HORIZON_PRESETS = {
    "15m": ("m15", 1), "1h": ("h1", 1), "4h": ("h1", 4),
    "session": ("h1", 8), "24h": ("h1", 24),
}


def _split_samples(samples: list[Sample], train_fraction: float):
    """Purge training labels whose outcomes reach the held-out window."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(len(samples) * train_fraction)
    test = samples[split:]
    if not test:
        raise ValueError("empty evaluation window")
    train = [s for s in samples[:split]
             if s.outcome_ts is not None and s.outcome_ts < test[0].ts]
    if not train:
        raise ValueError("empty training window after purging overlapping labels")
    return train, test


def run(code: str, horizon: str = "4h", weights: dict[str, float] | None = None,
        kappa: float | None = None, train_fraction: float = 0.7,
        max_samples: int = 6000, min_confidence: float = 0.0,
        frame: pd.DataFrame | None = None) -> Report:
    """Walk-forward replay of the chart-derived engine over one horizon."""
    timeframe, bars = HORIZON_PRESETS.get(horizon, ("h1", 4))
    samples = build_samples(code, timeframe, bars, max_samples, frame=frame)
    if len(samples) < 200:
        raise ValueError(f"only {len(samples)} usable samples for {code} {horizon} — "
                         f"not enough history to say anything")

    train, test = _split_samples(samples, train_fraction)
    analogues = _analogue_map(train)     # fitted on train only

    weights = weights or {k: config.normalised_base_weights()[k] for k in REPLAYABLE}
    kappa = config.KAPPA_MAX if kappa is None else kappa

    probs = [predict_sample(s, weights, kappa, horizon, analogues) for s in test]
    m = _metrics(test, probs)
    b = _baseline(test, train)
    m["brier_skill"] = 1 - m["brier"] / b["brier"] if b["brier"] else float("nan")
    cal = _calibration(test, probs)
    m["calibration_error"] = float(np.average(cal["gap"].abs(), weights=cal["n"]))

    # How many samples apart two non-overlapping trades must be.
    bar_hours = {"m15": 0.25, "h1": 1.0, "d1": 24.0}[timeframe]
    spacing_h = ((test[1].ts - test[0].ts).total_seconds() / 3600
                 if len(test) > 1 else bar_hours)
    step = max(1, int(round(bars * bar_hours / max(spacing_h, 1e-6))))

    return Report(
        code=code, timeframe=timeframe, horizon_bars=bars,
        horizon_label=f"{horizon} horizon ({bars} x {timeframe} bars)",
        n=len(test), period=(test[0].ts, test[-1].ts),
        metrics=m, baseline=b, calibration=cal,
        by_regime=_grouped(test, probs, lambda s: s.regime, train),
        by_session=_grouped(test, probs, lambda s: s.session, train),
        class_report=_class_report(test, probs),
        trading=_trading_sim(test, probs, code, min_confidence, step=step),
        weights=weights, kappa=kappa,
        note=("Replays technical / price action / volatility / structure only — "
              "macro, news and Polymarket have no point-in-time archive and are "
              "excluded rather than backfilled."),
        extras={"n_train": len(train), "analogue_regimes": list(analogues)})


# --------------------------------------------------------------------------
# weight optimisation
# --------------------------------------------------------------------------

def optimise(code: str, horizon: str = "4h", trials: int = 300, seed: int = 7,
             train_fraction: float = 0.7, max_samples: int = 6000) -> dict:
    """Random search over the replayable weights and kappa, scored out-of-sample.

    Random rather than grid because four weights plus kappa is a five-dimensional
    space where a grid coarse enough to finish is too coarse to find anything.
    Fitted on the training slice, reported on the held-out slice, and the search
    is over *relative* weights within the replayable four — the four weights the
    replay cannot see are left at their defaults, because there is no honest way
    to fit them here.
    """
    timeframe, bars = HORIZON_PRESETS.get(horizon, ("h1", 4))
    samples = build_samples(code, timeframe, bars, max_samples)
    if len(samples) < 300:
        raise ValueError(f"not enough samples ({len(samples)}) to optimise {code}")

    train, test = _split_samples(samples, train_fraction)
    analogues = _analogue_map(train)
    rng = np.random.default_rng(seed)

    base = {k: config.normalised_base_weights()[k] for k in REPLAYABLE}
    best = {"weights": base, "kappa": config.KAPPA_MAX, "train_brier": np.inf}

    for _ in range(trials):
        raw = rng.dirichlet(np.ones(len(REPLAYABLE)) * 1.4)
        cand = dict(zip(REPLAYABLE, raw))
        kappa = float(rng.uniform(0.0, config.KAPPA_MAX))
        probs = [predict_sample(s, cand, kappa, horizon, analogues) for s in train]
        score = float(np.mean([brier({c: p[c] for c in CLASSES}, s.actual)
                               for s, p in zip(train, probs)]))
        if score < best["train_brier"]:
            best = {"weights": cand, "kappa": kappa, "train_brier": score}

    def _eval(w, k, data):
        probs = [predict_sample(s, w, k, horizon, analogues) for s in data]
        return float(np.mean([brier({c: p[c] for c in CLASSES}, s.actual)
                              for s, p in zip(data, probs)]))

    result = {
        "code": code, "horizon": horizon,
        "weights": {k: round(v, 4) for k, v in best["weights"].items()},
        "kappa": round(best["kappa"], 4),
        "train_brier": round(best["train_brier"], 5),
        "test_brier": round(_eval(best["weights"], best["kappa"], test), 5),
        "default_test_brier": round(_eval(base, config.KAPPA_MAX, test), 5),
        "n_train": len(train), "n_test": len(test), "trials": trials,
    }
    result["improved"] = result["test_brier"] < result["default_test_brier"]
    return result


def to_overrides(results: list[dict]) -> dict:
    """Turn optimiser output into a `weights.json` payload.

    The optimised relative weights are scaled back into the full eight-component
    vector so the four unfitted channels keep their intended share — otherwise
    "optimising" would quietly delete macro, news and Polymarket from the model.
    """
    usable = [r for r in results if r.get("improved")]
    if not usable:
        return {}
    base = config.normalised_base_weights()
    replay_share = sum(base[k] for k in REPLAYABLE)
    mean_w = {k: float(np.mean([r["weights"][k] for r in usable])) for k in REPLAYABLE}
    total = sum(mean_w.values())
    scaled = {k: v / total * replay_share for k, v in mean_w.items()}
    out = {**base, **scaled}
    return {"BASE_WEIGHTS": {k: round(v, 4) for k, v in out.items()},
            "KAPPA_MAX": round(float(np.mean([r["kappa"] for r in usable])), 4)}
