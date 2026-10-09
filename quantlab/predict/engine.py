"""The probability engine: evidence in, three calibrated probabilities out.

The whole model in one page:

    band        RANGE is defined, not vibed: |return| <= band_k * sigma_slow(h),
                where sigma_slow comes from ATR(14). Because the *predictive*
                sigma comes from fast EWMA vol, an expansion of realised vol
                against its own baseline shrinks RANGE mechanically.

    shape       returns ~ mu + sigma_fast(h) * T(nu), a standardised Student-t
                with nu from realised kurtosis. Fat tails are not decoration:
                a Gaussian materially understates P(UP)+P(DOWN) on these assets.

    time        sigma(h) = sigma_daily * sqrt(variance fraction of [now, now+h]),
                taken from the hour-of-day variance profile, so the 15-minute
                question is different at the New York open than at 03:00 UTC,
                and a horizon spanning the weekend carries almost no variance.

    direction   enters ONLY as mu = kappa * score * sigma(h). kappa is capped
                at 0.30, scaled by model confidence, by horizon, and cut again
                into a scheduled event. Maxed out, that moves P(UP) from about
                31% to about 40% — never to 87%.

    score       weighted blend of the eight components, where the weights are
                base * horizon tilt * regime tilt, renormalised over the
                components that actually had data, then nudged toward the
                historical analogue for this regime and time of day.

Every number the engine prints can be traced back to a fetched observation.
Where an input is missing the model says so, redistributes its weight, and
lowers its own confidence — it never substitutes a neutral guess for evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.predict import (assets as asset_spec, config, macro_data,
                              market_data, news_data, polymarket_data, quality,
                              regime as regime_mod, technical)
from quantlab.predict.component import ComponentScore, clamp, missing

_t_cache: dict[float, np.ndarray] = {}


# --------------------------------------------------------------------------
# standardised Student-t
# --------------------------------------------------------------------------

def _t_sample(nu: float) -> np.ndarray:
    """Sorted sample of a unit-variance Student-t, cached per rounded nu.

    A fixed seed matters more than it looks: without it, two consecutive runs
    would differ by Monte-Carlo noise and the "probability changed because..."
    explanation would be narrating resampling error as market information.
    """
    key = round(float(np.clip(nu, config.NU_MIN, config.NU_MAX)), 1)
    cached = _t_cache.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(config.MC_SEED)
    z = rng.standard_t(key, config.MC_DRAWS)
    z /= np.sqrt(key / (key - 2))          # unit variance
    z.sort()
    _t_cache[key] = z
    return z


def t_cdf(x: float, nu: float) -> float:
    """P(T <= x) for the standardised t, by empirical CDF of the cached sample."""
    if not np.isfinite(x):
        return 0.5
    sample = _t_sample(nu)
    return float(np.searchsorted(sample, x, side="right") / sample.size)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class HorizonPrediction:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    hours: float
    up: float
    down: float
    range_: float
    confidence: float
    confidence_label: str
    band: float          # log-return half-width of the RANGE band
    band_price: float    # the same band expressed in price units
    sigma: float
    drift: float
    score: float
    variance_fraction: float
    weights: dict[str, float]
    contributions: dict[str, float]
    event_level: str
    event_names: list[str]
    analogue: regime_mod.Analogue | None
    poly_influence_pts: float = 0.0
    confidence_parts: dict[str, float] = field(default_factory=dict)
    anchored_at_open: bool = False
    explanation: list[str] = field(default_factory=list)
    prev_up: float | None = None

    @property
    def up_pct(self) -> float:
        return self.up * 100

    @property
    def down_pct(self) -> float:
        return self.down * 100

    @property
    def range_pct(self) -> float:
        return self.range_ * 100

    def rounded(self) -> tuple[int, int, int]:
        """Whole percentages that still sum to exactly 100.

        Largest-remainder, because three independently rounded numbers summing
        to 99 or 101 would quietly break the one guarantee the output makes.
        """
        raw = [self.up * 100, self.down * 100, self.range_ * 100]
        floors = [int(np.floor(v)) for v in raw]
        remainder = 100 - sum(floors)
        order = np.argsort([-(v - f) for v, f in zip(raw, floors)])
        for i in range(remainder):
            floors[order[i % 3]] += 1
        return tuple(floors)

    def to_dict(self) -> dict:
        up, down, rng = self.rounded()
        return {
            "horizon": self.key, "label": self.label,
            "start": str(self.start), "end": str(self.end),
            "hours": round(self.hours, 3),
            "up_pct": up, "down_pct": down, "range_pct": rng,
            "up": round(self.up, 5), "down": round(self.down, 5),
            "range": round(self.range_, 5),
            "confidence": round(self.confidence, 1),
            "confidence_label": self.confidence_label,
            "band": self.band, "band_price": self.band_price,
            "sigma": self.sigma, "drift": self.drift,
            "score": round(self.score, 4),
            "variance_fraction": round(self.variance_fraction, 5),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "event_level": self.event_level, "event_names": self.event_names,
            "poly_influence_pts": round(self.poly_influence_pts, 2),
            "confidence_parts": {k: round(v, 3) for k, v in self.confidence_parts.items()},
            "anchored_at_open": self.anchored_at_open,
            "analogue": (None if self.analogue is None or not self.analogue.available
                         else {"up_rate": round(self.analogue.up_rate, 4),
                               "n": self.analogue.n,
                               "base_rate": round(self.analogue.base_rate, 4),
                               "description": self.analogue.description}),
            "explanation": self.explanation,
            "prev_up_pct": None if self.prev_up is None else round(self.prev_up * 100, 1),
        }


@dataclass
class Prediction:
    code: str
    name: str
    ts: pd.Timestamp
    price: float
    change: float
    change_pct: float
    price_status: str
    price_source: str
    regime: regime_mod.RegimeState
    tech: technical.TechnicalState
    components: dict[str, ComponentScore]
    horizons: dict[str, HorizonPrediction]
    quality: quality.QualityReport
    risk_events: list[macro_data.Event]
    calendar_problems: list[str]
    model_version: str = config.MODEL_VERSION
    notes: list[str] = field(default_factory=list)

    @property
    def headline(self) -> HorizonPrediction:
        return self.horizons.get("1h") or next(iter(self.horizons.values()))

    def key_drivers(self, limit: int = 5) -> list[str]:
        """The channels doing the most work, phrased from the data itself.

        Named by channel and by that channel's *net* contribution, because a
        single evidence line can point the opposite way to the channel it came
        from — the rates channel can be net bearish while its loudest single
        driver is bullish, and reporting only the line would misattribute it.
        """
        h = self.headline
        ranked = sorted(h.contributions.items(), key=lambda kv: -abs(kv[1]))
        out: list[str] = []
        for name, contribution in ranked:
            if abs(contribution) < 0.005:
                continue
            comp = self.components.get(name)
            if comp is None or not comp.evidence:
                continue
            direction = "bullish" if contribution > 0 else "bearish"
            out.append(f"{name.replace('_', ' ')} → net {direction} "
                       f"{contribution:+.3f} · " + "; ".join(comp.evidence[:2]))
            if len(out) >= limit:
                break
        return out

    def to_dict(self) -> dict:
        return {
            "asset": self.code, "name": self.name, "timestamp": str(self.ts),
            "price": self.price, "change": self.change, "change_pct": self.change_pct,
            "price_status": self.price_status, "price_source": self.price_source,
            "regime": self.regime.to_dict(),
            "trend": self.tech.trend_label, "volatility": self.tech.vol_label,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "horizons": {k: v.to_dict() for k, v in self.horizons.items()},
            "data_quality": self.quality.to_dict(),
            "risk_events": [e.to_dict(self.ts) for e in self.risk_events[:6]],
            "calendar_problems": self.calendar_problems,
            "untracked_events": macro_data.UNTRACKED_EVENTS,
            "key_drivers": self.key_drivers(),
            "model_version": self.model_version,
            "notes": self.notes,
            "indicators": {
                "rsi14_d": _num(self.tech.rsi14_d), "rsi14_h1": _num(self.tech.rsi14_h1),
                "rsi14_m15": _num(self.tech.rsi14_m15),
                "atr14": _num(self.tech.atr14),
                "sigma_ewma_daily_pct": _num(self.tech.sigma_ewma * 100),
                "sigma_atr_daily_pct": _num(self.tech.sigma_atr * 100),
                "vol_ratio": _num(self.tech.vol_ratio),
                "adx14": _num(self.tech.adx14),
                "sma20": _num(self.tech.sma20), "sma50": _num(self.tech.sma50),
                "sma200": _num(self.tech.sma200),
                "ema20": _num(self.tech.ema20), "ema50": _num(self.tech.ema50),
                "day_high": _num(self.tech.day_high), "day_low": _num(self.tech.day_low),
                "prev_high": _num(self.tech.prev_high), "prev_low": _num(self.tech.prev_low),
                "swing_high_20d": _num(self.tech.swing_high_20d),
                "swing_low_20d": _num(self.tech.swing_low_20d),
                "nearest_support": (list(self.tech.nearest_support)
                                    if self.tech.nearest_support else None),
                "nearest_resistance": (list(self.tech.nearest_resistance)
                                       if self.tech.nearest_resistance else None),
                "mom_1d": _num(self.tech.mom_1d), "mom_5d": _num(self.tech.mom_5d),
                "mom_20d": _num(self.tech.mom_20d),
                "volume_z": _num(self.tech.volume_z) if self.tech.volume_z is not None else None,
                "volume_note": self.tech.volume_note,
                "nu": _num(self.tech.nu),
            },
        }


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def resolve_weights(horizon_key: str, tilt_keys: list[str],
                    available: dict[str, bool]) -> tuple[dict[str, float], float]:
    """Base * horizon tilt * regime tilt, renormalised over available inputs.

    Returns the normalised weights and the *availability*: the share of the
    intended model weight that was actually backed by data. That number is the
    single biggest term in model confidence, which is how a missing source
    lowers confidence instead of silently changing the answer.
    """
    base = config.normalised_base_weights()
    htilt = config.HORIZON_TILT.get(horizon_key, {})
    raw: dict[str, float] = {}
    for name, w in base.items():
        value = w * htilt.get(name, 1.0)
        for key in tilt_keys:
            value *= config.REGIME_TILT.get(key, {}).get(name, 1.0)
        raw[name] = value

    total_all = sum(raw.values())
    live = {k: v for k, v in raw.items() if available.get(k, False)}
    total_live = sum(live.values())
    if total_live <= 0:
        return {k: 0.0 for k in base}, 0.0
    return ({k: v / total_live for k, v in live.items()},
            total_live / total_all if total_all > 0 else 0.0)


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------

def _agreement(weights: dict[str, float], comps: dict[str, ComponentScore]) -> float:
    """1.0 when the live components agree, 0.0 when they are at war."""
    pairs = [(comps[k].score, w) for k, w in weights.items() if w > 0 and k in comps]
    if len(pairs) < 2:
        return 0.5
    total = sum(w for _, w in pairs)
    mean = sum(s * w for s, w in pairs) / total
    var = sum(w * (s - mean) ** 2 for s, w in pairs) / total
    dispersion = float(np.sqrt(var))
    # Weight sitting on the minority side of zero is the sharper conflict test.
    pos = sum(w for s, w in pairs if s > 0.02)
    neg = sum(w for s, w in pairs if s < -0.02)
    minority = min(pos, neg) / total if (pos + neg) > 0 else 0.0
    return float(np.clip(1.0 - dispersion / 0.60, 0, 1) * (1.0 - minority))


def _freshness(price_status: str) -> float:
    return {"live": 1.0, "delayed": 0.65, "closed": 0.5,
            "stale": 0.2, "unavailable": 0.0}.get(price_status, 0.4)


def _vol_penalty(vol_ratio: float) -> float:
    if not np.isfinite(vol_ratio) or vol_ratio <= 0:
        return 0.5
    # log-symmetric: 2x and 0.5x are equally unusual.
    return float(np.clip(1.0 - abs(np.log(vol_ratio)) / np.log(2.2), 0, 1))


def compute_confidence(availability: float, agreement: float, vol_ratio: float,
                       event_level: str, price_status: str,
                       track_record: float | None) -> tuple[float, dict[str, float]]:
    parts = {
        "availability": float(np.clip(availability, 0, 1)),
        "agreement": float(np.clip(agreement, 0, 1)),
        "vol_regime": _vol_penalty(vol_ratio),
        "event_risk": {"HIGH": 0.35, "medium": 0.7, "none": 1.0}.get(event_level, 1.0),
        "freshness": _freshness(price_status),
        "track_record": 0.5 if track_record is None else float(np.clip(track_record, 0, 1)),
    }
    score = sum(config.CONFIDENCE_WEIGHTS[k] * v for k, v in parts.items())
    return float(np.clip(score, 0, 100)), parts


def confidence_label(value: float) -> str:
    if value >= config.CONFIDENCE_HIGH:
        return "HIGH"
    if value >= config.CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# the horizon calculation
# --------------------------------------------------------------------------

def _probabilities(band: float, sigma: float, drift: float, nu: float
                   ) -> tuple[float, float, float]:
    """Three-way split of a t-distributed return around a defined band."""
    if not np.isfinite(sigma) or sigma <= 0:
        # No variance in the window (a closed market): the honest answer is
        # that it stays where it is.
        return 0.0, 0.0, 1.0
    z_up = (band - drift) / sigma
    z_dn = (-band - drift) / sigma
    up = 1.0 - t_cdf(z_up, nu)
    down = t_cdf(z_dn, nu)
    rng = max(0.0, 1.0 - up - down)
    total = up + down + rng
    return up / total, down / total, rng / total


def _horizon_window(h: config.Horizon, now: pd.Timestamp,
                    session: regime_mod.SessionState
                    ) -> tuple[pd.Timestamp, pd.Timestamp, bool, str]:
    """Start, end, whether it had to be pushed to the next open, and a label."""
    anchored = False
    start = now
    if not market_data.market_open(now):
        start = market_data.next_open(now)
        anchored = True

    if h.key == "session":
        end = session.next_end
        start = max(start, session.next_start) if anchored else start
        label = f"Next trading session ({session.next_name}, to {session.next_end:%H:%M} UTC)"
        return start, end, anchored, label

    return start, start + pd.Timedelta(hours=h.hours), anchored, h.label


def _build_horizon(h: config.Horizon, now: pd.Timestamp, snap: market_data.MarketSnapshot,
                   ts: technical.TechnicalState, reg: regime_mod.RegimeState,
                   comps: dict[str, ComponentScore], events: list[macro_data.Event],
                   track_record: float | None) -> HorizonPrediction:
    a = asset_spec.get(snap.code)
    start, end, anchored, label = _horizon_window(h, now, reg.session)
    hours = max((end - start).total_seconds() / 3600, 1 / 60)

    # ---- variance time -----------------------------------------------------
    frac = technical.variance_fraction(ts.profile, start, end)
    sigma_fast = ts.sigma_ewma * np.sqrt(frac) if np.isfinite(ts.sigma_ewma) else float("nan")
    sigma_slow = ts.sigma_atr * np.sqrt(frac) if np.isfinite(ts.sigma_atr) else sigma_fast

    # ---- event risk inside this window ------------------------------------
    risk = macro_data.event_risk(macro_data.events_for(snap.code, events), now, end)
    vol_mult = config.EVENT_VOL_MULT.get(risk["level"], 1.0)
    kappa_mult = config.EVENT_KAPPA_MULT.get(risk["level"], 1.0)
    sigma_fast = sigma_fast * vol_mult if np.isfinite(sigma_fast) else sigma_fast

    band = config.BAND_K * sigma_slow if np.isfinite(sigma_slow) else float("nan")
    if np.isfinite(band) and snap.price > 0:
        # Never call a move smaller than the round-trip cost a direction.
        band = max(band, config.BAND_MIN_SPREADS * a.spread / snap.price)

    # ---- weights and blended score ----------------------------------------
    available = {k: v.available for k, v in comps.items()}
    weights, availability = resolve_weights(h.key, reg.tilt_keys, available)
    contributions = {k: weights.get(k, 0.0) * comps[k].score for k in comps}
    score = sum(contributions.values())

    # ---- historical analogue ----------------------------------------------
    frame, bars = _analog_frame(snap, hours)
    analog = None
    if frame is not None and bars >= 1:
        session_hours = (reg.session.start.hour, reg.session.end.hour) \
            if (reg.session.is_open and hours <= 8) else None
        analog = regime_mod.analogue(frame, reg.trend, bars, session_hours)
        if analog.available:
            analog_score = clamp((analog.up_rate - analog.base_rate) * 4)
            score = (1 - config.ANALOG_WEIGHT) * score + config.ANALOG_WEIGHT * analog_score

    score = clamp(score)

    # ---- confidence, then drift (which confidence shrinks) -----------------
    agreement = _agreement(weights, comps)
    conf, conf_parts = compute_confidence(availability, agreement, ts.vol_ratio,
                                          risk["level"], snap.price_status,
                                          track_record)
    kappa = (config.KAPPA_MAX * (conf / 100.0)
             * config.KAPPA_HORIZON.get(h.key, 1.0) * kappa_mult)
    drift = kappa * score * sigma_fast if np.isfinite(sigma_fast) else 0.0

    up, down, rng = _probabilities(band, sigma_fast, drift, ts.nu)

    # ---- how much of that was Polymarket -----------------------------------
    poly_pts = 0.0
    if comps.get("polymarket") is not None and comps["polymarket"].available:
        without = {k: v for k, v in available.items()}
        without["polymarket"] = False
        w2, avail2 = resolve_weights(h.key, reg.tilt_keys, without)
        score2 = clamp(sum(w2.get(k, 0.0) * comps[k].score for k in comps))
        if analog is not None and analog.available:
            analog_score = clamp((analog.up_rate - analog.base_rate) * 4)
            score2 = (1 - config.ANALOG_WEIGHT) * score2 + config.ANALOG_WEIGHT * analog_score
        conf2, _ = compute_confidence(avail2, _agreement(w2, comps), ts.vol_ratio,
                                      risk["level"], snap.price_status, track_record)
        kappa2 = (config.KAPPA_MAX * (conf2 / 100.0)
                  * config.KAPPA_HORIZON.get(h.key, 1.0) * kappa_mult)
        drift2 = kappa2 * score2 * sigma_fast if np.isfinite(sigma_fast) else 0.0
        up2, _, _ = _probabilities(band, sigma_fast, drift2, ts.nu)
        poly_pts = (up - up2) * 100

    return HorizonPrediction(
        key=h.key, label=label, start=start, end=end, hours=hours,
        up=up, down=down, range_=rng,
        confidence=conf, confidence_label=confidence_label(conf),
        band=float(band) if np.isfinite(band) else float("nan"),
        band_price=float(band * snap.price) if np.isfinite(band) else float("nan"),
        sigma=float(sigma_fast) if np.isfinite(sigma_fast) else float("nan"),
        drift=float(drift), score=score, variance_fraction=frac,
        weights=weights, contributions=contributions,
        event_level=risk["level"], event_names=risk["names"],
        analogue=analog, poly_influence_pts=poly_pts,
        confidence_parts=conf_parts, anchored_at_open=anchored)


def _analog_frame(snap: market_data.MarketSnapshot, hours: float):
    """Pick the timeframe whose bar size suits this horizon, and the bar count."""
    if hours <= 0.75:
        frame = snap.bars.get("m15")
        return frame, max(1, int(round(hours * 4)))
    frame = snap.bars.get("h1")
    if frame is None or len(frame) < 300:
        frame = snap.bars.get("d1")
        return frame, max(1, int(round(hours / 24)))
    return frame, max(1, int(round(hours)))


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------

def explain(current: Prediction, previous: dict | None) -> None:
    """Fill in each horizon's explanation, from data only.

    With a previous run to compare against, the explanation is a genuine diff:
    which component contributions moved, and what underlying reading moved
    them. Without one, it states what is carrying the current call. Nothing
    here is generated from a template of plausible-sounding market reasons.
    """
    for key, hp in current.horizons.items():
        lines: list[str] = []
        prev_h = None
        if previous:
            prev_h = (previous.get("horizons") or {}).get(key)

        if prev_h:
            prev_up = float(prev_h.get("up", np.nan))
            hp.prev_up = prev_up if np.isfinite(prev_up) else None
            delta_pts = (hp.up - prev_up) * 100 if np.isfinite(prev_up) else 0.0
            prev_contrib = prev_h.get("contributions") or {}
            moves = []
            for name, now_c in hp.contributions.items():
                before = float(prev_contrib.get(name, 0.0))
                if abs(now_c - before) >= config.EXPLAIN_MIN_DELTA:
                    moves.append((abs(now_c - before), name, before, now_c))
            moves.sort(reverse=True)

            if abs(delta_pts) < config.EXPLAIN_MIN_PROB_PTS and not moves:
                lines.append(f"P(UP) unchanged at {hp.up * 100:.0f}% — no input moved "
                             f"enough to shift it since the last update.")
            else:
                if abs(delta_pts) < config.EXPLAIN_MIN_PROB_PTS:
                    # Inputs moved but cancelled out; saying "because" here would
                    # claim a change the number does not show.
                    lines.append(f"P(UP) held at {hp.up * 100:.0f}% — inputs moved "
                                 f"but offset each other:")
                else:
                    lines.append(f"P(UP) {prev_up * 100:.0f}% → {hp.up * 100:.0f}% "
                                 f"({delta_pts:+.0f} pts) because:")
                for _, name, before, now_c in moves[:4]:
                    comp = current.components.get(name)
                    detail = comp.evidence[0] if (comp and comp.evidence) else comp.note if comp else ""
                    lines.append(f"• {name.replace('_', ' ')} contribution "
                                 f"{before:+.3f} → {now_c:+.3f} — {detail}")
                prev_conf = float(prev_h.get("confidence", hp.confidence))
                if abs(hp.confidence - prev_conf) >= 5:
                    lines.append(f"• model confidence {prev_conf:.0f} → "
                                 f"{hp.confidence:.0f}/100")
                prev_events = set(prev_h.get("event_names") or [])
                if set(hp.event_names) - prev_events:
                    lines.append("• newly inside the window: "
                                 + ", ".join(sorted(set(hp.event_names) - prev_events)))
        else:
            ranked = sorted(hp.contributions.items(), key=lambda kv: -abs(kv[1]))
            lines.append(f"First reading for this horizon — P(UP) {hp.up * 100:.0f}%, "
                         f"carried by:")
            for name, contribution in ranked[:3]:
                if abs(contribution) < 0.005:
                    continue
                comp = current.components.get(name)
                detail = comp.evidence[0] if (comp and comp.evidence) else ""
                lines.append(f"• {name.replace('_', ' ')} {contribution:+.3f} — {detail}")

        missing_now = [c.name for c in current.components.values() if not c.available]
        if missing_now:
            lines.append("• not used this run (no data): " + ", ".join(missing_now))
        hp.explanation = lines


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def predict(code: str, now: pd.Timestamp | None = None,
            previous: dict | None = None,
            universe: list | None = None,
            events: list[macro_data.Event] | None = None,
            calendar_problems: list[str] | None = None,
            use_twelvedata: bool = True,
            track_record: dict[str, float] | None = None) -> Prediction:
    """One full prediction for one asset, across every horizon.

    Shared inputs (the Polymarket universe, the calendar) can be passed in so a
    multi-asset run fetches them once instead of three times.
    """
    a = asset_spec.get(code)
    now = now or pd.Timestamp.now(tz="UTC")

    snap = market_data.get_snapshot(a.code, now, use_twelvedata=use_twelvedata)
    ts = technical.compute(snap)

    if events is None:
        events, calendar_problems = macro_data.calendar(now)
    calendar_problems = calendar_problems or []

    asset_events = macro_data.events_for(a.code, events)
    day_risk = macro_data.event_risk(asset_events, now, now + pd.Timedelta(hours=24))
    reg = regime_mod.classify(ts, now, day_risk["level"])

    comps: dict[str, ComponentScore] = {
        "technical": technical.score_technical(ts),
        "price_action": technical.score_price_action(ts),
        "volatility": technical.score_volatility(ts),
        "structure": technical.score_structure(ts),
    }
    comps["macro"] = _safe(lambda: macro_data.score_macro(a.code), "macro")
    comps["rates"] = _safe(lambda: macro_data.score_rates(a.code), "rates")
    comps["news"] = _safe(lambda: news_data.score_news(a.code, now), "news")
    comps["polymarket"] = _safe(
        lambda: polymarket_data.score_polymarket(a.code, now, universe), "polymarket")

    # ---- data quality ------------------------------------------------------
    report = quality.QualityReport()
    for status in snap.statuses:
        report.add(status)
    for name, comp in comps.items():
        if name in ("technical", "price_action", "volatility", "structure"):
            continue
        report.add(quality.SourceStatus(
            {"macro": "Macro (FRED)", "rates": "Rate expectations (FRED)",
             "news": "News sentiment", "polymarket": "Polymarket"}[name],
            comp.status if comp.available else "unavailable",
            comp.note or (comp.evidence[0][:80] if comp.evidence else "")))
    report.add(quality.SourceStatus(
        "Economic calendar",
        "live" if events else "unavailable",
        f"{len(asset_events)} scheduled release(s) tracked in the next "
        f"{macro_data.CALENDAR_DAYS} days"
        + ("; " + "; ".join(calendar_problems[:2]) if calendar_problems else "")))

    base_w = config.normalised_base_weights()
    availability = sum(base_w[k] for k, c in comps.items() if c.available)
    quality.grade(report, availability, snap.price_status,
                  config.QUALITY_GOOD, config.QUALITY_LIMITED)

    # ---- horizons ----------------------------------------------------------
    horizons: dict[str, HorizonPrediction] = {}
    if snap.ok:
        for h in config.HORIZONS:
            tr = (track_record or {}).get(h.key)
            horizons[h.key] = _build_horizon(h, now, snap, ts, reg, comps,
                                             events, tr)

    notes: list[str] = []
    if not snap.ok:
        notes.append("LIVE DATA UNAVAILABLE — no usable price or history this run; "
                     "no probabilities were produced.")
    if not reg.session.is_open:
        notes.append(f"Market is closed (weekend). Horizons are anchored at the next "
                     f"open, {market_data.next_open(now):%a %d %b %H:%M} UTC.")
    if snap.feed_offset_hours:
        notes.append(f"Twelve Data bar timestamps corrected by "
                     f"{snap.feed_offset_hours:+.0f}h onto UTC.")
    if snap.basis:
        notes.append(f"GC=F futures levels shifted onto spot by a "
                     f"{snap.basis:+,.2f} basis.")
    if not a.has_volume:
        notes.append("No tick or contract volume exists behind spot FX bars, so "
                     "volume confirmation is not part of this asset's read.")

    prediction = Prediction(
        code=a.code, name=a.name, ts=now, price=snap.price, change=snap.change,
        change_pct=snap.change_pct, price_status=snap.price_status,
        price_source=snap.price_source, regime=reg, tech=ts, components=comps,
        horizons=horizons, quality=report, risk_events=asset_events,
        calendar_problems=calendar_problems, notes=notes)

    explain(prediction, previous)
    return prediction


def _safe(fn, name: str) -> ComponentScore:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — one dead source must not kill a run
        return missing(name, f"{type(exc).__name__}: {str(exc)[:90]}")


def predict_all(codes: list[str] | None = None, now: pd.Timestamp | None = None,
                previous: dict[str, dict] | None = None,
                use_twelvedata: bool = True,
                track_record: dict[str, dict[str, float]] | None = None
                ) -> dict[str, Prediction]:
    """Every asset, sharing one Polymarket fetch and one calendar fetch."""
    codes = codes or asset_spec.DEFAULT_CODES
    now = now or pd.Timestamp.now(tz="UTC")

    try:
        universe = polymarket_data.fetch_universe()
    except Exception:
        universe = []
    events, problems = macro_data.calendar(now)

    out: dict[str, Prediction] = {}
    for code in codes:
        out[code] = predict(
            code, now=now, previous=(previous or {}).get(code),
            universe=universe or None, events=events, calendar_problems=problems,
            use_twelvedata=use_twelvedata,
            track_record=(track_record or {}).get(code))
    return out
