"""Tunable configuration for the probability engine.

Nothing here is a permanent constant. Every number below is a *default* that
`weights.json` (written by the backtester's optimiser, or by hand) can override
at import time, so the model can be re-fit without editing code:

    from quantlab.predict import config
    config.load_overrides()            # called automatically on import
    config.save_overrides({"kappa_max": 0.22})

The defaults are deliberately conservative. `kappa_max` in particular is the
single most important anti-overconfidence knob: it caps how far a perfect
signal is allowed to shift the distribution, in units of the horizon's own
standard deviation. At 0.30 a maxed-out score moves P(UP) from roughly 31% to
roughly 40% — a real edge, and nothing like a certainty.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

MODEL_VERSION = "predict-1.0.0"

_HERE = Path(__file__).resolve().parent
OVERRIDES_PATH = _HERE / "weights.json"


# --------------------------------------------------------------------------
# horizons
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Horizon:
    key: str
    label: str
    hours: float | None   # None = variable length, resolved at runtime (session)


HORIZONS: list[Horizon] = [
    Horizon("15m", "Next 15 minutes", 0.25),
    Horizon("1h", "Next 1 hour", 1.0),
    Horizon("4h", "Next 4 hours", 4.0),
    Horizon("session", "Next trading session", None),
    Horizon("24h", "Next 24 hours", 24.0),
]

HORIZON_KEYS = [h.key for h in HORIZONS]


# --------------------------------------------------------------------------
# signal components
# --------------------------------------------------------------------------

#: The eight evidence channels. Each scores in [-1, +1] (negative = bearish)
#: and declares whether it actually had data this run.
COMPONENTS = (
    "technical",
    "price_action",
    "macro",
    "news",
    "rates",
    "polymarket",
    "volatility",
    "structure",
)

#: Baseline blend, before horizon and regime tilts. Sums to 1.0.
BASE_WEIGHTS: dict[str, float] = {
    "technical": 0.20,
    "price_action": 0.15,
    "macro": 0.20,
    "news": 0.10,
    "rates": 0.10,
    "polymarket": 0.10,
    "volatility": 0.10,
    "structure": 0.05,
}

#: Multipliers applied per horizon, then renormalised. Fifteen minutes from now
#: is a microstructure question; 24 hours from now is a macro question. A CPI
#: print does not tell you where the next three candles go, and the position of
#: price inside today's range tells you very little about tomorrow.
HORIZON_TILT: dict[str, dict[str, float]] = {
    "15m": {"technical": 1.30, "price_action": 1.60, "structure": 1.50,
            "volatility": 1.20, "macro": 0.30, "news": 0.40,
            "rates": 0.30, "polymarket": 0.30},
    "1h": {"technical": 1.25, "price_action": 1.35, "structure": 1.30,
           "volatility": 1.10, "macro": 0.55, "news": 0.70,
           "rates": 0.55, "polymarket": 0.50},
    "4h": {"technical": 1.10, "price_action": 1.00, "structure": 1.00,
           "volatility": 1.00, "macro": 0.90, "news": 1.00,
           "rates": 0.90, "polymarket": 0.85},
    "session": {"technical": 1.00, "price_action": 0.85, "structure": 0.90,
                "volatility": 0.95, "macro": 1.15, "news": 1.10,
                "rates": 1.15, "polymarket": 1.10},
    "24h": {"technical": 0.90, "price_action": 0.60, "structure": 0.70,
            "volatility": 0.90, "macro": 1.45, "news": 1.15,
            "rates": 1.45, "polymarket": 1.40},
}

#: Multipliers applied per market regime, then renormalised. In a range, trend
#: indicators are noise and the edges of the range are the information; in a
#: strong trend the opposite holds and fading structure is how accounts die.
REGIME_TILT: dict[str, dict[str, float]] = {
    "strong_bullish": {"technical": 1.25, "price_action": 1.25, "structure": 0.55},
    "weak_bullish": {"technical": 1.10, "price_action": 1.05, "structure": 0.85},
    "range": {"technical": 0.65, "price_action": 0.85, "structure": 1.70,
              "volatility": 1.10},
    "weak_bearish": {"technical": 1.10, "price_action": 1.05, "structure": 0.85},
    "strong_bearish": {"technical": 1.25, "price_action": 1.25, "structure": 0.55},
    "high_volatility": {"technical": 0.80, "price_action": 0.90, "structure": 0.70,
                        "macro": 1.25, "news": 1.30, "polymarket": 1.20},
    "low_volatility": {"technical": 0.90, "price_action": 0.95, "structure": 1.35},
}


# --------------------------------------------------------------------------
# distribution and calibration
# --------------------------------------------------------------------------

#: How wide the RANGE band is, in units of the *baseline* (ATR-implied) sigma
#: for the horizon. 0.5 puts roughly 38% of a driftless normal inside it.
BAND_K = 0.50

#: Floor on the band, in round-trip spreads. A move smaller than the cost of
#: taking the trade is not a direction, it is noise — this matters at the
#: 15-minute horizon where the vol-scaled band can shrink below the spread.
BAND_MIN_SPREADS = 1.0

#: Hard cap on drift, in units of the horizon's predictive sigma. The ceiling
#: on how opinionated the model is ever allowed to be.
KAPPA_MAX = 0.30

#: Extra drift shrinkage per horizon — short-horizon signals decay fast.
KAPPA_HORIZON: dict[str, float] = {
    "15m": 0.85, "1h": 1.00, "4h": 1.00, "session": 0.95, "24h": 0.85,
}

#: How much of the blended score is replaced by the historical analogue — how
#: this asset actually behaved next, in the same regime, at this time of day.
#: Small on purpose: conditional base rates are a prior, not a forecast.
ANALOG_WEIGHT = 0.12

#: Volatility multiplier when a scheduled event lands inside the horizon.
#: Widening the distribution is what mechanically converts event risk into
#: "less likely to sit still", which is the observable truth about data days.
EVENT_VOL_MULT: dict[str, float] = {"HIGH": 1.35, "medium": 1.15}

#: Drift is less trustworthy into an event, so shrink it as well.
EVENT_KAPPA_MULT: dict[str, float] = {"HIGH": 0.55, "medium": 0.80}

#: Student-t tail index bounds (method of moments on realised kurtosis).
NU_MIN, NU_MAX = 4.5, 30.0

#: Monte-Carlo sample size for the standardised-t CDF (fixed seed: run-to-run
#: probability deltas must be real signal, never resampling noise).
MC_DRAWS = 400_000
MC_SEED = 42


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------

#: Points out of 100, by contribution. Summing to 100 keeps the label honest.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "availability": 30.0,   # share of model weight that actually had data
    "agreement": 25.0,      # do the components point the same way
    "vol_regime": 15.0,     # extreme vol => the model knows less
    "event_risk": 10.0,     # imminent high-impact release
    "freshness": 10.0,      # how old is the price feed
    "track_record": 10.0,   # realised Brier vs baseline, once enough samples
}

CONFIDENCE_HIGH = 70.0
CONFIDENCE_MEDIUM = 45.0

#: Minimum scored predictions before the stored track record is trusted.
TRACK_RECORD_MIN_SAMPLES = 30


# --------------------------------------------------------------------------
# data quality
# --------------------------------------------------------------------------

QUALITY_GOOD = 0.80     # weighted availability at/above this and price live
QUALITY_LIMITED = 0.50

#: A price older than this many seconds is no longer "live" (market hours).
PRICE_LIVE_SECONDS = 180
PRICE_STALE_SECONDS = 900


# --------------------------------------------------------------------------
# live loop: fetch cadences and recompute triggers
# --------------------------------------------------------------------------

#: Seconds between fetches, per source. Twelve Data's free tier allows 8
#: calls/minute and 800/day; at 60s for three assets that is 180 calls/hour of
#: quote traffic, so the daily budget below is the binding constraint.
FETCH_INTERVAL: dict[str, int] = {
    "price": 60,
    "bars": 300,
    "polymarket": 300,
    "news": 900,
    "macro": 3600,
    "calendar": 3600,
}

#: Recompute when any of these fire (or when MAX_RECOMPUTE_GAP elapses).
TRIGGER_PRICE_ATR = 0.25        # price moved this fraction of ATR(14) since last
TRIGGER_POLY_PTS = 3.0          # a relevant Polymarket probability moved 3 points
TRIGGER_VOL_REL = 0.10          # EWMA sigma moved 10% relative
MAX_RECOMPUTE_GAP = 300         # seconds; heartbeat recompute even if quiet
MIN_RECOMPUTE_GAP = 20          # seconds; never thrash faster than this

TWELVEDATA_DAILY_BUDGET = int(os.getenv("QUANTLAB_TD_DAILY_BUDGET", "700"))
SERVER_PORT = int(os.getenv("QUANTLAB_PREDICT_PORT", "8787"))

#: Explanations only mention a driver whose contribution moved by this much.
EXPLAIN_MIN_DELTA = 0.02
#: A probability change smaller than this is not worth narrating.
EXPLAIN_MIN_PROB_PTS = 1.0


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------

_OVERRIDABLE_SCALARS = {"BAND_K", "BAND_MIN_SPREADS", "KAPPA_MAX", "ANALOG_WEIGHT",
                        "CONFIDENCE_HIGH", "CONFIDENCE_MEDIUM"}
_OVERRIDABLE_DICTS = {"BASE_WEIGHTS", "HORIZON_TILT", "REGIME_TILT",
                      "KAPPA_HORIZON", "CONFIDENCE_WEIGHTS", "EVENT_VOL_MULT"}

#: Populated by load_overrides() so the UI can say the model is not stock.
ACTIVE_OVERRIDES: dict = {}


def normalised_base_weights() -> dict[str, float]:
    """BASE_WEIGHTS scaled to sum to exactly 1.0."""
    total = sum(BASE_WEIGHTS.values())
    if total <= 0:
        raise ValueError("BASE_WEIGHTS sum to zero")
    return {k: v / total for k, v in BASE_WEIGHTS.items()}


def load_overrides(path: Path | None = None) -> dict:
    """Apply `weights.json` over the module defaults. Returns what was applied."""
    global ACTIVE_OVERRIDES
    p = path or OVERRIDES_PATH
    if not p.exists():
        ACTIVE_OVERRIDES = {}
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"predict.config: ignoring unreadable {p.name} ({exc})")
        ACTIVE_OVERRIDES = {}
        return {}

    applied: dict = {}
    g = globals()
    for key, value in data.items():
        name = key.upper()
        if name in _OVERRIDABLE_SCALARS and isinstance(value, (int, float)):
            g[name] = float(value)
            applied[name] = value
        elif name in _OVERRIDABLE_DICTS and isinstance(value, dict):
            merged = {**g[name]}
            for k, v in value.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            g[name] = merged
            applied[name] = value
    ACTIVE_OVERRIDES = applied
    return applied


def save_overrides(data: dict, path: Path | None = None) -> Path:
    """Write an override file (used by the backtester's weight optimiser)."""
    p = path or OVERRIDES_PATH
    payload = {**data, "_written": "quantlab.predict.backtest optimiser",
               "_model_version": MODEL_VERSION}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


load_overrides()
