"""Per-asset specification: symbols, costs, macro drivers, Polymarket rules.

One place to answer "what moves this thing, and where do I get it". Adding a
fourth asset means adding one `Asset` here — the engine reads everything it
needs from this table.

The `sign` on every driver is a **stated modelling assumption, not a
measurement**: +1 means "a rise in this reading is bullish for this asset".
They are chosen from the standard transmission channels and are visible in the
output so you can argue with them. The one that surprises people is inflation
and gold: a hot CPI print sends gold *down* on the day, because the market
immediately prices a more hawkish Fed and real yields jump. Gold as an
inflation hedge is a multi-year channel, not a release-day reaction, and the
sign below is the release-day one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# driver + rule types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MacroDriver:
    """One FRED series feeding the macro or rates component."""
    series: str
    label: str
    transform: str        # "level" | "yoy" | "chg" | "spread"
    sign: int             # +1 = a rise in this reading is bullish for the asset
    channel: str          # "market" (repriced continuously) | "release"
    max_age_days: int     # older than this and the reading is reported missing
    why: str
    against: str | None = None   # second series, for transform="spread"


@dataclass(frozen=True)
class PolyRule:
    """How one family of Polymarket questions maps onto this asset.

    `kind` decides how the probability is read:

    - ``fed_meeting``  aggregate the whole meeting ladder into expected bps;
      `sign` applies to *hawkishness*.
    - ``cb_binary``    a single central-bank question. Its hawkish/dovish
      direction is parsed from the wording, because "will the ECB cut" and
      "will the ECB hike" point opposite ways; `sign` applies to hawkishness of
      that central bank. A question whose direction cannot be parsed is
      **excluded and reported**, never guessed.
    - ``binary``       a state question (recession, shutdown); `sign` applies
      directly to the probability of the state being true.
    - ``gold_level``   a market on the price itself; only the *change* in
      probability is used, since the level means nothing without its threshold.
    """
    key: str
    pattern: re.Pattern
    kind: str             # "fed_meeting" | "cb_binary" | "binary" | "gold_level"
    sign: int             # +1 = a rise in this reading is bullish for the asset
    weight: float
    why: str


@dataclass(frozen=True)
class Asset:
    code: str
    name: str
    twelvedata: str          # Twelve Data spot symbol
    yahoo_intraday: str      # Yahoo symbol for 15m/1h bars
    yahoo_daily: str         # Yahoo symbol for daily bars
    uses_basis: bool         # daily source is a future -> shift levels onto spot
    decimals: int
    unit: str                # display unit for distances
    unit_size: float         # one "unit" in price terms (pip / dollar)
    spread: float            # typical retail round-trip spread, price units
    round_step: float        # spacing of the round numbers traders watch
    has_volume: bool         # is there a real volume series behind the bars
    macro_drivers: tuple[MacroDriver, ...]
    rate_drivers: tuple[MacroDriver, ...]
    poly_rules: tuple[PolyRule, ...]
    news_query: str
    news_context: tuple[str, ...]        # article must mention one of these
    news_bullish: tuple[str, ...]
    news_bearish: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


# --------------------------------------------------------------------------
# shared driver definitions
# --------------------------------------------------------------------------

# US drivers common to all three (every pair here has USD on one side).
_US_DOLLAR = MacroDriver(
    "DTWEXBGS", "Broad trade-weighted dollar", "level", -1, "market", 21,
    "All three assets are quoted in dollars, so a broad dollar move is "
    "mechanically half the price move.")
_US_RISK = MacroDriver(
    "VIXCLS", "VIX", "level", +1, "market", 7,
    "Risk-off bids the dollar and, separately, bids gold.")
_US_CPI = MacroDriver(
    "CPIAUCSL", "US CPI", "yoy", -1, "release", 70,
    "Hot US inflation pulls the Fed hawkish, lifts real yields and the dollar.")
_US_CORE_PCE = MacroDriver(
    "PCEPILFE", "US core PCE", "yoy", -1, "release", 70,
    "The Fed's actual target measure — the most policy-relevant inflation print.")
_US_PAYROLLS = MacroDriver(
    "PAYEMS", "US nonfarm payrolls", "chg", -1, "release", 60,
    "Strong jobs remove the urgency to cut: yields and dollar up.")
_US_UNRATE = MacroDriver(
    "UNRATE", "US unemployment rate", "level", +1, "release", 60,
    "Rising unemployment pulls the Fed toward easing and weakens the dollar.")
_US_BREAKEVEN = MacroDriver(
    "T10YIE", "10y breakeven inflation", "level", +1, "market", 7,
    "Expected inflation; rising breakevens against sticky nominal yields "
    "compress real yields.")

_US_2Y = MacroDriver(
    "DGS2", "US 2y yield (Fed path)", "level", -1, "market", 7,
    "The 2y *is* the market's Fed forecast. It rises when cuts get priced out.")
_US_10Y = MacroDriver(
    "DGS10", "US 10y yield", "level", -1, "market", 7,
    "The headline discount rate; drives the dollar and the real yield inside it.")
_US_REAL = MacroDriver(
    "DFII10", "US 10y real yield (TIPS)", "level", -1, "market", 7,
    "The opportunity cost of holding a zero-coupon asset.")
_US_CURVE = MacroDriver(
    "T10Y2Y", "US 10y-2y curve", "level", +1, "market", 7,
    "Bull steepening out of inversion accompanies cutting cycles.")
_US_FF = MacroDriver(
    "DFF", "Fed funds effective", "level", -1, "market", 10,
    "The policy rate itself — the cash yield competing with everything else.")


# --------------------------------------------------------------------------
# Polymarket rules
# --------------------------------------------------------------------------

# The Fed-meeting family is handled specially: the engine groups all "Will the
# Fed {increase,decrease} rates by N bps / no change after the <month> meeting"
# markets, renormalises their probabilities to sum to 1, and collapses them
# into one number — the expected policy move in basis points. Positive is
# hawkish. Every asset here is short the dollar by construction, so all three
# carry sign -1 against hawkishness.
# Order-independent lookaheads, not a left-to-right chain. The chain version of
# this missed "Will there be no change in Fed interest rates after the September
# meeting?" — because "no change" sits *before* "Fed" — which silently dropped
# the largest leg of the ladder and left the renormalised expected move roughly
# three times too hawkish. Matching on presence rather than order is the fix.
_FED_MEETING = _rx(r"(?=.*\b(fed|fomc|federal reserve)\b)"
                   r"(?=.*\brates?\b)"
                   r"(?=.*\b(increase|increases|decrease|decreases|hike|hikes|"
                   r"cut|cuts|no change|unchanged|hold)\b)"
                   r"(?=.*\b(meeting|decision|fomc)\b)")
_FED_YEAR = _rx(r"\bfed\b.*(rate (hike|cut)|hike|cut).*\b20\d\d\b")
_RECESSION = _rx(r"\b(us |u\.s\. )?recession\b")
_SHUTDOWN = _rx(r"\bgovernment shutdown\b")
_GOLD_LEVEL = _rx(r"\bgold\b.*\$?\s?[\d,]{3,}|\bgold\b.*(reach|hit|above|below)")
_ECB = _rx(r"\becb\b|\beuropean central bank\b")
_BOE = _rx(r"\bbank of england\b|\bboe\b")

_FED_RULES = (
    PolyRule("fed_meeting", _FED_MEETING, "fed_meeting", -1, 1.0,
             "Real money pricing the next FOMC outcome; hawkish repricing lifts "
             "the dollar and real yields."),
    # Read as a single hawkish/dovish question rather than a ladder: it quotes
    # no basis points, so there is nothing to aggregate — only a direction.
    PolyRule("fed_year", _FED_YEAR, "cb_binary", -1, 0.5,
             "The full-year Fed path — slower moving than the meeting markets "
             "but the same channel."),
)


# --------------------------------------------------------------------------
# news lexicons
# --------------------------------------------------------------------------

_DOLLAR_BULL_FOR_USD = ("hawkish", "rate hike", "hikes rates", "strong jobs",
                        "inflation accelerat", "hotter than expected",
                        "yields jump", "higher for longer", "dollar surges",
                        "dollar strengthens", "taper")
_DOLLAR_BEAR_FOR_USD = ("dovish", "rate cut", "cuts rates", "inflation cools",
                        "cooler inflation", "weak jobs", "jobs miss",
                        "recession", "easing", "dollar weakens", "dollar slides",
                        "yields fall")

_US_NEWS_CONTEXT = ("federal reserve", "fed ", "fomc", "powell", "inflation",
                    "treasury", "yield", "interest rate", "rate cut", "rate hike",
                    "dollar index", "monetary policy", "central bank", "payrolls",
                    "jobs report", "cpi")


# --------------------------------------------------------------------------
# the assets
# --------------------------------------------------------------------------

ASSETS: dict[str, Asset] = {
    "XAUUSD": Asset(
        code="XAUUSD",
        name="Gold spot",
        twelvedata="XAU/USD",
        yahoo_intraday="GC=F",
        yahoo_daily="GC=F",
        uses_basis=True,
        decimals=2,
        unit="$",
        unit_size=1.0,
        spread=0.35,
        round_step=25.0,
        has_volume=True,
        macro_drivers=(
            _US_DOLLAR, _US_RISK, _US_BREAKEVEN, _US_CPI, _US_CORE_PCE,
            _US_PAYROLLS, _US_UNRATE,
            MacroDriver("M2SL", "US M2 money stock", "yoy", +1, "release", 90,
                        "Long-run debasement: more dollars chasing a fixed "
                        "above-ground gold stock."),
        ),
        rate_drivers=(_US_REAL, _US_2Y, _US_10Y, _US_CURVE, _US_FF),
        poly_rules=_FED_RULES + (
            PolyRule("recession", _RECESSION, "binary", +1, 0.6,
                     "Recession odds pull the Fed dovish and bid the safe haven."),
            PolyRule("shutdown", _SHUTDOWN, "binary", +1, 0.3,
                     "Fiscal dysfunction is a mild, well-documented gold bid."),
            PolyRule("gold_level", _GOLD_LEVEL, "gold_level", +1, 0.8,
                     "A market directly on the gold price — the only Polymarket "
                     "input that is about this asset rather than its drivers."),
        ),
        news_query=('("gold price" OR bullion OR "spot gold" OR "precious metals" OR '
                    '"Federal Reserve" OR FOMC OR Powell OR "Treasury yields" OR '
                    '"real yields" OR "dollar index")'),
        news_context=_US_NEWS_CONTEXT + ("gold price", "gold prices", "bullion",
                                         "xau", "spot gold", "precious metal"),
        news_bullish=_DOLLAR_BEAR_FOR_USD + (
            "safe haven", "central bank buying", "war", "escalat", "sanction",
            "debt ceiling", "downgrade", "stimulus", "flight to safety"),
        news_bearish=_DOLLAR_BULL_FOR_USD + (
            "risk-on", "profit taking", "ceasefire", "peace deal", "de-escalat"),
        notes=("Daily bars come from GC=F futures and are shifted onto spot by "
               "the futures/spot basis before any level is used.",),
    ),

    "EURUSD": Asset(
        code="EURUSD",
        name="Euro / US Dollar",
        twelvedata="EUR/USD",
        yahoo_intraday="EURUSD=X",
        yahoo_daily="EURUSD=X",
        uses_basis=False,
        decimals=5,
        unit="pip",
        unit_size=0.0001,
        spread=0.8 * 0.0001,
        round_step=0.0050,
        has_volume=False,
        macro_drivers=(
            _US_DOLLAR, _US_CPI, _US_CORE_PCE, _US_PAYROLLS, _US_UNRATE,
            MacroDriver("VIXCLS", "VIX", "level", -1, "market", 7,
                        "Risk-off is dollar-positive; EURUSD is the funding-side "
                        "loser in a scramble for dollars."),
            MacroDriver("CP0000EZ19M086NEST", "Euro-area HICP", "yoy", +1, "release", 75,
                        "Hot euro-area inflation pulls the ECB hawkish and lifts "
                        "the euro leg of the pair."),
        ),
        rate_drivers=(
            _US_2Y, _US_10Y, _US_REAL,
            MacroDriver("ECBDFR", "ECB deposit facility rate", "level", +1, "market", 10,
                        "The euro's policy anchor; the pair trades the *gap* "
                        "between this and the Fed."),
            MacroDriver("DGS2", "US 2y minus ECB depo (policy gap)", "spread", -1,
                        "market", 10,
                        "The rate differential is the single cleanest driver of a "
                        "major: a widening US premium pulls capital into dollars.",
                        against="ECBDFR"),
        ),
        poly_rules=_FED_RULES + (
            PolyRule("ecb", _ECB, "cb_binary", +1, 0.6,
                     "An ECB policy market prices the euro leg directly: a "
                     "hawkish ECB lifts the euro."),
            PolyRule("recession", _RECESSION, "binary", -1, 0.3,
                     "US recession odds are dollar-negative long-run but "
                     "risk-off-positive short-run; weighted low for that reason."),
        ),
        news_query=('("EUR/USD" OR euro OR "European Central Bank" OR ECB OR Lagarde OR '
                    '"Federal Reserve" OR FOMC OR Powell OR "dollar index" OR '
                    '"eurozone inflation")'),
        news_context=_US_NEWS_CONTEXT + ("euro", "eurozone", "euro area", "ecb",
                                         "lagarde", "eur/usd", "eurusd"),
        news_bullish=_DOLLAR_BEAR_FOR_USD + (
            "euro rallies", "euro strengthens", "ecb hawkish", "eurozone growth",
            "germany stimulus"),
        news_bearish=_DOLLAR_BULL_FOR_USD + (
            "euro falls", "euro weakens", "ecb dovish", "eurozone recession",
            "eurozone slowdown"),
        notes=("Twelve Data returns a gapless 24/7 FX series and pads the "
               "Friday-to-Sunday closure with synthetic bars; those are dropped "
               "before anything is computed.",),
    ),

    "GBPUSD": Asset(
        code="GBPUSD",
        name="British Pound / US Dollar",
        twelvedata="GBP/USD",
        yahoo_intraday="GBPUSD=X",
        yahoo_daily="GBPUSD=X",
        uses_basis=False,
        decimals=5,
        unit="pip",
        unit_size=0.0001,
        spread=1.2 * 0.0001,
        round_step=0.0050,
        has_volume=False,
        macro_drivers=(
            _US_DOLLAR, _US_CPI, _US_CORE_PCE, _US_PAYROLLS, _US_UNRATE,
            MacroDriver("VIXCLS", "VIX", "level", -1, "market", 7,
                        "Sterling is the higher-beta major: risk-off hits it "
                        "harder than the euro."),
            MacroDriver("LRHUTTTTGBM156S", "UK unemployment rate", "level", -1,
                        "release", 100,
                        "A loosening UK labour market pulls the BoE dovish and "
                        "weakens sterling."),
        ),
        rate_drivers=(
            _US_2Y, _US_10Y, _US_REAL,
            MacroDriver("IUDSOIA", "SONIA (UK overnight rate)", "level", +1, "market", 10,
                        "The live sterling policy rate; the BoE path in one daily "
                        "number."),
            MacroDriver("DGS2", "US 2y minus SONIA (policy gap)", "spread", -1,
                        "market", 10,
                        "The US/UK rate differential — a widening US premium pulls "
                        "capital out of sterling.", against="IUDSOIA"),
            MacroDriver("IRLTLT01GBM156N", "UK 10y gilt yield", "level", +1, "release", 75,
                        "Long gilts; monthly on FRED, so a slow channel rather "
                        "than a live one."),
        ),
        poly_rules=_FED_RULES + (
            PolyRule("boe", _BOE, "cb_binary", +1, 0.6,
                     "A Bank of England policy market prices the sterling leg "
                     "directly: a hawkish BoE lifts sterling."),
            PolyRule("recession", _RECESSION, "binary", -1, 0.4,
                     "Sterling is the higher-beta major and sells off into "
                     "global risk-off."),
        ),
        news_query=('("GBP/USD" OR sterling OR pound OR "Bank of England" OR BoE OR Bailey OR '
                    '"Federal Reserve" OR FOMC OR Powell OR "dollar index" OR "UK inflation")'),
        news_context=_US_NEWS_CONTEXT + ("sterling", "pound", "bank of england",
                                         "boe", "bailey", "gbp/usd", "gbpusd",
                                         "uk inflation", "gilt"),
        news_bullish=_DOLLAR_BEAR_FOR_USD + (
            "pound rallies", "sterling strengthens", "boe hawkish", "uk growth"),
        news_bearish=_DOLLAR_BULL_FOR_USD + (
            "pound falls", "sterling weakens", "boe dovish", "uk recession",
            "gilt selloff"),
        notes=("Same weekend-padding filter as EURUSD.",),
    ),
}

DEFAULT_CODES = list(ASSETS)


def get(code: str) -> Asset:
    try:
        return ASSETS[code.upper()]
    except KeyError:
        raise KeyError(f"Unknown asset {code!r}. Known: {', '.join(ASSETS)}") from None


def fmt_price(code: str, value: float) -> str:
    a = get(code)
    if value is None or value != value:  # NaN
        return "—"
    return f"{value:,.{a.decimals}f}"


def fmt_distance(code: str, value: float, signed: bool = True) -> str:
    """A price distance in the unit a trader of this asset thinks in."""
    a = get(code)
    if value is None or value != value:
        return "—"
    n = value / a.unit_size
    sign = "+" if (signed and n >= 0) else ""
    if a.unit == "pip":
        return f"{sign}{n:.1f} pip"
    return f"{sign}${n:,.2f}"
