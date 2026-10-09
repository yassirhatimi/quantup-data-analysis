"""Polymarket as one input — never as truth, and never forced.

What this module refuses to do is as important as what it does. A naive
keyword filter on "euro" matches *CCT Europe Series* Counter-Strike matches,
and a naive read of a central-bank market treats "will the ECB cut" and "will
the ECB hike" as the same signal. Both mistakes would inject confident noise
into the prediction. So:

- every question must clear a **finance-context** regex and survive an
  **exclusion** regex for sport, entertainment and crypto-price markets;
- a market must carry real money (`MIN_LIQUIDITY` / `MIN_VOLUME_24H`) — a
  thin book is a quote, not a probability;
- central-bank questions have their hawkish/dovish direction **parsed from the
  wording**, and any question whose direction cannot be established is
  excluded and listed in the output rather than guessed at;
- the whole Fed-meeting ladder is aggregated properly: outcome probabilities
  are renormalised to sum to 1 (they routinely sum to 1.02 across separate
  binary markets) and collapsed into one number, the **expected policy move in
  basis points**.

The signal leans on *change* rather than level. The level of Fed pricing is
already in the two-year yield, which the `rates` component reads directly;
what Polymarket adds intraday is the repricing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data import polymarket as poly_client
from quantlab.data.base import get_json
from quantlab.predict import assets as asset_spec
from quantlab.predict.component import ComponentScore, clamp, missing, squash

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO / "data" / "predict" / "poly_snapshots.json"

GAMMA_URL = "https://gamma-api.polymarket.com"

MIN_LIQUIDITY = 5_000.0
MIN_VOLUME_24H = 1_000.0
PAGES = 3
PAGE_SIZE = 100
_TTL = 240.0

#: A market must talk about one of these to be considered at all.
FINANCE_CONTEXT = re.compile(
    r"\b(fed|fomc|federal reserve|powell|interest rate|rate cut|rate hike|"
    r"inflation|cpi|pce|recession|gold|bullion|ecb|european central bank|"
    r"lagarde|bank of england|boe|bailey|treasury|yield|dollar|"
    r"unemployment|jobs report|payrolls|government shutdown|tariff)\b", re.I)

#: ...and must not be one of these, whatever else it says.
EXCLUDE = re.compile(
    r"\b(counter-strike|cs2|dota|valorant|league of legends|esports|"
    r"nba|nfl|mlb|nhl|soccer|football|tennis|ufc|boxing|f1|"
    r"premier league|champions league|world cup|olympics|"
    r"bitcoin|ethereum|solana|xrp|dogecoin|nominee|oscar|grammy|"
    r"super bowl|election winner)\b|\bbo[35]\b|\bvs\.?\b", re.I)

HAWKISH = re.compile(r"\b(hike|hikes|increase|increases|raise|raises|higher|hawkish)\b", re.I)
DOVISH = re.compile(r"\b(cut|cuts|decrease|decreases|lower|lowers|dovish|reduce|reduces)\b", re.I)
NO_CHANGE = re.compile(r"\bno change\b|\bunchanged\b|\bhold\b", re.I)
BPS = re.compile(r"(\d+)\s*\+?\s*bps", re.I)

_cache: dict[str, tuple[float, object]] = {}


def clear_cache() -> None:
    _cache.clear()


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------

@dataclass
class Market:
    slug: str
    question: str
    prob: float                  # probability of the first outcome ("Yes")
    outcomes: list[str]
    volume_24h: float
    liquidity: float
    end_date: pd.Timestamp | None
    event_title: str | None = None
    change: float | None = None          # probability change since last snapshot
    change_window_min: float | None = None
    rule: str = ""
    relevance: str = ""
    signed_contribution: float = 0.0

    def time_remaining(self, now: pd.Timestamp) -> str:
        if self.end_date is None or pd.isna(self.end_date):
            return "no end date"
        days = (self.end_date - now).total_seconds() / 86400
        if days < 0:
            return "expired"
        if days < 1:
            return f"{days * 24:.0f}h left"
        return f"{days:.0f}d left"

    def to_dict(self, now: pd.Timestamp) -> dict:
        return {
            "question": self.question,
            "probability": round(self.prob, 4),
            "probability_change": round(self.change, 4) if self.change is not None else None,
            "change_window_min": self.change_window_min,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "time_remaining": self.time_remaining(now),
            "end_date": str(self.end_date) if self.end_date is not None else None,
            "timestamp": str(now),
            "rule": self.rule,
            "relevance": self.relevance,
            "contribution": round(self.signed_contribution, 4),
            "slug": self.slug,
        }


def _parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def fetch_universe(pages: int = PAGES) -> list[Market]:
    """Active markets by 24h volume and by liquidity, de-duplicated."""
    def fetch() -> list[Market]:
        seen: dict[str, dict] = {}
        for order in ("volume24hr", "liquidity"):
            for page in range(pages):
                raw = get_json(f"{GAMMA_URL}/markets", params={
                    "limit": PAGE_SIZE, "offset": page * PAGE_SIZE,
                    "active": "true", "closed": "false",
                    "order": order, "ascending": "false"})
                if not raw:
                    break
                for m in raw:
                    if m.get("slug"):
                        seen[m["slug"]] = m
        out: list[Market] = []
        for slug, m in seen.items():
            prices = [float(p) for p in (_parse_json_field(m.get("outcomePrices")) or [])]
            if not prices:
                continue
            events = m.get("events") or []
            out.append(Market(
                slug=slug,
                question=str(m.get("question") or ""),
                prob=prices[0],
                outcomes=list(_parse_json_field(m.get("outcomes")) or []),
                volume_24h=float(m.get("volume24hr") or 0.0),
                liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0.0),
                end_date=pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"),
                event_title=(events[0].get("title") if events else None),
            ))
        return out

    return _cached("universe", _TTL, fetch)


def _cached(key: str, ttl: float, fn):
    hit = _cache.get(key)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


# --------------------------------------------------------------------------
# snapshots, for probability change
# --------------------------------------------------------------------------

def _load_snapshots() -> dict:
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_snapshots(data: dict) -> None:
    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def record_and_diff(markets: list[Market], now: pd.Timestamp,
                    lookback_min: float = 90.0) -> None:
    """Attach each market's probability change and append the new observation.

    The comparison point is the newest stored observation that is at least ten
    minutes old, so a fast poll loop does not diff a market against itself.
    """
    store = _load_snapshots()
    for m in markets:
        history = store.get(m.slug, [])
        candidates = []
        for ts_iso, p in history:
            try:
                ts = pd.Timestamp(ts_iso)
            except ValueError:
                continue
            age_min = (now - ts).total_seconds() / 60
            if 10 <= age_min <= lookback_min:
                candidates.append((age_min, float(p)))
        if candidates:
            age_min, prior = min(candidates, key=lambda c: c[0])
            m.change = m.prob - prior
            m.change_window_min = round(age_min, 1)
        history.append([now.isoformat(), round(m.prob, 5)])
        store[m.slug] = history[-80:]
    _save_snapshots(store)


# --------------------------------------------------------------------------
# relevance and reading
# --------------------------------------------------------------------------

def _is_candidate(m: Market) -> bool:
    if m.liquidity < MIN_LIQUIDITY and m.volume_24h < MIN_VOLUME_24H:
        return False
    q = m.question
    return bool(FINANCE_CONTEXT.search(q)) and not EXCLUDE.search(q)


def _hawkish_direction(question: str) -> int | None:
    """+1 for a hike question, -1 for a cut question, 0 for 'no change'.

    None when the wording does not establish a direction — in which case the
    market is dropped, because a probability with an unknown sign is worse than
    no probability at all.
    """
    if NO_CHANGE.search(question):
        return 0
    hawk, dove = bool(HAWKISH.search(question)), bool(DOVISH.search(question))
    if hawk and not dove:
        return 1
    if dove and not hawk:
        return -1
    return None


def _bps_of(question: str) -> float | None:
    """Signed basis points implied by a meeting-ladder question."""
    direction = _hawkish_direction(question)
    if direction is None:
        return None
    if direction == 0:
        return 0.0
    match = BPS.search(question)
    if not match:
        return None
    return float(match.group(1)) * direction


@dataclass
class FedLadder:
    """One meeting's outcome ladder, renormalised."""
    label: str
    expected_bps: float
    expected_bps_change: float | None
    raw_sum: float
    outcomes: list[tuple[str, float, float]] = field(default_factory=list)  # bps, p, dp
    volume_24h: float = 0.0
    liquidity: float = 0.0
    markets: list[Market] = field(default_factory=list)


def _build_ladders(markets: list[Market]) -> list[FedLadder]:
    groups: dict[str, list[Market]] = {}
    for m in markets:
        bps = _bps_of(m.question)
        if bps is None:
            continue
        key = m.event_title or "Fed policy"
        groups.setdefault(key, []).append(m)

    ladders: list[FedLadder] = []
    for label, group in groups.items():
        if len(group) < 2:
            continue    # a single leg is not a ladder and cannot be renormalised
        rows, total = [], 0.0
        for m in group:
            bps = _bps_of(m.question)
            rows.append((bps, m.prob, m.change))
            total += m.prob
        if total <= 0:
            continue
        expected = sum(bps * p for bps, p, _ in rows) / total
        d_rows = [(bps, dp) for bps, _, dp in rows if dp is not None]
        expected_change = (sum(bps * dp for bps, dp in d_rows) / total
                           if d_rows else None)
        ladders.append(FedLadder(
            label=label, expected_bps=expected, expected_bps_change=expected_change,
            raw_sum=total,
            outcomes=[(f"{bps:+.0f} bps" if bps else "no change", p, dp or 0.0)
                      for bps, p, dp in sorted(rows, key=lambda r: -r[1])],
            volume_24h=sum(m.volume_24h for m in group),
            liquidity=sum(m.liquidity for m in group),
            markets=group))
    return sorted(ladders, key=lambda l: -l.volume_24h)


def relevant_markets(code: str, universe: list[Market] | None = None,
                     now: pd.Timestamp | None = None
                     ) -> tuple[list[Market], list[str]]:
    """Markets this asset genuinely cares about, plus what was rejected and why."""
    a = asset_spec.get(code)
    now = now or pd.Timestamp.now(tz="UTC")
    universe = universe if universe is not None else fetch_universe()

    picked: list[Market] = []
    rejected: list[str] = []
    for m in universe:
        if not _is_candidate(m):
            continue
        for rule in a.poly_rules:
            if not rule.pattern.search(m.question):
                continue
            if rule.kind in ("fed_meeting", "cb_binary"):
                if _hawkish_direction(m.question) is None:
                    rejected.append(f"{m.question[:70]} — policy direction not "
                                    f"parseable from the wording")
                    break
            m.rule = rule.key
            m.relevance = rule.why
            picked.append(m)
            break
    return picked, rejected


def score_polymarket(code: str, now: pd.Timestamp | None = None,
                     universe: list[Market] | None = None) -> ComponentScore:
    a = asset_spec.get(code)
    now = now or pd.Timestamp.now(tz="UTC")

    try:
        universe = universe if universe is not None else fetch_universe()
    except Exception as exc:
        return missing("polymarket", f"Gamma API unreachable: {type(exc).__name__}")
    if not universe:
        return missing("polymarket", "Gamma API returned no active markets")

    picked, rejected = relevant_markets(code, universe, now)
    if not picked:
        note = f"no relevant market among {len(universe)} active markets"
        if rejected:
            note += f"; {len(rejected)} matched but were excluded"
        return missing("polymarket", note)

    record_and_diff(picked, now)

    rules = {r.key: r for r in a.poly_rules}
    parts: list[tuple[float, float]] = []
    ev: list[str] = []
    used: list[Market] = []

    # ---- Fed / central-bank ladders ---------------------------------------
    ladder_markets = [m for m in picked if rules[m.rule].kind == "fed_meeting"]
    ladders = _build_ladders(ladder_markets)
    for ladder in ladders[:2]:
        rule = rules[ladder.markets[0].rule]
        # Level: expected move scaled so a full 25bp priced hike is one unit.
        level = squash(ladder.expected_bps / 25.0, 1.0)
        # Change: 5bp of repricing inside the window is a full unit — repricing
        # is a much sharper signal than the standing level, which yields
        # already reflect.
        change_term = (squash(ladder.expected_bps_change / 5.0, 1.0)
                       if ladder.expected_bps_change is not None else None)
        combined = (0.4 * level + 0.6 * change_term) if change_term is not None else level
        contribution = rule.sign * combined
        parts.append((contribution, rule.weight * (1.0 if change_term is not None else 0.7)))

        book = " · ".join(f"{name} {p:.0%}" for name, p, _ in ladder.outcomes[:4])
        moved = ""
        if ladder.expected_bps_change is not None and abs(ladder.expected_bps_change) >= 0.2:
            moved = (f", repriced {ladder.expected_bps_change:+.1f} bps since the "
                     f"last check")
        # Plain text, no markdown emphasis: this string is rendered into both a
        # terminal and an HTML table, and asterisks are literal in both.
        ev.append(f"{ladder.label}: market prices {ladder.expected_bps:+.1f} bps "
                  f"({book}){moved} → "
                  f"{'bullish' if contribution > 0 else 'bearish'} {abs(contribution):.2f}")
        for m in ladder.markets:
            m.signed_contribution = contribution / max(len(ladder.markets), 1)
        used.extend(ladder.markets)

    # ---- single central-bank questions ------------------------------------
    for m in picked:
        rule = rules[m.rule]
        if rule.kind != "cb_binary":
            continue
        direction = _hawkish_direction(m.question)
        if direction is None or direction == 0:
            continue
        level = clamp((m.prob - 0.5) * 2) * direction
        change_term = clamp(m.change * 8) * direction if m.change is not None else None
        combined = (0.4 * level + 0.6 * change_term) if change_term is not None else level
        m.signed_contribution = rule.sign * combined
        parts.append((m.signed_contribution, rule.weight * _money_weight(m)))
        ev.append(f"{m.question[:80]} — {m.prob:.0%}"
                  + (f" ({m.change:+.1%} in {m.change_window_min:.0f}m)"
                     if m.change is not None else "")
                  + f" → {'bullish' if m.signed_contribution > 0 else 'bearish'}")
        used.append(m)

    # ---- state questions (recession, shutdown) ----------------------------
    for m in picked:
        rule = rules[m.rule]
        if rule.kind != "binary":
            continue
        level = clamp((m.prob - 0.5) * 1.2)
        change_term = clamp(m.change * 8) if m.change is not None else None
        combined = (0.35 * level + 0.65 * change_term) if change_term is not None else level
        m.signed_contribution = rule.sign * combined
        parts.append((m.signed_contribution, rule.weight * _money_weight(m)))
        ev.append(f"{m.question[:80]} — {m.prob:.0%}"
                  + (f" ({m.change:+.1%})" if m.change is not None else ""))
        used.append(m)

    # ---- markets on the price itself: change only -------------------------
    for m in picked:
        rule = rules[m.rule]
        if rule.kind != "gold_level":
            continue
        if m.change is None:
            # The level of "will gold reach $X" says nothing without X and a
            # baseline; only its repricing is informative, so this market waits
            # for a second observation rather than contributing a guess.
            m.relevance += (" — level alone carries no signal without the threshold; "
                            "waiting for a second observation to measure the change")
            continue
        m.signed_contribution = rule.sign * clamp(m.change * 8)
        parts.append((m.signed_contribution, rule.weight * _money_weight(m)))
        ev.append(f"{m.question[:80]} — {m.prob:.0%} "
                  f"({m.change:+.1%} in {m.change_window_min:.0f}m)")
        used.append(m)

    if not parts:
        note = (f"{len(picked)} relevant market(s) found but none produced a "
                f"usable signal yet (change-based inputs need a second "
                f"observation)")
        return ComponentScore("polymarket", 0.0, available=False, status="delayed",
                              note=note,
                              detail={"markets": [m.to_dict(now) for m in picked]})

    total_w = sum(w for _, w in parts)
    score = clamp(sum(v * w for v, w in parts) / total_w) if total_w > 0 else 0.0

    note = f"{len(used)} relevant market(s) of {len(universe)} scanned"
    if rejected:
        note += f"; {len(rejected)} excluded (unparseable direction)"

    return ComponentScore("polymarket", score, evidence=ev, note=note, detail={
        "n_universe": len(universe), "n_relevant": len(used),
        "rejected": rejected[:6],
        "expected_fed_bps": ladders[0].expected_bps if ladders else None,
        "expected_fed_bps_change": (ladders[0].expected_bps_change
                                    if ladders else None),
        "ladder_label": ladders[0].label if ladders else None,
        "ladder_raw_sum": round(ladders[0].raw_sum, 4) if ladders else None,
        "markets": [m.to_dict(now) for m in used],
    })


def _money_weight(m: Market) -> float:
    """Down-weight thin books: a 90% quote on $6k of liquidity is not 90%."""
    depth = max(m.liquidity, m.volume_24h)
    return float(np.clip(np.log10(max(depth, 1.0)) / 5.0, 0.3, 1.2))
