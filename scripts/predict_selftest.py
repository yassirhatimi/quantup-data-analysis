"""Invariant checks for the probability engine.

    python scripts/predict_selftest.py            # full run (hits the live APIs)
    python scripts/predict_selftest.py --offline  # maths and degradation only

These are the properties that must hold no matter what the market is doing.
Most of them exist because the brief is explicit about them: probabilities that
sum to 100, no fabricated values, confidence that falls when inputs go missing,
and stale data that is labelled rather than laundered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantlab.predict import (assets as asset_spec, component, config, engine,
                              market_data, quality, regime, store, technical)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((PASS if condition else FAIL, name, detail))
    return condition


# --------------------------------------------------------------------------
# offline: the maths
# --------------------------------------------------------------------------

def test_distribution() -> None:
    # A standardised t must have unit variance, or every sigma downstream lies.
    for nu in (4.5, 8.0, 30.0):
        sample = engine._t_sample(nu)
        check(f"standardised t(nu={nu}) has unit variance",
              abs(float(sample.std()) - 1.0) < 0.02,
              f"sd={sample.std():.4f}")

    # Driftless and symmetric: UP and DOWN must agree.
    up, down, rng = engine._probabilities(band=0.5, sigma=1.0, drift=0.0, nu=8.0)
    check("driftless split is symmetric", abs(up - down) < 0.005,
          f"up={up:.4f} down={down:.4f} range={rng:.4f}")
    check("driftless probabilities sum to 1", abs(up + down + rng - 1) < 1e-9)

    # Positive drift must move probability from DOWN to UP, monotonically.
    prev_up = -1.0
    for drift in (0.0, 0.05, 0.15, 0.30):
        u, d, _ = engine._probabilities(0.5, 1.0, drift, 8.0)
        if drift and not u > prev_up:
            check("P(UP) increases monotonically with drift", False,
                  f"drift={drift} up={u:.4f}")
            return
        prev_up = u
    check("P(UP) increases monotonically with drift", True, f"max up={prev_up:.3f}")

    # The anti-fake-precision guarantee, stated as a test.
    u, d, r = engine._probabilities(0.5, 1.0, config.KAPPA_MAX, 8.0)
    check("a maxed-out signal cannot produce a near-certain call", u < 0.55,
          f"P(UP) at kappa={config.KAPPA_MAX} is {u:.1%}")

    # A wider band must mean more RANGE.
    _, _, narrow = engine._probabilities(0.2, 1.0, 0.0, 8.0)
    _, _, wide = engine._probabilities(0.9, 1.0, 0.0, 8.0)
    check("a wider band raises P(RANGE)", wide > narrow,
          f"{narrow:.1%} -> {wide:.1%}")

    # No variance in the window (a shut market) must not invent movement.
    u, d, r = engine._probabilities(0.5, 0.0, 0.0, 8.0)
    check("zero variance gives RANGE = 100%", r == 1.0 and u == 0.0)


def test_rounding() -> None:
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(2000):
        p = rng.dirichlet([1, 1, 1])
        hp = engine.HorizonPrediction(
            key="t", label="t", start=pd.Timestamp.now(tz="UTC"),
            end=pd.Timestamp.now(tz="UTC"), hours=1, up=p[0], down=p[1], range_=p[2],
            confidence=50, confidence_label="MEDIUM", band=0.1, band_price=1.0,
            sigma=0.1, drift=0.0, score=0.0, variance_fraction=0.1,
            weights={}, contributions={}, event_level="none", event_names=[],
            analogue=None)
        if sum(hp.rounded()) != 100:
            bad += 1
    check("rounded probabilities always sum to exactly 100", bad == 0,
          f"{bad} failures in 2000 random splits")


def test_weight_redistribution() -> None:
    everything = {c: True for c in config.COMPONENTS}
    w_all, avail_all = engine.resolve_weights("1h", ["range"], everything)
    check("full weights sum to 1", abs(sum(w_all.values()) - 1) < 1e-9)
    check("full availability is 1", abs(avail_all - 1) < 1e-9)

    degraded = dict(everything)
    for name in ("macro", "news", "polymarket", "rates"):
        degraded[name] = False
    w_few, avail_few = engine.resolve_weights("1h", ["range"], degraded)
    check("degraded weights still sum to 1", abs(sum(w_few.values()) - 1) < 1e-9)
    check("availability falls when sources go missing", avail_few < avail_all,
          f"{avail_all:.2f} -> {avail_few:.2f}")
    check("missing components carry no weight",
          all(w_few.get(n, 0) == 0 for n in ("macro", "news", "polymarket", "rates")))


def test_confidence_degrades() -> None:
    full, _ = engine.compute_confidence(1.0, 0.9, 1.0, "none", "live", None)
    thin, _ = engine.compute_confidence(0.45, 0.9, 1.0, "none", "live", None)
    fight, _ = engine.compute_confidence(1.0, 0.1, 1.0, "none", "live", None)
    event, _ = engine.compute_confidence(1.0, 0.9, 1.0, "HIGH", "live", None)
    stale, _ = engine.compute_confidence(1.0, 0.9, 1.0, "none", "stale", None)
    wild, _ = engine.compute_confidence(1.0, 0.9, 2.5, "none", "live", None)

    check("missing sources cut confidence", thin < full, f"{full:.0f} -> {thin:.0f}")
    check("conflicting signals cut confidence", fight < full, f"{full:.0f} -> {fight:.0f}")
    check("event risk cuts confidence", event < full, f"{full:.0f} -> {event:.0f}")
    check("stale price cuts confidence", stale < full, f"{full:.0f} -> {stale:.0f}")
    check("extreme volatility cuts confidence", wild < full, f"{full:.0f} -> {wild:.0f}")
    check("confidence labels are ordered",
          engine.confidence_label(90) == "HIGH"
          and engine.confidence_label(55) == "MEDIUM"
          and engine.confidence_label(20) == "LOW")


def test_missing_component_is_not_neutral_evidence() -> None:
    absent = component.missing("polymarket", "no relevant market")
    check("a missing component scores exactly zero and is flagged",
          absent.score == 0.0 and not absent.available)
    present = component.ComponentScore("technical", 0.8)
    check("clamping holds the score inside [-1, 1]",
          component.ComponentScore("x", 5.0).score == 1.0
          and component.ComponentScore("x", -5.0).score == -1.0
          and present.score == 0.8)


def test_market_calendar() -> None:
    sat = pd.Timestamp("2026-08-08 12:00", tz="UTC")
    sun_early = pd.Timestamp("2026-08-09 12:00", tz="UTC")
    sun_late = pd.Timestamp("2026-08-09 22:00", tz="UTC")
    fri_late = pd.Timestamp("2026-08-07 22:00", tz="UTC")
    mon = pd.Timestamp("2026-08-10 12:00", tz="UTC")
    check("Saturday is closed", not market_data.market_open(sat))
    check("Sunday morning is closed", not market_data.market_open(sun_early))
    check("Sunday 22:00 is open", market_data.market_open(sun_late))
    check("Friday 22:00 is closed", not market_data.market_open(fri_late))
    check("Monday midday is open", market_data.market_open(mon))
    check("next_open from Saturday lands in the Sunday session",
          market_data.market_open(market_data.next_open(sat)),
          str(market_data.next_open(sat)))

    idx = pd.date_range("2026-08-07 18:00", "2026-08-10 02:00", freq="15min", tz="UTC")
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
    kept = market_data.drop_closed(frame)
    check("the weekend closure is dropped from bars",
          not any(market_data.market_open(t) is False for t in kept.index),
          f"{len(frame)} bars -> {len(kept)}")


def test_variance_time() -> None:
    profile = pd.Series(1 / 24, index=range(24))
    start = pd.Timestamp("2026-08-10 10:00", tz="UTC")           # Monday
    quarter = technical.variance_fraction(profile, start, start + pd.Timedelta(hours=6))
    day = technical.variance_fraction(profile, start, start + pd.Timedelta(hours=24))
    check("variance accumulates with the length of the window", day > quarter,
          f"6h={quarter:.3f} 24h={day:.3f}")
    check("a full flat day is one day of variance", abs(day - 1.0) < 0.02, f"{day:.4f}")

    sat = pd.Timestamp("2026-08-08 10:00", tz="UTC")
    shut = technical.variance_fraction(profile, sat, sat + pd.Timedelta(hours=6))
    check("a window inside the weekend carries no variance", shut == 0.0, f"{shut}")


def test_polymarket_direction_parsing() -> None:
    from quantlab.predict import polymarket_data as pm
    cases = [
        ("Will the Fed increase interest rates by 25 bps after the September meeting?", 25.0),
        ("Will the Fed decrease interest rates by 50+ bps after the September meeting?", -50.0),
        ("Will there be no change in Fed interest rates after the September meeting?", 0.0),
    ]
    ok = all(pm._bps_of(q) == expected for q, expected in cases)
    check("Fed ladder questions parse to signed basis points", ok,
          "; ".join(f"{pm._bps_of(q)}" for q, _ in cases))
    check("an ambiguous policy question is refused, not guessed",
          pm._hawkish_direction("Will the Fed change rates?") is None)
    check("esports markets are excluded from the universe",
          bool(pm.EXCLUDE.search("Counter-Strike: Black Phoenix vs CYBERSHOKE - CCT Europe")))
    check("a euro-area policy question is not excluded",
          not pm.EXCLUDE.search("Will the ECB cut rates in December?")
          and bool(pm.FINANCE_CONTEXT.search("Will the ECB cut rates in December?")))


def test_quality_grading() -> None:
    report = quality.QualityReport()
    report.add(quality.SourceStatus("Polymarket", "unavailable", "no relevant market"))
    graded = quality.grade(report, 0.9, "live", config.QUALITY_GOOD, config.QUALITY_LIMITED)
    check("an unavailable source is named in the reasons",
          any("Polymarket" in r for r in graded.reasons), str(graded.reasons))

    report2 = quality.QualityReport()
    graded2 = quality.grade(report2, 0.95, "unavailable", config.QUALITY_GOOD,
                            config.QUALITY_LIMITED)
    check("no price means POOR quality regardless of everything else",
          graded2.label == "POOR", graded2.label)


# --------------------------------------------------------------------------
# online: a real prediction
# --------------------------------------------------------------------------

def test_live(codes: list[str]) -> None:
    preds = engine.predict_all(codes)
    for code, pred in preds.items():
        payload = pred.to_dict()
        check(f"{code}: produced a prediction", bool(pred.horizons),
              pred.notes[0] if pred.notes and not pred.horizons else "")
        if not pred.horizons:
            continue
        for key, h in pred.horizons.items():
            up, down, rng = h.rounded()
            if not check(f"{code} {key}: probabilities sum to 100", up + down + rng == 100,
                         f"{up}+{down}+{rng}"):
                continue
            check(f"{code} {key}: all probabilities are finite and in range",
                  all(0 <= v <= 100 for v in (up, down, rng)))
            check(f"{code} {key}: confidence is inside 0-100",
                  0 <= h.confidence <= 100, f"{h.confidence:.1f}")
            check(f"{code} {key}: an explanation was generated from data",
                  bool(h.explanation))
        check(f"{code}: price is finite", np.isfinite(pred.price), str(pred.price))
        check(f"{code}: price status is labelled",
              pred.price_status in ("live", "delayed", "closed", "stale", "unavailable"),
              pred.price_status)
        check(f"{code}: data quality is graded",
              payload["data_quality"]["label"] in ("GOOD", "LIMITED", "POOR"),
              payload["data_quality"]["label"])
        check(f"{code}: every component declares availability",
              all("available" in c for c in payload["components"].values()))
        check(f"{code}: the payload is JSON-serialisable",
              _json_ok(payload))
        check(f"{code}: no API key leaks into the payload", not _has_secret(payload))


def _json_ok(payload: dict) -> bool:
    import json
    try:
        json.dumps(payload, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _has_secret(payload: dict) -> bool:
    """The rendered payload reaches a browser — it must carry no credentials."""
    import json
    import os
    blob = json.dumps(payload, default=str)
    for name in ("TWELVE_DATA_API_KEY", "FINNHUB_API_KEY", "FRED_API_KEY",
                 "NEWSAPI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value and value in blob:
            return True
    return "apikey" in blob.lower() or "api_key" in blob.lower()


def test_store_roundtrip(codes: list[str]) -> None:
    store.init_db()
    summary = store.summary()
    check("the SQLite store initialises", summary["rows"] >= 0, summary["db"])
    check("realized_class respects the band",
          store.realized_class(0.01, 0.005) == "UP"
          and store.realized_class(-0.01, 0.005) == "DOWN"
          and store.realized_class(0.001, 0.005) == "RANGE")
    perfect = store.brier({"UP": 1.0, "DOWN": 0.0, "RANGE": 0.0}, "UP")
    worst = store.brier({"UP": 0.0, "DOWN": 1.0, "RANGE": 0.0}, "UP")
    check("Brier score is 0 when perfect and 2 when maximally wrong",
          perfect == 0.0 and worst == 2.0, f"{perfect} / {worst}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip the checks that call live APIs")
    ap.add_argument("assets", nargs="*", default=None)
    args = ap.parse_args(argv)
    codes = [c.upper() for c in (args.assets or asset_spec.DEFAULT_CODES)]

    print(f"Self-test · model {config.MODEL_VERSION}")
    test_distribution()
    test_rounding()
    test_weight_redistribution()
    test_confidence_degrades()
    test_missing_component_is_not_neutral_evidence()
    test_market_calendar()
    test_variance_time()
    test_polymarket_direction_parsing()
    test_quality_grading()
    test_store_roundtrip(codes)
    if not args.offline:
        print("  ...running live checks (this calls the real APIs)")
        test_live(codes)

    width = max(len(name) for _, name, _ in results) + 2
    print()
    for status, name, detail in results:
        line = f"  [{status}] {name:<{width}}"
        if detail:
            line += f" {detail}"
        print(line)

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:")
        for _, name, detail in failed:
            print(f"  - {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
