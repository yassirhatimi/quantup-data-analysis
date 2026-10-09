"""Backtest and calibrate the probability engine.

    python scripts/predict_backtest.py                       # all assets, all horizons
    python scripts/predict_backtest.py XAUUSD --horizon 4h
    python scripts/predict_backtest.py --optimise            # fit weights, print result
    python scripts/predict_backtest.py --optimise --write    # ...and save weights.json
    python scripts/predict_backtest.py --live                # accuracy of real stored calls

The headline question this answers is the one that matters: **when the model
says 60%, does it happen 60% of the time?** That is the calibration table, and
it is printed for every run.

What is replayed is the chart-derived half of the engine. Macro, news and
Polymarket have no point-in-time archive — FRED revises in place, the free
NewsAPI window is a month, and today's Fed markets did not exist last year — so
they are excluded rather than backfilled with values that would leak the
future. `--live` reports the *whole* engine's accuracy from the SQLite store,
which fills up as predictions mature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantlab.predict import assets as asset_spec
from quantlab.predict import backtest, config, store

HORIZONS = ["15m", "1h", "4h", "session", "24h"]


def print_report(report: backtest.Report, full: bool = True) -> None:
    print()
    print(report.summary_text())
    if not full:
        return

    print("\n  Calibration — does a forecast of X% happen X% of the time?")
    print("    " + "-" * 62)
    print(f"    {'forecast bucket':<18}{'n':>7}{'predicted':>12}{'observed':>11}{'gap':>10}")
    for _, r in report.calibration.iterrows():
        if r["n"] < 20:
            continue
        print(f"    {str(r['bucket']):<18}{int(r['n']):>7}"
              f"{r['mean_forecast']:>11.1%}{r['observed']:>11.1%}{r['gap']:>+10.1%}")

    print("\n  Per class")
    print(f"    {'class':<8}{'predicted':>11}{'actual':>9}{'precision':>11}"
          f"{'recall':>9}{'f1':>8}")
    for _, r in report.class_report.iterrows():
        print(f"    {r['class']:<8}{int(r['predicted']):>11}{int(r['actual']):>9}"
              f"{_pct(r['precision']):>11}{_pct(r['recall']):>9}{_pct(r['f1']):>8}")

    for title, table in (("By market regime", report.by_regime),
                         ("By session", report.by_session)):
        if table.empty:
            continue
        print(f"\n  {title}")
        print(f"    {'group':<18}{'n':>7}{'brier':>9}{'baseline':>10}{'skill':>9}"
              f"{'accuracy':>10}")
        for _, r in table.iterrows():
            print(f"    {r['group']:<18}{int(r['n']):>7}{r['brier']:>9.4f}"
                  f"{r['baseline_brier']:>10.4f}{r['skill']:>+9.1%}"
                  f"{r['accuracy']:>10.1%}")


def _pct(v) -> str:
    return "—" if v is None or v != v else f"{v:.1%}"


def run_live_report() -> int:
    """Accuracy of the *whole* engine, from predictions that have matured."""
    summary = store.summary()
    print("\nStored prediction history (the full engine, live)")
    print(f"  rows {summary['rows']:,} · scored {summary['scored']:,} · "
          f"pending {summary['pending']:,}")
    print(f"  db: {summary['db']}")
    if not summary["scored"]:
        print("  Nothing has matured yet. Run the live engine for a while — the "
              "15m horizon starts scoring within the hour, 24h by tomorrow.")
        return 0
    print(f"  mean Brier {summary['mean_brier']:.4f} · "
          f"mean log loss {summary['mean_log_loss']:.4f} · "
          f"hit rate {summary['hit_rate']:.1%}")
    cal = store.calibration_table()
    if not cal.empty:
        print("\n  Calibration of live predictions")
        for _, r in cal.iterrows():
            print(f"    {str(r['bucket']):<18}{int(r['n']):>7}"
                  f"{r['mean_forecast']:>11.1%}{r['observed_frequency']:>11.1%}"
                  f"{r['gap']:>+10.1%}")
    record = store.track_record()
    if record:
        print("\n  Brier skill vs climatology, by asset and horizon")
        for code, per in record.items():
            for tf, value in per.items():
                print(f"    {code:<8}{tf:<10}{value:.2f}  (0.5 = no skill)")
    return 0


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="*", default=None)
    ap.add_argument("--horizon", default=None, choices=HORIZONS)
    ap.add_argument("--optimise", "--optimize", action="store_true", dest="optimise")
    ap.add_argument("--write", action="store_true",
                    help="save optimised weights to quantlab/predict/weights.json")
    ap.add_argument("--live", action="store_true",
                    help="report stored live-prediction accuracy instead")
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--brief", action="store_true", help="summary lines only")
    args = ap.parse_args(argv)

    if args.live:
        return run_live_report()

    codes = [c.upper() for c in (args.assets or asset_spec.DEFAULT_CODES)]
    horizons = [args.horizon] if args.horizon else HORIZONS

    print(f"Backtesting {', '.join(codes)} over {', '.join(horizons)}")
    print(f"Model {config.MODEL_VERSION} · band {config.BAND_K:g} sigma · "
          f"kappa cap {config.KAPPA_MAX:g}"
          + (f" · overrides active: {list(config.ACTIVE_OVERRIDES)}"
             if config.ACTIVE_OVERRIDES else ""))

    if args.optimise:
        results = []
        for code in codes:
            for horizon in horizons:
                try:
                    r = backtest.optimise(code, horizon, trials=args.trials,
                                          max_samples=args.max_samples)
                except ValueError as exc:
                    print(f"  {code} {horizon}: {exc}")
                    continue
                results.append(r)
                flag = "IMPROVED" if r["improved"] else "no improvement"
                print(f"  {code:<8}{horizon:<9} test Brier {r['test_brier']:.5f} vs "
                      f"default {r['default_test_brier']:.5f}  [{flag}]  "
                      f"kappa {r['kappa']:.2f}  "
                      + " ".join(f"{k[:4]}={v:.2f}" for k, v in r["weights"].items()))
        payload = backtest.to_overrides(results)
        if not payload:
            print("\nNo configuration beat the defaults out of sample — "
                  "weights.json left untouched. That is a result, not a failure.")
            return 0
        print("\nProposed overrides:")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        if args.write:
            path = config.save_overrides(payload)
            print(f"\nWritten to {path} — it is applied on the next import.")
        else:
            print("\nRe-run with --write to save these to weights.json.")
        return 0

    failures = 0
    for code in codes:
        for horizon in horizons:
            try:
                report = backtest.run(code, horizon, max_samples=args.max_samples)
            except Exception as exc:
                print(f"\n{code} {horizon}: FAILED — {type(exc).__name__}: {exc}")
                failures += 1
                continue
            print_report(report, full=not args.brief)

    print("\n" + "-" * 70)
    print("Replayed: technical, price action, volatility, structure, regime, "
          "historical analogue.")
    print("Not replayed (no point-in-time archive): macro, news, Polymarket. "
          "Their live accuracy accrues in the SQLite store — "
          "see `--live`.")
    print("The trading simulation books one unit per non-overlapping signal at the "
          "asset's round-trip spread. It is a yardstick, not a strategy.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
