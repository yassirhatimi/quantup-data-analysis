"""Reproduce aggregate review evidence from the existing local SQLite snapshot.

No network calls and no writes to the source database. Output files are local
review artifacts, not independently cleared redistribution material.
"""
from pathlib import Path
import json
import sqlite3
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab.predict import backtest, market_data


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source = ROOT / "gold_wave_analysis/gold_waves.sqlite"
    if not source.exists():
        raise SystemExit("Local research snapshot missing; acquire permitted data first.")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        bars = pd.read_sql_query("SELECT timestamp_utc, open, high, low, close FROM price_bars WHERE timeframe='1h' AND quality_flag='ok' ORDER BY timestamp_utc", conn)
        quality = pd.read_sql_query("SELECT * FROM data_quality", conn)
    bars.index = pd.to_datetime(bars.pop("timestamp_utc"), utc=True) + pd.Timedelta(hours=1)
    bars.columns = [c.title() for c in bars.columns]
    bars = market_data.drop_closed(bars)
    report = backtest.run("XAUUSD", horizon="4h", frame=bars, max_samples=1200)
    payload = {
        "review_date": "2026-10-07",
        "source": "Local Yahoo GC=F hourly futures snapshot; not spot XAU/USD",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bar_timestamp": "Source bar-start labels shifted one hour to completion time",
        "scope": "Chart-derived replay, four-hour windows with all four bars present; closure filter is approximate",
        "pricing_limit": "XAUUSD model settings applied to a futures proxy; not executable spot performance",
        "split": "Chronological 70/30 with training outcomes purged at evaluation boundary",
        "n_train": report.extras["n_train"], "n_test": report.n,
        "test_start": str(report.period[0]), "test_end": str(report.period[1]),
        "metrics": report.metrics, "baseline": report.baseline,
        "calibration": report.calibration.assign(bucket=report.calibration.bucket.astype(str)).to_dict("records"),
        "quality": quality.to_dict("records"),
    }
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    out = ROOT / "docs/validation_metrics.json"
    out.write_text(json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8")
    print(report.summary_text())
    print("Saved aggregate evidence to docs/validation_metrics.json")


if __name__ == "__main__":
    main()
