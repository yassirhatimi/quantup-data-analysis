# Quantup — Financial Data Analysis

Python tools for checking market data, analysing gold price movements, and
building repeatable research reports with pandas, SQLite and matplotlib.

The main question: after a large gold price move, how can we describe the
following correction without using future information or counting unfinished
events as successes?

## Start with an offline example

Requires Python 3.10 or later. From this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install .
python -m examples.offline_gold
```

The example uses small, explicitly synthetic price paths. It needs no internet
access or API keys. It demonstrates three event outcomes; these are not
historical market findings.

## Method

- Retain raw observations and flag invalid or unfinished bars.
- Confirm turning points only after a defined reversal.
- Track large moves, corrections and unresolved events separately.
- Store observations and results in SQLite so summaries can be checked.

Start with [the example](examples/offline_gold.py), then read
[the gold analysis](gold_wave_analysis/gold_waves.py) and its
[tests](gold_wave_analysis/test_gold_waves.py).

Daily, hourly and 15-minute series are analysed separately. The optional
download workflow uses GC=F futures, not spot XAU/USD. Fixed dollar thresholds
are assumptions and can behave differently at different price levels.

## Project map

| Folder | Contents |
| --- | --- |
| gold_wave_analysis | Gold event analysis and data-quality checks |
| quantlab/data | Price, economic and news data clients |
| quantlab/analysis | Return, risk and signal calculations |
| quantlab/backtest | Strategy research with cost assumptions |
| quantlab/predict | Experimental forecasts and evaluation tools |
| dashboard | Reports that fetch external data |
| tests | Offline regression checks |

Forecasts are experimental. No profitable trading strategy is established,
and confidence scores are not measured win probabilities.

## Verification

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s gold_wave_analysis -p test_gold_waves.py -v
python scripts/predict_selftest.py --offline
```

See [local verification](docs/RELEASE_CHECKS.md). An offline GitHub Actions
workflow is included.

## Optional external data

Copy `.env.example` to `.env` and supply only the keys needed. Keep it private.
For an internet-backed example, run `python -m examples.quickstart`.

Saved provider data, reports and notebooks are excluded from this edition.
`verify_saved.py`, `review_saved_research.py` and `build_case_study.py` need a
separate local research snapshot; they cannot run from this copy alone.

## License

Original code: [MIT](LICENSE). See the [data policy](docs/DATA_POLICY.md)
for external-data handling.

Maintained by [yassirhatimi-code](https://github.com/yassirhatimi-code).
