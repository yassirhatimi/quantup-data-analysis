"""Regression checks for historical information boundaries (offline)."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from quantlab.predict import backtest


def bars(n=700):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-06", periods=n, freq="h", tz="UTC")
    close = 2000 * np.exp(np.cumsum(rng.normal(0, .001, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.002,
                         "Low": close * .998, "Close": close}, index=idx)


class ReplayTests(unittest.TestCase):
    def test_future_prices_cannot_change_earlier_features(self):
        original = bars()
        changed = original.copy()
        changed.iloc[230:] *= np.linspace(1, 3, len(changed) - 230)[:, None]
        a = backtest._vectorised_features(original, "XAUUSD")
        b = backtest._vectorised_features(changed, "XAUUSD")
        pd.testing.assert_frame_equal(a.iloc[:230], b.iloc[:230])

    def test_future_prices_cannot_change_earlier_predictions(self):
        original = bars()
        changed = original.copy()
        changed.iloc[450:] *= np.linspace(1, 3, len(changed) - 450)[:, None]
        a = {s.ts: s for s in backtest.build_samples("XAUUSD", frame=original)}
        b = {s.ts: s for s in backtest.build_samples("XAUUSD", frame=changed)}
        checked = 0
        for ts, sample in a.items():
            if sample.outcome_ts < original.index[450]:
                self.assertEqual(sample, b[ts])
                checked += 1
        self.assertGreater(checked, 100)

    def test_split_excludes_unmatured_training_labels(self):
        samples = backtest.build_samples("XAUUSD", horizon_bars=24, frame=bars())
        train, test = backtest._split_samples(samples, .7)
        self.assertTrue(all(s.outcome_ts < test[0].ts for s in train))
        self.assertLess(len(train) + len(test), len(samples))

    def test_baseline_does_not_fit_evaluation_outcomes(self):
        sample = backtest.build_samples("XAUUSD", frame=bars())[0]
        training = [replace(sample, actual="UP"), replace(sample, actual="DOWN")]
        evaluation = [replace(sample, actual="RANGE")]
        result = backtest._baseline(evaluation, training)
        self.assertEqual(result["frequencies"], {"UP": .5, "DOWN": .5, "RANGE": 0.})
        self.assertEqual(result["accuracy"], 0.)

    def test_missing_bars_do_not_extend_the_horizon(self):
        frame = bars().drop(bars().index[300:303])
        samples = backtest.build_samples("XAUUSD", horizon_bars=4, frame=frame)
        self.assertTrue(samples)
        self.assertTrue(all(s.outcome_ts - s.ts == pd.Timedelta(hours=4) for s in samples))
        self.assertFalse(any(s.ts == frame.index[298] for s in samples))

    def test_first_trade_loss_counts_as_drawdown(self):
        sample = backtest.build_samples("XAUUSD", frame=bars())[0]
        result = backtest._trading_sim([replace(sample, forward_return=-.01)],
                  [{"UP": .8, "DOWN": .1, "confidence": 80}], "XAUUSD")
        self.assertLess(result["max_drawdown"], -.01)


if __name__ == "__main__":
    unittest.main()
