import unittest

import numpy as np
import pandas as pd

from dashboard.build_dashboard import correlation_stats, fmt, render_correlation


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-08-20", periods=51)
        self.returns = np.random.default_rng(7).normal(0, 0.01, 51)
        self.prices = pd.Series(100 * np.cumprod(1 + self.returns), index=self.dates)

    def test_returns_and_yield_changes(self):
        opposite = pd.Series(100 * np.cumprod(1 - self.returns), index=self.dates)
        yields = pd.Series(4 + np.cumsum(self.returns), index=self.dates)
        result = correlation_stats({"XAUUSD": self.prices, "EURUSD": opposite,
                                    "US10Y": yields}, as_of="2026-10-09")
        self.assertAlmostEqual(result["matrix"].loc["XAUUSD", "EURUSD"], -1)
        self.assertAlmostEqual(result["matrix"].loc["XAUUSD", "US10Y"], 1)
        self.assertEqual(result["counts"].loc["XAUUSD", "EURUSD"], 30)
        self.assertEqual(result["end"], "2026-10-08")

    def test_current_day_is_excluded(self):
        modified = self.prices.copy()
        modified.loc["2026-10-09"] = 100000
        before = correlation_stats({"XAUUSD": self.prices}, as_of="2026-10-09")
        after = correlation_stats({"XAUUSD": modified}, as_of="2026-10-09")
        pd.testing.assert_frame_equal(before["matrix"], after["matrix"])
        pd.testing.assert_frame_equal(before["counts"], after["counts"])

    def test_missing_data_is_not_filled(self):
        missing = self.prices.copy()
        missing.loc["2026-10-01"] = np.nan
        result = correlation_stats({"XAUUSD": missing}, as_of="2026-10-09")
        self.assertEqual(result["counts"].loc["XAUUSD", "XAUUSD"], 28)
        self.assertTrue(np.isnan(result["matrix"].loc["XAUUSD", "DXY"]))

    def test_insufficient_and_constant_series(self):
        result = correlation_stats({"EURUSD": self.prices.tail(5),
                                    "USOIL": self.prices * 0 + 50}, as_of="2026-10-09")
        self.assertTrue(result["matrix"].isna().all().all())
        self.assertIn("N/A", render_correlation(result))
        self.assertEqual(fmt(1.12345, "EURUSD"), "1.12345")
        self.assertEqual(fmt(75.12, "USOIL"), "75.12")


if __name__ == "__main__":
    unittest.main()
