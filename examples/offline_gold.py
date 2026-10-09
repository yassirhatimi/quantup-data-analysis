"""Demonstrate event labels with synthetic prices; no network or saved data.

Run from the repository root: python -m examples.offline_gold
"""
import pandas as pd
from gold_wave_analysis.gold_waves import impulse_events


def main():
    paths = {
        "Continuation after correction": [4300, 4360, 4450, 4600, 4650, 4550, 4500, 4660],
        "Full retracement": [4300, 4360, 4600, 4500, 4300],
        "Unfinished at end of data": [4300, 4360, 4600, 4650],
    }
    print("SYNTHETIC EXAMPLES - not observed market prices.\n")
    print("Defaults: $50 reversal, $300 move, $100 correction, 60-day horizon.\n")
    for label, prices in paths.items():
        times = pd.date_range("2025-01-01", periods=len(prices), freq="h", tz="UTC")
        print(label)
        print(f"  Prices: {prices}")
        for event in impulse_events(prices, times):
            print(f"  Outcome: {event['outcome']}")
        print()
    print("Unfinished events are not counted as wins or losses.")
    print("These examples demonstrate labels, not predictive accuracy.")


if __name__ == "__main__":
    main()
