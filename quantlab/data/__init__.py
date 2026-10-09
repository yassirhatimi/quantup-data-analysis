"""Data clients, one module per source.

Import the module you need explicitly, e.g.::

    from quantlab.data import yahoo, binance, fred

Each module is self-contained so a missing optional dependency or API key
in one source never breaks the others.
"""

__all__ = [
    "yahoo",
    "finnhub",
    "twelvedata",
    "fred",
    "news",
    "polymarket",
    "binance",
]
