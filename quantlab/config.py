"""API-key handling.

Keys are read from environment variables, optionally loaded from a `.env`
file (see `.env.example` in the project root). Sources that need no key:
Yahoo Finance, Binance public market data, Polymarket.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; plain env vars still work
    load_dotenv = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MissingAPIKeyError(RuntimeError):
    """A required API key is not set in the environment or .env file."""


_SIGNUP_URLS = {
    "FINNHUB_API_KEY": "https://finnhub.io/register",
    "TWELVE_DATA_API_KEY": "https://twelvedata.com",
    "FRED_API_KEY": "https://fred.stlouisfed.org/docs/api/api_key.html",
    "NEWSAPI_API_KEY": "https://newsapi.org/register",
}


def load_env() -> None:
    """Load `.env` from the project root and the current working directory."""
    if load_dotenv is None:
        return
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv()  # cwd .env, if any — does not override existing vars


load_env()


def get_key(name: str, required: bool = True) -> str | None:
    """Return the API key `name`, or raise a helpful error if it is missing."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    if not required:
        return None
    msg = (
        f"Missing API key: {name}. Set it as an environment variable or in a "
        f"`.env` file in the project root (copy `.env.example` to `.env`)."
    )
    if name in _SIGNUP_URLS:
        msg += f" Get a free key at {_SIGNUP_URLS[name]}"
    raise MissingAPIKeyError(msg)


def has_key(name: str) -> bool:
    """True if the key is configured (useful to skip optional notebook cells)."""
    return bool(os.getenv(name, "").strip())
