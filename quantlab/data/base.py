"""Shared HTTP plumbing for the REST data clients."""

from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_TIMEOUT = 20
USER_AGENT = "quantlab/0.1 (quant research toolkit)"

_session: requests.Session | None = None


class DataSourceError(RuntimeError):
    """An API returned an error status or an unusable payload."""


def session() -> requests.Session:
    """Process-wide session so connections are reused across requests."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
    return _session


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """GET a JSON endpoint with retries on rate limits and transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session().get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                last_error = DataSourceError(f"HTTP 429 (rate limited) from {url}")
                time.sleep(backoff * (2**attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            time.sleep(backoff * (2**attempt))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body = exc.response.text[:300] if exc.response is not None else ""
            raise DataSourceError(f"HTTP {status} from {url}: {body}") from exc
        except ValueError as exc:  # not JSON
            raise DataSourceError(f"Non-JSON response from {url}") from exc
    raise DataSourceError(f"Failed to fetch {url} after {retries} attempts") from last_error
