"""
API Clients — Shared HTTP session and retry logic for external data sources.

Centralises rate-limiting, retry backoff, and session reuse so individual
tools (news_tool, stock_tool, economic_tool) don't each manage their own
connection pools.
"""
import os
import time
from typing import Any, Dict, Optional
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger


def _build_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Build a requests.Session with automatic retry and backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Module-level shared session (reused across calls)
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Return the module-level shared requests session (lazy-initialized)."""
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def get(url: str, params: Optional[Dict] = None, timeout: int = 10, **kwargs) -> requests.Response:
    """Perform a GET request using the shared session."""
    logger.debug(f"GET {url} params={params}")
    return get_session().get(url, params=params, timeout=timeout, **kwargs)


def post(url: str, json: Optional[Dict] = None, timeout: int = 10, **kwargs) -> requests.Response:
    """Perform a POST request using the shared session."""
    logger.debug(f"POST {url}")
    return get_session().post(url, json=json, timeout=timeout, **kwargs)
