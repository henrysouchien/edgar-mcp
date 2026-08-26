"""Portable process-local SEC transport for the public edgar-mcp package."""

from __future__ import annotations

from collections.abc import Mapping
import threading
import time
from typing import Any

import requests


_SEC_USER_AGENT = "edgar-parser henry@edgarparser.com"
_THROTTLE_LOCK = threading.Lock()
_last_call_monotonic: float | None = None


def sec_get(
    url: str,
    *,
    caller_label: str,
    headers: Mapping[str, Any] | None = None,
    timeout: tuple[float, float] | float | None = None,
    **kwargs: Any,
) -> requests.Response:
    """Issue one portable SEC GET with the established process-local pacing."""

    if not isinstance(caller_label, str) or not caller_label.strip():
        raise ValueError("caller_label must be a non-empty string")
    request_headers = {
        name: value
        for name, value in dict(headers or {}).items()
        if str(name).lower() != "user-agent"
    }
    request_headers["User-Agent"] = _SEC_USER_AGENT

    global _last_call_monotonic
    with _THROTTLE_LOCK:
        now = time.monotonic()
        if _last_call_monotonic is not None:
            delay = 1.0 - (now - _last_call_monotonic)
            if delay > 0:
                time.sleep(delay)
        _last_call_monotonic = time.monotonic()
        return requests.get(
            url,
            headers=request_headers,
            timeout=timeout,
            **kwargs,
        )


__all__ = ["sec_get"]
