from __future__ import annotations

import os
import time

import httpx

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


class RateClient:
    def __init__(self, headers: dict[str, str], pause_s: float = 0.8):
        self._client = httpx.Client(headers=headers, timeout=30.0, follow_redirects=True)
        self._pause_s = pause_s

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        time.sleep(self._pause_s)
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()
