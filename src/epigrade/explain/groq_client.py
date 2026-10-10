"""Minimal Groq chat-completions client (OpenAI-compatible REST), using `requests` - one call
does not justify an SDK dependency.

Every failure (no key, timeout, HTTP error, rate limit, malformed reply) returns None so the
caller falls back to the deterministic template. The API key is read from GROQ_API_KEY and is
never logged, cached or returned.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import requests
import yaml

from epigrade import paths

KEY_VARIABLE = "GROQ_API_KEY"


@lru_cache(maxsize=1)
def config() -> dict:
    with open(paths.repo_root() / "config" / "explain.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_key() -> bool:
    return bool(os.environ.get(KEY_VARIABLE, "").strip())


def chat(messages: list[dict]) -> tuple[str | None, str]:
    """Returns (text, status). status is "ok", or the reason text is None: "no_api_key",
    "timeout", "rate_limited", "http_<code>", "network_error", "bad_response"."""
    key = os.environ.get(KEY_VARIABLE, "").strip()
    if not key:
        return None, "no_api_key"
    cfg = config()
    payload = {
        "model": cfg["model"], "messages": messages,
        "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    attempts = 1 + int(cfg.get("retry_on_rate_limit", 1))
    for attempt in range(attempts):
        try:
            response = requests.post(cfg["endpoint"], json=payload, headers=headers,
                                     timeout=cfg["timeout_seconds"])
        except requests.Timeout:
            return None, "timeout"
        except requests.RequestException:
            return None, "network_error"
        if response.status_code == 429:
            if attempt + 1 < attempts:
                wait = response.headers.get("retry-after")
                try:
                    delay = min(float(wait), 30.0) if wait else cfg["rate_limit_backoff_seconds"]
                except ValueError:
                    delay = cfg["rate_limit_backoff_seconds"]
                time.sleep(delay)
                continue
            return None, "rate_limited"
        if response.status_code != 200:
            return None, f"http_{response.status_code}"
        try:
            text = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return None, "bad_response"
        return (text.strip() if isinstance(text, str) and text.strip() else None,
                "ok" if text else "bad_response")
    return None, "rate_limited"
