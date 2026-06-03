"""Fetch and cache model pricing data from models.dev/api.json."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_URL = "https://models.dev/api.json"
CACHE_PATH = Path.home() / ".cache" / "ducktrace" / "models.json"
TTL_SECONDS = 86_400  # 24 hours


def _cache_load() -> tuple[Optional[dict], float]:
    """Return (data, age_seconds) from the on-disk cache."""
    if not CACHE_PATH.exists():
        return None, float("inf")
    try:
        raw = json.loads(CACHE_PATH.read_text())
        age = time.time() - raw.get("_ts", 0)
        return raw.get("data"), age
    except Exception:
        return None, float("inf")


def _cache_write(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"_ts": time.time(), "data": data}))


def get_models() -> dict:
    """Return models.dev API payload, fetching from network if the cache is stale.

    Set ``DUCKTRACE_MODELS_CACHE_OFFLINE=1`` to disable network calls entirely
    and use only the on-disk cache (or an empty dict as last resort).  This is
    useful in CI/test environments where outbound network access is restricted.
    """
    cached, age = _cache_load()
    if cached is not None and age < TTL_SECONDS:
        return cached
    if os.environ.get("DUCKTRACE_MODELS_CACHE_OFFLINE"):
        logger.debug("DUCKTRACE_MODELS_CACHE_OFFLINE set; skipping network fetch")
        return cached if cached is not None else {}
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=10) as resp:
            if resp.status != 200:
                raise ValueError(f"models.dev returned HTTP {resp.status}")
            data = json.loads(resp.read())
        _cache_write(data)
        return data
    except Exception:
        return cached if cached is not None else {}


def find_model_cost(model_id: str) -> Optional[dict]:
    """Return the ``cost`` entry for *model_id* from the models.dev cache.

    Searches the canonical provider first (``anthropic`` for claude-* models,
    ``openai`` for all others), then all remaining providers.  The returned
    dict uses USD per-million-token rates with keys ``input``, ``output``,
    and optionally ``cache_read`` / ``cache_write``.  Returns ``None`` when
    the model is not found.
    """
    data = get_models()
    if not data:
        return None

    prefix = (model_id.split("-")[0] or "").lower()
    preferred = "anthropic" if prefix == "claude" else "openai"
    search_order = [preferred] + [k for k in data if k != preferred]

    for provider_id in search_order:
        models = data.get(provider_id, {}).get("models", {})
        if model_id in models:
            cost = models[model_id].get("cost")
            inp = cost.get("input") if cost else None
            out = cost.get("output") if cost else None
            if (
                cost
                and isinstance(inp, (int, float)) and inp > 0
                and isinstance(out, (int, float)) and out > 0
            ):
                return cost
    return None
