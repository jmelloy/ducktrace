"""Model pricing tables and cost calculation.

Ported from ../ai-observer/backend/internal/pricing. Rates are USD per token
(the source JSON uses per-million-token rates; we divide by 1e6 here).

Cost model:
  * Claude: input + output + cache-read + cache-creation, each at its own rate.
            ``claude_cost`` always computes from tokens. The caller is responsible
            for also capturing ``costUSD`` from the JSONL as ``stated_cost``.
  * Codex:  input (non-cached) + cache-read + output. Codex reports *cumulative*
            token counts, so the caller passes per-turn deltas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from claude_analysis.models_cache import find_model_cost

logger = logging.getLogger(__name__)

_M = 1_000_000.0
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class ModelPricing:
    input: float          # per token
    output: float         # per token
    cache_read: float     # per token
    cache_write: float    # per token (Claude only; 0 for Codex)


def _mk(input_m, output_m, cache_read_m=0.0, cache_write_m=0.0) -> ModelPricing:
    return ModelPricing(input_m / _M, output_m / _M, cache_read_m / _M, cache_write_m / _M)


# --- Claude (Anthropic) -------------------------------------------------------
# canonical name -> pricing, plus alias -> canonical
_CLAUDE: dict[str, ModelPricing] = {
    "claude-opus-4-6-20260301": _mk(5, 25, 0.5, 6.25),
    "claude-sonnet-4-6-20260301": _mk(3, 15, 0.3, 3.75),
    "claude-sonnet-4-5-20250929": _mk(3, 15, 0.3, 3.75),
    "claude-haiku-4-5-20251001": _mk(1, 5, 0.1, 1.25),
    "claude-opus-4-5-20251101": _mk(5, 25, 0.5, 6.25),
    "claude-opus-4-1-20250805": _mk(15, 75, 1.5, 18.75),
    "claude-sonnet-4-20250514": _mk(3, 15, 0.3, 3.75),
    "claude-opus-4-20250514": _mk(15, 75, 1.5, 18.75),
    "claude-3-7-sonnet-20250219": _mk(3, 15, 0.3, 3.75),
    "claude-3-5-sonnet-20241022": _mk(3, 15, 0.3, 3.75),
    "claude-3-5-sonnet-20240620": _mk(3, 15, 0.3, 3.75),
    "claude-3-5-haiku-20241022": _mk(0.8, 4, 0.08, 1),
    "claude-3-opus-20240229": _mk(15, 75, 1.5, 18.75),
    "claude-3-sonnet-20240229": _mk(3, 15, 0.3, 3.75),
    "claude-3-haiku-20240307": _mk(0.25, 1.25, 0.03, 0.3),
}

_CLAUDE_ALIASES: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4-6-20260301",
    "claude-opus-4-6-latest": "claude-opus-4-6-20260301",
    "claude-opus-4-8": "claude-opus-4-6-20260301",  # newer family, fall back to opus 4.6 rates
    "claude-opus-4-8-1m": "claude-opus-4-6-20260301",
    "claude-sonnet-4-6": "claude-sonnet-4-6-20260301",
    "claude-sonnet-4-6-latest": "claude-sonnet-4-6-20260301",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-latest": "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-latest": "claude-haiku-4-5-20251001",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-opus-4-5-latest": "claude-opus-4-5-20251101",
    "claude-opus-4-1": "claude-opus-4-1-20250805",
    "claude-opus-4-1-latest": "claude-opus-4-1-20250805",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    "claude-sonnet-4-latest": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-opus-4-0": "claude-opus-4-20250514",
    "claude-opus-4-latest": "claude-opus-4-20250514",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest": "claude-3-7-sonnet-20250219",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-v2": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-v1": "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-haiku-3-5": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-opus-latest": "claude-3-opus-20240229",
    "claude-opus-3": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}

# --- Codex (OpenAI) -----------------------------------------------------------
_CODEX: dict[str, ModelPricing] = {
    "gpt-5": _mk(1.25, 10, 0.125),
    "gpt-5.1": _mk(1.25, 10, 0.125),
    "gpt-5.2": _mk(1.75, 14, 0.175),
    "gpt-5.4": _mk(2.5, 15, 0.25),
    "gpt-5.5": _mk(5, 30, 0.5),          # per developers.openai.com/api/docs/pricing
    "gpt-5.5-codex": _mk(5, 30, 0.5),
    "gpt-5.4-mini": _mk(0.75, 4.5, 0.075),
    "gpt-5.4-nano": _mk(0.2, 1.25, 0.02),
    "gpt-5.4-pro": _mk(30, 180, 0),
    "gpt-5-mini": _mk(0.25, 2, 0.025),
    "gpt-5-nano": _mk(0.05, 0.4, 0.005),
    "gpt-5-pro": _mk(15, 120, 0),
    "gpt-5.2-pro": _mk(21, 168, 0),
    "gpt-5-chat-latest": _mk(1.25, 10, 0.125),
    "gpt-5.1-chat-latest": _mk(1.25, 10, 0.125),
    "gpt-5.2-chat-latest": _mk(1.75, 14, 0.175),
    "gpt-5.3-chat-latest": _mk(1.75, 14, 0.175),
    "gpt-5-codex": _mk(1.25, 10, 0.125),
    "gpt-5.1-codex": _mk(1.25, 10, 0.125),
    "gpt-5.1-codex-max": _mk(1.25, 10, 0.125),
    "gpt-5.1-codex-mini": _mk(0.25, 2, 0.025),
    "gpt-5.3-codex": _mk(1.75, 14, 0.175),
    "gpt-5-search-api": _mk(1.25, 10, 0.125),
    "codex-mini-latest": _mk(1.5, 6, 0.375),
    "gpt-4.1": _mk(2, 8, 0.5),
    "gpt-4.1-mini": _mk(0.4, 1.6, 0.1),
    "gpt-4.1-nano": _mk(0.1, 0.4, 0.025),
    "gpt-4o": _mk(2.5, 10, 1.25),
    "gpt-4o-mini": _mk(0.15, 0.6, 0.075),
    "o1": _mk(15, 60, 7.5),
    "o1-mini": _mk(1.1, 4.4, 0.55),
    "o1-pro": _mk(150, 600, 0),
    "o3": _mk(2, 8, 0.5),
    "o3-mini": _mk(1.1, 4.4, 0.55),
    "o3-pro": _mk(20, 80, 0),
    "o4-mini": _mk(1.1, 4.4, 0.275),
}

_CODEX_ALIASES: dict[str, str] = {
    "gpt-5-chat": "gpt-5-chat-latest",
    "gpt-5.1-chat": "gpt-5.1-chat-latest",
    "gpt-5.2-chat": "gpt-5.2-chat-latest",
    "gpt-5.3-chat": "gpt-5.3-chat-latest",
}


def _lookup(table, aliases, model: str) -> ModelPricing | None:
    if not model:
        return None
    if model in table:
        return table[model]
    if model in aliases:
        return table.get(aliases[model])
    return None


def _api_pricing(model_id: str) -> ModelPricing | None:
    """Build ModelPricing from the models.dev API cache.

    Tries the raw model ID first; if not found, strips an 8-digit date suffix
    (e.g. ``-20260301``) and retries so versioned canonical IDs resolve to
    the dateless IDs used by models.dev.

    NOTE: callers should merge the returned pricing with hardcoded values so
    that cache rates fall back to the hardcoded table when the API omits them
    (models.dev does not always include cache_read / cache_write).
    """
    cost = find_model_cost(model_id)
    if cost is None:
        stripped = _DATE_SUFFIX_RE.sub("", model_id)
        if stripped != model_id:
            cost = find_model_cost(stripped)
    if cost is None:
        return None
    return _mk(
        cost.get("input", 0.0),
        cost.get("output", 0.0),
        cost.get("cache_read", 0.0),
        cost.get("cache_write", 0.0),
    )


def _merge_cache_rates(api: ModelPricing, hardcoded: ModelPricing | None) -> ModelPricing:
    """Return *api* pricing, filling in zero cache rates from *hardcoded* if available."""
    if hardcoded is None:
        return api
    if api.cache_read > 0 and api.cache_write > 0:
        return api
    return ModelPricing(
        input=api.input,
        output=api.output,
        cache_read=api.cache_read if api.cache_read > 0 else hardcoded.cache_read,
        cache_write=api.cache_write if api.cache_write > 0 else hardcoded.cache_write,
    )


def claude_pricing(model: str) -> ModelPricing | None:
    """Return per-token pricing for a Claude model.

    PRECEDENCE: models.dev API data is preferred over the hardcoded table for
    input/output rates.  If the API omits cache rates (cache_read / cache_write),
    hardcoded rates are used as a fallback.  If models.dev ever returns incorrect
    input/output prices for a well-known model, those wrong rates will be used;
    monitor logs at DEBUG level for pricing source details.
    """
    model = (model or "").strip()
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    api = _api_pricing(model)
    hardcoded = _lookup(_CLAUDE, _CLAUDE_ALIASES, model)
    if api is not None:
        logger.debug("claude_pricing(%s): using models.dev API rates (input=%s, output=%s)", model, api.input, api.output)
        return _merge_cache_rates(api, hardcoded)
    return hardcoded


def codex_pricing(model: str) -> ModelPricing | None:
    """Return per-token pricing for a Codex/OpenAI model.

    PRECEDENCE: models.dev API data is preferred over the hardcoded table for
    input/output rates.  If the API omits cache rates, hardcoded rates are used
    as a fallback.  If models.dev ever returns incorrect prices, those wrong
    rates will be used; monitor logs at DEBUG level for pricing source details.
    """
    model = (model or "").strip()
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    api = _api_pricing(model)
    hardcoded = _lookup(_CODEX, _CODEX_ALIASES, model)
    if api is not None:
        logger.debug("codex_pricing(%s): using models.dev API rates (input=%s, output=%s)", model, api.input, api.output)
        return _merge_cache_rates(api, hardcoded)
    return hardcoded


def claude_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> float:
    """Cost for a Claude assistant turn computed from token counts."""
    p = claude_pricing(model)
    if p is None:
        return 0.0
    return (
        max(0, input_tokens) * p.input
        + max(0, output_tokens) * p.output
        + max(0, cache_creation_tokens) * p.cache_write
        + max(0, cache_read_tokens) * p.cache_read
    )


def codex_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """Cost for a Codex turn given per-turn token deltas.

    ``input_tokens`` is the total input for the turn; ``cached_tokens`` (a subset)
    is billed at the cache-read rate, the remainder at the input rate.
    """
    p = codex_pricing(model)
    if p is None:
        return 0.0
    inp = max(0, input_tokens)
    cached = min(max(0, cached_tokens), inp)
    non_cached = inp - cached
    return non_cached * p.input + cached * p.cache_read + max(0, output_tokens) * p.output
