"""Pop the fields we already promoted to typed columns out of ``attributes``.

After this runs, ``attributes`` holds only the *remainder* — the bits we have
not yet lifted into a column — which makes it much easier to see what's still
unmodelled and decide what to promote next. Anything reconstructable from a
column (ids, types, timestamps, cwd/branch, model, tool name/role) is removed,
as is the body text that now lives in the ``text`` column. For ``usage``, only
the four token fields lifted into typed columns are removed; extra fields
(service_tier, billed_units, …) are preserved. Genuinely un-promoted data
(tool inputs, apply_patch bodies, ``stop_reason``, ``isSidechain``,
``turn_id``, …) is kept.

Pass ``keep_used=True`` (CLI ``--keep-used-attributes``) to skip popping and
keep the fuller JSON.
"""

from __future__ import annotations

# Claude — keys removed at each level of the {line, message, block} attributes.
_CLAUDE_LINE = {"type", "timestamp", "sessionId", "uuid", "parentUuid",
                "requestId", "cwd", "gitBranch"}
_CLAUDE_MSG = {"id", "model"}
# Only the usage subfields that were lifted into typed columns; the rest stays.
_CLAUDE_USAGE = {"input_tokens", "output_tokens",
                 "cache_creation_input_tokens", "cache_read_input_tokens"}
_CLAUDE_BLOCK_STRUCT = {"type", "id", "tool_use_id", "name"}
_CLAUDE_BLOCK_TEXT = {"text", "thinking", "content"}  # now in the `text` column

# Codex — keys removed from the {timestamp, type, payload} attributes.
_CODEX_TOP = {"timestamp", "type"}
_CODEX_PAYLOAD_STRUCT = {"type", "role", "call_id", "name"}
_CODEX_PAYLOAD_TEXT = {"content", "summary"}  # message/reasoning body -> `text`


def _pop_keys(d, keys) -> None:
    if isinstance(d, dict):
        for k in keys:
            d.pop(k, None)


def _pop_value_equals(d, text) -> None:
    """Drop any key whose (string) value is exactly what we copied into `text`
    — e.g. function_call ``arguments``, tool ``output``, ``prUrl``."""
    if not isinstance(d, dict) or not isinstance(text, str) or not text:
        return
    for k in [k for k, v in d.items() if isinstance(v, str) and v == text]:
        d.pop(k, None)


def pop_used_event(ev: dict) -> dict:
    a = ev.get("attributes")
    if not isinstance(a, dict):
        return ev
    text = ev.get("text")

    if ev.get("source") == "claude":
        if a.keys() & {"line", "message", "block"}:
            _pop_keys(a.get("line"), _CLAUDE_LINE)
            _pop_keys(a.get("message"), _CLAUDE_MSG)
            usage = (a.get("message") or {}).get("usage")
            if isinstance(usage, dict):
                _pop_keys(usage, _CLAUDE_USAGE)
                if not usage:
                    a.get("message", {}).pop("usage", None)
            block = a.get("block")
            if isinstance(block, str):
                a.pop("block", None)  # plain user text -> `text` column
            elif isinstance(block, dict):
                _pop_keys(block, _CLAUDE_BLOCK_STRUCT)
                _pop_keys(block, _CLAUDE_BLOCK_TEXT)
            # drop now-empty containers for tidiness
            for k in ("line", "message", "block"):
                if a.get(k) == {} :
                    a.pop(k, None)
        else:
            # raw non-message entry (pr-link, system, snapshot, …)
            _pop_keys(a, _CLAUDE_LINE)
            _pop_value_equals(a, text)  # prUrl / aiTitle / lastPrompt
    else:  # codex
        _pop_keys(a, _CODEX_TOP)
        payload = a.get("payload")
        _pop_keys(payload, _CODEX_PAYLOAD_STRUCT)
        _pop_keys(payload, _CODEX_PAYLOAD_TEXT)
        _pop_value_equals(payload, text)  # arguments / input / output == text
        if payload == {}:
            a.pop("payload", None)

    return ev
