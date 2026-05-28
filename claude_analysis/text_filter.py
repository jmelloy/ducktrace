"""Free-text reduction for stored rows.

Keeps the *structure* and the extracted scalar columns intact, but trims the
bulky free text that dominates DB size and slows writes:

  * drops ``signature`` blobs (thinking-block base64 — pure noise for analysis);
  * truncates the ``text`` column to ``max_text`` chars;
  * truncates any long string *inside* ``attributes`` to ``max_field`` chars.

Truncated strings get a ``…[+N chars]`` marker so it's obvious data was
elided and how much. Set ``keep_full=True`` (CLI ``--keep-full-text``) to store
everything verbatim instead.
"""

from __future__ import annotations

# Keys whose (string) values are dropped entirely rather than truncated —
# opaque base64 blobs with no analytical value (thinking signatures, Codex
# encrypted reasoning).
_DROP_KEYS = {"signature", "encrypted_content"}


def _truncate(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"…[+{len(s) - limit} chars]"


def _clean(value, max_field: int):
    if isinstance(value, str):
        return _truncate(value, max_field)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _DROP_KEYS and isinstance(v, str):
                out[k] = f"[dropped {len(v)} chars]"
            else:
                out[k] = _clean(v, max_field)
        return out
    if isinstance(value, list):
        return [_clean(v, max_field) for v in value]
    return value


def shrink_event(ev: dict, *, max_text: int, max_field: int, keep_full: bool) -> dict:
    """Mutate and return an event row with its free text reduced."""
    if keep_full:
        return ev
    if isinstance(ev.get("text"), str):
        ev["text"] = _truncate(ev["text"], max_text)
    if ev.get("attributes") is not None:
        ev["attributes"] = _clean(ev["attributes"], max_field)
    return ev


def shrink_session(sess: dict, *, max_field: int, keep_full: bool) -> dict:
    if keep_full:
        return sess
    if sess.get("attributes") is not None:
        sess["attributes"] = _clean(sess["attributes"], max_field)
    return sess
