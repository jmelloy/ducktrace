"""Small shared helpers: timestamp parsing, file extensions, line counting,
and apply-patch parsing for file-level metrics."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
)


def parse_ts(ts: str | None) -> datetime | None:
    """Parse an RFC3339-ish timestamp to a tz-aware UTC datetime, or None."""
    if not ts:
        return None
    # datetime.fromisoformat handles most forms (incl. 'Z' on 3.11+).
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def file_ext(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).suffix.lower() or "(no ext)"


def count_lines(text: str | None) -> int:
    return len(text.splitlines()) if text else 0


# A patch file header line, e.g. "*** Add File: /path/to/x.go"
_PATCH_FILE_RE = re.compile(r"^\*\*\*\s+(Add|Update|Delete|Move)\s+File:\s+(.+?)\s*$", re.I)


def parse_apply_patch(patch_text: str) -> list[dict]:
    """Parse a Codex ``apply_patch`` payload (the ``*** Begin Patch`` format)
    into per-file edit records: ``{file, op, added, removed}``.

    Lines starting with a single ``+``/``-`` count as added/removed; the
    ``+++``/``---`` diff headers and the ``*** …`` directives are ignored."""
    if not patch_text:
        return []
    files: dict[str, dict] = {}
    current: str | None = None
    for line in patch_text.splitlines():
        m = _PATCH_FILE_RE.match(line)
        if m:
            op, path = m.group(1).lower(), m.group(2).strip()
            current = path
            files.setdefault(path, {"file": path, "op": op, "added": 0, "removed": 0})
            files[path]["op"] = op
            continue
        if current is None:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            files[current]["added"] += 1
        elif line.startswith("-"):
            files[current]["removed"] += 1
    return list(files.values())
