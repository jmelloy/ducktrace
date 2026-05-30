#!/usr/bin/env python3
"""Show token/cost totals for a single session ID.

Finds the session across all Claude project files, runs the same dedup and
summation logic as build_db, and prints per-model token/cost totals alongside
the raw API usage block values for comparison.

Usage: python session_stats.py <session-id>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from claude_analysis import claude_parser


def _fmt(n: int | float | None) -> str:
    if n is None:
        return "-"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _find_session_files(target: str) -> list[str]:
    """Find files that belong to target session without full parsing.

    Strategy (fast to slow):
    1. Direct name match: <base>/<target>.jsonl
    2. Sub-agent files:   <base>/<target>/subagents/*.jsonl
    3. Resume files:      any other .jsonl that contains the target string
       (cheap text scan, no JSON parsing, no tokenization)
    """
    bases = [Path(p) for p in claude_parser.config_paths()]
    found: list[str] = []

    for base in bases:
        # 1. main session file
        main = base.rglob(f"{target}.jsonl")
        for p in main:
            found.append(str(p))

        # 2. sub-agent files under <target>/subagents/
        for p in base.rglob(f"{target}/subagents/*.jsonl"):
            found.append(str(p))

    # 3. resume files: scan remaining files for the session ID string
    found_set = set(found)
    all_files = claude_parser.find_session_files([str(b) for b in bases])
    for f in all_files:
        if f in found_set:
            continue
        try:
            with open(f, "r", errors="replace") as fh:
                if target in fh.read():
                    found.append(f)
        except OSError:
            pass

    return found


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: session_stats.py <session-id>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1].strip()
    candidates = _find_session_files(target)

    events_by_id: dict[str, dict] = {}
    found_files: list[str] = []

    for f in candidates:
        try:
            result = claude_parser.parse_file(f)
        except Exception as exc:
            print(f"  parse error {f}: {exc}", file=sys.stderr)
            continue
        if result is None:
            continue
        meta, evs = result
        if meta["session_id"] != target:
            continue
        found_files.append(f)
        for ev in evs:
            events_by_id[ev["event_id"]] = ev

    if not found_files:
        print(f"Session {target!r} not found in {candidates}", file=sys.stderr)
        sys.exit(1)

    print(f"Session: {target}")
    for f in found_files:
        print(f"  {f}")
    print()

    events = list(events_by_id.values())

    # --- parsed event totals (what goes into the DB) ---
    parsed: dict[str, dict] = defaultdict(lambda: dict(input=0, output=0, cache_read=0, cache_write=0, cost=0.0, reasoning_tokens=0))
    for ev in events:
        m = ev.get("model") or "(no model)"
        for col, key in (
            ("output_tokens", "output"),
            ("cache_read_tokens", "cache_read"),
            ("cache_creation_tokens", "cache_write"),
            ("input_tokens", "input"),
            ("reasoning_tokens", "reasoning_tokens"),
        ):
            v = ev.get(col)
            
            if v is not None:
                parsed[m][key] += v
        c = ev.get("inferred_cost")
        if c is not None:
            parsed[m]["cost"] += c

    print("Parsed event totals:")
    for m, t in sorted(parsed.items()):
        print(
            f"  {m}:  {_fmt(t['input'])} input, {_fmt(t['output'])} output, "
            f"{_fmt(t['cache_read'])} cache read, {_fmt(t['cache_write'])} cache write "
            f"(${t['cost']:.2f})"
            f"{', ' + _fmt(t['reasoning_tokens']) + ' reasoning' if t['reasoning_tokens'] else ''}"
        )

    # --- raw API usage block totals (ground truth) ---
    # Only read from #0 events to avoid double-counting multi-block messages.
    # usage is stored in attributes["message"] (msg_meta = message dict minus content).
    api: dict[str, dict] = defaultdict(lambda: dict(input=0, output=0, cache_read=0, cache_write=0))
    seen_message_ids: set[str] = set()
    for ev in events:
        if (ev.get("block_index") or 0) != 0:
            continue
        mid = ev.get("message_id")
        if mid is not None:
            if mid in seen_message_ids:
                continue
            seen_message_ids.add(mid)
        attrs = ev.get("attributes") or {}
        msg_meta = attrs.get("message") or {}
        usage = msg_meta.get("usage") if isinstance(msg_meta, dict) else None
        if not isinstance(usage, dict):
            continue
        m = ev.get("model") or "(no model)"
        api[m]["input"] += usage.get("input_tokens", 0) or 0
        api[m]["output"] += usage.get("output_tokens", 0) or 0
        api[m]["cache_read"] += usage.get("cache_read_input_tokens", 0) or 0
        api[m]["cache_write"] += usage.get("cache_creation_input_tokens", 0) or 0

    if api:
        print()
        print("Raw API usage blocks:")
        for m, t in sorted(api.items()):
            print(
                f"  {m}:  {_fmt(t['input'])} input, {_fmt(t['output'])} output, "
                f"{_fmt(t['cache_read'])} cache read, {_fmt(t['cache_write'])} cache write"
            )

    print()
    print(f"Events: {len(events)}  Files: {len(found_files)}")


if __name__ == "__main__":
    main()
