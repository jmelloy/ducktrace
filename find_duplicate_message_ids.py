#!/usr/bin/env python3
"""Scan all Claude Code session files and report sessions with duplicate message_ids.

A duplicate message_id means two or more events within the same session share
the same message.id value, which can indicate replayed or re-ingested API calls.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from claude_analysis import claude_parser


def main() -> None:
    paths = claude_parser.config_paths()
    if not paths:
        print("No Claude session directories found.", file=sys.stderr)
        sys.exit(1)

    files = claude_parser.find_session_files(paths)
    print(f"Scanning {len(files)} session file(s) in {paths}…", file=sys.stderr)

    # session_id -> {message_id -> [event dicts]}
    sessions: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    errors = 0
    for f in files:
        try:
            result = claude_parser.parse_file(f)
        except Exception as exc:
            print(f"  error parsing {f}: {exc}", file=sys.stderr)
            errors += 1
            continue
        if result is None:
            continue
        meta, events = result
        sid = meta["session_id"]
        for ev in events:
            mid = ev.get("message_id")
            if mid is None:
                continue
            sessions[sid][mid].append(ev)

    found = 0
    for sid, mid_map in sorted(sessions.items()):
        dupes = {mid: evs for mid, evs in mid_map.items() if len(evs) > 1}
        if not dupes:
            continue
        found += 1
        print(f"\nsession: {sid}")
        for mid, evs in sorted(dupes.items()):
            print(f"  message_id={mid}  ({len(evs)} events)")
            for ev in evs:
                inp = ev.get("input_tokens")
                out = ev.get("output_tokens")
                print(f"    event_id={ev['event_id']}  input_tokens={inp}  output_tokens={out}")

    print(f"\n{'='*60}", file=sys.stderr)
    if found:
        print(f"Found {found} session(s) with duplicate message_ids.", file=sys.stderr)
    else:
        print("No duplicate message_ids found.", file=sys.stderr)
    if errors:
        print(f"{errors} file(s) failed to parse.", file=sys.stderr)


if __name__ == "__main__":
    main()
