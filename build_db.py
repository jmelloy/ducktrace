#!/usr/bin/env python3
"""Build the sessions/events DuckDB store from Claude Code and Codex CLI logs.

Usage:
  python build_db.py [--db data/sessions.duckdb]
                     [--source claude|codex|all]
                     [--reset] [--limit N] [--quiet]

Discovers session files in the standard locations (overridable via
CLAUDE_ANALYSIS_CLAUDE_PATH / CLAUDE_ANALYSIS_CODEX_PATH, comma-separated) and
parses each into event rows plus per-file session metadata. A session can span
several files (Claude sub-agent transcripts under ``<session>/subagents/``, and
resumes), so events are grouped by ``session_id`` and each session is aggregated
*once* over the union of its events — sub-agent tokens/tool-calls roll into the
parent. Writes are idempotent on the primary keys (session_id / event_id).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from claude_analysis import claude_parser, codex_parser
from claude_analysis.aggregate import aggregate_session
from claude_analysis.db import Store

# session metadata keys passed through to aggregate_session
_META_KEYS = (
    "repository", "repository_url", "git_branch", "git_commit", "model",
    "cli_version", "originator", "model_provider", "cwd",
    "pr_repositories", "pr_numbers", "extra_attributes",
)

_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_creation_tokens", "reasoning_tokens", "stated_cost", "inferred_cost",
)


def _dedup_usage(evs: list[dict]) -> None:
    """When multiple events share a message_id (same API response replayed into
    several JSONL files), only the last one should carry usage data. Zero out
    token/cost fields on all earlier occurrences so aggregate_session doesn't
    multiply-count them. 'Last' is by seq (line number) then event_id for
    stability across files."""
    by_mid: dict[str, list[int]] = defaultdict(list)
    for i, ev in enumerate(evs):
        mid = ev.get("message_id")
        if mid is not None:
            by_mid[mid].append(i)
    for indices in by_mid.values():
        if len(indices) <= 1:
            continue
        indices.sort(key=lambda i: (evs[i].get("seq") or 0, evs[i].get("event_id") or ""))
        for i in indices[:-1]:
            for field in _USAGE_FIELDS:
                evs[i][field] = None


def _better_repo(a: str, b: str) -> str:
    """Prefer a canonical owner/repo over a bare name, else any non-empty."""
    cands = [x for x in (a, b) if x]
    if not cands:
        return a or b or ""
    slashed = [x for x in cands if "/" in x]
    return slashed[0] if slashed else cands[0]


def _merge_meta(acc: dict | None, m: dict) -> dict:
    """Combine per-file metadata for one session across its files. The main
    session file sorts before its ``subagents/`` dir, so ``acc`` (seen first)
    keeps identity fields like ``file_path``."""
    if acc is None:
        return dict(m)
    for k in ("repository_url", "git_branch", "git_commit", "model",
              "cli_version", "originator", "model_provider", "cwd"):
        if not acc.get(k) and m.get(k):
            acc[k] = m[k]
    acc["repository"] = _better_repo(acc.get("repository", ""), m.get("repository", ""))
    if m.get("custom_title"):
        acc["custom_title"] = m["custom_title"]
    if m.get("ai_title"):
        acc["ai_title"] = m["ai_title"]
    acc["pr_repositories"] = sorted(set(acc.get("pr_repositories", [])) | set(m.get("pr_repositories", [])))
    acc["pr_numbers"] = sorted(set(acc.get("pr_numbers", [])) | set(m.get("pr_numbers", [])))
    ea = dict(acc.get("extra_attributes") or {})
    ea.update(m.get("extra_attributes") or {})
    acc["extra_attributes"] = ea
    return acc


def _ingest(
    parser_mod, events_by, meta_by, *,
    limit, quiet, seen_files: dict, force: bool,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    """Parse a source's files into the per-session event/meta maps.

    Returns (n_parsed, n_skipped, new_file_stats) where new_file_stats is a
    list of (path, mtime_ns, size_bytes) for every file that was actually parsed
    this run (so the caller can update the cache).
    """
    paths = parser_mod.config_paths()
    files = parser_mod.find_session_files(paths)
    if limit:
        files = files[:limit]
    label = parser_mod.SOURCE
    if not quiet:
        print(f"[{label}] {len(files)} session file(s) in {paths or '(none found)'}", file=sys.stderr)

    n_parsed = 0
    n_skipped = 0
    new_stats: list[tuple[str, int, int]] = []

    for i, f in enumerate(files, 1):
        path_str = str(f)
        if not force:
            try:
                st = os.stat(f)
                mtime_ns, size = st.st_mtime_ns, st.st_size
            except OSError:
                mtime_ns, size = -1, -1
            cached = seen_files.get(path_str)
            if cached is not None and cached == (mtime_ns, size):
                n_skipped += 1
                continue
        else:
            try:
                st = os.stat(f)
                mtime_ns, size = st.st_mtime_ns, st.st_size
            except OSError:
                mtime_ns, size = -1, -1

        try:
            result = parser_mod.parse_file(f)
        except Exception as exc:  # keep going; report the offender
            print(f"[{label}] error parsing {f}: {exc}", file=sys.stderr)
            continue
        if result is None:
            continue
        meta, evs = result
        sid = meta["session_id"]
        bucket = events_by[sid]
        for ev in evs:
            bucket[ev["event_id"]] = ev  # dedup by event_id (overlapping resumes)
        meta_by[sid] = _merge_meta(meta_by.get(sid), meta)
        new_stats.append((path_str, mtime_ns, size))
        n_parsed += 1
        if not quiet and i % 200 == 0:
            print(f"[{label}] {i}/{len(files)} files…", file=sys.stderr)
    return n_parsed, n_skipped, new_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/sessions.duckdb", help="output DuckDB path")
    ap.add_argument("--source", choices=("claude", "codex", "all"), default="all")
    ap.add_argument("--reset", action="store_true", help="clear existing rows first")
    ap.add_argument("--limit", type=int, default=0, help="max files per source (0 = all)")
    ap.add_argument("--no-canonicalize", action="store_true",
                    help="skip promoting bare repo names to owner/repo")
    ap.add_argument("--max-text", type=int, default=4000,
                    help="truncate the events.text column to N chars (0 = unlimited)")
    ap.add_argument("--max-field", type=int, default=4000,
                    help="truncate long strings inside attributes JSON to N chars (0 = unlimited)")
    ap.add_argument("--keep-full-text", action="store_true",
                    help="store all text verbatim (lossless; no truncation or signature stripping)")
    ap.add_argument("--keep-used-attributes", action="store_true",
                    help="keep fields in attributes even when promoted to a column (don't pop)")
    ap.add_argument("--force", action="store_true",
                    help="re-parse all files even if mtime/size are unchanged")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser()
    if args.reset and db_path.exists():
        db_path.unlink()

    store = Store(
        str(db_path),
        max_text=args.max_text,
        max_field=args.max_field,
        keep_full_text=args.keep_full_text,
        pop_used=not args.keep_used_attributes,
    )

    start = time.time()
    seen_files = store.get_seen_files()
    events_by: dict[str, dict] = defaultdict(dict)  # session_id -> {event_id: event}
    meta_by: dict[str, dict] = {}                    # session_id -> merged meta
    all_new_stats: list[tuple[str, int, int]] = []

    ingest_kwargs = dict(limit=args.limit, quiet=args.quiet,
                         seen_files=seen_files, force=args.force)
    if args.source in ("claude", "all"):
        n, skipped, ns = _ingest(claude_parser, events_by, meta_by, **ingest_kwargs)
        all_new_stats.extend(ns)
        if not args.quiet and skipped:
            print(f"[claude] skipped {skipped} unchanged file(s)", file=sys.stderr)
    if args.source in ("codex", "all"):
        n, skipped, ns = _ingest(codex_parser, events_by, meta_by, **ingest_kwargs)
        all_new_stats.extend(ns)
        if not args.quiet and skipped:
            print(f"[codex] skipped {skipped} unchanged file(s)", file=sys.stderr)

    # aggregate each session once over the union of its (deduped) events
    total_e = 0
    for sid, bucket in events_by.items():
        evs = list(bucket.values())
        _dedup_usage(evs)
        m = meta_by[sid]
        repository = m.get("repository") or ""
        for ev in evs:  # re-stamp so events agree with the merged session repo
            ev["repository"] = repository
        title = m.get("custom_title") or m.get("ai_title") or None
        session = aggregate_session(
            sid, m["file_path"], m["source"], evs,
            title=title,
            **{k: m.get(k) for k in _META_KEYS},
        )
        store.write_session(session, evs)
        total_e += len(evs)

    store.mark_files_seen(all_new_stats)

    if not args.no_canonicalize:
        mapped = store.canonicalize_repositories()
        if mapped and not args.quiet:
            pairs = sorted(set(mapped))
            print(f"\nCanonicalized {len(mapped)} session(s), {len(pairs)} name(s):", file=sys.stderr)
            for bare, canonical in pairs:
                print(f"  {bare} -> {canonical}", file=sys.stderr)

    store.close()
    if not args.quiet:
        print(
            f"\nDone in {time.time()-start:.1f}s → {args.db}\n"
            f"  {len(events_by)} sessions, {total_e} events",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
