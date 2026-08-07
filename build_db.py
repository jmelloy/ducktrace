#!/usr/bin/env python3
"""Build the sessions/events DuckDB store from Claude Code and Codex CLI logs.

Usage:
  python build_db.py [--db data/sessions.duckdb]
                     [--source claude|codex|pi|all]
                     [--reset] [--limit N] [--quiet]

Discovers session files in the standard locations (overridable via
CLAUDE_ANALYSIS_CLAUDE_PATH / CLAUDE_ANALYSIS_CODEX_PATH /
CLAUDE_ANALYSIS_PI_PATH, comma-separated) and
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

from claude_analysis import claude_parser, codex_parser, pi_parser, pricing
from claude_analysis.aggregate import aggregate_session
from claude_analysis.db import Store

# session metadata keys passed through to aggregate_session
_META_KEYS = (
    "repository", "repository_url", "git_branch", "git_commit", "model",
    "cli_version", "originator", "model_provider", "cwd",
    "pr_repositories", "pr_numbers", "extra_attributes",
)


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


def _stat(f) -> tuple[int, int]:
    try:
        st = os.stat(f)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return -1, -1


def _ingest(
    parser_mod, events_by, meta_by, *,
    limit, quiet, seen_files: dict, file_sessions: dict, force: bool,
) -> tuple[int, int, list[tuple[str, int, int, str]]]:
    """Parse a source's files into the per-session event/meta maps.

    A session spans several files (main + ``subagents/`` + resumes). Aggregation
    reads *all* of a session's events at once, so if only some of its files
    changed we must still re-parse the unchanged ones — otherwise the session
    row is rewritten from a partial view, dropping tokens / pr_numbers / repo
    from the untouched files. ``file_sessions`` (cached file->session_id) lets us
    pull those siblings back in for any session that had a change.

    Returns (n_parsed, n_skipped, new_file_stats) where new_file_stats is a list
    of (path, mtime_ns, size_bytes, session_id) for every file parsed this run.
    """
    paths = parser_mod.config_paths()
    files = parser_mod.find_session_files(paths)
    if limit:
        files = files[:limit]
    label = parser_mod.SOURCE
    if not quiet:
        print(f"[{label}] {len(files)} session file(s) in {paths or '(none found)'}", file=sys.stderr)

    # classify: files whose mtime/size changed must parse now; unchanged files
    # with a known cached session are pull-in candidates. Unchanged files with
    # no cached session (first run after upgrade) parse so we learn the mapping.
    stats: dict[str, tuple[int, int]] = {}
    to_parse: list[str] = []            # definitely parse (changed / unknown)
    candidates: dict[str, str] = {}     # path -> cached session_id (pull in if touched)
    for f in files:
        path_str = str(f)
        stats[path_str] = _stat(f)
        cached_sid = file_sessions.get(path_str)
        if force or seen_files.get(path_str) != stats[path_str] or cached_sid is None:
            to_parse.append(path_str)
        else:
            candidates[path_str] = cached_sid

    by_path = {str(f): f for f in files}

    def _parse(path_str):
        try:
            return parser_mod.parse_file(by_path[path_str])
        except Exception as exc:  # keep going; report the offender
            print(f"[{label}] error parsing {path_str}: {exc}", file=sys.stderr)
            return None

    # pass 1: parse the changed/unknown files to discover which sessions changed
    parsed: dict[str, tuple] = {}
    touched: set[str] = set()
    for path_str in to_parse:
        result = _parse(path_str)
        if result is None:
            continue
        parsed[path_str] = result
        touched.add(result[0]["session_id"])

    # pass 2: pull in the unchanged sibling files of any touched session
    for path_str, sid in candidates.items():
        if sid not in touched:
            continue
        result = _parse(path_str)
        if result is not None:
            parsed[path_str] = result

    # merge in sorted path order (main file sorts before its subagents/ dir, so
    # _merge_meta keeps the main file's identity fields)
    new_stats: list[tuple[str, int, int, str]] = []
    for path_str in sorted(parsed):
        meta, evs = parsed[path_str]
        sid = meta["session_id"]
        bucket = events_by[sid]
        for ev in evs:
            bucket[ev["event_id"]] = ev  # dedup by event_id (overlapping resumes)
        meta_by[sid] = _merge_meta(meta_by.get(sid), meta)
        mtime_ns, size = stats[path_str]
        new_stats.append((path_str, mtime_ns, size, sid))

    n_skipped = len(files) - len(parsed)
    return len(parsed), n_skipped, new_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/sessions.duckdb", help="output DuckDB path")
    ap.add_argument("--source", choices=("claude", "codex", "pi", "all"), default="all")
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
    file_sessions = store.get_file_sessions()
    events_by: dict[str, dict] = defaultdict(dict)  # session_id -> {event_id: event}
    meta_by: dict[str, dict] = {}                    # session_id -> merged meta
    all_new_stats: list[tuple[str, int, int, str]] = []

    ingest_kwargs = dict(limit=args.limit, quiet=args.quiet,
                         seen_files=seen_files, file_sessions=file_sessions,
                         force=args.force)
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
    if args.source in ("pi", "all"):
        n, skipped, ns = _ingest(pi_parser, events_by, meta_by, **ingest_kwargs)
        all_new_stats.extend(ns)
        if not args.quiet and skipped:
            print(f"[pi] skipped {skipped} unchanged file(s)", file=sys.stderr)

    # aggregate each session once over the union of its (deduped) events
    total_e = 0
    for sid, bucket in events_by.items():
        evs = list(bucket.values())
        m = meta_by[sid]
        repository = m.get("repository") or ""
        for ev in evs:  # re-stamp so events agree with the merged session repo
            ev["session_id"] = sid
            ev["source"] = m["source"]
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

    n_repo_rows = store.build_session_repositories()
    if not args.quiet:
        print(f"Built session_repositories: {n_repo_rows} (session, repo) rows", file=sys.stderr)

    store.close()
    if not args.quiet:
        print(
            f"\nDone in {time.time()-start:.1f}s → {args.db}\n"
            f"  {len(events_by)} sessions, {total_e} events",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
