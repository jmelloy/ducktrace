"""Incremental rebuild must re-parse *all* files of a session when only some
changed — otherwise the session row is rewritten from a partial view and loses
tokens / pr_numbers / repo carried by the unchanged files."""

from __future__ import annotations

import os
from collections import defaultdict
from types import SimpleNamespace

import build_db
from claude_analysis.db import Store


def _fake_parser(tmp_path, contents: dict[str, dict]):
    """A parser module over on-disk files. ``contents`` maps filename -> meta;
    each file belongs to session 's1'. The main file carries pr_numbers."""
    for name in contents:
        (tmp_path / name).write_text("x")  # real bytes so mtime/size are stat-able

    def parse_file(path):
        meta = dict(contents[os.path.basename(str(path))])
        meta.update(session_id="s1", source="fake", file_path=str(path))
        ev = {"event_id": f"{os.path.basename(str(path))}:1", "session_id": "s1"}
        return meta, [ev]

    return SimpleNamespace(
        SOURCE="fake",
        config_paths=lambda: [str(tmp_path)],
        find_session_files=lambda paths: sorted(str(p) for p in tmp_path.glob("*.jsonl")),
        parse_file=parse_file,
    )


def test_unchanged_sibling_pulled_in_when_one_file_changes(tmp_path):
    parser = _fake_parser(tmp_path, {
        "s1.jsonl": {"pr_numbers": [15], "pr_repositories": ["o/r"]},   # main
        "s1_subagents.jsonl": {"pr_numbers": [], "pr_repositories": []},  # sibling
    })
    store = Store(str(tmp_path / "db.duckdb"))

    def run():
        events_by, meta_by = defaultdict(dict), {}
        _, _, stats = build_db._ingest(
            parser, events_by, meta_by,
            limit=0, quiet=True,
            seen_files=store.get_seen_files(),
            file_sessions=store.get_file_sessions(),
            force=False,
        )
        store.mark_files_seen(stats)
        return events_by, meta_by, [s[0] for s in stats]

    # round 1: cold — both files parsed, session has the PR number
    _, meta_by, parsed = run()
    assert meta_by["s1"]["pr_numbers"] == [15]
    assert len(parsed) == 2

    # touch only the sibling; the main file (with pr_numbers) is unchanged
    os.utime(tmp_path / "s1_subagents.jsonl", ns=(10**18, 10**18))

    # round 2: sibling changed -> main must be pulled back in so pr_numbers survive
    events_by, meta_by, parsed = run()
    assert meta_by["s1"]["pr_numbers"] == [15], "unchanged main file was dropped"
    assert len(events_by["s1"]) == 2
    assert set(os.path.basename(p) for p in parsed) == {"s1.jsonl", "s1_subagents.jsonl"}
    store.close()


def test_fully_unchanged_session_is_skipped(tmp_path):
    parser = _fake_parser(tmp_path, {"s1.jsonl": {"pr_numbers": [15]}})
    store = Store(str(tmp_path / "db.duckdb"))
    events_by, meta_by = defaultdict(dict), {}
    _, _, stats = build_db._ingest(
        parser, events_by, meta_by, limit=0, quiet=True,
        seen_files=store.get_seen_files(), file_sessions=store.get_file_sessions(), force=False)
    store.mark_files_seen(stats)

    # nothing touched -> nothing re-parsed
    events_by, meta_by = defaultdict(dict), {}
    n, skipped, stats = build_db._ingest(
        parser, events_by, meta_by, limit=0, quiet=True,
        seen_files=store.get_seen_files(), file_sessions=store.get_file_sessions(), force=False)
    assert (n, skipped, stats) == (0, 1, [])
    store.close()
