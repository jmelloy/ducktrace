"""session_repositories fans a session out to 1-n repos and apportions cost by
lines touched. The invariant that matters: per-session apportioned cost is
conserved (shares sum to 1), and lines land on the right repo."""

from __future__ import annotations

from claude_analysis.db import EVENT_COLUMNS, SESSION_COLUMNS, Store


def _row(cols, **vals):
    return {c: vals.get(c) for c in cols}


def _seed(store):
    # one session in repo A that also references a PR in repo B.
    sessions = [_row(SESSION_COLUMNS, session_id="s1", source="fake",
                     repository="o/a", stated_cost=10.0, inferred_cost=6.0)]
    events = [
        # 30 lines edited in the session's own repo (no referenced_repository)
        _row(EVENT_COLUMNS, event_id="e1", session_id="s1", role="tool_use",
             file_path="a.py", lines_added=20, lines_removed=10),
        # 10 lines touched against a PR in another repo
        _row(EVENT_COLUMNS, event_id="e2", session_id="s1", role="tool_use",
             file_path="b.py", lines_added=10, lines_removed=0,
             referenced_repository="o/b", pr_number=42),
    ]
    store._bulk_insert("sessions", SESSION_COLUMNS, sessions, "session_id")
    store._bulk_insert("events", EVENT_COLUMNS, events, "event_id")


def test_fanout_apportion_and_conservation(tmp_path):
    store = Store(str(tmp_path / "db.duckdb"))
    _seed(store)
    store.build_session_repositories()

    rows = store.con.execute("""
        SELECT repository, is_primary, files_touched, lines_added, lines_removed,
               pr_numbers, round(lines_share, 4), round(stated_cost, 4), round(inferred_cost, 4)
        FROM session_repositories ORDER BY repository
    """).fetchall()
    by_repo = {r[0]: r for r in rows}

    assert set(by_repo) == {"o/a", "o/b"}
    # o/a: primary, 30 of 40 lines -> 75% of cost; edits attributed here, no PR
    a = by_repo["o/a"]
    assert a[1] is True and a[3] == 20 and a[4] == 10 and a[5] == []
    assert a[6] == 0.75 and a[7] == 7.5 and a[8] == 4.5
    # o/b: referenced repo, 10 of 40 lines -> 25%, carries PR 42
    b = by_repo["o/b"]
    assert b[1] is False and b[3] == 10 and b[5] == [42]
    assert b[6] == 0.25 and b[7] == 2.5 and b[8] == 1.5

    # cost conservation: apportioned shares sum back to the session total
    tot = store.con.execute(
        "SELECT round(sum(stated_cost), 4), round(sum(inferred_cost), 4) FROM session_repositories"
    ).fetchone()
    assert tot == (10.0, 6.0)
    store.close()


def test_zero_lines_puts_all_cost_on_primary(tmp_path):
    store = Store(str(tmp_path / "db.duckdb"))
    store._bulk_insert("sessions", SESSION_COLUMNS,
                       [_row(SESSION_COLUMNS, session_id="s2", source="fake",
                             repository="o/a", stated_cost=4.0)], "session_id")
    # a PR reference in another repo but zero lines touched anywhere
    store._bulk_insert("events", EVENT_COLUMNS,
                       [_row(EVENT_COLUMNS, event_id="e1", session_id="s2", role="assistant",
                             referenced_repository="o/b", pr_number=7)], "event_id")
    store.build_session_repositories()
    got = dict(store.con.execute(
        "SELECT repository, round(stated_cost, 4) FROM session_repositories").fetchall())
    assert got == {"o/a": 4.0, "o/b": 0.0}  # no even split (would be fiction)
    store.close()
