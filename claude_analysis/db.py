"""DuckDB schema and writer for the sessions/events store.

Design goals (per request):
  * First-class ``session_id`` and ``event_id``.
  * Pull the *useful* scalars out of the raw JSON into typed columns (ids,
    types, timestamps, repo/file/token info) for cheap querying, while the rest
    of the original JSON is preserved in an ``attributes`` JSON column.
  * Bulk-append rows via Arrow (fast), upserting on the primary keys so re-runs
    are idempotent.
  * Reduce bulky free text on the way in (``text_filter``) so the store stays
    compact; pass ``keep_full_text=True`` to keep everything verbatim.

Two tables: ``sessions`` (one row per session) and ``events`` (one row per
content block / response item / standalone line).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa

from .pop import pop_used_event
from .text_filter import shrink_event, shrink_session

# Column name -> DuckDB type. Order matters for INSERT.
EVENT_COLUMNS: dict[str, str] = {
    "event_id": "VARCHAR",
    "session_id": "VARCHAR",
    "source": "VARCHAR",            # 'claude' | 'codex'
    "seq": "INTEGER",               # line number within the session file (1-based)
    "block_index": "INTEGER",       # content-block index within the line (0-based)
    "timestamp": "TIMESTAMP",
    "type": "VARCHAR",              # top-level entry type
    "subtype": "VARCHAR",           # content-block / payload subtype
    "role": "VARCHAR",              # user | assistant | tool_use | tool_result | system | reasoning | ...
    "parent_id": "VARCHAR",         # parentUuid (claude) / linking id
    "message_id": "VARCHAR",        # message.id (claude) / messageId
    "request_id": "VARCHAR",        # requestId (claude)
    "tool_use_id": "VARCHAR",       # tool_use id / tool_use_id / call_id  (links call<->result)
    "tool_name": "VARCHAR",
    "model": "VARCHAR",
    "cwd": "VARCHAR",
    "git_branch": "VARCHAR",
    "repository": "VARCHAR",        # resolved owner/repo or repo name (session-level)
    "file_path": "VARCHAR",
    "file_ext": "VARCHAR",
    "lines_added": "INTEGER",
    "lines_removed": "INTEGER",
    "pr_number": "INTEGER",             # PR this event references (pr-link / mined)
    "pr_url": "VARCHAR",
    "pr_action": "VARCHAR",             # 'pr-link' or 'gh pr <verb>'
    "referenced_repository": "VARCHAR", # owner/repo mined from --repo / PR URL
    "input_tokens": "BIGINT",
    "output_tokens": "BIGINT",
    "cache_read_tokens": "BIGINT",
    "cache_creation_tokens": "BIGINT",
    "reasoning_tokens": "BIGINT",
    "cost_usd": "DOUBLE",
    "text": "VARCHAR",              # extracted body/text (may be large)
    "attributes": "JSON",           # full raw entry (lossless)
}

SESSION_COLUMNS: dict[str, str] = {
    "session_id": "VARCHAR",
    "source": "VARCHAR",
    "title": "VARCHAR",             # manual rename (custom-title) if set, else ai-title
    "file_path": "VARCHAR",
    "started_at": "TIMESTAMP",
    "ended_at": "TIMESTAMP",
    "duration_sec": "DOUBLE",
    "cwd": "VARCHAR",
    "repository": "VARCHAR",
    "repository_url": "VARCHAR",
    "git_branch": "VARCHAR",
    "git_commit": "VARCHAR",
    "model": "VARCHAR",
    "cli_version": "VARCHAR",
    "originator": "VARCHAR",
    "model_provider": "VARCHAR",
    "event_count": "INTEGER",
    "message_count": "INTEGER",
    "tool_call_count": "INTEGER",
    "files_touched": "INTEGER",
    "input_tokens": "BIGINT",
    "output_tokens": "BIGINT",
    "cache_read_tokens": "BIGINT",
    "cache_creation_tokens": "BIGINT",
    "reasoning_tokens": "BIGINT",
    "total_tokens": "BIGINT",
    "cost_usd": "DOUBLE",
    "pr_repositories": "JSON",      # list of owner/repo seen via pr-link
    "pr_numbers": "JSON",           # list of PR numbers
    "attributes": "JSON",           # extra session metadata (worktree, cwd_counts, …)
}

_JSON_COLS = {"attributes", "pr_repositories", "pr_numbers"}


def _create_table(con, name: str, columns: dict[str, str], pk: str) -> None:
    cols_sql = ",\n  ".join(f'"{c}" {t}' for c, t in columns.items())
    con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (\n  {cols_sql},\n  PRIMARY KEY ("{pk}")\n)')


def _column_arrays(rows: list[dict], columns: dict[str, str]) -> dict[str, list]:
    """Pivot row dicts into column-wise lists, JSON-encoding JSON columns."""
    data: dict[str, list] = {c: [] for c in columns}
    for r in rows:
        for col in columns:
            val = r.get(col)
            if col in _JSON_COLS and val is not None and not isinstance(val, str):
                val = json.dumps(val, default=str)
            data[col].append(val)
    return data


def _dedup_last(rows: list[dict], key: str) -> list[dict]:
    """Keep the last row per primary key (a session resumed across files reuses
    ids; a single bulk INSERT can't contain duplicate keys)."""
    seen: dict[str, dict] = {}
    for r in rows:
        seen[r.get(key)] = r
    return list(seen.values())


class Store:
    """DuckDB wrapper that creates the schema and bulk-appends rows via Arrow.

    Rows from ``write_session`` are buffered and flushed in large batches
    (``batch_size`` events). Each flush builds an Arrow table and does a single
    ``INSERT OR REPLACE … SELECT`` — DuckDB ingests Arrow column-at-a-time,
    which is ~15-20x faster than row-by-row ``executemany``. Free text is
    reduced on the way in (see ``text_filter``). Call ``close`` (or use as a
    context manager) to flush the tail.
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 20_000,
        *,
        max_text: int = 4000,
        max_field: int = 4000,
        keep_full_text: bool = False,
        pop_used: bool = True,
    ):
        self.path = path
        self.batch_size = batch_size
        self.max_text = max_text
        self.max_field = max_field
        self.keep_full_text = keep_full_text
        self.pop_used = pop_used
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(path)
        _create_table(self.con, "sessions", SESSION_COLUMNS, pk="session_id")
        _create_table(self.con, "events", EVENT_COLUMNS, pk="event_id")
        self._session_buf: list[dict] = []
        self._event_buf: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self) -> None:
        self.con.execute("DELETE FROM events")
        self.con.execute("DELETE FROM sessions")

    def _bulk_insert(self, table: str, columns: dict[str, str], rows: list[dict], pk: str) -> None:
        if not rows:
            return
        rows = _dedup_last(rows, pk)
        arrow = pa.table(_column_arrays(rows, columns))
        view = f"_arrow_{table}"
        self.con.register(view, arrow)
        try:
            sel = ", ".join(f'"{c}"' for c in columns)
            self.con.execute(f'INSERT OR REPLACE INTO "{table}" ({sel}) SELECT {sel} FROM "{view}"')
        finally:
            self.con.unregister(view)

    def write_session(self, session: dict, events: list[dict]) -> None:
        self._session_buf.append(
            shrink_session(session, max_field=self.max_field, keep_full=self.keep_full_text)
        )
        for ev in events:
            if self.pop_used:
                pop_used_event(ev)
            self._event_buf.append(
                shrink_event(ev, max_text=self.max_text, max_field=self.max_field,
                             keep_full=self.keep_full_text)
            )
        if len(self._event_buf) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Bulk-append all buffered rows (events + sessions) in one transaction."""
        if not self._event_buf and not self._session_buf:
            return
        self.con.execute("BEGIN TRANSACTION")
        try:
            self._bulk_insert("events", EVENT_COLUMNS, self._event_buf, "event_id")
            self._bulk_insert("sessions", SESSION_COLUMNS, self._session_buf, "session_id")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        self._event_buf.clear()
        self._session_buf.clear()

    def canonicalize_repositories(self) -> list[tuple[str, str]]:
        """Promote bare repo names (e.g. ``dnsid``, from a worktree session with
        no pr-link or git remote) to their canonical ``owner/repo`` form when
        exactly one canonical match exists. Ambiguous names are left untouched.
        Returns the list of (bare -> canonical) mappings applied."""
        self.flush()
        rows = self.con.execute(
            """
            WITH bare AS (
                SELECT DISTINCT repository AS name FROM sessions
                WHERE repository IS NOT NULL AND repository <> '' AND repository NOT LIKE '%/%'
            ),
            canon AS (
                SELECT b.name, c.repository AS canonical
                FROM bare b
                JOIN (SELECT DISTINCT repository FROM sessions WHERE repository LIKE '%/%') c
                  ON c.repository LIKE '%/' || b.name
            )
            SELECT name, min(canonical) AS canonical
            FROM canon GROUP BY name HAVING count(DISTINCT canonical) = 1
            """
        ).fetchall()
        for bare, canonical in rows:
            self.con.execute("UPDATE sessions SET repository = ? WHERE repository = ?", [canonical, bare])
            self.con.execute("UPDATE events SET repository = ? WHERE repository = ?", [canonical, bare])
        return rows

    def close(self) -> None:
        self.flush()
        self.con.close()
