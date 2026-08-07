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
    "calculated_input_tokens": "BIGINT",
    "output_tokens": "BIGINT",
    "cache_read_tokens": "BIGINT",
    "cache_creation_tokens": "BIGINT",
    "reasoning_tokens": "BIGINT",
    "stated_cost": "DOUBLE",        # costUSD if present, else computed from calculated_input_tokens
    "inferred_cost": "DOUBLE",      # computed from raw input_tokens (usage payload)
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
    "calculated_input_tokens": "BIGINT",
    "output_tokens": "BIGINT",
    "cache_read_tokens": "BIGINT",
    "cache_creation_tokens": "BIGINT",
    "reasoning_tokens": "BIGINT",
    "total_tokens": "BIGINT",
    "stated_cost": "DOUBLE",        # sum of stated_cost across events
    "inferred_cost": "DOUBLE",      # sum of inferred_cost across events
    "pr_repositories": "JSON",      # list of owner/repo seen via pr-link
    "pr_numbers": "JSON",           # list of PR numbers
    "attributes": "JSON",           # extra session metadata (worktree, cwd_counts, …)
}

_JSON_COLS = {"attributes", "pr_repositories", "pr_numbers"}


def _create_table(con, name: str, columns: dict[str, str], pk: str) -> None:
    cols_sql = ",\n  ".join(f'"{c}" {t}' for c, t in columns.items())
    con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (\n  {cols_sql},\n  PRIMARY KEY ("{pk}")\n)')


def _create_file_cache(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS file_cache (
            file_path  VARCHAR PRIMARY KEY,
            mtime_ns   BIGINT,
            size_bytes BIGINT,
            session_id VARCHAR
        )
    """)
    # migrate older caches that predate the session_id column
    con.execute("ALTER TABLE file_cache ADD COLUMN IF NOT EXISTS session_id VARCHAR")


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
        _create_file_cache(self.con)
        self._session_buf: list[dict] = []
        self._event_buf: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self) -> None:
        self.con.execute("DELETE FROM events")
        self.con.execute("DELETE FROM sessions")
        self.con.execute("DELETE FROM file_cache")

    def get_seen_files(self) -> dict[str, tuple[int, int]]:
        """Return {file_path: (mtime_ns, size_bytes)} for all cached files."""
        rows = self.con.execute("SELECT file_path, mtime_ns, size_bytes FROM file_cache").fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def get_file_sessions(self) -> dict[str, str]:
        """Return {file_path: session_id} so an incremental build can find the
        unchanged sibling files of a session that had *some* file change."""
        rows = self.con.execute(
            "SELECT file_path, session_id FROM file_cache WHERE session_id IS NOT NULL"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def mark_files_seen(self, entries: list[tuple[str, int, int, str]]) -> None:
        """Upsert (file_path, mtime_ns, size_bytes, session_id) rows."""
        if not entries:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO file_cache (file_path, mtime_ns, size_bytes, session_id) VALUES (?, ?, ?, ?)",
            entries,
        )

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
        """Promote bare repo names to a canonical ``owner/repo`` form.

        A bare name (e.g. ``dnsid``, or a subdir leaf like ``backend``) is
        upgraded when an ``owner/repo`` seen elsewhere in the data unambiguously
        matches — either the bare name itself, or a segment of the session's cwd
        path (so a session working in ``…/pioneer-square/backend`` resolves to
        ``jmelloy/pioneer-square``). Repo-parts mapping to more than one owner are
        left untouched. Returns the (bare -> canonical) mappings applied."""
        from pathlib import PurePosixPath

        self.flush()
        applied: list[tuple[str, str]] = []

        # 0) Unify case variants of the same owner/repo (GitHub is
        # case-insensitive; a lowercase URL the agent typed can otherwise split a
        # repo in two). Canonical casing = the variant used by the most sessions.
        variants: dict[str, list] = {}
        for full, n in self.con.execute(
            "SELECT repository, count(*) FROM sessions WHERE repository LIKE '%/%' GROUP BY 1"
        ).fetchall():
            variants.setdefault(full.lower(), []).append((full, n))
        for group in variants.values():
            if len(group) < 2:
                continue
            canonical = max(group, key=lambda v: (v[1], any(c.isupper() for c in v[0])))[0]
            for full, _ in group:
                if full != canonical:
                    self.con.execute("UPDATE sessions SET repository=? WHERE repository=?", [canonical, full])
                    self.con.execute("UPDATE events SET repository=? WHERE repository=?", [canonical, full])
                    applied.append((full, canonical))

        # repo-part (segment after the slash) -> owner/repo, only when unique.
        part_to_full: dict[str, set] = {}
        for (full,) in self.con.execute(
            "SELECT DISTINCT repository FROM sessions WHERE repository LIKE '%/%'"
        ).fetchall():
            part_to_full.setdefault(full.split("/")[-1].lower(), set()).add(full)
        canon = {p: next(iter(s)) for p, s in part_to_full.items() if len(s) == 1}
        bare = self.con.execute(
            """SELECT session_id, repository, cwd FROM sessions
               WHERE repository IS NOT NULL AND repository <> '' AND repository NOT LIKE '%/%'"""
        ).fetchall()
        for sid, repo, cwd in bare:
            target = canon.get(repo.lower())
            if target is None and cwd:
                # deepest path segment first, so a subdir's own repo wins
                for seg in reversed([s.lower() for s in PurePosixPath(cwd).parts]):
                    if seg in canon:
                        target = canon[seg]
                        break
            if target and target != repo:
                self.con.execute("UPDATE sessions SET repository=? WHERE session_id=?", [target, sid])
                self.con.execute("UPDATE events SET repository=? WHERE session_id=?", [target, sid])
                applied.append((repo, target))
        return applied

    def build_session_repositories(self) -> int:
        """(Re)build the ``session_repositories`` junction: one row per
        (session, repository), so a session that works in its own repo and
        references PRs in others fans out to 1-n rows.

        A file change is attributed to ``referenced_repository`` when the event
        carries one (a PR/`--repo` reference), else to the session's own repo —
        which is where the edits actually happened. Cost is apportioned across a
        session's repos by share of lines touched (added+removed); the shares sum
        to 1 so ``sum(stated_cost)`` per session is conserved. A session with no
        lines touched puts all its cost on its primary repo rather than splitting
        evenly (an even split would be fiction). ``pr_numbers`` collects the PRs
        seen against each repo in that session.

        ponytail: cwd is *not* used to split — most within-session cwd variety is
        subdirs/worktrees of the same repo (dnsid/compliance-tests ⊂ dnsid), so
        splitting on it fragments one repo. The session's canonical repository
        already resolves that. Pure post-step SQL; no parser changes.
        """
        self.flush()
        self.con.execute("DROP TABLE IF EXISTS session_repositories")
        self.con.execute("""
        CREATE TABLE session_repositories AS
        WITH ev AS (
          SELECT e.session_id,
                 coalesce(nullif(e.referenced_repository, ''), s.repository) AS repository,
                 e.file_path, e.role, e.lines_added, e.lines_removed, e.pr_number
          FROM events e JOIN sessions s USING(session_id)
        ),
        agg AS (
          SELECT session_id, repository,
                 count(DISTINCT CASE WHEN role='tool_use' THEN file_path END) AS files_touched,
                 coalesce(sum(lines_added), 0)   AS lines_added,
                 coalesce(sum(lines_removed), 0) AS lines_removed,
                 list(DISTINCT pr_number) FILTER (pr_number IS NOT NULL) AS pr_numbers
          FROM ev WHERE repository IS NOT NULL AND repository <> '' GROUP BY 1, 2
        ),
        combined AS (  -- guarantee the session's own repo always gets a row
          SELECT * FROM agg
          UNION ALL
          SELECT session_id, repository, 0, 0, 0, []::BIGINT[]
          FROM sessions WHERE repository IS NOT NULL AND repository <> ''
        ),
        rolled AS (
          SELECT session_id, repository,
                 sum(files_touched) AS files_touched,
                 sum(lines_added)   AS lines_added,
                 sum(lines_removed) AS lines_removed,
                 list_distinct(flatten(list(pr_numbers))) AS pr_numbers
          FROM combined GROUP BY 1, 2
        ),
        tot AS (SELECT session_id, sum(lines_added + lines_removed) AS total_lines FROM rolled GROUP BY 1)
        SELECT r.session_id, r.repository,
               (r.repository = s.repository) AS is_primary,
               r.files_touched, r.lines_added, r.lines_removed, r.pr_numbers,
               CASE WHEN t.total_lines > 0 THEN (r.lines_added + r.lines_removed) * 1.0 / t.total_lines
                    WHEN r.repository = s.repository THEN 1.0 ELSE 0.0 END AS lines_share,
               s.stated_cost   * (CASE WHEN t.total_lines > 0 THEN (r.lines_added + r.lines_removed) * 1.0 / t.total_lines
                                       WHEN r.repository = s.repository THEN 1.0 ELSE 0.0 END) AS stated_cost,
               s.inferred_cost * (CASE WHEN t.total_lines > 0 THEN (r.lines_added + r.lines_removed) * 1.0 / t.total_lines
                                       WHEN r.repository = s.repository THEN 1.0 ELSE 0.0 END) AS inferred_cost
        FROM rolled r JOIN sessions s USING(session_id) JOIN tot t USING(session_id)
        """)
        self.con.execute(
            "ALTER TABLE session_repositories ADD PRIMARY KEY (session_id, repository)"
        )
        return self.con.execute("SELECT count(*) FROM session_repositories").fetchone()[0]

    def close(self) -> None:
        self.flush()
        self.con.close()
