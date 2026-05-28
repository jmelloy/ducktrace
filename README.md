# ducktrace

Parses **Claude Code** and **Codex CLI** session logs into a single DuckDB
store with two tables — `sessions` and `events` — for usage, cost, repository,
and file-edit analysis.

Structure and cost logic are adapted from the `../ai-observer` Go importer
(`internal/importer/{claude,codex}.go`, `internal/pricing`, `cmd/debug_repos`),
reworked into a small Python package.

## Design

- **First-class ids.** Every event has a stable `event_id`
  (Claude's per-line `uuid`, suffixed `#<block>` when a line expands into
  multiple content blocks; `<session>:L<lineno>` for Codex and for Claude
  lines without a uuid). Every event and session carries `session_id`.
- **Extract the useful scalars; `attributes` holds only the remainder.** Ids,
  types, timestamps, model, repository, file path/ext, line counts and
  token/cost are pulled into typed columns. Fields promoted to a column (and the
  body text, which lives in the `text` column) are then *popped* out of the
  `attributes` JSON, so it contains only what hasn't been modelled yet
  (`stop_reason`, `isSidechain`, tool inputs, apply_patch bodies, Codex
  `rate_limits`, …) — making it easy to see what's left to promote. Pass
  `--keep-used-attributes` to keep the fuller JSON.
- **One session row per `session_id`, aggregated across files.** A session can
  span several files — Claude sub-agent transcripts under
  `<session>/subagents/` carry the parent `session_id`, and resumes append new
  files. Events are grouped by `session_id` and each session is aggregated once
  over the union of its (event-id-deduped) events, so sub-agent tokens, cost and
  tool calls roll into the parent rather than overwriting it.
- **Session title** — `title` is the manual rename (`custom-title`) if the
  session was renamed, otherwise the latest AI-generated `ai-title`.
- **Fast, idempotent writes.** Rows are buffered and bulk-appended to DuckDB via
  Arrow (one `INSERT OR REPLACE … SELECT` per batch — ~15–20× faster than
  row-by-row inserts). `INSERT OR REPLACE` upserts on the primary keys
  (`session_id` / `event_id`), so re-runs are idempotent.
- **Bulky free text is trimmed on the way in.** Thinking-block `signature`
  blobs are dropped, the `text` column is capped at `--max-text` chars (default
  4000), and long strings inside `attributes` at `--max-field` (default 4000),
  each with a `…[+N chars]` marker. Pass `--keep-full-text` to store everything
  verbatim (lossless).
- **Repository attribution** (strongest signal first):
  1. GitHub `owner/repo` from a Claude `pr-link` (`prRepository`) or a Codex
     `session_meta.git.repository_url` (credentials stripped, `.git` removed);
  2. a git worktree's original working dir (Claude `worktree-state`);
  3. the session's most-frequent `cwd`, with on-disk worktree resolution
     (a `.git` *file* pointing into `…/.git/worktrees/<name>` → parent repo
     directory name).

  After import, a canonicalization pass promotes a bare repo name (e.g. a
  worktree session that only knew `dnsid`) to its `owner/repo` form when
  exactly one canonical match exists elsewhere in the data (ambiguous names
  are left as-is). Disable with `--no-canonicalize`.
- **PR / URL mining.** Beyond structured Claude `pr-link` entries, we mine
  `gh pr <verb>` commands, `--repo owner/name` flags, and
  `github.com/<owner>/<repo>/pull/<N>` URLs out of shell commands and tool
  output/text (placeholders like `owner/repo` are ignored). These populate the
  `pr_number` / `pr_url` / `pr_action` / `referenced_repository` event columns,
  roll up into session `pr_repositories` / `pr_numbers`, and serve as a
  last-resort repo signal — which is the *only* PR signal for Codex, since it
  writes no pr-link records.
- **File metrics.** Edit/Write/MultiEdit/NotebookEdit (Claude) and `apply_patch`
  (Codex) populate `file_path`, `file_ext`, `lines_added`, `lines_removed`.
  Line counts live only on the tool **call** event (not its result) to avoid
  double counting; multi-file Codex patches expand to one event per file.
- **Cost.** Claude uses `costUSD` when present, else prices tokens. Codex diffs
  its cumulative `token_count` events into per-turn deltas and prices those.
  Token columns live on a single carrier event per turn, so session totals are
  simple `SUM`s with no double counting.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Build the store

```bash
.venv/bin/python build_db.py --db data/sessions.duckdb --reset
# options:
#   --source claude|codex|all   --limit N   --quiet   --no-canonicalize
#   --max-text N   --max-field N   --keep-full-text        (free-text controls)
#   --keep-used-attributes   (don't pop promoted fields out of attributes)
```

A full build of ~700 session files (~87k events) takes a few seconds.

Re-runs are idempotent (`INSERT OR REPLACE` on the primary keys). Source
locations are auto-discovered (`~/.claude/projects`, `~/.config/claude/projects`,
`~/.codex/sessions`) and overridable via `CLAUDE_ANALYSIS_CLAUDE_PATH` /
`CLAUDE_ANALYSIS_CODEX_PATH` (comma-separated).

## Tables

`sessions` (one row per session): `session_id`, `source`, `title`, `file_path`,
`started_at`/`ended_at`/`duration_sec`, `cwd`, `repository`, `repository_url`,
`git_branch`, `git_commit`, `model`, `cli_version`, `originator`,
`model_provider`, `event_count`, `message_count`, `tool_call_count`,
`files_touched`, token columns, `total_tokens`, `cost_usd`,
`pr_repositories`/`pr_numbers` (JSON), `attributes` (JSON).

`events` (one row per content block / response item / standalone line):
`event_id`, `session_id`, `source`, `seq`, `block_index`, `timestamp`, `type`,
`subtype`, `role`, `parent_id`, `message_id`, `request_id`, `tool_use_id`,
`tool_name`, `model`, `cwd`, `git_branch`, `repository`, `file_path`,
`file_ext`, `lines_added`, `lines_removed`, token columns, `cost_usd`, `text`,
`attributes` (JSON).

## Example queries

```sql
-- cost & tokens by repository
SELECT repository, count(*) AS sessions, round(sum(cost_usd),2) AS usd,
       sum(total_tokens) AS tokens
FROM sessions GROUP BY 1 ORDER BY usd DESC;

-- lines changed by file type and source
SELECT source, file_ext, sum(lines_added) AS added, sum(lines_removed) AS removed
FROM events WHERE role = 'tool_use' AND file_path IS NOT NULL
GROUP BY 1,2 ORDER BY added DESC;

-- most-used tools
SELECT source, tool_name, count(*) FROM events
WHERE role = 'tool_use' GROUP BY 1,2 ORDER BY 3 DESC;

-- PR activity (structured pr-link + mined gh pr / pull URLs)
SELECT source, pr_action, count(*) FROM events
WHERE pr_action IS NOT NULL GROUP BY 1,2 ORDER BY 3 DESC;

-- dig into anything still in the raw blob
SELECT attributes->>'$.line.gitBranch' AS branch, count(*)
FROM events WHERE source='claude' GROUP BY 1;
```
