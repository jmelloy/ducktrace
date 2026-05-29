"""Roll a session's event rows up into a single session row.

Token/cost totals sum over events. build_db._dedup_usage strips token fields
from all but the last event per message_id before this runs, so sessions that
span multiple files (resumes, sub-agents) don't multiply-count API calls.
"""

from __future__ import annotations


def aggregate_session(session_id: str, path: str, source: str, events: list[dict], **meta) -> dict:
    def _sum(col):
        vals = [v for ev in events if (v := ev.get(col)) is not None]
        return sum(vals) if vals else None

    timestamps = [ev["timestamp"] for ev in events if ev.get("timestamp")]
    started = min(timestamps) if timestamps else None
    ended = max(timestamps) if timestamps else None
    duration = (ended - started).total_seconds() if started and ended else None

    # A "message" is one user/assistant turn. Count the first event of each such
    # line (block_index 0) so multi-block assistant lines — incl. tool-only turns
    # where no block has the assistant role — are counted once. We count rather
    # than collect seq, because seq is per-file and collides when a session spans
    # multiple files (sub-agents, resumes).
    message_count = sum(
        1 for ev in events
        if (ev.get("block_index") or 0) == 0
        and (ev.get("type") in ("user", "assistant") or ev.get("role") in ("user", "assistant"))
    )
    tool_calls = sum(1 for ev in events if ev.get("role") == "tool_use")
    files = {ev["file_path"] for ev in events if ev.get("file_path") and ev.get("role") == "tool_use"}

    inp, out = _sum("input_tokens"), _sum("output_tokens")
    cr, cc = _sum("cache_read_tokens"), _sum("cache_creation_tokens")
    reasoning = _sum("reasoning_tokens")

    # Roll PR/repo references (structured pr-link + mined from commands/text)
    # up to the session. ``meta`` carries the structured pr-link values; events
    # carry both those and anything mined.
    pr_repos = set(meta.get("pr_repositories") or [])
    pr_nums = set(meta.get("pr_numbers") or [])
    pr_actions: set[str] = set()
    for ev in events:
        if ev.get("referenced_repository"):
            pr_repos.add(ev["referenced_repository"])
        if ev.get("pr_number"):
            pr_nums.add(ev["pr_number"])
        if ev.get("pr_action"):
            pr_actions.add(ev["pr_action"])

    return {
        "session_id": session_id,
        "source": source,
        "title": meta.get("title"),
        "file_path": path,
        "started_at": started,
        "ended_at": ended,
        "duration_sec": duration,
        "cwd": meta.get("cwd"),
        "repository": meta.get("repository"),
        "repository_url": meta.get("repository_url"),
        "git_branch": meta.get("git_branch"),
        "git_commit": meta.get("git_commit"),
        "model": meta.get("model"),
        "cli_version": meta.get("cli_version"),
        "originator": meta.get("originator"),
        "model_provider": meta.get("model_provider"),
        "event_count": len(events),
        "message_count": message_count,
        "tool_call_count": tool_calls,
        "files_touched": len(files),
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cr,
        "cache_creation_tokens": cc,
        "reasoning_tokens": reasoning,
        "total_tokens": sum(v for v in [inp, out, cr, cc, reasoning] if v is not None),
        "stated_cost": _sum("stated_cost"),
        "inferred_cost": _sum("inferred_cost"),
        "pr_repositories": sorted(pr_repos),
        "pr_numbers": sorted(pr_nums),
        "attributes": {**meta.get("extra_attributes", {}),
                       **({"pr_actions": sorted(pr_actions)} if pr_actions else {})},
    }
