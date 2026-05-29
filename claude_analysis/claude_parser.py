"""Parse Claude Code session JSONL files into session + event rows.

Claude Code writes one JSON object per line under ``~/.claude/projects/<slug>/
<session-uuid>.jsonl`` (and the XDG ``~/.config/claude/projects`` location).
Each line has a stable ``uuid`` (our event id) and ``parentUuid`` (lineage).
Assistant lines carry ``message.usage`` (tokens) and sometimes ``costUSD``.
Repository attribution prefers ``pr-link`` entries, then worktree origin,
then the most-frequent ``cwd``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from . import pricing, prmatch
from .repos import resolve_session_repository
from .util import count_lines, file_ext, parse_ts

SOURCE = "claude"

# Tools whose inputs describe a file edit, and where to find the file path.
_EDIT_TOOLS = {"Edit", "str_replace", "str_replace_based_edit_tool"}
_WRITE_TOOLS = {"Write", "write_file", "create", "create_file"}
_MULTI_EDIT_TOOLS = {"MultiEdit"}
_NOTEBOOK_TOOLS = {"NotebookEdit"}


def config_paths() -> list[str]:
    """Locations to search, newest layout first (matches ai-observer)."""
    env = os.getenv("AI_OBSERVER_CLAUDE_PATH") or os.getenv("CLAUDE_ANALYSIS_CLAUDE_PATH")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    home = Path.home()
    out = []
    for rel in (Path(".config") / "claude" / "projects", Path(".claude") / "projects"):
        p = home / rel
        if p.exists():
            out.append(str(p))
    return out


def find_session_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for base in paths:
        files.extend(str(p) for p in Path(base).rglob("*.jsonl"))
    return sorted(files)


def _file_path_from_input(inp: dict) -> str:
    for k in ("file_path", "path", "notebook_path", "filePath"):
        v = inp.get(k)
        if v:
            return v
    return ""


def _edit_metrics(name: str, inp: dict) -> tuple[str, int | None, int | None]:
    """Return (file_path, lines_added, lines_removed) for an edit-like tool."""
    fpath = _file_path_from_input(inp)
    if name in _EDIT_TOOLS:
        old = inp.get("old_string", inp.get("old_str", ""))
        new = inp.get("new_string", inp.get("new_str", ""))
        return fpath, count_lines(new), count_lines(old)
    if name in _WRITE_TOOLS:
        body = inp.get("content", inp.get("file_text", ""))
        return fpath, count_lines(body), 0
    if name in _MULTI_EDIT_TOOLS:
        added = removed = 0
        for e in inp.get("edits", []) or []:
            if isinstance(e, dict):
                added += count_lines(e.get("new_string", ""))
                removed += count_lines(e.get("old_string", ""))
        return fpath, added, removed
    if name in _NOTEBOOK_TOOLS:
        return fpath, count_lines(inp.get("new_source", "")), 0
    return fpath, None, None


def _content_text(content) -> str:
    """Flatten a tool_result content (str or list of blocks) into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" and b.get("text"):
                    parts.append(b["text"])
                elif b.get("text"):
                    parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return ""


def parse_file(path: str) -> tuple[dict, list[dict]] | None:
    """Parse one Claude JSONL file -> (session_meta, [event_rows]). Returns None
    if the file has no usable entries. The session row itself is aggregated in
    build_db once per session_id, since a session can span several files
    (sub-agent transcripts, resumes)."""
    session_id = Path(path).stem
    lines: list[tuple[int, dict]] = []
    try:
        with open(path, "r", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append((lineno, json.loads(raw)))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    if not lines:
        return None

    # --- pass 1: session-level signals ---------------------------------------
    pr_repositories: list[str] = []
    pr_numbers: list[int] = []
    original_cwd = ""
    cwd_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    cli_version = ""
    entrypoint = ""
    git_branch = ""
    ai_title = ""
    custom_title = ""

    for _, e in lines:
        t = e.get("type")
        sid = e.get("sessionId")
        if sid:
            session_id = sid
        if t == "ai-title" and e.get("aiTitle"):
            ai_title = e["aiTitle"]            # last (most recent) wins
        elif t == "custom-title" and e.get("customTitle"):
            custom_title = e["customTitle"]    # manual rename; overrides ai-title
        if t == "worktree-state":
            ws = e.get("worktreeSession") or {}
            if ws.get("originalCwd"):
                original_cwd = ws["originalCwd"]
        elif t == "pr-link":
            repo = e.get("prRepository", "")
            if repo:
                pr_repositories.append(repo)
            if e.get("prNumber"):
                pr_numbers.append(e["prNumber"])
        if e.get("cwd"):
            cwd_counts[e["cwd"]] = cwd_counts.get(e["cwd"], 0) + 1
        if e.get("version"):
            cli_version = e["version"]
        if e.get("entrypoint"):
            entrypoint = e["entrypoint"]
        if e.get("gitBranch"):
            git_branch = e["gitBranch"]
        msg = e.get("message")
        if isinstance(msg, dict) and msg.get("model"):
            model_counts[msg["model"]] = model_counts.get(msg["model"], 0) + 1

    main_cwd = max(cwd_counts, key=cwd_counts.get) if cwd_counts else ""
    main_model = max(model_counts, key=model_counts.get) if model_counts else ""
    # repository is resolved after pass 2, once command/URL-mined repo references
    # (referenced_repository) are available to combine with the cwd.

    # --- pass 2: emit events --------------------------------------------------
    events: list[dict] = []

    def base_event(lineno, e, *, uuid_suffix=None):
        uuid = e.get("uuid") or e.get("messageId") or e.get("leafUuid")
        eid = uuid or f"{SOURCE}:{session_id}:L{lineno}"
        if uuid_suffix is not None:
            eid = f"{eid}#{uuid_suffix}"
        ts = parse_ts(e.get("timestamp"))
        return {
            "event_id": eid,
            "session_id": session_id,
            "source": SOURCE,
            "seq": lineno,
            "block_index": uuid_suffix or 0,
            "timestamp": ts,
            "type": e.get("type"),
            "subtype": None,
            "role": None,
            "parent_id": e.get("parentUuid"),
            "message_id": (e.get("message") or {}).get("id") if isinstance(e.get("message"), dict) else e.get("messageId"),
            "request_id": e.get("requestId"),
            "tool_use_id": None,
            "tool_name": None,
            "model": (e.get("message") or {}).get("model") if isinstance(e.get("message"), dict) else None,
            "cwd": e.get("cwd"),
            "git_branch": e.get("gitBranch"),
            "repository": None,  # resolved + stamped after mining (see below)
            "file_path": None,
            "file_ext": None,
            "lines_added": None,
            "lines_removed": None,
            "pr_number": None,
            "pr_url": None,
            "pr_action": None,
            "referenced_repository": None,
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "reasoning_tokens": None,
            "stated_cost": None,
            "inferred_cost": None,
            "text": None,
            "attributes": None,
        }

    def mine(ev: dict, *extra_text: str) -> None:
        """Populate PR/repo columns from mined references in command + text."""
        found = prmatch.extract(ev.get("text") or "", *extra_text)
        if found["pr_actions"]:
            ev["pr_action"] = found["pr_actions"][0]
        if found["pr_urls"]:
            ev["pr_url"] = found["pr_urls"][0]
        if found["pr_numbers"]:
            ev["pr_number"] = found["pr_numbers"][0]
        if found["repos"]:
            ev["referenced_repository"] = found["repos"][0]

    for lineno, e in lines:
        t = e.get("type")
        msg = e.get("message") if isinstance(e.get("message"), dict) else None
        content = msg.get("content") if msg else None
        line_meta = {k: v for k, v in e.items() if k != "message"}
        msg_meta = {k: v for k, v in (msg or {}).items() if k != "content"}

        # Non-message structural entries: one event, full raw kept.
        if t not in ("user", "assistant") or content is None:
            ev = base_event(lineno, e)
            ev["role"] = t
            ev["attributes"] = e
            if t == "pr-link":
                ev["text"] = e.get("prUrl")
                # structured pr-link -> same columns as mined references
                ev["pr_action"] = "pr-link"
                ev["pr_url"] = e.get("prUrl")
                ev["pr_number"] = e.get("prNumber") or None
                ev["referenced_repository"] = e.get("prRepository") or None
            elif t in ("ai-title",):
                ev["text"] = e.get("aiTitle")
            elif t == "custom-title":
                ev["text"] = e.get("customTitle")
            elif t == "agent-name":
                ev["text"] = e.get("agentName")
            elif t in ("last-prompt",):
                ev["text"] = e.get("lastPrompt")
            events.append(ev)
            continue

        # message line with string content (typical user prompt)
        if isinstance(content, str):
            ev = base_event(lineno, e)
            ev["role"] = t
            ev["subtype"] = "text"
            ev["text"] = content
            ev["attributes"] = {"line": line_meta, "message": msg_meta, "block": content}
            _attach_usage(ev, e, msg, main_model)
            mine(ev)
            events.append(ev)
            continue

        # message line with a list of content blocks
        if not isinstance(content, list) or not content:
            ev = base_event(lineno, e)
            ev["role"] = t
            ev["attributes"] = {"line": line_meta, "message": msg_meta}
            _attach_usage(ev, e, msg, main_model)
            events.append(ev)
            continue

        tool_use_result = e.get("toolUseResult")
        for i, block in enumerate(content):
            ev = base_event(lineno, e, uuid_suffix=i)
            btype = block.get("type") if isinstance(block, dict) else "text"
            ev["subtype"] = btype
            ev["attributes"] = {"line": line_meta, "message": msg_meta, "block": block}

            if btype == "thinking":
                ev["role"] = t
                ev["text"] = block.get("thinking", "")
            elif btype == "text":
                ev["role"] = t
                ev["text"] = block.get("text", "")
                mine(ev)
            elif btype == "tool_use":
                ev["role"] = "tool_use"
                ev["tool_name"] = block.get("name")
                ev["tool_use_id"] = block.get("id")
                inp = block.get("input") or {}
                cmd = ""
                if isinstance(inp, dict):
                    fpath, added, removed = _edit_metrics(block.get("name", ""), inp)
                    if fpath:
                        ev["file_path"] = fpath
                        ev["file_ext"] = file_ext(fpath)
                        ev["lines_added"] = added
                        ev["lines_removed"] = removed
                    cmd = inp.get("command", "") if block.get("name") in ("Bash", "bash") else ""
                ev["text"] = f"Tool call: {block.get('name')}"
                mine(ev, cmd)  # the Bash command, not the placeholder text
            elif btype == "tool_result":
                ev["role"] = "tool_result"
                ev["tool_use_id"] = block.get("tool_use_id")
                ev["text"] = _content_text(block.get("content"))
                # Attribute the result to a file when the structured result names one,
                # but leave line counts on the tool_use event to avoid double counting.
                if isinstance(tool_use_result, dict):
                    fpath = tool_use_result.get("filePath") or (
                        tool_use_result.get("file", {}) or {}).get("filePath")
                    if fpath:
                        ev["file_path"] = fpath
                        ev["file_ext"] = file_ext(fpath)
                mine(ev)  # gh pr create prints the new PR URL into its output
            else:
                ev["role"] = btype
                ev["text"] = block.get("text") if isinstance(block, dict) else None

            if i == 0:
                _attach_usage(ev, e, msg, main_model)
            events.append(ev)

    # Resolve the repository now that we've mined owner/repo references from the
    # session's own commands/output (referenced_repository), and combine them
    # with the structured pr-link repos, most-frequent first. The cwd decides
    # which of these is real (see resolve_session_repository).
    ref_counts = Counter(ev["referenced_repository"] for ev in events if ev.get("referenced_repository"))
    for r in pr_repositories:
        ref_counts[r] += 1
    candidates = [r for r, _ in ref_counts.most_common()]
    repository = resolve_session_repository(
        candidate_repositories=candidates,
        original_cwd=original_cwd,
        cwd=main_cwd,
    )
    for ev in events:
        ev["repository"] = repository

    # --- per-file session metadata (aggregated once per session in build) -----
    meta = {
        "session_id": session_id,
        "source": SOURCE,
        "file_path": path,
        "repository": repository,
        "repository_url": "",
        "git_branch": git_branch,
        "git_commit": "",
        "model": main_model,
        "cli_version": cli_version,
        "originator": entrypoint,
        "model_provider": "anthropic",
        "cwd": main_cwd,
        "custom_title": custom_title,
        "ai_title": ai_title,
        "pr_repositories": sorted(set(pr_repositories)),
        "pr_numbers": sorted(set(pr_numbers)),
        "extra_attributes": {
            "original_cwd": original_cwd,
            "cwd_counts": cwd_counts,
            "model_counts": model_counts,
        },
    }
    return meta, events


def _attach_usage(ev: dict, e: dict, msg: dict | None, fallback_model: str) -> None:
    """Attach token + cost columns from an assistant message's usage block."""
    if e.get("type") != "assistant" or not msg:
        return
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return
    model = msg.get("model") or fallback_model
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cc = usage.get("cache_creation_input_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    ev["input_tokens"] = inp
    ev["output_tokens"] = out
    ev["cache_creation_tokens"] = cc
    ev["cache_read_tokens"] = cr
    ev["stated_cost"] = e.get("costUSD") or None
    ev["inferred_cost"] = pricing.claude_cost(model, inp, out, cc, cr)
