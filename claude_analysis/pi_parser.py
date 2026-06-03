"""Parse pi (Earendil ``pi`` coding-agent) session JSONL files into session +
event rows.

pi writes ``~/.pi/agent/sessions/<encoded-cwd>/<ISO-ts>_<uuid>.jsonl``. The
encoded-cwd directory name is the working directory with ``/`` replaced by
``-`` and wrapped in ``--…--``. Each line is a JSON object with a top-level
``type``:

  * ``session``               – ``{version, id, timestamp, cwd}`` (one per file)
  * ``model_change``          – ``{provider, modelId}`` (active model switches)
  * ``thinking_level_change`` – reasoning-effort switches
  * ``message``               – the transcript, with a nested ``message`` whose
                                ``role`` is one of:
        - ``user``          : ``content`` = [{type:text}]
        - ``assistant``     : ``content`` = [{type:thinking|text|toolCall}],
                              plus ``api``/``provider``/``model``/``usage``/
                              ``stopReason`` (token counts + cost live here)
        - ``toolResult``    : ``toolCallId``/``toolName``/``content``/``isError``
        - ``bashExecution`` : a user-run ``! command`` with ``output``/``exitCode``

Unlike Claude/Codex, pi records both token usage *and* dollar cost directly on
each assistant turn (``usage.cost.total``), so ``stated_cost`` comes straight
from the log; ``inferred_cost`` is recomputed from the pricing tables as a
cross-check (and falls back to the stated cost when the model isn't priced,
e.g. a local Ollama model). Repository comes from the session ``cwd`` reconciled
with any owner/repo references mined from commands/output.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from . import pricing, prmatch
from .repos import resolve_session_repository
from .util import count_lines, file_ext, parse_ts

SOURCE = "pi"

_EDIT_TOOLS = {"edit"}
_WRITE_TOOLS = {"write"}


def config_paths() -> list[str]:
    env = os.getenv("CLAUDE_ANALYSIS_PI_PATH") or os.getenv("AI_OBSERVER_PI_PATH")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    pi_home = os.getenv("PI_HOME")
    base = Path(pi_home) if pi_home else Path.home() / ".pi"
    p = base / "agent" / "sessions"
    return [str(p)] if p.exists() else []


def find_session_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for base in paths:
        files.extend(str(p) for p in Path(base).rglob("*.jsonl"))
    return sorted(files)


def _text_blocks(content) -> str:
    """Flatten text blocks from a content list into one string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for c in content:
        if isinstance(c, dict) and c.get("type") in ("text", None) and c.get("text"):
            parts.append(c["text"])
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def _edit_metrics(name: str, args: dict) -> tuple[str, int | None, int | None]:
    """(file_path, lines_added, lines_removed) for pi's edit/write tools."""
    fpath = args.get("path") or args.get("file_path") or ""
    if name in _EDIT_TOOLS:
        edits = args.get("edits")
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except json.JSONDecodeError:
                edits = []
        added = removed = 0
        for e in edits or []:
            if isinstance(e, dict):
                added += count_lines(e.get("newText", ""))
                removed += count_lines(e.get("oldText", ""))
        return fpath, added, removed
    if name in _WRITE_TOOLS:
        return fpath, count_lines(args.get("content", "")), 0
    return fpath, None, None


def parse_file(path: str) -> tuple[dict, list[dict]] | None:
    lines: list[dict] = []
    try:
        with open(path, "r", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    if not lines:
        return None

    session_id = Path(path).stem.split("_", 1)[-1]  # strip leading ISO timestamp
    cwd = ""
    version = None
    current_model = ""
    current_provider = ""
    model_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    events: list[dict] = []
    seq = 0

    def base_event(eid_suffix, ts, etype, *, block=0):
        nonlocal seq
        seq += 1
        eid = f"{SOURCE}:{session_id}:{eid_suffix}"
        if block:
            eid = f"{eid}#{block}"
        return {
            "event_id": eid,
            "session_id": session_id,
            "source": SOURCE,
            "seq": seq,
            "block_index": block,
            "timestamp": ts,
            "type": etype,
            "subtype": None,
            "role": None,
            "parent_id": None,
            "message_id": None,
            "request_id": None,
            "tool_use_id": None,
            "tool_name": None,
            "model": current_model or None,
            "cwd": cwd or None,
            "git_branch": None,
            "repository": None,  # filled at the end once resolved
            "file_path": None,
            "file_ext": None,
            "lines_added": None,
            "lines_removed": None,
            "pr_number": None,
            "pr_url": None,
            "pr_action": None,
            "referenced_repository": None,
            "input_tokens": None,
            "calculated_input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "reasoning_tokens": None,
            "stated_cost": None,
            "inferred_cost": None,
            "text": None,
            "attributes": None,
        }

    for idx, entry in enumerate(lines):
        etype = entry.get("type")
        ts = parse_ts(entry.get("timestamp"))
        eid = entry.get("id") or f"L{idx}"

        if etype == "session":
            cwd = entry.get("cwd", cwd) or cwd
            version = entry.get("version", version)
            if entry.get("id"):
                session_id = entry["id"]
            ev = base_event(eid, ts, "session")
            ev["session_id"] = session_id
            ev["subtype"] = "session_meta"
            ev["role"] = "system"
            ev["cwd"] = cwd or None
            ev["attributes"] = entry
            events.append(ev)
            continue

        if etype == "model_change":
            current_provider = entry.get("provider", current_provider) or current_provider
            if entry.get("modelId"):
                current_model = entry["modelId"]
            ev = base_event(eid, ts, "model_change")
            ev["subtype"] = "model_change"
            ev["role"] = "system"
            ev["parent_id"] = entry.get("parentId")
            ev["attributes"] = entry
            events.append(ev)
            continue

        if etype == "thinking_level_change":
            ev = base_event(eid, ts, "thinking_level_change")
            ev["subtype"] = "thinking_level_change"
            ev["role"] = "system"
            ev["parent_id"] = entry.get("parentId")
            ev["attributes"] = entry
            events.append(ev)
            continue

        if etype != "message":
            ev = base_event(eid, ts, etype or "unknown")
            ev["attributes"] = entry
            events.append(ev)
            continue

        msg = entry.get("message") or {}
        role = msg.get("role")
        parent = entry.get("parentId")

        if role == "user":
            ev = base_event(eid, ts, "user")
            ev["subtype"] = "message"
            ev["role"] = "user"
            ev["parent_id"] = parent
            ev["message_id"] = eid
            ev["text"] = _text_blocks(msg.get("content"))
            ev["attributes"] = entry
            events.append(ev)
            continue

        if role == "assistant":
            mdl = msg.get("model") or current_model
            prov = msg.get("provider") or current_provider
            if mdl:
                model_counts[mdl] = model_counts.get(mdl, 0) + 1
            if prov:
                provider_counts[prov] = provider_counts.get(prov, 0) + 1
            content = msg.get("content") if isinstance(msg.get("content"), list) else []
            if not content:
                content = [{"type": "text", "text": ""}]
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                ev = base_event(eid, ts, "assistant", block=bi)
                ev["parent_id"] = parent
                ev["message_id"] = eid
                ev["model"] = mdl or None
                btype = block.get("type")
                if btype == "thinking":
                    ev["subtype"] = "thinking"
                    ev["role"] = "reasoning"
                    ev["text"] = block.get("thinking", "")
                elif btype == "toolCall":
                    name = block.get("name") or "tool"
                    ev["subtype"] = "tool_call"
                    ev["role"] = "tool_use"
                    ev["tool_name"] = name
                    ev["tool_use_id"] = block.get("id")
                    fpath, added, removed = _edit_metrics(name, block.get("arguments") or {})
                    if fpath:
                        ev["file_path"] = fpath
                        ev["file_ext"] = file_ext(fpath)
                        ev["lines_added"] = added
                        ev["lines_removed"] = removed
                    args = block.get("arguments")
                    ev["text"] = json.dumps(args, default=str) if args is not None else f"Tool call: {name}"
                else:  # text or anything else
                    ev["subtype"] = "text"
                    ev["role"] = "assistant"
                    ev["text"] = block.get("text", "")
                # token usage + cost ride on the first block of the turn
                if bi == 0:
                    _attach_pi_usage(ev, msg, mdl)
                ev["attributes"] = entry if bi == 0 else {"block": block, "message_id": eid}
                events.append(ev)
            continue

        if role == "toolResult":
            ev = base_event(eid, ts, "tool_result")
            ev["subtype"] = "tool_result"
            ev["role"] = "tool_result"
            ev["parent_id"] = parent
            ev["tool_use_id"] = msg.get("toolCallId")
            ev["tool_name"] = msg.get("toolName")
            ev["text"] = _text_blocks(msg.get("content")) or msg.get("errorMessage", "")
            ev["attributes"] = entry
            events.append(ev)
            continue

        if role == "bashExecution":
            ev = base_event(eid, ts, "bash_execution")
            ev["subtype"] = "bash_execution"
            ev["role"] = "tool_use"
            ev["parent_id"] = parent
            ev["tool_name"] = "bash"
            cmd = msg.get("command", "")
            out = msg.get("output", "")
            ev["text"] = f"$ {cmd}\n{out}" if out else f"$ {cmd}"
            ev["attributes"] = entry
            events.append(ev)
            continue

        # unknown message role: keep it, lossless
        ev = base_event(eid, ts, "message")
        ev["subtype"] = role
        ev["parent_id"] = parent
        ev["attributes"] = entry
        events.append(ev)

    # mine PR/repo references from each event's text (commands, tool output, …)
    for ev in events:
        found = prmatch.extract(ev.get("text") or "")
        if found["pr_actions"]:
            ev["pr_action"] = found["pr_actions"][0]
        if found["pr_urls"]:
            ev["pr_url"] = found["pr_urls"][0]
        if found["pr_numbers"]:
            ev["pr_number"] = found["pr_numbers"][0]
        if found["repos"]:
            ev["referenced_repository"] = found["repos"][0]

    ref_counts = Counter(ev["referenced_repository"] for ev in events if ev.get("referenced_repository"))
    candidates = [r for r, _ in ref_counts.most_common()]
    repository = resolve_session_repository(
        candidate_repositories=candidates,
        cwd=cwd,
    )
    for ev in events:
        ev["repository"] = repository

    main_model = max(model_counts, key=model_counts.get) if model_counts else current_model
    main_provider = max(provider_counts, key=provider_counts.get) if provider_counts else current_provider

    meta = {
        "session_id": session_id,
        "source": SOURCE,
        "file_path": path,
        "repository": repository,
        "repository_url": "",
        "git_branch": "",
        "git_commit": "",
        "model": main_model,
        "cli_version": str(version) if version is not None else "",
        "originator": "pi",
        "model_provider": main_provider,
        "cwd": cwd,
        "custom_title": "",
        "ai_title": "",
        "pr_repositories": [],
        "pr_numbers": [],
        "extra_attributes": {
            "model_counts": model_counts,
            "provider_counts": provider_counts,
        },
    }
    return meta, events


def _attach_pi_usage(ev: dict, msg: dict, model: str) -> None:
    """pi records per-turn token usage and dollar cost on the assistant message.
    ``stated_cost`` is taken verbatim; ``inferred_cost`` is recomputed from the
    pricing tables (falling back to the stated cost when the model isn't priced,
    e.g. a local Ollama model)."""
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return
    inp = int(usage.get("input") or 0)
    out = int(usage.get("output") or 0)
    cr = int(usage.get("cacheRead") or 0)
    cc = int(usage.get("cacheWrite") or 0)
    ev["input_tokens"] = inp
    ev["output_tokens"] = out
    ev["cache_read_tokens"] = cr
    ev["cache_creation_tokens"] = cc

    cost = usage.get("cost")
    stated = None
    if isinstance(cost, dict) and cost.get("total") is not None:
        stated = float(cost["total"])
    ev["stated_cost"] = stated

    inferred = pricing.pi_cost(model, inp, out, cc, cr)
    ev["inferred_cost"] = inferred or stated
