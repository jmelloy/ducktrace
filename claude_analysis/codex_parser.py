"""Parse Codex CLI rollout JSONL files into session + event rows.

Codex writes ``~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl``. Each
line is ``{timestamp, type, payload}`` where ``type`` is one of:
  * ``session_meta``  – id, cwd, model(_provider), cli_version, git{...}
  * ``turn_context``  – may update the active model
  * ``response_item`` – the transcript: message / function_call /
                        function_call_output / reasoning / custom_tool_call /
                        custom_tool_call_output / web_search_call
  * ``event_msg``     – runtime events incl. ``token_count`` (cumulative usage)

Codex reports *cumulative* token usage, so we diff successive ``token_count``
events to get per-turn deltas and price those. Repository comes from
``session_meta.git.repository_url`` (authoritative), falling back to cwd.
Event ids are ``<session>:L<lineno>`` (line numbers are unique per file);
the tool ``call_id`` lives in ``tool_use_id`` so calls link to their outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from collections import Counter

from . import pricing, prmatch
from .repos import resolve_session_repository
from .util import file_ext, parse_apply_patch, parse_ts

SOURCE = "codex"


def config_paths() -> list[str]:
    env = os.getenv("AI_OBSERVER_CODEX_PATH") or os.getenv("CLAUDE_ANALYSIS_CODEX_PATH")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        p = Path(codex_home) / "sessions"
        return [str(p)] if p.exists() else []
    p = Path.home() / ".codex" / "sessions"
    return [str(p)] if p.exists() else []


def find_session_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for base in paths:
        files.extend(str(p) for p in Path(base).rglob("*.jsonl"))
    return sorted(files)


def _message_text(content) -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for c in content:
        if isinstance(c, dict) and c.get("type") in ("input_text", "output_text") and c.get("text"):
            parts.append(c["text"])
    return "\n".join(parts)


def _reasoning_text(summary) -> str:
    if not isinstance(summary, list):
        return ""
    parts = [s.get("text", "") for s in summary if isinstance(s, dict) and s.get("type") == "summary_text"]
    return "\n".join(p for p in parts if p)


def _patch_text(payload: dict) -> str:
    """apply_patch carries the patch in either ``input`` (custom_tool_call) or
    ``arguments`` (function_call, sometimes JSON-wrapped)."""
    raw = payload.get("input") or payload.get("arguments") or ""
    if not raw:
        return ""
    if raw.lstrip().startswith("{"):
        try:
            obj = json.loads(raw)
            for k in ("input", "patch", "content"):
                if isinstance(obj.get(k), str):
                    return obj[k]
        except json.JSONDecodeError:
            pass
    return raw


def parse_file(path: str) -> tuple[dict, list[dict]] | None:
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

    # session-level state, filled as we encounter session_meta
    session_id = Path(path).stem
    cwd = repository_url = git_branch = git_commit = ""
    cli_version = originator = model_provider = ""
    current_model = ""
    model_counts: dict[str, int] = {}
    last_token_total: dict | None = None
    events: list[dict] = []

    def base_event(lineno, ts, etype, *, suffix=None):
        eid = f"{SOURCE}:{session_id}:L{lineno}"
        if suffix is not None:
            eid = f"{eid}#{suffix}"
        return {
            "event_id": eid,
            "session_id": session_id,
            "source": SOURCE,
            "seq": lineno,
            "block_index": suffix or 0,
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
            "git_branch": git_branch or None,
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

    for lineno, entry in lines:
        etype = entry.get("type")
        payload = entry.get("payload") or {}
        ts = parse_ts(entry.get("timestamp"))

        if etype == "session_meta":
            if payload.get("id"):
                session_id = payload["id"]
            cwd = payload.get("cwd", cwd)
            originator = payload.get("originator", originator)
            cli_version = payload.get("cli_version", cli_version)
            model_provider = payload.get("model_provider", model_provider)
            if payload.get("model"):
                current_model = payload["model"]
            git = payload.get("git") or {}
            repository_url = git.get("repository_url", repository_url)
            git_branch = git.get("branch", git_branch)
            git_commit = git.get("commit_hash", git_commit)
            ev = base_event(lineno, ts, etype)
            ev["session_id"] = session_id
            ev["subtype"] = "session_meta"
            ev["role"] = "system"
            ev["cwd"] = cwd or None
            ev["git_branch"] = git_branch or None
            ev["attributes"] = entry
            events.append(ev)
            continue

        if etype == "turn_context":
            if payload.get("model"):
                current_model = payload["model"]
            ev = base_event(lineno, ts, etype)
            ev["subtype"] = "turn_context"
            ev["role"] = "system"
            ev["attributes"] = entry
            events.append(ev)
            continue

        if etype == "event_msg":
            sub = payload.get("type")
            ev = base_event(lineno, ts, etype)
            ev["subtype"] = sub
            ev["role"] = "event"
            ev["attributes"] = entry
            if sub == "token_count":
                _attach_codex_tokens(ev, payload, current_model, last_token_total)
                info = payload.get("info") or {}
                total = info.get("total_token_usage")
                if isinstance(total, dict):
                    last_token_total = total
                if current_model:
                    model_counts[current_model] = model_counts.get(current_model, 0) + 1
            events.append(ev)
            continue

        if etype == "response_item":
            sub = payload.get("type")
            call_id = payload.get("call_id")

            if sub == "message":
                ev = base_event(lineno, ts, etype)
                ev["subtype"] = "message"
                ev["role"] = payload.get("role") or "assistant"
                ev["text"] = _message_text(payload.get("content"))
                ev["attributes"] = entry
                events.append(ev)

            elif sub == "reasoning":
                ev = base_event(lineno, ts, etype)
                ev["subtype"] = "reasoning"
                ev["role"] = "reasoning"
                ev["text"] = _reasoning_text(payload.get("summary"))
                ev["attributes"] = entry
                events.append(ev)

            elif sub in ("function_call", "custom_tool_call", "local_shell_call", "web_search_call"):
                name = payload.get("name") or sub
                if name == "apply_patch":
                    files = parse_apply_patch(_patch_text(payload))
                    if files:
                        for i, f in enumerate(files):
                            ev = base_event(lineno, ts, etype, suffix=(i if i else None))
                            ev["subtype"] = sub
                            ev["role"] = "tool_use"
                            ev["tool_name"] = name
                            ev["tool_use_id"] = call_id
                            ev["file_path"] = f["file"]
                            ev["file_ext"] = file_ext(f["file"])
                            ev["lines_added"] = f["added"]
                            ev["lines_removed"] = f["removed"]
                            ev["text"] = f"apply_patch: {f['op']} {f['file']}"
                            ev["attributes"] = entry if i == 0 else {"file": f, "call_id": call_id}
                            events.append(ev)
                        continue
                ev = base_event(lineno, ts, etype)
                ev["subtype"] = sub
                ev["role"] = "tool_use"
                ev["tool_name"] = name
                ev["tool_use_id"] = call_id
                ev["text"] = payload.get("arguments") or payload.get("input") or f"Tool call: {name}"
                ev["attributes"] = entry
                events.append(ev)

            elif sub in ("function_call_output", "custom_tool_call_output"):
                ev = base_event(lineno, ts, etype)
                ev["subtype"] = sub
                ev["role"] = "tool_result"
                ev["tool_use_id"] = call_id
                out = payload.get("output")
                if isinstance(out, str):
                    ev["text"] = out
                elif out is not None:
                    ev["text"] = json.dumps(out, default=str)
                ev["attributes"] = entry
                events.append(ev)

            else:
                ev = base_event(lineno, ts, etype)
                ev["subtype"] = sub
                ev["role"] = "tool_use" if "call" in (sub or "") else "other"
                ev["tool_use_id"] = call_id
                ev["attributes"] = entry
                events.append(ev)
            continue

        # unknown top-level type: keep it, lossless
        ev = base_event(lineno, ts, etype)
        ev["attributes"] = entry
        events.append(ev)

    # mine PR/repo references from each event's text (covers exec command
    # arguments and tool outputs, where Codex's only PR signal lives)
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

    # resolve repository once, stamp every event. Codex's git remote (from
    # session_meta) is authoritative; otherwise fall back to the cwd reconciled
    # with mined owner/repo references, most-frequent first.
    ref_counts = Counter(ev["referenced_repository"] for ev in events if ev.get("referenced_repository"))
    candidates = [r for r, _ in ref_counts.most_common()]
    repository = resolve_session_repository(
        candidate_repositories=candidates,
        git_repository_url=repository_url,
        cwd=cwd,
    )
    for ev in events:
        ev["repository"] = repository

    main_model = max(model_counts, key=model_counts.get) if model_counts else current_model

    meta = {
        "session_id": session_id,
        "source": SOURCE,
        "file_path": path,
        "repository": repository,
        "repository_url": repository_url,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "model": main_model,
        "cli_version": cli_version,
        "originator": originator,
        "model_provider": model_provider or "openai",
        "cwd": cwd,
        "custom_title": "",
        "ai_title": "",
        "pr_repositories": [],
        "pr_numbers": [],
        "extra_attributes": {"model_counts": model_counts},
    }
    return meta, events


def _attach_codex_tokens(ev: dict, payload: dict, model: str, last_total: dict | None) -> None:
    """Diff cumulative ``total_token_usage`` against the previous token_count to
    get per-turn deltas, then price them."""
    info = payload.get("info") or {}
    total = info.get("total_token_usage")
    if not isinstance(total, dict):
        return

    def cur(k, *alts):
        v = total.get(k)
        if v is None:
            for a in alts:
                if total.get(a) is not None:
                    return total[a]
            return 0
        return v

    cur_in = cur("input_tokens")
    cur_out = cur("output_tokens")
    cur_cc = cur("cache_creation_input_tokens")
    cur_cr = cur("cache_read_input_tokens", "cached_input_tokens")
    cur_reason = cur("reasoning_output_tokens")

    if last_total is None:
        d_in, d_out, d_cc, d_cr, d_reason = cur_in, cur_out, cur_cc, cur_cr, cur_reason
    else:
        def prev(k, *alts):
            v = last_total.get(k)
            if v is None:
                for a in alts:
                    if last_total.get(a) is not None:
                        return last_total[a]
                return 0
            return v
        d_in = cur_in - prev("input_tokens")
        d_out = cur_out - prev("output_tokens")
        d_cc = cur_cc - prev("cache_creation_input_tokens")
        d_cr = cur_cr - prev("cache_read_input_tokens", "cached_input_tokens")
        d_reason = cur_reason - prev("reasoning_output_tokens")

    ev["input_tokens"] = max(0, d_in)
    # calculated_input_tokens left None for Codex — token counts come directly
    # from the API cumulative diff and don't need a local estimate
    ev["output_tokens"] = max(0, d_out)
    ev["cache_creation_tokens"] = max(0, d_cc)
    ev["cache_read_tokens"] = max(0, d_cr)
    ev["reasoning_tokens"] = max(0, d_reason)
    cost = pricing.codex_cost(model, max(0, d_in), max(0, d_cr), max(0, d_out))
    ev["stated_cost"] = cost or None
    ev["inferred_cost"] = cost or None
