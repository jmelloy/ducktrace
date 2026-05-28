#!/usr/bin/env python3
"""
cc_stats.py — Claude Code project JSONL analyzer

Usage:
  python cc_stats.py [--dir ~/.claude/projects] [--days 30] [--json] [--project SLUG]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_ts(ts):
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            pass
    return None


def ext(path):
    return Path(path).suffix.lower() or "(no ext)"


def count_lines(text):
    return len(text.splitlines()) if text else 0


def iter_jsonl(path):
    try:
        f = open(path, "r", errors="replace")
    except OSError:
        return
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] {path.name}:{lineno}: {e}", file=sys.stderr)


PR_PATTERNS = [
    re.compile(r"gh\s+pr\s+(create|merge|close|checkout|list|view|diff|review)", re.I),
    re.compile(r"git\s+(push|pull)\b.*(?:--pr|pull-request)", re.I),
    re.compile(r"(?:pull.request|pr)[#/\s]+(\d+)", re.I),
    re.compile(r"(?:open|create|merge|close)\s+(?:a\s+)?pr\b", re.I),
]

def extract_pr_actions(cmd):
    return [m.group(0)[:60] for pat in PR_PATTERNS for m in [pat.search(cmd)] if m]


REPO_FLAG_RE = re.compile(r"--repo\s+(\S+)", re.I)

def repo_from_gh_flag(cmd):
    """Extract --repo owner/name from a gh command; returns last path component."""
    m = REPO_FLAG_RE.search(cmd)
    if not m:
        return None
    val = m.group(1)
    return val.split("/")[-1] if "/" in val else val


def resolve_repo_name(cwd):
    """Return repo name for cwd, resolving git worktrees to the main repo name."""
    if not cwd:
        return ""
    p = Path(cwd)
    git_path = p / ".git"
    if git_path.is_file():
        # Worktree: .git is a file containing "gitdir: /path/to/.git/worktrees/<name>"
        try:
            content = git_path.read_text().strip()
            if content.startswith("gitdir:"):
                gitdir = Path(content[len("gitdir:"):].strip())
                parts = gitdir.parts
                try:
                    dot_git_idx = next(i for i, seg in enumerate(parts) if seg == ".git")
                    if dot_git_idx > 0:
                        return Path(*parts[:dot_git_idx]).name
                except StopIteration:
                    pass
        except OSError:
            pass
    return p.name


def process_entry(entry, day_tokens, day_edits, ext_edits, repo_stats, pr_events, cutoff):
    ts = parse_ts(entry.get("timestamp"))
    if ts and cutoff and ts < cutoff:
        return

    day    = ts.date().isoformat() if ts else "unknown"
    msg    = entry.get("message", {}) or {}
    usage  = msg.get("usage", {}) or {}
    cwd    = entry.get("cwd", "")
    branch = entry.get("gitBranch", "")
    repo   = resolve_repo_name(cwd)

    # tokens
    if usage:
        d = day_tokens[day]
        d["input"]        += usage.get("input_tokens", 0)
        d["output"]       += usage.get("output_tokens", 0)
        d["cache_read"]   += usage.get("cache_read_input_tokens", 0)
        d["cache_create"] += usage.get("cache_creation_input_tokens", 0)
        d["cost_usd"]     += entry.get("costUSD", 0.0)

    # repo
    if repo:
        rs = repo_stats[repo]
        rs["days"].add(day)
        if branch:
            rs["branches"].add(branch)

    # top-level PR fields
    pr_number = entry.get("prNumber")
    pr_url    = entry.get("prUrl", "")
    if pr_number or pr_url:
        pr_events.append({
            "day": day, "repo": repo, "branch": branch,
            "action": f"prNumber={pr_number}", "url": pr_url,
        })

    # tool calls
    content = msg.get("content", [])
    if not isinstance(content, list):
        return

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue

        name  = block.get("name", "")
        inp   = block.get("input", {}) or {}
        fpath = inp.get("path", inp.get("file_path", ""))

        if name in ("Edit", "str_replace", "str_replace_based_edit_tool"):
            old     = inp.get("old_string", inp.get("old_str", ""))
            new     = inp.get("new_string", inp.get("new_str", ""))
            removed = count_lines(old)
            added   = count_lines(new)
            e = ext(fpath) if fpath else "(unknown)"
            ext_edits[e]["added"]   += added
            ext_edits[e]["removed"] += removed
            day_edits[day]["added"]   += added
            day_edits[day]["removed"] += removed

        elif name in ("Write", "write_file", "create", "create_file"):
            content_text = inp.get("content", inp.get("file_text", ""))
            added = count_lines(content_text)
            e = ext(fpath) if fpath else "(unknown)"
            ext_edits[e]["added"]   += added
            day_edits[day]["added"] += added

        elif name in ("Bash", "bash", "run_bash", "execute_bash"):
            cmd = inp.get("command", inp.get("cmd", ""))
            if cmd:
                cmd_repo = repo_from_gh_flag(cmd) or repo
                for a in extract_pr_actions(cmd):
                    pr_events.append({"day": day, "repo": cmd_repo, "branch": branch, "action": a, "url": ""})


def load_all(projects_dir, days, project_filter, debug=False):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)) if days > 0 else None

    day_tokens = defaultdict(lambda: dict(input=0, output=0, cache_read=0, cache_create=0, cost_usd=0.0))
    day_edits  = defaultdict(lambda: dict(added=0, removed=0))
    ext_edits  = defaultdict(lambda: dict(added=0, removed=0))
    repo_stats = defaultdict(lambda: dict(days=set(), branches=set()))
    pr_events  = []

    jsonl_files = sorted(projects_dir.rglob("*.jsonl"))
    if project_filter:
        jsonl_files = [f for f in jsonl_files if project_filter.lower() in str(f).lower()]

    print(f"Scanning {len(jsonl_files)} JSONL file(s) in {projects_dir}", file=sys.stderr)

    for jf in jsonl_files:
        if debug:
            print(f"  [debug] reading {jf}", file=sys.stderr)
        entry_count = 0
        pr_before = len(pr_events)
        for entry in iter_jsonl(jf):
            process_entry(entry, day_tokens, day_edits, ext_edits, repo_stats, pr_events, cutoff)
            entry_count += 1
        if debug:
            pr_found = len(pr_events) - pr_before
            repos_seen = {e["repo"] for e in pr_events[pr_before:] if e["repo"]}
            repo_note = f"  repos={sorted(repos_seen)}" if repos_seen else ""
            print(f"  [debug]   {entry_count} entries, {pr_found} PR events{repo_note}", file=sys.stderr)

    return day_tokens, day_edits, ext_edits, repo_stats, pr_events


def hr(char="─", width=72):
    print(char * width)


def render_text(day_tokens, day_edits, ext_edits, repo_stats, pr_events):
    hr()
    print("TOKEN USAGE BY DAY")
    hr()
    total = dict(input=0, output=0, cache_read=0, cache_create=0, cost_usd=0.0)
    if day_tokens:
        print(f"{'Date':<12} {'Input':>10} {'Output':>10} {'CacheRd':>10} {'CacheCr':>10} {'Cost($)':>9}")
        hr("-")
        for day in sorted(day_tokens):
            d = day_tokens[day]
            print(f"{day:<12} {d['input']:>10,} {d['output']:>10,} {d['cache_read']:>10,} {d['cache_create']:>10,} {d['cost_usd']:>9.4f}")
            for k in total:
                total[k] += d[k]
        hr("-")
        print(f"{'TOTAL':<12} {total['input']:>10,} {total['output']:>10,} {total['cache_read']:>10,} {total['cache_create']:>10,} {total['cost_usd']:>9.4f}")
    else:
        print("  (no token data)")

    hr()
    print("LINES EDITED BY DAY")
    hr()
    if day_edits:
        print(f"{'Date':<12} {'Added':>10} {'Removed':>10} {'Net':>10}")
        hr("-")
        ta, tr = 0, 0
        for day in sorted(day_edits):
            d = day_edits[day]
            net = d["added"] - d["removed"]
            print(f"{day:<12} {d['added']:>10,} {d['removed']:>10,} {net:>+10,}")
            ta += d["added"]; tr += d["removed"]
        hr("-")
        print(f"{'TOTAL':<12} {ta:>10,} {tr:>10,} {ta-tr:>+10,}")
    else:
        print("  (no edit data)")

    hr()
    print("EDITS BY FILE TYPE")
    hr()
    if ext_edits:
        print(f"{'Ext':<16} {'Added':>10} {'Removed':>10} {'Net':>10}")
        hr("-")
        for e, d in sorted(ext_edits.items(), key=lambda x: -(x[1]["added"] + x[1]["removed"])):
            print(f"{e:<16} {d['added']:>10,} {d['removed']:>10,} {d['added']-d['removed']:>+10,}")
    else:
        print("  (no file-type data)")

    hr()
    print("REPOSITORY ACTIVITY")
    hr()
    if repo_stats:
        print(f"{'Repo':<30} {'Active days':>12}  Branches")
        hr("-")
        for repo, rs in sorted(repo_stats.items(), key=lambda x: -len(x[1]["days"])):
            branches = ", ".join(sorted(rs["branches"])[:5])
            if len(rs["branches"]) > 5:
                branches += f" (+{len(rs['branches'])-5} more)"
            print(f"{repo:<30} {len(rs['days']):>12}  {branches}")
    else:
        print("  (no repository data)")

    hr()
    print(f"PR EVENTS  ({len(pr_events)} total)")
    hr()
    if pr_events:
        seen = set()
        deduped = []
        for ev in pr_events:
            key = ev.get("url") or f"{ev['day']}:{ev['action']}"
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        for ev in deduped[-30:]:
            url_part = f"  {ev['url']}" if ev.get("url") else ""
            print(f"  {ev['day']}  [{ev['repo'] or '?'}:{ev['branch'] or '?'}]  {ev['action']}{url_part}")
        if len(deduped) > 30:
            print(f"  ... ({len(deduped)-30} earlier omitted; use --json for full list)")
        if len(pr_events) != len(deduped):
            print(f"  ({len(pr_events) - len(deduped)} duplicate references deduplicated)")
    else:
        print("  (no PR events detected)")

    hr()


def render_json(day_tokens, day_edits, ext_edits, repo_stats, pr_events):
    print(json.dumps({
        "token_usage_by_day": {d: {**v, "cost_usd": round(v["cost_usd"], 6)} for d, v in sorted(day_tokens.items())},
        "lines_edited_by_day": {d: dict(v) for d, v in sorted(day_edits.items())},
        "edits_by_extension": {e: dict(v) for e, v in sorted(ext_edits.items())},
        "repositories": {r: {"active_days": len(v["days"]), "branches": sorted(v["branches"])} for r, v in sorted(repo_stats.items())},
        "pr_events": pr_events,
    }, indent=2))


def main():
    default_dir = Path.home() / ".claude" / "projects"
    ap = argparse.ArgumentParser(description="Analyze Claude Code project JSONL files")
    ap.add_argument("--dir",     default=str(default_dir))
    ap.add_argument("--days",    type=int, default=0, help="Last N days (0 = all)")
    ap.add_argument("--project", default=None,        help="Filter by project name substring")
    ap.add_argument("--json",    action="store_true", help="Output JSON")
    ap.add_argument("--debug",   action="store_true", help="Print per-file debug info to stderr")
    args = ap.parse_args()

    projects_dir = Path(args.dir).expanduser()
    if not projects_dir.exists():
        sys.exit(f"Error: not found: {projects_dir}")

    data = load_all(projects_dir, args.days, args.project, debug=args.debug)

    if args.json:
        render_json(*data)
    else:
        label = f"Last {args.days} days" if args.days else "All time"
        print(f"\n{label}  |  {projects_dir}\n")
        render_text(*data)


if __name__ == "__main__":
    main()
