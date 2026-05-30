"""Sanitize Claude Code session JSONL files for use as test fixtures.

Redacts PII from all string values recursively: home paths, IPs, API keys,
email addresses, git author lines, and machine-specific hostnames.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

# /home/<user>/..., /Users/<user>/..., /root/...
_RE_HOME_PATH = re.compile(r"(/home/|/Users/|/root/)[^\s\"',:;)>\]]+")

# /tmp/ paths containing worktree-style worker IDs (e.g. /tmp/pioneer-work/... or /tmp/w-abc123/...)
_RE_TMP_WORKTREE = re.compile(r"/tmp/[^\s\"',:;)>\]]*/w-[a-z0-9]+[^\s\"',:;)>\]]*")

# Hyphenated variants of the same paths (slashes replaced by hyphens in tool output, memory dirs, etc.)
# e.g. "-tmp-pioneer-work-dcktrc-w-i4t6em-t-3c3q02-ducktrace"
_RE_TMP_WORKTREE_HYPH = re.compile(r"(?m)^-tmp-[a-z0-9\-]+-w-[a-z0-9]+[a-z0-9\-]*")

# IPv4 addresses
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 addresses (simplified; catches most real-world forms)
_RE_IPV6 = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:)+:[0-9a-fA-F]{0,4}\b"
    r"|\b::[0-9a-fA-F:]+\b"
)

# Anthropic API keys
_RE_SK_ANT = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")

# Bearer tokens
_RE_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.=+/]{8,}")

# Authorization headers (entire value portion, to end of line)
_RE_AUTH_HEADER = re.compile(r"(Authorization:\s*).+", re.IGNORECASE)

# Email addresses
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Git author/committer lines: "Author: Name <email>" or "Committer: Name <email>"
_RE_GIT_AUTHOR = re.compile(
    r"((?:Author|Committer|author|committer):\s*)[^<\r\n]*<[^>\r\n]*>",
    re.IGNORECASE,
)

# Single-label hostnames heuristic: word chars only, 2–63 chars, not "localhost"
# Anchored to common surrounding contexts to avoid false positives.
_RE_HOSTNAME = re.compile(
    r"(?<![./\w])([a-zA-Z][a-zA-Z0-9\-]{1,62})(?![./\w])"
)
_HOSTNAME_SKIP = {"localhost", "hostname"}


def _redact_string(s: str, counts: dict[str, int]) -> str:
    """Apply all redaction patterns to a single string. Returns cleaned string."""

    def sub(pattern, replacement, label, value=s):
        nonlocal s
        result, n = pattern.subn(replacement, s)
        if n:
            counts[label] += n
        s = result

    # Git author lines first (before email, so we replace the whole line)
    orig = s
    s, n = _RE_GIT_AUTHOR.subn(r"\1User <user@example.com>", s)
    if n:
        counts["git_author"] += n

    sub(_RE_TMP_WORKTREE, "/tmp/workdir", "tmp_worktree")
    sub(_RE_TMP_WORKTREE_HYPH, "-tmp-workdir", "tmp_worktree")
    sub(_RE_HOME_PATH, "/home/user/...", "home_path")
    sub(_RE_SK_ANT, "[REDACTED]", "api_key")
    sub(_RE_BEARER, r"\1[REDACTED]", "bearer_token")
    sub(_RE_AUTH_HEADER, r"\1[REDACTED]", "auth_header")
    sub(_RE_EMAIL, "user@example.com", "email")
    sub(_RE_IPV4, "0.0.0.0", "ipv4")
    sub(_RE_IPV6, "0.0.0.0", "ipv6")

    return s


def _redact_value(value, counts: dict[str, int]):
    """Recursively redact PII from any JSON value."""
    if isinstance(value, str):
        return _redact_string(value, counts)
    if isinstance(value, dict):
        return {k: _redact_value(v, counts) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, counts) for item in value]
    return value


def clean_record(record: dict) -> tuple[dict, dict[str, int]]:
    """Sanitize a single parsed JSONL record dict.

    Returns (cleaned_record, counts) where counts maps redaction label → hit count.
    """
    counts: dict[str, int] = defaultdict(int)
    cleaned = _redact_value(record, counts)
    return cleaned, dict(counts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_file(src: Path, output_dir: Path) -> dict[str, int]:
    """Sanitize one JSONL file, write to output_dir. Returns aggregate counts."""
    total_counts: dict[str, int] = defaultdict(int)
    out_path = output_dir / src.name
    with (
        src.open(encoding="utf-8", errors="replace") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write("\n")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue
            cleaned, counts = clean_record(record)
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            for k, v in counts.items():
                total_counts[k] += v
    return dict(total_counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize Claude Code session JSONL files for test fixtures."
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help=".jsonl session files")
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures/sessions",
        help="Destination directory (default: tests/fixtures/sessions/)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grand_total: dict[str, int] = defaultdict(int)
    for fpath in args.files:
        src = Path(fpath)
        if not src.exists():
            print(f"WARNING: {fpath} not found, skipping.", file=sys.stderr)
            continue
        counts = _process_file(src, output_dir)
        out_path = output_dir / src.name
        print(f"{src.name} -> {out_path}")
        if counts:
            for label, n in sorted(counts.items()):
                print(f"  {label}: {n} replacement(s)")
        else:
            print("  (no PII detected)")
        for k, v in counts.items():
            grand_total[k] += v

    if len(args.files) > 1:
        print("\nTotal replacements:")
        for label, n in sorted(grand_total.items()):
            print(f"  {label}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
