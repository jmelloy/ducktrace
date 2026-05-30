"""Sanitize Claude Code session JSONL files for use as test fixtures.

Redacts PII from all string values recursively: home paths, IPs, API keys,
email addresses, git author lines, UUIDs, and internal branch names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

# /home/<user>/..., /Users/<user>/..., /root/...
# Uses the same negative-class terminator set as _RE_TMP_WORKTREE so that paths
# ending before ':' (e.g. in JSON "key":"/home/user/file") are fully captured,
# and paths containing '(' or '[' are not cut short.
_RE_HOME_PATH = re.compile(r"/(?:home|Users|root)/[^\s\"',;:()\[\]>]+")

# /tmp/ paths containing worktree-style worker IDs (e.g. /tmp/pioneer-work/... or /tmp/w-abc123/...)
_RE_TMP_WORKTREE = re.compile(r"/tmp/[^\s\"',;:()\[\]>]*/w-[a-z0-9]+[^\s\"',;:()\[\]>]*")

# Hyphenated variants of the same paths (slashes replaced by hyphens in tool output, memory dirs, etc.)
# Anchored to the known worker-ID segment to avoid over-matching.
# e.g. "-tmp-pioneer-work-dcktrc-w-i4t6em-t-3c3q02-ducktrace"
_RE_TMP_WORKTREE_HYPH = re.compile(r"-tmp-w-[a-z0-9]+(?:-[a-z0-9]+)+")

# IPv4 addresses
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 addresses (simplified; catches most real-world forms)
_RE_IPV6 = re.compile(
    r"(?:\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b)"
    r"|(?:\b(?:[0-9a-fA-F]{1,4}:)+:[0-9a-fA-F]{0,4}\b)"
    r"|(?:\b::(?:[0-9a-fA-F]{1,4}:){1,6}[0-9a-fA-F]{1,4}\b)"
)

# Anthropic API keys
_RE_SK_ANT = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")

# Bearer tokens (requires token of ≥8 chars; short tokens fall through to _RE_AUTH_HEADER).
# _RE_AUTH_HEADER is the unconditional catch-all fallback: it redacts the entire value
# of any Authorization: header regardless of scheme or token length, covering cases
# where _RE_BEARER won't fire (e.g. short tokens, custom schemes like "Token abc").
_RE_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.=+/]{8,}")
_RE_AUTH_HEADER = re.compile(r"(Authorization:\s*).+", re.IGNORECASE)

# Email addresses
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Git author/committer lines: "Author: Name <email>" or "Committer: Name <email>"
_RE_GIT_AUTHOR = re.compile(
    r"((?:Author|Committer|author|committer):\s*)[^<\r\n]*<[^>\r\n]*>",
    re.IGNORECASE,
)

# UUIDs in standard format (version-agnostic)
_RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Git branch names with username/description-t-taskid pattern
# Matches branches like "claude/my-feature-t-3c3q" but not "main" or "feature/normal"
_RE_GIT_BRANCH = re.compile(
    r"\b[a-z][a-z0-9_\-]*/[a-z0-9][a-z0-9_\-]+-t-[a-z0-9]{4,}\b"
)


def _uuid_placeholder(match: re.Match) -> str:
    """Return a deterministic UUID-shaped placeholder derived from the original UUID."""
    h = hashlib.md5(match.group(0).lower().encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _redact_string(s: str, counts: dict[str, int]) -> str:
    """Apply all redaction patterns to a single string. Returns cleaned string."""

    def sub(pattern, replacement, label):
        nonlocal s
        result, n = pattern.subn(replacement, s)
        if n:
            counts[label] += n
        s = result

    # Git author lines first (before email, so we replace the whole line)
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
    sub(_RE_IPV6, "::", "ipv6")

    # UUIDs: replace with deterministic hash-based placeholder (preserves referential integrity)
    result, n = _RE_UUID.subn(_uuid_placeholder, s)
    if n:
        counts["uuid"] += n
    s = result

    # Git branch names with task IDs
    sub(_RE_GIT_BRANCH, "feature/redacted-branch", "git_branch")

    return s


def _redact_value(value, counts: dict[str, int], redact_keys: bool = False):
    """Recursively redact PII from any JSON value.

    Keys are not redacted by default because session files use stable, well-known
    key names (type, sessionId, uuid, etc.) that never contain PII; redacting them
    would corrupt the schema and make fixtures unusable as test data. Pass
    redact_keys=True (via --redact-keys) only when key names may themselves contain
    user data (uncommon).
    """
    if isinstance(value, str):
        return _redact_string(value, counts)
    if isinstance(value, dict):
        if redact_keys:
            return {_redact_string(k, counts): _redact_value(v, counts, redact_keys) for k, v in value.items()}
        return {k: _redact_value(v, counts, redact_keys) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, counts, redact_keys) for item in value]
    return value


def clean_record(record: dict, redact_keys: bool = False) -> tuple[dict, dict[str, int]]:
    """Sanitize a single parsed JSONL record dict.

    Returns (cleaned_record, counts) where counts maps redaction label → hit count.
    """
    counts: dict[str, int] = defaultdict(int)
    cleaned = _redact_value(record, counts, redact_keys=redact_keys)
    return cleaned, dict(counts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_file(
    src: Path,
    output_dir: Path,
    skip_malformed: bool = False,
    redact_keys: bool = False,
) -> dict[str, int]:
    """Sanitize one JSONL file, write to output_dir atomically. Returns aggregate counts."""
    total_counts: dict[str, int] = defaultdict(int)
    out_path = output_dir / src.name

    fd, tmp_path = tempfile.mkstemp(dir=output_dir, prefix=".tmp-", suffix=".jsonl")
    try:
        with src.open(encoding="utf-8", errors="replace") as fin, os.fdopen(fd, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line.strip():
                    fout.write("\n")
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"WARNING: non-JSON line encountered in {src.name}",
                        file=sys.stderr,
                    )
                    if skip_malformed:
                        continue
                    line_counts: dict[str, int] = defaultdict(int)
                    redacted_line = _redact_string(line, line_counts)
                    for k, v in line_counts.items():
                        total_counts[k] += v
                    fout.write(redacted_line + "\n")
                    continue
                cleaned, counts = clean_record(record, redact_keys=redact_keys)
                fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                for k, v in counts.items():
                    total_counts[k] += v
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

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
    parser.add_argument(
        "--redact-keys",
        action="store_true",
        default=False,
        help="Also redact dictionary keys (off by default; keys are normally stable schema names)",
    )
    parser.add_argument(
        "--skip-malformed",
        action="store_true",
        default=False,
        help="Skip lines that cannot be parsed as JSON instead of writing regex-redacted fallback",
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
        counts = _process_file(
            src, output_dir, skip_malformed=args.skip_malformed, redact_keys=args.redact_keys
        )
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
