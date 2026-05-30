"""Tests for scripts/clean_session.py PII redaction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from clean_session import clean_record, main  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleaned(s: str) -> str:
    record = {"text": s}
    result, _ = clean_record(record)
    return result["text"]


def _counts(s: str) -> dict:
    _, counts = clean_record({"text": s})
    return counts


# ---------------------------------------------------------------------------
# PII pattern tests
# ---------------------------------------------------------------------------

class TestHomePaths:
    def test_linux_home(self):
        assert "/home/user/..." in _cleaned("/home/alice/projects/foo")

    def test_users_home(self):
        assert "/home/user/..." in _cleaned("/Users/bob/code/bar")

    def test_root_home(self):
        assert "/home/user/..." in _cleaned("/root/.ssh/id_rsa")

    def test_non_home_path_untouched(self):
        assert _cleaned("/etc/passwd") == "/etc/passwd"
        assert _cleaned("/tmp/file.txt") == "/tmp/file.txt"

    def test_tmp_worktree_path(self):
        result = _cleaned("/tmp/pioneer-work/dcktrc/w-i4t6em/t-3c3q02/ducktrace")
        assert "pioneer-work" not in result
        assert "w-i4t6em" not in result
        assert "/tmp/workdir" in result

    def test_tmp_worktree_path_variants(self):
        assert "/tmp/workdir" in _cleaned("/tmp/foo/w-abc123/project")
        assert "/tmp/workdir" in _cleaned("/tmp/pioneer-work/proj/w-xyz999/repo")

    def test_count_incremented(self):
        assert _counts("/home/alice/x /home/bob/y")["home_path"] == 2


class TestIPAddresses:
    def test_ipv4(self):
        assert "0.0.0.0" in _cleaned("connect to 192.168.1.100:8080")

    def test_ipv4_full_replace(self):
        assert _cleaned("addr=10.0.0.1") == "addr=0.0.0.0"

    def test_ipv6(self):
        assert "::" in _cleaned("2001:db8:85a3::8a2e:370:7334")
        assert "2001:db8" not in _cleaned("2001:db8:85a3::8a2e:370:7334")

    def test_loopback_replaced(self):
        # 127.0.0.1 is an IPv4 address and should be redacted
        assert "0.0.0.0" in _cleaned("127.0.0.1")

    def test_ipv6_no_false_positive_version_string(self):
        assert _cleaned("v1.2:3") == "v1.2:3"

    def test_ipv6_no_false_positive_hex_fragment(self):
        assert _cleaned("#abc123") == "#abc123"

    def test_ipv6_no_false_positive_semver(self):
        assert _cleaned("1.0.0") == "1.0.0"


class TestAPIKeys:
    def test_sk_ant_key(self):
        result = _cleaned("key=sk-ant-api03-abc123XYZ")
        assert "sk-ant" not in result
        assert "[REDACTED]" in result

    def test_bearer_token(self):
        # Test the Bearer pattern standalone (no Authorization: prefix)
        result = _cleaned("Bearer eyJhbGciOiJIUzI1NiJ9.payload")
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "[REDACTED]" in result
        assert "Bearer" in result  # label preserved

    def test_auth_header(self):
        # Full Authorization header line: entire value is redacted
        result = _cleaned("Authorization: Token abc123secret")
        assert "abc123secret" not in result
        assert "[REDACTED]" in result

    def test_count_incremented(self):
        assert _counts("sk-ant-foo sk-ant-bar")["api_key"] == 2


class TestEmailAddresses:
    def test_simple_email(self):
        assert _cleaned("contact@company.com") == "user@example.com"

    def test_email_in_sentence(self):
        result = _cleaned("send to user@corp.io please")
        assert "user@corp.io" not in result
        assert "user@example.com" in result

    def test_already_redacted_unchanged(self):
        assert _cleaned("user@example.com") == "user@example.com"


class TestGitAuthorLines:
    def test_author_line(self):
        result = _cleaned("Author: Alice Smith <alice@corp.com>")
        assert "Alice Smith" not in result
        assert "alice@corp.com" not in result
        assert "User <user@example.com>" in result

    def test_committer_line(self):
        result = _cleaned("Committer: Bob Jones <bob@org.net>")
        assert "Bob Jones" not in result
        assert "User <user@example.com>" in result

    def test_count_incremented(self):
        text = "Author: A <a@b.com>\nCommitter: B <b@c.com>"
        assert _counts(text)["git_author"] == 2


# ---------------------------------------------------------------------------
# Structure preservation tests
# ---------------------------------------------------------------------------

class TestStructurePreservation:
    def test_dict_keys_preserved(self):
        record = {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Hello /home/alice/file.py"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        cleaned, _ = clean_record(record)
        assert set(cleaned.keys()) == {"type", "message", "timestamp"}
        assert set(cleaned["message"].keys()) == {"role", "content"}

    def test_non_string_values_unchanged(self):
        record = {"count": 42, "flag": True, "nothing": None, "data": [1, 2, 3]}
        cleaned, _ = clean_record(record)
        assert cleaned["count"] == 42
        assert cleaned["flag"] is True
        assert cleaned["nothing"] is None
        assert cleaned["data"] == [1, 2, 3]

    def test_nested_list_of_dicts(self):
        record = {
            "content": [
                {"type": "text", "text": "/home/user/secret.txt"},
                {"type": "text", "text": "normal text"},
            ]
        }
        cleaned, _ = clean_record(record)
        assert "/home/user/..." in cleaned["content"][0]["text"]
        assert cleaned["content"][1]["text"] == "normal text"

    def test_deterministic(self):
        record = {"cwd": "/home/alice/work", "email": "alice@corp.com"}
        out1, _ = clean_record(record)
        out2, _ = clean_record(record)
        assert out1 == out2


# ---------------------------------------------------------------------------
# JSONL file processing tests (via CLI main())
# ---------------------------------------------------------------------------

class TestCLI:
    def test_line_count_preserved(self, tmp_path):
        src = tmp_path / "session.jsonl"
        records = [
            {"type": "user", "cwd": "/home/alice/project"},
            {"type": "assistant", "message": {"content": "hi"}},
            {"type": "ai-title", "title": "Test"},
        ]
        src.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])

        out_file = out_dir / "session.jsonl"
        assert out_file.exists()
        out_lines = [l for l in out_file.read_text().splitlines() if l.strip()]
        assert len(out_lines) == len(records)

    def test_pii_stripped_in_output(self, tmp_path):
        src = tmp_path / "s.jsonl"
        src.write_text(json.dumps({"type": "user", "email": "secret@corp.com"}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "s.jsonl").read_text()
        assert "secret@corp.com" not in content
        assert "user@example.com" in content

    def test_key_names_preserved_in_output(self, tmp_path):
        src = tmp_path / "k.jsonl"
        record = {"type": "assistant", "sessionId": "abc-123", "cwd": "/home/bob/x"}
        src.write_text(json.dumps(record) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        out = json.loads((out_dir / "k.jsonl").read_text())
        assert out["type"] == "assistant"
        assert out["sessionId"] == "abc-123"
        assert out["cwd"] == "/home/user/..."

    def test_output_filename_preserved(self, tmp_path):
        src = tmp_path / "my-session-uuid.jsonl"
        src.write_text(json.dumps({"type": "user"}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        assert (out_dir / "my-session-uuid.jsonl").exists()

    def test_empty_lines_preserved(self, tmp_path):
        src = tmp_path / "e.jsonl"
        src.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"b": 2}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        raw = (out_dir / "e.jsonl").read_text()
        assert raw.count("\n") >= 3

    def test_invalid_json_line_gets_regex_redaction(self, tmp_path):
        src = tmp_path / "bad.jsonl"
        src.write_text("not valid json but has secret@corp.com in it\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "bad.jsonl").read_text()
        assert "secret@corp.com" not in content
        assert "user@example.com" in content
