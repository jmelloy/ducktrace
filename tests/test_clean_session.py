"""Tests for scripts/clean_session.py PII redaction."""

from __future__ import annotations

import hashlib
import json
import re
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
        assert "/home/user/[REDACTED]" in _cleaned("/home/alice/projects/foo")

    def test_users_home(self):
        # Prefix style preserved: /Users/ stays /Users/ rather than becoming /home/
        assert "/Users/user/[REDACTED]" in _cleaned("/Users/bob/code/bar")

    def test_root_home(self):
        # /root paths keep /root prefix so the replacement isn't misleadingly Linux home-like
        assert "/root/[REDACTED]" in _cleaned("/root/.ssh/id_rsa")

    def test_bare_root(self):
        assert "/root/[REDACTED]" in _cleaned("/root")

    def test_root_trailing_slash_only(self):
        assert "/root/[REDACTED]" in _cleaned("/root/")

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

    def test_path_with_parens_fully_captured(self):
        # The entire path including components after closing parens must be redacted.
        # Previously, `)` in the exclusion set caused the match to stop at `(1`, leaving
        # `)/subdir` in the output — that would make both assertions below coincidentally
        # pass while still leaking path content.
        result = _cleaned("/home/user/dir(1)/subdir")
        assert result == "/home/user/[REDACTED]"  # entire input consumed

    def test_path_with_brackets_fully_captured(self):
        result = _cleaned("/home/user/dir[0]/file")
        assert result == "/home/user/[REDACTED]"  # entire input consumed


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
        assert "/home/user/[REDACTED]" in cleaned["content"][0]["text"]
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
        real_uuid = "550e8400-e29b-41d4-a716-446655440000"
        record = {"type": "assistant", "sessionId": real_uuid, "cwd": "/home/bob/x"}
        src.write_text(json.dumps(record) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        out = json.loads((out_dir / "k.jsonl").read_text())
        assert out["type"] == "assistant"
        # Key name is preserved; the UUID value is hashed to a deterministic placeholder
        assert "sessionId" in out
        assert out["sessionId"] != real_uuid
        assert _UUID_PATTERN.fullmatch(out["sessionId"]), "sessionId should be UUID-shaped after hashing"
        assert out["cwd"] == "/home/user/[REDACTED]"

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

    def test_invalid_json_always_emits_warning(self, tmp_path, capsys):
        src = tmp_path / "warn.jsonl"
        src.write_text("not valid json at all, no pii here\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "non-JSON" in captured.err

    def test_invalid_json_unicode_escape_decoded(self, tmp_path):
        # @ is the @ sign — a raw JSON line with a unicode-escaped email must
        # still be redacted even though the @ is not literally present in the text.
        src = tmp_path / "uescape.jsonl"
        src.write_text("not json: user\\u0040corp.com is here\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "uescape.jsonl").read_text()
        assert "corp.com" not in content
        assert "user@example.com" in content


# ---------------------------------------------------------------------------
# Dict-key redaction tests
# ---------------------------------------------------------------------------

class TestDictKeyRedaction:
    # Keys are NOT redacted by default; pass redact_keys=True to opt in.

    def test_keys_not_redacted_by_default(self):
        record = {"user@corp.com": "some value"}
        cleaned, counts = clean_record(record)
        assert "user@corp.com" in cleaned
        assert counts.get("email", 0) == 0

    def test_email_in_key_redacted_with_flag(self):
        record = {"user@corp.com": "some value"}
        cleaned, counts = clean_record(record, redact_keys=True)
        assert "user@corp.com" not in cleaned
        assert "user@example.com" in cleaned
        assert counts.get("email", 0) >= 1

    def test_api_key_in_key_redacted_with_flag(self):
        record = {"sk-ant-api03-abc123XYZ": "token value"}
        cleaned, counts = clean_record(record, redact_keys=True)
        assert "sk-ant-api03-abc123XYZ" not in cleaned
        assert "[REDACTED]" in cleaned
        assert counts.get("api_key", 0) >= 1

    def test_nested_dict_key_redacted_with_flag(self):
        record = {"outer": {"user@nested.com": "val"}}
        cleaned, _ = clean_record(record, redact_keys=True)
        assert "user@nested.com" not in str(cleaned)
        assert "user@example.com" in str(cleaned)

    def test_clean_keys_pass_through(self):
        record = {"type": "assistant", "sessionId": "abc-123"}
        cleaned, _ = clean_record(record)
        assert "type" in cleaned
        assert "sessionId" in cleaned

    def test_redact_keys_cli_flag(self, tmp_path):
        src = tmp_path / "keys.jsonl"
        src.write_text('{"user@corp.com": "value"}\n')
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir), "--redact-keys"])
        content = (out_dir / "keys.jsonl").read_text()
        assert "user@corp.com" not in content
        assert "user@example.com" in content


# ---------------------------------------------------------------------------
# --skip-malformed flag tests
# ---------------------------------------------------------------------------

class TestSkipMalformed:
    def test_skip_malformed_omits_line(self, tmp_path):
        src = tmp_path / "mixed.jsonl"
        src.write_text(
            '{"type": "user"}\n'
            'not valid json\n'
            '{"type": "assistant"}\n'
        )
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir), "--skip-malformed"])
        out_lines = [l for l in (out_dir / "mixed.jsonl").read_text().splitlines() if l.strip()]
        assert len(out_lines) == 2
        assert all(json.loads(l) for l in out_lines)

    def test_skip_malformed_still_warns(self, tmp_path, capsys):
        src = tmp_path / "bad.jsonl"
        src.write_text("not json\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir), "--skip-malformed"])
        assert "WARNING" in capsys.readouterr().err

    def test_default_writes_malformed_line(self, tmp_path):
        src = tmp_path / "bad.jsonl"
        src.write_text("not json but secret@corp.com here\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "bad.jsonl").read_text()
        assert content.strip() != ""
        assert "secret@corp.com" not in content


# ---------------------------------------------------------------------------
# UUID redaction tests
# ---------------------------------------------------------------------------

_UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SAMPLE_UUID = "a1164fe6-b620-4c21-acf6-8aa27cab7e59"
_SAMPLE_UUID2 = "034e2365-4ff6-467a-baf7-cfb8276a17db"


class TestUUIDRedaction:
    def test_uuid_value_is_redacted(self):
        result = _cleaned(f"session {_SAMPLE_UUID} started")
        assert _SAMPLE_UUID not in result

    def test_replacement_looks_like_uuid(self):
        result = _cleaned(_SAMPLE_UUID)
        assert _UUID_PATTERN.fullmatch(result), f"{result!r} is not UUID-shaped"

    def test_replacement_is_deterministic(self):
        assert _cleaned(_SAMPLE_UUID) == _cleaned(_SAMPLE_UUID)

    def test_different_uuids_get_different_placeholders(self):
        r1 = _cleaned(_SAMPLE_UUID)
        r2 = _cleaned(_SAMPLE_UUID2)
        assert r1 != r2

    def test_non_uuid_string_untouched(self):
        assert _cleaned("abc-123") == "abc-123"
        assert _cleaned("v1.2.3") == "v1.2.3"

    def test_count_incremented(self):
        assert _counts(f"{_SAMPLE_UUID} and {_SAMPLE_UUID2}")["uuid"] == 2

    def test_uuid_in_nested_record(self):
        record = {"sessionId": _SAMPLE_UUID, "message": {"requestId": _SAMPLE_UUID2}}
        cleaned, counts = clean_record(record)
        assert cleaned["sessionId"] != _SAMPLE_UUID
        assert cleaned["message"]["requestId"] != _SAMPLE_UUID2
        assert counts["uuid"] == 2

    def test_uuid_redaction_in_jsonl(self, tmp_path):
        src = tmp_path / "s.jsonl"
        src.write_text(json.dumps({"sessionId": _SAMPLE_UUID}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "s.jsonl").read_text()
        assert _SAMPLE_UUID not in content


# ---------------------------------------------------------------------------
# Git branch name redaction tests
# ---------------------------------------------------------------------------

class TestGitBranchRedaction:
    def test_task_branch_is_redacted(self):
        result = _cleaned("claude/clean-feature-t-3c3q")
        assert "claude" not in result
        assert "t-3c3q" not in result
        assert "feature/redacted-branch" in result

    def test_real_fixture_branch_is_redacted(self):
        branch = "claude/clean-claude-session-files-for-test-fixtures-t-3c3q"
        result = _cleaned(branch)
        assert "claude" not in result
        assert "feature/redacted-branch" in result

    def test_main_branch_not_redacted(self):
        assert _cleaned("main") == "main"

    def test_feature_branch_without_task_id_not_redacted(self):
        assert _cleaned("feature/add-login") == "feature/add-login"

    def test_branch_in_record_field(self):
        record = {"gitBranch": "claude/my-work-t-3c3q"}
        cleaned, counts = clean_record(record)
        assert "claude" not in cleaned["gitBranch"]
        assert cleaned["gitBranch"] == "feature/redacted-branch"
        assert counts["git_branch"] == 1

    def test_branch_redaction_in_jsonl(self, tmp_path):
        src = tmp_path / "b.jsonl"
        src.write_text(json.dumps({"gitBranch": "claude/my-feature-t-abc1"}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "b.jsonl").read_text()
        assert "claude" not in content
        assert "feature/redacted-branch" in content


# ---------------------------------------------------------------------------
# Request/message ID hashing tests
# ---------------------------------------------------------------------------

def _expected_req_hash(s: str) -> str:
    """Compute the expected deterministic hash for a request/message ID."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


_SAMPLE_REQ_ID = "req_01abc123defghij"
_SAMPLE_MSG_ID = "msg_01abc123defghij"
_SAMPLE_REQ_ID2 = "req_01abc123defghijk"


class TestRequestIDRedaction:
    def test_req_id_hashed(self):
        result = _cleaned(f"request {_SAMPLE_REQ_ID} started")
        assert _SAMPLE_REQ_ID not in result
        assert "[REDACTED]" not in result
        assert _expected_req_hash(_SAMPLE_REQ_ID) in result

    def test_msg_id_hashed(self):
        result = _cleaned(f"message {_SAMPLE_MSG_ID} received")
        assert _SAMPLE_MSG_ID not in result
        assert "[REDACTED]" not in result
        assert _expected_req_hash(_SAMPLE_MSG_ID) in result

    def test_hash_is_deterministic(self):
        assert _cleaned(_SAMPLE_REQ_ID) == _cleaned(_SAMPLE_REQ_ID)

    def test_different_ids_get_different_hashes(self):
        r1 = _cleaned(_SAMPLE_REQ_ID)
        r2 = _cleaned(_SAMPLE_REQ_ID2)
        assert r1 != r2

    def test_count_incremented(self):
        text = "req_01abcdefghij1 and msg_01abcdefghij2"
        assert _counts(text)["request_id"] == 2

    def test_short_req_not_redacted(self):
        # Fewer than 10 chars after underscore — too short to be a real ID
        assert _cleaned("req_short") == "req_short"

    def test_req_id_in_nested_record(self):
        record = {"requestId": "req_01abc123defghijklmno"}
        cleaned, counts = clean_record(record)
        assert "req_01abc123defghijklmno" not in cleaned["requestId"]
        assert cleaned["requestId"] == _expected_req_hash("req_01abc123defghijklmno")
        assert counts["request_id"] == 1

    def test_req_id_in_jsonl(self, tmp_path):
        src = tmp_path / "r.jsonl"
        req_id = "req_01abc123defghijklmno"
        src.write_text(json.dumps({"id": req_id}) + "\n")
        out_dir = tmp_path / "out"
        main([str(src), "--output-dir", str(out_dir)])
        content = (out_dir / "r.jsonl").read_text()
        assert req_id not in content
        assert _expected_req_hash(req_id) in content
