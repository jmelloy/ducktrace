# raw/

This directory contains **intentionally unredacted** session JSONL files used as input for the `clean_session.py` test suite.

These files are synthetic fixtures — they do not contain real user data — but they are structured to include realistic PII-like values (home paths, UUIDs, branch names, API keys, etc.) so the cleaner's redaction logic can be validated against them.

**Do not** add real session files here. The output of running `clean_session.py` on these files is stored in the parent `sessions/` directory.
