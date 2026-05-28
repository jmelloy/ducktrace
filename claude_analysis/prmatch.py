"""Mine PR / repository references out of free text and shell commands.

Sources covered (none of which are structured fields):
  * ``gh pr <verb>`` invocations in Bash / exec commands  -> pr_action
  * ``--repo owner/name`` flags                           -> referenced repo
  * ``github.com/<owner>/<repo>/pull/<N>`` URLs anywhere  -> repo + pr number + url
    (these also show up in ``gh pr create`` command output, i.e. tool results)

This recovers PR activity that has no ``pr-link`` entry — notably *all* Codex
PR activity, since Codex never writes pr-link records.
"""

from __future__ import annotations

import re

from .repos import github_repo_from_url

_GH_PR_RE = re.compile(
    r"\bgh\s+pr\s+(create|merge|close|reopen|checkout|view|diff|review|list|edit|comment|ready|status)\b",
    re.I,
)
_REPO_FLAG_RE = re.compile(r"--repo[ =]\s*['\"]?([^\s'\"]+)", re.I)
_PULL_URL_RE = re.compile(r"https?://github\.com/([^/\s]+/[^/\s]+?)/pull/(\d+)", re.I)


# Documentation/example placeholders that are not real repositories.
_PLACEHOLDER_REPOS = {
    "owner/repo", "owner/name", "org/repo", "user/repo", "your-org/your-repo",
    "username/repo", "owner/repository",
}


def _is_placeholder(repo: str) -> bool:
    r = repo.lower()
    return r in _PLACEHOLDER_REPOS or "<" in repo or ">" in repo


def _normalize_repo_flag(val: str) -> str:
    """A ``--repo`` value is usually ``owner/repo`` but may be a full URL."""
    gh = github_repo_from_url(val)
    if gh:
        return "" if _is_placeholder(gh) else gh
    parts = [p for p in val.strip().rstrip("/").split("/") if p]
    if len(parts) >= 2 and "." not in parts[-2]:  # avoid host.com/path being read as a repo
        repo = "/".join(parts[-2:]).removesuffix(".git")
        return "" if _is_placeholder(repo) else repo
    return ""


def _dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def extract(*texts: str) -> dict:
    """Scan one or more strings; return mined PR/repo references.

    Keys: ``pr_actions`` (list[str]), ``pr_urls`` (list[str]),
    ``pr_numbers`` (list[int]), ``repos`` (list[str] of owner/repo)."""
    actions: list[str] = []
    urls: list[str] = []
    numbers: list[int] = []
    repos: list[str] = []
    for text in texts:
        if not text:
            continue
        for m in _GH_PR_RE.finditer(text):
            actions.append("gh pr " + m.group(1).lower())
        for m in _PULL_URL_RE.finditer(text):
            repo = m.group(1).removesuffix(".git")
            if _is_placeholder(repo):
                continue
            repos.append(repo)
            numbers.append(int(m.group(2)))
            urls.append(m.group(0))
        for m in _REPO_FLAG_RE.finditer(text):
            r = _normalize_repo_flag(m.group(1))
            if r:
                repos.append(r)
    return {
        "pr_actions": _dedup(actions),
        "pr_urls": _dedup(urls),
        "pr_numbers": _dedup(numbers),
        "repos": _dedup(repos),
    }
