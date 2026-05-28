"""Repository attribution.

Ported and extended from ../ai-observer's repo logic (importer/claude.go,
cmd/debug_repos). Goal: map a session to a canonical ``owner/repo`` (or at
least a repo name) using the strongest signal available.

Signal priority (best first):
  1. An explicit GitHub ``owner/repo`` from a Claude ``pr-link`` entry
     (``prRepository``) or a Codex ``session_meta.git.repository_url``.
  2. A git worktree's *original* working dir (Claude ``worktree-state``).
  3. The session's most-frequent ``cwd``, with on-disk worktree resolution
     (a ``.git`` file pointing into ``…/.git/worktrees/<name>`` resolves to
     the parent repo's directory name).
"""

from __future__ import annotations

import re
from pathlib import Path

_GITHUB_URL_RE = re.compile(r"(?:^|://|@)(?:[^/@]+@)?github\.com[:/]+([^/]+/[^/?#]+)", re.I)


def github_repo_from_url(url: str) -> str:
    """Extract ``owner/repo`` from any GitHub URL (https or ssh), stripping
    embedded credentials and a trailing ``.git``. Returns "" on no match."""
    if not url:
        return ""
    m = _GITHUB_URL_RE.search(url)
    if not m:
        return ""
    return m.group(1).removesuffix(".git")


def normalize_git_url(url: str) -> str:
    """Normalize any git remote URL to ``owner/repo`` when possible.

    Handles ``git@host:owner/repo.git``, ``https://host/owner/repo.git`` and
    ``https://<token>@host/owner/repo.git`` (credentials are dropped). Falls
    back to the last two path segments for non-GitHub hosts."""
    gh = github_repo_from_url(url)
    if gh:
        return gh
    if not url:
        return ""
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url.strip(), flags=re.I)  # scheme
    u = re.sub(r"^[^@/]+@", "", u)                                     # user/token@
    u = u.replace(":", "/", 1)                                        # ssh host:path
    u = re.sub(r"\.git$", "", u)
    parts = [p for p in u.split("/") if p]
    if len(parts) >= 3:          # host / owner / repo …
        return "/".join(parts[-2:])
    return u


def resolve_repo_name(cwd: str) -> str:
    """Repo *name* for a working dir, resolving on-disk git worktrees to the
    main repo's directory name (mirrors ai-observer). If the path no longer
    exists we just return its basename."""
    if not cwd:
        return ""
    p = Path(cwd)
    git_path = p / ".git"
    try:
        if git_path.is_file():
            content = git_path.read_text(errors="replace").strip()
            if content.startswith("gitdir:"):
                gitdir = Path(content[len("gitdir:"):].strip())
                parts = gitdir.parts
                try:
                    idx = next(i for i, seg in enumerate(parts) if seg == ".git")
                    if idx > 0:
                        return Path(*parts[:idx]).name
                except StopIteration:
                    pass
    except OSError:
        pass
    return p.name


def resolve_session_repository(
    *,
    pr_repositories: list[str] | None = None,
    git_repository_url: str = "",
    original_cwd: str = "",
    cwd: str = "",
) -> str:
    """Apply the signal-priority chain and return the best repository id."""
    if pr_repositories:
        # All pr-link repos in one session are virtually always identical; take
        # the first non-empty (already canonical owner/repo).
        for r in pr_repositories:
            if r:
                return r
    if git_repository_url:
        norm = normalize_git_url(git_repository_url)
        if norm:
            return norm
    if original_cwd:
        return resolve_repo_name(original_cwd)
    return resolve_repo_name(cwd)
