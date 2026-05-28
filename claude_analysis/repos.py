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
import subprocess
from functools import lru_cache
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


@lru_cache(maxsize=8192)
def cwd_git_remote(path: str) -> str:
    """The `owner/repo` of the cwd's ``origin`` remote, read from disk, or "".

    This is the authoritative repo for a session whose working directory still
    exists locally — it's the real remote, so it's correct even when the local
    directory name differs from the repo name (e.g. dir ``claude-analysis`` →
    repo ``jmelloy/ducktrace``). Sessions whose cwd has moved/disappeared just
    fall through to the other signals. Cached per path."""
    if not path:
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", path, "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return normalize_git_url(out.stdout.strip())


def resolve_session_repository(
    *,
    candidate_repositories: list[str] | None = None,
    git_repository_url: str = "",
    original_cwd: str = "",
    cwd: str = "",
) -> str:
    """Return the best repository id for a session.

    ``candidate_repositories`` are ``owner/repo`` references seen for the session
    — from a Claude ``pr-link`` *and* from the agent's own commands/output
    (``gh pr --repo``, git remotes, clone/PR URLs), ideally ordered most-frequent
    first.

    The working directory says *which* repo we're in; the candidates supply the
    ``owner/`` that the cwd alone can't (Claude's logs never record the git
    remote). So the highest-confidence answer is a candidate whose repo name
    matches the cwd's name or a path segment of it (covers working in a subdir
    like ``<repo>/backend``). A candidate that doesn't match the cwd is treated
    as a stray — e.g. some other repo's PR URL merely mentioned in the chat — and
    used only when there's no working directory at all. Otherwise we fall back to
    the bare cwd name, which canonicalization may later upgrade.
    """
    # Codex carries the cwd's actual git remote — authoritative and consistent.
    if git_repository_url:
        norm = normalize_git_url(git_repository_url)
        if norm:
            return norm

    path = original_cwd or cwd

    # The cwd's own git remote on disk is authoritative (handles dir name ≠ repo
    # name, and ignores repos merely mentioned in the chat).
    on_disk = cwd_git_remote(path)
    if on_disk:
        return on_disk

    cwd_name = resolve_repo_name(path)
    candidates = [r for r in (candidate_repositories or []) if r]

    if candidates and path:
        segments = {seg.lower() for seg in Path(path).parts}
        for r in candidates:
            name = r.split("/")[-1].lower()
            if name == cwd_name.lower() or name in segments:
                return r

    if cwd_name:
        return cwd_name

    # No working directory to go on — a referenced repo is better than nothing.
    return candidates[0] if candidates else ""
