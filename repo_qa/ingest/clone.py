"""Get a repository onto disk: validate the GitHub URL, then do a safe shallow clone.

Safety notes:
- Only public https://github.com/<owner>/<repo> URLs are accepted, so a pasted string
  can never become a git option, a local path, or a request to another host.
- git runs without a shell, with the URL after `--`, with credential prompts turned
  off (a private repo fails fast instead of waiting for a password forever), and with a
  hard timeout.
- Symlinks are written as plain files (core.symlinks=false), so a malicious repo cannot
  point us at files outside the clone.
- The clone goes into a temporary folder that is renamed only on success, so a timeout
  or crash never leaves a half-finished repository that looks complete.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common.errors import CloneError

log = logging.getLogger(__name__)

_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9._-]{1,100}?"
_REF = r"[A-Za-z0-9._/-]{1,200}"
_URL_RE = re.compile(
    rf"^https://github\.com/(?P<owner>{_OWNER})/(?P<repo>{_REPO})(?:\.git)?"
    rf"(?:/tree/(?P<ref>{_REF}))?/?$"
)
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    ref: str | None  # branch or tag from a /tree/<ref> URL, else None (default branch)

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"

    @property
    def dir_name(self) -> str:
        return f"{self.owner}__{self.repo}" + (f"@{self.ref.replace('/', '_')}" if self.ref else "")


def looks_like_github_url(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped.startswith(("https://github.com/", "http://github.com/", "github.com/"))


def parse_repo_url(url: str) -> RepoRef:
    """Validate a GitHub URL and split it into owner, repo and optional branch."""
    text = url.strip()
    if text.lower().startswith("github.com/"):
        text = "https://" + text
    if text.lower().startswith("http://github.com/"):
        text = "https://" + text[len("http://"):]
    match = _URL_RE.match(text)
    if not match:
        raise CloneError(
            f"Not a supported repository URL: {url!r}. Use a public GitHub address such as "
            "https://github.com/owner/repo"
        )
    ref = match.group("ref")
    if ref and (ref.startswith(("-", "/")) or ".." in ref or ref.endswith("/")):
        raise CloneError(f"Unsupported branch name in URL: {ref!r}")
    repo = match.group("repo")
    if repo in (".", ".."):
        raise CloneError(f"Unsupported repository name in URL: {url!r}")
    return RepoRef(owner=match.group("owner"), repo=repo, ref=ref)


def clone_repo(
    ref: RepoRef,
    repos_dir: Path,
    *,
    timeout_s: float,
    git: str = "git",
    refresh: bool = False,
    runner: Runner = subprocess.run,
) -> Path:
    """Shallow-clone `ref` into repos_dir and return the folder.

    An existing clone is reused unless refresh=True.
    """
    dest = repos_dir / ref.dir_name
    if dest.exists() and not refresh:
        if (dest / ".git").exists():
            log.info("Reusing existing clone at %s (use refresh to re-download)", dest)
            return dest
        raise CloneError(f"{dest} exists but is not a git clone. Remove it or pick another data_dir.")
    run_git_clone(
        ref.clone_url, dest, branch=ref.ref, timeout_s=timeout_s, git=git, runner=runner
    )
    return dest


def run_git_clone(
    clone_url: str,
    dest: Path,
    *,
    branch: str | None,
    timeout_s: float,
    git: str = "git",
    runner: Runner = subprocess.run,
) -> None:
    """Run `git clone --depth 1` into a temporary folder, then move it to dest.

    Separate from clone_repo so tests can clone from a local repository.
    """
    if shutil.which(git) is None:
        raise CloneError(f"'{git}' was not found. Install git (https://git-scm.com) and retry.")

    partial = dest.with_name(dest.name + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _remove_tree(partial)

    cmd = [git, "-c", "core.symlinks=false", "clone", "--depth", "1", "--single-branch", "--no-tags"]
    if branch:
        cmd += ["--branch", branch]
    cmd += ["--", clone_url, str(partial)]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", LC_ALL="C")

    log.info("Cloning %s (timeout %.0fs)", clone_url, timeout_s)
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, check=False)
    except subprocess.TimeoutExpired as exc:
        _remove_tree(partial)
        raise CloneError(
            f"Cloning took longer than {timeout_s:.0f}s and was stopped. The repository may be "
            "very large; raise CHATBOT_CLONE_TIMEOUT_S or choose a smaller repository."
        ) from exc
    except OSError as exc:
        _remove_tree(partial)
        raise CloneError(f"Could not run git: {type(exc).__name__}") from exc

    if proc.returncode != 0:
        _remove_tree(partial)
        raise CloneError(_explain_git_failure(proc.stderr or ""))
    try:
        if dest.exists():  # refresh: replace the old clone only now that the new one is complete
            _remove_tree(dest)
        partial.rename(dest)
    except OSError as exc:
        _remove_tree(partial)
        raise CloneError(f"Could not finish the clone: {type(exc).__name__}") from exc


def _explain_git_failure(stderr: str) -> str:
    lowered = stderr.lower()
    if "remote branch" in lowered and "not found" in lowered:  # must precede the generic check
        return "That branch or tag does not exist in the repository."
    if "not found" in lowered or "could not read username" in lowered or "authentication" in lowered:
        return (
            "Repository not found, or it is private. Only public GitHub repositories are "
            "supported. Check the address for typos."
        )
    if "could not resolve host" in lowered or "unable to access" in lowered or "timed out" in lowered:
        return "Could not reach GitHub. Check your internet connection, VPN or proxy."
    tail = " ".join(stderr.strip().splitlines()[-2:])[:300]
    return f"git clone failed: {tail or 'unknown error'}"


def _remove_tree(path: Path) -> None:
    """Delete a folder tree, including read-only files git leaves behind on Windows."""

    def _make_writable_and_retry(func: Callable[[str], object], target: str, _exc: object) -> None:
        os.chmod(target, 0o700)
        func(target)

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_make_writable_and_retry)  # type: ignore[call-arg]
    else:
        shutil.rmtree(path, onerror=_make_writable_and_retry)  # type: ignore[arg-type]
