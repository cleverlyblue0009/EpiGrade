"""Pre-push guard: refuse to push if any staged or to-be-pushed file is larger than 50 MB or
sits under data/. Data and large artifacts must never reach the repository (see .gitignore and
docs/SETUP.md); .gitignore alone does not stop a `git add -f`.

    python scripts/pre_push_guard.py            # check staged files + commits not yet on origin
    python scripts/pre_push_guard.py --install  # install as .git/hooks/pre-push

Exits 1 and lists every offending path if the check fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
FORBIDDEN_PREFIXES = ("data/",)
REPO_ROOT = Path(__file__).resolve().parents[1]

HOOK = """#!/bin/sh
# Installed by scripts/pre_push_guard.py --install
exec python scripts/pre_push_guard.py
"""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def staged_files() -> set[str]:
    return set(_git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines())


def unpushed_files() -> set[str]:
    """Files touched by commits on HEAD that the upstream does not have yet. With no upstream
    configured (a fresh clone's first push), every tracked file is checked instead."""
    try:
        upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}").strip()
    except subprocess.CalledProcessError:
        return set(_git("ls-files").splitlines())
    return set(
        _git("diff", "--name-only", "--diff-filter=ACMR", f"{upstream}..HEAD").splitlines()
    )


def blob_size(path: str, staged: bool) -> int:
    """Size of the blob git would actually push - the index version for a staged file, the
    committed version otherwise - not whatever happens to be in the working tree."""
    ref = f":{path}" if staged else f"HEAD:{path}"
    try:
        return int(_git("cat-file", "-s", ref).strip())
    except subprocess.CalledProcessError:
        return 0


def find_violations(files: dict[str, int]) -> list[str]:
    """`files` maps repo-relative path -> size in bytes. Pure, so it can be unit-tested."""
    problems = []
    for path in sorted(files):
        normalised = path.replace("\\", "/")
        if normalised.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{path}: under data/ (data is never committed)")
        if files[path] > MAX_BYTES:
            problems.append(
                f"{path}: {files[path] / 1024 / 1024:.1f} MB exceeds the "
                f"{MAX_BYTES // 1024 // 1024} MB limit"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="install as a git pre-push hook")
    args = parser.parse_args()

    if args.install:
        hook_path = REPO_ROOT / ".git" / "hooks" / "pre-push"
        hook_path.write_text(HOOK, encoding="utf-8", newline="\n")
        hook_path.chmod(0o755)
        print(f"Installed {hook_path}")
        return 0

    staged = staged_files()
    files = {p: blob_size(p, staged=True) for p in staged}
    for p in unpushed_files() - staged:
        files[p] = blob_size(p, staged=False)

    problems = find_violations(files)
    if problems:
        print("pre-push guard FAILED:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"pre-push guard passed ({len(files)} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
