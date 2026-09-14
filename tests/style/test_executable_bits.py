#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self test for scripts/check_executable_bits.py.

A mode check that had stopped seeing modes would pass every tree it was handed,
so it is handed a fixture with both kinds of wrong mode planted and must name
exactly those two files. Every file in the fixture carries the opposite mode on
disk from the one in its index, so a checker that read the filesystem instead
would name the other two. The same fixture with its index put right must pass,
and a directory with no index of its own, or a subdirectory of some other work
tree, must be refused with exit status 2 rather than passed.

The fixture is a repository with its own index in a temporary directory, and git
runs there with no global or system configuration and with discovery stopped at
that directory, so nothing here can read or write the index of the repository
under test.

Dependency free apart from git, and run under ctest, like
tests/style/check_linter.py. Exit status 2 means git is not installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = ROOT / "scripts" / "check_executable_bits.py"

# Each file's content and the executable bit its index entry is given.
FIXTURE = {
    "run.sh": ("#!/bin/sh\necho run\n", "+x"),
    "notes.txt": ("plain text\n", "-x"),
    "tool.py": ("#!/usr/bin/env python3\nprint(1)\n", "-x"),
    "data.csv": ("a,b\n1,2\n", "+x"),
}
# The two that disagree: a script committed 100644 and data committed 100755.
PLANTED = {"tool.py", "data.csv"}


def git(repo: Path, env: dict[str, str], *arguments: str) -> None:
    subprocess.run(["git", "-C", str(repo), *arguments], env=env, capture_output=True,
                   timeout=60, check=True)


def check(root: Path, env: dict[str, str]) -> tuple[int, str]:
    finished = subprocess.run([sys.executable, str(CHECKER), str(root)], env=env,
                              capture_output=True, text=True, timeout=120, check=False)
    return finished.returncode, finished.stdout + finished.stderr


def main() -> int:
    if shutil.which("git") is None:
        print("test_executable_bits: git is not installed, so no fixture index can be built")
        return 2

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch).resolve()
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CEILING_DIRECTORIES": str(base)}

        repo = base / "fixture"
        repo.mkdir()
        git(repo, env, "init", "--quiet")
        for name, (text, _) in FIXTURE.items():
            (repo / name).write_text(text, encoding="utf-8")
        git(repo, env, "add", "--", *FIXTURE)
        for name, (_, bit) in FIXTURE.items():
            git(repo, env, "update-index", f"--chmod={bit}", "--", name)
            (repo / name).chmod(0o644 if bit == "+x" else 0o755)

        code, output = check(repo, env)
        named = {name for name in FIXTURE if name in output}
        if code != 1:
            failures.append(f"the fixture with two wrong modes should exit 1, exited {code}:\n"
                            f"{output}")
        if named != PLANTED:
            failures.append(f"expected exactly {sorted(PLANTED)} to be named, got "
                            f"{sorted(named)}; the modes on disk are the opposite of the index, "
                            f"so the other two mean the filesystem was read:\n{output}")

        git(repo, env, "update-index", "--chmod=+x", "--", "tool.py")
        git(repo, env, "update-index", "--chmod=-x", "--", "data.csv")
        code, output = check(repo, env)
        if code != 0:
            failures.append(f"the fixture with its index put right should exit 0, exited "
                            f"{code}:\n{output}")

        archive = base / "archive"
        archive.mkdir()
        (archive / "tool.py").write_text(FIXTURE["tool.py"][0], encoding="utf-8")
        code, output = check(archive, env)
        if code != 2 or "not in a git work tree" not in output:
            failures.append(f"a directory with no index should exit 2 and say why, exited "
                            f"{code}:\n{output}")

        nested = repo / "vendored"
        nested.mkdir()
        (nested / "tool.py").write_text(FIXTURE["tool.py"][0], encoding="utf-8")
        code, output = check(nested, env)
        if code != 2 or "rather than at its top" not in output:
            failures.append(f"a subdirectory of another work tree should exit 2 and say why, "
                            f"exited {code}:\n{output}")

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print(f"{len(failures)} failed")
        return 1

    print("  pass  both planted modes are named, and only those, with the disk saying the "
          "opposite")
    print("  pass  the same fixture with its index put right passes")
    print("  pass  no index, and a subdirectory of another work tree, are refused with 2")
    print("3 passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
