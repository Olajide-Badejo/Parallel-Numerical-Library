#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check that the executable bit and the shebang agree, as git records them.

A tracked file that begins with `#!` must be committed with mode 100755, and a
file committed with mode 100755 must begin with `#!`. ruff's EXE001 and EXE002
state the same rule, but they read the mode from the filesystem, which is where
it cannot be seen on a checkout that keeps no modes. In this repository's WSL
working copy `core.fileMode` is false and ruff skips those rules altogether, so
six scripts were committed 100644 and the first continuous integration runner
was the first thing to notice. This reads every mode from the index, which is
what a clone receives, and every first line from the blob the index names, so
it gives the same answer on every machine and never looks at the filesystem.

Exit status, because a caller has to be able to tell the three apart:

    0   every tracked file agrees
    1   at least one does not, and each is named with the command that fixes it
    2   the check could not be made at all, with the reason: git is missing, or
        the root is not the top of a git work tree, which is what an unpacked
        source archive looks like

A check that could not be made is never reported as agreement.

Usage:

    python3 scripts/check_executable_bits.py [ROOT]
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTABLE = "100755"
REGULAR = "100644"


class CannotCheck(Exception):
    """The index could not be read. Exit status 2, never 0."""


def git(root: Path, *arguments: str, stdin: bytes | None = None) -> bytes:
    finished = subprocess.run(
        ["git", "-C", str(root), *arguments], input=stdin, capture_output=True, check=False
    )
    if finished.returncode != 0:
        detail = finished.stderr.decode("utf-8", errors="replace").strip()
        raise CannotCheck(f"git {arguments[0]} failed in {root}: {detail}")
    return finished.stdout


def index_entries(root: Path) -> list[tuple[str, str, str]]:
    """Mode, blob and path of every regular file in the index of the tree at root."""
    if shutil.which("git") is None:
        raise CannotCheck("git is not installed, so there is no index to read the modes from")
    found = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, check=False
    )
    if found.returncode != 0:
        raise CannotCheck(
            f"{root} is not in a git work tree, which is what an unpacked source archive looks "
            "like; the modes are recorded in git's index and there is none here to read"
        )
    top = Path(found.stdout.decode("utf-8", errors="surrogateescape").strip()).resolve()
    if top != root.resolve():
        raise CannotCheck(
            f"{root} is inside the git work tree at {top} rather than at its top, so the index "
            "there describes some other tree"
        )

    entries: list[tuple[str, str, str]] = []
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        mode, blob, stage = meta.decode("ascii").split(" ")
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != "0":
            raise CannotCheck(f"{path} is unmerged in the index, so its mode is not settled")
        if mode in (EXECUTABLE, REGULAR):
            entries.append((mode, blob, path))
    if not entries:
        raise CannotCheck(f"the index of {root} lists no regular files")
    return entries


def blobs_with_shebang(root: Path, blobs: set[str]) -> set[str]:
    """The blobs among these whose content begins with #!, read from the object store."""
    ordered = sorted(blobs)
    output = git(root, "cat-file", "--batch", stdin=("\n".join(ordered) + "\n").encode("ascii"))
    found: set[str] = set()
    position = 0
    for blob in ordered:
        end = output.index(b"\n", position)
        header = output[position:end].split(b" ")
        if len(header) != 3 or header[1] != b"blob":
            raise CannotCheck(f"object {blob} is not a readable blob: {header!r}")
        size = int(header[2])
        start = end + 1
        if size >= 2 and output[start:start + 2] == b"#!":
            found.add(blob)
        position = start + size + 1
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else ROOT
    try:
        entries = index_entries(root)
        shebangs = blobs_with_shebang(root, {blob for _, blob, _ in entries})
    except CannotCheck as error:
        print(f"check_executable_bits: cannot check: {error}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for mode, blob, path in entries:
        if blob in shebangs and mode != EXECUTABLE:
            problems.append(f"{path} begins with #! and is committed {mode}; "
                            f"git update-index --chmod=+x {shlex.quote(path)}")
        elif mode == EXECUTABLE and blob not in shebangs:
            problems.append(f"{path} is committed {mode} and does not begin with #!; "
                            f"git update-index --chmod=-x {shlex.quote(path)}")

    if problems:
        for problem in problems:
            print(f"check_executable_bits: {problem}", file=sys.stderr)
        print(f"check_executable_bits: {len(problems)} of {len(entries)} tracked files disagree",
              file=sys.stderr)
        return 1

    scripts = sum(1 for _, blob, _ in entries if blob in shebangs)
    print(f"check_executable_bits: {len(entries)} tracked files agree: {scripts} begin with #! "
          "and are committed 100755, and no other file is")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
