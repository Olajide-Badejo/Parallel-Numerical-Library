#!/usr/bin/env python3
"""Self test for scripts/check_no_dashes.py.

A linter that silently stopped detecting anything would let the ground rule rot
while every gate stayed green, so the linter itself is tested: it must find the
violations planted in the fixtures and must not flag the legitimate uses beside
them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = ROOT / "scripts" / "check_no_dashes.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run(paths: list[Path]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECKER), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    failures: list[str] = []

    # The LaTeX fixture: exactly two prose violations, on lines 4 and 14.
    code, output = run([FIXTURES / "ligatures.tex"])
    if code == 0:
        failures.append("the ligature fixture should have failed the check")
    reported = {
        int(line.split(":")[1])
        for line in output.splitlines()
        if "ligatures.tex:" in line and "latex ligature" in line
    }
    if reported != {4, 14}:
        failures.append(
            f"expected ligature violations on lines 4 and 14, got {sorted(reported)}; "
            "a comment, a URL, an inline verb span, a verbatim block and an lstlisting "
            "block must all be ignored"
        )

    # The character fixture: one em dash and one en dash.
    code, output = run([FIXTURES / "characters.md"])
    if code == 0:
        failures.append("the character fixture should have failed the check")
    if "U+2014" not in output:
        failures.append("the em dash in characters.md was not detected")
    if "U+2013" not in output:
        failures.append("the en dash in characters.md was not detected")

    # The clean fixture must pass.
    code, output = run([FIXTURES / "clean.tex"])
    if code != 0:
        failures.append(f"the clean fixture was wrongly flagged:\n{output}")

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print(f"{len(failures)} failed")
        return 1

    print("  pass  linter detects planted violations and ignores legitimate ones")
    print("1 passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
