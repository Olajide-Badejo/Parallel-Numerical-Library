#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assert that -ffast-math is refused by the headers, with the intended message.

Ground rule 8 says a flag the numerical contract depends on gets a test that
fails when the flag is missing. The static half of that contract is the
`#if defined(__FAST_MATH__)` guard at the top of `include/pnl/core/types.hpp`,
and the only way to test a guard that stops a compile is to run a compile and
look at what came out.

Two compiles, and the second is why this is a test rather than a gesture. With
`-ffast-math` the compile must fail *and* say why, because an exit status alone
is satisfied by a missing file, a typo in the probe or a compiler that is not
there. Without `-ffast-math` the same file must compile cleanly, which is what
rules those out.

Dependency free and run under ctest, like tests/style/check_linter.py. Takes the
C++ compiler CMake configured with and the source root, so it tests the compiler
the project is actually built with rather than whichever one is first on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MESSAGE = "pnl requires IEEE arithmetic: -ffast-math voids the bit identity guarantee"


def compile_probe(compiler: str, root: Path, fast_math: bool) -> subprocess.CompletedProcess[str]:
    command = [compiler, "-std=c++20"]
    if fast_math:
        command.append("-ffast-math")
    command += ["-fsyntax-only", "-I", str(root / "include")]
    command.append(str(root / "tests" / "unit" / "fast_math_probe.cpp"))
    print("$ " + " ".join(command))
    return subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check_fast_math.py <cxx-compiler> <source-root>")
        return 2
    compiler = argv[1]
    root = Path(argv[2]).resolve()

    rejected = compile_probe(compiler, root, fast_math=True)
    output = rejected.stdout + rejected.stderr
    print(output, end="" if output.endswith("\n") else "\n")

    failures = []
    if rejected.returncode == 0:
        failures.append(
            "the compile with -ffast-math succeeded; the guard in "
            "include/pnl/core/types.hpp is not stopping it"
        )
    if MESSAGE not in output:
        failures.append(f"the compiler output does not contain the intended message: {MESSAGE!r}")

    accepted = compile_probe(compiler, root, fast_math=False)
    if accepted.returncode != 0:
        failures.append(
            "the same file does not compile without -ffast-math, so the failure above "
            "proves nothing about the guard:\n" + accepted.stdout + accepted.stderr
        )

    for failure in failures:
        print(f"check_fast_math: {failure}")
    if failures:
        return 1
    print("check_fast_math: -ffast-math is rejected with the intended message")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
