#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assert that a bad command line is refused, and refused rather than survived.

Two rows of Section 4.7 live in `parse()`. It ran outside the `try` in `main`,
so `pnl --size abc` ended in `std::terminate` on an unhandled `std::invalid_
argument` whose message was the single word "stoll"; and `--reps 0` was accepted
and then indexed an empty vector of timings three times to build a row whose
durations nobody measured.

What is asserted here is what a user sees: a non zero exit status that is a
status and not a signal, and a message that names the flag that was wrong. The
signal check is the one that matters most, because a process killed by SIGABRT
also "fails", and telling the two apart is the whole point of the change.

Dependency free and run under ctest, like the other Python gates here. Takes the
path of the driver, so it tests the binary this build produced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TIMEOUT_SECONDS = 120


def run(binary: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = [str(binary), *arguments]
    print("$ " + " ".join(command))
    return subprocess.run(
        command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
    )


def check_refused(binary: Path, arguments: list[str], flag: str) -> list[str]:
    """Check one bad command line. Returns the failures it found."""
    result = run(binary, *arguments)
    output = result.stdout + result.stderr
    print(output, end="" if output.endswith("\n") else "\n")
    print(f"  exit {result.returncode}")

    failures = []
    if result.returncode == 0:
        failures.append(f"{' '.join(arguments)} was accepted, and it should not be")
    if result.returncode < 0:
        failures.append(
            f"{' '.join(arguments)} killed the process with signal {-result.returncode} "
            "rather than exiting; an unhandled exception is not a diagnostic"
        )
    if "terminate called" in output:
        failures.append(f"{' '.join(arguments)} reached std::terminate: {output.strip()!r}")
    if flag not in output:
        failures.append(f"the message for {' '.join(arguments)} does not name {flag}: {output!r}")
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_cli_errors.py <pnl-binary>")
        return 2
    binary = Path(argv[1]).resolve()

    failures = []
    # A value that is not a number at all.
    failures += check_refused(binary, ["--size", "abc"], "--size")
    # A number with something after it, which is a typo rather than a number.
    failures += check_refused(binary, ["--iterations", "12abc"], "--iterations")
    # A number that is one, and is still not a legal repetition count.
    failures += check_refused(binary, ["--reps", "0"], "--reps")

    # And the binary works, so that the three refusals above are refusals and
    # not a driver that cannot start.
    working = run(binary, "--version")
    if working.returncode != 0:
        failures.append(
            "the driver does not run at all, so the refusals above prove nothing:\n"
            + working.stdout
            + working.stderr
        )

    for failure in failures:
        print(f"check_cli_errors: {failure}")
    if failures:
        return 1
    print("check_cli_errors: every bad command line was refused with a message naming the flag")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
