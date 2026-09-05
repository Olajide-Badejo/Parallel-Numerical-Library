#!/usr/bin/env python3
"""Self test for scripts/migrate_summary.py.

A migration is trusted with the only copy of every measurement in the
repository, and it runs once, unattended, at the start of a sweep. So the four
properties it claims are tested rather than asserted in a docstring: the new
columns arrive in the binary's order, the values that were already there come
back byte for byte, the declared defaults are what lands, and a second run is a
no operation. The fifth case is the one that protects data: a column the binary
no longer emits must stop the migration rather than be quietly dropped.

Dependency free and run under ctest, like tests/style/check_linter.py.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "migrate_summary.py"

# The 1.0.0 header, and the eight columns release 1.1.0 appends to it. Written
# out here rather than read from the binary so that the test needs no build.
OLD_HEADER = (
    "problem,unknowns,solver,backend,workers,ranks,threads_per_rank,pinning,reduction,"
    "schedule,mode,iterations,converged,stop_reason,relative_residual,omega,blocks,"
    "check_interval,seconds_median,seconds_min,seconds_max,reps,updates_per_second,"
    "gib_per_second,bytes_per_unknown,seed,commit,label"
)
NEW_COLUMNS = [
    "sweeps",
    "passes",
    "dram_bytes_per_unknown_per_sweep",
    "pinning_status",
    "measured_at",
    "seconds_reps",
    "kernels",
    "kernel_variant",
]
NEW_HEADER = OLD_HEADER + "," + ",".join(NEW_COLUMNS)

# Three rows in the shape the 1.0.0 binary wrote, one of them on the device.
# CRLF line endings, because that is what csv.DictWriter produced and therefore
# what the committed summary carries; preserving them is part of the contract.
ROWS = [
    "poisson2d_rich_1023,1046529,jacobi,openmp,20,1,1,none,deterministic,static,fixed,100,0,"
    "iteration_cap,5.353741e-02,1.993883,1023,1000000,0.062997,0.058905,0.070180,5,"
    "1.661243e+09,37.1291,24.0,20260802,dce9cf4b25f0,scaling",
    "poisson2d_rich_1023,1046529,gauss_seidel_rb,cuda,1,1,1,none,device,static,fixed,300,0,"
    "iteration_cap,2.224451e-02,1.993883,1023,1000000,0.019289,0.018628,0.019870,5,"
    "2.167327e+10,484.4354,24.0,20260802,4abf914a7ea2.dirty,"
    "device_comparison kernel=0.014486 transfer=0.002055",
    "dense_spd_512,512,cg,serial,1,1,1,none,deterministic,static,fixed,50,0,"
    "iteration_cap,1.104972e-03,1.000000,16,1000000,0.041215,0.040988,0.041902,5,"
    "6.211077e+08,9.4874,4096.0,20260802,dce9cf4b25f0,dense",
]
# Seven commas: the one that closes `label`, then the six columns that stay
# empty because the 1.0.0 generation never recorded them.
EXPECTED = [
    ",,,,,,,cxx,cpp",
    ",,,,,,,device,device",
    ",,,,,,,cxx,cpp",
]


def write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(
        header + "\r\n" + "".join(row + "\r\n" for row in rows), encoding="utf-8", newline=""
    )


def migrate(path: Path, header: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--header", header],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as scratch:
        summary = Path(scratch) / "summary.csv"
        write(summary, OLD_HEADER, ROWS)
        before = summary.read_text(encoding="utf-8", newline="")

        code, output = migrate(summary, NEW_HEADER)
        if code != 0:
            failures.append(f"the migration failed on a 1.0.0 summary:\n{output}")
        migrated = summary.read_text(encoding="utf-8", newline="")
        lines = migrated.splitlines()

        # The new columns exist, in the binary's order, and no others appeared.
        if lines and lines[0] != NEW_HEADER:
            failures.append(f"the migrated header is not the binary's header:\n  {lines[0]}")

        # Every stored value survives byte for byte, and only fields are added.
        for index, (old, new) in enumerate(zip(before.splitlines(), lines, strict=False)):
            if not new.startswith(old + ","):
                failures.append(f"line {index + 1} was rewritten rather than extended:\n"
                                f"  was {old}\n  now {new}")

        # The declared defaults land, and the device row gets the device ones.
        for index, (row, expected) in enumerate(zip(lines[1:], EXPECTED, strict=True)):
            if not row.endswith(expected):
                failures.append(f"row {index + 1} should end with {expected!r}, got "
                                f"{row[-len(expected) - 20:]!r}")

        # The line endings are the ones the file already had.
        if "\r\n" not in migrated or migrated.count("\r\n") != len(ROWS) + 1:
            failures.append("the migration did not preserve the CRLF line endings")

        # A second run changes nothing at all.
        code, output = migrate(summary, NEW_HEADER)
        if code != 0:
            failures.append(f"the second migration should have succeeded:\n{output}")
        if summary.read_text(encoding="utf-8", newline="") != migrated:
            failures.append("the second migration changed the file; it is not idempotent")
        if "nothing to do" not in output:
            failures.append(f"the second migration should have said so, printed:\n{output}")

        # A column the binary no longer emits is refused, by name.
        removed = Path(scratch) / "removed.csv"
        write(removed, OLD_HEADER + ",retired_column", [row + ",7" for row in ROWS])
        code, output = migrate(removed, NEW_HEADER)
        if code != 2:
            failures.append(f"a removed column should exit 2, exited {code}:\n{output}")
        if "retired_column" not in output:
            failures.append(f"the refusal should name the removed column, printed:\n{output}")
        if removed.read_text(encoding="utf-8", newline="") != (
            OLD_HEADER + ",retired_column\r\n" + "".join(row + ",7\r\n" for row in ROWS)
        ):
            failures.append("the refused file was modified anyway")

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print(f"{len(failures)} failed")
        return 1

    print("  pass  columns appended in order, values preserved, defaults declared")
    print("  pass  a second run is a no operation")
    print("  pass  a removed column is refused by name")
    print("3 passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
