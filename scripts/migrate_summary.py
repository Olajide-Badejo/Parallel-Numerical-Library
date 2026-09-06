#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add to a stored summary the columns the binary has gained, with declared defaults.

`run_sweep.py` compares the header of the summary it is merging into against the
header the binary emits, with strict list equality, and exits 2 on any
difference. That is the right default, because mixing two schemas in one file
silently shifts every field, but it means that every column added to the schema
breaks `make sweep` against the committed summary until a full re measurement
finishes. This script is the other half of that rule: it makes the stored file
match the current binary, by adding what is missing and nothing else.

What it will and will not do.

  Add.      A column the binary emits and the file lacks is appended to every
            row with its default from the table below.

  Refuse.   A column the file has and the binary no longer emits is a removal,
            and a removal loses data. There is no safe guess, so the script
            names the column and exits 2.

  Refuse.   A header that is not a prefix of the binary's header means a column
            was inserted or reordered rather than appended. Appending fields
            cannot express that, so the script says so and exits 2.

  Preserve. Existing rows are extended by appending text to the line, so every
            byte a row already carried, including its line ending and any
            quoting, survives unchanged. Only the new fields are written.

  Repeat.   Running it on an already migrated file writes nothing and says so.

The defaults for the columns release 1.1.0 adds. The 1.0.0 generation recorded
no sweep count, no second byte model, no pinning outcome, no timestamp and no
repetition list, so `sweeps`, `passes`, `dram_bytes_per_unknown_per_sweep`,
`pinning_status`, `measured_at` and `seconds_reps` are empty, which pandas reads
as NaN. A migration must not invent a measurement, and a plausible looking
number in one of those fields would be exactly that. `kernels` and
`kernel_variant` are not measurements but statements about which code ran, and
for 1.0.0 the answer is known exactly: the C++ kernel table in its C++
implementation, `cxx` and `cpp`, because no others existed. A row whose
`backend` is `cuda` ran neither, and carries `device` for both, which is what
its `reduction` column already does.

Usage:

    python3 scripts/migrate_summary.py experiments/results/summary.csv
    python3 scripts/migrate_summary.py SUMMARY --header-from build/pnl
    python3 scripts/migrate_summary.py SUMMARY --header "problem,unknowns,..."
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The declared default of every column this generation adds. See the docstring
# for why six of them are empty and two are not.
DEFAULTS = {
    "sweeps": "",
    "passes": "",
    "dram_bytes_per_unknown_per_sweep": "",
    "pinning_status": "",
    "measured_at": "",
    "seconds_reps": "",
    "kernels": "cxx",
    "kernel_variant": "cpp",
}

# What those two non empty defaults become on a device row.
DEVICE_BACKEND = "cuda"
DEVICE_DEFAULTS = {"kernels": "device", "kernel_variant": "device"}


def binary_header(binary: Path) -> list[str] | None:
    """The header the binary emits, or None with a message on stderr."""
    try:
        proc = subprocess.run(
            [str(binary), "--header"], capture_output=True, text=True, timeout=120, check=False
        )
    except OSError as error:
        print(f"migrate_summary: cannot run {binary}: {error}", file=sys.stderr)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"migrate_summary: {binary} --header printed nothing: {proc.stderr.strip()}",
              file=sys.stderr)
        return None
    return proc.stdout.strip().split(",")


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file in the same directory and rename it.

    The same four lines as `write_atomic` in benchmarks/run_sweep.py, copied
    rather than imported: that one takes parsed rows and writes them back
    through a csv.DictWriter, while this script appends text to the original
    lines so that stored values survive byte for byte. The property that
    matters is shared, and it is the reason both do it this way: a reader can
    never observe a half written summary at the real path.
    """
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        temporary = handle.name
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def split_terminator(line: str) -> tuple[str, str]:
    """The line without its ending, and the ending, whether CRLF, LF or absent."""
    text = line.rstrip("\r\n")
    return text, line[len(text):]


def extend(line: str, values: list[str]) -> str:
    """Append fields to a line, leaving every byte before them alone."""
    text, terminator = split_terminator(line)
    return text + "," + ",".join(values) + terminator


def migrate(summary: Path, header: list[str]) -> int:
    raw = summary.read_text(encoding="utf-8", newline="")
    lines = raw.splitlines(keepends=True)
    if not lines:
        print(f"migrate_summary: {summary} is empty", file=sys.stderr)
        return 2

    existing = next(csv.reader([lines[0]]))
    removed = [name for name in existing if name not in header]
    if removed:
        print(f"migrate_summary: {summary} has column(s) the binary no longer emits: "
              f"{', '.join(removed)}. A removal loses data and there is no safe default "
              "for it, so this refuses rather than guessing.", file=sys.stderr)
        return 2
    if header[:len(existing)] != existing:
        print(f"migrate_summary: {summary} does not begin with the binary's columns in the "
              "binary's order, so a column was inserted or reordered rather than appended. "
              "Appending fields cannot express that.", file=sys.stderr)
        return 2

    missing = header[len(existing):]
    if not missing:
        print(f"migrate_summary: {summary} already has every column the binary emits, "
              f"{len(existing)} of them; nothing to do")
        return 0
    undeclared = [name for name in missing if name not in DEFAULTS]
    if undeclared:
        print(f"migrate_summary: no declared default for {', '.join(undeclared)}. Add one to "
              "DEFAULTS, with the reason, rather than letting the column be filled by "
              "accident.", file=sys.stderr)
        return 2

    body = lines[1:]
    rows = list(csv.reader(body))
    if len(rows) != len(body):
        print(f"migrate_summary: {summary} has a field containing a newline, which this "
              "script's line by line append cannot preserve.", file=sys.stderr)
        return 2
    backend_column = existing.index("backend") if "backend" in existing else -1

    output = [extend(lines[0], missing)]
    devices = 0
    for line, row in zip(body, rows, strict=True):
        if not row:
            output.append(line)
            continue
        device = 0 <= backend_column < len(row) and row[backend_column] == DEVICE_BACKEND
        devices += 1 if device else 0
        table = DEFAULTS | DEVICE_DEFAULTS if device else DEFAULTS
        output.append(extend(line, [table[name] for name in missing]))

    write_atomic(summary, "".join(output))
    print(f"migrate_summary: {summary}: added {', '.join(missing)} "
          f"to {len(rows)} row(s), {devices} of them on the device")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summary", type=Path, help="the summary CSV to migrate in place")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--header-from", type=Path, default=ROOT / "build" / "pnl",
                        metavar="BINARY",
                        help="take the target header from this binary's --header output")
    source.add_argument("--header", metavar="LINE",
                        help="the target header, as a literal comma separated line")
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"migrate_summary: {args.summary} not found", file=sys.stderr)
        return 2

    if args.header:
        header = args.header.strip().split(",")
    else:
        found = binary_header(args.header_from)
        if found is None:
            return 2
        header = found

    return migrate(args.summary, header)


if __name__ == "__main__":
    sys.exit(main())
