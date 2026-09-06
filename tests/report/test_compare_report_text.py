#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self test for scripts/compare_report_text.py.

The comparison is the gate that would have caught finding 4.3, so the thing it
must never do is report agreement over a difference. It is also the gate most
easily made useless by accident, because widening the volatile allowlist turns a
red run green and nothing else in the repository would notice.

Both directions are checked here, on synthetic extracts rather than on PDFs, so
this test needs no poppler and no report build and runs in every job:

  - two identical extracts agree;
  - an extract with one number changed does not, and the message names both the
    old value and the new one;
  - an extract whose volatile lines changed and whose measurements did not still
    agrees, which is the allowlist doing its job;
  - a sign that changed is a difference, because a table cell that went from 0.4
    to -0.4 has changed by more than a table cell that went from 0.4 to 0.5;
  - an extract that gained a number at the end is a difference, which a token by
    token walk alone would not report.

Dependency free and run under ctest, like tests/style/check_linter.py and
tests/report/test_gen_report_assets.py: no test framework, no fixture files, no
network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "compare_report_text.py"

# Two pages, separated the way pdftotext separates them. The first page carries
# a build date and a table of measurements, the second a matplotlib version
# string, a thousands separated integer and a signed difference, which is one of
# each thing the tokeniser has a rule for.
PAGE_ONE = "\n".join([
    "Parallel Numerical Library, main report",
    "generated on 2026-08-01 by the asset generator",
    "",
    "Table 5.1: host bandwidth against worker count",
    "   workers        GiB/s       spread",
    "         1         12.5         0.30",
    "        28         52.5         1.10",
    "",
    "The 28 worker figure is -0.4 GiB/s away from the 24 worker one.",
])

PAGE_TWO = "\n".join([
    "matplotlib 3.9.2 drew every chart in this document",
    "Total unknowns 1,046,529 over 425 rows of summary.csv.",
])

EXTRACT = PAGE_ONE + "\f" + PAGE_TWO


def load_script() -> Any:
    """Import the comparison script as a module, which its __main__ guard allows."""
    spec = importlib.util.spec_from_file_location("compare_report_text", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed, because the script defines a frozen
    # dataclass and dataclasses resolves annotations through sys.modules. A
    # module that is not registered there fails at the decorator with an
    # AttributeError on None, which says nothing about what went wrong.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_identical(module: Any, failures: list[str]) -> None:
    problems = module.compare_texts(EXTRACT, EXTRACT)
    if problems:
        failures.append(f"two identical extracts were reported as differing: {problems}")


def check_one_number_changed(module: Any, failures: list[str]) -> None:
    changed = EXTRACT.replace("52.5", "55.1")
    problems = module.compare_texts(changed, EXTRACT)
    if not problems:
        failures.append("a table figure that changed from 52.5 to 55.1 was reported as agreement")
        return
    if len(problems) != 1:
        failures.append(f"one changed number should report once, reported {len(problems)} times")
    message = problems[0]
    for expected in ("55.1", "52.5", "page 1"):
        if expected not in message:
            failures.append(f"the message for a changed number does not name {expected}: {message}")


def check_volatile_line_changed(module: Any, failures: list[str]) -> None:
    changed = EXTRACT.replace("2026-08-01", "2026-09-06")
    changed = changed.replace("matplotlib 3.9.2", "matplotlib 3.10.0")
    problems = module.compare_texts(changed, EXTRACT)
    if problems:
        failures.append(
            f"a build date and a matplotlib version that changed were reported as a "
            f"difference in the measurements: {problems}"
        )
    tokens, dropped = module.tokens_of(EXTRACT)
    if sum(dropped.values()) != 2:
        failures.append(f"expected two volatile lines dropped, dropped {dict(dropped)}")
    if any(token.text.startswith("2026") for token in tokens):
        failures.append("a date survived into the compared tokens")


def check_sign_changed(module: Any, failures: list[str]) -> None:
    changed = EXTRACT.replace("is -0.4 GiB/s", "is 0.4 GiB/s")
    problems = module.compare_texts(changed, EXTRACT)
    if not problems:
        failures.append("a difference that changed sign from -0.4 to 0.4 was reported as agreement")
    elif "-0.4" not in problems[0]:
        failures.append(f"the message for a changed sign does not name -0.4: {problems[0]}")


def check_token_added(module: Any, failures: list[str]) -> None:
    problems = module.compare_texts(EXTRACT + "\nOne more line carrying 99.", EXTRACT)
    if not problems:
        failures.append("an extract that gained a number at the end was reported as agreement")
    elif "99" not in problems[0]:
        failures.append(f"the message for an added number does not name 99: {problems[0]}")


def main() -> int:
    if not SCRIPT.exists():
        print(f"test_compare_report_text: {SCRIPT} is missing", file=sys.stderr)
        return 1
    module = load_script()

    failures: list[str] = []
    check_identical(module, failures)
    check_one_number_changed(module, failures)
    check_volatile_line_changed(module, failures)
    check_sign_changed(module, failures)
    check_token_added(module, failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("test_compare_report_text: identical extracts agree, a changed number, a changed sign "
          "and an added number are all reported, and the volatile allowlist drops dates and "
          "versions without hiding a measurement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
