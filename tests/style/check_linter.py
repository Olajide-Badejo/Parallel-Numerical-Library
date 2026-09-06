#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self test for scripts/check_no_dashes.py.

A linter that silently stopped detecting anything would let the ground rule rot
while every gate stayed green, so the linter itself is tested: it must find the
violations planted in the fixtures and must not flag the legitimate uses beside
them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = ROOT / "scripts" / "check_no_dashes.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The extracted text of a compiled report, in miniature. Everything before the
# bibliography heading is prose and gets no allowance; inside the bibliography a
# page range is allowed and a dash between words is not. Written here rather
# than compiled, because the carve out is a property of the scanner and a test
# that had to build a PDF to reach it would be skipped wherever LaTeX is absent.
#
# The en dashes below are built with chr on purpose. This file is not a
# fixture, so the recursive scan walks into it, and a literal U+2013 here would
# be a violation of the rule the file exists to test.
EN = chr(0x2013)
PDF_TEXT_LINES = [
    "Chapter 5",                                          # 1
    "The host reached 61.4 GiB/s at 4 workers.",          # 2
    f"A prose en dash {EN} right here.",                  # 3, reported
    f"A prose range 19{EN}26 before the bibliography.",   # 4, reported
    "Bibliography",                                       # 5, the heading
    f"[7] Dormand, J. R. A family of formulae. 6(1):19{EN}26, 1980.",  # 6, allowed
    f"[8] Example, A. A title with an en dash {EN} in it. 1999.",      # 7, reported
]


def run(paths: list[Path]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECKER), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def load_checker() -> Any:
    """Import the checker as a module, which its __main__ guard allows."""
    spec = importlib.util.spec_from_file_location("check_no_dashes", CHECKER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {CHECKER}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed, because the checker defines a frozen
    # dataclass under `from __future__ import annotations` and the dataclass
    # machinery resolves that class's annotations through sys.modules. Without
    # this the import fails inside dataclasses with an AttributeError on None.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_bib_carve_out(failures: list[str]) -> None:
    """Ground rule 1 as amended: a page range may carry a dash, prose may not.

    The exemption is one field wide. If it ever widens to the whole file the
    title on line 17 of the fixture stops being reported and this fails, which
    is the only way anyone would notice.
    """
    code, output = run([FIXTURES / "pages.bib"])
    if code == 0:
        failures.append("the bib fixture should have failed the check: the double hyphen "
                        "in a title is prose and is not a page range")
    reported = {
        int(line.split(":")[1])
        for line in output.splitlines()
        if "pages.bib:" in line and "bibtex ligature" in line
    }
    if reported != {17}:
        failures.append(
            f"expected the bibtex ligature on line 17 and nothing else, got "
            f"{sorted(reported)}; the two pages fields on lines 11 and 19 are the "
            "carve out and the comment on line 5 is a comment")


def check_pdf_carve_out(failures: list[str]) -> None:
    """The same carve out in the compiled PDF, anchored on the bibliography.

    An en dash is allowed only between two digits and only after the
    bibliography heading. Everything else the rule bans is still banned, which
    is the property that keeps this a carve out rather than a hole.
    """
    module = load_checker()
    findings = module.scan_pdf_text(Path("synthetic.pdf"), "\n".join(PDF_TEXT_LINES))
    reported = sorted(finding.line for finding in findings)
    if reported != [3, 4, 7]:
        failures.append(
            f"expected the compiled PDF scan to report lines 3, 4 and 7 and to allow "
            f"the page range on line 6, got {reported}. Line 4 is a range before the "
            "bibliography heading and line 7 is a dash between words after it; neither "
            "is a page range and both stay banned.")


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

    check_bib_carve_out(failures)
    check_pdf_carve_out(failures)

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print(f"{len(failures)} failed")
        return 1

    print("  pass  linter detects planted violations and ignores legitimate ones")
    print("  pass  the page range carve out is one bib field wide and one PDF region "
          "wide")
    print("2 passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
