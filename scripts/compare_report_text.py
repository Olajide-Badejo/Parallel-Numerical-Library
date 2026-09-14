#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a freshly built report against the tracked copy, by its numbers.

This is the gate that would have caught finding 4.3 of the version 2
specification. The bandwidth figures did not reconcile across four artifacts,
one of them the tracked PDF the README links to, and continuous integration
stayed green throughout, because the reports job regenerated the assets, checked
that the output files were not empty, and never compared them against anything
at all.

Byte comparison is the wrong instrument here, for exactly the reason the
workflow already gives for the PNG charts. A PDF carries the fonts, the producer
string and the compression of the machine that built it, so bytes produced on a
runner will never equal bytes produced on a development machine, and a gate on
them would go red on every run for a reason that has nothing to do with the
data. What is worth comparing is what a reader reads, and in a report of
measurements that is the numbers.

So: `pdftotext -layout` on both documents, drop the small list of volatile lines
below, take every numeric token in reading order, and fail on the first
divergence with the page and the line it sits on. The tokens are compared as an
ordered sequence rather than as a set, so a value that moved from one table to
another is a divergence and not a match, and a table that gained or lost a row
is reported as a count difference at the point where the two documents stop
lining up.

Exit status, because a caller has to be able to tell the three apart:

    0   every numeric token agrees
    1   the documents diverge, and the first divergence is printed
    2   the comparison could not be made at all, with the reason

A skipped comparison is never reported as agreement.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Lines dropped before any number is read, and the list is deliberately short.
#
# The rule for adding to it: a line qualifies only if it changes between two
# runs of the same generator over the same data, and only if no reader would
# take the number on it for a measurement. Anything else that differs between a
# rebuild and the tracked copy is the finding this script exists to report, and
# widening the list to make a red run green is the failure mode to guard
# against. Every drop is counted and printed, so a pattern that starts eating
# real content shows up as a count that grew.
VOLATILE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:built|generated) on \d{4}-\d{2}-\d{2}\b", re.IGNORECASE), "a build date"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"), "a manifest timestamp"),
    (re.compile(r"\bmatplotlib[ \t]+\d+\.\d+", re.IGNORECASE), "a matplotlib version string"),
]

# A run of digits, with thousands separators and any number of dotted groups, so
# that 1,046,529 is one token and so are 5.2.1 and 1.1.0. An exponent is part of
# the token it belongs to.
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)*(?:[eE][-+]?\d+)?")

# Characters after which a leading minus or plus belongs to the number rather
# than to the text around it. Without this rule the two plus signs of "C++20"
# would turn 20 into +20, and with a rule that simply drops signs a table cell
# that changed from 0.4 to -0.4 would compare equal.
SIGN_LEADERS = " \t([{=<>,:;/"


class ComparisonError(Exception):
    """The comparison could not be made. Exit status 2, never 0."""


@dataclass(frozen=True)
class Token:
    """One numeric token, and enough of its surroundings to find it again."""

    text: str
    page: int
    line: int
    context: str

    def where(self) -> str:
        return f"page {self.page} line {self.line}: {self.context}"


def numbers_in(line: str) -> list[str]:
    """Every numeric token on one line, in reading order, signs attached."""
    tokens: list[str] = []
    for match in NUMBER.finditer(line):
        text = match.group(0)
        start = match.start()
        if start > 0 and line[start - 1] in "-+":
            before = line[start - 2] if start >= 2 else " "
            if start == 1 or before in SIGN_LEADERS:
                text = line[start - 1] + text
        tokens.append(text)
    return tokens


def tokens_of(text: str) -> tuple[list[Token], Counter[str]]:
    """Numeric tokens of an extracted document, and what was dropped for each."""
    tokens: list[Token] = []
    dropped: Counter[str] = Counter()
    for page_number, page in enumerate(text.split("\f"), start=1):
        for line_number, line in enumerate(page.splitlines(), start=1):
            reason = next(
                (why for pattern, why in VOLATILE_PATTERNS if pattern.search(line)), None
            )
            if reason is not None:
                dropped[reason] += 1
                continue
            for token in numbers_in(line):
                tokens.append(Token(token, page_number, line_number, line.strip()))
    return tokens, dropped


def compare_texts(fresh_text: str, tracked_text: str) -> list[str]:
    """Compare two extracted documents. An empty list means they agree.

    Both halves of the comparison matter. A token that differs is reported at
    the first one, because after a table has shifted every token after it
    differs too and the hundredth report says nothing the first did not. A
    difference in count with no differing token before it is a document that
    gained or lost numbers at the end, which a token by token walk alone would
    not notice.
    """
    fresh, fresh_dropped = tokens_of(fresh_text)
    tracked, tracked_dropped = tokens_of(tracked_text)
    problems: list[str] = []

    for index, (left, right) in enumerate(zip(fresh, tracked, strict=False), start=1):
        if left.text != right.text:
            problems.append(
                f"numeric token {index} of the rebuilt report reads {left.text} "
                f"where the tracked report reads {right.text}\n"
                f"    rebuilt  {left.where()}\n"
                f"    tracked  {right.where()}"
            )
            return problems

    if len(fresh) != len(tracked):
        common = min(len(fresh), len(tracked))
        longer, side = (fresh, "rebuilt") if len(fresh) > len(tracked) else (tracked, "tracked")
        extra = longer[common]
        problems.append(
            f"the rebuilt report has {len(fresh)} numeric tokens and the tracked report has "
            f"{len(tracked)}; the first {common} agree and the {side} report carries more, "
            f"beginning {extra.text} at {extra.where()}"
        )

    for reason in sorted(set(fresh_dropped) | set(tracked_dropped)):
        if fresh_dropped[reason] != tracked_dropped[reason]:
            problems.append(
                f"{reason} was dropped from {fresh_dropped[reason]} line(s) of the rebuilt "
                f"report and {tracked_dropped[reason]} line(s) of the tracked one, so the "
                f"two documents do not carry the same volatile lines"
            )

    return problems


def extract(pdf: Path) -> str:
    """The layout preserving text of a PDF, through pdftotext."""
    if shutil.which("pdftotext") is None:
        raise ComparisonError(
            "pdftotext is not installed, so nothing can be compared. It comes from "
            "poppler-utils, which the reports job installs."
        )
    if not pdf.is_file():
        raise ComparisonError(f"{pdf} does not exist")
    finished = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        check=False,
    )
    if finished.returncode != 0:
        detail = finished.stderr.decode("utf-8", errors="replace").strip()
        raise ComparisonError(f"pdftotext failed on {pdf}: {detail}")
    return finished.stdout.decode("utf-8", errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a freshly built report against the tracked copy, by its numbers."
    )
    parser.add_argument("fresh", type=Path, help="the PDF just built, for example report/main.pdf")
    parser.add_argument(
        "tracked", type=Path, help="the tracked copy, for example assets/reports/main_report.pdf"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fresh_text = extract(args.fresh)
        tracked_text = extract(args.tracked)
    except ComparisonError as error:
        print(f"compare_report_text: {error}", file=sys.stderr)
        return 2

    problems = compare_texts(fresh_text, tracked_text)
    if problems:
        print(
            f"compare_report_text: {args.fresh} does not agree with {args.tracked}",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    tokens, dropped = tokens_of(fresh_text)
    pages = fresh_text.count("\f") + 1
    summary = ", ".join(
        f"{count} line(s) holding {reason}" for reason, count in sorted(dropped.items())
    )
    print(
        f"compare_report_text: {len(tokens)} numeric tokens across {pages} pages agree between "
        f"{args.fresh} and {args.tracked}"
    )
    print(f"  volatile lines dropped: {summary or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
