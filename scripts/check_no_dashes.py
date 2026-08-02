#!/usr/bin/env python3
"""Repository wide guard against em dashes and en dashes.

Ground rule 1 of the build specification forbids U+2014 and U+2013 in every file
type, and additionally forbids the LaTeX ligatures ``--`` and ``---`` in .tex
prose because TeX typesets those as the very characters the rule bans. This
script enforces both, including inside compiled PDFs, and is wired into
``make check-style`` and the report build.

Exit status is 0 when the tree is clean and 1 when anything is found, so it can
be used directly as a CI gate.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The two characters the ground rule names explicitly.
BANNED_PRIMARY = {
    "\u2014": "EM DASH",
    "\u2013": "EN DASH",
}

# Visually identical stand ins used as punctuation. Banning only the two above
# would leave an obvious loophole, so these are treated the same way.
#
# U+2212 MINUS SIGN is deliberately NOT in this list. It was, briefly, and it
# made the compiled report unbuildable: every equation in a numerical methods
# document contains subtraction, and LaTeX typesets mathematical minus as
# U+2212. That is a mathematical operator, not a dash used as punctuation, and
# the ground rule is about typography rather than arithmetic. Banning it would
# forbid writing A = M minus N in mathematics, which is absurd.
BANNED_LOOKALIKE = {
    "\u2012": "FIGURE DASH",
    "\u2015": "HORIZONTAL BAR",
    "\u2E3A": "TWO EM DASH",
    "\u2E3B": "THREE EM DASH",
    "\uFE58": "SMALL EM DASH",
    "\uFE63": "SMALL HYPHEN MINUS",
    "\uFF0D": "FULLWIDTH HYPHEN MINUS",
}

BANNED = {**BANNED_PRIMARY, **BANNED_LOOKALIKE}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "_build",
    ".cache",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    "third_party",
}

# Binary or generated formats with no prose to police. PDFs are handled
# separately because the rule explicitly covers them.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".zip", ".gz", ".xz", ".tar", ".whl",
    ".ttf", ".otf", ".woff", ".woff2",
    ".aux", ".fdb_latexmk", ".fls", ".synctex", ".out", ".toc", ".lof", ".lot",
    ".bbl", ".blg", ".nav", ".snm", ".vrb",
}

TEXT_SUFFIX_HINT = {
    ".md", ".txt", ".tex", ".bib", ".py", ".sh", ".yaml", ".yml", ".json",
    ".cpp", ".hpp", ".h", ".cu", ".cuh", ".cmake", ".toml", ".cfg", ".ini",
    ".csv", ".clang-format", ".gitignore", ".mk",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    kind: str
    detail: str
    excerpt: str

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return (
            f"{shown}:{self.line}:{self.column}: {self.kind}: {self.detail}\n"
            f"    {self.excerpt}"
        )


def is_probably_text(path: Path) -> bool:
    """Cheap binary sniff so the walker does not choke on stray artefacts."""
    if path.suffix.lower() in TEXT_SUFFIX_HINT or path.name in TEXT_SUFFIX_HINT:
        return True
    try:
        chunk = path.open("rb").read(4096)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


# The linter's own test fixtures contain planted violations on purpose, so a
# recursive scan of the tree must not walk into them. They are still checked,
# but by tests/style/check_linter.py, which names them explicitly and asserts
# exactly which lines are reported.
FIXTURE_DIR = Path("tests") / "style" / "fixtures"


def is_fixture(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return FIXTURE_DIR in relative.parents


def iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    scan_root = root
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if is_fixture(path, scan_root):
            continue
        out.append(path)
    return out


def caret_excerpt(line_text: str, column: int) -> str:
    """Render the offending line with a caret, clipped around long lines."""
    stripped = line_text.rstrip("\n")
    start = max(0, column - 40)
    end = min(len(stripped), column + 40)
    window = stripped[start:end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(stripped) else ""
    caret_pos = len(prefix) + (column - 1 - start)
    return f"{prefix}{window}{suffix}\n    {' ' * max(caret_pos, 0)}^"


def scan_characters(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            name = BANNED.get(char)
            if name is None:
                continue
            findings.append(
                Finding(
                    path=path,
                    line=lineno,
                    column=column,
                    kind="banned character",
                    detail=f"U+{ord(char):04X} {name}",
                    excerpt=caret_excerpt(line, column),
                )
            )
    return findings


# LaTeX regions where a literal ``--`` is legitimate: it is code or a URL, not
# prose, and TeX will not typeset it as a dash.
TEX_VERBATIM_ENVS = ("verbatim", "lstlisting", "minted", "Verbatim", "alltt")
TEX_URL_MACROS = re.compile(r"\\(?:url|href|path|nolinkurl)\s*\{[^}]*\}")
TEX_VERB_INLINE = re.compile(r"\\(?:verb|lstinline)(\*?)(.)(.*?)\2")


def mask_tex_noncode(line: str) -> str:
    """Blank out the parts of a .tex line where ``--`` carries no typographic
    meaning, so only genuine prose is checked."""
    line = TEX_URL_MACROS.sub(lambda m: " " * len(m.group(0)), line)
    line = TEX_VERB_INLINE.sub(lambda m: " " * len(m.group(0)), line)
    # Strip a trailing comment, honouring the \% escape.
    out: list[str] = []
    idx = 0
    while idx < len(line):
        ch = line[idx]
        if ch == "\\" and idx + 1 < len(line):
            out.append(line[idx : idx + 2])
            idx += 2
            continue
        if ch == "%":
            break
        out.append(ch)
        idx += 1
    return "".join(out)


def scan_tex_ligatures(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    in_verbatim = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        lowered = raw.strip()
        if any(lowered.startswith(rf"\begin{{{env}}}") for env in TEX_VERBATIM_ENVS):
            in_verbatim = True
            continue
        if any(lowered.startswith(rf"\end{{{env}}}") for env in TEX_VERBATIM_ENVS):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        prose = mask_tex_noncode(raw)
        for match in re.finditer(r"-{2,}", prose):
            column = match.start() + 1
            run = len(match.group(0))
            typeset = "en dash" if run == 2 else "em dash"
            findings.append(
                Finding(
                    path=path,
                    line=lineno,
                    column=column,
                    kind="latex ligature",
                    detail=f"{'-' * run} typesets as an {typeset}; write the words out",
                    excerpt=caret_excerpt(raw, column),
                )
            )
    return findings


def scan_pdf(path: Path) -> list[Finding]:
    """Check compiled PDFs, which the definition of done covers explicitly."""
    tool = shutil.which("pdftotext")
    if tool is None:
        print(
            f"note: skipping {path.name}, pdftotext not installed "
            "(install poppler-utils to cover PDFs)",
            file=sys.stderr,
        )
        return []
    try:
        proc = subprocess.run(
            [tool, "-q", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"note: could not read {path.name}: {exc}", file=sys.stderr)
        return []
    text = proc.stdout.decode("utf-8", errors="replace")
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            name = BANNED.get(char)
            if name is None:
                continue
            findings.append(
                Finding(
                    path=path,
                    line=lineno,
                    column=column,
                    kind="banned character in compiled PDF",
                    detail=f"U+{ord(char):04X} {name}",
                    excerpt=caret_excerpt(line, column),
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to scan (default: repository root)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="skip compiled PDFs even when pdftotext is available",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the summary line",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    targets = args.paths or [root]

    candidates: list[Path] = []
    for requested in targets:
        target = requested.resolve()
        if target.is_dir():
            candidates.extend(iter_files(target))
        elif target.is_file():
            candidates.append(target)

    findings: list[Finding] = []
    scanned = 0
    for path in candidates:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if not args.no_pdf:
                scanned += 1
                findings.extend(scan_pdf(path))
            continue
        if suffix in SKIP_SUFFIXES:
            continue
        if not is_probably_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        findings.extend(scan_characters(path, text))
        if suffix == ".tex":
            findings.extend(scan_tex_ligatures(path, text))

    if findings and not args.quiet:
        for finding in findings:
            print(finding.render(root))
        print()

    if findings:
        print(f"check_no_dashes: {len(findings)} violation(s) across {scanned} file(s)")
        return 1

    print(f"check_no_dashes: clean, {scanned} file(s) scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
