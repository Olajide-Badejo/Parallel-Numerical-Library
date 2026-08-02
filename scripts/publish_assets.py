#!/usr/bin/env python3
"""Copy the compiled reports into assets/reports/ for the repository landing page.

LaTeX builds its PDFs next to their sources, where they are build output and are
not tracked. The landing page needs stable, tracked paths it can link to and
that a release can attach, so the finished PDFs are copied here under names that
say what they are.

Idempotent: it overwrites whatever is there and reports what it did.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "assets" / "reports"

# Source, published name, and whether its absence is a failure.
REPORTS = [
    (ROOT / "report" / "main.pdf", "main_report.pdf", True),
    (ROOT / "report_debug" / "debug_report.pdf", "debug_report.pdf", True),
]


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    for source, published, required in REPORTS:
        if not source.exists():
            if required:
                missing.append(str(source.relative_to(ROOT)))
            continue
        destination = TARGET / published
        shutil.copy2(source, destination)
        size = destination.stat().st_size / 1024
        print(f"  report  {destination.relative_to(ROOT)} ({size:.0f} KiB)")

    if missing:
        print("publish_assets: missing " + ", ".join(missing) +
              "; run make reports first", file=sys.stderr)
        return 1

    print("publish_assets: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
