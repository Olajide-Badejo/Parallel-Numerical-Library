#!/usr/bin/env python3
"""Self test for scripts/gen_report_assets.py.

The generator is where ground rule 7 lives. It decides which differences the
report is allowed to print as a number and which print as a phrase, how wide the
knee interval is, and whether an asset may be built from a dirty tree at all.
None of that is exercised by building the report, which succeeds whatever the
generator decides, so it is tested here against a summary with known values.

Dependency free and run under ctest, like tests/style/check_linter.py: no test
framework, no fixture files, no network. It does import the generator, which
needs matplotlib and pandas, and where those are missing it says so and stops
rather than failing a build that was never going to run the reports job.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "gen_report_assets.py"
SUMMARY = ROOT / "experiments" / "results" / "summary.csv"

# The synthetic scaling curve: speedup rises one for one up to this many workers
# and then almost flattens, so the two segment fit has exactly one right answer
# and the bootstrap can be asked whether its interval contains it.
KNEE = 6
WORKER_COUNTS = list(range(1, 13))
FLAT_SLOPE = 0.1

# Fifteen repetitions, symmetric about zero so the median of the whole set is the
# unjittered value and the recorded minimum and maximum are its ends.
JITTER = [round(-0.014 + 0.002 * index, 3) for index in range(15)]


def load_generator() -> Any:
    """Import the generator as a module, which its __main__ guard allows."""
    spec = importlib.util.spec_from_file_location("gen_report_assets", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_header() -> list[str]:
    """The committed schema, read rather than copied.

    A synthetic summary with a hand written header would keep passing after a
    column was added, while testing a file format the generator no longer reads.
    """
    with SUMMARY.open(encoding="utf-8") as handle:
        return next(csv.reader(handle))


def scaling_rows(commit: str, with_repetitions: bool) -> list[dict[str, Any]]:
    """One backend's scaling block, with a knee planted at KNEE workers."""
    rows: list[dict[str, Any]] = []
    for workers in WORKER_COUNTS:
        speedup = workers if workers <= KNEE else KNEE + FLAT_SLOPE * (workers - KNEE)
        seconds = 1.0 / speedup
        repetitions = [seconds * (1.0 + jitter) for jitter in JITTER]
        rows.append({
            "problem": "poisson2d_rich_1023",
            "unknowns": 1046529,
            "solver": "jacobi",
            "backend": "openmp",
            "workers": workers,
            "pinning": "none",
            "reduction": "deterministic",
            "schedule": "static",
            "mode": "fixed",
            "iterations": 100,
            "converged": 0,
            "seconds_median": f"{seconds:.9f}",
            "seconds_min": f"{min(repetitions):.9f}",
            "seconds_max": f"{max(repetitions):.9f}",
            "reps": len(JITTER),
            "gib_per_second": 1.0,
            "commit": commit,
            "label": "scaling",
            "seconds_reps": (";".join(f"{value:.9f}" for value in repetitions)
                             if with_repetitions else ""),
        })
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    header = summary_header()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})


def check_spread(module: Any, failures: list[str]) -> None:
    """One definition of spread, and no guess where a column is missing."""
    value = module.spread({"seconds_median": 2.0, "seconds_min": 1.0, "seconds_max": 3.0})
    if value is None or abs(value - 1.0) > 1e-12:
        failures.append(f"spread of (1, 2, 3) should be 1.0, got {value}")
    value = module.spread({"seconds_median": 0.5, "seconds_min": 0.45, "seconds_max": 0.55})
    if value is None or abs(value - 0.2) > 1e-12:
        failures.append(f"spread of (0.45, 0.5, 0.55) should be 0.2, got {value}")
    for row in ({"seconds_median": 2.0, "seconds_min": 1.0, "seconds_max": ""},
                {"seconds_median": 0.0, "seconds_min": 0.0, "seconds_max": 0.0}):
        if module.spread(row) is not None:
            failures.append(f"a row with no usable timings has no spread: {row}")


def check_separability(module: Any, failures: list[str]) -> None:
    """Ground rule 7: an effect inside the noise is a phrase, not a number."""
    fast = {"seconds_median": 1.0, "seconds_min": 0.99, "seconds_max": 1.01}
    slow = {"seconds_median": 2.0, "seconds_min": 1.99, "seconds_max": 2.01}
    effect = module.separable_effect(fast, slow)
    if not isinstance(effect, float) or abs(effect + 0.5) > 1e-12:
        failures.append(f"a halving against one percent of noise is separable, got {effect}")

    noisy = {"seconds_median": 1.00, "seconds_min": 0.90, "seconds_max": 1.10}
    also_noisy = {"seconds_median": 1.05, "seconds_min": 0.95, "seconds_max": 1.15}
    effect = module.separable_effect(noisy, also_noisy)
    if effect != module.NOT_SEPARABLE:
        failures.append(f"five percent against twenty percent of noise is not separable, "
                        f"got {effect}")

    # No recorded extremes means no known noise floor, which is the case the rule
    # exists for, so the phrase rather than a confident number.
    bare = {"seconds_median": 1.0, "seconds_min": "", "seconds_max": ""}
    other = {"seconds_median": 2.0, "seconds_min": "", "seconds_max": ""}
    if module.separable_effect(bare, other) != module.NOT_SEPARABLE:
        failures.append("an effect with no measured spread must not print as a number")


def check_bootstrap(module: Any, failures: list[str]) -> None:
    """The interval must contain the fit, and must name which resampling it used."""
    for with_repetitions, expected in ((True, module.MEASURED_REPS),
                                       (False, module.TRIANGULAR)):
        frame = module.pd.DataFrame(scaling_rows("abc123def456", with_repetitions))
        for column in ("workers", "seconds_median", "seconds_min", "seconds_max"):
            frame[column] = module.pd.to_numeric(frame[column])
        knee = module.bootstrap_knee(frame, samples=300)
        if not knee:
            failures.append(f"the bootstrap returned nothing for {expected}")
            continue
        if int(knee["workers"]) != KNEE:
            failures.append(f"the fit should find the knee at {KNEE} workers, got "
                            f"{knee['workers']} for {expected}")
        if knee["method"] != expected:
            failures.append(f"expected the {expected} resampling, got {knee['method']}")
        low, high = knee["low"], knee["high"]
        if low is None or high is None or not low <= KNEE <= high:
            failures.append(f"the interval {low} to {high} does not contain the fitted "
                            f"knee at {KNEE} for {expected}")


def check_dirty_rule(module: Any, failures: list[str], workspace: Path) -> None:
    """Ground rule 6, and the flag that lets this phase be developed at all."""
    module.ROOT = workspace
    module.SUMMARY = workspace / "summary.csv"
    module.MANIFEST = workspace / "session_manifest.json"
    module.FIGURES = workspace / "figures"
    module.TABLES = workspace / "tables"
    module.ASSET_FIGURES = workspace / "assets"

    rows = scaling_rows("4abf914a7ea2.dirty", with_repetitions=True)
    write_summary(module.SUMMARY, rows)

    errors = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
        refused = module.main([])
    if refused == 0:
        failures.append("the generator built assets from .dirty rows with no flag")
    message = errors.getvalue()
    if str(len(rows)) not in message:
        failures.append(f"the refusal must name how many rows are dirty: {message!r}")

    with contextlib.redirect_stdout(io.StringIO()):
        allowed = module.main(["--allow-dirty"])
    if allowed != 0:
        failures.append(f"--allow-dirty should generate anyway, exited {allowed}")

    dispersion = module.TABLES / "dispersion.tex"
    if not dispersion.exists():
        failures.append("dispersion.tex was not written")
    else:
        columns = [line for line in dispersion.read_text(encoding="utf-8").splitlines()
                   if line.startswith("block &")]
        if not columns:
            failures.append("dispersion.tex has no header row")
        elif "p90" in columns[0].lower():
            failures.append("dispersion.tex must not carry a p90 column")

    # A clean row needs no flag. Without this the refusal could be unconditional
    # and every check above would still pass.
    write_summary(module.SUMMARY, scaling_rows("4abf914a7ea2", with_repetitions=True))
    with contextlib.redirect_stdout(io.StringIO()):
        clean = module.main([])
    if clean != 0:
        failures.append(f"a summary with no dirty row needs no flag, exited {clean}")


def main() -> int:
    if not SUMMARY.exists():
        print(f"test_gen_report_assets: {SUMMARY} is missing, nothing to test against",
              file=sys.stderr)
        return 1
    try:
        module = load_generator()
    except ImportError as error:
        print(f"test_gen_report_assets: skipping, the generator cannot be imported here "
              f"({error}). It needs matplotlib and pandas, which the reports job installs "
              f"and the build job does not have to.")
        return 0

    failures: list[str] = []
    check_spread(module, failures)
    check_separability(module, failures)
    check_bootstrap(module, failures)
    with tempfile.TemporaryDirectory() as directory:
        check_dirty_rule(module, failures, Path(directory))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("test_gen_report_assets: spread, separability, the knee bootstrap and the "
          "dirty rule all behave as ground rules 6 and 7 require")
    return 0


if __name__ == "__main__":
    sys.exit(main())
