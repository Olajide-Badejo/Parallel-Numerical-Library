#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self test for scripts/gen_report_assets.py.

The generator is where ground rule 7 lives. It decides which differences the
report is allowed to print as a number and which print as a phrase, how wide the
knee interval is, whether an asset may be built from a dirty tree at all, and
which generation of rows a report is built from. None of that is exercised by
building the report, which succeeds whatever the generator decides, so it is
tested here against a summary with known values.

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
import json
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


# The device comparison block the counted byte model is read from. The size is
# the one the committed sweep uses and its working set is comfortably larger than
# either device's last level cache, so the table quotes a streaming efficiency
# rather than the words "cache resident".
DEVICE_UNKNOWNS = 16769025
DEVICE_ITERATIONS = 300

# solver, backend, GiB/s as the binary wrote it, sweeps, passes, median seconds.
# The two cuda rows carry a bandwidth computed from the kernel time beside a
# median that is the kernel plus the transfer, which is the arrangement MEAS-14
# was about: a counted figure recomputed from the median lands on a different
# clock from the declared figure beside it, and 500 GiB/s over 0.5 s at this size
# cannot both be true.
DEVICE_ROWS = (
    ("jacobi", "openmp", 50.0, 1, 1, 2.0),
    ("gauss_seidel_rb", "openmp", 30.0, 1, 2, 3.0),
    ("jacobi", "cuda", 500.0, 1, 1, 0.5),
    ("gauss_seidel_rb", "cuda", 260.0, 1, 2, 0.9),
)


def device_rows(commit: str) -> list[dict[str, Any]]:
    """One device comparison block, two methods that differ only in pass count."""
    rows: list[dict[str, Any]] = []
    for solver, backend, gib, sweeps, passes, seconds in DEVICE_ROWS:
        rows.append({
            "problem": "poisson2d_rich_4095",
            "unknowns": DEVICE_UNKNOWNS,
            "solver": solver,
            "backend": backend,
            "workers": 20 if backend == "openmp" else 1,
            "pinning": "none",
            "reduction": "deterministic" if backend == "openmp" else "device",
            "schedule": "static",
            "mode": "fixed",
            "iterations": DEVICE_ITERATIONS,
            "converged": 0,
            "seconds_median": f"{seconds:.9f}",
            "seconds_min": f"{seconds * 0.99:.9f}",
            "seconds_max": f"{seconds * 1.01:.9f}",
            "reps": 3,
            "gib_per_second": gib,
            "bytes_per_unknown": 24.0,
            "dram_bytes_per_unknown_per_sweep": 32.0,
            "sweeps": sweeps,
            "passes": passes,
            "commit": commit,
            "label": "device_comparison",
            "seconds_reps": ";".join(f"{seconds * factor:.9f}"
                                     for factor in (0.99, 1.0, 1.01)),
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


def write_manifest(path: Path, recounted: bool = True) -> None:
    """A session manifest with the two probes the generator divides by.

    `recounted` carries the plain triad counted the way the counted column
    counts a kernel, which is the denominator that column has to be divided by.
    A manifest written before that probe existed has no such entry, and the
    counted percentages must then be left underived rather than divided by a
    triad counted the other way.
    """
    bandwidth: dict[str, Any] = {
        "host": {"gib_per_second": 60.0, "detail": "synthetic"},
        "gpu": {"gib_per_second": 550.0, "detail": "synthetic"},
    }
    if recounted:
        bandwidth["derived"] = {
            "host_plain_at_32_bytes": {"value": 80.0, "note": "synthetic, 60 x 32 / 24"},
        }
    path.write_text(json.dumps({
        "commit": path.name,
        "declared": 12,
        "executed": 12,
        "skipped_already_present": 0,
        "bandwidth": bandwidth,
    }, indent=2), encoding="utf-8")


def point_at(module: Any, workspace: Path) -> None:
    """Send every path the generator reads or writes into a workspace."""
    module.ROOT = workspace
    module.SUMMARY = workspace / "summary.csv"
    module.FIGURES = workspace / "figures"
    module.TABLES = workspace / "tables"
    module.ASSET_FIGURES = workspace / "assets"


def device_table_cells(module: Any, workspace: Path,
                       recounted: bool = True) -> dict[tuple[str, str], list[str]]:
    """Generate a device comparison table and return its cells, row by row.

    Read out of the written table rather than out of the functions behind it,
    because MEAS-14 was a defect of what the table said and not of any one
    function: the numerator came from one place, the denominator from another,
    and each was defensible alone.
    """
    point_at(module, workspace)
    write_manifest(workspace / "manifest-4abf914a7ea2-20260905T151102Z.json", recounted)
    write_summary(module.SUMMARY, device_rows("4abf914a7ea2"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if module.main([]) != 0:
            return {}
    cells: dict[tuple[str, str], list[str]] = {}
    table = module.TABLES / "device_comparison.tex"
    if not table.exists():
        return {}
    for line in table.read_text(encoding="utf-8").splitlines():
        if "&" not in line or f"{DEVICE_UNKNOWNS:,}" not in line:
            continue
        columns = [column.strip() for column in line.rstrip("\\ ").split("&")]
        backend = "cuda" if "RTX" in columns[2] else "openmp"
        cells[(columns[0].replace("\\_", "_"), backend)] = columns
    return cells


def check_counted_model(module: Any, failures: list[str], workspace: Path) -> None:
    """MEAS-14: one byte accounting on both sides of a ratio, one clock per row.

    An efficiency is a ratio of two achieved bandwidths and means nothing unless
    the same byte count is applied to both. With the triad recounted the way the
    kernel is, the read for ownership charge cancels and a counted efficiency is
    the declared one times passes over sweeps: the same number for a one pass
    method, twice it for a two pass one. Each of the four checks below is a cell
    the 1.1.0 generation printed above one hundred percent.
    """
    # GiB/s declared, GiB/s counted, percent declared, percent counted.
    declared_gib, counted_gib, declared, counted = 4, 5, 6, 7
    cells = device_table_cells(module, workspace)
    if len(cells) != len(DEVICE_ROWS):
        failures.append(f"the device comparison table has {len(cells)} of "
                        f"{len(DEVICE_ROWS)} rows at {DEVICE_UNKNOWNS:,} unknowns")
        return

    for backend in ("openmp", "cuda"):
        row = cells[("jacobi", backend)]
        if row[declared] != row[counted]:
            failures.append(f"jacobi on {backend} is one pass to one sweep, so its "
                            f"counted efficiency must equal its declared "
                            f"{row[declared]}, got {row[counted]}")
        row = cells[("gauss_seidel_rb", backend)]
        if abs(float(row[counted]) - 2.0 * float(row[declared])) > 0.1:
            failures.append(f"gauss_seidel_rb on {backend} is two passes to one sweep, "
                            f"so its counted efficiency must be twice its declared "
                            f"{row[declared]}, got {row[counted]}")

    # The counted figure of a device row is derived from the declared one, so it
    # is on the clock the binary used, the kernel time. Recomputing it from
    # seconds_median, which is the kernel plus the transfer, is the second half
    # of MEAS-14 and lands somewhere else entirely.
    row = cells[("jacobi", "cuda")]
    seconds = next(item[5] for item in DEVICE_ROWS if item[:2] == ("jacobi", "cuda"))
    kernel_clock = float(row[declared_gib]) * 32.0 / 24.0
    wall_clock = (32.0 * DEVICE_UNKNOWNS * DEVICE_ITERATIONS / seconds) / 1024.0 ** 3
    if abs(float(row[counted_gib]) - kernel_clock) > 0.1:
        failures.append(f"a device row's counted bandwidth must come out on the same "
                        f"clock as its declared one, {kernel_clock:.1f}, got "
                        f"{row[counted_gib]}")
    if abs(float(row[counted_gib]) - wall_clock) < 1.0:
        failures.append(f"the counted bandwidth {row[counted_gib]} is the wall clock "
                        f"derivation {wall_clock:.1f}, which is the clock the declared "
                        f"column beside it did not use")

    # The denominator, and the refusal when the session cannot supply it.
    triad = module.recounted_triad({"host": {"gib_per_second": 60.0},
                                    "gpu": {"gib_per_second": 550.0},
                                    "derived": {"host_plain_at_32_bytes": {"value": 80.0}}})
    if triad["host"] != 80.0 or abs(triad["gpu"] - 550.0 * 32.0 / 24.0) > 1e-9:
        failures.append(f"the recounted triads should be 80.0 and 733.3, got {triad}")

    cells = device_table_cells(module, workspace, recounted=False)
    if len(cells) != len(DEVICE_ROWS):
        failures.append("the table must still be written when the manifest carries no "
                        "recounted triad")
        return
    for key, row in cells.items():
        # The host denominator is measured and is read from the manifest, so a
        # session that predates the probe leaves those cells underived rather
        # than dividing by the triad counted the other way. The device
        # denominator is arithmetic the generator does itself on a figure every
        # manifest carries, so those cells are unaffected, and saying which is
        # which is the point of the two branches.
        expected = module.PREDATES if key[1] == "openmp" else None
        if expected is not None and row[counted] != expected:
            failures.append(f"with no recounted host triad to divide by, the counted "
                            f"efficiency of {key} must read {expected!r}, got "
                            f"{row[counted]!r}")
        if expected is None and row[counted] == module.PREDATES:
            failures.append(f"the device denominator is derived from the manifest's own "
                            f"gpu figure, so {key} should still carry a counted "
                            f"efficiency")


def check_generation_rule(module: Any, failures: list[str], workspace: Path) -> None:
    """One generation per summary, and a manifest that names its commit.

    Release 1.0.0 chose between two generations with the commit of the last row
    in the file, which is a statement about append order. Both refusals here are
    that selector's replacement, so both need a test that fails if either is
    quietly relaxed back into a choice.
    """
    point_at(module, workspace)
    write_manifest(workspace / "manifest-4abf914a7ea2-dirty-20260905T140311Z.json")

    two = (scaling_rows("4abf914a7ea2.dirty", with_repetitions=True)
           + scaling_rows("cd57032941a8.dirty", with_repetitions=True))
    write_summary(module.SUMMARY, two)
    errors = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
        refused = module.main(["--allow-dirty"])
    if refused == 0:
        failures.append("the generator built assets from two generations in one summary")
    for commit in ("4abf914a7ea2.dirty", "cd57032941a8.dirty"):
        if commit not in errors.getvalue():
            failures.append(f"the refusal must name {commit}: {errors.getvalue()!r}")

    # One generation whose commit no manifest carries. Every efficiency figure
    # divides by a bandwidth from that manifest, so this is a refusal and not a
    # gap to be filled with nothing.
    write_summary(module.SUMMARY, scaling_rows("0282876a0b99", with_repetitions=True))
    errors = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
        refused = module.main([])
    if refused == 0:
        failures.append("the generator ran with no manifest for the commit it published")
    if "0282876a0b99" not in errors.getvalue():
        failures.append(f"the refusal must name the commit: {errors.getvalue()!r}")


def check_dirty_rule(module: Any, failures: list[str], workspace: Path) -> None:
    """Ground rule 6, and the flag that lets this phase be developed at all."""
    point_at(module, workspace)
    write_manifest(workspace / "manifest-4abf914a7ea2-dirty-20260905T140311Z.json")
    write_manifest(workspace / "manifest-4abf914a7ea2-20260905T151102Z.json")

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
    with tempfile.TemporaryDirectory() as directory:
        check_generation_rule(module, failures, Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        check_counted_model(module, failures, Path(directory))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("test_gen_report_assets: spread, separability, the knee bootstrap, the dirty "
          "rule, the one generation rule and the counted byte model all behave as "
          "ground rules 6, 7 and 9 require")
    return 0


if __name__ == "__main__":
    sys.exit(main())
