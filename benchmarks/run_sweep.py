#!/usr/bin/env python3
"""Resolve sweep_matrix.yaml, run every configuration, and assemble summary.csv.

Section 7 of the specification requires three things of this script and it does
all three.

  Resumable.  A completed (solver, backend, size, workers, pinning, mode,
              reduction, schedule, commit) row is skipped on a rerun. The sweep
              takes over an hour, so it has to survive an interruption without
              redoing work. Pass --force to redo anyway.

  Reproducible.  Every row carries the seed and the commit hash, both stamped by
              the binary rather than by this script, so a row cannot claim
              provenance the binary did not have.

  Honest.     Both devices' measured bandwidth probes run once per session and
              land in the session manifest. Every efficiency number in the
              report divides by those, never by a specification sheet figure.
              A configuration that fails is recorded as a failure with its
              stderr, not dropped.

The merge into summary.csv is atomic: rows are written to a temporary file in
the same directory and renamed over the target, so an interrupted run cannot
leave a half written summary that the report would then build from.

The header of that summary must equal the header the binary emits, exactly,
because mixing two schemas in one file shifts every field silently. When the
binary has gained a column, --migrate adds it to the stored rows with the
default declared in scripts/migrate_summary.py rather than making the sweep
refuse; --dry-run does the header check and the resume calculation and stops,
which is the cheap way to see what a sweep would do before spending an hour
finding out.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("run_sweep needs PyYAML: python3 -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(2) from None

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
MATRIX = ROOT / "benchmarks" / "sweep_matrix.yaml"
MANIFEST = RESULTS / "session_manifest.json"
MIGRATE = ROOT / "scripts" / "migrate_summary.py"

# The columns the binary emits, in order. Kept here so a mismatch is caught
# rather than silently shifting every field.
EXPECTED_HEADER_PREFIX = "problem,unknowns,solver,backend,workers"

# Top level keys of the matrix that declare something other than a block of
# runs. "meta" holds the defaults every block inherits and "preregistered" holds
# the predictions of ground rule 10, which are written down before the
# measurement and are read by a human and by the report, never expanded into
# configurations.
NON_BLOCK_KEYS = frozenset({"meta", "preregistered"})

# Rows of `pnl --bandwidth` whose first field starts with this prefix are
# arithmetic on the two probes rather than a third probe, and are filed under
# `bandwidth.derived` in the manifest so nothing can mistake one for a
# measurement.
DERIVED_PREFIX = "derived_"

# Fields that together identify a configuration for resume purposes.
#
# "label" carries the block name and is part of the identity for a reason that
# cost real data before it was found: two blocks can declare the same
# configuration and differ only in a field that is not here, such as the
# iteration count. The device comparison block and the backend cost block both
# run Jacobi on OpenMP at twenty workers in fixed mode, differing only in
# running 300 sweeps against 200. Without the block in the identity the second
# one looked already complete, was skipped, and its rows were silently absent
# from the results. The same collision removed several scaling and pinning
# points.
#
# The block name is used rather than the whole label because the device path
# appends timing detail to it, which would never match on a rerun.
#
# "kernels" and "kernel_variant" are here from the commit that added the columns,
# not from the commit that adds a second value to either. A Fortran row and a C++
# row that differ in nothing else would otherwise collide, the second would be
# skipped as already complete, and half the new data would never be measured.
# That is SWEEP-03 a second time, and it cost real data the first time.
IDENTITY_FIELDS = (
    "problem",
    "solver",
    "backend",
    "unknowns",
    "workers",
    "pinning",
    "mode",
    "reduction",
    "schedule",
    "kernels",
    "kernel_variant",
    "label",
    "commit",
)

# The position of the commit inside an identity tuple, so that the dry run can
# report a row that matches in everything except the build that produced it.
COMMIT_FIELD = IDENTITY_FIELDS.index("commit")


@dataclass
class Run:
    """One configuration to execute."""

    block: str
    solver: str
    backend: str
    problem: str
    size: int
    workers: int
    mode: str
    iterations: int
    check_interval: int
    pinning: str = "none"
    reduction: str = "deterministic"
    schedule: str = "static"
    # Which kernel table runs, and which implementation of it. Both have one
    # value in this release and are carried anyway, because they are part of the
    # resume identity and a row without them cannot be told apart from the
    # Fortran and assembly rows that release 1.2.0 adds. Neither is passed on the
    # command line yet: the driver grows the flags in 1.2.0, and sending it a
    # flag it does not know would fail every run in the block.
    kernels: str = "cxx"
    kernel_variant: str = "cpp"
    rhs: str = "rich"
    blocks: int = 0
    threads_per_rank: int = 1
    tolerance: float = 1.0e-8
    repetitions: int = 5
    seed: int = 20260802

    def command(self, binary: Path, mpirun: str | None) -> list[str]:
        argv: list[str] = []
        if self.backend in ("mpi", "hybrid"):
            ranks = self.workers if self.backend == "mpi" else max(
                1, self.workers // max(1, self.threads_per_rank)
            )
            argv += [mpirun or "mpirun", "--oversubscribe", "-n", str(ranks)]
        argv += [
            str(binary),
            "--solver", self.solver,
            "--backend", self.backend,
            "--problem", self.problem,
            "--rhs", self.rhs,
            "--size", str(self.size),
            "--workers", str(self.workers),
            "--threads-per-rank", str(self.threads_per_rank),
            "--pinning", self.pinning,
            "--reduction", self.reduction,
            "--schedule", self.schedule,
            "--mode", self.mode,
            "--iterations", str(self.iterations),
            "--check-interval", str(self.check_interval),
            "--tolerance", repr(self.tolerance),
            "--reps", str(self.repetitions),
            "--seed", str(self.seed),
            "--label", self.block,
        ]
        if self.blocks > 0:
            argv += ["--blocks", str(self.blocks)]
        return argv

    def predicted_problem(self) -> tuple[str, int]:
        """The problem name and unknown count the binary will report.

        Predicted rather than read back, because the resume check has to happen
        before the run, not after it. An earlier version compared identities
        only after executing the command, which made the sweep re-run every
        configuration and skip merely the write; "resumable" was then a claim
        the code did not honour.
        """
        if self.problem == "poisson":
            kind = "sine" if self.rhs == "sine" else "rich"
            return f"poisson2d_{kind}_{self.size}", self.size * self.size
        kind = "spd" if self.problem == "dense_spd" else "dd"
        return f"dense_{kind}_{self.size}", self.size

    def predicted_workers(self) -> int:
        """The worker count the binary will report, which is not always the one
        requested: the serial and device backends report one however many were
        asked for, and the hybrid backend reports ranks times threads."""
        # fortran_dc_serial is here before the backend exists, because this is
        # the commit that rewrote these lines and the omission is the same defect
        # as SWEEP-05 wearing a different field name. The backend arrives in
        # release 1.2.0, on a build whose do concurrent probe came back negative,
        # and it reports one worker however many were asked for, exactly as
        # serial does. Without the case the prediction would carry the request,
        # the stored row would carry 1, and every one of those rows would be re
        # run on every sweep.
        if self.backend in ("serial", "cuda", "fortran_dc_serial"):
            return 1
        # The hybrid case was written for a binary that did not yet exist. Until
        # MEAS-09 the backend inherited worker_count() from MpiBackend and
        # reported its rank count, so the prediction below and the row on disk
        # disagreed and every hybrid configuration was re run on every sweep.
        # The clamp matters as much as the product: `command` launches
        # max(1, threads_per_rank) threads per rank and the backend clamps the
        # same way, so predicting the unclamped product would put a zero in the
        # identity of a configuration the driver runs with one thread.
        if self.backend == "hybrid":
            threads = max(1, self.threads_per_rank)
            ranks = max(1, self.workers // threads)
            return ranks * threads
        return self.workers

    def predicted_identity(self, commit: str) -> tuple[str, ...]:
        """The identity tuple this configuration's row will carry."""
        problem, unknowns = self.predicted_problem()
        device = self.backend == "cuda"
        # The backend column is not normalised, and getting this wrong was
        # SWEEP-05. The driver prints the literal "cuda" in that column, and
        # every stored CUDA row carries "cuda"; predicting "device" here made a
        # CUDA identity that could never match a stored row, so the resume check
        # always missed and every CUDA configuration was re run on every sweep.
        # What the device path does normalise is reduction, pinning, kernels and
        # kernel_variant, all four of which it prints as a fixed value that
        # ignores what was requested.
        reduction = "device" if device else self.reduction
        pinning = "none" if device else self.pinning
        kernels = "device" if device else self.kernels
        kernel_variant = "device" if device else self.kernel_variant
        values = {
            "problem": problem,
            "solver": self.solver,
            "backend": self.backend,
            "unknowns": str(unknowns),
            "workers": str(self.predicted_workers()),
            "pinning": pinning,
            "mode": self.mode,
            "reduction": reduction,
            "schedule": self.schedule,
            "kernels": kernels,
            "kernel_variant": kernel_variant,
            "label": self.block,
            "commit": commit,
        }
        return tuple(values[name] for name in IDENTITY_FIELDS)

    def describe(self) -> str:
        detail = f"{self.solver}/{self.backend}/{self.size}/w{self.workers}"
        if self.pinning != "none":
            detail += f"/{self.pinning}"
        if self.reduction != "deterministic":
            detail += f"/{self.reduction}"
        if self.schedule != "static":
            detail += f"/{self.schedule}"
        return detail


@dataclass
class Session:
    """Everything about this run of the sweep that a reader might need."""

    started: str
    commit: str
    host: dict[str, Any]
    toolchain: dict[str, str]
    bandwidth: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)


def as_list(value: Any) -> list[Any]:
    """Accept either a scalar or a list in the matrix, always return a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def backend_sides(spec: dict[str, Any]) -> list[tuple[str, int]]:
    """The (backend, workers) pairs a block asks for.

    Most blocks give one list of backends and one list of worker counts. The
    device comparison block instead names a cpu side and a gpu side, because the
    GPU is one device however many host threads exist and pairing it against
    every CPU worker count would be meaningless.
    """
    pairs: list[tuple[str, int]] = []
    if "cpu" in spec or "gpu" in spec:
        for side in ("cpu", "gpu"):
            if side not in spec:
                continue
            for backend in as_list(spec[side].get("backends")):
                for workers in as_list(spec[side].get("workers", [1])):
                    pairs.append((str(backend), int(workers)))
        return pairs
    for backend in as_list(spec.get("backends")):
        for workers in as_list(spec.get("workers", [1])):
            pairs.append((str(backend), int(workers)))
    return pairs


def expand_block(block: str, spec: dict[str, Any], meta: dict[str, Any]) -> list[Run]:
    """Turn one declared block into its configurations."""
    # The kernels and variants axes both default to the single value this
    # release has, so every block written before they existed expands to exactly
    # the same configurations it did before.
    axes = itertools.product(
        as_list(spec.get("problem", "poisson")),
        as_list(spec.get("sizes")),
        backend_sides(spec),
        as_list(spec.get("solvers")),
        as_list(spec.get("pinnings", ["none"])),
        as_list(spec.get("reductions", ["deterministic"])),
        as_list(spec.get("schedules", ["static"])),
        as_list(spec.get("kernels", ["cxx"])),
        as_list(spec.get("variants", ["cpp"])),
    )
    iterations = int(spec.get("iterations", spec.get("max_iterations", 100000)))
    runs: list[Run] = []
    for (problem, size, (backend, workers), solver, pinning, reduction, schedule,
         kernels, variant) in axes:
        runs.append(
            Run(
                block=block,
                solver=str(solver),
                backend=backend,
                problem=str(problem),
                size=int(size),
                workers=workers,
                mode=str(spec.get("mode", "solve")),
                iterations=iterations,
                check_interval=int(spec.get("check_interval", 1)),
                pinning=str(pinning),
                reduction=str(reduction),
                schedule=str(schedule),
                kernels=str(kernels),
                kernel_variant=str(variant),
                rhs=str(spec.get("rhs", "rich")),
                blocks=int(spec.get("blocks", 0)),
                threads_per_rank=(
                    int(spec.get("hybrid_threads_per_rank", 4)) if backend == "hybrid" else 1
                ),
                tolerance=float(meta.get("tolerance", 1.0e-8)),
                # A block may override the repetition count. The convergence
                # blocks use one, since an iteration count is deterministic and
                # repeating it measures nothing.
                repetitions=int(spec.get("repetitions", meta.get("repetitions", 5))),
                seed=int(meta.get("seed", 20260802)),
            )
        )
    return runs


def expand(matrix: dict[str, Any], only: set[str] | None) -> list[Run]:
    """Turn the declared matrix into a flat list of configurations."""
    meta = matrix.get("meta", {})
    runs: list[Run] = []
    for block, spec in matrix.items():
        if block in NON_BLOCK_KEYS or not isinstance(spec, dict):
            continue
        if only and block not in only:
            continue
        runs.extend(expand_block(block, spec, meta))
    return runs


def load_existing(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def migrate_existing(summary: Path, binary: Path) -> int:
    """Add to an older summary the columns the binary has gained.

    Run as a subprocess rather than imported, so that there is one migration and
    one table of declared defaults: scripts/migrate_summary.py is what a reader
    runs by hand, and this runs the same command with the same arguments.
    """
    proc = subprocess.run(
        [sys.executable, str(MIGRATE), str(summary), "--header-from", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    for stream, text in ((sys.stdout, proc.stdout), (sys.stderr, proc.stderr)):
        if text.strip():
            print(text.strip(), file=stream)
    return proc.returncode


def block_of(label: str) -> str:
    """The block name from a label, discarding any appended detail."""
    return str(label).split()[0] if str(label).strip() else ""


def identity(row: dict[str, str]) -> tuple[str, ...]:
    values = {name: str(row.get(name, "")) for name in IDENTITY_FIELDS}
    values["label"] = block_of(values["label"])
    return tuple(values[name] for name in IDENTITY_FIELDS)


def write_atomic(path: Path, header: list[str], rows: Iterable[dict[str, str]]) -> None:
    """Write the summary through a temporary file and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        temporary = handle.name
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    # Rename only after the file is closed and flushed, so a reader can never
    # observe a partially written summary at the real path.
    os.replace(temporary, path)


def run_command(argv: list[str], timeout: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f} s"
    except OSError as error:
        return 127, "", str(error)
    return proc.returncode, proc.stdout, proc.stderr


def probe_commit(binary: Path, header: list[str]) -> str:
    """Ask the binary which commit it was built from.

    Taken from the binary rather than from the first existing row, because those
    rows may predate a rebuild. Using a stale commit for the resume check would
    silently skip configurations whose code has since changed, which is exactly
    the failure the commit column exists to prevent. A four point grid keeps the
    probe instant.
    """
    code, out, _ = run_command(
        [str(binary), "--solver", "jacobi", "--backend", "serial", "--size", "4",
         "--mode", "fixed", "--iterations", "1", "--reps", "1"],
        120,
    )
    if code != 0 or not out.strip():
        return ""
    parsed = list(csv.DictReader(out.strip().splitlines(), fieldnames=header))
    return parsed[0].get("commit", "") if parsed else ""


def collect_session(binary: Path, commit: str) -> Session:
    """Environment and both bandwidth probes, once per session."""
    session = Session(
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        commit=commit,
        host={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        toolchain={},
    )

    for name, argv in (
        ("cxx", ["g++-16", "--version"]),
        ("cmake", ["cmake", "--version"]),
        ("mpi", ["mpirun", "--version"]),
        ("nvcc", ["nvcc", "--version"]),
    ):
        code, out, err = run_command(argv, 60)
        text = (out or err).strip().splitlines()
        session.toolchain[name] = text[0] if text else "unavailable"

    # The probe prints one row per device, plus rows prefixed `derived_` that
    # are arithmetic on those. The two are kept apart in the manifest: a derived
    # figure sitting in a `gib_per_second` field beside two measured ones is
    # exactly how a number nobody measured ends up quoted as one.
    code, out, err = run_command([str(binary), "--bandwidth", "--backend", "openmp"], 1800)
    if code == 0:
        derived: dict[str, Any] = {}
        for line in out.strip().splitlines()[1:]:
            parts = line.split(",", 2)
            if len(parts) != 3:
                continue
            name, value, detail = parts
            if name.startswith(DERIVED_PREFIX):
                derived[name[len(DERIVED_PREFIX):]] = {
                    "value": float(value) if value else None,
                    "note": detail,
                }
            else:
                session.bandwidth[name] = {
                    "gib_per_second": float(value) if value else None,
                    "detail": detail,
                }
        if derived:
            session.bandwidth["derived"] = derived
    else:
        session.bandwidth["error"] = err.strip()

    code, out, err = run_command([str(binary), "--topology"], 900)
    if code == 0:
        verdict = ""
        for line in out.splitlines():
            if line.startswith("verdict:"):
                verdict = line.split(":", 1)[1].strip()
        session.topology = {"verdict": verdict, "raw": out}
    else:
        session.topology = {"error": err.strip()}

    return session


def refresh_bandwidth(binary: Path, header: list[str], quiet: bool) -> int:
    """Re-probe both devices and update the manifest, running nothing else.

    The host triad probe is sensitive to whatever else is using the memory
    system: measured on an idle machine it reads about 55 GiB/s, and measured
    while a build was running it read 39.8. Since every host efficiency figure in
    the report divides by this number, a depressed reading would inflate every
    one of them. The probe therefore needs to run on a quiet machine, which is
    not something the sweep driver can arrange for itself when it runs the probe
    first and then works the machine hard for an hour.

    The device probe is insensitive to host load, as expected, and reads about
    547 GiB/s either way.
    """
    if not MANIFEST.exists():
        print(f"refresh_bandwidth: {MANIFEST} not found; run the sweep first",
              file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    previous = manifest.get("bandwidth", {})

    session = collect_session(binary, probe_commit(binary, header))
    manifest["bandwidth"] = session.bandwidth
    manifest["bandwidth_note"] = (
        "Re-probed on an idle machine. The reading taken at the start of the sweep "
        "session was made while the machine was otherwise busy and understated host "
        "bandwidth; every efficiency figure divides by the value recorded here."
    )
    manifest["bandwidth_previous"] = previous
    manifest["bandwidth_refreshed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if not quiet:
        for device, entry in session.bandwidth.items():
            if device == "derived":
                for name, item in entry.items():
                    print(f"  derived {name} = {item.get('value')}", file=sys.stderr)
                continue
            was = (previous.get(device) or {}).get("gib_per_second")
            now = entry.get("gib_per_second")
            print(f"  {device:17s} {now} GiB/s (was {was})", file=sys.stderr)
        print(f"manifest: {MANIFEST}", file=sys.stderr)
    return 0


def without_commit(key: tuple[str, ...]) -> tuple[str, ...]:
    """An identity with the commit removed, for reporting rather than resuming."""
    return key[:COMMIT_FIELD] + key[COMMIT_FIELD + 1:]


def dry_run(runs: list[Run], rows: list[dict[str, str]], done: set[tuple[str, ...]],
            binary: Path, commit: str) -> int:
    """Report what a sweep would do, and run nothing.

    The header check and the resume calculation are the two things that can be
    wrong before a sweep starts and expensive to discover an hour into one, so
    this exercises both and stops. The only thing it executes is the four point
    grid that reads the commit stamp out of the binary; neither bandwidth probe
    runs, because both are minutes of work and neither affects what would run.

    A stored row that matches in everything but the commit is reported
    separately rather than counted as present. Resuming across builds is exactly
    what the commit column exists to prevent, but a sweep that would redo
    everything because the binary was rebuilt looks identical to one that would
    redo everything because an identity is predicted wrongly, and telling those
    two apart by hand is how SWEEP-05 survived a release.
    """
    commits: dict[tuple[str, ...], set[str]] = {}
    for key in done:
        commits.setdefault(without_commit(key), set()).add(key[COMMIT_FIELD])

    mpirun = shutil.which("mpirun")
    present = 0
    stale = 0
    stale_commits: set[str] = set()
    pending: dict[str, int] = {}
    for run in runs:
        key = run.predicted_identity(commit)
        if key in done:
            status = "have"
            present += 1
        elif without_commit(key) in commits:
            status = "have-earlier"
            stale += 1
            stale_commits |= commits[without_commit(key)]
        else:
            status = "run"
            pending[run.backend] = pending.get(run.backend, 0) + 1
        print(f"{status:12s} {run.block:20s} {' '.join(run.command(binary, mpirun))}")

    # And the same question from the other side. A stored row that no declared
    # configuration predicts is either a block that was removed from the matrix
    # or an identity this script gets wrong, and the second is worth finding:
    # a configuration that never resumes and one that was never measured look
    # identical from the outside.
    predicted = {without_commit(run.predicted_identity(commit)) for run in runs}
    unpredicted: dict[str, int] = {}
    for row in rows:
        if without_commit(identity(row)) not in predicted:
            name = str(row.get("backend", ""))
            unpredicted[name] = unpredicted.get(name, 0) + 1

    print(f"\n{len(runs)} configurations declared, {len(rows)} rows in the summary")
    print(f"{present} already present at commit {commit or 'unknown'}, which is what a sweep "
          "would skip")
    if stale:
        print(f"{stale} present at {', '.join(sorted(stale_commits))} and at no other commit, "
              "which a sweep from this build would measure again")
    total = sum(pending.values())
    print(f"{total} have no stored row at any commit"
          + (": " + ", ".join(f"{name} {count}" for name, count in sorted(pending.items()))
             if pending else ""))
    print(f"{sum(unpredicted.values())} stored rows that no declared configuration predicts"
          + (": " + ", ".join(f"{name} {count}" for name, count in sorted(unpredicted.items()))
             if unpredicted else ""))
    return 0


def format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 359999:
        return "--:--"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Bar:
    """A tqdm style bar that also works when stderr is not a terminal."""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = total
        self.done = 0
        self.start = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.enabled = enabled
        self.last = 0.0

    def update(self, detail: str, force: bool = False) -> None:
        if not self.enabled:
            return
        self.done += 1
        now = time.monotonic()
        if not force and self.tty and now - self.last < 0.2:
            return
        self.last = now
        elapsed = now - self.start
        fraction = self.done / self.total if self.total else 1.0
        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
        width = 28
        filled = int(fraction * width)
        bar = "#" * filled + "." * (width - filled)
        line = (
            f"sweep [{bar}] {fraction * 100:3.0f}% {self.done}/{self.total} "
            f"[{format_eta(elapsed)}<{format_eta(eta)}] {detail}"
        )
        if self.tty:
            print(f"\r\033[2K{line}", end="", file=sys.stderr, flush=True)
        elif self.done == 1 or self.done == self.total or self.done % 25 == 0:
            print(line, file=sys.stderr, flush=True)

    def finish(self) -> None:
        if self.enabled and self.tty:
            print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build",
                        help="build directory containing the pnl binary")
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--out", type=Path, default=RESULTS / "summary.csv")
    parser.add_argument("--only", action="append", default=[],
                        help="run only this block; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="rerun configurations that already have a row")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the header, do the resume calculation, report what would "
                             "run, and exit without running or probing anything")
    parser.add_argument("--migrate", action="store_true",
                        help="add to an existing summary the columns the binary has gained, "
                             "with the defaults declared in scripts/migrate_summary.py, "
                             "instead of refusing to merge into it")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="seconds allowed per configuration")
    parser.add_argument("--refresh-bandwidth", action="store_true",
                        help="re-probe both devices and update the manifest, running no "
                             "configurations")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    binary = args.build / "pnl"
    if not binary.exists():
        print(f"run_sweep: {binary} not found; build first", file=sys.stderr)
        return 2

    code, header_line, err = run_command([str(binary), "--header"], 60)
    if args.refresh_bandwidth:
        return refresh_bandwidth(binary, header_line.strip().split(","), args.quiet)

    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    runs = expand(matrix, set(args.only) if args.only else None)

    if code != 0 or not header_line.startswith(EXPECTED_HEADER_PREFIX):
        print(f"run_sweep: could not read the result header from the binary: {err}",
              file=sys.stderr)
        return 2
    header = header_line.strip().split(",")

    existing_header, existing_rows = load_existing(args.out)
    if existing_header and existing_header != header:
        if not args.migrate:
            print("run_sweep: the existing summary has a different set of columns than the "
                  "binary now emits. Pass --migrate to add the missing ones with their "
                  "declared defaults, or move the file aside rather than mixing schemas.",
                  file=sys.stderr)
            return 2
        code = migrate_existing(args.out, binary)
        if code != 0:
            return code
        existing_header, existing_rows = load_existing(args.out)
        if existing_header != header:
            print("run_sweep: the summary still does not match the binary's header after the "
                  "migration, so the two schemas differ by more than missing columns.",
                  file=sys.stderr)
            return 2

    done = {identity(row) for row in existing_rows} if not args.force else set()

    if args.dry_run:
        return dry_run(runs, existing_rows, done, binary, probe_commit(binary, header))

    RESULTS.mkdir(parents=True, exist_ok=True)
    commit = probe_commit(binary, header)
    session = collect_session(binary, commit)

    if not args.quiet:
        host = session.bandwidth.get("host", {}).get("gib_per_second")
        gpu = session.bandwidth.get("gpu", {}).get("gib_per_second")
        print(f"measured bandwidth: host {host} GiB/s, device {gpu} GiB/s", file=sys.stderr)
        print(f"{len(runs)} configurations declared, {len(done)} already complete",
              file=sys.stderr)

    mpirun = shutil.which("mpirun")
    rows = list(existing_rows)
    bar = Bar(len(runs), enabled=not args.quiet)
    executed = 0
    skipped = 0

    for run in runs:
        # Skip before executing, not after. The identity is predicted from the
        # configuration; if a prediction is ever wrong the only cost is one
        # redundant run, after which the real row makes the match exact.
        if not args.force and run.predicted_identity(commit) in done:
            skipped += 1
            bar.update(f"have: {run.describe()}")
            continue

        argv = run.command(binary, mpirun)
        code, out, err = run_command(argv, args.timeout)

        if code != 0:
            # A solver that does not apply to a problem exits 3, which is a
            # declared outcome rather than a failure, so it is recorded quietly.
            level = "inapplicable" if code == 3 else "failed"
            session.failures.append({
                "block": run.block,
                "config": run.describe(),
                "status": level,
                "exit_code": str(code),
                "message": err.strip().splitlines()[-1] if err.strip() else "",
            })
            bar.update(f"{level}: {run.describe()}")
            continue

        parsed = list(csv.DictReader(out.strip().splitlines(), fieldnames=header))
        if not parsed:
            session.failures.append({
                "block": run.block,
                "config": run.describe(),
                "status": "no output",
                "exit_code": "0",
                "message": err.strip()[:200],
            })
            bar.update(f"empty: {run.describe()}")
            continue

        for row in parsed:
            key = identity(row)
            rows = [existing for existing in rows if identity(existing) != key]
            rows.append(row)
            done.add(key)
            executed += 1

        # Persist after every configuration. The sweep is long; losing an hour
        # of it to an interruption would be its own kind of bug.
        write_atomic(args.out, header, rows)
        bar.update(run.describe())

    bar.finish()

    session.counts = {
        "declared": len(runs),
        "executed": executed,
        "skipped_already_present": skipped,
        "failures": len(session.failures),
        "rows_total": len(rows),
    }
    MANIFEST.write_text(json.dumps(session.__dict__, indent=2, default=str), encoding="utf-8")

    if not args.quiet:
        print(f"\n{executed} rows written, {skipped} skipped, "
              f"{len(session.failures)} not recorded", file=sys.stderr)
        print(f"summary:  {args.out}", file=sys.stderr)
        print(f"manifest: {MANIFEST}", file=sys.stderr)
        for failure in session.failures[:10]:
            print(f"  {failure['status']}: {failure['config']}: {failure['message']}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
