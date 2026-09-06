# Archived measurement output

What is here is kept because it was published from, not because it is usable.
Nothing in this directory is an input to the report. The generator reads
`../summary.csv` and the session manifest that names the commit those rows
carry; it reaches into this directory only under `--allow-dirty`, and it says so
on every run when it does.

## `summary-4abf914a7ea2-dirty.csv`

The 425 rows of the superseded generation, moved out of `../summary.csv` in
phase A8a. Release 1.0.0 shipped both generations in one 850 row file and told
them apart with the commit of the last row, which is a statement about the order
two sweeps were appended in. Both generations have 425 rows, so that choice was
decided by nothing a reader could see. The generation left in `../summary.csv`
was the one the 1.0.0 generator selected, at commit `cd57032941a8.dirty`, it is
what every figure release 1.0.0 published was built from, and it is the file
below.

## `summary-cd57032941a8-dirty.csv`

The 425 rows release 1.0.0 published, moved out of `../summary.csv` in phase
A8b. The commit is part of a row's identity, so the publication sweep of
release 1.1.0 appended its own 425 rows beside these rather than replacing
them, and for the length of that session the file held two generations again
and the generator refused it. This file is byte for byte the rows that were in
`../summary.csv` at commit `fba9872e407f`, header included, and nothing else
was done to them.

## `manifest-cd57032941a8-dirty.json`

The file that was tracked as `../session_manifest.json`. It is archived rather
than deleted because it is the provenance record release 1.0.0 shipped, and it
is archived rather than kept in place because it does not describe the session
whose rows were published. Its counts are

```json
"declared": 8, "executed": 0, "skipped_already_present": 8
```

Eight configurations declared, none executed, eight found already present. The
sweep that produced the 425 published rows declared 440. So this manifest
records a later eight configuration re run that measured nothing, and its
`started` timestamp, host block, toolchain versions and both bandwidth probes
belong to that re run and not to the sweep the report quotes.

It also has no `bandwidth_refreshed` key, which `run_sweep.py` writes whenever
the refresh runs. `make bandwidth-refresh` therefore never ran in the committed
session, and the host figure it carries, 61.35 GiB/s, is the reading taken at
the start of a session, on a machine that had just built and tested. That is the
mechanism behind the four irreconcilable bandwidth figures of Section 4.3 of the
V2 specification. See `PROV-04` in `../../../docs/ENGINEERING_LOG.md`.

Its `toolchain.cxx` field reads `g++-16`, and that field was a literal written
into the sweep harness rather than a probe of the build that ran, which is
`PROV-05`. It happens to name the compiler release 1.0.0 was in fact built
with, for a reason no reader could check from the file.

## Reproducing either generation

| | the 1.0.0 generation | the 1.1.0 generation |
| --- | --- | --- |
| where | `summary-cd57032941a8-dirty.csv` | `../summary.csv` |
| rows | 425 | 425 |
| commit stamp | `cd57032941a8.dirty` | `fba9872e407f` |
| in git history | no, the tree was dirty when it was measured | yes |
| session manifest | `manifest-cd57032941a8-dirty.json`, which describes a different session | `../manifest-fba9872e407f-20260906T203409Z.json`, which describes this one |
| C++ compiler | GCC 16.0.1 20260322 experimental, an unreleased trunk snapshot, from the 1.0.0 toolchain table in `../../../PROGRESS.md` | `g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, read from the build cache and recorded in the manifest as `cxx` beside the `cxx_probe` that answered |
| C++ standard | C++23 | C++20, decision 20 |
| byte model | one, `bytes_per_unknown` 24 on the stencil. `dram_bytes_per_unknown_per_sweep` is empty in every row because the column did not exist | both, 24 declared and 32 counted, published side by side because the selection rule came out unresolved on this machine |
| sweeps per iteration | not recorded, and charged as one for every method | recorded per row: 1 for every method except `gauss_seidel_s` and `ssor`, which are 2 |
| host triad every efficiency divides by | 61.35 GiB/s, taken at the start of a session on a machine that had just built and tested, never refreshed | 70.007 GiB/s plain and 71.518 GiB/s non temporal, re probed on an idle machine after the sweep, `bandwidth_refreshed` in the manifest |

Only the right hand column can be reproduced. `git checkout fba9872e407f && make
setup && make all` rebuilds the 1.1.0 binary from the source that measured it.
There is no such command for the left hand column: its commit stamp ends in
`.dirty`, so the source that produced those numbers is not in git history and
nobody can check it out, including its author.

## The 1.0.0 timings are superseded and its iterates are not

Every timing in the left hand column is superseded. The compiler changed, the
C++ standard changed, phase A1 removed a full state copy from the Jacobi path,
phase A5 took allocation out of the timed region, phase A2 corrected the work
unit and phase A6 corrected the hybrid worker count, so a second of 1.0.0 and a
second of 1.1.0 are not measuring the same program doing the same work.

The values are another matter. Joining the two files on everything that
identifies a configuration except the commit gives 410 configurations named by
both. Of those, 396 carry the same `relative_residual` and the same
`iterations` to every digit the row prints. The 14 that differ are the whole of
the `reduction_cost` block, whose fixed iteration count phase A7 raised from 200
to 3000, and the whole of the `schedule_cost` block, raised from 100 to 1100:
they ran further, so they arrived somewhere else, and no row in either block
disagrees for any other reason. The 15 configurations present only on the left
are the hybrid rows that recorded 5 workers, and the 15 present only on the
right are the same configurations recording the 20 they actually ran, which is
`MEAS-09`.

So the arithmetic did not move between the two releases and the clock did.
`PNL_REDUCTION_ACCUMULATORS` stays at 1, which is what holds the iterates fixed,
and Section 0.1.2 of the V2 specification is where that is stated for the
release as a whole.

## Every generation in this directory is dirty

Every row in both archived files carries a `.dirty` commit stamp, so the exact
source that produced every number in either generation is not in git history.
The three mechanisms that made the tree dirty are `PROV-01`; they were fixed in
phase A0 and the sweep now refuses to run from a tree in that state. Neither
generation can be reproduced, by anyone, including its author. They are kept as
the record of what was published and not as data to build on. Phase A8b
replaced `../summary.csv` with a generation measured from a clean tree, and that
generation is the first in this repository that a stranger can reproduce.
