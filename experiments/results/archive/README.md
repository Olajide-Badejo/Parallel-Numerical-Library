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
is the one the 1.0.0 generator selected, at commit `cd57032941a8.dirty`, and it
is what every published figure was built from.

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

## Both generations are dirty

Every row in both files carries a `.dirty` commit stamp, so the exact source
that produced every number in either generation is not in git history. The three
mechanisms that made the tree dirty are `PROV-01`; they were fixed in phase A0
and the sweep now refuses to run from a tree in that state. Neither generation
can be reproduced, by anyone, including its author. They are kept as the record
of what was published and not as data to build on. Phase A8b replaces
`../summary.csv` with a generation measured from a clean tree.
