# How the CPU versus GPU comparison is done, and how to read it

This document is written for someone who is going to quote a number from this
repository out of context. It is here so that the honest framing is the easiest
one to find.

## The short version

**This repository does not contain a GPU speedup figure that means anything on
its own.** It contains three quantities, and they answer three different
questions:

1. **Iterations to a fixed tolerance.** A property of the mathematics. Same on
   every machine that ever runs it.
2. **Efficiency against each device's own measured bandwidth.** A dimensionless
   ratio. Says how well a device is being used, and is comparable between
   devices precisely because it has been divided by each device's own ceiling.
3. **Seconds.** A property of one specific CPU paired with one specific GPU,
   under one specific set of compiler flags, on one specific day.

The third is the one people quote and the first two are the ones that transfer.
The report presents them in that order deliberately.

## Why the naive comparison is misleading

The tempting experiment is: run the solver on the CPU, run it on the GPU, divide
the times, print the ratio. Three things go wrong.

**It compares different algorithms.** Natural ordering Gauss Seidel is
sequentially dependent, so it cannot run on a GPU at all. If the CPU runs
Gauss Seidel and the GPU runs Jacobi, the ratio conflates two effects: the
device, and the fact that Jacobi needs about twice as many iterations to reach
the same tolerance. This repository pairs like against like: GPU Jacobi against
CPU Jacobi, GPU red black Gauss Seidel against CPU red black Gauss Seidel. The
ordering penalty is then charged to both sides equally, and reported separately
so it can be seen.

**It compares different amounts of effort.** A GPU port written by someone who
cares, measured against a CPU version compiled without vectorisation or
threading, produces an impressive number about nothing. Here both sides run the
same solver code over the same problem, the CPU side uses every core and is
compiled with the machine's own instruction set, and the equivalence suite
proves the two are computing bit identical values.

**It hides what actually limits the calculation.** A five point stencil sweep
does three memory operations per unknown and about six floating point
operations. It is bandwidth bound on every machine ever built. A ratio of times
between two devices is therefore, to within a small correction, just the ratio
of their memory bandwidths, dressed up as a statement about parallel
programming.

## What is measured instead

### Bytes per unknown per sweep: counted, not estimated

A Jacobi sweep writes one value, reads one right hand side entry, and reads the
previous iterate. The four stencil neighbours of consecutive unknowns overlap,
so in the streaming limit each is already in cache from a nearby access. That
is **three doubles, 24 bytes, per unknown per sweep**, and the same number is
returned by `Poisson2D::bytes_per_unknown_per_sweep()` on the host and set by
the device solver, so the figure the report uses comes from the implementation
rather than from a back of an envelope.

That count is the conservative one, and it is not the only defensible one. The
next section says why, and what is being done about it.

### The traffic model is unsettled, and the rule that settles it is fixed in advance

A store to a cache line that the cache does not already hold has to fetch that
line from memory before it can modify it. This is read for ownership, and it is
a property of a write allocate cache rather than of the algorithm. A Jacobi
sweep writes an output array it never reads, so if the host pays read for
ownership then every one of those writes costs a read as well, and the count
above is short by one double per unknown.

**Both counts are therefore published and neither replaces the other.** Every
result row carries `bytes_per_unknown`, the conservative count, and
`dram_bytes_per_unknown_per_sweep`, the same count with read for ownership
charged. Every table and figure in the report that shows an achieved bandwidth
shows both, labelled *declared* and *counted*. Until the question is settled,
quoting one of them without the other is quoting half the evidence.

The two candidates, derived from the code as it stands and not estimated:

| | conservative | with read for ownership |
| --- | --- | --- |
| reads the right hand side | 8 | 8 |
| reads the previous iterate | 8 | 8 |
| writes the output | 8 | 16, the write plus the line fetched first |
| **per unknown per pass** | **24** | **32** |

An earlier draft tabulated 56 and 80 bytes for these two models. Those totals
were per iteration and included the full state copy the solver driver used to
make with `swap_ranges`, which has since been removed: 24 plus 32
conservatively, and 32 plus 48 with read for ownership. With the copy gone a
Jacobi iteration is one pass and the two candidates are 24 and 32.

**The instrument.** Two host STREAM triads over the same arrays, the same
sizes, the same worker counts and the same repetitions, differing in the store
instruction and in nothing else. `measure_host_triad` is a plain C++ loop, so
the compiler emits ordinary stores into an array the loop never reads, which is
the store under suspicion. `measure_host_triad_nontemporal` writes through
`_mm256_stream_pd` with one `_mm_sfence` after the loop, which does not fetch
the line it overwrites. Both
report against the same declared 24 bytes per element, so the ratio of the two
is the ratio of the traffic they really move: one if the plain loop pays
nothing extra, four thirds if it pays a read for ownership on every line. Both
probes run from `pnl --bandwidth` and land in the session manifest, the second
as an additional entry beside the first and never as a replacement for it.

**The rule, fixed before the measurement.** With
`ratio = host_nontemporal / host`:

- a ratio **above 1.20** selects the read for ownership model, 32 bytes per
  unknown per pass;
- a ratio **below 1.10** selects the conservative model, 24 bytes;
- a ratio **from 1.10 to 1.20** is recorded as unresolved, and both models are
  carried.

The reciprocal is recorded beside the ratio in the manifest so the rule cannot
be read in the wrong direction; in that orientation the thresholds are 0.833
and 0.909 and the inequalities reverse. The rule is written down in full in
`benchmarks/sweep_matrix.yaml` under `preregistered.traffic_model`, together
with the sentence this document and the report will carry under each of the
three outcomes:

1. **If the read for ownership model is selected.** The non temporal triad
   reached a ratio of `<ratio>` against the plain triad on the publication
   machine, above the 1.20 threshold fixed before the measurement, so the read
   for ownership model is selected: a Jacobi pass over the five point stencil
   moves 32 bytes per unknown and not 24, the counted column is the achieved
   bandwidth this report compares against each device's own triad, and the host
   figures rise by a third while the device figures do not move, which is why
   host and device efficiency converge on the Jacobi row.
2. **If the conservative model is selected.** The non temporal triad reached a
   ratio of `<ratio>` against the plain triad on the publication machine, below
   the 1.10 threshold fixed before the measurement, so the conservative model is
   selected: a Jacobi pass moves 24 bytes per unknown, the declared column
   stands as the achieved bandwidth of this report, and the read for ownership
   argument is refuted on this machine rather than confirmed. The Jacobi
   efficiency gap between host and device is then a real gap and not an artefact
   of the denominator.
3. **If it is unresolved.** The non temporal triad reached a ratio of `<ratio>`
   against the plain triad on the publication machine, between the 1.10 and 1.20
   thresholds fixed before the measurement, so the traffic model is recorded as
   unresolved: both counts, 24 and 32 bytes per unknown per pass, are carried in
   every table and figure of this report, neither is presented on its own as the
   achieved bandwidth, and no claim is made that rests on one of them and would
   fail under the other.

**Amendment of 2026-09-06, for MEAS-07.** The thresholds above and the three
sentences are exactly as registered on 2026-09-05 and are not touched. What is
amended is how the statistic they are applied to is taken, and the decision was
made before the publication session took the measurement it governs.

The registered statistic is a ratio of two figures, each the best over the
worker sweep, and MEAS-07 found two faults with it. Run three times on an idle
machine it read 1.0990, 1.0121 and 1.0601: a spread of 0.087, which is the
1.10 to 1.20 undecided band over again. And the two bests need not come from
the same worker count. On the first of those runs the plain probe peaked at
eight workers and the non temporal probe at sixteen, so the ratio compared two
arms that differed in the store instruction and in the worker count at once. A
statistic whose own spread is the width of the band it must fall inside cannot
decide anything, which is ground rule 7 pointed at the denominator instead of
at the numerator.

The amended statistic:

- **Five repetitions, interleaved.** `pnl --bandwidth` runs the plain triad and
  the non temporal triad five times each at every worker count, alternating the
  two, so that a repetition's two arms see the same machine. All five figures
  per worker count are recorded for both probes, not only the best.
- **Matched worker count.** `w*` is the worker count at which the plain triad's
  median over the five repetitions is highest. `S` is the median over the five
  repetitions of `nontemporal(w*) / plain(w*)`, each repetition's ratio taken
  between the two probes of that repetition at that count. No ratio is taken
  between two different worker counts.
- **Rule 7 applies to the statistic itself.** If the interval from the smallest
  to the largest of the five ratios at `w*` contains either threshold, the
  outcome is unresolved whatever `S` is, and this document and the report say
  so. Otherwise `S` decides on the thresholds above.

The session manifest records `w*`, the five plain figures, the five non
temporal figures, the five ratios, `S`, the interval and the outcome under a
`traffic_model` key, so the phase that applies the rule reads an outcome rather
than recomputing one. The registered single figures are recorded beside it and
keep their meaning, so the statistic as registered stays visible next to the
statistic as amended.

**Which sentence applies is pending.** The measurement that decides it is the
publication session's and it has not been taken. Phase A8b applies the rule for
release 1.1.0; the assembly triad of phase D4 confirms it in 1.2.0 rather than
gating it.

**No theoretical peak is quoted as corroboration, and none can be.** The read
for ownership corrected figure for the plain triad, `plain x 32 / 24`, is
recorded in the session manifest as a derived number and is labelled as one. It
is not compared against a manufacturer's bandwidth for this machine's memory,
because **no memory speed is recorded anywhere in this repository**: the
environment table names the CPU and not the memory, and `dmidecode` is not
installed in the WSL guest the measurements run in, so `dmidecode -t memory`
cannot supply one either. Such a comparison would in any case have cut against
the read for ownership argument rather than supporting it, since a STREAM triad
does not reach the high nineties as a percentage of peak. The two triads
measured against each other are the evidence, and they are the only evidence.

### Each device's own bandwidth: measured, not quoted

Both devices are probed with the same STREAM triad kernel, `a[i] = b[i] + q c[i]`,
two reads and one write per element with no reuse (McCalpin, IEEE TCCA
Newsletter, December 1995). The host probe runs over the execution backend, not
on one thread, because a single core cannot saturate a twenty core part and
dividing by a single core figure would make every parallel result look
superlinear. Both host probes run five times at each worker count and the
device probe once, in one sweep session, and land in that session's own
manifest, `experiments/results/manifest-<commit>-<timestamp>.json`.

Manufacturer bandwidth figures are never used. They are a bus width times a
clock, no real kernel reaches them, and dividing by one would make both devices
look equally inefficient while saying nothing about which is being used well.

### The efficiency number

```text
efficiency = achieved GiB/s / that device's own measured triad GiB/s
```

This is the number to compare. It is dimensionless, it is bounded above by one,
and it transfers: a reader with different hardware can measure their own triad,
apply the same fraction, and predict what they would get.

## Reading the result

The measured decomposition on this machine, at the largest size in the device
comparison block. Everything between the two markers below is written by
`scripts/gen_report_assets.py --markdown` from the committed generation and its
session manifest, which is the same pair the report's own device comparison
table is built from. It is not maintained by hand, and a number typed into this
section by hand would be finding 4.3 starting again.

<!-- generated:start -->

The tables in this section are generated by `scripts/gen_report_assets.py --markdown` from the same rows and the same session manifest as the report's own device comparison table. Do not edit them by hand.

Generation: 425 rows at commit `fba9872e407f`.

Measured at 16,769,025 unknowns, the largest size in the block and the one where both devices are unambiguously streaming rather than partly served from cache.

|  | host, all threads | device |
| --- | --- | --- |
| measured triad bandwidth | 70.0 GiB/s | 550.4 GiB/s |
| achieved, jacobi | 53.3 GiB/s | 540.2 GiB/s |
| efficiency, jacobi | 76.1 percent | 98.1 percent |
| spread, jacobi | 2.6 percent | 8.5 percent |
| achieved, gauss_seidel_rb | 30.7 GiB/s | 272.0 GiB/s |
| efficiency, gauss_seidel_rb | 43.9 percent | 49.4 percent |
| spread, gauss_seidel_rb | 1.5 percent | 2.4 percent |
| achieved, sor_rb | 30.6 GiB/s | 271.3 GiB/s |
| efficiency, sor_rb | 43.8 percent | 49.3 percent |
| spread, sor_rb | 2.6 percent | 2.2 percent |
| achieved, cg | 13.7 GiB/s | 110.2 GiB/s |
| efficiency, cg | 19.6 percent | 20.0 percent |
| spread, cg | 1.8 percent | 7.1 percent |

Applying the decomposition, with the bandwidth ratio at 7.9:

| method | efficiency ratio | predicted speedup | measured |
| --- | --- | --- | --- |
| jacobi | 1.29 | 10.1 | 8.5 |
| gauss_seidel_rb | 1.13 | 8.9 | 8.1 |
| sor_rb | 1.13 | 8.9 | 8.1 |
| cg | 1.02 | 8.0 | 7.6 |

The efficiency ratio is the second factor of the decomposition and the bandwidth ratio is the first. A method whose efficiency ratio is near one is used equally well on both devices, so the whole of its advantage is the memory system and none of it is attributable to the port. Every figure above carries the spread of the row it came from, and a difference smaller than that spread is not a difference this data can see.
<!-- generated:end -->

Read the efficiency ratios rather than the seconds. A method whose two devices
come out within a percentage point or two of one another is used **equally
well** on both, so the whole of the graphics card's advantage on that kernel is
that its memory system is faster, and nothing about the port, the language or
the programming model contributes anything the measurement can see.

Jacobi is the row to read last, and the leading explanation for it is the open
question of the traffic model above: the host implementation writes a separate
output array, and if it pays a read for ownership on every cache line it writes
then it moves a third more traffic than the conservative byte model charges,
while the GPU does not. That would be a property of how a write allocate cache
handles stores, not of the algorithm. It is stated here as the hypothesis it is.
The two triads decide it, the rule that reads them is fixed above, and until the
publication session takes that measurement this row is the one place in the
comparison where the declared and the counted models disagree enough to matter.

The honest one line summary is therefore: *on a bandwidth bound stencil sweep
this GPU moves an order of magnitude more data per second than this CPU, and on
most of these kernels that is the whole of the difference.* That is a useful
thing to know and much less exciting than a bare speedup figure, which is why
the report says the first and not the second.

## Results from release 1.0.0 against results from release 1.1.0

The two generations are comparable to reduction tolerance and are not the same
numbers, and a reader has to be able to tell which of the two differences they
are looking at.

**The values did not change.** Release 1.1.0 keeps
`PNL_REDUCTION_ACCUMULATORS` at 1, so the reduction is the same ascending chain
it was in 1.0.0 and every iterate and every residual this library computes is
bit identical to the one 1.0.0 computed. That is a supported configuration and
not a debugging leftover. The reduction association changes only when that
number changes, which is phase D3 of release 1.2.0, and the changelog entry for
it leads with the escape hatch.

**Every timing changed, and for three reasons that are all deliberate.**

1. **The compiler.** 1.0.0 was measured with an unreleased GCC 16 trunk snapshot
   that nobody outside the target machine can install. 1.1.0 is measured with
   the released GCC 15.2.0. A change of compiler changes every timing.
2. **The Jacobi algorithm.** 1.0.0's Jacobi solver copied the whole state on the
   calling thread once per iteration. That copy is gone, so the Jacobi rows are
   not a faster measurement of the same work, they are a measurement of less
   work.
3. **The timed region.** 1.0.0 timed a call that allocated its own workspace,
   so every repetition paid first touch page fault cost inside the measurement.
   The workspace is now allocated outside the timed region.

The work unit changed with them: `sweeps` and `passes` are separate columns
now, Richardson is no longer charged for a residual it evaluates once, and the
symmetric methods are charged the two sweeps they perform. So a 1.1.0 row is
not the same quantity as a 1.0.0 row even where the seconds happen to be close.

The before and after for every headline number that moved is recorded per phase
in the Release 1.1.0 section of [PROGRESS.md](../PROGRESS.md), against the phase
that moved it, with the gate output that measured it. It is deliberately not
copied here: a number typed into this document is a number that stops being
true the next time the sweep runs, which is the fault this whole section exists
to avoid. The archived 1.0.0 generation keeps its own summary and its own
manifest under `experiments/results/archive/`, so either generation can be
rebuilt.

### The crossover, derived rather than asserted

Because iteration counts and per iteration costs are measured separately, the
point at which one method overtakes another can be derived instead of guessed.
For method A on device 1 against method B on device 2:

```text
time = iterations(method, n)  x  unknowns(n)  x  bytes per unknown
       -------------------------------------------------------
                achieved bandwidth(device, method, n)
```

Every term on the right is measured and tabulated in the report. The crossover
is where the two expressions are equal. This is why the report can say where GPU
Jacobi stops being worth it against CPU red black SOR, rather than declaring a
winner.

## What this comparison does not establish

- **Nothing about single solve latency.** Transfer time is reported separately
  and is not folded into the bandwidth figure. For one small solve, moving the
  problem across PCIe can cost more than the solve; for a simulation that keeps
  the state resident, it is paid once and vanishes.
- **Nothing about other problem classes.** These conclusions hold for bandwidth
  bound stencil work. A dense factorisation is compute bound and would rank the
  devices differently.
- **Nothing about other hardware.** Both rooflines are published in the report
  precisely so that a reader can substitute their own.
- **Nothing about power or cost.** Not measured, so not claimed.

## If you are quoting this

Quote the efficiency percentages and the measured bandwidths. If you must quote
a speedup, quote it with the sentence that decomposes it, and say which grid
size it came from, because it changes with size: at 1023 by 1023 the GPU has not
yet reached its streaming regime, and the ratio is smaller.
