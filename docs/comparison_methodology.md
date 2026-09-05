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
superlinear. Both probes run once per sweep session and land in
`experiments/results/session_manifest.json`.

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

The measured decomposition on this machine, at 4095 by 4095, which is the
smallest of the three sizes where both devices are unambiguously streaming
rather than partly served from cache:

| | host, 20 threads | RTX 5070 |
| --- | --- | --- |
| measured triad bandwidth | 62.3 GiB/s | 549.7 GiB/s |
| achieved, Jacobi | 17.5 GiB/s | 515.5 GiB/s |
| efficiency, Jacobi | 28.1 percent | 93.8 percent |
| achieved, red black Gauss Seidel | 28.5 GiB/s | 258.8 GiB/s |
| efficiency, red black Gauss Seidel | 45.7 percent | 47.1 percent |
| achieved, conjugate gradient | 12.6 GiB/s | 110.2 GiB/s |
| efficiency, conjugate gradient | 20.2 percent | 20.1 percent |

The bandwidth ratio between the two devices is 8.8. Applying the decomposition:

| method | efficiency ratio | predicted speedup | measured |
| --- | --- | --- | --- |
| red black Gauss Seidel | 1.03 | 9.1 | 8.3 |
| conjugate gradient | 1.00 | 8.8 | 8.4 |
| Jacobi | 3.34 | 29 | 25 |

Read the first two rows carefully, because they are the interesting ones. For
red black Gauss Seidel and for conjugate gradient the two devices are used
**equally well**, within a percentage point or two. The entire advantage of the
graphics card on those kernels is that its memory system is 8.8 times faster.
Nothing about the port, the language, or the programming model contributes
anything measurable.

Jacobi is the exception, and the leading explanation is the open question of the
traffic model above: the host implementation writes a separate output array, and
if it pays a read for ownership on every cache line it writes then it moves a
third more traffic than the conservative byte model charges, while the GPU does
not. That would be a property of how a write allocate cache handles stores, not
of the algorithm. It is stated here as the hypothesis it is. The two triads
decide it, the rule that reads them is fixed above, and until the publication
session takes that measurement this row is the one place in the comparison where
the declared and the counted models disagree enough to matter.

The honest one line summary is therefore: *on a bandwidth bound stencil sweep,
this GPU moves about nine times more data per second than this CPU, and on two of
three kernels that is the whole of the difference.* That is a useful thing to
know and much less exciting than a bare speedup figure, which is why the report
says the first and not the second.

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
