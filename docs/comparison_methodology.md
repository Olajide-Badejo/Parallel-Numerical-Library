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

Jacobi is the exception, and the reason is instructive: the host implementation
writes a separate output array, so it pays a read for ownership on every cache
line it writes and moves more traffic than the byte model assumes, while the GPU
does not. That is a property of how CPUs handle writes, not of the algorithm.

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
