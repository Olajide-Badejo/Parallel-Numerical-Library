# Parallel Numerical Library

A C++23 numerical methods library where twelve iterative solvers are written
**once**, against a single problem interface, and run unchanged over **seven
execution backends**: serial, OpenMP, POSIX threads, a `std::jthread` pool, MPI,
hybrid MPI with threads, and CUDA.

The point is not that any one backend is fast. It is that the same numerical work
runs over all of them, so a benchmark can isolate what each programming model
costs without the comparison being contaminated by a separate implementation per
model.

## The claim this rests on

**Every shared memory backend, at every worker count, produces bit identical
iterates. The GPU sweeps are bit identical to the CPU too.**

Not close. Identical. The test suite asserts exact equality across twelve
solvers, two problem families, four backends and seven worker counts.

This is possible because floating point addition is not associative and the
library refuses to let the association vary: the reduction chunk grid is fixed by
the problem size alone, never by the worker count, so partial sums are always
combined in the same order. Getting the GPU to agree as well required disabling
fused multiply add contraction on both sides, because the SOR update has the
shape `a*b + c*d` and the host was fusing it while the device was not.

Two limits are documented rather than hidden: across MPI **rank** counts results
agree to reduction tolerance, since rank boundaries do not align with the chunk
grid; and GPU **reductions** use a tree, which is not an ordered sum.

## Quick start

```bash
make setup     # check the toolchain, report what is missing
make build     # configure and compile
make test      # every gate: unit, convergence, equivalence, MPI, CUDA, style
make sweep     # the benchmark matrix into experiments/results
make reports   # all three PDFs
make all       # everything above, in order
```

Needs GCC 16, CMake 4.4 and Ninja. MPI and CUDA are optional and detected; a
build without either configures cleanly and skips those backends.

### Measured wall clock

Times from an actual `make clean && make all` on the target machine, not
estimates:

| stage | measured |
| --- | --- |
| configure and build, 6 jobs | 55 s |
| full test suite, 10 binaries | 8 s |
| benchmark sweep, 440 configurations | **21 min 05 s** |
| bandwidth refresh | 45 s |
| all three PDFs | 25 s |
| **`make all` end to end** | **about 23 min** |

The sweep is resumable within a commit, so an interrupted run costs only the
configuration it was on. Re-running it unchanged takes **1 min 32 s**, of which
almost all is the per session environment capture: the bandwidth probe sweeps
eight worker counts and the topology probe times a kernel on each of 28 logical
processors. The 440 declared configurations themselves are recognised and skipped
in a few seconds.

Note that a new commit invalidates the whole sweep, because the commit is part of
the resume key. That is deliberate and the reasoning, with the alternative that
was rejected, is in `docs/DESIGN_DECISIONS.md`.

```bash
# One configuration, one result row
./build/pnl --solver cg --backend openmp --size 1023 --workers 20

./build/pnl --list        # solvers and available backends
./build/pnl --topology    # probe and describe the CPU
./build/pnl --bandwidth   # measured STREAM triad, host and device
```

## Measured results

All numbers below are from `experiments/results/summary.csv`, produced by
`make sweep` on the machine described at the end of this file, and regenerated
into the report by script. Nothing here is typed by hand.

### Convergence theory holds to the digit

Iterations to a relative residual of 1e-8 on the 2D Poisson problem:

| method | 63 by 63 | 127 by 127 | 255 by 255 | growth |
| --- | --- | --- | --- | --- |
| richardson | 20,602 | 85,458 | 314,427 | O(n squared) |
| jacobi | 11,255 | 45,880 | 173,278 | O(n squared) |
| gauss seidel forward | 5,192 | 21,383 | 78,516 | O(n squared) |
| gauss seidel red black | 5,293 | 21,938 | 80,906 | O(n squared) |
| block gauss seidel | 2,615 | 10,701 | 39,254 | O(n squared) |
| SOR at optimal omega | 216 | 432 | 868 | **O(n)** |
| red black SOR | 197 | 414 | 815 | **O(n)** |
| conjugate gradient | 197 | 385 | 762 | **O(n)** |

What the tests check against closed form theory, and all pass:

- Jacobi spectral radius matches cos(pi h) to 1e-3 relative.
- Gauss Seidel needs exactly half the iterations of Jacobi, which is Young's
  rho_GS = rho_J squared.
- Young's optimal omega is a genuine minimum: perturbing it either way costs
  iterations.
- Optimal SOR changes the *order* of the work. Doubling the grid multiplies the
  Jacobi count by about four and the SOR count by 2.00.
- Red black ordering costs 2.6 percent more iterations than natural ordering.
  That is the measured price of the parallelism.
- The five point stencil is second order accurate with error constant 0.823
  against the pi squared over 12 = 0.8225 that the truncation analysis predicts.

### Measured bandwidth, and the knee

Measured with a STREAM triad on this machine, never quoted from a specification
sheet:

| workers | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host GiB/s | 42.4 | **62.3** | 61.2 | 60.0 | 56.8 | 56.7 | 54.7 | 37.7 |

RTX 5070: **549.7 GiB/s**.

The scaling knee for a Jacobi sweep sits at **three to five workers**, and beyond
it the slope is slightly negative. The memory system is saturated by four
workers, flat to about twelve, and falls off a cliff once every hyperthread is
engaged.

So for this workload the binding constraint is the memory system, not the
performance versus efficiency core split that the project set out to find. The
machine runs out of memory bandwidth long before it runs out of fast cores. That
is a more useful result than the one it was looking for, and it would have been
invisible to a study that measured only speedup.

The host probe must run on an idle machine: measured during a build it reads
39.8 GiB/s against 62.3 idle, and since every host efficiency figure divides by
it, a depressed reading would inflate all of them. `make all` therefore re-probes
after the sweep rather than before it.

### CPU versus GPU, decomposed

At 4095 by 4095 (16.8 million unknowns, 384 MiB working set), where both devices
are unambiguously streaming:

| method | device | GiB/s | percent of own peak | seconds |
| --- | --- | --- | --- | --- |
| jacobi | host, 20 threads | 17.5 | 28.1 | 6.436 |
| jacobi | RTX 5070 | 515.5 | **93.8** | 0.259 |
| red black gauss seidel | host, 20 threads | 28.5 | 45.7 | 3.951 |
| red black gauss seidel | RTX 5070 | 258.8 | **47.1** | 0.474 |
| conjugate gradient | host, 20 threads | 12.6 | 20.2 | 8.946 |
| conjugate gradient | RTX 5070 | 110.2 | **20.1** | 1.065 |

**Read the efficiency column, not the seconds column.** The efficiency is
dimensionless and says how well each device is used; the seconds are a property
of this particular pair of devices.

The speedup decomposes as `bandwidth ratio x efficiency ratio`, and the measured
numbers bear that out:

- **Red black Gauss Seidel**: efficiencies 45.7 against 47.1 percent, so the two
  devices are used almost identically well. Predicted speedup 8.8 x 1.03 = 9.1;
  measured 8.3.
- **Conjugate gradient**: 20.2 against 20.1 percent, essentially identical.
  Predicted 8.8 x 1.00 = 8.8; measured 8.4.
- **Jacobi**: 28.1 against 93.8 percent, so here the GPU is also used far better.
  Predicted 8.8 x 3.34 = 29; measured 25.

For two of the three methods the entire advantage is the memory system, and the
implementations are equally good. That is the result, and it is the one an
undecomposed speedup figure would have hidden.

The honest one line summary: **on a bandwidth bound stencil sweep this GPU moves
about nine times more data per second than this CPU, and on one of the three
kernels is also used considerably more efficiently.** That is much less exciting
than a bare speedup number, which is why `docs/comparison_methodology.md` exists.

## What is in here

```text
include/pnl/
  core/       types, exceptions, the diagnostics record on every result
  backend/    the interface, seven implementations, topology probing
  solvers/    twelve solvers, one splitting family plus conjugate gradient
  numerics/   roots, quadrature, ODE, LU, QR, Thomas
  problems/   2D Poisson stencil, seeded dense systems
src/          backend implementations, CUDA kernels, the CLI driver
tests/        unit, convergence, equivalence, MPI, CUDA, style
benchmarks/   the declarative sweep matrix and its resumable driver
docs/         backends, solvers, comparison methodology, decisions, log
report/       main report; report_debug/ and report_for_me/ the other two
```

Documentation worth reading in order: `docs/solvers.md` for the family tree,
`docs/backends.md` for the interface contract and what each backend does
underneath, `docs/comparison_methodology.md` before quoting any speedup, and
`docs/DESIGN_DECISIONS.md` for the choices and the alternatives rejected.

`docs/ENGINEERING_LOG.md` records every significant fault with symptom, root
cause, options, fix and verification. Several describe faults that produced
entirely plausible numbers, which are the ones worth reading.

## Reproducing

Every result row carries the problem, solver, backend, worker count, rank count,
pinning, reduction mode, schedule, seed and the commit hash stamped into the
binary. Problems are generated from recorded seeds. The sweep is resumable and
merges atomically, and a configuration that fails or does not apply is recorded
as such rather than dropped.

```bash
git clone <this repository> && cd parallel-numerical-lab
make setup && make all
```

## Environment

| | |
| --- | --- |
| CPU | Intel Core i7-14700K, 8 performance plus 12 efficiency cores, 28 threads |
| GPU | NVIDIA GeForce RTX 5070, 12 GB, sm_120, 48 SMs, driver 610.62 |
| OS | Windows 11 Pro, all work inside WSL2 Ubuntu 26.04 |
| Compiler | GCC 16.0.1, C++23, with GCC 14 as the CUDA host compiler |
| CMake | 4.4.0, Ninja 1.13.2 |
| MPI | OpenMPI 5.0.10 |
| CUDA | 13.3 |

The guest does not expose the hybrid core topology, and thread affinity binds to
a virtual processor the hypervisor may place anywhere. The library measures
rather than assumes, and reports what it could not establish.

## Licence

MIT. Copyright 2026 Olajide Badejo.
