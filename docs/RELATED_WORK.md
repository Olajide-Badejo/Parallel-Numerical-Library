# Related work, and what this project adds that they do not

Deterministic reduction is established technique. Nothing in this repository
invented it, and a reader who came here for that should read the work below
instead. This document names the prior art, says plainly what each piece of it
established, and then states the narrower thing this project adds.

The bibliography of the report carries the same entries with full metadata, at
`report/refs.bib`.

## Reproducible arithmetic

**Intel oneAPI Math Kernel Library, Conditional Numerical Reproducibility.** A
production library that will give bitwise identical results across runs and
across a range of Intel processors, subject to conditions the vendor documents:
a fixed thread count, aligned arrays, and a selected code path. It is the
clearest statement of the shape of the problem, which is that reproducibility is
bought by fixing the things a performance library would otherwise be free to
vary. There is no paper; the developer guide is the reference.

**Demmel and Nguyen, reproducible summation.** Fast reproducible floating point
summation (ARITH, 2013) and parallel reproducible summation (IEEE Transactions
on Computers, 2015) give summation algorithms whose result does not depend on
the order or the number of summands, at a bounded cost and with an error bound.
Ahrens, Demmel and Nguyen (ACM Transactions on Mathematical Software, 2020)
develops the binning algorithm behind **ReproBLAS**, which is a BLAS whose
reductions are reproducible independently of how the work was split. That is a
stronger guarantee than this project makes: ReproBLAS is reproducible across
different decompositions, and the fixed chunk grid here is reproducible only
because the decomposition is held fixed.

## Performance portability layers

**Kokkos** (Edwards, Trott and Sunderland, 2014; Trott and others, 2022) and
**RAJA** (Beckingsale and others, 2019) are the two established C++ layers for
writing a kernel once and running it over several execution models, including
GPUs. Both provide reductions, and both document what is and is not
deterministic about them. They solve the portability problem this library also
solves, at a scale and with a maturity this library does not approach, and they
are the right choice for production work.

## Solver libraries

**Ginkgo** (Anzt and others, 2022) is a modern C++ sparse linear algebra
framework built around linear operators, with GPU backends and a strong
reproducibility and benchmarking culture. **PETSc** (Balay and others, 1997, and
the current users manual) is the reference distributed solver toolkit, with the
Krylov methods and preconditioners this repository implements a small subset of.
**hypre** (Falgout and Yang, 2002) is the reference library for scalable
multigrid preconditioners. Each of the three is a real answer to "what should I
use to solve this system", and this library is not.

## What this project adds

Not the deterministic reduction. The contribution is narrower and, as far as the
work above goes, not covered by it:

1. **One invariant held across seven execution models at once.** Serial, OpenMP,
   POSIX threads, a `std::jthread` pool, MPI, hybrid MPI with threads, and CUDA
   run the same twelve solvers over the same problem interface and produce the
   same values. The comparison across programming models is only meaningful
   because the thing being executed is provably identical, and holding one
   numerical contract across a set of models that includes both a GPU and a
   distributed backend is the part that is unusual.
2. **Asserted as exact equality, in a test suite, rather than described.** The
   equivalence suite compares with `==`, across twelve solvers, two problem
   families, four backends and seven worker counts, and fails on the first bit
   that differs. A one bit fused multiply add discrepancy on the device was
   found that way and would have vanished into any tolerance.
3. **Priced.** What the invariant costs is measured against each model's own
   native reduction and reported with its run to run spread, including where
   that spread is wide enough that the honest answer is that the cost cannot be
   separated from the noise.

The limits are documented rather than hidden: across MPI **rank** counts the
results agree only to reduction tolerance, because rank boundaries are chosen
for load balance and do not align with the chunk grid, and GPU reductions use a
tree, which is not an ordered sum.

## What is not here, and when it will be

**There is no baseline comparison in this report.** Nothing here says what a
competent existing implementation achieves on the same machine on the same
problem, so a statement of the form "this solver reaches 45 percent of host
STREAM" cannot be read as good or bad. That is a real gap and it is stated as
one rather than left for a reader to notice.

A single baseline against **PETSc** is planned for release 1.2.0: the same 2D
Poisson stencil, `KSPCG` with `PCJACOBI` and with `PCSOR`, the same tolerance,
the same machine, the same reported metrics, emitting rows in the same CSV
schema the sweep already uses. One baseline is enough, and the result is worth
publishing whichever way it comes out.
