# Design decisions

The choices that shaped the codebase, with the alternatives that were rejected
and why. Ordered roughly by how much they constrained everything else.

---

## 1. Chunk level callbacks, not element level

**Decision.** `parallel_for(n, body)` calls `body` once per contiguous chunk with
a `Range`, and the body loops over it.

**Rejected.** An element level callback, `body(i)`, which reads more naturally.

**Why.** Two reasons, both about not contaminating the measurement. The
`std::function` indirection is paid once per chunk rather than once per element,
so it is O(workers) per sweep instead of O(unknowns). And the innermost loop
stays a plain loop over contiguous memory that the compiler can vectorise, so
each backend measures its own dispatch and synchronisation cost rather than a
penalty this abstraction imposed on all of them equally.

An element level interface would have made every backend look equally slow, and
the study would have measured `std::function`.

---

## 2. Determinism as a design property, not a hope

**Decision.** The reduction chunk grid is fixed by problem size alone,
`min(512, n)` chunks, never by the worker count. Partials land in fixed slots and
are summed in index order.

**Rejected.** Using each model's native reduction and comparing results with a
tolerance, which is what most libraries do.

**Why.** This is the keystone of the whole project. Because every shared memory
backend produces bit identical iterates at every worker count, the differences
the study measures are attributable to the execution model and to nothing else.
It also turns the equivalence suite into a far sharper instrument: asserting
exact equality fails on the first bit that differs, whereas a tolerance would
absorb real faults silently. The fused multiply add discrepancy (CUDA-02) was one
bit and no reasonable tolerance would have caught it.

**Cost.** Real, and measured rather than excused: the sweep prices the
deterministic mode against each model's native reduction. A production solver
with different priorities could reasonably choose the native mode, and the
library offers it.

---

## 3. Sequential methods keep their semantics, even when that means no speedup

**Decision.** Natural ordering Gauss Seidel, SOR and block Gauss Seidel preserve
exact sequential semantics on every backend, including across MPI ranks through a
pipelined token chain.

**Rejected.** Substituting a red black reordering whenever more than one worker
is present, which would have produced attractive scaling curves for all of them.

**Why.** That substitution would be a lie of a particular kind: not a wrong
number, but a right number for a different algorithm. Doing it honestly means
these methods appear as flat lines in the scaling figures, and that is the
correct result. The reader learns that a sequential recurrence has nothing to
offer a parallel machine, and that red black ordering is the price of getting
parallelism back, measured at 2.6 percent more iterations.

It also improved the device comparison: because the red black penalty is
measured, it can be charged to the CPU and GPU sides equally.

---

## 4. Block count is a solver parameter, never the worker count

**Decision.** Block Jacobi and block Gauss Seidel take an explicit block count,
defaulting to the problem's natural choice.

**Rejected.** One block per worker, which is what a naive parallel implementation
does and what several textbooks describe.

**Why.** Tying the block count to the worker count makes the iterates depend on
how many workers happen to be running, which destroys the equivalence invariant
for two of the twelve solvers. Making it a solver parameter keeps every method
comparable across backends, and blocks are distributed across workers rather than
defined by them.

---

## 5. Two right hand sides for the Poisson problem

**Decision.** `ManufacturedSine` has a closed form solution;
`SpectrallyRich` is a seeded pseudo random source. Each is used where it is
honest.

**Rejected.** A single source. A polynomial manufactured solution such as
x(1-x)y(1-y), whose fourth derivatives vanish so the stencil is exact for it and
the second order accuracy test loses its subject. A sum of a few sine modes,
which only moves the problem since three eigenvectors make conjugate gradient
terminate in three steps.

**Why.** The eigenvectors of the five point stencil are sin(p pi x) sin(q pi y),
so the obvious manufactured solution is exactly one of them, and conjugate
gradient converges in one iteration at every grid size. That is degeneracy, not
speed. The same choice is *ideal* for the stationary methods, because the slowest
decaying Jacobi mode is that same lowest mode, so it isolates the asymptotic rate
and reproduces cos(pi h) to six digits. One source is right for one family and
useless for the other, so both exist. See NUM-01.

---

## 6. Relaxation defaults to zero, meaning "ask the solver"

**Decision.** `SolverOptions::relaxation = 0.0`. Each solver interprets zero as a
request for its own natural factor.

**Rejected.** Defaulting to 1.0, which reads as the identity and is harmless for
SOR.

**Why.** It is not harmless for Richardson: omega = 1 on the Poisson operator
gives iteration matrix I minus A, whose eigenvalues reach minus 7, and it
diverges immediately. Each method's natural factor is different, so the only
default correct for all of them is one that defers. See NUM-02.

---

## 7. Gershgorin, not power iteration, for the Richardson step

**Decision.** When no relaxation factor is given, Richardson uses 1 divided by
the problem's Gershgorin bound.

**Rejected.** Estimating the largest eigenvalue by power iteration, which is more
accurate.

**Why.** More accurate but wrong in the direction that matters. The Rayleigh
quotient approaches lambda_max from below, so an unconverged estimate produces a
step larger than 1/lambda_max, which can leave the convergence interval entirely.
Gershgorin overestimates by construction, so its reciprocal is always admissible.
It is also free, deterministic and identical on every backend, none of which the
power iteration was. See NUM-03.

---

## 8. CUDA behind a C ABI boundary

**Decision.** Every CUDA entry point is `extern "C"` taking plain pointers and
scalars. The `.cu` files are compiled by nvcc driving GCC 14; everything else by
GCC 16.

**Rejected.** Dropping the whole project to GCC 14, which would cost `<mdspan>`
and drop OpenMP from 5.2 to 4.5. Passing `-allow-unsupported-compiler`, which
does not help because the failure is a genuine parse error in a system header.

**Why.** CUDA 13.3 rejects GCC above 15 and cannot parse GCC 15's own libstdc++
headers either. With a C ABI boundary the two compilers never have to agree on a
C++ ABI, only on the platform C ABI, which they do by definition. It also
enforces at the ABI level the rule that backend files never leak their model's
types. See ENV-01.

**Amended 2026-09-05.** Neither cost named under Rejected was ever real. The
header this decision leans on is not included by any translation unit here: the
grep for it and for every other C++23 only construct, quoted under Phase A0.6 in
`PROGRESS.md`, returns nothing across `include`, `src`, `tests` and
`benchmarks`. And `openmp.hpp` asserts `_OPENMP >= 201511`, which is OpenMP 4.5,
so the 5.2 that GCC 16 reports was never a level this project spent. Dropping
below GCC 16 therefore costs nothing at all, and the publication compiler for
release 1.1.0 is GCC 15.2.0, for the reasons in decision 20. What the decision
itself says stands unchanged and is still needed at 15.2.0, because nvcc 13.3
cannot parse GCC 15's own libstdc++ headers either: the C ABI boundary is what
lets the CUDA half and the C++ half be built by different compilers, and after
this phase those are g++-14 and g++-15. Adopting `<mdspan>` stays future work
and is deliberately not done, because it would put the floor back at GCC 15 and
libc++ 18 and kill the gcc-14, gcc-15 and clang-18 matrix Phase B2 asks for.

---

## 9. The CUDA path is not a `Backend`

**Decision.** The device solver is a separate path with its own driver, not an
implementation of the parallelism interface.

**Rejected.** Making CUDA the seventh `Backend`, which would have been tidier
architecturally.

**Why.** The entire point of running on a GPU is that state stays in device
memory for the whole solve. Behind `parallel_for` that would mean a host callback
per chunk, which would measure PCIe latency and nothing else. Tidiness is not
worth a benchmark that measures the wrong thing. The numerics are the same and
are verified against the serial backend.

---

## 10. Contraction disabled on both host and device

**Decision.** `-ffp-contract=off` on the host, `--fmad=false` on the device.

**Rejected.** Comparing SOR to a tolerance instead. Enabling FMA on the device,
which would fix SOR and break Jacobi.

**Why.** The SOR update has the shape `a*b + c*d`, which a compiler may legally
contract. The host was contracting and the device was not, and they disagreed by
one bit. A claim of identical numerics that holds only when the compiler happens
not to contract is a much weaker claim than it looks, and would break silently on
a compiler upgrade. On bandwidth bound kernels the cost is not measurable. See
CUDA-02.

---

## 11. MPI ranks replicate the grid

**Decision.** Each rank allocates the whole grid and computes only its own band.

**Rejected.** Allocating the local slab only, which is what a production code
does.

**Why.** The communication volume is identical either way, since only halo rows
are exchanged, so nothing the study measures changes. What the choice buys is
that the solver code is literally identical across every backend, which is the
point of the whole design. At 4096 squared a vector is 134 MB, which fits the 12
GB guest budget. This is the one place where the project deliberately spends
memory to buy clarity, and it is listed under further work.

---

## 12. Symbol probes, not version macros, for MPI features

**Decision.** The build uses `check_cxx_symbol_exists` rather than testing
`MPI_VERSION`.

**Why.** OpenMPI 5.0.10 advertises MPI 3.1 while shipping many MPI 4.0 entry
points, all of which link. Guarding on the macro would disable functionality that
demonstrably works; assuming the functions exist would break on a stricter
library. Probing is the only approach that is both correct and portable. See
ENV-03.

---

## 13. The core knee comes from the aggregate curve

**Decision.** The performance versus efficiency core knee is located as the two
segment least squares split of the scaling curve. Per processor classification is
attempted, reported, and used only when it demonstrably succeeds.

**Rejected.** Assuming the conventional enumeration, that host CPUs 0 to 15 are
performance core threads and 16 to 27 are efficiency cores, and labelling
accordingly.

**Why.** WSL2 synthesises a homogeneous topology: fourteen uniform cores,
identical cache sizes, no hybrid flag, no cpufreq. And affinity binds to a guest
virtual processor the hypervisor may place anywhere. The assumed mapping is
unverifiable from inside the guest and would produce confident labels with
nothing behind them. The aggregate curve depends only on how many cores are
engaged, not on which, and therefore survives virtualisation. The claim got
weaker and became true. See ENV-04.

---

## 14. A dependency free test framework

**Decision.** About 150 lines of assertion macros in `tests/pnl_test.hpp`.

**Rejected.** Catch2, doctest or GoogleTest.

**Why.** The suite has to run in CI on a plain Ubuntu image and under `mpirun`
with several ranks. A framework would add a fetch step and a set of MPI
interactions to debug, for no benefit at this size. What is actually needed is
named cases, assertions that print both values, a relative comparison, an exact
comparison for the bit identity claims, and an exit status.

---

## 15. Fixed iteration mode alongside solve to tolerance

**Decision.** `RunMode::FixedIterations` runs exactly N sweeps with no
convergence test.

**Why.** Not a convenience. Jacobi on a 1024 squared grid needs of order four
million sweeps to reach 1e-8, which is days. Per iteration cost is what the
bandwidth study needs at those sizes and it is measurable in seconds. Iteration
counts are measured where they are affordable and checked against closed form
rates, which is what makes extrapolating them legitimate. Separating the two
quantities is more rigorous than measuring them together, not less.

---

## 16. The resume key includes the commit, and therefore any commit invalidates the sweep

**Decision.** The resume identity is (problem, solver, backend, unknowns,
workers, pinning, mode, reduction, schedule, block, commit). A new commit means
every configuration is measured again.

**Considered.** Replacing the commit with a digest of only the sources that can
change a result, meaning `include/`, `src/` and `CMakeLists.txt`. That would give
the property one actually wants day to day: a solver change invalidates
measurements and a documentation change does not. As it stands, editing a README
and committing discards an hour of benchmarks.

**Why the commit was kept anyway.** Section 7 of the specification names the
commit explicitly as part of the resume key, and there is a real argument behind
that choice: a digest over a hand picked set of directories is a judgement about
what can affect a measurement, and that judgement can be wrong. A compiler flag
moved into a preset, a change to the sweep driver's timing loop, a new dependency
version, all of these can move a number without touching the hashed set. The
commit is the only key that cannot be too narrow.

The cost is a full re-measurement per commit, which is an hour. The mitigation is
that the sweep is resumable within a commit, so an interrupted run costs nothing,
and that measurements are committed alongside the code that produced them. If
this becomes painful the digest is the change to make, and it should hash the
build configuration as well as the sources.

---

## 17. The dash linter checks compiled PDFs

**Decision.** `check_no_dashes.py` runs `pdftotext` over generated PDFs, and the
LaTeX check covers the `--` and `---` ligatures in prose while ignoring comments,
`\url`, `\verb`, `verbatim` and `lstlisting`.

**Why.** The ground rule covers all file types, and a rule enforced only on
sources would be broken by the first BibTeX page range, which is written `19--26`
by convention and typesets as an en dash. Page ranges in `refs.bib` are therefore
written out as "19 to 26". The linter has its own test, because a linter that
silently stopped detecting anything would let the rule rot while every gate
stayed green.

---

## 18. The build specification stays in history

**Decision.** The V1 build specification stays where it is in the history. The
ignore rules added in Phase A0 stop a future re add, and the past is left
alone. `docs/BUILD_SPECIFICATION.md` is present in the trees of `5faf41d`
through `1e0a7aa` and was removed at `a619b58`, so
`git show cd57032:docs/BUILD_SPECIFICATION.md` still recovers it from any
clone. That remains true after this commit, deliberately.

**Rejected.** `git filter-repo --path docs/BUILD_SPECIFICATION.md
--invert-paths` followed by a force push, which is what the V2 specification
suggests for the case where the file is genuinely not for publication.

**Why.** Rewriting changes every commit hash after `5faf41d`. The `commit`
column of every published result row names a hash, and the `v1.0.0` tag names
a commit; after a rewrite none of those names resolve to the objects they were
written for. Part A exists to repair the provenance of those rows, so scrubbing
the history would destroy the thing being repaired in order to hide a document
that is a build specification, not a credential. A force push additionally
breaks every existing clone and every link into the history from outside. The
file describes how this project was built, which is embarrassing at worst.

The option stays open. The owner can scrub later, accepting the same costs,
and nothing in V2 depends on the file staying. What is not acceptable is
paying those costs silently in the middle of a measurement repair. One
consequence is recorded rather than hidden: the A0 gate line
`git log --all -- docs/BUILD_SPECIFICATION.md` returning nothing is not met,
and `PROGRESS.md` says so under Phase A0.

---

## 19. Release 1.1.0 is Parts A, B, E2, E4 and E5

**Decision.** Release 1.1.0 is Part A, Part B, E2, E4 and E5 of the V2 build
specification. Part C, which is Fortran, Part D, which is assembly, E1, which is
the container, and E3, which is the PETSc baseline, are release 1.2.0. The
reduction accumulator count stays at 1 for 1.1.0, so no committed residual
changes and the release stays bit compatible with 1.0.0; Part D measures four
accumulators as a variant in the `kernel_reduction` block instead of adopting it
as the default.

**Rejected.** One release with one definition of done, covering all five parts.

**Why.** Section 11.5 of the specification prices the whole of V2 at 57 to 72
sessions plus 8 to 12 hours of exclusive machine time, which is five to seven
weeks full time or five to eight months of evenings. A single definition of done
across that span leaves the repository half migrated for months, in a state
where `make all` does not build a report and the published numbers match neither
the old code nor the new. The split gives a release that stands on its own: the
measurements are valid, the numbers reproduce, and the library is installable
for the first time. Holding the accumulator count at 1 is the same argument
applied to the numerics, and is Section 10.4's own recommendation. Changing it
moves every `relative_residual` in `summary.csv` in its last digits and can move
an iteration count that sits on the tolerance boundary, which forces a complete
re measurement and a complete re publication, for a gain the specification
itself pre registers at under 3 percent at 20 workers. The result that makes the
work interesting survives without it, because the finding is that the one kernel
where explicit vectorisation wins is the one kernel where the invariant is at
stake, and stating that requires the variant to be measured, not adopted.

---

## 20. GCC 15.2.0 is the publication compiler, and the standard is C++20

**Decision.** Every number published in release 1.1.0 is produced by GCC 15.2.0,
with gfortran 15.2.0 alongside it. Both are stock Ubuntu 26.04 packages,
`g++-15` and `gfortran-15`. The declared standard drops to C++20, which is what
the code has always used, and `pnl_core` carries `cxx_std_20` as a public
compile feature so a consumer inherits the requirement rather than guessing it.
The CUDA host compiler stays `g++-14`.

**Rejected.** GCC 14.3.0, which is already installed as the CUDA host compiler
and would have needed no new package. GCC 16.0.1, the trunk snapshot that
produced the 1.0.0 numbers.

**Why.** A published number is worth what a reader can reproduce, and nobody
outside this machine can install an unreleased trunk snapshot, so GCC 16 was not
a candidate for 1.1.0 whatever its merits. Between the two released options, 15
wins on one concrete ground: `-fdo-concurrent=parallel` and DO CONCURRENT REDUCE
arrived in GCC 15, so the Fortran work of release 1.2.0 needs no second compiler
switch. A switch there would throw away a whole measurement session, because
changing compilers changes every timing number. That is also why this decision
lands before Phase A1 rather than in Part B where the specification first put
it: taken after Phase A8, it would invalidate the measurement session Phase A8
exists to produce.

The floor is lower than the publication compiler, deliberately. The code needs
C++20 and OpenMP 4.5, which is roughly GCC 11 and Clang 14, and that is what
makes the compiler matrix of Phase B2 and the platform work of Phase B5 nearly
free. Nominating a publication compiler says which binary produced the figures;
it does not raise what a user needs to build the library.
