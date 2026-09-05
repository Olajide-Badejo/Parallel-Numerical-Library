# Engineering log

Dated entries, written as problems happen. Each carries symptom, root cause,
options considered, the fix and why it was chosen, and how it was verified.
Phase 8 converts this into `report_debug/debug_report.pdf`, grouped by theme.

---

## 2026-08-02 ENV-01 nvcc rejects the project compiler, and rejects its own default too

**Symptom.** `cmake -S . -B build` failed inside `project(... CUDA)` with a
compiler identification error. Running `nvcc` by hand on a trivial kernel with
the system default host compiler failed differently, inside a system header:

```text
/usr/include/x86_64-linux-gnu/c++/15/bits/c++config.h(586): error: expected a "("
      if consteval { return true; } else { return false; }
```

**Root cause.** Two separate problems that presented as one.

1. CUDA 13.3 supports GCC up to 15: `crt/host_config.h` line 137 refuses
   anything newer, and this project uses GCC 16 for the host code.
2. Even at GCC 15, which nvcc claims to support, libstdc++ 15 uses the C++23
   `if consteval` statement in `c++config.h`, and the EDG frontend inside nvcc
   cannot parse it. So the nominally supported combination is also broken.

**Options.**

- Drop the host code to GCC 14 throughout. Rejected: it costs `<mdspan>` and
  drops the OpenMP level from 5.2 to 4.5, both of which this project uses.
- Pass `-allow-unsupported-compiler` and hope. Rejected: the failure is a real
  parse failure in a system header, not a version check being cautious.
- Compile only the `.cu` files with an older host compiler and keep everything
  else on GCC 16, with a C ABI boundary between them. Chosen.

**Fix.** `nvcc -ccbin g++-14 -std=c++20`, and every CUDA entry point is declared
`extern "C"` taking plain pointers and scalars. No libstdc++ type crosses the
boundary, so the two compilers never have to agree on a C++ ABI, only on the
platform C ABI, which they do by definition. This also satisfies the Section 11
rule that backend files never leak their model's types through the interface.

**Verification.** A kernel built this way links against GCC 16 host code that
uses `<mdspan>` and runs on the GPU, returning the expected values.

**Correction, 2026-09-05.** Two claims in this entry were wrong when it was
written, and are corrected here rather than edited out. The first option,
dropping the host code to GCC 14, costs neither of the two things listed against
it: no translation unit in this project includes the header named there, then or
now, and the drop from OpenMP 5.2 to 4.5 costs nothing, because `openmp.hpp`
asserts 4.5 and the backend uses nothing above it. The Verification paragraph is
wrong for the same reason. What the kernel actually linked against was host code
using `std::jthread` and `std::barrier`, which are C++20. The conclusion the
entry reached is unaffected, because the second root cause is the one that
decides it: nvcc 13.3 cannot parse GCC 15's `c++config.h` either, so a separate
CUDA host compiler is required at any host GCC of 15 or newer, and the C ABI
boundary is still the fix. Phase A0.6 nominates GCC 15.2.0 as the publication
compiler on exactly that basis, and the host pair is now g++-14 under nvcc and
g++-15 everywhere else. See decisions 8 and 20 in `docs/DESIGN_DECISIONS.md`.

---

## 2026-08-02 ENV-02 CMAKE_CUDA_HOST_COMPILER has to be set before project()

**Symptom.** After fixing ENV-01, configuration still failed the same way even
though `set(CMAKE_CUDA_HOST_COMPILER g++-14)` was in `CMakeLists.txt`.

**Root cause.** The line sat after `project(... LANGUAGES CXX CUDA)`. Compiler
identification runs inside `project()`, so the setting arrived too late and the
identification step used the default host compiler, which nvcc rejects.

**Fix.** Set it as a cache variable before `project()`. The comment in
`CMakeLists.txt` says so explicitly so it cannot be tidied back down.

**Verification.** Configure and build succeed in a path that also contains
spaces, which was briefly suspected of being the cause and was not.

---

## 2026-08-02 ENV-03 OpenMPI 5.0.10 advertises MPI 3.1 while implementing MPI 4 entry points

**Symptom.** The specification names MPI 5.0 as the reference document and asks
which of its features the installed library actually implements.
`MPI_Get_version` returns 3.1 and the `MPI_VERSION` compile time macro is 3.

**Root cause.** OpenMPI 5.0.x does not claim conformance to MPI 4.0 as a whole,
so it leaves the version macro at 3.1, but it ships many MPI 4.0 entry points.
Link probes confirmed all of these resolve: `MPI_Session_init`,
`MPI_Comm_create_from_group`, `MPI_Isendrecv`, `MPI_Allreduce_init`,
`MPI_Info_create_env`, `MPI_Barrier_init`, `MPI_Comm_idup_with_info`.

**Consequence.** Guarding a feature on `MPI_VERSION >= 4` would disable working
functionality; assuming it is present would break on a stricter library.

**Fix.** The build probes for symbols with `check_cxx_symbol_exists` rather than
testing the version macro, and defines `PNL_HAVE_MPI_ISENDRECV` when the probe
succeeds. The core communication path uses only MPI 3.1 guaranteed API so the
code is portable; the probed extras are used where they measurably help.

**Verification.** Configure log records the probe result; the halo exchange has
both paths and the tests pass on either.

---

## 2026-08-02 ENV-04 WSL2 hides the hybrid core topology entirely

**Symptom.** Objective 3 asks for the performance versus efficiency core knee on
an i7-14700K, which has 8 P cores with two threads each and 12 E cores. Inside
the guest, `lscpu` reports 14 uniform cores of two threads, every logical
processor reports the same 2048K L2 and `cpu_capacity` of 1024, the `hybrid`
flag is absent from `/proc/cpuinfo`, and `/sys/.../cpufreq` does not exist.

**Root cause.** WSL2 is a Hyper-V virtual machine and synthesises a homogeneous
topology. Separately, `sched_setaffinity` binds a thread to a guest virtual
processor; the hypervisor remains free to place that virtual processor on any
host logical processor, so pinning is a hint about the guest, not a guarantee
about the silicon.

**Options.**

- Assume the conventional enumeration, that host CPUs 0 to 15 are P core threads
  and 16 to 27 are E cores, and label accordingly. Rejected: unverifiable from
  inside the guest, and would produce confident labels with nothing behind them.
- Reboot to bare metal Linux. Out of scope, and the specification fixes WSL2 as
  the environment.
- Measure instead of asking, and take the knee from a measurement that survives
  virtualisation. Chosen.

**Fix.** Two parts. `probe_topology()` times an identical compute kernel on each
logical processor and only reports a performance versus efficiency split when
the throughputs separate into two groups by a margin that dominates the within
group spread; otherwise it says so and the labels are not used. The knee that
the report leads with is taken from the aggregate scaling curve, which depends
only on how many cores are engaged and not on which, and is therefore valid
under virtualisation.

**Verification.** Recorded in Phase 6 with the probe output beside the scaling
curve, so a reader can see both what could and what could not be established.

---

## 2026-08-02 ENV-05 The CUDA host compiler's libstdc++ hijacks the link

**Symptom.** Enabling the CUDA backend broke a link that had worked for weeks,
with undefined references to `std::__detail::__wait_impl`,
`std::__detail::__notify_impl` and `std::__detail::__wait_args::_M_setup_proxy_wait`,
all reached from `JthreadBackend`. Nothing in the CUDA code uses threads, and
the failing objects were compiled by GCC 16, not by nvcc.

**Root cause.** Those symbols are the out of line half of GCC 16's atomic wait
and notify support, which `std::jthread` and `std::barrier` need. They live in
GCC 16's libstdc++. CMake detects the CUDA implicit link directories by asking
nvcc, and nvcc is driving g++-14, so the list contained
`/usr/lib/gcc/x86_64-linux-gnu/14`. That `-L` landed on the link line ahead of
the search path g++-16 adds for itself, so `-lstdc++` resolved to GCC 14's copy,
which has none of those symbols.

Two wrong turns before finding it. The first guess was that spaces in the
project path were to blame, which a controlled test disproved. The second was
that the link was being driven by the wrong compiler, which `LINKER_LANGUAGE
CXX` did not fix because g++-16 was already driving it; the problem was the
search order, not the driver.

**Fix.** Filter every version specific GCC directory out of
`CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES`. That is safe exactly because the link is
driven by the C++ compiler, which contributes its own. The configure step prints
each directory it drops, so the filtering is visible rather than silent.

**Verification.** Inspecting the generated link line confirms the stray `-L` is
gone; the full suite links and passes with CUDA enabled.

---

## 2026-08-02 CUDA-01 Kernels in a header become duplicate device stubs

**Symptom.** After the link directory fix, a new link failure: multiple
definition of `__device_stub__ZN8pnl_cuda11xpby_kernelEPKddPdii`, reported
against generated files named `tmpxft_00280375_00000000-6_stream_probe.cudafe1.cpp`.

**Root cause.** The shared kernels were defined in `cuda_common.cuh`. A
`__global__` function in a header is emitted into every translation unit that
includes it, along with its host side device stub, so two `.cu` files including
the header produced two definitions of each. The error names a mangled stub in
a generated file and gives no hint that a header is responsible.

**Fix.** `cuda_common.cuh` now holds error handling, launch geometry and
declarations only. Each kernel is defined in exactly one `.cu`, and anything
needed across files goes through a plain launcher function. The header says so,
so it does not get undone.

---

## 2026-08-02 CUDA-02 The host was contracting to FMA and the device was not

**Symptom.** GPU Jacobi and GPU red black Gauss Seidel came out bit identical to
the CPU, but GPU red black SOR differed by 1.6e-18, one bit in the last place.

**Root cause.** The `.cu` files are compiled with `--fmad=false`, so the device
evaluates a multiply and an add separately. The host had no such restriction,
and the SOR update has the shape `a*b + c*d`, which GCC is free to contract into
a fused multiply add. It did. Jacobi has no such shape, which is why only SOR
disagreed and why the discrepancy looked mysterious rather than systematic.

**Options.**

- Compare SOR to a tolerance and note it. Rejected: the central claim of this
  library is that identical numerics run over every execution model, and
  "identical unless the compiler decided to contract" is a much weaker claim
  that happens to be invisible in the test output.
- Enable FMA on the device. Rejected: it would fix SOR and break Jacobi, since
  the two targets would still contract by different rules.
- Turn contraction off on both sides. Chosen.

**Fix.** `-ffp-contract=off` on the host, alongside `--fmad=false` on the device.
The cost is nil in practice: these kernels are bandwidth bound, and the measured
sweep rates were unchanged. The gain is that results no longer depend on whether
a particular compiler on a particular target felt like contracting, which also
makes them stable across compiler upgrades.

**Verification.** All three GPU sweeps are now bit identical to the CPU, and the
device and host red black runs agree on iteration count exactly (21938 each).

---

## 2026-08-02 CUDA-03 A device to device copy given a host pointer

**Symptom.** Every device solve failed immediately with
`cudaMemcpy(d_work, x, bytes, cudaMemcpyDeviceToDevice) failed: invalid argument`.

**Root cause.** A plain slip: `x` is the caller's host array, and the source of
that copy should have been `d_x`, the device copy made two lines earlier. The
CUDA runtime caught it and said so precisely, which is why this cost minutes
rather than hours; recording it because the fix is one character and the class
of mistake is common at a C ABI boundary where pointers carry no indication of
which address space they belong to.

**Fix.** Copy from `d_x`.

---

## 2026-08-02 NUM-01 The manufactured solution is an eigenvector, and conjugate gradient exploits it

**Symptom.** The test asserting that conjugate gradient needs O(n) iterations on
the model problem failed: the count did not grow at all between a 31 by 31 and a
63 by 63 grid. Measured directly, CG converged in exactly one iteration at every
grid size.

**Root cause.** Not a bug in CG. The eigenvectors of the five point stencil are
sin(p pi x) sin(q pi y), and the manufactured solution u = sin(pi x) sin(pi y)
is exactly the lowest of them. With a zero initial guess the residual is a
single eigenvector, so the Krylov subspace CG builds is one dimensional and the
first step is exact. The problem was degenerate, not the method fast.

Worth noting because it cuts the other way for the stationary methods: the
slowest decaying mode of the Jacobi iteration matrix is that same lowest mode,
so a single mode source isolates the asymptotic rate cleanly and reproduces the
closed form spectral radius to six digits. The same choice is ideal for one
family and useless for the other.

**Options.**

- Change the manufactured solution to a polynomial such as x(1-x)y(1-y).
  Rejected: its fourth derivatives vanish, so the five point stencil becomes
  exact for it and the second order accuracy test loses its subject.
- Use a sum of a few sine modes. Rejected: it only moves the problem, since a
  sum of three eigenvectors makes CG terminate in three steps.
- Offer both right hand sides and use each where it is honest. Chosen.

**Fix.** `PoissonRhs::ManufacturedSine` keeps the closed form solution and is
used by the discretisation error tests and the stationary rate tests.
`PoissonRhs::SpectrallyRich` builds a seeded pseudo random source that excites
the whole spectrum, and is used for anything involving CG and for the benchmark
sweep, where a degenerate spectrum would flatter one method over the others.

**Verification.** With the rich source CG grows from 41 to 93 iterations as the
grid doubles, consistent with the O(n) bound from a condition number of
O(h^-2). A dedicated test now asserts the one iteration behaviour on the single
mode source, so the trap is recorded rather than merely avoided.

---

## 2026-08-02 NUM-02 Richardson diverged from a default that was right for the other solvers

**Symptom.** Richardson reported `diverged` on both the 4 by 4 dense system and
the 15 by 15 Poisson problem, while its own dedicated convergence test passed.

**Root cause.** `SolverOptions::relaxation` defaulted to 1.0. The tests that
passed set the factor explicitly to 0, meaning "ask the solver"; the tests that
failed left the default. Richardson with omega = 1 on the Poisson operator has
iteration matrix I - A, whose eigenvalues reach 1 - 8 = -7, so the spectral
radius is 7 and it diverges immediately. The default was chosen thinking of SOR,
where 1 means Gauss Seidel and is harmless.

**Fix.** The default is now 0, meaning each solver picks its own: Young's closed
form optimum for SOR, one for SSOR, and the reciprocal of the Gershgorin bound
for Richardson. Zero is the only default that is correct for every member of the
family.

**Verification.** Richardson now converges on both problems and takes about twice
the iterations of Jacobi, which is what a step of 1/8 against Jacobi's effective
1/4 predicts on this operator.

---

## 2026-08-02 NUM-03 Power iteration is the wrong way to pick the Richardson step

**Symptom.** The first Richardson implementation estimated the largest eigenvalue
by power iteration and diverged even when asked to choose its own step.

**Root cause.** Two faults. The Rayleigh quotient approaches lambda_max from
below, so an unconverged estimate yields a step larger than 1 / lambda_max and
can leave the convergence interval. Separately, the starting vector was written
across the whole padded grid including the Dirichlet boundary ring, which
corrupts the operator.

**Fix.** Replaced entirely by `Problem::gershgorin_bound()`. Gershgorin gives an
overestimate of the spectral radius by construction, so 1 / bound is always
strictly inside the interval 0 < omega < 2 / lambda_max. It is exact for the
stencil (8, against a true largest eigenvalue of 4 + 4 cos(pi h)), free to
compute, deterministic, and identical on every backend, which the power
iteration was not.

**Verification.** The convergence suite checks Richardson converges from the
default step and that its iteration count sits in the expected ratio to Jacobi.

---

## 2026-08-02 NUM-04 The attainable residual floor, measured rather than assumed

**Symptom.** The discretisation error tests asked for a relative residual of
1e-13 and stopped converging when the solver they used changed from Gauss
Seidel to optimally relaxed SOR.

**Root cause.** Not a regression. Stationary iterations have a rounding limited
residual floor, and over relaxation raises it because it amplifies rounding
noise. Measured on this operator in double precision:

| grid | forward Gauss Seidel | SOR at optimal omega | conjugate gradient |
| --- | --- | --- | --- |
| 15 by 15 | 4.4e-15 | 9.5e-15 | 8.1e-17 |
| 31 by 31 | 1.7e-14 | 5.6e-14 | 8.6e-17 |
| 63 by 63 | 7.0e-14 | 2.9e-13 | 9.4e-17 |

The requested 1e-13 sat below the floor at 63 by 63, so the iteration could
never report convergence however long it ran.

**Fix.** Those tests now ask for 1e-11, which is three orders above the measured
floor and seven orders below the discretisation error of about 2e-4 that they
are actually measuring. The table above is quoted in the test comment so the
number is not mistaken for a guess.

**Verification.** The second order accuracy test recovers a fitted order of 2.00
across three grid sizes.

---

## 2026-08-02 SWEEP-01 The resumable sweep was not resumable

**Symptom.** Restarting the sweep after an interruption reported "38 already
complete" and then immediately began re-running the first configuration, which
was a four minute Richardson solve that was already recorded.

**Root cause.** The skip test was in the right loop but on the wrong side of the
work. Rows were parsed from the binary's output and only then compared against
the completed set, so every configuration was executed in full and the only
thing skipped was writing the row. The sweep was resumable in the sense that it
did not corrupt the summary, and in no other sense.

The reason it was written that way is that the identity tuple was taken from the
emitted row, which does not exist until the run has happened. Circular, and the
circularity was invisible because the counters still reported plausible numbers.

**Fix.** `Run.predicted_identity()` derives the same tuple from the declared
configuration before anything is launched. The predictions that are not simply
the declared value are worth naming, because they are where this could go wrong
again: the serial and CUDA backends report one worker whatever was requested,
the hybrid backend reports ranks times threads, and the CUDA path reports its
backend as "device". If a prediction is ever wrong the cost is one redundant
run, after which the real row makes the match exact, so the failure mode is
waste rather than a missing measurement.

The commit is now probed from the binary with a four point solve rather than
read from the first existing row. A stale commit in the resume check would
silently skip configurations whose code had changed since, which is precisely
what the commit column exists to prevent.

**Verification.** Re-running the `convergence_counts` block now reports 36
skipped in under a second, against roughly ten minutes of recomputation before.

---

## 2026-08-02 SWEEP-03 Two sweep blocks collided in the resume identity, and rows vanished

**Symptom.** The generated device comparison table was missing every host row
except the largest size. The sweep had reported success with no failures.

**Root cause.** The identity tuple used for the resume check did not include the
block. The device comparison block and the backend cost block both run Jacobi on
OpenMP at twenty workers in fixed mode with the same pinning, reduction and
schedule; they differ only in running 300 sweeps against 200, and the iteration
count is not part of the identity. Backend cost runs first, so its row made the
device comparison configuration look already complete, and it was skipped.

The same collision quietly removed the twenty worker point from the scaling
curve and the unpinned points from the pinning block. None of this showed up as
an error: the sweep reported those configurations as already present, which is
exactly what a working resume looks like.

This is the second fault in the same mechanism, after SWEEP-01, and the pair are
instructive together. The first made the resume do too little work; the second
made it do too little work in a way that removed data. Both reported plausible
counts throughout.

**Options.**

- Add the iteration count to the identity. Rejected: it fixes this instance and
  not the general problem, since two blocks could differ in any field.
- Add the block name to the identity. Chosen. Blocks are distinct experiments by
  definition, so two rows from different blocks are different measurements even
  when every other field agrees.

**Fix.** `label` joins the identity fields, normalised to its first token because
the device path appends timing detail that would never match on a rerun. Re-running
the sweep then filled exactly the missing configurations and left the rest alone.

**Verification.** The device comparison table now carries host and device rows at
all three sizes, and the scaling curve has its twenty worker point back.

---

## 2026-08-02 SWEEP-04 Efficiency above one hundred percent, and what it was telling me

**Symptom.** The generated device comparison table reported the GPU achieving
149 percent of its own measured STREAM triad bandwidth at the smallest size, and
108 percent at the middle one. An efficiency above one is impossible under the
roofline model the report is built on.

**Root cause.** Not an arithmetic error. The efficiency ratio divides achieved
bandwidth, computed as unknowns times bytes per unknown divided by time, by the
device's DRAM triad figure. That is only meaningful when the sweep actually
streams from DRAM. At 1023 squared the three vectors total 24 MiB, which fits
inside the L2 of an RTX 5070, so the kernel is largely served from cache and
genuinely moves less DRAM traffic than the model assumes. The number above one
is the model breaking, and it was correctly reporting that it had broken.

Worth noting what the same figures say once read properly: at the largest size,
where the working set is 384 MiB and both devices are unambiguously streaming,
the GPU reaches 97.8 percent of its own triad on a Jacobi sweep. That is not
suspicious, it is expected: a perfectly cached five point stencil moves exactly
24 bytes per unknown, and so does a STREAM triad element, so the two kernels have
the same ratio and a good implementation should approach the same bandwidth.

**Fix.** The table now carries the working set size per row and marks any row
whose working set is below that device's last level cache as "cache resident"
rather than printing a percentage. The efficiency figure is a streaming
efficiency and is only shown where streaming is what is happening. The
efficiency figure additionally restricts itself to sizes where both devices are
streaming, and the caption says why.

**Verification.** No row reports above one hundred percent; the cache resident
rows are visibly labelled as such; and the comparison the report leads with is
taken from the largest size.

---

## 2026-08-02 SWEEP-02 Two O(n^2) methods declared in an O(n) block

**Symptom.** Caught by arithmetic rather than by waiting: the
`convergence_counts_large` block listed `ssor` and `block_gauss_seidel` at grid
sizes 511 and 1023, on the reasoning that they were the "fast" methods.

**Root cause.** They are not. Symmetric Gauss Seidel at a relaxation factor of
one is 1 minus O(h^2), and line Gauss Seidel likewise: both need of order n^2
iterations. At 1023 that is about a million sweeps over a million unknowns, some
hours per configuration, for a number the closed form already predicts. Only
optimally relaxed SOR, its red black form, and conjugate gradient are genuinely
O(n) here.

**Fix.** The block now lists exactly those three, with a comment giving the
reason so the list is not helpfully extended later. A per configuration timeout
was also added to the sweep driver, so a future misdeclaration costs fifteen
minutes rather than a night.

**Verification.** The block completes in the expected time and the growth order
it demonstrates matches the closed form.

---

## 2026-08-02 CONC-01 A destructor ordering hazard in the jthread pool

**Symptom.** Found by inspection before it could bite, during review of the
shutdown path.

**Root cause.** Workers wait on a barrier and, once released, read an atomic
`stopping_` flag before returning. `std::jthread` joins in its own destructor,
and members are destroyed in reverse declaration order, which placed `stopping_`
after `threads_`. A worker still reading the flag while it was being destroyed
would be a use after free, in a window that would almost never reproduce.

**Fix.** The destructor body now joins every worker explicitly after releasing
the barrier, before any member can be destroyed. Relying on the jthread
destructor was tidier to read and wrong.

**Verification.** The equivalence suite constructs and destroys the pool
hundreds of times across worker counts under the normal build; the shutdown path
is exercised on every one.

---

## 2026-08-02 BUILD-01 CMake scans for C++20 modules that do not exist

**Symptom.** Every compile line carried `-fmodules-ts -fmodule-mapper=...` and
`-fdeps-format=p1689r5`, and the first build failed inside that machinery.

**Root cause.** From C++20 onward CMake scans each translation unit for module
dependencies by default. This project is entirely header based and declares no
modules, so the scan is pure cost, and it routes GCC through its experimental
modules path.

**Fix.** `set(CMAKE_CXX_SCAN_FOR_MODULES OFF)`.

**Verification.** Compile lines are clean and the build works.

---

## 2026-08-02 STYLE-01 The specification itself contained an em dash

**Symptom.** The dash checker's first run over the tree reported one violation,
in the build specification, at the sentence asking for the personal report to be
thorough.

**Root cause.** The document that sets the no dash ground rule contains one em
dash in its own prose.

**Fix.** The specification is preserved as `docs/BUILD_SPECIFICATION.md` for
provenance, with that one character replaced by a comma. Nothing else changed.
Recorded here rather than fixed silently, because the input to the build is part
of the record.

**Verification.** `check_no_dashes.py` reports the tree clean.

---

## 2026-09-05 PROV-01 Every published result row is stamped dirty

**Symptom.** `experiments/results/summary.csv` holds 850 rows across two
commits and every one of them carries a `.dirty` stamp in the `commit` column:

```text
    425 4abf914a7ea2.dirty
    425 cd57032941a8.dirty
```

So the exact source that produced every published number does not exist as a
commit in this repository, and the stamp is not a one off. `make clean`
followed by `make build` reproduces it deterministically, because the three
mechanisms below are standing conditions of the tree rather than accidents of
one session. Ground rule 6 of the V2 specification refuses to measure from a
dirty tree, so the re measurement in Phase A8 could not have satisfied its own
gate while any of them stood.

**Root cause.** Three independent mechanisms, all of which had to go together.

1. `report/main.pdf` and `report_debug/debug_report.pdf` were tracked and at
   the same time matched by their own ignore lines. Once a path is tracked the
   ignore rule is inert, so they behaved as tracked files nobody knew were
   tracked. `make clean` deletes both, at `Makefile` lines 181 and 184, and
   `git status --porcelain` then reports two deleted tracked files. Every build
   that followed a clean was dirty before it compiled anything.
2. The agent instruction files were untracked and unignored. Immediately before
   this commit `git status --porcelain` read:

   ```text
   ?? BOARD.md
   ?? BUILD_SPECIFICATION.md
   ?? CLAUDE.md
   ?? "Parallel Numerical library V2.md"
   ?? tasks/
   ```

   An untracked file dirties the tree exactly as a deleted one does. The root
   copy of the build specification was not covered because the ignore rule was
   anchored at `docs/`.
3. The check in `CMakeLists.txt` ran `git status --porcelain` over the whole
   tree, so a regenerated figure, a rebuilt table or a fresh `summary.csv`
   counted towards the dirtiness of the binary. That makes `make all` non
   idempotent on rule 6's own terms: the first run regenerates the tracked
   report assets and the second run's sweep refuses to start.

**Options.**

- Filter the porcelain output inside CMake so that known outputs do not count.
  Rejected: that is a second copy of the ignore rules, written in a different
  language, in a file nobody opens when they edit `.gitignore`. It would drift
  within a phase.
- Track the generated assets and the measurements as sources, so the tree is
  clean by construction. Rejected: it makes every report build produce a commit
  and it hides exactly the churn rule 6 exists to detect.
- Give `git status` a pathspec, so only the sources which determine the
  binary's behaviour count, and fix the two states that let ordinary files
  dirty the tree. Chosen. A pathspec is a statement about what can change a
  measurement, which is the judgement the stamp is supposed to encode, and it
  sits three lines above the code that reads it.

**Fix.** One commit, four changes. `git rm --cached` untracks the two PDFs; the
published copies under `assets/reports/` stay tracked and they are what the
README links. `BUILD_SPECIFICATION.md` unanchored, `CLAUDE.md`,
`Parallel Numerical library V2.md`, `BOARD.md`, `tasks/` and `.claude/` join
the private working material block of `.gitignore`; `.claude/` had been covered
only by `.git/info/exclude`, which does not travel with a clone. The dirty
check becomes

```cmake
execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=normal --
                        CMakeLists.txt cmake include src tests benchmarks scripts Makefile
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE PNL_GIT_DIRTY
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
```

`cmake/` does not exist yet and is named for the layout of Section 6 of the
specification; a pathspec entry that matches nothing is not an error.

**Verification.** The six paths that used to dirty the tree are each matched by
a rule now, and `git check-ignore -v --no-index` names the line that catches
each one:

```text
.gitignore:69:BUILD_SPECIFICATION.md	BUILD_SPECIFICATION.md
.gitignore:70:CLAUDE.md	CLAUDE.md
.gitignore:71:Parallel Numerical library V2.md	Parallel Numerical library V2.md
.gitignore:72:BOARD.md	BOARD.md
.gitignore:73:tasks/	tasks/PROTOCOL.md
.gitignore:74:.claude/	.claude/settings.json
```

Listing the tracked PDFs prints nothing, so neither is tracked any more, and
both are still matched by their ignore lines so they cannot return as
untracked. `git status --porcelain` is empty after the commit. `make clean`
followed by `make build`, then one run of the solver, prints the twelve
character short hash of this commit in column 27 with no `.dirty` suffix;
`PROGRESS.md` quotes the run. That last pair is the point of the whole phase,
and no amount of care in Phase A8 would have produced it.

---

## 2026-09-05 PROV-02 An excluded directory that no later negation can re include

**Symptom.** Found by inspection while writing the ignore rules for PROV-01,
before Phase A8 could hit it. `.gitignore` excluded `experiments/results/*` and
re included three named files below it. Phase A8 moves the existing 425 row
measurement set into `experiments/results/archive/` and writes
`experiments/results/manifest-<commit>-<timestamp>.json` beside it. Staging
that with `git add -A` would have committed the manifest and silently dropped
all 425 archived rows, and `git status` would not have mentioned them.

**Root cause.** `experiments/results/*` matches the `archive` directory itself,
and git does not descend into an excluded directory. A negation of a path
inside it is therefore never consulted, because the walk stopped one level
above. The rule reads as though it should work, which is what makes it worth an
entry.

The V2 specification says both the archive and the manifest would have been
invisible. Only half of that is true, and the true half is the worse one. The
manifest is a file directly under `experiments/results`, so an explicit
negation does re include it; the directory is the thing that cannot be reached.
Replaying both rule sets against the same two probe paths:

```text
=== what git add would see under the old rules ===
experiments/results/manifest-4abf914a7ea2-20260905.json
=== what git add sees under the new rules ===
experiments/results/archive/summary-4abf914a7ea2.csv
experiments/results/manifest-4abf914a7ea2-20260905.json
```

A run that commits a manifest and no archive is worse than one that commits
neither, because the manifest is the record which says the archive exists.

**Options.**

- Move the archive out from under the excluded directory, to
  `experiments/archive/`. Rejected: it separates a measurement set from the
  summary it was cut from, and Section 6 of the specification puts it under
  `experiments/results/`.
- Re include the directory and then its contents, two lines rather than one.
  Chosen, with the reason written above the block, because the next person to
  tidy `.gitignore` will otherwise delete the line that looks redundant.

**Fix.** `!experiments/results/archive/` immediately before
`!experiments/results/archive/**`, plus `!experiments/results/manifest-*.json`.
The `session_manifest.json` negation is kept rather than dropped as the
specification's listing has it: that file is tracked today, and removing its
negation would put it in exactly the tracked and ignored state PROV-01 had to
undo for the two PDFs. Phase A8a retires the file and removes the line then.

**Verification.** The two probe paths above were created under
`experiments/results/`. Under the new rules `git ls-files --others` lists both;
under the old rules it lists only the manifest. `git check-ignore -v` on both
paths names the negation line that re includes them. The probe files were
removed afterwards and `git status --porcelain` returned to what it was.

---

## 2026-09-05 MEAS-01 Jacobi copies the whole state on the calling thread, every iteration

**Symptom.** Two readings of one defect. The committed `scaling` block has
Jacobi on OpenMP speeding up by 1.822 at seven workers and then falling away,
to 1.453 at 20 and 0.600 at 28, and no worker count moved it further. And
Section 8.3 of the report puts host Jacobi beside device Jacobi under a
normalised efficiency comparison, in the one section written to be hard to
misquote, while the two were not running the same algorithm.

**Root cause.** `SweepFunction` returned `void`, so a sweep had no way to tell
the driver where it had put the new iterate. The only remaining way to say it
was to put the iterate back where the driver already expected it, and
`include/pnl/solvers/jacobi.hpp` did that with

```cpp
problem.jacobi_sweep(backend, x, work);
std::swap_ranges(x.begin(), x.end(), work.begin());
```

`std::swap_ranges` runs on the calling thread. It never sees the backend, so it
is serial whatever `--workers` says, and it moves both arrays: at 4095 squared
that is two state vectors of about 134 MB each, read and written on one core,
against a sweep that reads two arrays and writes one across twenty. Per unknown
the sweep moves 24 bytes and the swap moves another 32, or 48 if the write is
charged a read for ownership, so between a half and two thirds of the traffic
was on a thread no backend could touch. That was the whole gap.

`src/cuda/jacobi_sweep.cu` never paid it. After the kernel it swaps two device
pointers and the next launch reads the other buffer. The host is what was doing
the extra work, and it was doing it inside the comparison.

I am not quoting an Amdahl asymptote for the serial fraction here, and the
specification is right to forbid one. The committed curve peaks at seven
workers and then decreases, while Amdahl's law is monotone in the worker count,
so whatever shaped that curve is not only the copy. Bandwidth saturation and
hyperthread contention are in it too, and at 1023 squared the three vectors
total about 25 MB against a 33 MiB last level cache, so the streaming model does
not even apply at that size. The copy is real, it is large, and the number it is
worth is the measured one below.

**Options.**

- Copy `work` into `result.solution` rather than swapping. Rejected. It is the
  same serial traffic at three quarters of the volume, and it leaves the host
  and the device still running different algorithms.
- Keep a parity counter in `run_stationary` and read the iterate out of
  `result.solution` or `work` according to its low bit. Rejected, and this is
  the option that would have been a defect rather than a missed improvement.
  Parity is a property of the driver's bookkeeping, not of the sweep. Every in
  place method, which is Richardson, all four Gauss Seidel variants, all three
  SOR variants and both block methods, updates the buffer it was handed and has
  nothing to flip, so its parity never advances. A counter would therefore have
  to carry a second fact, "this method does not flip", that nothing in the sweep
  sets. Deciding the final copy on a parity that never moved is how the in place
  solvers would have quietly returned the wrong buffer.
- Return the view that now holds the iterate, so that `SweepFunction` becomes
  `std::function<VectorView(VectorView, VectorView)>`. Chosen. The view carries
  the fact the parity counter had to be told separately: an in place sweep
  returns its first argument, so nothing flips, and Jacobi returns its second,
  so the two buffers trade places. The driver compares data pointers and never
  has to know which method it is driving.

**Fix.** `SweepFunction` returns `VectorView`. `run_stationary` holds `current`,
initially `result.solution`, and `spare`, initially the work buffer, and after
each sweep takes `current` from the returned view, moving the old `current` into
`spare` when the two differ. `Jacobi::solve` sweeps into `work` and returns
`work`; every other sweep returns its first argument. Three places in the driver
had to move with it, and only the first of them is obvious:

1. The residual inside the loop now reads `current`. Left on `result.solution`
   it would have tested the previous iterate on alternate iterations and shifted
   every reported iteration count in the study by one.
2. `problem.synchronise` is applied to `current`, before the copy. The gather
   collects each rank's own rows out of the buffer it is given, so gathering
   `result.solution` while the iterate still sat in the work buffer would have
   collected stale rows and returned an ungathered answer on every rank.
3. The copy into `result.solution` happens once, at the end, and only when
   `current.data()` is not already `result.solution.data()`.

The diff is 8 files, 93 insertions and 11 deletions. It is not a four line
change, and Section 10.2 must not describe it as one where it contrasts K0
against the assembly kernels.

**The dense audit, which found no defect.** `DenseProblem::jacobi_sweep` writes
only `out[i]` for the rows a rank owns, so with `swap_ranges` gone the inactive
buffer's non local rows are stale after a flip, and a dense matrix vector
product reads every row of the iterate rather than a two row halo. That is a
real hazard and it does not fire, for a reason that is not the one the phrase
"the current swap_ranges keeps both buffers consistent" suggests. Every dense
sweep opens with `backend.exchange_halo(x, 0, n_)`, and `MpiBackend::exchange_halo`
with a row stride of zero delegates to `gather_rows`, which is an
`MPI_Allgatherv` with `MPI_IN_PLACE`: every non local entry of the buffer is
overwritten by its owner's current value before a single row is read. `apply`
and `residual` open the same way. Tracing 1.0.0 shows `swap_ranges` was never
the mechanism either. It left the freshly swapped buffer holding the previous
iterate's non local rows, and the next sweep's gather is what corrected them,
exactly as it does now. The invariant was already carried by the exchange, and
the flip does not disturb it, so no code change was needed. What I did change is
the comment: `problem.hpp` now states the distributed contract on
`jacobi_sweep`, that an implementation must make its input consistent at entry
and may not assume the caller left the buffer complete, and `dense_generator.hpp`
says the same above the exchange that does it. The gather looks redundant to a
reader who assumes a complete vector arrives, and deleting it would turn a
correct program into one that computes with stale rows and reports nothing. The
padded Poisson grid is safe for the reason the specification gives: its boundary
ring is the homogeneous Dirichlet data and no sweep writes it, and its halo rows
are refreshed by the exchange at the top of every sweep.

**Verification.** Two things had to be shown: that no iterate moved, and how
much the copy was costing.

Bit identity first, because the speed is worthless without it.
`ctest -L equivalence` passes with `tests/equivalence/test_equivalence.cpp`
unmodified, `git diff --exit-code 35d8a6f -- tests/equivalence/` exiting 0, and
that suite asserts bit identical iterates across every backend and worker count.
It compares the new code against itself, though, so I also built 35d8a6f into a
scratch build tree and ran both binaries over the same 122 configurations:
twelve solvers on the Poisson and dense problems, on serial and OpenMP, run to a
tolerance of 1e-8 and at a fixed 137 iterations, and all twelve at four MPI
ranks. `diff` of the two result sets is empty. Every iteration count, stop
reason and residual agrees, including Jacobi's 11255 iterations to 1e-8 at 63
squared, a count a one iteration shift or a one bit difference would have moved,
and the same solvers declined the same configurations in both. `make test` is
green at 10 of 10, MPI at one, two and four ranks and CUDA included.

Then the cost. Four runs at the published sizes, `--mode fixed --iterations
300`, five repetitions at 1023 squared and three at 4095 squared, on an
otherwise idle machine, before at `35d8a6f686ec` and after at `dce9cf4b25f0`:

| size | workers | before median | after median | before spread | after spread | before over after |
| --- | --- | --- | --- | --- | --- | --- |
| 1023 | 1 | 0.755255 | 0.519628 | 0.045 | 0.095 | 1.4535 |
| 1023 | 20 | 0.237309 | 0.062997 | 0.183 | 0.178 | 3.7670 |
| 4095 | 1 | 15.136418 | 11.197806 | 0.046 | 0.025 | 1.3517 |
| 4095 | 20 | 9.112059 | 4.970442 | 0.004 | 0.010 | 1.8332 |

Medians in seconds. Spread is `(max - min) / median` over the repetitions of
that run.

Speedup at 20 workers over 1 goes from 1.6611 to 2.2529 at 4095 squared, and
from 3.1826 to 8.2485 at 1023 squared. Wall clock at 20 workers improves by
1.8332 at 4095 squared, which lands inside the range Section 10.2 pre registered
for K0, 1.8 to 2.5, at its lower end. The gain is larger where the copy was a
larger share of the work: at one worker the sweep and the copy both run on that
worker and the ratio is 1.3517, while at twenty the sweep is spread across the
workers and the copy is not, so removing it is worth 1.8332.

The achieved bandwidth column, still divided by the declared 24 bytes per
unknown, goes from 12.3402 to 22.6228 GiB/s at 4095 squared and 20 workers.
That figure is not the whole story, and Phase A3a is where it gets its second
denominator. What changed here is that the numerator now measures the sweep
rather than the sweep plus a serial copy.

The two residuals are a free extra check on the identity claim. Both halves of
the table report `relative_residual` 3.207319e-02 at 1023 squared and
3.262973e-02 at 4095 squared after 300 fixed sweeps, unchanged to every digit
the row prints.

---

## 2026-09-05 SWEEP-05 The resume check normalised a column the driver never normalised, so no CUDA row has ever resumed

**Symptom.** Nothing visible, which is the reason it survived a release. The
sweep is resumable, the committed summary holds 24 CUDA rows, and every sweep re
ran all 24 of them and wrote the same numbers back. A dry run against the
committed summary, with `predicted_identity` as 1.0.0 wrote it, reports the two
halves of it:

```text
42 have no stored row at any commit: cuda 12, hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
54 stored rows that no declared configuration predicts: cuda 24, hybrid 30
```

Twelve declared CUDA configurations the harness believes have never been
measured, and 24 stored CUDA rows the harness cannot attribute to any
configuration. They are the same rows, at two commits, seen from both sides.

**Root cause.** One line, `benchmarks/run_sweep.py:172` in 1.0.0:

```python
backend = "device" if self.backend == "cuda" else self.backend
```

The driver does not print `device` in the backend column. `src/main.cpp` writes
the device row through a format string whose fourth field is the literal `cuda`,
and the one `device` argument in that call lands in `reduction`. Every committed
CUDA row reads `...,cuda,1,1,1,none,device,static,...`. Since `backend` is in
`IDENTITY_FIELDS`, a predicted CUDA identity could never equal a stored one, the
resume check missed every time, and a row that never resumes is
indistinguishable from a row that was never measured unless somebody counts.

The mistake is a plausible one. The device path really does normalise fields, and
this was written as though `backend` were among them: it normalises `reduction`,
which the format string fixes at `device`, and `pinning`, which it fixes at
`none`. It does not normalise the backend name, because `cuda` is what the user
types, what the matrix declares and what the report calls it.

**Options.**

- Make the binary print `device` in the backend column, so that the prediction
  becomes true. Rejected. It renames a backend in every published row, in the
  report and in the sweep matrix, to repair a resume check.
- Take `backend` out of `IDENTITY_FIELDS`. Rejected, and it is the worst of the
  three: the same solver on two backends is two measurements, and collapsing
  them is `SWEEP-03` again.
- Predict `cuda`, and normalise exactly the fields the device path prints as a
  fixed value. Chosen, and those fields are now listed in a comment beside the
  code, so the next reader can check the list against the format string.

**Fix.** `predicted_identity` carries `self.backend` unchanged and normalises
`reduction`, `pinning`, `kernels` and `kernel_variant` to `device`. The last two
are new in this commit, and adding a second normalised field beside a first one
that was wrong is how a defect becomes a convention, which is why the repair
lands here rather than in the phase that first needs `kernels`.

`predicted_workers` gained the `fortran_dc_serial` case in the same commit and
for the same reason. That backend arrives in release 1.2.0, on a build whose `do
concurrent` probe comes back negative, and it reports one worker however many
were asked for, exactly as `serial` does. Without the case the prediction would
carry the request while the stored row carried 1, and `workers` is in
`IDENTITY_FIELDS`, so it is this defect wearing a different field name.

**Verification.** The same dry run against the same migrated summary, with only
that line changed:

```text
before   42 have no stored row at any commit: cuda 12, hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
         54 stored rows that no declared configuration predicts: cuda 24, hybrid 30
after    30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
         30 stored rows that no declared configuration predicts: hybrid 30
```

All twelve CUDA configurations now match a stored row and all 24 stored CUDA rows
are attributed. Each is reported as present at `4abf914a7ea2.dirty` rather than
at the current build, because `commit` is in the identity and those rows predate
this binary. That is the commit column doing its job, and it is why the dry run
reports the two cases separately: already present at this commit, which is what a
sweep would skip, and present at another commit, which is what tells a reader
that the identity matched.

**What the same output says about two other things, neither of which is this
phase's to fix.** The fifteen hybrid configurations and thirty hybrid rows that
remain unmatched are finding 4.6: the binary reports `workers` 5 on a hybrid row
of five ranks times four threads while the harness predicts 20, so those rows
never resume either, for the same structural reason. Phase A6 owns that repair
and this phase leaves it alone. The fifteen remaining are `cg` on `dense_dd` at
three sizes on five backends, which is the declared inapplicable skip:
conjugate gradient requires a symmetric operator, the driver exits 3, and the
harness records the configuration in `session.failures` rather than as a row.
That one is working as designed.

---

## 2026-09-05 SWEEP-06 A strict schema check with no repair, and eight columns queued up behind it

**Symptom.** `run_sweep.py` compares the header of the summary it is merging into
against the header the binary emits, with strict list equality, and on any
difference prints "Move it aside rather than mixing schemas" and exits 2. V2 adds
eight columns to that header. Every one of them, on the day it lands, makes `make
sweep` exit 2 against the committed summary and therefore breaks `make all`,
until a full re measurement finishes. Stopping anywhere in between leaves the
repository strictly worse than 1.0.0: it does not build a report.

**Root cause.** The check is right, and it is half a mechanism. Mixing two
schemas in one CSV shifts every field after the first difference and produces a
file that reads without error and means something else, so refusing is correct.
There is no backfill anywhere in the file, though, so the only remedy the message
offers is to throw the measurements away, and a guard with no matching repair
turns every schema change into a re measurement.

**Options.**

- Relax the comparison to a subset check and let `csv.DictWriter` fill the rest.
  Rejected. It writes the new columns empty for the old rows and leaves the file
  carrying two generations of row under one header with nothing to say which is
  which, which is the mixing the check exists to prevent.
- Take the re measurement each time. Rejected. Hours per column, six times over,
  and it makes adding a column expensive enough that columns get added in a
  hurry at the end.
- Keep the strict check and write the missing half. Chosen.

**Fix.** `scripts/migrate_summary.py` adds the columns the binary has gained,
from a table of declared defaults at the top of the file, refuses on a column
that was removed rather than guessing what to do with it, and writes through the
same atomic temporary file and rename that `run_sweep.py` uses. `--migrate` calls
it instead of exiting 2, `make sweep` and `make sweep-force` pass it, and
`--dry-run` does the header check and the resume calculation and stops, so the
state of the summary can be read without starting an hour of measurement.

Then all eight columns landed in one commit rather than eight, per Section 7.
Six of them are printed empty until the phase that fills them, which pandas reads
as NaN. A placeholder that looked like a number would be a measurement nobody
made, and the two that are not empty, `kernels` and `kernel_variant`, are not
measurements: they say which code ran, and for this generation the answer is
known exactly.

**Verification.** The 1.0.0 summary, taken from the commit that published it and
migrated against the current binary:

```text
$ git show ec406a7:experiments/results/summary.csv > /tmp/summary-1.0.0.csv
$ python3 scripts/migrate_summary.py /tmp/summary-1.0.0.csv --header-from build/pnl
migrate_summary: /tmp/summary-1.0.0.csv: added sweeps, passes,
dram_bytes_per_unknown_per_sweep, pinning_status, measured_at, seconds_reps,
kernels, kernel_variant to 850 row(s), 24 of them on the device
```

`tests/sweep/test_migrate_summary.py` holds the properties the migration claims,
on a fixture rather than on the repository's only copy of its measurements: the
columns arrive in the binary's order, every stored byte survives, the declared
defaults land with `device` on the CUDA row, the CRLF line endings the file
already had are kept, a second run writes nothing, and a removed column exits 2
naming the column.

The committed summary was migrated in this commit and
`scripts/gen_report_assets.py` regenerated from it. The eight LaTeX tables and
the twelve PNG figures are byte identical to the ones generated from the
unmigrated file. The six PDF figures are not, and not because of the migration:
two consecutive runs from the same input differ too, and the difference is the
`/CreationDate` matplotlib stamps into every PDF. With that string removed the
files are equal byte for byte, and `pdftotext` output is identical for all six.
No regenerated asset is committed here.
