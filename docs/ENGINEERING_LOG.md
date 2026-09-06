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

---

## 2026-09-05 MEAS-02 Richardson evaluates the residual twice per iteration

**Symptom.** Richardson is timed against Jacobi, Gauss Seidel and conjugate
gradient on the same problem at the same size, and its seconds per iteration are
higher than the method's arithmetic accounts for. At 1023 squared, 200 fixed
iterations on the serial backend, the same run costs 0.639220 s at the driver's
default `check_interval = 1` and 0.295283 s at the sweep matrix's 1000000. A
method whose iterate does not depend on the check interval should not more than
double in cost when the interval changes.

**Root cause.** Two evaluations of the operator per iteration where the method
needs one. `include/pnl/solvers/richardson.hpp` computed
`r_k = b - A x_k` inside its sweep, because the step direction is the residual,
and `include/pnl/solvers/splitting.hpp` computed `b - A x_{k+1}` again at the
end of the same iteration whenever the check fired. At the default interval that
is every iteration, and the second evaluation is at a different iterate, so it
cannot be dropped without changing what the row reports.

Under this project's byte model a residual moves 24 bytes per unknown, reading
the iterate and the right hand side and writing the residual, and the axpy moves
24, reading the residual and the iterate and writing the iterate. The measured
cost was therefore residual plus axpy plus residual, 72 bytes per unknown per
iteration, against a true cost of 48: an overcount of 1.5x, not the 2x that
counting evaluations alone would suggest. Richardson was compared against
methods that pay for one traversal.

The measurement is worse than 1.5x and the reason is instructive. Each
`problem.residual` call also takes the norm of what it wrote, which is a second
traversal of that array, so the pass counts are five against three, or 1.67.
Above that, the old arrangement touched four distinct state vectors per
iteration, the iterate, the right hand side, the sweep's residual and the
driver's own, which at 1023 squared is 33.6 MB against a 33 MiB last level
cache, while the three the new one touches is 25.2 MB and fits. At 511 squared,
where four vectors are 8.4 MB and everything fits, the ratio is 1.695 and
matches the pass count. At 1023 squared it is 2.165.

**Options.**

- Leave it and correct the reported number in the report. Rejected. The row
  would still be measured wrong, and every future reader would have to find the
  correction.
- Thread a residual hint through `run_stationary` so the driver reuses the one
  the sweep computed. Rejected, and not on cost grounds. The hint available is
  the residual at `x_k`, since that is the one the sweep used to take its step,
  while the driver reports the residual at `x_{k+1}`. Accepting it would report
  Richardson one iteration behind every other method, change its iteration
  counts to tolerance, and break the property that
  `splitting.hpp` documents as the entire reason a shared driver exists.
- Specialise Richardson and carry its residual vector across iterations. Chosen.

**Fix.** `Richardson::solve` runs its own loop. It updates
`x_{k+1} = x_k + omega r_k` from the residual it already holds, then evaluates
`r_{k+1} = b - A x_{k+1}` once, uses that norm for the convergence test when the
check is due, and keeps the vector for the next update. One evaluation per
iteration, reported at `x_{k+1}` exactly as every other method reports it.

The convergence test itself is not duplicated. `detail::check_due` and
`detail::apply_check` were factored out of the driver and both paths call them,
so the criterion, the quantity tested and the iterate it is tested at are still
the driver's; only the loop around them is Richardson's. `detail::finalise_reason`
went the same way and conjugate gradient now uses it too, which removed a third
copy of the same four lines.

**Verification.** The iterates do not move. The arithmetic is the same axpy over
the same `r_k` in the same order; the evaluation that produces `r_k` has only
moved from the top of iteration k to the bottom of iteration k - 1, and
`problem.residual` is a pure function of the iterate. Measured, on the serial
backend, before and after:

```text
                          before      after
size 511  ci 1          0.097504   0.056097   relative residual 3.955925e-02 both
size 511  ci 1000000    0.057531   0.061224   relative residual 3.955925e-02 both
size 1023 ci 1          0.639220   0.276260   relative residual 3.940177e-02 both
size 1023 ci 1000000    0.295283   0.274997   relative residual 3.940177e-02 both
```

Seconds are the median of five repetitions at 200 fixed iterations. At the
default check interval the change is 1.738x at 511 squared and 2.314x at 1023
squared. After it, the two check intervals cost the same to within the run to
run spread, which is what one evaluation per iteration means. The 511 squared
pair after the change, 0.056097 against 0.061224, has overlapping minimum to
maximum ranges of 0.054553 to 0.062296 and 0.055092 to 0.067782, so the apparent
ordering there is noise and not an effect.

Run to tolerance, where a single changed bit would move the count:

```text
$ build/pnl --solver richardson --backend serial --size N --mode solve \
      --tolerance 1e-8 --reps 1 --check-interval C

  N   C   iterations   relative residual      before and after
 31   1         6052        9.992941e-09      identical
 31   5         6055        9.920937e-09      identical
 63   1        20602        9.996886e-09      identical
 63   5        20605        9.978835e-09      identical
```

Every count and every residual is the same to all seven printed digits, at a
check interval of five as well as one, and the check interval of five is the
case where the old code evaluated the residual on iterations the new code also
evaluates it on but did not test. `ctest` is green on all eleven tests including
the unmodified equivalence suite. A new unit case asserts that Richardson
performs `iterations + 1` operator applications, and another asserts for all
eleven methods that measure a true residual that the residual the row reports
equals `b - A x` recomputed at the iterate the row returns, which is the
property the rejected hint would have broken.

---

## 2026-09-05 MEAS-03 The symmetric methods were charged one sweep, and the mitigation the code claimed did not exist

**Symptom.** `src/main.cpp` computed `updates = unknowns * iterations`, and
`updates_per_second` and `gib_per_second` are both derived from it. Symmetric
Gauss Seidel and SSOR perform two full sweeps per iteration, so both numbers
were half of what those two methods achieved. `include/pnl/solvers/gauss_seidel.hpp`
said, of exactly this, that "the result rows record sweeps as well as iterations
so the report can compare on equal work". There was no `sweeps` column and
`Diagnostics::evaluations` was set to the iteration count, so the mitigation the
comment described was not in the code.

**Root cause.** One number was being asked to mean two things and was given the
value of neither. A row needs updates per unknown per iteration, which is what
`updates` multiplies by and what a work comparison divides by, and it separately
needs streams over the array per iteration, which is what a traffic model
divides by. For nine of the twelve methods the two agree and the confusion is
invisible. For the red black methods they differ: `coloured_sweep` steps
`j += 2`, so each colour writes half the unknowns and the pair writes each
unknown exactly once, in two strided traversals of the whole array. One sweep of
work, two passes over memory.

That is why the correction is not "give the two sweep methods a 2". The red
black methods were already right, and incrementing their sweep count would have
created a factor of two error where none existed, in the two solvers that carry
the Section 8.3 device comparison. The methods that were wrong are symmetric
Gauss Seidel and SSOR, which really do write every unknown twice.

**Options.**

- One column named `sweeps`, defined as passes. Rejected: it corrects the byte
  model and breaks `updates` for the red black methods.
- One column named `sweeps`, defined as updates. Rejected: it corrects
  `updates` and leaves the byte model with nothing to divide by, which is the
  input phase A3a needs.
- Two columns. Chosen. They were added to the schema in phase A1.5 with the
  other seven and left empty; this phase fills them.

**Fix.** `Diagnostics` gains `sweeps` and `passes`, documented at the field so a
reader of the CSV can follow the definitions without reading a solver.
`Solver::work_unit()` is pure virtual, so a new solver cannot be added without
declaring both, and the device path in `main.cpp` reads the same declaration
through the host solver of the same name rather than keeping a second table.
`run_stationary` records them and charges `sweeps` operator applications per
iteration to `evaluations`. `updates` in `main.cpp` is now
`unknowns * iterations * sweeps`.

The twelve, at ten fixed iterations with the residual checked every iteration:

```text
solver                sweeps  passes   evaluations
richardson                 1       2            11
jacobi                     1       1            21
gauss_seidel_f             1       1            21
gauss_seidel_b             1       1            21
gauss_seidel_s             2       2            31
gauss_seidel_rb            1       2            21
sor                        1       1            21
ssor                       2       2            31
sor_rb                     1       2            21
block_jacobi               1       2            21
block_gauss_seidel         1       1            21
cg                         1       6            11
```

`evaluations` is now the count of operator applications actually performed:
the initial residual, one per sweep, and one for each residual the check
interval asked for. Conjugate gradient reports `iterations + 1`, the initial
residual plus one matrix vector product per iteration, where it used to report
`iterations`; its inner products are not operator applications and are not
counted. Block Jacobi takes two passes because the lagged coupling obliges
`block_sweep` to snapshot the previous iterate, and block Gauss Seidel takes one
because reading the blocks already updated removes the snapshot. Conjugate
gradient takes six: the matrix vector product, two inner products and the three
axpy like updates of x, r and p.

The comment in `gauss_seidel.hpp` now names the two columns and says that the
mitigation did not exist until this phase.

**Verification.** `gauss_seidel_s` and `ssor` report 2 and 2, the four red black
and symmetric rows report `passes` 2, `cg` reports 11 evaluations at ten
iterations, and `gauss_seidel_rb` and `sor_rb` report `sweeps` 1 so their
`updates` is unchanged. A unit case holds the whole table and fails if the
registry gains a solver it does not declare. No iterate moved: `ctest` is green
on all eleven tests, the equivalence suite unmodified.

---

## 2026-09-05 MEAS-04 A relaxation factor recorded on rows that never used one

**Symptom.** Every result row carried an `omega`. A Jacobi row carried 1.906455
at 63 squared, which is Young's optimum for SOR on that grid and is a number
Jacobi has no use for. A conjugate gradient row carried it too.

**Root cause.** `src/main.cpp` asked `Sor::resolve_relaxation` for every row
regardless of which solver ran. Eight of the twelve methods take no relaxation
factor at all, so eight twelfths of the rows recorded a parameter the run never
read. Worse, two of the four that do take one recorded the wrong value:
Richardson uses the reciprocal of the Gershgorin bound, which is 0.125 on the
five point stencil, and SSOR defaults to 1, and both were reported as the SOR
optimum. Only `sor` and `sor_rb` were right, and they were right by coincidence
of asking the class that happens to own their default.

**Options.**

- A boolean on the solver saying whether it uses a factor. Rejected. It would
  empty the column for the eight, which is the visible half of the defect, and
  leave Richardson and SSOR reporting a factor they do not use, which is the
  half that produces a wrong number rather than a spurious one.
- `Solver::relaxation_factor(problem, options)`, returning the factor the solver
  will actually apply or zero when it has none. Chosen. One virtual answers both
  questions, and each solver's `solve` calls it too, so the row and the run
  cannot disagree.

**Fix.** The virtual is on `Solver` and defaults to zero, which is what the
eight inherit. Richardson returns its step, `Sor` returns
`resolve_relaxation`, `SymmetricSor` returns its own default of one and
`SorRedBlack` returns the SOR optimum, which carries over to the red black
ordering unchanged. `main.cpp` prints the value when it is positive and leaves
the field empty otherwise, which pandas reads as NaN. The device path asks the
host solver of the same name, and gets an empty field for `jacobi`,
`gauss_seidel_rb` and `cg`, which matches `src/cuda/jacobi_sweep.cu`, where the
kernel uses omega for `PNL_CUDA_SOR_RB` alone.

**Verification.** At 63 squared, ten fixed iterations, serial: `omega` is empty
for `jacobi`, `gauss_seidel_f`, `gauss_seidel_b`, `gauss_seidel_s`,
`gauss_seidel_rb`, `block_jacobi`, `block_gauss_seidel` and `cg`, reads 1.906455
for `sor` and `sor_rb`, 1.000000 for `ssor`, and 0.125000 for `richardson`. The
last two are the values those methods take and are not the values the column
carried before. A unit case asserts the presence or absence of a factor for all
twelve.

---

## 2026-09-05 MEAS-05 Conjugate gradient reports the recurrence residual, not the true one

**Symptom.** Found while writing the check that every method reports the
residual at the iterate it returns, which is the property phase A2's Richardson
work had to preserve. Eleven of the twelve methods pass it exactly. Conjugate
gradient does not: after ten fixed iterations at 63 squared it reports
7.1757867344208716e-15 while `b - A x` at the solution it returned is
1.0952924615673637e-13.

**Root cause.** Not a defect in the implementation. Conjugate gradient updates
its residual by the recurrence `r_{k+1} = r_k - alpha_k A p_k` rather than
recomputing `b - A x_{k+1}`, which is the standard formulation and is the whole
reason the method costs one matrix vector product per iteration instead of two.
The recurrence and the true residual agree in exact arithmetic and drift apart
as rounding accumulates, and the drift here is at the scale rounding predicts.

**Options.** None taken. Recomputing the true residual would double the cost of
the method and would make its `evaluations` count `2 * iterations + 1`, which
would be a worse comparison, not a better one. Recomputing it once at the end,
for the report only, would make the reported residual disagree with the
convergence test the run actually stopped on.

**Fix.** None. The fact is recorded, the unit case exempts conjugate gradient
from the true residual property and says why at the exemption, and the field
documentation on `Diagnostics::error_estimate` already says the meaning is
documented per routine.

**Verification.** The exemption is one boolean in the test's table, so a future
change that made conjugate gradient recompute the true residual would have to
clear it deliberately. At the tolerances this project uses, 1e-8 in the sweep
and 1e-10 to 1e-12 in the tests, a drift of 1e-13 changes no reported iteration
count, and the convergence suite is green.

---

## 2026-09-05 MEAS-06 The byte model is undercounted, and the triad it is divided by pays the same read for ownership it does not charge

**Symptom.** `Poisson2D::bytes_per_unknown_per_sweep()` returns 24, three
doubles, and every achieved bandwidth in the repository is that figure times a
work unit over a time. The same repository divides those bandwidths by a host
STREAM triad that declares 24 bytes per element. Both numbers are counted the
same way, and if that way is wrong they are wrong together and in the same
direction, which is why the error has never shown up as an inconsistency.

**Root cause.** Neither count charges read for ownership. A store to a cache
line the cache does not hold has to fetch that line from memory before it can
modify it, so an array a pass writes without having read it first costs a read
as well as a write. A Jacobi pass has exactly one such array, the output: it
reads the right hand side and the previous iterate and writes `out`, and `out`
is never read. Under that model the pass moves 32 bytes per unknown, not 24.
`measure_host_triad` is in the same position for the same reason. It is a plain
C++ loop, `ap[i] = bp[i] + q * cp[i]`, so the compiler emits ordinary stores
into an array the loop never reads, and under the same model it moves 32 bytes
per element while reporting against a declared 24. Whether it does is not
asserted anywhere: it is the thing measured.

Section 4.2 of the specification tabulates the two candidates as 56 and 80
bytes. Those totals are per iteration and include the full state copy the
solver driver made with `swap_ranges`, which phase A1 removed: 24 plus 32
conservatively, and 32 plus 48 with read for ownership. With the copy gone a
Jacobi iteration is one pass and the candidates are 24 and 32. The counts in
this repository are derived from the code as it now is and are not the 56 and
80 of that table.

**Options.**

- Correct the model to 32 and reissue every bandwidth figure. Rejected twice
  over. It decides by argument a question that can be measured, and it changes
  the meaning of a published column underneath anyone who has already quoted
  it, which is what ground rule 9 exists to forbid.
- Leave the model at 24 and note the doubt in prose. Rejected. A doubt in prose
  beside a number in a table is not carried by anyone who reads the table.
- Publish both counts in every row and in every table and figure, build the
  instrument that decides between them, and fix the rule that reads the
  instrument before taking the measurement. Chosen.

**Fix.** `Problem` gains `dram_bytes_per_unknown_per_sweep()` beside
`bytes_per_unknown_per_sweep()`. Both are documented as one pass over the
arrays, which is what `Diagnostics::passes` multiplies, and both write their
derivation out array by array: which arrays the pass reads, which it writes, and
which written array is read first and so pays nothing extra. `Poisson2D`
returns 24 and 32. `DenseProblem` returns `(n + 3) * 8` and `(n + 4) * 8`, where
the correction is one double against a term of order n and is 0.2 percent at
n equal to 512, so the dense rows cannot separate the two models. Every result
row carries both, host and device. `gib_per_second` keeps dividing by the
conservative count it has always divided by, and the report derives the second
bandwidth from the second column, labelled `counted` beside `declared`.

The instrument is `measure_host_triad_nontemporal`, a second host probe over the
same arrays, sizes, worker counts and repetitions as the first, differing in the
store instruction and in nothing else: `_mm256_stream_pd` with one `_mm_sfence`
after the loop, guarded on `__AVX__`, with a scalar fallback that reports itself
as the fallback rather than passing an ordinary store off as a streaming one.
`objdump -d build/pnl` shows the `vmovntpd` and the `sfence`. It is reported in
the session manifest as an additional entry beside the plain probe, never as a
replacement, with the ratio in both orientations and with the read for ownership
corrected figure for the plain triad, `plain * 32 / 24`, filed under
`bandwidth.derived` and labelled as arithmetic rather than as a measurement.

That corrected figure is not corroborated against a theoretical peak and cannot
be. No memory speed is recorded anywhere in this repository, the environment
table names the CPU and not the memory, and `dmidecode` is not installed in the
WSL guest the measurements run in, so `dmidecode -t memory` cannot supply one
either. `docs/comparison_methodology.md` says so in those words.

**The question is open, and this entry does not close it.** The selection rule
is fixed in `benchmarks/sweep_matrix.yaml` under `preregistered.traffic_model`
before any measurement, together with the sentence the report carries under each
of its three outcomes: a ratio of the non temporal triad to the plain triad
above 1.20 selects the read for ownership model, below 1.10 selects the
conservative one, and from 1.10 to 1.20 is recorded as unresolved with both
carried. Phase A8b applies that rule to the publication session's measurement
and records the outcome as `ASM-01`; phase D4 of release 1.2.0 rebuilds the same
triad in assembly and confirms it rather than gating it.

**Verification.** At 63 squared, ten fixed iterations, serial backend, the
Jacobi row reads `bytes_per_unknown` 24.0 and
`dram_bytes_per_unknown_per_sweep` 32.0. `gauss_seidel_rb` and `sor_rb` read
24.0 and 32.0 with `passes` 2, so an iteration of those methods moves twice
either figure. A dense row at n equal to 512 reads 4120.0 and 4128.0. The device
row carries the same two figures, since the device kernels move the same three
arrays and the device path declares no second count of its own. The eleven tests
pass, `tests/equivalence/` is untouched, and no iterate moves: nothing in this
phase is on a numerical path.

---

## 2026-09-05 MEAS-07 The triad probe's run to run spread is wider than the decision band the traffic model is selected on

**Symptom.** Found immediately after building the instrument of `MEAS-06`, by
running it three times on the same quiet machine with nothing else scheduled.
The selection statistic, the non temporal triad over the plain triad, read
1.0990, then 1.0121, then 1.0601. The pre registered undecided band is 1.10 to
1.20 and is 0.10 wide. The spread of the statistic over three runs is 0.087,
which is that band over again.

The underlying figures move as much. The plain triad's best over the worker
sweep read 61.205, 69.083 and 61.670 GiB/s, and at eight workers specifically it
read 61.2, 69.1 and 61.7, a spread of thirteen percent.
`refresh_bandwidth` in `benchmarks/run_sweep.py` already documents this probe as
load sensitive, 55 GiB/s idle against 39.8 while a build ran, and the committed
session manifest carries 61.35. What these runs add is that the sensitivity
survives a quiet machine.

**Root cause.** Two contributions, and they are separable.

The first is ordinary run to run variance in a memory bound probe on a WSL2
guest, where the Windows host is scheduling underneath it. The probe keeps the
best of five repetitions at each worker count, which suppresses variance within
a run and says nothing about variance between runs, and nothing anywhere
records a spread for it. This is ground rule 7 pointed at the denominator
instead of at the numerator: an effect smaller than its own spread is being
asked to decide something.

The second is that the pre registered statistic is a ratio of two bests over the
worker sweep, and the two bests need not come from the same worker count. On the
first run the plain probe peaked at eight workers and the non temporal probe at
sixteen, so the ratio compared two arms that differed in the store instruction
and in the worker count at once. At matched worker counts the same run gives
1.0441 at eight, 1.1198 at sixteen and 1.1085 at twenty, which spans the
threshold on its own. Section 10.3 makes exactly this objection about K1 against
K2 and answers it with a controlled experiment; the same objection applies here
and is not yet answered.

**Options.**

- Amend the pre registered rule now, to a ratio at matched worker counts, or to
  a rule that requires several probe runs and reports the spread of the ratio.
  **Rejected, and the reason is the whole point of the phase.** The rule was
  written before the measurement and the measurement has now been seen. Changing
  the statistic afterwards, when the change is known to move the answer across
  the threshold, is precisely what ground rule 10 forbids, and a phase whose
  deliverable is a pre registration cannot be the phase that edits one after
  looking.
- Strengthen the probe here, by repeating the whole sweep and reporting the
  spread of the ratio. Rejected as another phase's work. Dispersion is phase A7
  and the publication measurement is phase A8b. Widening this phase to cover
  them would put the fix in the same commit as the pre registration it changes
  the meaning of.
- Record the defect, leave the rule exactly as registered, and hand the decision
  to the phase that has to take the measurement. Chosen.

**Fix.** None here, deliberately. The rule in
`benchmarks/sweep_matrix.yaml` stands as written. What this entry adds is the
requirement that **before** phase A8b takes the publication measurement, and
before phase D4 takes the assembly one, a decision is made and written down on
two questions this run raises and does not settle: whether the selection
statistic should be taken at matched worker counts rather than as a ratio of two
bests, and how many probe runs the statistic needs before its spread is narrower
than the band it has to fall inside. Both decisions have to be recorded before
the measurement, which is the same discipline the pre registration itself is
under.

**Verification.** Three `build/pnl --bandwidth --backend openmp` runs on an
otherwise idle machine, quoted in full in `PROGRESS.md` under phase A3a. None is
the publication measurement and all three are labelled as observations. The
instrument itself is sound: `objdump -d build/pnl` shows one `vmovntpd` and the
fence. The non temporal arm is ahead at twenty two of the twenty four worker
points across the three runs and behind at two of them, both in the third run,
at four workers and at twelve. The direction is the one read for ownership
predicts and the exceptions sit inside the spread above. It is the size of the
effect, not its sign, that this instrument cannot yet resolve, and a statistic
whose own spread is the width of the band it must fall inside cannot decide
anything.
## 2026-09-05 MEAS-08 Two pinning policies bound nothing and said they had

**Symptom.** `cpu_for_worker` in `include/pnl/backend/topology.hpp` returned
minus 1 when it could not tell the performance cores from the efficiency ones,
and every caller read that minus 1 exactly as it read the answer for
`Pinning::None`: skip the binding, count no failure, carry on.
`src/backend/pthreads_pool.cpp` line 21 read `if (cpu >= 0 &&
!pin_this_thread(cpu)) ++pinning_failures_;`, and the OpenMP, jthread and MPI
backends each carried their own copy of the same test. On this guest the
classification never succeeds, so `--pinning pcore` and `--pinning ecore` bound
no thread at all and printed a row whose `pinning` column read `pcore`.

The classification fails for the reason the file's own header gives: WSL2 is a
Hyper-V guest and does not pass the heterogeneity of the i7-14700K through. The
probe taken during this phase measured a largest gap in per processor throughput
of 0.003038 against a within group spread of 0.017394, which is one group and
not two.

**The defect is latent, not realised, and the pinning block stands.**
`Pinning::Compact` returns `worker % logical` and `Pinning::Scatter` returns
`topology.core_leaders[worker]`. Both are always non negative, both genuinely
bind, and both are what the committed sweep used.
`experiments/results/summary.csv` holds 850 rows: 802 with `pinning=none`, 24
with `compact`, 24 with `scatter`, and **zero** with `pcore` or `ecore`. The
`pinning` labelled block is 36 rows at each of the two commit generations in the
file, 12 `none`, 12 `compact` and 12 `scatter` across `openmp`, `pthreads` and
`jthread`. Every one of those rows measured what it says it measured. Nothing in
the report is retracted. What this entry is about is the next `--pinning pcore`
anybody types.

**Root cause.** One sentinel value for two answers that call for opposite
responses. "Nobody asked for pinning" and "the classification this policy needs
did not succeed" were both minus 1, so no caller could tell them apart, and the
only behaviour that serves the first of them is silence. `pinning_failures()`
counted a third answer, "the operating system refused", but nothing read the
counter and no column carried it, so even the case that was detected was
invisible.

**Options.**

- Fall back to `compact` when the classification fails. Rejected. The row would
  carry `pcore` while the threads were placed by a different policy, which is
  worse than the defect: a wrong answer wearing the shape of a right one.
- Report the failure in a column and let the row stand. Rejected on its own,
  though the column is part of the fix. A reader who filters on `pinning` and
  never looks at the second column has the original defect back.
- Refuse the run. Chosen. `make_backend` throws `BackendFailure` when `pcore` or
  `ecore` is asked for and the classification did not succeed; every shared
  memory backend throws from its constructor when a worker was refused; and the
  driver refuses to write a row whose `pinning` is not `none` and whose
  `pinning_status` is not `bound`. The three guards overlap deliberately. The
  third is what makes the invariant a property of the CSV rather than of the
  backends that happen to exist today.

**Fix.** `cpu_for_worker` returns a `PinTarget`, which is a `PinOutcome` and a
processor number, and the outcome separates `NotRequested`, `Bound`,
`NotApplicable` and `Refused`. `pin_worker` turns a target into an outcome by
calling `pin_this_thread`, and it is now the only place a binding is attempted,
so the four backends no longer carry four copies of the rule.
`Backend::pinning_status()` joins the interface as a pure virtual returning the
outcome's name, with the refusal count appended after a colon when it is not
zero, and `src/main.cpp` fills the `pinning_status` column from it. `Compact`
and `Scatter` return exactly the processors they returned before, which is what
keeps the committed rows comparable with anything measured later.

`SerialBackend` binds its one thread now instead of ignoring the request, so
`--backend serial --pinning compact` reports `bound` rather than a policy it
never applied. The two distributed backends record what their rank thread did
and do not throw: a rank local throw leaves the other ranks waiting in the next
collective, which is the hang Section 4.7 of the specification records against
`mpi.hpp` and which phase B6 owns. The driver's refusal on the root rank is
their loud failure instead.

**A neighbouring fault, recorded here and deliberately not fixed here.** The
hybrid backend's OpenMP team inherits the affinity mask of the rank thread that
`MpiBackend` binds, so `--backend hybrid` under any policy other than `none`
would confine a whole rank's threads to one logical processor rather than spread
them across the rank's share. No committed hybrid row uses a pinning policy: all
30 carry `pinning=none`, so nothing published is affected. Binding per thread
inside the rank changes what the hybrid backend measures, and that belongs to
the phase that re measures it.

**Verification.** `build/pnl --solver jacobi --backend pthreads --size 63 --mode
fixed --iterations 10 --reps 1 --workers 4 --pinning pcore` exits 1 with
`pnl: backend failure: pinning policy 'pcore' needs a performance core
classification and this machine did not yield one`, and writes no row. The same
configuration under `compact` on pthreads, `scatter` on openmp and `none` on
jthread writes rows whose `pinning_status` reads `bound`, `bound` and
`not_requested`. The header is still 36 columns and so is every row: this phase
fills a column, it does not add one. All four runs are quoted in `PROGRESS.md`
under phase A4.

## 2026-09-05 CONC-02 A counter incremented under the mutex, incremented without it, and read without it

**Symptom.** `PthreadsBackend::pinning_failures_` was a plain `int` with three
accesses and no single rule. The constructor incremented it at
`src/backend/pthreads_pool.cpp:21` with no lock held, each worker incremented it
at `:77` under `mutex_`, and `pinning_failures()` at
`include/pnl/backend/pthreads.hpp:55` read it with no lock at all.

**Root cause.** The constructor's increment runs before `pthread_create`, so it
happens before every worker and races with nothing. The workers' increments are
under the mutex and race with nothing either. The only pair that is a data race
by the letter of the standard is the guarded increment against the unguarded
read, and until this phase nothing anywhere called `pinning_failures()`, so the
read never happened and the race had no second access to be a race with.

This is a defect of consistency, not a bug anybody could have observed. It is
fixed because a counter with three access rules is a trap for whoever touches it
next, and not because a sanitizer would catch it: ThreadSanitizer reports a race
between two accesses that both executed, and there were never two. The
specification says the same thing in its sanitizer section, and phase B2 should
not be waiting for a report that cannot arrive.

**Options.**

- Take the mutex for the constructor's increment too, and for the read.
  Rejected. It makes the reader's rule the mutex as well, and `pinning_status()`
  is called from the driver after the pool is up, where taking a worker dispatch
  mutex to read an int is a lock nobody needs.
- Make it `std::atomic<int>` and drop the mutex for it everywhere. Chosen. One
  rule, stated at the declaration, and a rule that still holds if a later caller
  reads the counter from a worker thread.

**Fix.** `pinning_failures_` is `std::atomic<int>`, incremented with
`fetch_add(1, std::memory_order_relaxed)` in the constructor and in
`worker_loop`, and read with `load(std::memory_order_relaxed)` in
`pinning_failures()`. Relaxed ordering is enough because the counter is only
added to, never used to publish anything, and only read after a happens before
edge that the new pinning handshake already supplies. The mutex is not taken for
it anywhere.

That handshake is new and is the reason the counter is read at all. Each
`pthreads` worker writes its own slot of `pin_outcomes_`, then takes `mutex_`,
increments `pin_reports_` and signals `pin_done_`; the constructor waits until
every spawned worker has reported before it aggregates the outcomes and decides
whether the pool may exist. `pin_reports_` is a plain `int` guarded by `mutex_`
at every access, which is the same one rule applied to a counter for which the
mutex is the right answer. The jthread pool gets the same happens before edge
from a `std::latch` and needs no counter of its own.

Both pools also had to learn to unwind. A constructor that throws leaves no
object, so no destructor runs, and the pool it had already started would be
joined by member destructors that cannot release a worker parked on a barrier.
`PthreadsBackend::stop_workers` and `JthreadBackend::shutdown` are the teardown
the destructor used to do inline, called now from the destructor and from the
failure paths. The `pthread_create` failure path uses the same function, so
there is one teardown rather than three.

**Verification.** `make test` is green, 11 of 11, including `test_equivalence`
and the MPI suite at 1, 2 and 4 ranks. `clang-format --dry-run --Werror` over
`include src tests` exits zero and `ruff check benchmarks scripts tests` passes.
## 2026-09-05 MEAS-09 The hybrid backend plotted its scaling curve against its rank count

**Symptom.** Every hybrid row in `experiments/results/summary.csv` has
`workers == ranks`. All 30 of them read `workers=5, ranks=5,
threads_per_rank=4`, so a configuration running 20 threads across the job was
recorded at 5 workers and compared, on the same axis, against `openmp`,
`pthreads`, `jthread` and `mpi` rows that were recorded at 20.

**Root cause.** `HybridBackend`'s constructor sets `config_.workers = ranks_ *
threads_` and says in a comment that the product is "the number the scaling
curve is plotted against". It then did not override `worker_count()`, which is
inherited from `MpiBackend` and returns `ranks_`. The result row does not read
`config_.workers`: `src/main.cpp` takes the `workers` column from
`execution->worker_count()`. The right number was computed, stored in the field
the row does not read, and documented in a comment that described a behaviour
the class did not have.

**A second defect fell out of the first, and it is the more expensive one.**
`Run.predicted_workers` in `benchmarks/run_sweep.py` has special cased hybrid as
`ranks * threads` since `cd57032`, the commit that added the sweep harness. The
prediction was therefore right about what the binary should report and wrong
about what it did report, so the predicted identity of every hybrid
configuration has never matched the row the sweep itself wrote. The resume check
missed on all 15 of them, on every sweep, from the harness's first run onwards.
That is `SWEEP-05` again in a different column: a configuration that never
resumes and one that was never measured look identical from the outside, which
is why the dry run learned to print the question from both sides. It prints the
answer today, unprompted:

```text
30 stored rows that no declared configuration predicts: hybrid 30
```

**Options.**

- Change the prediction to `ranks`, matching the binary. Rejected. It makes the
  resume check agree at the cost of leaving the axis wrong, and the axis is the
  thing the phase exists to fix.
- Override `worker_count()` and leave the prediction alone. Chosen for the
  number, since the prediction was already the correct one.
- Do both and clamp the prediction the way the binary clamps. Chosen in full.
  `command` launches `max(1, workers // max(1, threads_per_rank))` ranks and the
  backend takes `max(1, threads_per_rank)` threads, so the prediction now
  applies the same clamp to the product. It changes no number for any
  configuration the matrix declares, all of which are 20 workers at 4 threads
  per rank and predict 20 either way; it removes the one input, a zero or
  negative thread count, on which the two could still have disagreed.

**Fix.** `HybridBackend::worker_count()` returns `ranks_ * threads_`.
`predicted_workers` clamps the thread count before multiplying. The comment in
`mpi_runtime.cpp` that described the intended behaviour now describes the actual
one.

**The 30 committed hybrid rows carry the old count and have to be re measured.**
They are not wrong about the run that produced them, which really was 5 ranks of
4 threads, and both other columns say so; they are wrong about the axis the
report plots them on. They cannot be repaired in place, because `workers` is in
`IDENTITY_FIELDS` and editing it would forge an identity no run ever produced.
The dry run reports all 15 hybrid configurations as having no stored row at any
commit, which is the correct answer and is stated in `PROGRESS.md` rather than
papered over. Phase A8b re measures everything anyway; what this entry adds is
that the hybrid rows in the file today must not be carried forward.

**Verification.** `mpirun -np 2 build/pnl --backend hybrid --workers 8
--threads-per-rank 4` writes `workers=8, ranks=2, threads_per_rank=4`, and the
sweep's own shape, 5 ranks at 4 threads, writes `workers=20, ranks=5,
threads_per_rank=4`. `make test` is green at 11 of 11, including `test_mpi` at
1, 2 and 4 ranks. All quoted in `PROGRESS.md` under phase A6.

---

## 2026-09-05 MEAS-10 The timed region allocated on every call, and the state was only the largest part of it

**Symptom.** `src/main.cpp` timed `solver->solve(...)`, and `solve` allocated
every full size vector it needed on entry and freed them on return. At 4095
squared the padded state is 4097 squared doubles, 128.06 MiB, so a Jacobi
repetition mapped and unmapped 384.2 MiB and a conjugate gradient repetition
512.3 MiB, inside the clock, on every repetition. The untimed warm up before the
loop could not help: the memory it faulted in was freed before the first timed
repetition asked for its own.

**The specification's arithmetic is right about the sizes and wrong about which
solver.** Section 7 A5 writes "403 MB across three state vectors, or 537 MB for
Richardson's four". Counted from the code, Richardson allocates two, the iterate
and the residual vector it carries across iterations, which is 269 MB. The four
vector method is conjugate gradient, which needs the iterate, the residual, the
search direction and the operator product live at once, and 537 MB is its
figure. The shared stationary driver allocates three, 403 MB, which is the
larger number's correct owner too.

**Root cause, measured rather than reasoned about.** The gate for this phase,
`tests/unit/test_no_allocation.cpp`, replaces the global `operator new` and arms
a counter at the two ends of the timed region. Built against the tree at
`3faf2fc` and pointed at `Poisson2D` at 63 squared, 12 fixed iterations, check
interval 1, one warm up call and then one counted call, it reported:

```text
grid 3969 squared, 12 fixed iterations, check interval 1
serial     richardson                 45 allocations        69180 bytes
serial     jacobi                     45 allocations       102947 bytes
serial     gauss_seidel_f             57 allocations       103595 bytes
serial     gauss_seidel_b             57 allocations       103595 bytes
serial     gauss_seidel_s             81 allocations       104723 bytes
serial     gauss_seidel_rb            82 allocations       105613 bytes
serial     sor                        59 allocations       103747 bytes
serial     ssor                       83 allocations       104774 bytes
serial     sor_rb                     84 allocations       105673 bytes
serial     block_jacobi               82 allocations       528239 bytes
serial     block_gauss_seidel         71 allocations       122562 bytes
serial     cg                         82 allocations       138136 bytes
openmp     richardson                 45 allocations        69180 bytes
openmp     jacobi                     45 allocations       102947 bytes
openmp     gauss_seidel_f             57 allocations       103595 bytes
openmp     gauss_seidel_b             57 allocations       103595 bytes
openmp     gauss_seidel_s             81 allocations       104723 bytes
openmp     gauss_seidel_rb            82 allocations       105613 bytes
openmp     sor                        59 allocations       103747 bytes
openmp     ssor                       83 allocations       104774 bytes
openmp     sor_rb                     84 allocations       105673 bytes
openmp     block_jacobi              118 allocations       582671 bytes
openmp     block_gauss_seidel         71 allocations       122562 bytes
openmp     cg                         82 allocations       138136 bytes
```

Twelve iterations, and Jacobi allocates forty five times. The state is three of
those forty five. Decomposing the row exactly: three state vectors, three
preconditions in the driver whose message literals are longer than the small
string buffer, one reduction for the right hand side norm, two dispatches for
the initial residual, and three dispatches per iteration for twelve iterations,
which is 3 + 3 + 1 + 2 + 36 = 45. The state is most of the bytes and almost none
of the count, and the count is what runs per iteration. Six separate causes,
none of which the phase's paragraph anticipates:

- **Every `parallel_for`, `reduce` and `run_ordered` allocated.** `RangeBody` was
  `std::function<void(Range)>`, and libstdc++ stores a callable inside a
  `std::function` only when it is trivially copyable and fits in sixteen bytes.
  A sweep body captures the row range, two spans, the right hand side pointer
  and `this`, which is forty bytes, so every dispatch took a heap block and gave
  it back. A Jacobi iteration makes three dispatches, so that is three
  allocations per iteration for a type erasure that outlives nothing: every
  backend here already held a bare pointer to the caller's object while its
  workers ran.
- **Every precondition allocated.** `require` took `const std::string&`, so a
  call site constructed a string from its message literal on the path that
  succeeds. `relaxation_sweep`, `coloured_sweep` and `block_sweep` check theirs
  on every sweep, which is once or twice per iteration, and only the shortest of
  those messages fits the small string buffer.
- **Three solvers built a reason string unconditionally.** `cg` and the two red
  black methods pass `inapplicable_reason(problem)` to `require`, and an
  argument is evaluated whether or not the check fires. Each of those reasons is
  a sentence.
- **Five solvers had a sweep lambda too large to store locally.** `SweepFunction`
  was a `std::function` too. SOR, SSOR and red black SOR capture a relaxation
  factor on top of the two references, and both block methods capture a block
  count, which is twenty four bytes against the sixteen available.
- **Both block sweeps allocated inside the iteration, and that is where the
  bytes are.** A full state snapshot for the lagged coupling, plus a scratch
  vector of 3n per chunk, which under OpenMP is one per chunk per sweep.
  `block_jacobi` on `openmp` counts 118 allocations and 582 kB against
  `jacobi`'s 45 and 103 kB, and 36 of those 73 extra allocations are the four
  chunk scratch vectors of twelve iterations.
- **The progress bar allocated for the longer names.** `ProgressBar` took its
  label as a `std::string` by value and the driver built one from a
  `string_view` on every solve. `block_gauss_seidel` is eighteen characters and
  does not fit the small string buffer, which is what the eleven allocation gap
  between `block_gauss_seidel` at 71 and `gauss_seidel_f` at 57 partly is.

**Options.**

- Hoist the state and stop there, which is what the phase asks for literally.
  Rejected once the counter had run. It removes the bytes and leaves three
  allocations per iteration, so the gate the phase also asks for, zero
  allocations between the two ends of the timed region, would not have passed
  and there would have been nothing to hold the repair in place.
- Keep `std::function` and shrink the captures. Rejected. It is a discipline
  that has to be re imposed at every new call site, it cannot be enforced by a
  compiler, and sixteen bytes is two pointers: the sweep bodies need five.
- Make the chunk callbacks non owning references and the precondition message a
  view. Chosen. Both encode a promise the code already kept, and neither changes
  a single call site: the lambdas are still lambda literals and the messages are
  still literals.
- Give `block_sweep` a scratch parameter for its snapshot, and give the Thomas
  recurrence a per chunk scratch parameter too. Chosen for the snapshot, which
  comes from the workspace. Rejected for the Thomas scratch, in favour of not
  needing it: see below.

**Fix.**

`Solver::solve` takes a `SolverWorkspace` the caller allocates once outside its
repetition loop and leaves the iterate in it, reported as a view. Each solver
says how many full size vectors it wants, so the driver's three, Richardson's
two, conjugate gradient's four and block Jacobi's four are each exactly what
that method needs. The three argument `solve` survives as a convenience that
allocates a workspace and copies the answer out; nothing timed calls it and
every test does, so `tests/equivalence/` compiles unchanged.

`RangeBody`, `RangeReducer` and the new `OrderedWork` are `FunctionRef`, a two
word non owning reference to a callable. `SweepFunction` is one as well.
`require` takes a `std::string_view`, and the three call sites whose message has
to be computed guard the computation. `ProgressBar` holds a view of its label.

`Poisson2D::solve_line` needs no scratch at all now. Its tridiagonal block is 4
on the diagonal and -1 off it for every grid line at every iteration, so the two
coefficient arrays of the Thomas recurrence depend on nothing but the grid size
and are computed once in the constructor. The eliminated right hand side is
written straight into the destination row, entry j of it at `dr[j + 1]`, which
is where the back substitution wants it anyway; that is safe under both
couplings because a five point stencil couples a line to lines i-1 and i+1 only,
so nothing the sweep reads lives in the row it is writing.

The per repetition timed call moves into `include/pnl/bench/timed_solve.hpp`, so
there is one timed region in the repository and both the driver and the gate
drive it. The gate observes it through a hook called just inside each end, which
is what makes "the timed region allocates nothing" a statement about the region
that is timed rather than about a region that resembles it.

**What is deliberately left.** `DenseProblem::block_sweep` still allocates its
per block right hand side once per chunk. The dense problem is not on any timed
path in the sweep matrix and the gate the specification asks for is stated over
`Poisson2D`, so the allocation is recorded here rather than removed by a change
that would need a chunk indexed scratch array and a chunk index the backend
interface does not hand the body.

**Verification.** `ctest -R test_no_allocation` passes: zero allocations between
the two ends of the timed region, for all twelve solvers, on `serial` at one
worker and `openmp` at four. The same file carries a third case that allocates
on purpose inside the armed window and requires the count to move, because a
counter that counts nothing passes every test there is.

No iterate moves. Built against `3faf2fc` and against this tree, the same
program dumped every solver's solution, residual history, iteration count and
evaluation count on both Poisson sources at n of 1, 2, 31 and 63 and on both
dense families: 67 dumps, all bit identical. The 4095 squared observation rows
in `PROGRESS.md` agree to the last digit on `relative_residual`, at
`3.262973e-02` before and after.

## 2026-09-06 MEAS-11 Dispersion was collected on every row and read by nothing, and three headline claims sit inside it

**Symptom.** `seconds_min` and `seconds_max` are written into every one of the
850 rows of `summary.csv` and were read by no code in the repository.
`git grep -E "seconds_min|seconds_max" 31b4c91 -- scripts report docs` returns
nothing at all: not the asset generator, not a chapter, not a document. The
harness has been measuring how far each configuration moves between repetitions
since the sweep was written, and then throwing the answer away at the point
where it would have qualified a number.

Recomputed from the committed rows at `cd57032941a8`, spread defined as
`(max - min) / median` over five repetitions, by the same helper the generator
now uses everywhere:

| block | rows | median spread | max |
| --- | --- | --- | --- |
| `backend_cost` | 90 | 7.3% | 36.9% |
| `scaling` | 66 | 9.8% | 41.6% |
| `pinning` | 36 | 11.8% | 41.1% |
| `reduction_cost` | 8 | 9.3% | 77.5% |
| `schedule_cost` | 6 | 8.1% | 19.8% |
| `mpi_scaling` | 18 | 9.3% | 26.4% |
| `device_comparison` | 24 | 4.0% | 77.3% |
| `dense` | 135 | 7.7% | 676.4% |

The first six agree with Section 4.5 of the specification to the digit. The last
two are not in that table and are worse than any row in it. `device_comparison`
looks calm at 4.0 percent until you find the row behind the 77.3: `sor_rb` on
`cuda` at 1023 squared, median 0.0196 s, minimum 0.0184, maximum 0.0336. That is
one repetition taking seventy percent longer than the other four, on the block
that carries the whole host against device comparison. `dense` at 676.4 percent
is `jacobi` on `openmp` at `dense_dd_512`, median 0.0009 s against a maximum of
0.0069: a run so short that a single scheduling event is seven times the
measurement. Neither block appears in any published table with a dispersion
figure beside it, so neither has ever had to explain itself.

**Three headline claims are inside the noise, and the generator now says so.**

- *The three thread models land within 6 percent of one another.* They do: at
  4,190,209 unknowns the medians are `pthreads` 1.0973 s, `jthread` 1.1248 and
  `openmp` 1.1678, which is a 6.4 percent range. But the `openmp` point alone
  spans 1.1100 to 1.3330, a spread of 19.1 percent, while `pthreads` spans 3.0
  and `jthread` 4.7. Every one of the three pairwise comparisons now reads
  `not separable at this precision`: `openmp` against `pthreads` differs by 6.4
  percent against a wider spread of 19.1, `jthread` against `openmp` by 3.7
  against 19.1, and `jthread` against `pthreads` by 2.5 against 4.7. The
  conclusion was right and the sentence stated it backwards, as a measured 6
  percent difference rather than as an inability to separate them.
- *The reduction cost table.* All four rows now read the phrase. `openmp` 5.5
  percent against a wider spread of 9.4, `pthreads` 0.3 against 4.5, `jthread`
  2.5 against 10.4, and `mpi` 3.1 against 77.5. The table measured nothing, and
  the README quoted it as what reproducibility is worth here.
- *The schedule cost figures.* The specification predicted that the `pthreads`
  and `jthread` numbers might survive and the `openmp` one would not. Both
  halves hold. `pthreads` is 14.2 percent slower on a dynamic schedule against a
  wider spread of 9.1, and `jthread` 12.3 against 6.9, so both print as numbers.
  `openmp` differs by 6.7 percent against a wider spread of 19.8 and prints the
  phrase.

Across the four tables that state a difference, 27 of the 56 comparisons the
committed data supports are inside their own noise: 1 of 25 in `backend_cost`,
21 of 24 in `pinning`, 4 of 4 in `reduction_cost` and 1 of 3 in `schedule_cost`.

**The knee had no interval at all.** It came from a two segment least squares
fit over 22 medians each carrying about 10 percent spread, and it returned a
different answer per backend with the disagreement unexplained. Bootstrapped,
the point estimates are unchanged and the intervals are the finding: `openmp` 5
workers with a 95 percent interval of 4 to 20, `pthreads` 3 with 3 to 4,
`jthread` 3 with 3 to 3. The `openmp` curve is flat and noisy from 4 workers to
20, so its knee is not located at all by this data, and the three backends have
no worker count inside all three intervals. The disagreement survives the
bootstrap. It is a result.

**Root cause.** Two of the eleven ground rules were written down and never
expressed in code. Nothing in the pipeline could have caught this, because a
report built from a generator that quotes only medians is indistinguishable from
a correct one at every gate the project had: the tables are generated, the
figures are generated, the numbers match the CSV, and the CSV is right. The
missing thing was a rule about what may be said, and a rule that lives only in
prose is enforced by whoever last read the prose.

**Options.**

- Leave it and write the caveat into the report's prose. Rejected. That is where
  it already was: Section 4.5 exists because the caveat was written and the
  tables kept quoting the numbers anyway. Prose beside a table does not stop the
  next reader quoting the table.
- Add a dispersion appendix and leave the tables alone. Rejected for the same
  reason, one page further away. A reader who copies a percentage out of the
  reduction cost table will not turn to an appendix to find out that the
  percentage is smaller than the noise it sits in.
- Put the spread beside every number it qualifies, and make the generator refuse
  to print an effect smaller than its own spread. Chosen. It is the only version
  where the rule fires without anybody remembering it, and where a table that
  can no longer support its claim says so in the cell rather than in a footnote.
- For the knee, an analytic interval on the fitted break point. Rejected: it
  needs a noise model the repetitions do not justify, and it would put a normal
  assumption on five draws. The bootstrap resamples what was actually recorded.
- For the knee on 1.0.0 rows, silently reuse the min, median and max as if they
  were repetitions. Rejected. The fallback is a triangular draw on those three
  numbers, it is named in a column of the table, it is never mixed with measured
  repetitions inside one backend, and the caption says what it is worth.

**Fix.** One definition of spread, `(max - min) / median`, in one helper, used by
every table, every figure and both rules. Every timing table gained a spread
column and every timing figure min to max whiskers. Bounds on a derived quantity,
a speedup or an efficiency, are taken at the corners of its inputs' intervals,
the pairing that makes it largest and the pairing that makes it smallest, which
is stated in a constant, in a comment and on the face of every figure that draws
one. `separable_effect` returns either the signed effect or the phrase, deciding
on the magnitude against the larger of the two rows' spreads, and every table
that states a difference goes through it. Beside each such table the generator
writes a `<name>_verdicts.tex` fragment, one sentence per comparison, so the
prose can quote the generator instead of a number somebody typed while reading
the table. `report/tables/dispersion.tex` carries n, the median and the maximum
per block, and deliberately no p90: the smallest block here has 6 rows and a
ninetieth percentile over six is the single worst row wearing the name of a
statistic.

`--allow-dirty` is the flag Section 7 A8 step 6 asks for. Without it the
generator refuses to build an asset from any row whose commit stamp ends in
`.dirty` and names how many it found. All 425 rows of the published generation
are dirty, so it refuses on the committed data today, and the CI reports job and
`make report` both have to ask for the override until phase A8a re measures from
a clean tree. The Makefile passes `ASSET_FLAGS`, empty by default, so the rule
holds unless somebody types the override out loud.

**Verification.** `python3 scripts/gen_report_assets.py` exits 3 with
`refusing to build a published asset from 425 of 425 row(s) whose commit stamp
ends in .dirty`, and with the flag it exits 0 and writes 6 figures in two themes,
9 tables and 5 verdict fragments. `tests/report/test_gen_report_assets.py` is
registered under the `unit` label and asserts the spread helper on known values,
the separability helper on a pair that separates and a pair that does not, that
an effect with no measured spread never prints as a number, that the bootstrap
finds a planted knee at 6 workers and returns an interval containing it under
both resamplings, and that the generator refuses a `.dirty` row without the flag,
accepts it with the flag, and needs no flag for a clean row. `make test` is 13 of
13. `ruff check benchmarks scripts tests` and
`python3 scripts/check_no_dashes.py .` are clean, and the report rebuilds from
scratch in 40 pages with no overfull box.

## 2026-09-06 PROV-03 Two generations in one summary, told apart by the order they were appended in

**Symptom.** `experiments/results/summary.csv` held 850 rows across two commits,
425 at `4abf914a7ea2.dirty` and 425 at `cd57032941a8.dirty`, and
`scripts/gen_report_assets.py` chose between them with

```python
newest = str(data["commit"].iloc[-1])
```

The two generations are the same size, so nothing a reader could see decided
which one the report was built from. The comment above that line said "rows are
appended in run order, so the last row names the newest commit", which is true
of one uninterrupted append and of nothing else: `--force`, a resumed sweep, a
merge of two files, or a sort would each reorder the file without changing a
single measurement, and the report would then be built from the other generation
with no diagnostic anywhere.

**Root cause.** Row order was standing in for a timestamp, and a timestamp was
available. `measured_at` has been a column since phase A1.5 and nothing read it.
The accumulation itself is deliberate and correct: the commit is part of the
resume identity, so a rebuild appends a new generation rather than overwriting
the old one and the history is kept. What was missing is that the file is a
record of every generation while the report is a statement about one, and
nothing in the pipeline made that difference explicit. Choosing silently is the
worst of the three available behaviours, because it produces a report that is
wrong in a way that leaves no trace.

**Options.**

- Sort by `measured_at` and keep taking the last row's commit. Rejected. It
  fixes the ordering and leaves the silence: two generations in one file is a
  state somebody has to resolve, and a selector that resolves it quietly means
  nobody ever does.
- Drop every commit but the newest, with a warning. Rejected for the same
  reason. A warning printed in the middle of a hundred lines of generator output
  is not a decision anybody makes.
- Refuse on more than one commit, name them with their row counts, and point at
  the archive procedure. Chosen. The archive is where a superseded generation
  belongs, and this is the only version where the file cannot quietly change
  which measurement the report describes.

**Fix.** `select_generation` in the generator refuses when the summary holds more
than one commit, names each with its row count, and names the path the
superseded rows go to. With one generation present it sorts by `measured_at`, so
any later reading of "latest" reads a clock rather than an append order. The 425
superseded rows moved to
`experiments/results/archive/summary-4abf914a7ea2-dirty.csv`, which leaves one
generation in `summary.csv` and one only. The manifest is selected by the commit
in its name rather than by being the only file with that name, and where a commit
has several manifests the generator says which one it used.

**Verification.** On the unsplit 850 row summary the generator exits 4 with

```text
gen_report_assets: refusing to build assets from 2 generations in one summary:
4abf914a7ea2.dirty (425 rows), cd57032941a8.dirty (425 rows).
```

After the split, `wc -l` is 426 on each of the two files, header plus 425, and
`awk -F, 'NR>1 {print $27}' | sort -u` returns exactly one commit from each.
`tests/report/test_gen_report_assets.py` gained the case and fails if the refusal
is relaxed back into a choice. The interim sweep of this phase wrote 6 rows at
one clean commit and the generator built from them with no flag.

## 2026-09-06 PROV-04 The committed manifest described a different session, and the refresh it never recorded is why four bandwidth figures disagree

**Symptom.** Section 4.3 of the V2 specification tabulates four host bandwidth
figures for one machine, no two of which agree:

| Source | host GiB/s | at 28 workers |
| --- | --- | --- |
| `experiments/results/session_manifest.json`, the committed data | 61.35 | 52.5 |
| `assets/reports/main_report.pdf`, generated Table 5.1 | 64.3, peaking at 8 workers | 55.1 |
| `README.md` line 147 and `report/chapters/results.tex` line 148, hand typed | 62.3, peaking at 4 workers | 37.7 |
| `docs/comparison_methodology.md` line 240, hand typed | 62.3 | not given |

The value 37.7 appears in no machine generated artifact in the repository, and
the report's argument about the memory system collapsing once every hyperthread
is engaged rests on it.

The committed manifest is where the disagreement starts. Its counts are

```json
"declared": 8, "executed": 0, "skipped_already_present": 8
```

Eight configurations declared, none executed, eight already present. The sweep
that produced the 425 published rows declared 440. So the tracked provenance
record describes a later eight configuration re run that measured nothing, and
its host block, its toolchain versions, its start time and both its bandwidth
probes belong to that re run rather than to the session the report quotes. The
file also has no `bandwidth_refreshed` key, which `run_sweep.py` writes whenever
the refresh runs.

**Root cause.** Two mechanisms, and the second is the one that produced the four
figures.

The manifest had a fixed name, `session_manifest.json`, and every session
overwrote it. A file with one name cannot describe two sessions, and there is no
state in which it is honest: it either describes the current session and not the
one the tracked rows came from, or the reverse. Whichever session wrote it last
wins, and nothing records that a session was overwritten.

The absent `bandwidth_refreshed` says `make bandwidth-refresh` never ran in the
committed session. The refresh exists because the sweep driver probes at the
start of its session, which is immediately after a build and a test run, so the
machine is still busy and the host figure comes out low: `run_sweep.py` records
39.8 GiB/s measured that way against about 60 on an idle machine. Every host
efficiency figure in the report divides by that number. With the refresh never
run, the manifest carried a start of session reading, 61.35, the generated table
carried something else, and whoever wrote the prose measured again by hand and
typed a third number. Four artifacts, four numbers, and no procedure anywhere
that could have made them agree.

**Options.**

- Re-run the refresh and update the tracked manifest. Rejected. It would give one
  number for the wrong session: the manifest would then describe a re run that
  measured nothing, refreshed at a third time, against rows measured at a fourth.
  The file's problem is not that its figure is stale.
- Keep one manifest and add a session identifier inside it. Rejected. The name is
  what a reader and a script both look at, and a file that has to be opened to
  find out which session it belongs to will be misread exactly as this one was.
- One manifest per session, named for the commit and the moment, with the refresh
  updating the manifest of the commit it refreshes. Chosen. The name carries the
  identity, nothing is overwritten, and a generation with no manifest is visibly
  a generation with no manifest instead of a generation wearing somebody else's.

**Fix.** Sessions write `manifest-<commit>-<timestamp>.json`, UTC, with the dot
of a `.dirty` stamp written as a dash in the name. `--refresh-bandwidth` updates
the manifest of the commit it is refreshing and writes one only if that commit
has none, so a commit never ends up with two manifests disagreeing about whether
its figures were re-probed. The counts are written at the top level as well as
inside `counts`, from one dictionary in one statement, so a gate can read
`executed` without knowing the shape. The generator selects the manifest whose
name carries the commit it is publishing and refuses when there is none, because
every efficiency figure divides by a bandwidth from that session and there is no
such number without it. The committed file is archived as
`experiments/results/archive/manifest-cd57032941a8-dirty.json` with a README
saying what it actually records. The four hand typed figures are phase A8b's
work; this entry is the mechanism behind them.

**Verification.** The small block sweep of this phase wrote
`manifest-651511d59a43-20260906T005711Z.json` beside its summary, and the refresh
that followed updated that same file rather than adding a second:

```text
6 2026-09-06T02:59:10+0200
```

which is `executed` and `bandwidth_refreshed` read out of the newest manifest in
the directory. `ls experiments/results/interim/manifest-*.json` lists one file.
The archived manifest is byte for byte what was tracked, and
`experiments/results/archive/README.md` quotes its three counts and says why
`bandwidth_refreshed` is absent from it.

---

## 2026-09-06 BUILD-02 The one flag target carried the machine's architecture into every consumer's build

**Symptom.** Not a failure, because until this phase nothing could consume the
library: release 1.0.0 shipped no `install()`, no `export()` and no package
configuration file, so the defect had no way to reach anyone. It surfaced the
moment phase B1 tried to export what was there. `pnl_flags` is an `INTERFACE`
target linked `PUBLIC` into `pnl_core`, and it carried

```cmake
$<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -ffp-contract=off>
$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-march=native>
```

plus `-Werror` under an option that defaulted to `ON`. Exporting that target
hands every one of those to a stranger.

**Root cause.** Two unrelated kinds of flag were in one target because until
there was an install rule there was no reason to tell them apart. One kind is
the numerical contract: `-ffp-contract=off` and the language standard are what a
consumer must compile with for the bit identity claim to be true of their build,
and since the library is about three quarters headers, they have to travel with
the headers or they do not reach the code they are about. The other kind is this
developer's build policy: `-march=native` is a property of the machine the
benchmark runs on, and `-Werror` is a promise about a warning set that a newer
compiler will break in code the consumer did not write. A `FetchContent`
consumer already inherited both, and a binary built with `-march=native` on a
build farm faults on the machine it is shipped to.

**Options.**

- Export `pnl_flags` as it stands and document the flags as advice. Rejected:
  the flags are not advice, they are on the consumer's compile line.
- Strip the flags from the exported target only, with an install time genex.
  Rejected: the developer build and the exported build would then differ in a
  way no single reading of the file shows, which is how the next flag gets put
  in the wrong half.
- Two targets, split by whose contract the flag is. Chosen.

**Fix.** `pnl_flags` keeps `-ffp-contract=off` and the `cxx_std_20` compile
feature, and nothing else. A new `pnl_dev_flags` carries `-Wall -Wextra
-Wpedantic`, `-O3 -march=native` under Release and `-Werror` under `PNL_WERROR`,
is populated only inside `if(PROJECT_IS_TOP_LEVEL)` so an embedded consumer gets
an empty target, is linked `PRIVATE` into `pnl_core`, `pnl`, `pnl_test_main`,
every test binary and `test_mpi`, and is never installed. `PNL_WERROR` now
defaults to `OFF`; the Makefile's `configure` target passes `-DPNL_WERROR=ON`, so
the developer build is exactly as strict as it was, and CI passes it explicitly
already.

`pnl_cuda` deliberately does not link `pnl_dev_flags`. Every option in that
target is guarded by `$<COMPILE_LANGUAGE:CXX>` and `pnl_cuda` compiles only CUDA
sources, so it would contribute nothing to a compile line while adding
`pnl_dev_flags` to a link interface that `install(EXPORT)` validates. Its host
warning set stays spelled out in its own `-Xcompiler` list.

**Verification.** The trap in the other direction is the expensive one, so it is
checked rather than assumed. After the split, on the compile line of
`src/main.cpp`, which is the measured binary:

```text
$ grep -m1 'main.cpp' build/compile_commands.json |
      grep -o -e '-march=native' -e '-O3' -e '-ffp-contract=off' | sort -u
-O3
-ffp-contract=off
-march=native
```

The same three appear on `equivalence/test_equivalence.cpp` and on
`src/backend/factory.cpp`, and `-Werror` is still on the developer build's own
lines. `make build && make test` is green at 13 of 13. The staged install proves
the other half: `grep -rn 'march=native' build/stage/lib/cmake/pnl/` and the same
grep for `Werror` both return nothing.

---

## 2026-09-06 BUILD-03 The export set refuses a target that links a static library it does not contain

**Symptom.** The first `install(EXPORT pnlTargets ...)` that named only the two
targets the plan called for stopped the configure dead, at generate time rather
than at install time:

```text
CMake Error in CMakeLists.txt:
  install(EXPORT "pnlTargets" ...) includes target "pnl_core" which requires
  target "pnl_cuda" that is not in any export set.
```

**Root cause.** `pnl_core` links `pnl_cuda` `PUBLIC`, and `pnl_cuda` is a static
library. A static library does not link its own dependencies, so whoever links
`libpnl_core.a` must also link `libpnl_cuda.a`, and CMake will not write an
imported target whose link interface names an archive it has not been told how
to find. The rule is not about CUDA; it is about static libraries, and it fires
on any `PUBLIC` link to one that stays outside the set.

What makes it worth an entry rather than a fixed typo is where it hides. It
fires only on a machine that has `nvcc`, because `pnl_cuda` does not exist
anywhere else, and it fires at configure time, so a contributor without a CUDA
toolkit could add an install rule, watch it work, commit it, and hand the
maintainer of the one machine that measures a repository that does not
configure.

**Options.**

- Link `pnl_cuda` `PRIVATE` into `pnl_core`. Rejected: it is not private. The
  device backend's symbols are reached through `pnl_core` by the driver and by
  `test_cuda`, and a static library's private dependency still has to be on the
  final link line, so this changes the spelling and not the fact.
- Build `pnl_cuda` shared. Rejected: it buys an install rpath problem in a
  release that is trying to reduce them, and the static shape is what the
  Fortran provider of part C is going to copy.
- Put `pnl_cuda` in the export set whenever it was built. Chosen.

**Fix.** The install target list is built up rather than written out:
`pnl_core pnl_flags`, then `pnl_cuda` appended under `if(TARGET pnl_cuda)`, so
the export set matches what the build actually produced. Section 9.7 of the
version 2 specification says the same list grows again for `pnl_fortran` in
1.2.0, with the extra treatment that one needs for its link language.

**Verification.** It bit, and it was made to bite again on purpose. With the
guard replaced by `if(FALSE)`, a configure with CUDA enabled fails with exactly
the message above; with the guard restored, `make configure` completes, the
staged install writes `lib/libpnl_cuda.a` beside `lib/libpnl_core.a`, and
`pnlTargets.cmake` declares `pnl::pnl_cuda`. A throwaway consumer declaring
`LANGUAGES CXX` and nothing else links against the staged install and runs, so
the `IMPORTED_LINK_INTERFACE_LANGUAGES "CUDA"` that the release targets file
records does not by itself break a C++ only project here: `CUDA::cudart` names
the runtime by absolute path, and the device symbols are already resolved inside
the archive because `CUDA_SEPARABLE_COMPILATION` is off. The Fortran case of 9.7
is not so lucky and will need the treatment described there.

---

## 2026-09-06 BUILD-04 A private dependency of a static library is not private to the export set

**Symptom.** With `pnl_cuda` in the export set, the same command failed again,
naming the target that had just been created to be kept out of it:

```text
CMake Error in CMakeLists.txt:
  install(EXPORT "pnlTargets" ...) includes target "pnl_core" which requires
  target "pnl_dev_flags" that is not in any export set.
```

**Root cause.** `target_link_libraries(pnl_core PRIVATE pnl_dev_flags)` does not
mean what it looks like it means on a static library. A static library performs
no link of its own, so CMake has to tell whoever links the archive later about
every dependency, private ones included; it records them in
`INTERFACE_LINK_LIBRARIES` wrapped in `$<LINK_ONLY:...>`, which suppresses the
compile side of the usage requirement and keeps the link side. `install(EXPORT)`
then validates every target named there, and `pnl_dev_flags` is named. The
target that exists precisely so that `-march=native` never crosses the install
boundary was blocking the install.

**Options.**

- Install `pnl_dev_flags` as well. Rejected outright: it carries `-march=native`
  and `-Werror`, which is the whole defect BUILD-02 is about, and an
  `INTERFACE` target with no artifact would sail through the install looking
  harmless.
- Drop the target and repeat the flags on each of `pnl_core`, `pnl`,
  `pnl_test_main` and every test with `target_compile_options`. Rejected: four
  copies of one policy is how the copies drift apart.
- Wrap the entry so it evaluates to nothing in the install interface. Chosen.

**Fix.** `target_link_libraries(pnl_core PRIVATE $<BUILD_INTERFACE:pnl_dev_flags>)`.
The build tree is unchanged, and the exported interface records
`$<LINK_ONLY:>` with nothing inside it. Only `pnl_core` needs this, because it
is the only exported target that links the developer flags; `pnl_test_main` has
the same shape and is never installed.

**Verification.** The configure completes. The compile line of `src/main.cpp`,
of `src/backend/factory.cpp` and of `equivalence/test_equivalence.cpp` still
carries `-O3`, `-march=native` and `-ffp-contract=off`, so nothing measured
moved. `grep -rn 'march=native' build/stage/lib/cmake/pnl/` and the same grep
for `Werror` return nothing, and the whole of `pnl::core`'s exported link
interface reads

```text
"pnl::pnl_flags;\$<LINK_ONLY:>;Threads::Threads;OpenMP::OpenMP_CXX;MPI::MPI_CXX;pnl::pnl_cuda"
```

which is the empty genex where the developer flags used to be.

---

## 2026-09-06 NUM-05 The flag the bit identity claim rests on had nothing that could notice its absence

**Symptom.** No failure. This one was found by reading the contract rather than
by watching something break, which is why it is worth writing down: the library
promises that the same numerics run over every execution model and agree to the
last bit, and that promise depends on `-ffp-contract=off`. Nothing in the tree
could tell whether the flag was there. Removing it from `CMakeLists.txt` and
rebuilding leaves all seventeen tests green, because the equivalence suite
compares backends against each other inside one process, all compiled the same
way. Every one of them would contract, they would all contract identically, and
the suite would pass while the central claim of the report had quietly become
false against the device path, which is compiled `--fmad=false` and does not
contract.

The exposure is bigger than this repository. Phase B1 exported the package and
the library is about three quarters headers, so most of the arithmetic a
consumer runs is compiled on the consumer's own command line. `pnl_flags`
carries `-ffp-contract=off` publicly, which covers a consumer who uses the
exported target, and covers nobody who copies the headers or writes their own
build.

**Root cause.** Two flags, two different holes.

`-ffast-math` defines `__FAST_MATH__`, so a header can see it. Nothing looked.

`-ffp-contract` defines nothing at all. There is no predefined macro for it, and
`-ffp-contract=fast` is GCC's default, so the dangerous setting is the one a
consumer gets by doing nothing and the header cannot see it under any
circumstances. A compile time check is not available, so the check has to happen
at runtime, and a runtime check that nobody calls is decoration, so it also has
to be on a path every caller takes.

**Options.**

- A `#error` on `__FAST_MATH__` and stop there. Rejected as the whole answer. It
  catches the flag nobody sets by accident and misses the one that is on by
  default. Kept as the first of two mechanisms, not as the mechanism.
- Compare a contraction sensitive expression against the same expression written
  to defeat contraction. This is the obvious probe and it does not work. In a
  translation unit compiled with contraction on, either both sides contract and
  agree, or GCC folds both at compile time with correct rounding and they agree
  again. The check passes in exactly the case it exists to catch.
- Compare a genuinely emitted runtime computation against a hard coded literal
  of the correctly rounded unfused result. Chosen.
- Grep the compile line, in CMake or in CI. Rejected as the primary mechanism:
  it tests this repository's build system and says nothing about a consumer's,
  which is where the risk actually is.

**Fix.** Two mechanisms, in `include/pnl/core/types.hpp` and the new
`include/pnl/core/contract.hpp`.

The static one is four lines at the top of `types.hpp`, above every include, so
a `-ffast-math` build stops there.

The runtime one is `assert_no_contraction()`, which `make_backend` calls once
per process through a function local static. Its values are chosen so that the
fused and unfused answers are different doubles rather than the same double
reached two ways: with `a = b = 1 + 2^-27` and `c = -1`, the exact product is
`1 + 2^-26 + 2^-54`, one ulp there is `2^-52`, so round to nearest discards the
last term and the two step answer is exactly `2^-26`; a fused multiply add
rounds once and keeps it, giving `2^-26 + 2^-54`, which is itself exactly
representable. `volatile` on the three operands is what forces the expression to
be emitted instead of folded, and the comparison is against the literal `0x1p-26`
rather than against anything computed. The three operands and the expected
result are named `constexpr` values in the same header because release 1.2.0
runs the same probe on the Fortran side, as assertion 4 of Section 9.8, and a
number that has to be identical in two languages should have one definition in
each and no third spelling.

The residual limitation is stated in the header rather than left for someone to
discover. `assert_no_contraction` is inline, so a consumer's translation unit and
this library's `factory.cpp` each emit a copy and the linker keeps one; at `-O3`
the call inside `make_backend` is inlined into the copy `factory.cpp` compiled,
which is the copy built with the flag. A consumer who wants their own compile
line checked calls the function from a translation unit of their own, and that is
why it is public rather than a detail of the factory.

**Verification.** Ground rule 8 asks for a test that fails when the flag is
missing, so there are three, and the second is the one that carries the weight.

`test_contract` asserts the probe is quiet under the project flags and that the
constants are what their comment claims, including that `2^-26` and
`2^-26 + 2^-54` are different doubles, which is what makes `==` a legitimate
comparison here rather than a tolerance in disguise.

`test_contract_detects_fma` compiles the same source into a second executable
with `-ffp-contract=fast -O2 -march=native`, linking neither `pnl_flags` nor
`pnl_test_main` because both carry the flag that turns contraction off, and
asserts the probe throws. Without it the probe would pass just as happily with
an empty body. The compiler does fuse: at the guarded comparison the fused object
emits

```text
	vmovsd	40(%rsp), %xmm0
	vmovsd	48(%rsp), %xmm2
	vmovsd	56(%rsp), %xmm1
	vfmadd132sd	%xmm2, %xmm1, %xmm0
	vucomisd	.LC15(%rip), %xmm0
```

and the same source at `-ffp-contract=off` emits

```text
	vmulsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vucomisd	.LC15(%rip), %xmm0
```

one `vfmadd132sd` against zero.

`test_fast_math_rejected` runs the configured compiler over
`tests/unit/fast_math_probe.cpp` twice: with `-ffast-math` the compile must fail
and its output must contain the message, and without it the same file must
compile cleanly. The second compile is what stops the test passing on a typo or
a missing compiler, which an exit status alone would not.

**Addition, 2026-09-06, phase B4.** The residual limitation above was not a
limitation to live with, it was the bug. The probe checked `factory.cpp`, which
is compiled with this library's flags and is therefore the one translation unit
whose flags were never in doubt, and it checked nothing at all in the consumer's
own, which is where finding 4.8 says three quarters of the arithmetic is
compiled. Phase B4 had to rewrite `make_backend` for the registry anyway, so the
public `make_backend` is now an inline function in
`include/pnl/backend/backend.hpp` that calls `assert_no_contraction()` and then a
non inline `detail::make_backend_impl` in `factory.cpp` that does the registry
lookup. Being inline, the probe is emitted into the translation unit that
constructs the backend and so is compiled with the flags that apply to the code
that translation unit is about to run.

The function local static went with it, and that is deliberate rather than an
oversight. A function local static inside an inline function is one object for
the whole program, not one per translation unit, so a guard would have run the
probe in whichever translation unit constructed the first backend and silently
exempted every other one. That is the same hole in a smaller shape. The probe
runs on every construction instead: three volatile stores, a multiply, an add and
a compare, against a call that starts a thread pool.

`test_contract_detects_fma` gained the case that proves it. That executable
compiles `tests/unit/test_contract.cpp` with `-ffp-contract=fast` and now links
`pnl_core` through `$<LINK_ONLY:pnl_core>`, which links the archive and
propagates none of its compile usage requirements, so `-ffp-contract=off` does
not arrive with it. CMake appends an inherited interface option after a target's
own, so a plain link would have put `-ffp-contract=off` last on the compile line
and won. From that fused translation unit, `make_backend("serial", config)`
throws `ConfigurationError`; from the ordinarily compiled `test_contract` the
same call returns a backend. Before this change the first of those two calls
succeeded, which is the whole finding.

---

## 2026-09-06 NUM-06 The relaxation factor a foreign stencil never received

**Symptom.** No failure here either, and this one could not have produced one:
the library shipped no problem it was wrong for. `Sor::resolve_relaxation`
answered the question "what omega should SOR use when the caller asked for none"
with a downcast:

```cpp
if (const auto* poisson = dynamic_cast<const problems::Poisson2D*>(&problem)) {
    return poisson->theory().optimal_relaxation;
}
return 1.0;
```

A `Poisson2D` got Young's closed form optimum, `2 / (1 + sin(pi h))`, which at
63 squared is 1.906455. Everything else got 1.0, which turns SOR into Gauss
Seidel. `DenseProblem` got it, and so would a third party's own symmetric
positive definite stencil, silently, with no diagnostic and no way to say
otherwise however much its author knew about its spectrum. On the model problem
that is not a small difference: the optimum changes the asymptotic rate from
`1 - O(h^2)` to `1 - O(h)`, which is a change in the order of the iteration
count from `O(n^2)` to `O(n)`, so a consumer whose stencil had a known optimum
was being handed the slowest member of the family and told nothing.

Related, and already fixed elsewhere: `src/main.cpp` called this function for
*every* row regardless of solver, so `jacobi` and `cg` rows recorded an SOR
relaxation factor no run ever read. That half is MEAS-04, repaired by
`Solver::relaxation_factor`, and it is a different defect in the same function.
This entry is about the answer, not about who was asking.

Section 4.8 of the version 2 specification lists both halves; this is the first.

**Root cause.** The knowledge lives in the wrong class. Which relaxation factor
is right is a property of the operator's spectrum, so the operator is the only
object that can answer it, and the solver asked by testing the operator's type
against a list of one. A downcast in place of a virtual is a closed world
assumption written down: it is correct exactly while the set of problems is the
set the author enumerated, and it fails silently rather than loudly on the day it
is not, because the fallback is a legal value.

**Options.**

- Keep the downcast and document that a non `Poisson2D` problem gets `omega = 1`.
  Rejected. It converts a silent wrong answer into a documented wrong answer,
  which is an improvement in honesty and none at all in behaviour, and phase B4
  exists to open the extension points rather than to annotate the places they
  are closed.
- Extend the chain of casts as problems are added. Rejected on sight. It is the
  same defect once per problem, and it cannot reach a problem the library has
  never heard of, which is the whole population this finding is about.
- A trait or a policy object passed alongside the problem, so the caller supplies
  the factor. Rejected. `SolverOptions::relaxation` already is that, and a caller
  who knows the number can pass it today; what was missing is the answer for a
  caller who does not, which only the problem has.
- `virtual Real suggested_relaxation() const noexcept` on `Problem`, defaulting
  to 1.0, overridden where a closed form optimum exists. Chosen. It puts the
  question where the answer is, the default is exactly the old fallback so no
  existing behaviour moves, and a third party overrides one function.

**Fix.** In commit `42d5d74`. `Problem::suggested_relaxation()` returns 1.0 and
says why in its comment: one makes SOR into Gauss Seidel, which converges on any
symmetric positive definite or strictly diagonally dominant system, and claims
nothing about a spectrum the base class does not know. `Poisson2D` overrides it
and returns `theory_.optimal_relaxation`, which is the same member of the same
object the downcast used to reach, so the number is bit identical and not merely
equal. `Sor::resolve_relaxation` is two lines:

```cpp
if (options.relaxation > 0.0) return options.relaxation;
return problem.suggested_relaxation();
```

`SorRedBlack` still calls it, because the red black ordering is consistently
ordered in Young's sense and the closed form optimum carries over unchanged.
`SymmetricSor` still does not, because its own optimum is not the SOR optimum;
that is MEAS-04's arrangement and this change leaves it alone.
`include/pnl/solvers/sor.hpp` no longer includes `poisson2d.hpp` at all, which is
the visible sign that the coupling is gone.

**Verification.** Three things, and the first is the one that matters, because a
factor that is only *approximately* the old one would change every SOR iterate in
the sweep.

At 63 squared, 25 fixed iterations, serial, against `2 / (1 + sin(pi / 64))`
computed independently in Python as 1.906454701582762 and passed with `--omega`:

```text
sor      omega=1.906455 iterations=25 relative_residual=1.790482e-01
         fields that differ between --omega 0 and the explicit run, timing aside: none
ssor     omega=1.000000 iterations=25 relative_residual=2.758391e-02
         fields that differ: relative_residual, omega
sor_rb   omega=1.906455 iterations=25 relative_residual=1.430338e-01
         fields that differ between --omega 0 and the explicit run, timing aside: none
```

Every field of the `sor` and `sor_rb` rows agrees with the explicitly supplied
optimum, timings aside, so the default path returns that number and not a nearby
one. `ssor` differs because it keeps its own default of one and never asks
`resolve_relaxation`, which is the behaviour MEAS-04 established and is what it
should do.

Second, the convergence suite still finds the optimum where it always did:

```text
$ build/tests/test_convergence
  pass  convergence/SOR at the closed form optimum beats every nearby factor
  pass  convergence/optimal SOR changes the order of the iteration count
  pass  convergence/red black ordering costs iterations but keeps the rate order
13 passed, 0 failed
```

The first of those perturbs omega either way from `theory().optimal_relaxation`
and requires the iteration count to rise, so it is testing that the number the
solver uses really is the minimum. The second runs `sor` with `omega` left at
zero, which is the path this change rewrote, and asserts the count grows by less
than 2.6 between 31 squared and 63 squared, which is the `O(n)` order the optimum
buys and which `omega = 1` would not produce.

Third, `ctest` is green across the unit, convergence, equivalence, MPI and CUDA
levels, 19 of 19, and `grep -rn dynamic_cast include/pnl/solvers/` returns
nothing.

---

## 2026-09-06 BUILD-05 The self check for the allocation counter was deleted by the compiler it was meant to check

**Symptom.** The first run of the suite under clang 21.1.8 at `-O3
-march=native` failed a case that has never failed under GCC:

```text
  FAIL  no_allocation/the counter sees an allocation when there is one
        the counting allocator saw 0 allocations where a deliberate one had just been made, so the gate above proves nothing
        (counted >= 1 at .../tests/unit/test_no_allocation.cpp:251)
```

That case is the instrument check for the phase A5 gate. It allocates on purpose
inside the armed window and requires the counter to move, because a counter that
counts nothing passes every no allocation test there is.

**Root cause.** The compiler deleted the allocation. A new expression is one of
the few constructs an implementation may remove outright: the standard permits
it to omit a call to a replaceable global allocation function when the storage
does not escape. Here the array was written once, read never and deleted a few
lines later, so clang's optimiser paired the `operator new[]` with the
`operator delete[]` and dropped both. Nothing reached the replacement operator,
the counter stayed at zero, and the check correctly reported that it had no
evidence. Neither GCC 15.2.0 nor GCC 14.3.0 performs the elision on this shape,
which is why one compiler had been enough to keep it hidden.

The case was not wrong about the library, it was wrong about itself. The gate it
guards, that the timed region allocates nothing, is unaffected: those
allocations happen inside `std::function`, `std::string` and `std::vector` in
other translation units, where there is nothing to elide.

**Options.**

- Compile the file at `-O0`. Rejected: the gate has to watch the same optimised
  code the driver runs, and lowering the optimisation to protect the instrument
  changes what is being asserted.
- Allocate a size the compiler cannot fold. Rejected as insufficient: the
  elision does not depend on the size being a constant, it depends on the
  storage never escaping.
- Publish the pointer through a volatile object. Chosen. A volatile store is an
  observable side effect, so the pointer escapes and the omission is no longer
  permitted.

**Fix.** A file scope `void* volatile allocation_escape` in
`tests/unit/test_no_allocation.cpp`, assigned the pointer immediately after the
new expression and cleared after the delete. Four lines, and nothing about what
the gate asserts changes.

**Verification.** The case passes under clang, and still passes under both GCC
versions, so the fix did not move the problem to the compiler that never had it:

```text
$ build-clang/tests/test_no_allocation
  pass  no_allocation/the timed region allocates nothing on serial
  pass  no_allocation/openmp is not in this build, so its case is skipped
  pass  no_allocation/the counter sees an allocation when there is one
3 passed, 0 failed

$ build-gcc14/tests/test_no_allocation
  pass  no_allocation/the timed region allocates nothing on serial
  pass  no_allocation/the timed region allocates nothing on openmp
  pass  no_allocation/the counter sees an allocation when there is one
3 passed, 0 failed
```

---

## 2026-09-06 BUILD-06 A unit test named a backend the build is allowed not to contain

**Symptom.** In the same clang run, immediately before the case above:

```text
  FAIL  no_allocation/the timed region allocates nothing on openmp
        unexpected exception: invalid argument: backend 'openmp' is not available in this build; available backends are serial, pthreads, jthread, mpi
```

**Root cause.** There is no OpenMP runtime for clang on this machine, so
`find_package(OpenMP COMPONENTS CXX)` fails, `CMakeLists.txt` prints "pnl:
OpenMP not found, that backend will be skipped" and configures without
`PNL_WITH_OPENMP`, which is exactly what it is written to do. `pnl --list` in
that build offers `serial pthreads jthread mpi`. The test then asked
`make_backend` for `openmp` by name with no guard, so a configuration the build
system supports on purpose became a test failure.

Nothing else in the suite had the fault. The equivalence suite iterates over
`available_backends()` rather than naming anything, and `test_registry.cpp`
already guards its expectations with `#if defined(PNL_WITH_OPENMP)`. This one
file was the exception, and only a compiler without an OpenMP runtime could
show it.

**Options.**

- Drop the case when OpenMP is absent. Rejected on its own: a test that vanishes
  from the binary is a test whose absence nobody notices.
- Iterate over `available_backends()` as the equivalence suite does. Rejected:
  the case is deliberately about a runtime that starts a thread team, and
  running it over `serial` and `pthreads` too would widen the phase A5 gate
  rather than make it portable.
- Guard on `PNL_WITH_OPENMP` and assert the reason for the skip in the other
  branch. Chosen.

**Fix.** The case is compiled when `PNL_WITH_OPENMP` is defined. When it is not,
a case of the same shape takes its place, named so that the skip is visible in
the output, and it requires that `openmp` really is absent from
`available_backends()`. A build that has the backend can therefore never take
the skip path silently.

**Verification.** Under clang, where OpenMP is absent, the skip case runs and
`test_no_allocation` passes; under g++-15 and g++-14, where `find_package`
reports OpenMP 4.5 with spec date 201511, the original case runs and passes. The
two spellings of the middle line in the block above are the whole difference,
and `ctest` is 18 of 18 under clang, 18 of 18 under GCC 14 with CUDA off, and 19
of 19 in the default build.

---

## 2026-09-06 CONC-03 A body that throws kills the process, or stops it forever

**Symptom.** A scratch program dispatches a `parallel_for` over four workers
whose body throws `NumericalFailure` from exactly one chunk, and catches it
around the dispatch. Six runs, one per pool and per throwing chunk, against the
tree at 7110ab0:

```text
=== pthreads 4 -1 ===
pthreads at 4 workers, 4 chunks, throwing from chunk 3
terminate called after throwing an instance of 'pnl::NumericalFailure'
  what():  numerical failure: chunk refused to run
exit 134
=== pthreads 4 0 ===
pthreads at 4 workers, 4 chunks, throwing from chunk 0
caught on the dispatching thread: numerical failure: chunk refused to run
exit 0
=== jthread 4 -1 ===
jthread at 4 workers, 4 chunks, throwing from chunk 3
terminate called after throwing an instance of 'pnl::NumericalFailure'
  what():  numerical failure: chunk refused to run
exit 134
=== jthread 4 0 ===
jthread at 4 workers, 4 chunks, throwing from chunk 0
exit 124
=== openmp 4 -1 ===
openmp at 4 workers, 4 chunks, throwing from chunk 3
terminate called after throwing an instance of 'pnl::NumericalFailure'
  what():  numerical failure: chunk refused to run
exit 134
=== openmp 1 0 ===
openmp at 1 workers, 1 chunks, throwing from chunk 0
terminate called after throwing an instance of 'pnl::NumericalFailure'
  what():  numerical failure: chunk refused to run
exit 134
```

Exit 134 is `SIGABRT`. Exit 124 is `timeout` at twenty seconds: the jthread pool
with a throw on the dispatching thread does not fail, it stops, and stays
stopped.

**Root cause.** Three boundaries an exception may not cross, and the library
crossed all three.

A pthread entry point is one: `worker_entry` calls `worker_loop`, which calls
the body, and an exception that reaches the `extern "C"` frame is
`std::terminate`. A `std::jthread` body is the second, for the same reason. An
OpenMP structured block is the third and is the strictest: the standard requires
an exception thrown inside a region to be caught inside the same region, and
libgomp terminates when it is not. That accounts for the four aborts, including
the single threaded OpenMP case, where the team is one thread and the boundary
is still there.

The deadlock is a different fault with the same cause. `JthreadBackend::run_task`
opens the release barrier, runs worker zero's chunks on the dispatching thread,
and then waits on the collect barrier. A throw from worker zero's chunks skips
the collect barrier, so the exception propagates correctly to the caller and
every other worker waits on a barrier phase that will never complete. Nothing
times out, nothing prints, and the process holds its threads until it is killed.
`PthreadsBackend` survives the same case only by accident: its workers count
themselves down rather than meeting the dispatcher, so a throw on the dispatching
thread leaves the pool in a state the next call happens to repair.

**Options.**

- Require bodies not to throw and document it. Rejected. The library's own
  numerics throw `NumericalFailure` from inside sweeps, so the rule would be one
  this repository breaks itself, and an undiagnosable abort is what a consumer
  gets for breaking it.
- Catch, store and rethrow per worker. Rejected as written: a worker that stops
  at its first throw does not finish its chunks, and on the jthread pool a
  worker that returns early is a worker the barrier is still waiting for. The
  cure would reintroduce the deadlock it was fixing.
- Catch per chunk, store the first, always complete the protocol, rethrow on the
  dispatching thread after the join point. Chosen.

**Fix.** `detail::ExceptionRelay` in `pnl/backend/backend.hpp`. `capture()` is
`noexcept` and wraps one chunk; the first thrower wins an atomic exchange and
publishes a `std::exception_ptr` with a release store; `rethrow()` reads it with
an acquire load on the dispatching thread and clears the relay, so the backend
is usable again. The four dispatching paths that cross a thread boundary use it:
both pools, the OpenMP backend, and the OpenMP team inside `HybridBackend`. The
serial and distributed backends run their chunks on the calling thread, where a
throw already propagates, and are unchanged.

Two consequences are now documented on `Backend::parallel_for` rather than left
to be discovered. Every chunk of a dispatch is attempted, because a worker that
stopped short would not reach the barrier. And the exception that arrives is the
caller's own object, type and message intact, because what crosses the boundary
is a `std::exception_ptr` rather than a description of one.

**Verification.** `tests/unit/test_throwing_body.cpp`, on every pool this build
has, at 1, 2 and 8 workers, throwing from chunk zero and from the last chunk,
asserting the message, that every chunk which did not throw ran, and that a
further dispatch afterwards still covers the whole grid. Its CTest entry carries
a 30 second timeout rather than the suite's 900, so the deadlock cannot come
back as a fifteen minute hang.

```text
$ build/tests/test_throwing_body
  pass  throwing_body/a body that throws surfaces on the dispatching thread
  pass  throwing_body/a reducer that throws surfaces on the dispatching thread
        pools covered: openmp pthreads jthread
  pass  throwing_body/every pool this build has is covered
3 passed, 0 failed
```

---

## 2026-09-06 MPI-03 A failure on one rank hangs the job instead of failing it

**Symptom.** `PNL_TEST_FAIL_RANK=1` makes rank 1 fail one case of `test_mpi`,
from a point that case had collectives after. Two ranks, a 60 second limit
standing in for the 900 second CTest timeout:

```text
$ PNL_TEST_FAIL_RANK=1 timeout 60 mpirun -np 2 --oversubscribe build/tests/test_mpi
running at 2 rank(s)
  pass  mpi/the row decomposition covers the grid exactly once
exit 124 after 61 seconds
```

Exit 124 is `timeout`. Nothing failed; nothing finished either.

**Root cause.** Two places, one shape. An error on one rank is caught locally and
the ranks that did not see it are left in a collective that has lost a
participant.

In `src/main.cpp` the `catch` printed the message, set a status and fell through
to `MPI_Finalize`. The rank that threw is by then out of the solve while every
other rank is still inside it, in `MPI_Sendrecv` for the halo or in the
allgather behind the reduction. One rank finalising does not release them.

In `tests/mpi/test_mpi.cpp` the status was combined once, after every case had
run. That is a correct end of run check and it is not the problem: the problem
is that a rank which fails a collective case never reaches the end of the run,
so the combination is a collective nobody arrives at. The suite's own cases are
where this bites hardest, because `run_all` catches the assertion and moves to
the next case, so the failing rank then makes the *next* case's collectives
against neighbours still making the previous one's.

**Options.**

- Longer CTest timeouts. Rejected outright: it converts a hang into a slower
  hang and hides it behind a number nobody reads.
- `MPI_Abort` from the failing rank, and nothing else. Enough for the driver,
  where the failure is a configuration or a solve that stopped, and chosen
  there. Not enough for the test binary on its own: a failure that every rank
  sees would then abort rather than report which cases passed first.
- Combine the flag after every case, and nothing else. Rejected: the combination
  is itself a collective, so a rank that has left a case mid collective would
  deadlock in the combination instead of in the halo exchange.
- Both, with the reduction in its nonblocking form and a deadline on the failing
  path. Chosen for the test binary.

**Fix.** `src/main.cpp` calls `MPI_Abort(MPI_COMM_WORLD, status)` from the catch
when the backend is distributed, after flushing, so one rank's error takes the
whole job down with a status a caller can read.

`pnl_test.hpp` gains an optional `CaseHook` called after each case with that
case's status; without one, which is every single process binary here, the
runner behaves exactly as before. `test_mpi.cpp` installs a hook that combines
the flag with `MPI_Iallreduce`. A rank that passed waits for it. A rank that
failed polls for five seconds and then aborts the job, because a rank that has
just thrown is precisely the one that cannot assume the others are able to reach
this point. The nonblocking form is used on both paths because a blocking
collective and a nonblocking one do not match.

**Verification.** The same injection, and a distributed driver run whose
configuration is rejected:

```text
$ PNL_TEST_FAIL_RANK=1 timeout 120 mpirun -np 2 --oversubscribe build/tests/test_mpi
rank 1 failed 'mpi/order free solvers agree with serial on the Poisson problem'
and the other ranks did not reach the end of that case within 5 seconds, so they
are waiting in a collective this rank has left. Aborting the job.
exit 1 after 5 seconds

$ timeout 60 mpirun -np 2 --oversubscribe build/pnl --backend mpi --solver cg \
      --problem dense_dd --size 64
pnl: solver cg does not apply to problem dense_dd_64: conjugate gradient needs a
symmetric positive definite operator; ...
pnl: invalid argument: solver not applicable
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD
exit 3 after 0 seconds
```

`test_mpi_failure_mode` in `tests/CMakeLists.txt` keeps it closed: two ranks
with the variable set, `WILL_FAIL` so that a non zero exit is the pass
condition, and a 60 second timeout that CTest still applies, so a returning hang
is reported as a failure rather than inverted into a pass. The ordinary
`test_mpi_1rank`, `2rank` and `4rank` are unchanged and green.

---

## 2026-09-06 NUM-07 A matrix allocates its square before deciding the order is legal

**Symptom.** Two failures from the same line. In the ordinary build:

```text
$ build/tests/test_no_allocation "matrix order"
  FAIL  no_allocation/a matrix order that is refused allocates nothing first
        DenseMatrix(-1000) allocated 8000000 bytes before refusing the order
  FAIL  no_allocation/a matrix order whose square overflows is refused
        expected InvalidArgument from numerics::DenseMatrix(HUGE_ORDER)
0 passed, 2 failed, 3 filtered out
```

and under the address and undefined behaviour sanitizers:

```text
$ build-asan-ubsan/tests/test_no_allocation "matrix order"
include/pnl/numerics/lu.hpp:30:77: runtime error: signed integer overflow:
4294967296 * 4294967296 cannot be represented in type 'long int'
```

**Root cause.** The constructor was

```cpp
explicit DenseMatrix(Index n) : n_(n), data_(static_cast<std::size_t>(n * n), 0.0) {
    require(n >= 0, "DenseMatrix order must be non negative");
}
```

and member initialisers run before the body, so the `require` guarded nothing
that had not already happened. A negative order squares to a positive number,
so order minus 1000 allocated the eight megabytes of a thousand by thousand
matrix and then reported that the order was negative. `Matrix` in `qr.hpp` had
the identical shape, and there the rectangular case is worse: minus 1000 by 1000
is a negative product, which cast to `std::size_t` is 1.8e19 elements, so it
threw `std::length_error` from inside the vector rather than the
`InvalidArgument` the interface documents.

The overflow is the more serious half. `Index` is `std::ptrdiff_t`, so an order
above 2^31.5 overflows the product, and signed overflow is undefined behaviour
rather than wraparound: the sanitizer stops the process, and an optimiser is
entitled to assume it cannot happen and to delete a check written after it. At
order 2^32 the wrapped product is exactly zero, so the ordinary build allocated
nothing, constructed a matrix that claims an order of four billion over an empty
array, and reported no error at all. Every element access on it is then a read
far outside the allocation.

**Options.**

- Check in the body and allocate there too, leaving the member default
  constructed. Rejected: it makes the storage twice initialised on every valid
  construction to catch a case that should never be reached, and the class then
  has a window in which it is not an invariant.
- A delegating constructor that validates and forwards. Workable, and needs a
  private tag type to tell the two constructors apart, which is machinery for
  its own sake here.
- A helper called from the member initialiser, returning the element count.
  Chosen: the argument is evaluated before the vector is constructed, so the
  refusal genuinely precedes the allocation, and both matrix classes share one
  copy of the rule.

**Fix.** `pnl::numerics::detail::checked_extent(rows, cols, what)` in `lu.hpp`.
It refuses a negative extent, and tests the product against the index limit by
division, so finding out that the product would overflow does not overflow on
the way. `DenseMatrix` and `Matrix` both build their storage from it and their
bodies are now empty.

**Verification.** Two cases in `tests/unit/test_no_allocation.cpp`, which is
where the counting allocator lives. The counter gained the size of each request
alongside the count, because "allocated nothing" is not the assertion here: the
refusal builds the sentence it carries, so what separates the two states is
eight megabytes against a hundred bytes. Both cases pass in the ordinary build
and under the sanitizers, where the overflow report is gone.

```text
$ build-asan-ubsan/tests/test_no_allocation
  pass  no_allocation/the timed region allocates nothing on serial
  pass  no_allocation/the timed region allocates nothing on openmp
  pass  no_allocation/the counter sees an allocation when there is one
  pass  no_allocation/a matrix order that is refused allocates nothing first
  pass  no_allocation/a matrix order whose square overflows is refused
5 passed, 0 failed
```

---

## 2026-09-06 NUM-08 The Thomas solver writes past the end of an empty system

**Symptom.** Four empty spans, which the length check accepts because they are
all the same length. Under the address and undefined behaviour sanitizers:

```text
$ build-asan-ubsan/tests/test_numerics thomas
/usr/include/c++/15/span:302: constexpr std::span<_Type, _Extent>::element_type&
std::span<_Type, _Extent>::operator[](size_type) const: Assertion
'__idx < size()' failed.
exit 134
```

**Root cause.** The forward sweep begins outside the loop, because the first row
of the recurrence has no sub diagonal term:

```cpp
Vector c_prime(static_cast<std::size_t>(n), 0.0);
c_prime[0] = upper[0] / diagonal[0];
rhs[0] = rhs[0] / diagonal[0];
```

At `n == 0` there is no zeroth anything. `c_prime` is an empty vector whose
`data()` is null, and `upper` and `diagonal` are empty spans, so the line writes
one past the end of nothing and reads one past the end of two more. The
diagonal dominance loop above it runs zero times and so raises nothing, and the
length check is satisfied, so the function accepts the call and then walks off
the end of three objects.

**Options.**

- Throw on an empty system. Rejected. Nothing else in this library treats empty
  input as an error: `lu_solve` of an order zero matrix returns an empty vector,
  `parallel_for` over an empty range does nothing, `reduce` returns its initial
  value. A routine that refused it would be the odd one out, and the caller most
  likely to hand it an empty system is a loop over rows that happens to have
  none.
- Return early. Chosen: an empty right hand side is already its own solution.

**Fix.** `if (n == 0) return;` after the precondition checks and before the
sweep, with the reason recorded beside it.

**Verification.** Two cases in `tests/unit/test_numerics.cpp`: the empty system,
and the system of one unknown beside it, because the second is the case a
careless fix to the first would break. Both pass under the sanitizers, where the
assertion is gone, and in the ordinary build.

```text
$ build-asan-ubsan/tests/test_numerics thomas
  pass  lu/thomas solves a tridiagonal system
  pass  lu/thomas refuses a matrix that is not diagonally dominant
  pass  lu/thomas accepts an empty system and leaves it empty
  pass  lu/thomas solves a system of one unknown
4 passed, 0 failed, 20 filtered out
```

---

## 2026-09-06 NUM-09 A bracket that underflows to zero is not a bracket

**Symptom.** A function that is positive everywhere, handed to a bracketing root
finder over an interval that contains no root:

```text
$ build/tests/test_numerics bracket
  FAIL  roots/a bracket whose product underflows is still not a bracket
        expected InvalidArgument from bisection(tiny, 0.0, 2.0)
```

The finder accepted it. It then bisected an interval with no sign change in it
and returned the midpoint with `converged = true` and an error estimate below
the tolerance, which is a wrong answer wearing the evidence of a right one.

**Root cause.** `require(fa * fb <= 0.0, ...)`. The intent is "the ordinates
have opposite signs, or one of them is a root", and the product is a proxy for
it that fails in the small: `1e-200 * 1e-200` underflows to `+0.0`, and
`+0.0 <= 0.0` is true. Two ordinates need only be around 1e-154 for their
product to round to zero in double precision, which is well inside the range a
scaled residual reaches near a root, so the failure is not exotic.

The same proxy appears a third time inside Brent's loop as `fb * fc > 0.0`, the
test that decides whether the contraction point has to be reset. There the
underflow goes the other way: two same signed ordinates whose product rounds to
zero read as "not the same side", the reset is skipped, and the method carries
on with a pair that no longer brackets anything.

**Options.**

- Compare against a small positive threshold instead of zero. Rejected: it
  invents a scale the interface does not have, and it would reject genuine
  brackets whose ordinates are small.
- Take the sign of each ordinate separately. Chosen. The question is about signs
  and nothing else, so asking about signs neither underflows nor overflows.

**Fix.** `detail::brackets_root(x, y)` and `detail::same_side(x, y)` in
`roots.hpp`, used at all three sites. Zero on either endpoint is still a
bracket, because it is already a root and the finders return it as one; a NaN is
decided explicitly rather than left to whichever sign bit the platform's NaN
carries, which keeps the behaviour the product spelling had, since every
comparison against a NaN is false.

**Verification.** Two cases in `tests/unit/test_numerics.cpp`. The underflowing
pair is refused by both bisection and Brent; a genuine sign change whose product
overflows to negative infinity is still accepted, which is the regression the
fix could plausibly have introduced; and an endpoint that is exactly zero, in
both spellings of zero, is still accepted and returned as the root.

```text
$ build/tests/test_numerics bracket
  pass  roots/bisection rejects a bracket that does not change sign
  pass  roots/a bracket whose product underflows is still not a bracket
  pass  roots/an endpoint that is a root is accepted as a bracket
3 passed, 0 failed, 23 filtered out
```

---

## 2026-09-06 CLI-01 A mistyped flag terminates the driver, and a zero repetition count crashes it

**Symptom.** Three command lines, against the tree before this change:

```text
$ build/pnl --size abc
terminate called after throwing an instance of 'std::invalid_argument'
  what():  stoll
  exit -6

$ build/pnl --iterations 12abc
poisson2d_rich_255,65025,jacobi,serial,...,solve,12,0,iteration_cap,...
  exit 0

$ build/pnl --reps 0

  exit -11
```

Signal 6 is `SIGABRT`, signal 11 is `SIGSEGV`. The middle one is the quiet one:
`12abc` was accepted as twelve and a complete result row was printed for a run
the user did not ask for.

**Root cause.** Three faults sharing a function.

`parse()` was called outside the `try` in `main`. Every conversion in it was a
`std::sto*` call, which reports a bad value by throwing `std::invalid_argument`,
and with no handler anywhere that is `std::terminate`. The message those
functions carry is the name of the function, so the diagnostic for a mistyped
grid size was the word "stoll".

`std::stoll` also stops at the first character it cannot use and reports success
for what it read, so `12abc` is twelve to it. A benchmark driver that silently
reinterprets its arguments is worse than one that refuses them.

`--reps 0` passed straight through to the timing loop, which collects one
duration per repetition and then reads `timings[timings.size() / 2]`,
`timings.front()` and `timings.back()`. On an empty vector all three are reads
past the end, and the row it was building would have carried timings nobody
measured.

**Options.**

- Wrap `main` in one `try` and leave the conversions alone. Rejected: it stops
  the terminate but the message is still "stoll", and `12abc` is still twelve.
- Validate `--reps` where it is used. Rejected: the driver has three timing
  paths and the check would have to be in all of them, which is where a fourth
  path forgets it.
- Convert with `std::from_chars`, throw `InvalidArgument` naming the flag and
  the value, validate `--reps` at the point of parsing, and put `parse()` inside
  a `try`. Chosen.

**Fix.** `integer_argument` and `real_argument` in `main.cpp` convert with
`std::from_chars`, which refuses trailing text as well as text that is not a
number at all, and name the flag in the exception. `parse()` refuses a
repetition count below one with a message that says what it would have done.
`main` parses inside its own `try`, separate from the one around the run because
it has to finish before MPI is initialised: the flags are what decide whether
this process is part of a distributed job.

**Verification.** `tests/cli/check_cli_errors.py`, run by CTest as
`test_cli_parse_errors` against the binary this build produced. It asserts a non
zero exit that is a status and not a signal, that "terminate called" appears
nowhere, and that the message names the flag. It also runs `--version` to prove
the driver starts at all, so that three refusals cannot be three failures to
launch.

```text
$ python3 tests/cli/check_cli_errors.py build/pnl
pnl: invalid argument: --size needs a whole number, and 'abc' is not one
  exit 2
pnl: invalid argument: --iterations needs a whole number, and '12abc' is not one
  exit 2
pnl: invalid argument: --reps needs at least one repetition, and 0 would leave
the timings vector empty
  exit 2
check_cli_errors: every bad command line was refused with a message naming the flag
```

---

## 2026-09-06 CONC-04 The shared topology is rewritten under the references it handed out

**Symptom.** Take the cheap report, read it, ask for the probed one, read the
first again:

```text
$ build/tests/test_registry "shared topology"
  FAIL  registry/the shared topology a caller holds is not rewritten underneath it
        the verdict changed from 'not probed' to 'no reliable performance versus
        efficiency split visible from inside the guest: ...' under a held reference
```

**Root cause.** One `TopologyReport` at namespace scope served both passes. The
cheap pass filled it under one `once_flag`; the probing pass replaced it
wholesale under a second:

```cpp
if (!probed.probes.empty()) {
    cached_topology = probed;
}
```

`shared_topology` returns a reference into that object, so the assignment
rewrites what every earlier caller is still holding. On one thread that is the
surprise above. On two it is worse than a surprise: every backend constructor
copies a `TopologyReport` out of the reference it was handed, and copying a
`std::vector` whose source is being reassigned on another thread reads memory
the assignment has already freed. Nothing in the interface says a caller may not
hold the reference, and the reference is the whole of what the function returns.

The second `once_flag` is what makes this look safe and is not what makes it
safe. It guarantees the probe runs once; it says nothing about readers that
arrived before it.

**Options.**

- Return by value. Rejected: a `TopologyReport` carries two vectors and a
  sentence, and every backend construction would copy it whether it needed to or
  not.
- Hand out a `std::shared_ptr<const TopologyReport>`. Workable, and it changes
  the signature every backend factory and every constructor is written against
  for a lifetime problem that a write once object does not have.
- Two objects, each written exactly once. Chosen.

**Fix.** `cheap_topology` and `probed_topology`, each filled inside its own
`call_once` and never touched again; `shared_topology` returns a reference to
whichever the caller asked for. `call_once` is the happens before edge, so a
reader that arrives through it cannot see a partly written report, and a
reference handed out earlier keeps describing what it described.

The split changes no answer. `cpu_for_worker` reads only `logical_cpus` and
`core_leaders` for the none, compact and scatter policies, which is exactly what
the cheap report holds, and the two policies that read `probes` are the two that
ask for the probing pass. The probed report still falls back to the cheap facts
when probing yields no per processor timings, and still takes the probe's
verdict, which is the sentence the `pcore` refusal quotes:

```text
$ build/pnl --backend pthreads --pinning pcore --size 15
pnl: backend failure: pinning policy 'pcore' needs a performance core
classification and this machine did not yield one: no reliable performance
versus efficiency split visible from inside the guest: ...
```

**Verification.** `tests/unit/test_registry.cpp` holds the cheap reference
across a probing call and requires its address, its processor count, its core
leader list and its verdict to be unchanged, then requires both caches to answer
with the same objects on a second call. It costs the one probe, which is two
seconds on this machine. `test_registry` is 10 of 10.

---

## 2026-09-06 BUILD-07 A default argument on a virtual function, and the override that omitted it

**Symptom.** The same call, on the same object, through the two static types it
has:

```text
$ g++-15 -std=c++20 -Iinclude -fsyntax-only defaults_probe.cpp
compiles

$ g++-15 -std=c++20 -Iinclude -fsyntax-only -DPROBE_THROUGH_DERIVED=1 defaults_probe.cpp
defaults_probe.cpp:16:20: error: no matching function for call to
'pnl::backend::MpiBackend::run_ordered(main()::<lambda()>, bool)'
   16 |     mpi.run_ordered([&] { ++through_derived; }, true);
```

`base` is a `Backend&` bound to the same `MpiBackend`. Two arguments are enough
through the base and are not a call at all through the derived class.

**Root cause.** `Backend::run_ordered` was virtual and carried defaults for its
last three parameters; `MpiBackend::run_ordered` overrode it and declared none.
A default argument is part of the declaration the call is resolved against,
which is chosen from the *static* type of the expression, so the two
declarations of one function disagreed about how many arguments it takes. Here
that shows as a compile error. The worse form of the same fault is silent: had
the override supplied different defaults, the call would have compiled through
both types and done different things.

This is not something an override can fix by copying the defaults, because
copying them means maintaining the same values in two places and a later edit to
one of them reintroduces the fault without touching the other.

**Options.**

- Repeat the defaults in the override. Rejected, above.
- Drop the defaults and make every caller pass five arguments. Rejected: the
  three trailing parameters are meaningless for a shared memory backend and the
  call sites that do not need them would be noise.
- The non virtual interface: defaults on a non virtual public wrapper, which
  forwards to a virtual implementation that has none. Chosen, and it is the
  standard answer to exactly this.

**Fix.** `Backend::run_ordered` is a non virtual wrapper carrying the defaults,
and `Backend::run_ordered_impl` is the protected virtual with no defaults on it.
`MpiBackend` overrides the implementation. `HybridBackend` inherits that
override, and the four call sites in `poisson2d.hpp` and `dense_generator.hpp`
are unchanged, since they call the wrapper by the same name as before.

**Verification.** `tests/unit/test_default_args.cpp` records what the
implementation was handed and requires the two spellings of the call to agree,
argument by argument, and to be the documented defaults: an empty view and two
zeros. `tests/mpi/test_mpi.cpp` makes the same two calls on an `MpiBackend`,
which is the pair Section 4.7 names, and that file compiling at all is most of
the assertion.

```text
$ build/tests/test_default_args
  pass  default_args/an ordered sweep sees the same arguments through base and derived
  pass  default_args/an ordered sweep given every argument still receives them
2 passed, 0 failed
```

---

## 2026-09-06 CUDA-04 A grid index that wraps, and a refused launch that reads like a fault

**Symptom.** Two faults on the device path, neither of which any existing test
could reach.

`const int interior = n * n` in the solve driver wraps for `n >= 46341`, and
every `i * stride + j` inside the kernels wraps a little earlier, at `n = 46339`.
`main.cpp` cast `options.size` to `int` and passed it on with no range check, so
`--size 50000` was a solve over a wrapped index rather than a refusal.

And `pnl_cuda_launch_coloured` returned `void`. Section 4.7 is precise about
what was and was not wrong here: launch errors *were* checked at the call site
and execution faults *do* surface at the next synchronisation, so a corrupted
solve was never reported as a success. What was missing is narrower. There was
no `cudaGetLastError()` inside the colour helper, so a red half sweep the driver
refused was reported against the black one; and nothing in the wording said
whether a launch had been refused or a kernel had faulted, which are different
halves of the code to go and look at.

**Root cause.** The index arithmetic is 32 bit throughout the device sources,
which is the right choice for it, and nothing anywhere said what that implies
about the largest problem the kernels can address. An unstated bound is one
nobody can check.

**Options for the index.**

- Widen the kernels to 64 bit indices. Rejected, and this is the interesting
  one: the reduction kernel computes `k / side` and `k % side` per element, and
  a 64 bit integer division on a GPU is several times the cost of a 32 bit one.
  That kernel runs at every residual evaluation and its time is inside
  `kernel_seconds`, which is a published number. Widening would slow a measured
  path to reach sizes no device can hold: at the bound, each of the five arrays
  a solve allocates is 17 GB.
- State the bound, check it once at the entry point, and leave the arithmetic
  alone. Chosen.

**Fix.** `PNL_CUDA_MAX_SIDE = 46338` in `pnl/backend/cuda.hpp`, with the
derivation beside it: the largest index the kernels form is `n * (n + 2) + n`,
and 46338 is the largest `n` for which that stays inside a signed 32 bit
integer. `pnl_cuda_poisson_solve` refuses anything above it before it allocates
or launches, and `run_cuda` in `main.cpp` refuses it before it even looks for a
device, since it is a property of the request and not of the machine. The
interior count is formed as a 64 bit product and narrowed afterwards, so the
multiplication cannot overflow whatever it is handed and the narrowing is safe
by the bound above it.

`pnl_cuda::detail::launch_coloured` returns `cudaError_t`, checks each of its two
launches with `cudaGetLastError()` immediately, and records through a new
`record_launch_error` that names the half sweep and says the launch was refused
before any thread ran rather than a kernel having faulted.

**Verification.** Two cases in `tests/cuda/test_cuda.cpp`, on the RTX 5070.
`pnl_cuda_probe_launch_geometry` is a test hook that launches the colour helper
with the block geometry it is given; 64 by 64 is 4096 threads, four times the
device limit, so the driver refuses it. The case requires the wording and then
requires a legal geometry through the same path to be accepted, so the probe is
reporting the geometry rather than always failing. The bit identity cases either
side of it are unchanged, which is the evidence that no measured number moved.

```text
$ build/tests/test_cuda
  pass  cuda/the Jacobi sweep is bit identical to the CPU
  pass  cuda/the red black Gauss Seidel sweep is bit identical to the CPU
  pass  cuda/red black SOR matches the CPU at the optimal factor
  pass  cuda/conjugate gradient agrees with the CPU to reduction tolerance
        pnl_cuda_poisson_solve was asked for an interior side of 50000, above the
        largest side the device kernels can index, 46338; above that the 32 bit
        grid index wraps
  pass  cuda/an interior side the kernels cannot index is refused before anything runs
        the red half sweep of launch_coloured was rejected at launch at
        src/cuda/rb_gauss_seidel.cu:82: invalid argument. This is a launch
        configuration error, reported before any thread ran, and not an execution
        fault surfacing from an earlier kernel
  pass  cuda/a launch the driver refuses says so, and says it was the launch
        device triad: 549.1 GiB/s
  pass  cuda/the bandwidth probe returns a plausible figure
9 passed, 0 failed

$ build/pnl --backend cuda --size 50000
pnl: --size 50000 is above the largest interior side the device path can index,
46338. The kernels address the padded grid with 32 bit integers and the index
wraps above that; a grid at the limit is already 17 GB per array. Use a smaller
size, or a host backend.
exit 3
```

---

## 2026-09-06 NUM-10 An integration that missed its tolerance by twelve orders and called itself converged

**Symptom.** Decay at rate 50 over the unit interval, a tolerance of 1e-12 and a
minimum step of a tenth, which is far too coarse for that tolerance:

```text
$ build/tests/test_numerics dormand
  FAIL  ode/dormand prince refuses to call a run at the step floor converged
        a run whose worst step missed the tolerance by a factor of
        792173252276.25452 reported itself as converged
```

Ten steps, every one of them out of tolerance, and `converged = true` with
`StopReason::Converged` on the result. `Diagnostics::require_converged` passes
it, which is the interface a caller uses precisely so that it cannot proceed on
a bad answer.

**Root cause.** The acceptance test is

```cpp
if (error <= 1.0 || h <= options.min_step) {
```

and the second half of it is right: once the controller cannot shrink the step
any further, refusing the step means an integration that cannot advance at all.
The fault is that nothing recorded that the second half was what accepted it.
`converged` was set by one thing only, reaching the endpoint, so a run that got
there over steps the controller had given up on was indistinguishable from one
that met the tolerance at every step. `error_estimate` did carry the evidence,
at 7.9e11, but a caller who reads the converged flag never gets to it.

**Options.**

- Refuse the step and stop at the floor. Rejected: the caller then gets a
  partial trajectory and no answer, where what they usually want is the answer
  the integrator could produce, correctly labelled.
- Report `converged = false` with `StopReason::IterationCap`. Rejected: the
  iteration cap was not reached and saying it was sends a reader to the wrong
  option.
- A stop reason of its own. Chosen.

**Fix.** `StopReason::StepFloor`, spelled `step_floor`, and a flag in
`dormand_prince` set when a step with `error > 1` is accepted because `h` has
bottomed out. Reaching the endpoint is still necessary; it is no longer
sufficient. `error_estimate` continues to say by how far the worst step missed.

The enumerator is appended rather than inserted, and nothing in the tree
switches over `StopReason` except its own `to_string`, so no other code has to
change. The result row carries the spelling rather than the value.

**Verification.** A case in `tests/unit/test_numerics.cpp` that forces the floor
rather than approaching it, and first asserts that the endpoint really was
reached and the worst error really was above one, so that a future change which
stopped exercising the floor fails rather than passing vacuously. It then
requires `converged == false`, the reason to be `step_floor`, and
`require_converged` to throw. The two existing Dormand Prince cases, which do
meet their tolerances, still report converged.

```text
$ build/tests/test_numerics dormand
  pass  ode/dormand prince meets its requested tolerance
  pass  ode/dormand prince refuses to call a run at the step floor converged
  pass  ode/dormand prince takes fewer steps at a looser tolerance
3 passed, 0 failed, 24 filtered out
```

---

## 2026-09-06 TEST-01 The test framework had three defects, and nothing tested the test framework

**Symptom.** No failure, which is the shape of every defect in a checking
mechanism: the suite was green and had been green through every phase of this
release. Three separate holes, found by reading `tests/pnl_test.hpp` and the two
copies of its helpers the suites had made.

1. `run_all` caught `pnl::test::Failure` and `std::exception` and nothing else.
   A case that threw anything else, a bare `throw 42` being the canonical
   example, left the loop, left `main` and ended the process in
   `std::terminate`. CTest reports that as a crash with no case name, and every
   case after it in that binary is never attempted, so one bad case hides an
   unknown number of others.
2. `first_difference` in `tests/equivalence/test_equivalence.cpp` returned `0`
   when the two vectors were different lengths. Zero is the index of the first
   element, so a length mismatch was reported as a difference at index 0, and
   both call sites then formatted `candidate[0]` and `reference[0]` of vectors
   one of which may have no element zero at all. A solver that returned an empty
   iterate would have been diagnosed by a read past the end of it.
3. `worst_difference` in `tests/mpi/test_mpi.cpp` looped to
   `min(a.size(), b.size())`. A vector and a truncation of itself therefore came
   back as a worst difference of zero, and the three assertions in that file that
   read `difference <= 1.0e-12` and `difference == 0.0` all passed on one. The
   ordered solver case asserts bit identity against serial at every rank count,
   which is the strongest claim the distributed suite makes, and it was the claim
   most easily satisfied by a wrong length.

**Root cause.** The first is a missing `catch (...)`. The second and third are
the same root cause wearing two faces: the comparison helper was written twice,
once per suite, and neither copy treated "these are not the same shape" as a
difference. Both copies decided what to do about a length mismatch in the middle
of a loop that was about element values, and both chose the answer that let the
loop finish.

**Options.**

- Fix the three in place and leave the helpers where they are. Rejected. Two
  copies is how the defect arrived, and a third suite would have written a
  third copy.
- Move both helpers into `pnl_test.hpp`, make a length mismatch a difference in
  each, and delete the copies. Chosen.
- Have `first_difference` return a struct so a caller cannot index at all.
  Rejected as more than is needed: the callers want a sentence, so they get
  `describe_difference`, which is the only thing that indexes and which checks
  the lengths before it does.

**Fix.** In `tests/pnl_test.hpp`:

- `run_all` is now a thin wrapper over `run_cases(const std::vector<Case>&, ...)`
  and the loop has a `catch (...)` that reports the case by name and counts it
  as a failure. Splitting the list out of the runner is what makes the framework
  testable: a case cannot append to the registry it is itself being iterated out
  of, but it can build a list on the stack and run that.
- `first_difference` returns the first index at which the two vectors differ,
  and on a length mismatch returns the first index the shorter of them does not
  have. Identical still returns minus one, which is what every caller tests.
- `describe_difference` is the sentence a caller prints. It checks both lengths
  before it indexes, so the message that reports a mismatch cannot be the thing
  that reads past the end.
- `worst_difference` returns positive infinity on a length mismatch. Infinity
  fails `<= tolerance` and fails `== 0.0`, which is the only answer that leaves
  the distributed suite's assertions meaning what they say.

The two local copies are deleted and both suites call the shared ones.

**Verification.** `tests/unit/test_framework.cpp`, ten cases, registered first
in `tests/CMakeLists.txt` because a framework defect devalues every gate above
it. The deliberately failing cases run through `run_cases` over a list built in
the case body, and print a banner around themselves so that FAIL lines inside a
passing run read as deliberate.

```text
$ build/tests/test_framework
  pass  framework/a case that throws a non std::exception fails rather than terminating
  pass  framework/a case that throws a string literal fails rather than terminating
  pass  framework/the runner survives a throwing case and runs the ones after it
  pass  framework/a passing list passes and a failing assertion is a failure
  pass  framework/the filter selects by substring and reports what it skipped
  pass  framework/first_difference reports a length mismatch instead of index zero
  pass  framework/describe_difference never indexes a vector that is too short
  pass  framework/worst_difference fails a length mismatch instead of passing it
  pass  framework/the comparison helpers answer false for a NaN rather than throwing
  pass  framework/PNL_REQUIRE_THROWS refuses the wrong exception type
10 passed, 0 failed
```

Two of those cases are only fully asserted under the address sanitizer, which is
where a read of element zero of an empty vector is a report rather than a value:
the asan and ubsan preset runs the `unit` label and therefore runs this binary.

---

## 2026-09-06 NUM-11 Conjugate gradient calls an exact answer a proof that the operator is indefinite

**Symptom.** The one unknown boundary case added to the equivalence suite in
phase B7 failed on the first backend it reached, before any comparison was made:

```text
  FAIL  equivalence/boundary: one unknown is bit identical on every backend and worker count
        unexpected exception: numerical failure: conjugate gradient found a search direction
        with curvature 0.000000, which proves the operator is not positive definite
```

The operator is `Poisson2D(1)`, which is the one by one matrix `[4]`. It is as
positive definite as a matrix can be.

**Root cause.** The breakdown check in `cg.hpp` reads
`if (!(curvature > 0.0))`, and a curvature of zero has two causes that the
check does not tell apart.

One is a genuinely indefinite operator, which is what the message describes.
The other is a search direction that is the zero vector, and the zero vector has
zero curvature under every operator there is, including the most positive
definite one imaginable. Nothing can be concluded from it.

The recurrence produces the zero direction as soon as it has solved the system
exactly. At one unknown that is after one iteration: with `A = [4]`, `x0 = 0`
and `r0 = b`, the step is `alpha = b^2 / 4b^2 = 1/4`, so `x1 = b/4` and
`r1 = b - 4 (b/4) = 0` exactly. Then `beta = 0 / rr` is zero, `p = r + 0 p` is
the zero vector, and the next iteration computes a curvature of zero and throws.

It is reachable only under `RunMode::FixedIterations`. Under `ToTolerance` a
zero residual meets any tolerance and the loop breaks before it comes round
again. Conjugate gradient terminates in at most `n` steps in exact arithmetic,
so a fixed run of 25 iterations reaches an exactly zero residual on a problem of
fewer than about 25 unknowns and on no other, and the equivalence suite ran at
63 squared and at order 180. That is the whole reason it survived to release
1.1.0: nothing in the tree had ever run a fixed length solve on a problem small
enough to finish.

**Options.**

- Leave it and exclude conjugate gradient from the small sizes. Rejected. The
  boundary sizes exist to find exactly this, and a suite that skips the method
  that fails is not a suite.
- Loosen the check to a tolerance, `curvature > small`. Rejected, and it is the
  tempting wrong answer: it would swallow a genuinely indefinite operator whose
  curvature happened to be small, which is the one thing this check exists to
  refuse. The two cases differ in kind, not in magnitude.
- Keep iterating with a zero step so the fixed iteration count is honoured
  exactly. Rejected: it spends the remaining iterations doing arithmetic on
  zeros, which is neither a measurement of anything nor an answer.
- Answer the zero residual before the curvature is computed. Chosen.

**Fix.** A guard at the top of the loop in `include/pnl/solvers/cg.hpp`:

```cpp
if (rr == 0.0) {
    diagnostics.converged = true;
    diagnostics.reason = StopReason::Converged;
    break;
}
```

`rr` is the residual inner product the recurrence already carries, so the guard
costs one comparison per iteration and no reduction. `finalise_reason` relabels
the stop as an iteration cap under `FixedIterations`, which is correct and
deliberate: a fixed run makes no claim about convergence. What changes is that
it is a stop rather than an exception.

**What this cannot have changed.** The guard is only reachable when `rr` is
exactly zero, and every path that reaches it previously threw. No run that
completed before this change takes a different path after it, so no committed
number moves. The twelve golden files of phase B7 were generated before the
change and are reproduced bit for bit after it, which is the check that says so
rather than the argument that it must be so.

**Verification.** The boundary case that found it, at one unknown on both
problems, every solver in the registry, every shared memory backend and every
worker count this machine has:

```text
$ build/tests/test_equivalence boundary
filter: boundary
  pass  equivalence/boundary: the reduction is bit identical across the chunk count switch
  pass  equivalence/boundary: solves at the chunk count switch are bit identical
  pass  equivalence/boundary: an empty problem is refused the same way on every backend
  pass  equivalence/boundary: one unknown is bit identical on every backend and worker count
4 passed, 0 failed, 6 filtered out
```

and the golden files, unmoved:

```text
$ build/tests/test_golden
  pass  golden/every solver in the registry has a committed iterate
  pass  golden/the committed iterates are reproduced bit for bit
  pass  golden/the hexadecimal round trip is exact, so the files are the values
  pass  golden/the comparison distinguishes a signed zero from a zero
4 passed, 0 failed
```
