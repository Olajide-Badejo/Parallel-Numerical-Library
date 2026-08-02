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
