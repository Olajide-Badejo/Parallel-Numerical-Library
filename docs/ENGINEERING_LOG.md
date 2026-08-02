# Engineering log

Dated entries, written as problems happen. Each carries symptom, root cause,
options considered, the fix and why it was chosen, and how it was verified.
Phase 8 converts this into `report_debug/debug_report.pdf`, grouped by theme.

---

## 2026-08-02 ENV-01 nvcc rejects the project compiler, and rejects its own default too

**Symptom.** `cmake -S . -B build` failed inside `project(... CUDA)` with a
compiler identification error. Running `nvcc` by hand on a trivial kernel with
the system default host compiler failed differently, inside a system header:

```
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
|---|---|---|---|
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
