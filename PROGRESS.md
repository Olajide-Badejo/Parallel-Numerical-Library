# Progress

One entry per phase, with the checks that gated it and the output they produced.
Phases follow Section 16 of `docs/BUILD_SPECIFICATION.md`.

## Toolchain as installed

Recorded at Phase 0 and verified, not assumed. Where the installed version
differs from the floor the specification names, the substitution and its reason
are given.

| Component | Specification floor | Installed | Note |
| --- | --- | --- | --- |
| OS | Windows 11 Pro, WSL2 Ubuntu | Windows 11 Pro 10.0.26200, Ubuntu 26.04 LTS, kernel 6.18.33.2 | as specified |
| CPU | i7-14700K, 8 P plus 12 E, 28 threads | i7-14700K, 28 logical processors visible in guest | guest sees a synthesised uniform topology, see ENV-04 |
| RAM | 32 GB, about 10 GB free | 31.7 GB host, 12 GB budgeted to WSL, 10 GB free in guest | `.wslconfig` predates this project and carries a documented crash history |
| GPU | RTX 5070 12 GB, sm_120 | RTX 5070, 12227 MiB, compute capability 12.0, driver 610.62 | as specified |
| Host compiler | GCC 16.1 | GCC 16.0.1 20260322 experimental | 16.1 is not released; 16.0.1 is the newest packaged 16. Chosen over the stable GCC 15.2 on merit: it provides `<mdspan>`, which 15.2 lacks, and reports OpenMP 5.2 against 15.2's 4.5 |
| C++ standard | C++23 | `__cplusplus` 202302 | as specified |
| CMake | 4.4 | 4.4.0 via pipx | distribution ships 4.2.3, which is below the floor |
| MPI | OpenMPI 5.0.x | OpenMPI 5.0.10 | advertises MPI 3.1, implements MPI 4 entry points, see ENV-03 |
| OpenMP | 6.0 cited | GCC reports `_OPENMP` 202111, which is OpenMP 5.2 | 6.0 constructs are not used; the restriction is deliberate and asserted at compile time |
| CUDA | 13.3 | 13.3, V13.3.73 | host compiler forced to g++-14, see ENV-01 |
| CUDA host compiler | not specified | GCC 14 | nvcc 13.3 rejects GCC above 15 and miscompiles GCC 15 headers |
| Python | 3 with matplotlib, pandas | 3.14.4, matplotlib 3.10.7, pandas 2.3.3, PyYAML, tqdm | as specified |
| LaTeX | TeX Live with latexmk | latexmk 4.87 | as specified |
| Other | | ninja 1.13.2, clang-format 20.1.7, ruff 0.15.22, poppler-utils | |

### Deliberate deviations from the specification

1. **WSL processor count.** Section 3 asks for `processors=20` in `.wslconfig`.
   The file already sets `processors=28`, exposing every host logical processor,
   which is a superset of what the scaling sweep to 20 workers needs. It also
   carries a dated comment explaining that a lower count makes the hybrid core
   comparison impossible to measure, and documents a machine crash the memory
   budget in the same file exists to prevent. Left alone, deliberately.

2. **Compiler.** GCC 16.0.1 rather than 16.1, because 16.1 does not exist yet.
   See the table above for why 16 was preferred to the stable 15.2 regardless.

   **Corrected 2026-09-05, Phase A0.6.** That preference rested on two claims
   that were not true. The header named in the table is not included by any
   translation unit in this project, and `openmp.hpp` asserts OpenMP 4.5, so the
   5.2 that GCC 16 reports was never a level the code used. GCC 16.0.1 is also
   an unreleased trunk snapshot that a reader cannot install. Release 1.1.0 is
   built and measured with the released GCC 15.2.0 instead; see the Release
   1.1.0 toolchain table below and decision 20 in `docs/DESIGN_DECISIONS.md`.
   The table above is left as the record of what produced the 1.0.0 numbers.

3. **OpenMP level.** The 6.0 specification is cited as the reference document, as
   asked, but GCC implements to 5.2 and the code is restricted to that. The build
   asserts `_OPENMP >= 201511` so a downgraded toolchain fails loudly.

---

## Phase 0: toolchain, repository, dash lint

Done.

- Every version above verified by running the tool, not by reading a package
  list. Four environment findings recorded as ENV-01 to ENV-04 in the
  engineering log; two of them changed the build's structure.
- `scripts/check_no_dashes.py` written and self tested. It covers the two banned
  characters, eight visually identical stand ins, the LaTeX `--` and `---`
  ligatures in prose only, and compiled PDFs through `pdftotext`. It correctly
  ignores `%` comments, `\url`, `\verb`, `verbatim` and `lstlisting`.
- Its first run found an em dash in the build specification itself (STYLE-01).
- Repository skeleton, MIT licence, `.clang-format` and `.gitignore` in place
  before the first commit.

Gate: `check_no_dashes.py` reports the tree clean, and the linter's own self
test passes.

## Phase 1: numerics core and the serial solver zoo

Done.

- Numerics: bisection, Newton, Brent; adaptive Simpson, Gauss Legendre with
  nodes computed rather than tabulated, Romberg; RK4 and adaptive Dormand
  Prince 5(4); dense LU with partial pivoting, Householder QR, and the Thomas
  algorithm.
- Twelve solvers behind one interface: Richardson, Jacobi, forward, backward,
  symmetric and red black Gauss Seidel, SOR, SSOR, red black SOR, block Jacobi,
  block Gauss Seidel, and conjugate gradient. The nine of Objective 1 plus the
  three red black variants the GPU comparison needs.
- Two problem families: the 2D Poisson five point stencil with closed form
  theory, and seeded dense systems in diagonally dominant and symmetric positive
  definite flavours.
- Every result carries value, error estimate, iteration count, evaluation count,
  converged flag and an explicit stop reason.

Gate: 22 unit assertions across `test_numerics` and `test_solvers` pass,
including every solver on a hand checkable 4 by 4 system and on a 15 by 15
Poisson problem.

Findings: NUM-02 and NUM-03, both real divergence bugs in Richardson, and
NUM-04, the measured rounding floor of stationary iteration.

## Phase 2: convergence and theory ratios

Done. Thirteen assertions, each comparing a measurement against a closed form
prediction rather than a previous run.

Measured on the 2D Poisson model problem:

| Claim | Theory | Measured |
| --- | --- | --- |
| Jacobi contraction factor | cos(pi h) | matches to 1e-3 relative at n = 31 and 63 |
| Jacobi to Gauss Seidel iteration ratio | 2, from rho_GS = rho_J^2 | 2.00 within 2 percent |
| Forward against backward Gauss Seidel | equal rates on a symmetric operator | equal within 1 percent |
| SOR at Young's omega* = 2 / (1 + sin(pi h)) | a minimum | beats every perturbation of plus or minus 0.02 and 0.05 |
| Jacobi iteration count growth, grid doubled | O(n^2), factor about 4 | greater than 3 |
| Optimal SOR growth, grid doubled | O(n), factor about 2 | less than 2.6 |
| Red black against natural ordering Gauss Seidel | small penalty | ratio in [0.95, 1.3] |
| Line Jacobi to line Gauss Seidel | 2 | 2.00 within 0.15 |
| Conjugate gradient growth, grid doubled | O(n) from kappa = O(h^-2) | 41 to 93 iterations, factor 2.27 |
| RK4 global error order | 4 | 4.00 within 0.15 |
| Composite Simpson order | 4 | 4.00 within 0.15 |
| Five point stencil order | 2 | 2.00 within 0.1 |
| Discretisation error constant | pi^2 / 12 = 0.8225 | 0.823 |

Finding: NUM-01. The manufactured solution is an exact eigenvector of the
stencil, which makes conjugate gradient terminate in one iteration at every grid
size. Both right hand sides are now provided and each is used where it is
honest. A test now asserts the one iteration behaviour so the trap stays
recorded.

## Phase 3: shared memory backends and equivalence

Done. OpenMP, POSIX threads and a `std::jthread` pool, each idiomatic
underneath a common interface of `parallel_for`, `reduce`, `barrier` and an
ordered pass.

The equivalence gate is asserted in its strong form: with the deterministic
reduction mode, every backend at every worker count produces **bit identical**
iterates, not merely close ones. That is possible because the reduction chunk
grid depends on the problem size alone and never on the worker count, so
partials are always combined in the same order.

Gate: `test_equivalence` passes, covering every applicable solver on both the
Poisson and dense problems, across serial, OpenMP, pthreads and jthread, at 1,
2, 3, 4, 7, 8 and 16 workers, plus static against dynamic scheduling, plus the
remainder aware partition over 6 sizes and 33 part counts.

Finding: CONC-01, a destructor ordering hazard in the jthread pool, found by
inspection and fixed before it could reproduce.

## Phase 4: MPI and hybrid backends

Done. Remainder aware block row decomposition, halo exchange by two
`MPI_Sendrecv` calls, and a general `gather_rows` primitive that collects each
rank's range rather than assuming it.

Natural ordering Gauss Seidel keeps exact sequential semantics across ranks
through a pipelined token chain, so it is bit identical to serial at every rank
count and shows essentially no speedup. Both facts are the intended result.

Gate: `test_mpi` passes at 1, 2 and 4 ranks, on sizes chosen to leave a remainder
at each. Communication time is recorded separately for halo, reduction, barrier
and ordered pass.

Findings: MPI-01, four test failures from one cause, the returned iterate being
left distributed while the tests compared the whole vector. MPI-02, block ranges
and row ranges differ, found by review before it could corrupt anything.

## Phase 5: CUDA and bandwidth probes

Done. Jacobi, red black Gauss Seidel, red black SOR and conjugate gradient on the
device, with the whole iteration resident in device memory and transfer time
reported separately. Natural ordering Gauss Seidel is deliberately absent.

Gate: `test_cuda` passes. GPU Jacobi and red black sweeps are **bit identical**
to the CPU; conjugate gradient agrees to reduction tolerance, which is stated
rather than hidden. Device and host red black runs agree on iteration count
exactly.

Findings: ENV-05, the CUDA host compiler's library directory hijacking the link,
which took two wrong guesses to diagnose. CUDA-01, kernels in a header becoming
duplicate device stubs. CUDA-02, the host contracting `a*b + c*d` into an FMA
while the device did not, which is why contraction is now off on both sides.
CUDA-03, a host pointer given to a device to device copy.

## Phase 6: full sweep and comparison study

Done. 440 declared configurations across ten blocks, 425 rows recorded and 15
correctly reported as inapplicable (conjugate gradient on a non symmetric
system). Sweep wall clock 21 minutes 5 seconds.

Headline measurements are in the README and the results chapter. The one that
was not anticipated: the scaling knee sits at three to five workers, not at the
eight performance cores, because the memory system saturates first.

Findings: SWEEP-01, the resume check sitting after the work rather than before
it. SWEEP-02, two quadratic methods declared in a block meant for linear ones.
SWEEP-03, two blocks colliding in the resume identity, which silently removed 32
measurements. SWEEP-04, efficiency above one hundred percent, which turned out
to be the roofline model correctly announcing that a cache resident problem is
not streaming.

## Phase 7: documentation from real numbers

Done. `docs/backends.md`, `docs/solvers.md`, `docs/comparison_methodology.md`,
`docs/DESIGN_DECISIONS.md`, CONTRIBUTING, CHANGELOG and the README, all written
from the measured summary rather than from expectations.

## Phase 8: reports

Done. Three PDFs, all built by `make reports` and all dash clean including the
compiled output:

| report | pages |
| --- | --- |
| `report/main.pdf` | 37 |
| `report_debug/debug_report.pdf` | 16 |
| `report_for_me/report_for_me.pdf` | 16 |

Finding: the compiled PDF check earns its place here. The source linter passed a
`\verb|--fmad=false|` span that LaTeX could not honour inside a macro argument,
and only the check on the rendered PDF caught the en dash it produced.

**Note added 2026-09-06, at phase E4.** The third report,
`report_for_me/report_for_me.pdf`, is private and is not published. Its source
directory is ignored by `.gitignore` and is not in the repository, so a reader
of this file cannot obtain the document the table above lists, and neither the
`make reports` target nor `make all` builds it any more. The entry above is left
as it was written, because this file is a log of what was done. Two documents
are published, both under `assets/reports/`.

## Phase 9: final QA

Done.

- `make clean && make all` from a clean tree: exit 0, 23 minutes end to end.
- Dash check clean across 105 files including all three compiled PDFs.
- 10 of 10 test binaries pass: unit, convergence, equivalence, MPI at 1, 2 and 4
  ranks, CUDA, and both style gates.
- `ruff` clean; the C++ builds with `-Wall -Wextra -Wpedantic -Werror` and no
  warnings.
- README states measured wall clock, replacing the estimates of Section 2.

One claim was withdrawn during final review. An early bandwidth probe peaked at
exactly eight workers, matching the performance core count, which looked like
indirect evidence of the topology the guest hides. Repeated on an idle machine
the peak moved to four with eight 1.8 percent behind, so the coincidence does not
reproduce and the inference is gone from the report, replaced by a sentence
recording that it failed.

## Release 1.1.0

The phases below follow Section 12 of the V2 build specification, which is a
separate document from the V1 specification the phases above follow. Every
gate's output is quoted from the run rather than summarised, so a reader can
tell a gate that passed from a gate that was declared to have passed.

### Release 1.1.0 toolchain

Recorded at Phase A0.6 by running each tool, not by reading a package list. The
"Toolchain as installed" table near the top of this file is the 1.0.0 record and
is left alone. One entry deliberately changes: the host compiler. GCC 16.0.1 was
an unreleased trunk snapshot nobody outside this machine can install, so every
number published in 1.1.0 comes from the released GCC 15.2.0 instead. See
decision 20 in `docs/DESIGN_DECISIONS.md`.

| Component | Version, as the tool reports it | Note |
| --- | --- | --- |
| OS | Ubuntu 26.04 LTS, kernel 6.18.33.2-microsoft-standard-WSL2 | unchanged from 1.0.0 |
| C++ compiler | `g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0` | the publication compiler for 1.1.0 |
| Fortran compiler | `GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0` | same GCC major version, which Section 9.5 requires. Installed and recorded now, used from release 1.2.0 |
| CUDA host compiler | `g++-14 (Ubuntu 14.3.0-14ubuntu1) 14.3.0` | unchanged, see ENV-01 |
| C++ standard | `__cplusplus` is `202002L` under `-std=c++20` | declared at Phase A0.6, and what the code has always used |
| OpenMP | `_OPENMP` is `201511`, which is OpenMP 4.5 | exactly what `openmp.hpp` asserts. GCC 16 reported 202111, which nothing in the tree used |
| CMake | `cmake version 4.4.0` | via pipx; the distribution ships 4.2.3 |
| Ninja | `1.13.2` | |
| MPI | `Open MPI: 5.0.10`, from `ompi_info` | `mpirun --version` prints no version on this image, because the help file it wants is absent from both the pmix2 and the prrte3 packages. `mpicxx` drives the default `g++`, which is 15.2.0, so the wrapper and the project compiler now agree |
| CUDA | `Cuda compilation tools, release 13.3, V13.3.73` | unchanged |
| Python | `Python 3.14.4`, with matplotlib 3.10.7, pandas 2.3.3, numpy 2.3.5, PyYAML 6.0.3, tqdm 4.67.3 | unchanged |
| clang-format | `clang-format version 20.1.7` | |
| latexmk | `Latexmk ... Version 4.87` | |
| ruff | `ruff 0.15.22` | the selection is pinned in `ruff.toml` and the version in the workflow |
| binutils | `GNU ld (GNU Binutils for Ubuntu) 2.46` | recorded now because Part D drives GNU as from the same binutils |

### Phase A0: stop the leak, and fix the reason every row is dirty

Done. Five changes in one commit, before anything else in V2. Nothing about the
numerics, the solvers or the sweep changed.

1. `.gitignore` now ignores `BUILD_SPECIFICATION.md` unanchored, so the root
   copy and the `docs/` copy are both covered, and it adds `CLAUDE.md`,
   `Parallel Numerical library V2.md`, `BOARD.md`, `tasks/` and `.claude/`.
   These are instructions to the agent that built the project and the working
   files that schedule that work, not project documentation. `.claude/` had
   been covered only by `.git/info/exclude`, which does not travel with a
   clone.
2. The history is left alone, deliberately. See below.
3. `report/main.pdf` and `report_debug/debug_report.pdf` were tracked and
   ignored at the same time, so `make clean` deleted tracked files and every
   build that followed a clean was dirty before it compiled anything. Both are
   untracked with `git rm --cached`; the published copies under
   `assets/reports/` stay tracked and are what the README links.
4. The dirty check in `CMakeLists.txt` now takes a pathspec, so only
   `CMakeLists.txt`, `cmake`, `include`, `src`, `tests`, `benchmarks`,
   `scripts` and `Makefile` count towards dirtiness. A regenerated figure or a
   fresh `summary.csv` can no longer stamp a result row `.dirty`.
5. The `experiments/results` block re includes `archive/` and then
   `archive/**`, because git does not descend into an excluded directory, and
   re includes `manifest-*.json`.

Two findings are recorded, `PROV-01` and `PROV-02` in
`docs/ENGINEERING_LOG.md`. PROV-02 corrects the specification on one point: of
the two paths Phase A8 writes, only the archive directory was invisible to
`git add -A`; the manifest was already re included, which is the worse of the
two failures, because a committed manifest names an archive that was never
committed.

**The decision on A0.2, and the gate line it does not meet.** The history is
not rewritten.
`git filter-repo --path docs/BUILD_SPECIFICATION.md --invert-paths`
and a force push would change every commit hash after
`5faf41d`, which severs the `commit` column of every published result row and
the `v1.0.0` tag from the objects they name. That would destroy the provenance
Part A exists to repair, in order to hide a build specification, which is not a
credential. The force push would also break every existing clone. The ignore
rules stop a re add and the past is left as it is. This is recorded as decision
18 in `docs/DESIGN_DECISIONS.md`, and the option stays open for the owner to
take later at the same cost. The consequence is that the A0 gate line
`git log --all -- docs/BUILD_SPECIFICATION.md` returning nothing is **not
met**, and deliberately so:

```text
$ git log --all --oneline -- docs/BUILD_SPECIFICATION.md
a619b58 Prepare the repository for publication
5faf41d Add numerical core, solver zoo and shared memory backends
```

**Gate.** Run inside WSL2 Ubuntu, from the committed tree.

```text
$ git check-ignore -v --no-index BUILD_SPECIFICATION.md CLAUDE.md \
      "Parallel Numerical library V2.md" BOARD.md tasks/PROTOCOL.md \
      .claude/settings.json
.gitignore:69:BUILD_SPECIFICATION.md	BUILD_SPECIFICATION.md
.gitignore:70:CLAUDE.md	CLAUDE.md
.gitignore:71:Parallel Numerical library V2.md	Parallel Numerical library V2.md
.gitignore:72:BOARD.md	BOARD.md
.gitignore:73:tasks/	tasks/PROTOCOL.md
.gitignore:74:.claude/	.claude/settings.json

$ git ls-files | grep -E '^report(_debug)?/.*\.pdf$'
(no output, grep exits 1)

$ git status --porcelain
(no output)

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, N file(s) scanned
```

`check_no_dashes` reports the tree clean. Its file count is written as N above
rather than quoted, because the checker walks the working directory rather than
the index: it counts the compiled PDFs that `make clean` deletes, and the
private working files that are ignored but still on disk, so the number is a
property of the directory at that moment and not of this commit.

`make clean && make build` then completes, and one run of the solver from that
build prints the commit stamp in column 27:

```text
$ build/pnl --solver jacobi --backend serial --size 4 --mode fixed \
      --iterations 1 --reps 1 | awk -F, '{print $27}'
```

It prints the twelve character short hash of this commit with no `.dirty`
suffix. Before this commit the same command printed a hash that ended in
`.dirty` on every run, and `summary.csv` carries 850 such rows, 425 at
`4abf914a7ea2.dirty` and 425 at `cd57032941a8.dirty`. The hash itself is not
quoted here for the obvious reason that a file cannot carry the hash of the
commit that adds it; the run is reproducible from this tree with the command
above.

`make test` was not run. The only compiled change is the value of the commit
stamp, and the phase that touches a solver is A1.

### Phase A0.6: declare what is actually used, and choose the publication compiler

Done, in one commit. Two decisions and the document repairs they force. No
solver, no kernel and no measurement changed; the only compiled change is which
compiler produces the binary.

**Decision 19: release 1.1.0 is Parts A, B, E2, E4 and E5.** Part C, Part D, E1
and E3 are 1.2.0. This is Section 11.5's own recommendation, taken because a
single definition of done over 57 to 72 sessions leaves the repository half
migrated for months. Recorded as decision 19 in `docs/DESIGN_DECISIONS.md`,
which also records that `PNL_REDUCTION_ACCUMULATORS` stays at 1 for 1.1.0, per
Section 10.4, so no committed residual changes and the release stays bit
compatible with 1.0.0. The constant itself arrives in Part D; there is nothing
to code for it now.

**Decision 20: the publication compiler is GCC 15.2.0, with gfortran 15.2.0.**
Both are stock Ubuntu 26.04 packages. GCC 14.3.0 was the other released
candidate. 15 wins because `-fdo-concurrent=parallel` and DO CONCURRENT REDUCE
arrived in GCC 15, so the Fortran work of 1.2.0 needs no second compiler switch,
which would otherwise throw away a measurement session. GCC 16.0.1, which
produced the 1.0.0 numbers, is an unreleased trunk snapshot a reader cannot
obtain. The CUDA host compiler stays g++-14, because nvcc 13.3 cannot parse GCC
15's libstdc++ headers (ENV-01); the C ABI boundary of decision 8 is what makes
that coexistence work and it keeps working. This lands before Phase A1 because
switching compilers changes every timing number, and after Phase A8 it would
throw the whole measurement session away.

**The standard is now C++20.** `CMAKE_CXX_STANDARD` is 20, `pnl_core` carries
`target_compile_features(pnl_core PUBLIC cxx_std_20)` so a consumer inherits the
requirement, and the `project()` description says C++20. The only features above
C++17 in the tree are `std::jthread`, `std::barrier` and `std::span`, all C++20.
The specification says a grep for C++23 only constructs returns zero hits.
Verified:

```text
$ grep -rn -E 'std::expected|mdspan|std::print\(|std::println|std::to_underlying|std::unreachable|if consteval|\[\[assume|std::flat_|std::ranges::to|this auto|operator\[\] *\([^)]*,' include src tests benchmarks
(no output, grep exits 1)
```

`std::print` is written with its opening parenthesis so it does not match the
`std::printf` calls in `src/main.cpp` and the test harness, which are C, not
C++23.

**Documents corrected.** `CONTRIBUTING.md` claimed the build needs GCC 16 for a
header this project never includes and for OpenMP 5.2; it now states the real
floor, which is C++20 and OpenMP 4.5, roughly GCC 11 or Clang 14, and it names
GCC 15.2.0 as the publication compiler. The README badge reads C++20 and the environment table now says its
numbers came from the 1.0.0 trunk compiler and that 1.1.0 replaces them.
`docs/backends.md` and `include/pnl/backend/openmp.hpp` said GCC 16 reports
OpenMP 5.2; both now say what the build actually asserts, which is 4.5.
`jthread_pool.hpp` and `docs/backends.md` called the jthread pool C++23; it is
C++20. `include/pnl/backend/cuda.hpp` and the ENV-05 comments in `CMakeLists.txt`
no longer name g++-16. In the report, `main.tex`, `conclusion.tex`,
`parallelisation.tex` and `discussion.tex` carried the same claims in prose;
they are corrected, and no number was touched. Phase E4 does the full report
pass and Phase A8b rebuilds the tracked PDFs.

`docs/DESIGN_DECISIONS.md` decision 8 keeps its original text and gains an
"Amended 2026-09-05" paragraph: both costs it named against dropping below GCC
16 were imaginary, the decision it reached is unaffected because nvcc cannot
parse GCC 15's headers either, and adopting the header it names stays future
work rather than a plan. `docs/ENGINEERING_LOG.md` ENV-01 keeps its original
text and gains a "Correction, 2026-09-05" paragraph saying the same thing about
the entry that decision 8 was written from.

**Gate.** Run inside WSL2 Ubuntu through `tasks/run.sh`.

```text
$ g++-15 --version | head -1
g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0

$ gfortran-15 --version | head -1
GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0

$ make clean && make build
(elided: the ninja lines for 24 of 24 targets, all built, no warning under
 -Wall -Wextra -Wpedantic -Werror. The configure lines that matter:)
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5") found components: CXX
-- pnl: OpenMP 4.5 enabled, spec date 201511
-- pnl: MPI 3.1 enabled (/usr/bin/mpiexec)
-- pnl: dropping /usr/lib/gcc/x86_64-linux-gnu/14 from the CUDA implicit link directories
-- pnl: CUDA enabled, arch 120, host /usr/bin/g++-14
-- pnl: build type Release, C++ compiler GNU 15.2.0
[24/24] Linking CXX executable tests/test_cuda

$ grep -m1 'CMAKE_CXX_COMPILER:' build/CMakeCache.txt
CMAKE_CXX_COMPILER:STRING=/usr/bin/g++-15

$ grep -m1 'main.cpp' build/compile_commands.json | grep -o -- '-std=c++20'
-std=c++20

$ grep -m1 'main.cpp' build/compile_commands.json | grep -o -- '-march=native'
-march=native

$ make test
 1/10 Test  #1: test_numerics ....................   Passed    0.00 sec
 2/10 Test  #2: test_solvers .....................   Passed    0.01 sec
 3/10 Test  #3: test_convergence .................   Passed    0.83 sec
 4/10 Test  #4: test_equivalence .................   Passed    1.58 sec
 5/10 Test  #6: test_dash_checker_self ...........   Passed    0.25 sec
 6/10 Test  #7: test_mpi_1rank ...................   Passed    0.29 sec
 7/10 Test  #8: test_mpi_2rank ...................   Passed    0.27 sec
 8/10 Test  #9: test_mpi_4rank ...................   Passed    0.31 sec
 9/10 Test  #5: test_no_dashes ...................   Passed    2.61 sec
10/10 Test #10: test_cuda ........................   Passed    4.00 sec

100% tests passed out of 10

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 117 file(s) scanned

$ git ls-files -z -- '*.md' '*.tex' '*.hpp' '*.cpp' '*.txt' | xargs -0 grep -l mdspan
PROGRESS.md
docs/DESIGN_DECISIONS.md
docs/ENGINEERING_LOG.md
report_debug/debug_report.tex
```

The whole of that build and that test run used g++-15. The two lines worth
reading twice are `Found OpenMP: TRUE (found version "4.5")` and `spec date
201511`: under GCC 16 the same configure printed 5.2, and every test still
passes at 4.5, which is the direct evidence that the level the documents claimed
to need was never needed. The equivalence suite passing unmodified is the other
one, because it is the test that asserts bit identical iterates across every
shared memory backend, and it passes under a compiler the project had not used
before.

The build above was made from a dirty tree, since the tree still held this
phase's own edits, so the stamp it wrote reads `7025e8c29c4f.dirty`. After the
commit the tree is clean, `git status --porcelain` prints nothing, and

```text
$ make build
$ build/pnl --solver jacobi --backend serial --size 4 --mode fixed \
      --iterations 1 --reps 1 | awk -F, '{print $27}'
```

prints the twelve character short hash of this commit with no `.dirty` suffix.
The hash itself is not quoted, for the same reason it was not quoted under Phase
A0: a file cannot carry the hash of the commit that adds it.

Two things the gate does not show clean, and why.

1. Four tracked files still name the header, and three of them do so in lines
   this phase was told not to rewrite: the Rejected paragraph of decision 8,
   the Options and Verification paragraphs of ENV-01, and the host compiler row
   of the 1.0.0 toolchain table near the top of this file. Each of those now
   sits beside a dated correction saying the claim was false. The remaining
   mentions in this file are the construct grep quoted above and this note; the
   only forward looking one anywhere is the future work sentence in decision 8.
2. `report_debug/debug_report.tex` narrates ENV-01 and ENV-05 and repeats the
   same two false claims. It is the LaTeX rendering of the engineering log
   rather than a claim about the current toolchain, and it was left for Phase
   E4, which does the report pass, and Phase A8b, which rebuilds the tracked
   PDFs. Recorded here so it is not lost.

No engineering log entry was needed. The GCC 15.2.0 build surfaced no fault:
`make clean && make build` and the full `make test` both pass unmodified,
including the MPI runs at 1, 2 and 4 ranks and the CUDA tests on the device.

### Phase A0.7: apply clang-format once, in its own commit

Done, in one commit that changes formatting and this entry and nothing else.
`.clang-format` and `make format` have existed since 1.0.0 and had never been
run over the tree, so the first run rewrote most of `include/` and `src/`. It
lands here, before Phase A1 goes near a solver, so that every measurement diff
in Part A is readable instead of being buried under formatter churn.

**The clang-format the runner resolves is 20.1.7.** Two are installed: a pipx
one at `/home/elijah/.local/bin/clang-format` and Ubuntu's 21.1.8. The pipx
directory is first on the PATH `tasks/run.sh` exports, so it is the one
`make format` used and the one this commit's formatting is fixed to. Phase B2
must pin exactly `clang-format 20.1.7` in CI: 21 changes enough defaults that a
newer binary would fail `--dry-run --Werror` on a tree this one calls clean.

```text
$ which clang-format
/home/elijah/.local/bin/clang-format

$ clang-format --version
clang-format version 20.1.7
```

**The diff.** 46 files, 999 insertions and 661 deletions.

```text
$ git diff --stat
 include/pnl/backend/backend.hpp          |  26 ++-
 include/pnl/backend/chunking.hpp         |   8 +-
 include/pnl/backend/cuda.hpp             |  22 ++-
 include/pnl/backend/hybrid.hpp           |   6 +-
 include/pnl/backend/jthread_pool.hpp     |   7 +-
 include/pnl/backend/mpi.hpp              |  35 ++--
 include/pnl/backend/openmp.hpp           |  11 +-
 include/pnl/backend/pthreads.hpp         |   8 +-
 include/pnl/backend/serial.hpp           |   4 +-
 include/pnl/backend/topology.hpp         |  28 +--
 include/pnl/core/diagnostics.hpp         |  19 +-
 include/pnl/core/error.hpp               |  13 +-
 include/pnl/core/types.hpp               |   1 +
 include/pnl/numerics/lu.hpp              |  20 +-
 include/pnl/numerics/ode.hpp             |  43 +++--
 include/pnl/numerics/qr.hpp              |   8 +-
 include/pnl/numerics/quadrature.hpp      |  73 ++++++--
 include/pnl/numerics/roots.hpp           |  37 ++--
 include/pnl/problems/dense_generator.hpp |  44 +++--
 include/pnl/problems/poisson2d.hpp       |  44 +++--
 include/pnl/problems/problem.hpp         |  17 +-
 include/pnl/progress.hpp                 |  24 ++-
 include/pnl/solvers/block_solvers.hpp    |  14 +-
 include/pnl/solvers/cg.hpp               |   8 +-
 include/pnl/solvers/gauss_seidel.hpp     |  20 +-
 include/pnl/solvers/jacobi.hpp           |   5 +-
 include/pnl/solvers/registry.hpp         |  13 +-
 include/pnl/solvers/richardson.hpp       |  12 +-
 include/pnl/solvers/sor.hpp              |  15 +-
 include/pnl/solvers/splitting.hpp        |  18 +-
 src/backend/factory.cpp                  |   6 +-
 src/backend/mpi_runtime.cpp              |  77 ++++++--
 src/backend/pthreads_pool.cpp            |  14 +-
 src/cuda/cuda_common.cuh                 |  27 ++-
 src/cuda/jacobi_sweep.cu                 |  96 ++++++----
 src/cuda/rb_gauss_seidel.cu              |  16 +-
 src/cuda/stream_probe.cu                 |  11 +-
 src/main.cpp                             | 306 +++++++++++++++++++------------
 tests/convergence/test_convergence.cpp   |  45 ++---
 tests/cuda/test_cuda.cpp                 |  99 ++++++----
 tests/equivalence/test_equivalence.cpp   | 101 +++++-----
 tests/mpi/test_mpi.cpp                   |  56 +++---
 tests/pnl_test.hpp                       | 107 ++++++-----
 tests/test_main.cpp                      |   7 +-
 tests/unit/test_numerics.cpp             |  55 ++++--
 tests/unit/test_solvers.cpp              |  34 ++--
 46 files changed, 999 insertions(+), 661 deletions(-)
```

**How the diff was shown to be formatting only, and why the obvious check is
not the one that settles it.** The task's suggested check is that
`git diff --ignore-all-space --ignore-blank-lines --stat` comes out much
smaller than the plain stat. It does not: it reports 42 files, 818 insertions
and 481 deletions against the plain 46, 999 and 661. That is not evidence of a
semantic change. `BinPackArguments: false` and `BinPackParameters: false` put
each argument and each parameter on its own line, and `ColumnLimit: 100`
rewraps long expressions, so most of this diff is line boundaries moving.
Git's whitespace flags ignore whitespace *within* a line; they cannot see that
one line became four holding the same tokens. On a formatter that rewraps, the
check has no power.

What was run instead: for each of the 46 files, strip comments, lift the
`#include` lines into a set of their own, delete every remaining whitespace
character, and compare the result against `git show HEAD:` for the same file.
That is invariant under rewrapping, indentation, blank lines and include
reordering, so anything it flags is a real change to the token stream. It
flagged three files, all for the same harmless reason:
`include/pnl/backend/mpi.hpp`, `src/cuda/cuda_common.cuh` and
`tests/pnl_test.hpp` are the three files holding multi line macros, and in each
one a line continuation backslash moved to a different token boundary as the
macro body rewrapped. Deleting the newlines but keeping the backslashes is what
makes those show up; a backslash immediately before a newline is a line splice
and carries no meaning of its own. No include was added or removed in any file.
No other file differed by a single token.

**16 files had `#include` lines reordered**, which is `IncludeBlocks: Regroup`
plus the four `IncludeCategories` in `.clang-format` being enforced for the
first time: project headers first, then the standard library, then the
parallelism headers `mpi`, `omp`, `pthread` and `cuda`, then everything else.
In `include/pnl/backend/mpi.hpp`, for example, `<mpi.h>` moved from before
`<string>` and `<vector>` to after them. Nothing broke: `make build` compiled
all 24 targets with no warning under `-Wall -Wextra -Wpedantic -Werror`, so no
`// clang-format off` pair was needed anywhere and none was added. That is the
outcome worth stating plainly, because the demotion of `<mpi.h>` and
`<cuda_runtime.h>` below the standard library headers is exactly the change
that breaks a tree whose headers are not self contained, and this one's are.

**Gate.** Run inside WSL2 Ubuntu through `tasks/run.sh`.

```text
$ find include src tests \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format --dry-run --Werror {} +
(no output, exit 0)

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 117 file(s) scanned

$ make build
-- pnl: OpenMP 4.5 enabled, spec date 201511
-- pnl: MPI 3.1 enabled (/usr/bin/mpiexec)
-- pnl: dropping /usr/lib/gcc/x86_64-linux-gnu/14 from the CUDA implicit link directories
-- pnl: CUDA enabled, arch 120, host /usr/bin/g++-14
-- pnl: build type Release, C++ compiler GNU 15.2.0
(elided: the ninja lines for 24 of 24 targets, all rebuilt, no warning)
[24/24] Linking CXX executable tests/test_cuda

$ make test
 1/10 Test  #5: test_no_dashes ...................   Passed    2.92 sec
 2/10 Test #10: test_cuda ........................   Passed    4.55 sec
 3/10 Test  #4: test_equivalence .................   Passed    1.89 sec
 4/10 Test  #8: test_mpi_2rank ...................   Passed    0.26 sec
 5/10 Test  #3: test_convergence .................   Passed    0.77 sec
 6/10 Test  #7: test_mpi_1rank ...................   Passed    0.28 sec
 7/10 Test  #9: test_mpi_4rank ...................   Passed    0.33 sec
 8/10 Test  #2: test_solvers .....................   Passed    0.01 sec
 9/10 Test  #1: test_numerics ....................   Passed    0.00 sec
10/10 Test  #6: test_dash_checker_self ...........   Passed    0.32 sec

100% tests passed out of 10

Total Test time (real) =   5.71 sec
```

The `--dry-run --Werror` pass over all 46 files is the one that matters twice
over: it is the phase's gate, and it is the statement that clang-format 20.1.7
is a fixed point on this tree, so the next run of `make format` produces no
diff and Phase B2 can make it a CI failure rather than a suggestion.
`test_equivalence` passing is the second thing to read, for the same reason it
was under A0.6: it asserts bit identical iterates across every shared memory
backend, so it would catch a reordered include that changed which overload or
which floating point path a translation unit selected. It did not.

The `make build` above was made from a dirty tree, since the tree still held
this phase's own reformatting, so the stamp it wrote reads `a15ba3b36cc1.dirty`.
After the commit `git status --porcelain` prints nothing.

No engineering log entry. Nothing broke, so there is no fault to record: no
reordering had to be suppressed and no code semantics were touched. The one
observation worth carrying forward is the version pin above, which belongs to
Phase B2 and is written into the 1.1.0 toolchain table in this file rather than
into the log.

### Phase A1: remove the serial copy from the Jacobi path

`SweepFunction` now returns the view that holds the iterate rather than nothing,
`Jacobi::solve` sweeps into the work buffer and hands it back, and
`run_stationary` alternates the two buffers from that answer instead of calling
`std::swap_ranges` on the calling thread after every sweep. The host and the
device run the same algorithm for the first time: `src/cuda/jacobi_sweep.cu` has
always swapped two device pointers, and the host was paying a full state copy
the device never paid, inside the Section 8.3 comparison that puts the two side
by side. The finding is `MEAS-01`.

**Diff size.** 8 files, 93 insertions, 11 deletions, so 104 changed lines. Not a
four line change, and nothing downstream may call it one. `SweepFunction`'s
signature, all eleven sweep lambdas across five solver headers, the driver's
residual, gather and return paths, and the distributed contract on
`Problem::jacobi_sweep` all move together:

```text
include/pnl/problems/dense_generator.hpp |  8 +++++
include/pnl/problems/problem.hpp         | 10 ++++++
include/pnl/solvers/block_solvers.hpp    |  5 +++
include/pnl/solvers/gauss_seidel.hpp     | 10 ++++++
include/pnl/solvers/jacobi.hpp           |  9 +++--
include/pnl/solvers/richardson.hpp       |  2 ++
include/pnl/solvers/sor.hpp              |  4 +++
include/pnl/solvers/splitting.hpp        | 56 +++++++++++++++++++++++++++-----
8 files changed, 93 insertions(+), 11 deletions(-)
```

Three of those lines are the ones the phase turns on, and each is a different
way to get the same wrong answer. The in loop residual reads `current`, not
`result.solution`, or the convergence test reads the previous iterate on
alternate iterations and every iteration count in the study shifts by one.
`problem.synchronise` is applied to `current` before the copy, or the gather
collects each rank's stale rows and every rank returns an ungathered solution.
And the final copy is decided by comparing `current.data()` against
`result.solution.data()`, not by a parity counter, because the ten in place
solvers return the same view on every call and their parity never advances.

**The dense audit.** `DenseProblem::jacobi_sweep` writes only the rows a rank
owns, and a dense matrix vector product reads every row of the iterate, so the
flip leaves it reading non local rows that the previous flip left stale. It is
safe, and not for the reason the specification's note gives. Every dense sweep
opens with `backend.exchange_halo(x, 0, n_)`, which at a row stride of zero is
an `MPI_Allgatherv` with `MPI_IN_PLACE` that overwrites every non local entry
from its owner before anything reads it, and `apply` and `residual` do the same.
`swap_ranges` was never what kept them consistent: in 1.0.0 it left the swapped
buffer holding the previous iterate's non local rows and the next sweep's gather
corrected them, exactly as it does now. No code change was needed, so the finding
is recorded inside `MEAS-01` rather than as a second entry, and the invariant is
now written above `jacobi_sweep` in both `problem.hpp` and `dense_generator.hpp`
so that the gather is not deleted as redundant by someone who assumes a complete
vector arrives.

**Before and after.** `--solver jacobi --backend openmp --mode fixed
--iterations 300`, five repetitions at 1023 squared and three at 4095 squared,
on an otherwise idle machine, taken before any file was touched at
`35d8a6f686ec` and again after the commit at `dce9cf4b25f0`.

| size | workers | before median | after median | before spread | after spread | before over after |
| --- | --- | --- | --- | --- | --- | --- |
| 1023 | 1 | 0.755255 | 0.519628 | 0.045 | 0.095 | 1.4535 |
| 1023 | 20 | 0.237309 | 0.062997 | 0.183 | 0.178 | 3.7670 |
| 4095 | 1 | 15.136418 | 11.197806 | 0.046 | 0.025 | 1.3517 |
| 4095 | 20 | 9.112059 | 4.970442 | 0.004 | 0.010 | 1.8332 |

Medians in seconds, spread is `(max - min) / median` over that run's
repetitions.

| size | speedup at 20 over 1, before | after |
| --- | --- | --- |
| 1023 | 3.1826 | 8.2485 |
| 4095 | 1.6611 | 2.2529 |

The gate row asks for a Jacobi speedup at 20 workers above 1.7. Both readings of
that sentence clear it and both are recorded here rather than one being chosen:
the parallel speedup at 20 workers over 1 is 2.2529 at 4095 squared and 8.2485
at 1023 squared, and the wall clock improvement at 20 workers is 1.8332 at 4095
squared and 3.7670 at 1023 squared. The 1.8332 lands inside the 1.8 to 2.5 range
Section 10.2 pre registered for K0, at its lower end. No Amdahl ceiling is
stated, per Section 4.1. Note that the 1023 squared rows sit largely in the 33
MiB last level cache, where the streaming byte model does not apply, which is why
the two sizes behave so differently.

The recorded `relative_residual` is unchanged in all four rows, 3.207319e-02 at
1023 squared and 3.262973e-02 at 4095 squared, to every digit the row prints.

**Bit identity, beyond the required gate.** The equivalence suite passing
unmodified is the specification's proof and it passes, but it compares the new
code against itself. So I also built `35d8a6f` into a scratch build tree outside
the repository and ran both binaries over the same 122 configurations: twelve
solvers on the Poisson and dense problems, on serial and OpenMP, to a tolerance
of 1e-8 and at a fixed 137 iterations, plus all twelve at four MPI ranks.
Comparing solver, backend, workers, iterations, converged, stop reason and
relative residual, `diff` of the two sets is empty. Jacobi's 11255 iterations to
1e-8 at 63 squared is identical on serial, OpenMP and four ranks, and a shift of
one iteration or one bit would have moved it. The same solvers declined the same
configurations in both.

**Gate.** Run inside WSL2 Ubuntu through `tasks/run.sh`.

```text
$ git diff --exit-code 35d8a6f -- tests/equivalence/
(no output, exit 0)

$ make build
-- pnl: OpenMP 4.5 enabled, spec date 201511
-- pnl: MPI 3.1 enabled (/usr/bin/mpiexec)
-- pnl: dropping /usr/lib/gcc/x86_64-linux-gnu/14 from the CUDA implicit link directories
-- pnl: CUDA enabled, arch 120, host /usr/bin/g++-14
-- pnl: build type Release, C++ compiler GNU 15.2.0
(12 of 12 targets rebuilt, no warning under -Wall -Wextra -Wpedantic -Werror)

$ make test
 1/10 Test  #5: test_no_dashes ...................   Passed    2.28 sec
 2/10 Test #10: test_cuda ........................   Passed    4.13 sec
 3/10 Test  #4: test_equivalence .................   Passed    1.85 sec
 4/10 Test  #9: test_mpi_4rank ...................   Passed    0.29 sec
 5/10 Test  #6: test_dash_checker_self ...........   Passed    0.30 sec
 6/10 Test  #3: test_convergence .................   Passed    0.71 sec
 7/10 Test  #7: test_mpi_1rank ...................   Passed    0.25 sec
 8/10 Test  #2: test_solvers .....................   Passed    0.01 sec
 9/10 Test  #1: test_numerics ....................   Passed    0.00 sec
10/10 Test  #8: test_mpi_2rank ...................   Passed    0.27 sec

100% tests passed out of 10

Total Test time (real) =   5.14 sec

$ ctest --test-dir build --output-on-failure -L equivalence
1/1 Test #4: test_equivalence .................   Passed    1.57 sec

100% tests passed out of 1

$ find include src tests \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format --dry-run --Werror {} +
(no output, exit 0, clang-format 20.1.7)

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 117 file(s) scanned

$ git status --porcelain
(no output)
```

`test_mpi_4rank` is the one to read after `test_equivalence`. It compares the
distributed solution against the serial one for every solver on both problems,
so it is what decides whether the gather was applied to the right buffer and
whether the dense audit's conclusion holds. The after measurements were taken
from the committed tree, so their rows are stamped `dce9cf4b25f0` with no
`.dirty` suffix.

Findings: `MEAS-01`, a new family for the measurement validity findings of
Section 4. The dense audit is inside it, because it found no defect.

### Phase A1.5: make the schema migratable, then add every new column at once

The result header gained eight columns in one commit, and the summary gained a
way to receive them. Before this, `run_sweep.py` compared the stored header
against the binary's with strict list equality and exited 2 on any difference,
with no backfill anywhere in the file, so each column V2 adds would have broken
`make sweep` against the committed summary, and therefore `make all`, until a
full re measurement finished. Six hard stops became one, and this is the one.

**The eight columns**, appended after `label`, printed in both `printf` calls so
that the device row is not a column short of its own header:

| column | host row | device row | filled in by |
| --- | --- | --- | --- |
| `sweeps` | empty | empty | A2 |
| `passes` | empty | empty | A2 |
| `dram_bytes_per_unknown_per_sweep` | empty | empty | A3a |
| `pinning_status` | empty | empty | A4 |
| `measured_at` | `2026-09-05T21:01:30Z` | the same | now |
| `seconds_reps` | `0.000056;0.000031;0.000032;0.000031` | the same | now |
| `kernels` | `cxx` | `device` | part C, 1.2.0 |
| `kernel_variant` | `cpp` | `device` | part D, 1.2.0 |

Empty means an empty field between two commas, which pandas reads as NaN.
Nothing between this commit and the phase that fills a column can be mistaken
for a measurement, which is the whole reason the six are empty rather than zero
or `n/a`.

`seconds_reps` exists because Phase A7 bootstraps the knee fit, and a bootstrap
needs the repetitions rather than their median, minimum and maximum: three order
statistics cannot be resampled. The binary already held every repetition when it
computed those three, and now prints them, in run order, before the sort that
destroys the order. The semicolon list is safe inside one CSV field because no
other field uses a semicolon, and the header comment says so.

`measured_at` is stamped by the binary rather than by the harness, for the same
reason the commit hash and the seed are.

**The migration.** `scripts/migrate_summary.py` takes a summary and the binary's
current header, appends the missing columns with the defaults declared in one
table at the top of the file, and refuses with exit 2, naming the column, if the
stored file has a column the binary no longer emits. It preserves every existing
value byte for byte by appending text to each line rather than parsing and
rewriting it, which also keeps the CRLF endings `csv.DictWriter` gave the file.
It writes through the same atomic temporary file and rename `run_sweep.py` uses,
four lines copied rather than imported, because that helper writes parsed rows
and this one appends to raw lines. Running it twice is a no operation and says
so.

`run_sweep.py --migrate` calls it instead of exiting 2, and `make sweep` and
`make sweep-force` pass the flag. `--dry-run` now does the header check and the
resume calculation, reports what would run, and exits without probing bandwidth
or running a configuration; the only thing it executes is the four point grid
that reads the commit stamp out of the binary.

**The identity repairs in the same commit.** `IDENTITY_FIELDS` gained `kernels`
and `kernel_variant`, because adding the column without adding the field is
`SWEEP-03` a second time and that one cost real data. `Run` and `expand_block`
gained both axes, each defaulting to the single value this release has, so every
existing block expands to exactly the configurations it did before.
`predicted_identity` normalises both to `device` on the device path. And the two
neighbouring defects that the rewrite of those twenty lines exposed were
repaired: `SWEEP-05`, the predicted `backend` of a CUDA row, and the missing
`fortran_dc_serial` case in `predicted_workers`.

**The dry run, before and after `SWEEP-05`**, against the migrated committed
summary, with nothing changed but that one line:

```text
before   42 have no stored row at any commit: cuda 12, hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
         54 stored rows that no declared configuration predicts: cuda 24, hybrid 30
after    30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
         30 stored rows that no declared configuration predicts: hybrid 30
```

Twelve CUDA configurations and the 24 stored CUDA rows they account for, at two
commits, now resume. They are reported as present at `4abf914a7ea2.dirty` rather
than at the current build, because `commit` is in the identity and those rows
predate this binary, which is the commit column working as intended. What
remains unmatched is fifteen hybrid configurations, which is finding 4.6 and
Phase A6's to repair, and fifteen `cg` on `dense_dd` configurations, which is the
declared inapplicable skip.

**The committed summary was migrated** and `scripts/gen_report_assets.py` still
runs against it. The eight LaTeX tables and twelve PNG figures it writes are
byte identical to the ones generated from the unmigrated file. The six PDF
figures differ between any two runs, including two runs from the same input,
because matplotlib stamps a `/CreationDate` into each; with that string removed
they are equal byte for byte and `pdftotext` output is identical. No regenerated
asset is committed here.

**Gate.** Run before the commit, so the stamp is the parent commit with a dirty
tree.

```text
$ git show ec406a7:experiments/results/summary.csv > /tmp/summary-1.0.0.csv
$ python3 scripts/migrate_summary.py /tmp/summary-1.0.0.csv --header-from build/pnl
migrate_summary: /tmp/summary-1.0.0.csv: added sweeps, passes, dram_bytes_per_unknown_per_sweep, pinning_status, measured_at, seconds_reps, kernels, kernel_variant to 850 row(s), 24 of them on the device

$ head -1 /tmp/summary-1.0.0.csv
problem,...,seed,commit,label,sweeps,passes,dram_bytes_per_unknown_per_sweep,pinning_status,measured_at,seconds_reps,kernels,kernel_variant

$ python3 scripts/migrate_summary.py /tmp/summary-1.0.0.csv --header-from build/pnl
migrate_summary: /tmp/summary-1.0.0.csv already has every column the binary emits, 36 of them; nothing to do

$ python3 benchmarks/run_sweep.py --build build --dry-run
440 configurations declared, 850 rows in the summary
0 already present at commit 08a5fa8aa8d4.dirty, which is what a sweep would skip
410 present at 4abf914a7ea2.dirty, cd57032941a8.dirty and at no other commit, which a sweep from this build would measure again
30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
30 stored rows that no declared configuration predicts: hybrid 30
(exit 0)

$ ./benchmarks/run_sweep.sh --build build --dry-run          # 1.0.0 summary in place
run_sweep: the existing summary has a different set of columns than the binary now emits. Pass --migrate to add the missing ones with their declared defaults, or move the file aside rather than mixing schemas.
(exit 2)

$ ./benchmarks/run_sweep.sh --build build --dry-run --migrate
migrate_summary: .../experiments/results/summary.csv: added sweeps, passes, dram_bytes_per_unknown_per_sweep, pinning_status, measured_at, seconds_reps, kernels, kernel_variant to 850 row(s), 24 of them on the device
(exit 0, and the file it produced is byte identical to the migrated one committed here)

$ make build && make test
100% tests passed out of 11
      Start  7: test_migrate_summary
11/11 Test  #7: test_migrate_summary .............   Passed    0.29 sec

$ ruff check benchmarks scripts tests
All checks passed!

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 133 file(s) scanned

$ find include src tests \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format --dry-run --Werror {} +
(no output, exit 0, clang-format 20.1.7)

$ git status --porcelain
(no output)
```

No sweep was run. `--dry-run` exists so that this gate does not need one.

Findings: `SWEEP-05`, the CUDA resume defect, and `SWEEP-06`, the strict header
check with no repair.

### Phase A2: make the work unit honest

Done, in two commits, one for the code and one for these records. Four
corrections to what a result row says about the work it timed, and not one of
them changes an iterate: every solver's iteration count and residual on the
unit and convergence suites is what it was, and the equivalence suite passes
unmodified.

**Two fields, not one.** `Diagnostics` gains `sweeps`, updates per unknown per
iteration, and `passes`, streams over the state array per iteration. They are
separate because they disagree for exactly the red black methods:
`coloured_sweep` steps `j += 2`, so each colour writes half the unknowns and the
pair writes each unknown once, in two strided traversals of the whole array. One
sweep of work, two passes over memory. A single column would have been ambiguous
for those two, which are the solvers that carry the Section 8.3 device
comparison. `Solver::work_unit()` is pure virtual, so a solver added later
cannot omit them, and the CUDA path reads the same declaration through the host
solver of the same name rather than keeping a second table.

**The methods that were undercounted are the symmetric ones, not the red black
ones.** `updates` was `unknowns * iterations` and is now
`unknowns * iterations * sweeps`, which leaves `gauss_seidel_rb` and `sor_rb`
exactly where they were and doubles `gauss_seidel_s` and `ssor`, together with
the `updates_per_second` and `gib_per_second` derived from them. The claim in
`gauss_seidel.hpp` that the rows already recorded sweeps so the report could
compare on equal work is now true, and the comment says which columns carry it
and that the mitigation did not exist before this phase.

**`evaluations` is the operator application count.** The initial residual, one
per sweep, and one for each residual the check interval asked for. Conjugate
gradient reports `iterations + 1` where it reported `iterations`; its two inner
products per iteration are not operator applications and are not counted. There
is no `evaluations` column in the CSV and this phase did not add one, since
Section 12 forbids a column outside the A1.5 schema commit. The counts are held
by a unit case instead, printed below.

**Richardson evaluates its residual once.** It computed `b - A x_k` inside its
sweep and the driver computed `b - A x_{k+1}` again whenever the check fired,
which at the driver's default interval of one is every iteration. Richardson is
specialised rather than given a hint: the only residual the sweep has to offer
is the one at `x_k`, and a driver that accepted it would report Richardson one
iteration behind every other method. It now updates from the residual it holds,
evaluates the next one once, tests that when the check is due, and keeps the
vector. The check itself stays shared, factored into `detail::check_due` and
`detail::apply_check` which both paths call, so the criterion, the quantity and
the iterate it is measured at are still the driver's.

**`omega` only where a factor was used.** `main.cpp` asked
`Sor::resolve_relaxation` for every row. Eight of the twelve methods take no
relaxation factor, so eight twelfths of the rows carried one the run never read,
and two of the four that do take one, Richardson and SSOR, carried the SOR
optimum rather than the step they take. `Solver::relaxation_factor` answers both
questions at once and each solver's `solve` calls it, so the row and the run
cannot disagree.

**The twelve, at 63 squared, ten fixed iterations, residual checked every
iteration, serial backend.** `sweeps`, `passes` and `omega` are read from the
result rows, `evaluations` from the unit case that holds the table.

| solver | sweeps | passes | evaluations | omega |
| --- | --- | --- | --- | --- |
| `richardson` | 1 | 2 | 11 | 0.125000 |
| `jacobi` | 1 | 1 | 21 | empty |
| `gauss_seidel_f` | 1 | 1 | 21 | empty |
| `gauss_seidel_b` | 1 | 1 | 21 | empty |
| `gauss_seidel_s` | 2 | 2 | 31 | empty |
| `gauss_seidel_rb` | 1 | 2 | 21 | empty |
| `sor` | 1 | 1 | 21 | 1.906455 |
| `ssor` | 2 | 2 | 31 | 1.000000 |
| `sor_rb` | 1 | 2 | 21 | 1.906455 |
| `block_jacobi` | 1 | 2 | 21 | empty |
| `block_gauss_seidel` | 1 | 1 | 21 | empty |
| `cg` | 1 | 6 | 11 | empty |

Block Jacobi takes two passes because the lagged coupling makes `block_sweep`
snapshot the previous iterate, and block Gauss Seidel takes one because reading
the blocks already updated removes the snapshot. Conjugate gradient takes six:
the matrix vector product, the two inner products and the three axpy like
updates of x, r and p. Richardson takes two, a residual and an axpy, where
before this phase it took three.

**Richardson before and after.** Serial backend, 200 fixed iterations, median of
five repetitions, at the driver's default `check_interval` of 1 and at the
1000000 the sweep matrix uses.

| size | check interval | before, s | after, s | change |
| --- | --- | --- | --- | --- |
| 511 | 1 | 0.097504 | 0.056097 | 1.738x |
| 511 | 1000000 | 0.057531 | 0.061224 | inside the spread |
| 1023 | 1 | 0.639220 | 0.276260 | 2.314x |
| 1023 | 1000000 | 0.295283 | 0.274997 | 1.074x, ranges overlap |

The relative residual after those 200 iterations is 3.955925e-02 at 511 and
3.940177e-02 at 1023, before and after, at both check intervals.

The two rows at check interval 1000000 are the control, since the change should
not touch them: at that interval the old code already evaluated the residual
once per iteration, inside the sweep. Neither is separable from noise. At 511
the two runs span 0.055092 to 0.067782 and 0.054553 to 0.062296, and at 1023
they span 0.287792 to 0.312763 and 0.267415 to 0.288659, overlapping in both
cases. The small apparent gain at 1023 is consistent with the new code holding
three state vectors where the old held five, but it is inside the spread and is
recorded as an observation, not as a result.

The byte model predicts 1.5x at the default check interval, residual plus axpy
plus residual against residual plus axpy, at 24 bytes per unknown each. The
measurement is larger and for two reasons worth writing down. `problem.residual`
also takes the norm of what it wrote, which is a second traversal, so the pass
counts are five against three and the model's own prediction on passes is 1.67.
That is what 511 squared shows, 0.097504 against 0.057531, a ratio of 1.695.
At 1023 squared the old arrangement touched four distinct 8.4 MB vectors per
iteration, the iterate, the right hand side, the sweep's residual and the
driver's, which is 33.6 MB against a 33 MiB last level cache, while the new one
touches three, 25.2 MB, and fits. The ratio there is 2.165. The 1.5x of the
specification is a claim about counted bytes and it is right about those; the
time is worse than the bytes because of the cache.

The iterates do not move, and the run to tolerance is the sensitive check
because a single changed bit moves the count:

```text
$ build/pnl --solver richardson --backend serial --size N --mode solve \
      --tolerance 1e-8 --reps 1 --check-interval C

  N   C   iterations   relative residual      before and after
 31   1         6052        9.992941e-09      identical
 31   5         6055        9.920937e-09      identical
 63   1        20602        9.996886e-09      identical
 63   5        20605        9.978835e-09      identical
```

Check interval five is the interesting row: the old code did not evaluate the
residual on four iterations out of five and the new one does, and the count is
still the same, because the extra evaluation is only ever read when the check is
due.

**Gate.** Run inside WSL2 Ubuntu through `tasks/run.sh`, from the tree at the
code commit. `docs/ENGINEERING_LOG.md` and this file are the only things
outstanding, and neither is inside the pathspec the dirty check uses, so the
binary stamps a clean hash.

```text
$ for s in jacobi gauss_seidel_f gauss_seidel_b gauss_seidel_s gauss_seidel_rb \
      sor sor_rb ssor richardson block_jacobi block_gauss_seidel cg; do
      build/pnl --solver $s --backend serial --size 63 --mode fixed \
          --iterations 10 --reps 1
  done
poisson2d_rich_63,3969,jacobi,serial,...,1.777864e-01,,63,1,...,788212de2db5,,1,1,,,...
poisson2d_rich_63,3969,gauss_seidel_f,serial,...,8.865257e-02,,63,1,...,788212de2db5,,1,1,,,...
poisson2d_rich_63,3969,gauss_seidel_b,serial,...,8.867446e-02,,63,1,...,788212de2db5,,1,1,,,...
poisson2d_rich_63,3969,gauss_seidel_s,serial,...,5.645167e-02,,63,1,...,788212de2db5,,2,2,,,...
poisson2d_rich_63,3969,gauss_seidel_rb,serial,...,1.221406e-01,,63,1,...,788212de2db5,,1,2,,,...
poisson2d_rich_63,3969,sor,serial,...,4.783475e-01,1.906455,63,1,...,788212de2db5,,1,1,,,...
poisson2d_rich_63,3969,sor_rb,serial,...,6.241920e-01,1.906455,63,1,...,788212de2db5,,1,2,,,...
poisson2d_rich_63,3969,ssor,serial,...,5.645167e-02,1.000000,63,1,...,788212de2db5,,2,2,,,...
poisson2d_rich_63,3969,richardson,serial,...,1.809102e-01,0.125000,63,1,...,788212de2db5,,1,2,,,...
poisson2d_rich_63,3969,block_jacobi,serial,...,1.165387e-01,,63,1,...,788212de2db5,,1,2,,,...
poisson2d_rich_63,3969,block_gauss_seidel,serial,...,5.867176e-02,,63,1,...,788212de2db5,,1,1,,,...
poisson2d_rich_63,3969,cg,serial,...,1.473795e-01,,63,1,...,788212de2db5,,1,6,,,...
```

The rows are elided in the middle only, at the timing fields, which are not what
this gate reads. The fields shown are `relative_residual`, `omega`, `blocks`,
`check_interval`, then the commit, the empty `label`, and then `sweeps` and
`passes` followed by the two fields A3a and A4 still leave empty. `sweeps` is 1
for `gauss_seidel_rb` and `sor_rb` and 2 for `gauss_seidel_s` and `ssor`,
`passes` is 2 for all four, and `omega` is empty on eight rows.

```text
$ build/tests/test_solvers
  pass  solvers/every solver in the registry solves a 4x4 system
  pass  solvers/every solver in the registry solves the Poisson problem
  pass  solvers/conjugate gradient refuses a system that is not symmetric
  pass  solvers/red black methods refuse a dense system
  pass  solvers/SOR rejects a relaxation factor outside the Kahan interval
  pass  solvers/conjugate gradient terminates within n steps in exact arithmetic
  pass  solvers/a non converged result reports itself as such
  pass  solvers/fixed iteration mode runs exactly the requested count
solver                sweeps  passes   evaluations      omega
richardson                 1       2            11      0.125
jacobi                     1       1            21
gauss_seidel_f             1       1            21
gauss_seidel_b             1       1            21
gauss_seidel_s             2       2            31
gauss_seidel_rb            1       2            21
sor                        1       1            21 1.906454701582762
ssor                       2       2            31          1
sor_rb                     1       2            21 1.906454701582762
block_jacobi               1       2            21
block_gauss_seidel         1       1            21
cg                         1       6            11
  pass  solvers/every solver reports its work unit and its operator applications
  pass  solvers/Richardson applies the operator once per iteration
  pass  solvers/the Poisson solution matches the manufactured solution to O(h^2)
11 passed, 0 failed
(exit 0)

$ make build && make test          # tail of the ctest output
100% tests passed out of 11

Label Time Summary:
convergence    =   0.70 sec*proc (1 test)
cuda           =   4.51 sec*proc (1 test)
equivalence    =   1.80 sec*proc (1 test)
mpi            =   0.82 sec*proc (3 tests)
style          =   3.21 sec*proc (2 tests)
unit           =   0.27 sec*proc (3 tests)

$ git diff --exit-code 35d8a6f -- tests/equivalence/
(no output, exit 0)

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 133 file(s) scanned

$ clang-format --version
clang-format version 20.1.7
$ find include src tests \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \
      -o -name '*.cuh' \) -exec clang-format --dry-run --Werror {} +
(no output, exit 0)

$ ruff check benchmarks scripts tests
All checks passed!

$ git status --porcelain
 M docs/ENGINEERING_LOG.md
```

`test_cuda` is in the eleven and passes, so the device path's new `sweeps`,
`passes` and `omega` fields compile and run against a real device.

Findings: `MEAS-02`, Richardson's double residual, with the byte arithmetic and
the reason the measurement exceeds it; `MEAS-03`, the symmetric methods charged
one sweep while the red black methods were already correct, and the mitigation
`gauss_seidel.hpp` claimed did not exist; `MEAS-04`, a relaxation factor
recorded on rows that never used one, and the wrong one on two rows that did;
`MEAS-05`, found while writing the residual property test, that conjugate
gradient reports the recurrence residual rather than `b - A x`, which is the
standard formulation and is recorded rather than changed.

No sweep was run. The result rows above are single configurations at 63 squared
and 200 iteration timing runs at 511 and 1023 squared, taken to produce the
numbers in this section, and none of them is written to
`experiments/results/`.

### Phase A3a: publish both traffic models now, settle the question later

Done, in two commits, one for the models and the column and one for the pre
registration and the probe. Nothing here is on a numerical path: no iterate
moves, no residual moves, and `tests/equivalence/` is untouched.

**Both counts, derived from the code as it now is.** `Problem` gains
`dram_bytes_per_unknown_per_sweep()` beside `bytes_per_unknown_per_sweep()`.
Both are documented as one **pass** over the arrays, which is what phase A2's
`passes` column multiplies, and both carry their derivation array by array.

| problem | array | read | written | read before it is written | conservative | with read for ownership |
| --- | --- | --- | --- | --- | --- | --- |
| `Poisson2D` | `rhs_` | yes | no | not applicable | 8 | 8 |
| `Poisson2D` | `x` | yes | no | not applicable | 8 | 8 |
| `Poisson2D` | `out` | no | yes | **no** | 8 | 16 |
| `Poisson2D` | **total per unknown per pass** | | | | **24** | **32** |
| `DenseProblem` | `matrix_`, one row | yes | no | not applicable | 8 n | 8 n |
| `DenseProblem` | `rhs_` | yes | no | not applicable | 8 | 8 |
| `DenseProblem` | `x` | yes | no | not applicable | 8 | 8 |
| `DenseProblem` | `out` | no | yes | **no** | 8 | 16 |
| `DenseProblem` | **total per unknown per pass** | | | | **8(n + 3)** | **8(n + 4)** |

The `x` row of the stencil is charged once rather than five times because the
four stencil neighbours of consecutive unknowns are the same lines the sweep is
already walking. The `x` row of the dense problem is charged once rather than n
times because a row reads the whole iterate but the iterate is the same n
doubles for every row and is reused out of cache. A Jacobi pass has exactly one
array written without being read first, so the correction is one extra read of
the output line and no more.

**These are not the 56 and 80 of Section 4.2.** Those totals are per iteration
and include the full state copy the driver made with `swap_ranges`, which phase
A1 removed: 24 plus 32 conservatively, and 32 plus 48 with read for ownership.
With the copy gone a Jacobi iteration is one pass and the candidates are 24 and
32. On the dense problem the correction is one double against a term of order n,
0.2 percent at n equal to 512, so the dense rows cannot separate the two models
and the stencil is where the question has teeth.

The in place sweeps, `relaxation_sweep` and `coloured_sweep`, read each entry
before they overwrite it, so their written array pays no read for ownership at
all and the conservative count is exact for them. The column is the problem's
Jacobi model, which is what `bytes_per_unknown` has always been, and the
comments say so.

**The column, and both bandwidths in the report.** `src/main.cpp` fills
`dram_bytes_per_unknown_per_sweep` on the host row and on the device row. The
device path declares no second count of its own, so the device row carries the
host figure, and the comment says why: the device kernels move the same three
arrays per unknown, and what the column adds on top is a property of a write
allocate cache that the pre registration settles on the host triad and claims
nothing about on the device.

`gib_per_second` keeps its definition, which is ground rule 9: nothing changes
underneath a reader. `scripts/gen_report_assets.py` derives the second
bandwidth from `dram_bytes_per_unknown_per_sweep * passes * unknowns *
iterations / seconds_median` and emits both, labelled **declared** and
**counted**, in the device comparison table and in the device efficiency figure.
The two differ in two ways at once and both are deliberate: the byte count, and
the work unit, since `gib_per_second` uses `sweeps` and the counted figure uses
`passes`, which disagree for exactly the red black pair. Where either column is
empty the generator writes `predates the column` and derives nothing. Assuming
one pass for such a row would halve the figure for exactly the two methods the
device comparison turns on.

Every row of the committed summary predates both columns, so every counted cell
in the regenerated table reads `predates the column` today and the efficiency
figure carries the declared model alone and says so in its title. That is the
correct behaviour and not a gap: those rows were measured before phase A2.

**The third row of the header, at 63 squared, ten fixed iterations, serial.**

```text
$ build/pnl --solver jacobi --backend serial --size 63 --mode fixed --iterations 10 --reps 1
poisson2d_rich_63,3969,jacobi,serial,1,1,1,none,deterministic,static,fixed,10,0,
iteration_cap,1.777864e-01,,63,1,0.000059,0.000059,0.000059,1,6.736367e+08,
15.0570,24.0,20260802,b7a372f6d195,,1,1,32.0,,2026-09-05T22:06:17Z,
0.000059,cxx,cpp
```

Wrapped for width; the row is one line. The commit stamp carries no `.dirty`
suffix: this row and the four below it were taken from the clean tree at the
second of the two commits. Reading the two byte columns by name, and the same
three configurations on the red black and dense paths:

| configuration | `sweeps` | `passes` | `bytes_per_unknown` | `dram_bytes_per_unknown_per_sweep` |
| --- | --- | --- | --- | --- |
| `jacobi`, poisson 63 | 1 | 1 | 24.0 | 32.0 |
| `gauss_seidel_rb`, poisson 63 | 1 | 2 | 24.0 | 32.0 |
| `sor_rb`, poisson 63 | 1 | 2 | 24.0 | 32.0 |
| `jacobi`, `dense_spd` 512 | 1 | 1 | 4120.0 | 4128.0 |

**The probe.** `measure_host_triad_nontemporal` is a second host probe over the
same arrays, sizes, worker counts and repetitions as the plain one, differing in
the store instruction and in nothing else: `_mm256_stream_pd` with one
`_mm_sfence` after the loop, guarded on `__AVX__`, 32 byte aligned buffers, and
a scalar fallback that reports itself as the fallback rather than passing an
ordinary store off as a streaming one. `objdump -d build/pnl` finds one
`vmovntpd` and the fence. It is reported as an additional entry beside the
existing probe and never as a replacement, with the ratio in both orientations
and with the read for ownership corrected figure for the plain triad,
`plain * 32 / 24`, filed under `bandwidth.derived` in the manifest and labelled
as arithmetic rather than as a measurement. `docs/comparison_methodology.md`
states that the corrected figure is not corroborated against a theoretical peak
and cannot be: no memory speed is recorded anywhere in this repository and
`dmidecode` is not installed in the guest.

**The probe output, on an idle machine. An observation, not the publication
measurement.** Phase A8b takes that one and applies the rule to it. Three runs,
quoted in full; the third is the gate command on the committed tree:

```text
$ build/pnl --bandwidth --backend openmp        # first run
device,gib_per_second,detail
host,61.205,plain stores over the execution backend, best of workers
  2:39.3 4:50.1 8:61.2 12:57.2 16:60.1 20:59.9 24:59.4 28:58.0  over 256 MiB
  arrays; declares 24 bytes per element
host_nontemporal,67.267,_mm256_stream_pd with one sfence, best of workers
  2:47.2 4:52.9 8:63.9 12:59.8 16:67.3 20:66.4 24:65.5 28:59.4  over 256 MiB
  arrays; moves 24 bytes per element for real
derived_ratio_nontemporal_over_plain,1.0990,derived, not measured
derived_ratio_plain_over_nontemporal,0.9099,derived, not measured
derived_host_plain_at_32_bytes,81.606,derived, not measured
gpu,549.641,NVIDIA GeForce RTX 5070 sm_120 48 SMs 11.9 GiB, 512 MiB arrays,
  best of 5

$ build/pnl --bandwidth --backend openmp        # second run, same machine
host,69.083,plain stores over the execution backend, best of workers
  2:42.9 4:60.8 8:69.1 12:63.5 16:60.6 20:59.5 24:59.8 28:54.4  over 256 MiB
  arrays; declares 24 bytes per element
host_nontemporal,69.921,_mm256_stream_pd with one sfence, best of workers
  2:53.3 4:63.5 8:69.9 12:64.0 16:66.2 20:64.6 24:66.0 28:65.2  over 256 MiB
  arrays; moves 24 bytes per element for real
derived_ratio_nontemporal_over_plain,1.0121,derived, not measured
derived_ratio_plain_over_nontemporal,0.9880,derived, not measured
derived_host_plain_at_32_bytes,92.111,derived, not measured
gpu,548.515,NVIDIA GeForce RTX 5070 sm_120 48 SMs 11.9 GiB, 512 MiB arrays,
  best of 5
```

```text
$ build/pnl --bandwidth --backend openmp        # third run, committed tree
host,61.670,plain stores over the execution backend, best of workers
  2:41.7 4:60.6 8:61.7 12:59.5 16:57.8 20:60.0 24:56.1 28:47.1  over 256 MiB
  arrays; declares 24 bytes per element
host_nontemporal,65.375,_mm256_stream_pd with one sfence, best of workers
  2:45.2 4:57.0 8:64.4 12:58.8 16:62.1 20:60.7 24:65.4 28:48.0  over 256 MiB
  arrays; moves 24 bytes per element for real
derived_ratio_nontemporal_over_plain,1.0601,derived, not measured
derived_ratio_plain_over_nontemporal,0.9433,derived, not measured
derived_host_plain_at_32_bytes,82.227,derived, not measured
gpu,549.074,NVIDIA GeForce RTX 5070 sm_120 48 SMs 11.9 GiB, 512 MiB arrays,
  best of 5
```

The non temporal arm is ahead at twenty two of the twenty four worker points
across the three runs and behind at two, both in the third run, at four workers
and at twelve. The direction is the one read for ownership predicts. The size of
the effect is another matter: the selection statistic read 1.0990, 1.0121 and
1.0601, a spread of 0.087 against a pre registered undecided band 0.10 wide, and
the plain probe's own figure at eight workers spans thirteen percent across the
three runs. That is `MEAS-07`, and **the rule was not touched after seeing it.**

**The pre registration, in full, as written into `benchmarks/sweep_matrix.yaml`
before either run above.** It sits under a new top level `preregistered:` key
that the sweep driver ignores by name, through `NON_BLOCK_KEYS` in
`run_sweep.py`, rather than by expanding to zero configurations by accident.

- **The question.** Every achieved bandwidth in this repository is a byte count
  times a work unit divided by a time. The byte count is the open question. Does
  a Jacobi pass over the five point stencil move three doubles per unknown, or
  four, the fourth being the output line that an ordinary store has to fetch
  from memory before it can overwrite it?
- **The candidates.** Conservative, 24 bytes per unknown per pass, derived
  above. With read for ownership, 32 bytes per unknown per pass, derived above.
  The dense problem's pair is `8(n + 3)` and `8(n + 4)` and cannot separate
  them.
- **The instrument.** Two host STREAM triads over the same arrays, sizes, worker
  counts and repetitions, differing in the store instruction and in nothing
  else. Both report against the same declared 24 bytes per element, so the ratio
  of the two is the ratio of the traffic they really move: one if the plain loop
  pays nothing extra, four thirds if it pays a read for ownership on every line.
- **The selection statistic.** `ratio = host_nontemporal / host`, both from
  `pnl --bandwidth` on a quiet machine, each the best over the worker sweep. The
  reciprocal is recorded beside it in the manifest so the rule cannot be read in
  the wrong direction; in that orientation the thresholds are 0.833 and 0.909
  and the inequalities reverse.
- **The selection rule.** A ratio above 1.20 selects the read for ownership
  model. A ratio below 1.10 selects the conservative model. A ratio from 1.10 to
  1.20 inclusive is recorded as unresolved and both models are carried. The
  thresholds are fixed before any measurement and are not revisited afterwards.
- **Applied by.** Phase A8b, for release 1.1.0, which records the outcome as
  `ASM-01`. Phase D4 of release 1.2.0 rebuilds the same triad in assembly and
  asserts it agrees with the intrinsics arm to within run to run spread, which
  confirms the result rather than gating it.
- **The sentence the report carries if the read for ownership model is
  selected.** The non temporal triad reached a ratio of `<ratio>` against the
  plain triad on the publication machine, above the 1.20 threshold fixed before
  the measurement, so the read for ownership model is selected: a Jacobi pass
  over the five point stencil moves 32 bytes per unknown and not 24, the counted
  column is the achieved bandwidth this report compares against each device's
  own triad, and the host figures rise by a third while the device figures do
  not move, which is why host and device efficiency converge on the Jacobi row.
- **The sentence if the conservative model is selected.** The non temporal triad
  reached a ratio of `<ratio>` against the plain triad on the publication
  machine, below the 1.10 threshold fixed before the measurement, so the
  conservative model is selected: a Jacobi pass moves 24 bytes per unknown, the
  declared column stands as the achieved bandwidth of this report, and Section
  4.2's read for ownership argument is refuted on this machine rather than
  confirmed. The Jacobi efficiency gap between host and device is then a real
  gap and not an artefact of the denominator, and the only correction this
  release makes to the work unit is the pass count of phase A2.
- **The sentence if it is unresolved.** The non temporal triad reached a ratio
  of `<ratio>` against the plain triad on the publication machine, between the
  1.10 and 1.20 thresholds fixed before the measurement, so the traffic model is
  recorded as unresolved: both counts, 24 and 32 bytes per unknown per pass, are
  carried in every table and figure of this report, neither is presented on its
  own as the achieved bandwidth, and no claim is made that rests on one of them
  and would fail under the other.

`measured_ratio` and `outcome` both read `pending`, which is ground rule 3. The
same three sentences and the same rule are in `docs/comparison_methodology.md`
under a new heading, which also states that the model is unsettled between the
two candidates until the publication session measures it.

**The gate.**

```text
$ make build && make test
100% tests passed out of 11

$ find include src tests \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \
      -o -name '*.cuh' \) -exec clang-format --dry-run --Werror {} +
exit 0

$ build/pnl --solver jacobi --backend serial --size 63 --mode fixed \
      --iterations 10 --reps 1
  both byte columns present and non empty; the row is quoted above,
  bytes_per_unknown 24.0 and dram_bytes_per_unknown_per_sweep 32.0

$ build/pnl --bandwidth --backend openmp
  the plain triad, the non temporal triad, both ratios and the derived
  corrected figure; quoted in full above

$ python3 scripts/gen_report_assets.py 2>&1 | tail -5
  table   report/tables/reduction_cost.tex
  table   report/tables/pinning.tex
  table   report/tables/schedule_cost.tex
  table   report/tables/knee.tex
gen_report_assets: done

$ grep -n 'preregistered' benchmarks/sweep_matrix.yaml
29:preregistered:

$ python3 benchmarks/run_sweep.py --build build --dry-run
440 configurations declared, 850 rows in the summary
0 already present at commit b7a372f6d195, which is what a sweep would skip
410 present at 4abf914a7ea2.dirty, cd57032941a8.dirty and at no other commit,
  which a sweep from this build would measure again
30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3,
  pthreads 3, serial 3
30 stored rows that no declared configuration predicts: hybrid 30
exit 0

$ git diff --exit-code 35d8a6f -- tests/equivalence/
exit 0

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 133 file(s) scanned

$ ruff check benchmarks scripts tests
All checks passed!

$ git status --porcelain
  clean
```

The dry run's counts are unchanged by the new `preregistered:` key: 440
configurations declared, the same 440 the matrix declared before it.

The device comparison table now carries `GiB/s declared`, `GiB/s counted`,
`percent declared` and `percent counted`, and the bandwidth table carries the
plain probe, the non temporal probe and the derived rows. Running the generator
also rewrites the twelve tracked PNGs under `assets/figures/`, and it rewrites
them identically on a second run, so the generator is idempotent here; they
differ from the committed copies for reasons that predate this phase and are not
in it, so they were restored. Rebuilding published assets is phase A8b's job.

Findings: `MEAS-06`, the undercounted byte model and the triad that pays the
same read for ownership it does not charge, recorded with the question left
open and with the phase that settles it named; `MEAS-07`, found while running
the new probe three times, that the selection statistic's run to run spread is
as wide as the undecided band the rule uses, and that the statistic is a ratio
of two bests that need not come from the same worker count. The rule was deliberately left
exactly as registered, and `MEAS-07` records what has to be decided, and written
down, before phase A8b takes the publication measurement.

No sweep was run. The rows above are single configurations produced to fill this
section and none of them is written to `experiments/results/`.
### Phase A4: a row that says it pinned must have pinned

Done, in one commit. Nothing here is on a numerical path: no iterate moves, no
residual moves, `tests/equivalence/` is untouched, and the CSV header is the same
36 columns it was, with the `pinning_status` placeholder that phase A1.5 left
empty now filled.

**The tri state.** `cpu_for_worker` returned minus 1 both for "no pinning was
asked for" and for "the classification this policy needs did not succeed", and
every caller collapsed the two into silence. It returns a `PinTarget` now, an
outcome plus a processor number, and `PinOutcome` separates `NotRequested`,
`Bound`, `NotApplicable` and `Refused`. `pin_worker` is the single place that
turns a target into an outcome by calling `pin_this_thread`, so the rule lives
once instead of once per backend. `Compact` and `Scatter` compute exactly the
processors they computed before.

**The status, and who reports what.** `Backend::pinning_status()` is a pure
virtual returning one of `not_requested`, `bound`, `not_applicable` or
`refused`, with the count of refused workers appended after a colon when it is
not zero. Each backend reports what its own threads recorded: the OpenMP threads
write one slot each from inside the pinning region, the two pools write one slot
each and report through a handshake the constructor waits on, serial reports its
one thread, and the two distributed backends report the thread inside this rank.
The value is the worst outcome any worker saw, so one refusal is enough to make
the whole backend say `refused`.

**Three guards, deliberately overlapping.** `make_backend` throws
`BackendFailure` when `pcore` or `ecore` is requested and the classification did
not succeed, which on this guest is always. Every shared memory backend throws
from its constructor, after the pool is up, when a requested pinning did not bind
on every worker, and the message names the backend, the policy, the worker and
the outcome. The driver refuses to write a row whose `pinning` is not `none` and
whose `pinning_status` is not `bound`. With the first two the third cannot fire,
which is the point: it makes the invariant a property of the file rather than of
the backends that happen to exist today.

Throwing from a constructor that has already started threads means unwinding it
by hand, because a destructor never runs for an object that did not finish
constructing. `PthreadsBackend::stop_workers` and `JthreadBackend::shutdown` are
that teardown, shared with the destructors and with the `pthread_create` failure
path.

**The counter.** `pinning_failures_` was incremented without the mutex in the
constructor, with it in the worker, and read without it. It is a
`std::atomic<int>` now and the mutex is not taken for it anywhere. See `CONC-02`
for why that is a consistency fix and not a bug fix, and why ThreadSanitizer
will not find it.

**Gate A4.** The four runs are on the working tree at `ee508227fb32`, so their
`commit` column carries the `.dirty` suffix; they are gate output rather than a
measurement, none of them is written to `experiments/results/`, and the
orchestrator's re run of the gate on the committed tree is what produces the
clean stamp.

```text
$ build/pnl --solver jacobi --backend pthreads --size 63 --mode fixed \
    --iterations 10 --reps 1 --workers 4 --pinning pcore; echo "exit $?"
pnl: backend failure: pinning policy 'pcore' needs a performance core
classification and this machine did not yield one: no reliable performance
versus efficiency split visible from inside the guest: the largest gap in per
processor throughput was 0.003038 against a within group spread of 0.017394, so
the knee is taken from the aggregate scaling curve instead
exit 1
```

Wrapped for width; the message is one line on stderr, and standard output is
empty, so no row was written.

```text
$ build/pnl --solver jacobi --backend pthreads --size 63 --mode fixed \
    --iterations 10 --reps 1 --workers 4 --pinning compact
poisson2d_rich_63,3969,jacobi,pthreads,4,1,1,compact,deterministic,static,fixed,
10,0,iteration_cap,1.777864e-01,,63,1,0.001594,0.001594,0.001594,1,2.489628e+07,
0.5565,24.0,20260802,ee508227fb32.dirty,,1,1,32.0,bound,2026-09-05T23:12:16Z,
0.001594,cxx,cpp

$ build/pnl --solver jacobi --backend openmp --size 63 --mode fixed \
    --iterations 10 --reps 1 --workers 4 --pinning scatter
poisson2d_rich_63,3969,jacobi,openmp,4,1,1,scatter,deterministic,static,fixed,
10,0,iteration_cap,1.777864e-01,,63,1,0.000065,0.000065,0.000065,1,6.088170e+08,
13.6081,24.0,20260802,ee508227fb32.dirty,,1,1,32.0,bound,2026-09-05T23:12:16Z,
0.000065,cxx,cpp

$ build/pnl --solver jacobi --backend jthread --size 63 --mode fixed \
    --iterations 10 --reps 1 --workers 4 --pinning none
poisson2d_rich_63,3969,jacobi,jthread,4,1,1,none,deterministic,static,fixed,10,
0,iteration_cap,1.777864e-01,,63,1,0.001707,0.001707,0.001707,1,2.325579e+07,
0.5198,24.0,20260802,ee508227fb32.dirty,,1,1,32.0,not_requested,
2026-09-05T23:12:16Z,0.001707,cxx,cpp
```

Wrapped for width; each row is one line of 36 fields, the same count the header
prints. Reading the last three columns that matter by name: `pinning` and
`pinning_status` read `compact` and `bound`, `scatter` and `bound`, `none` and
`not_requested`. Two more runs outside the gate, kept because they exercise the
paths the gate does not: `--pinning ecore` fails the same way `pcore` does, and
`--backend serial --pinning compact` now writes a row reading `compact` and
`bound`, where before this phase serial ignored the request entirely.

```text
$ git diff --exit-code 35d8a6f -- tests/equivalence/
exit 0
$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 133 file(s) scanned
exit 0
$ make test
100% tests passed out of 11
```

`clang-format --dry-run --Werror` over `include src tests` exits zero at
clang-format 20.1.7 and `ruff check benchmarks scripts tests` reports all checks
passed.

Findings: `MEAS-08`, the two pinning policies that bound nothing and said they
had, recorded with the row counts that show the defect is latent and that the
committed pinning block stands, and with the neighbouring hybrid affinity
inheritance recorded and left to the phase that re measures it; `CONC-02`, the
`pinning_failures_` guarding, fixed as a consistency defect with the reason
ThreadSanitizer cannot see it written down.

No sweep was run. The rows above are single configurations produced to fill this
section and none of them is written to `experiments/results/`.
### Phase A5: take allocation out of the timed region

Done, in three commits: the code with its test, a repair of my own making, and
these records. No column is added or removed, the CSV keeps its 36 columns, and
no iterate moves.

**The workspace.** `Solver::solve` takes a `SolverWorkspace` the caller allocates
once, outside its repetition loop, and leaves the iterate in it as a view. Each
solver declares how many full size vectors it needs, so the shared stationary
driver asks for three, Richardson for two, conjugate gradient for four and block
Jacobi for four. The three argument `solve` stays as a convenience that
allocates a workspace and copies the answer out; nothing timed calls it and
every test does, which is why `tests/equivalence/` compiles unchanged.

**The specification's arithmetic, corrected.** Section 7 A5 says "403 MB across
three state vectors, or 537 MB for Richardson's four". Both sizes are right and
the second belongs to conjugate gradient: Richardson allocates two vectors, 269
MB, and `cg` allocates the four. Recorded in `MEAS-10`.

**One timed region.** The per repetition timed call is now
`pnl::bench::timed_repetition` in `include/pnl/bench/timed_solve.hpp`, and the
driver and the gate both drive it. Inside the clock: the barrier, the solve, the
barrier. Outside it: resetting the workspace to the problem's initial state,
which is a write over memory this process already owns.

**What the counting allocator found on its first run, which is the finding.**
The state was most of the bytes and almost none of the count. Against the tree
at `3faf2fc`, on `Poisson2D` at 63 squared with 12 fixed iterations, Jacobi
allocated 45 times per timed repetition and three of those were state vectors.
The rest were a `std::function` per `parallel_for`, `reduce` and `run_ordered`,
whose captures are far too large for the small object buffer, at three per
iteration; a `std::string` per precondition, built from a message literal on the
path that succeeds, once or twice per iteration inside the sweeps; a sweep
`std::function` on five of the twelve solvers; a full state snapshot and a
scratch vector per chunk inside both block sweeps; and a label string on the
progress bar. The full 24 row table and the arithmetic that accounts for all 45
are in `MEAS-10`.

**Gate A5.** From the working tree at `b7e8f807f938`, which is clean.

```text
$ make build && make test
100% tests passed out of 12

Label Time Summary:
convergence    =   0.68 sec*proc (1 test)
cuda           =   4.20 sec*proc (1 test)
equivalence    =   1.61 sec*proc (1 test)
mpi            =   0.76 sec*proc (3 tests)
style          =   3.76 sec*proc (2 tests)
unit           =   0.41 sec*proc (4 tests)
```

Eleven tests before this phase, twelve now; the new one is `test_no_allocation`.

```text
$ ctest --test-dir build --output-on-failure -R test_no_allocation
    Start 2: test_no_allocation
1/1 Test #2: test_no_allocation ...............   Passed    0.15 sec
100% tests passed out of 1

$ git diff --exit-code 35d8a6f -- tests/equivalence/
$ echo $?
0

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 137 file(s) scanned

$ git status --porcelain
$ echo $?
0
```

`clang-format --dry-run --Werror` over `include`, `src` and `tests` exits zero,
and `ruff check benchmarks scripts tests` reports "All checks passed!".

**The gate asserts zero, for all twelve solvers, on two backends.**
`tests/unit/test_no_allocation.cpp` replaces the global `operator new` and
`operator delete` and arms a counter at the two ends of the timed region,
through the observer hook of `timed_repetition`, so what is counted is what is
timed. It runs each of the twelve on `Poisson2D` at 63 squared on `serial` at
one worker and `openmp` at four, warms up once on the workspace it then measures
through, and requires zero. A third case allocates deliberately inside the armed
window and requires the count to move, so a counter that had stopped counting
could not pass the first two.

**No iterate moves, checked rather than assumed.** The same dump program built
against `3faf2fc` and against this tree wrote every solver's solution, residual
history, iteration count and evaluation count for both Poisson sources at n of
1, 2, 31 and 63 and for both dense families. All 67 dumps are bit identical. The
two observation rows below agree to the last digit on `relative_residual`.

**The observation, which is not a gate.** Jacobi at 4095 squared on `openmp` at
20 workers, 300 fixed iterations, 5 repetitions, on an otherwise quiet machine,
one run each side. Before is the tree at `3faf2fc`, after is `b7e8f807f938`.

| | seconds_median | seconds_min | seconds_max | spread (max - min) / median |
| --- | --- | --- | --- | --- |
| before | 5.085104 | 5.055086 | 5.228016 | 0.0340 |
| after | 4.890023 | 4.782769 | 4.943681 | 0.0329 |

The rows themselves:

```text
poisson2d_rich_4095,16769025,jacobi,openmp,20,1,1,none,deterministic,static,
fixed,300,0,iteration_cap,3.262973e-02,,4095,1,5.085104,5.055086,5.228016,5,
9.893028e+08,22.1126,24.0,20260802,3faf2fc7a1e9,,1,1,32.0,not_requested,
2026-09-05T23:30:58Z,5.113782;5.228016;5.057569;5.055086;5.085104,cxx,cpp

poisson2d_rich_4095,16769025,jacobi,openmp,20,1,1,none,deterministic,static,
fixed,300,0,iteration_cap,3.262973e-02,,4095,1,4.890023,4.782769,4.943681,5,
1.028770e+09,22.9948,24.0,20260802,b7e8f807f938,,1,1,32.0,not_requested,
2026-09-05T23:54:14Z,4.782769;4.848389;4.943681;4.918108;4.890023,cxx,cpp
```

Wrapped for width; each row is one line. Neither is written to
`experiments/results/`.

The median falls by 3.8 percent, the minimum by 5.4 percent. The spread moves
from 0.0340 to 0.0329, which is a change of one part in a thousand of the
median and is not evidence of anything: five repetitions on a shared WSL2 guest
cannot resolve a difference that size, and the specification says so in advance,
which is why this is an observation and the counting test is the gate.

**The first repetition, before and after, which is where the page fault cost was
supposed to show.** It did not show before. In run order the before
repetitions are 5.113782, 5.228016, 5.057569, 5.055086, 5.085104: the first is
5.113782 against a mean of 5.106444 for the other four, so it is 0.1 percent
slower, and the slowest repetition is the second, not the first. After, the run
order is 4.782769, 4.848389, 4.943681, 4.918108, 4.890023: the first is 2.4
percent *faster* than the mean of the other four.

The honest reading is that the per repetition first touch cost was real but was
not concentrated in the first repetition, because it was paid in every
repetition equally. Each repetition freed its vectors before the next allocated
its own, and glibc handed back memory the process had already faulted in, so
what every repetition paid was the mapping bookkeeping and the value initialising
write of 384 MiB rather than a fault storm that the first repetition could absorb
on behalf of the rest. That is consistent with the size of the win: 195
milliseconds of a 5.09 second repetition is about what writing 384 MiB costs on
this machine, and it is now paid once at start up instead of five times inside
the clock. The after run's fast first repetition is the remaining warm up effect
with nothing on top of it.

**One repair of my own making, in its own commit.** I edited the sources with a
script that wrote them in text mode from the Windows side, which turned every
line ending into CRLF, and `clang-format` keeps whatever ending a file already
has. The first commit therefore recorded a rewrite of every line of nineteen
files instead of the change its message describes. `b7e8f807f938` puts the line
endings back and changes nothing else, so the cumulative diff from `3faf2fc` is
the phase A5 change and nothing else. No tracked file outside
`experiments/results/summary.csv`, which was already CRLF before this phase,
contains a carriage return.

Findings: `MEAS-10`, allocation inside the timed region and the five causes
besides the state that the counting allocator turned up.

Design decisions: decision 1 amended, because the chunk level callbacks keep
their shape and lose their `std::function`; decision 21 added, for the workspace
and for why parallel first touch is future work on a single socket machine
rather than part of this phase.

No sweep was run. The two rows above are single configurations produced to fill
this section.
### Phase A6: the hybrid worker count

Done, in one commit. No iterate moves and no column is added or removed; what
moves is the number in an existing column, on the hybrid backend only.

**The override.** `HybridBackend::worker_count()` returns `ranks_ * threads_`.
It used to be inherited from `MpiBackend`, which returns the rank count, while
the constructor set `config_.workers` to the product and a comment beside it
called the product the number the scaling curve is plotted against. The row
takes its `workers` column from `worker_count()`, not from `config_.workers`, so
the right number was computed and then not used.

**`predicted_workers` was already right, and that is the finding.**
`benchmarks/run_sweep.py` has predicted `ranks * threads` for hybrid since the
commit that added the sweep harness. It was therefore predicting what the binary
ought to report and not what it did, so no hybrid configuration has ever
resumed: the sweep re ran all 15 of them every time. The dry run says so from
the other side, and said so before this phase touched anything, because the
stored rows match no declared configuration even with the commit ignored. The
only change to the prediction here is that it clamps the thread count to at
least one before multiplying, the same clamp the driver applies when it works
out the rank count and the same one the backend applies to the thread count. It
changes no number for any declared configuration, all of which are 20 workers at
4 threads per rank and predict 20 either way.

**The 30 committed hybrid rows must be re measured, and are now reported as
missing.** They describe their own run correctly, 5 ranks of 4 threads, and the
`ranks` and `threads_per_rank` columns beside the wrong one say so. What they
cannot do is sit on the same axis as the shared memory rows they are compared
against. They cannot be repaired in place either: `workers` is in
`IDENTITY_FIELDS`, so editing it would forge an identity that no run produced.
The dry run reports all 15 hybrid configurations as having no stored row at any
commit, which is the honest answer.

**Gate A6.** The rows below are from the working tree at `d2ec07bab134`, so
their `commit` column carries the `.dirty` suffix; they are gate output rather
than a measurement and none is written to `experiments/results/`.

```text
$ make test
100% tests passed out of 11
```

`test_mpi_1rank`, `test_mpi_2rank` and `test_mpi_4rank` are three of the eleven
and all three pass, which is this phase's requirement that the MPI suite still
runs at 1, 2 and 4 ranks.

The gate command as the task file writes it passes no `--threads-per-rank`, and
that flag is the only one the hybrid backend reads for its thread count:
`--workers` reaches `Config::workers`, which `MpiBackend` overwrites with the
rank count and `HybridBackend` overwrites with the product. Two ranks and the
default of one thread per rank is therefore two workers, and the row says two:

```text
$ mpirun -np 2 build/pnl --solver jacobi --backend hybrid --size 63 \
    --mode fixed --iterations 10 --reps 1 --workers 4
poisson2d_rich_63,3969,jacobi,hybrid,2,2,1,none,deterministic,static,fixed,10,0,
iteration_cap,1.777864e-01,,63,1,0.000071,0.000071,0.000071,1,5.559524e+08,
12.4265,24.0,20260802,d2ec07bab134.dirty,,1,1,32.0,not_requested,
2026-09-05T23:19:08Z,0.000071,cxx,cpp
```

Asking for the eight the gate's note anticipates, and for the twenty the sweep
actually launches:

```text
$ mpirun -np 2 build/pnl --solver jacobi --backend hybrid --size 63 \
    --mode fixed --iterations 10 --reps 1 --workers 8 --threads-per-rank 4
poisson2d_rich_63,3969,jacobi,hybrid,8,2,4,none,deterministic,static,fixed,10,0,
iteration_cap,1.777864e-01,,63,1,0.002047,0.002047,0.002047,1,1.939225e+07,
0.4335,24.0,20260802,d2ec07bab134.dirty,,1,1,32.0,not_requested,
2026-09-05T23:19:09Z,0.002047,cxx,cpp

$ mpirun --oversubscribe -n 5 build/pnl --solver jacobi --backend hybrid \
    --size 63 --mode fixed --iterations 10 --reps 1 --workers 20 \
    --threads-per-rank 4
poisson2d_rich_63,3969,jacobi,hybrid,20,5,4,none,deterministic,static,fixed,10,
0,iteration_cap,1.777864e-01,,63,1,0.000188,0.000188,0.000188,1,2.112339e+08,
4.7214,24.0,20260802,d2ec07bab134.dirty,,1,1,32.0,not_requested,
2026-09-05T23:19:09Z,0.000188,cxx,cpp
```

Wrapped for width; each row is one line. Reading the three columns by name:
`workers`, `ranks` and `threads_per_rank` are 2, 2, 1; then 8, 2, 4; then 20, 5,
4. The product holds in all three.

The dry run, hybrid lines only. All fifteen read `run`, which is the status for
a configuration with no stored row at any commit:

```text
$ python3 benchmarks/run_sweep.py --build build --dry-run | grep -i hybrid
run          backend_cost         /usr/bin/mpirun --oversubscribe -n 5 build/pnl
    --solver jacobi --backend hybrid --problem poisson --rhs rich --size 511
    --workers 20 --threads-per-rank 4 --pinning none --reduction deterministic
    --schedule static --mode fixed --iterations 200 --check-interval 1000000
    --tolerance 1e-08 --reps 5 --seed 20260802 --label backend_cost
```

One of the fifteen, wrapped for width; the other fourteen differ only in solver,
one of five, and size, one of three. The summary the same run prints:

```text
440 configurations declared, 850 rows in the summary
0 already present at commit d2ec07bab134.dirty, which is what a sweep would skip
410 present at 4abf914a7ea2.dirty, cd57032941a8.dirty and at no other commit,
which a sweep from this build would measure again
30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3,
pthreads 3, serial 3
30 stored rows that no declared configuration predicts: hybrid 30
```

The last line is the finding, and it is the reading the phase asked for: the 30
committed hybrid rows carry the old count, no declared configuration predicts
them, and they must be re measured rather than resumed. The fifteen non hybrid
entries on the line above are the `cg` on `dense_dd` configurations of the dense
block, five backends at each of three sizes, which the committed sweep never
produced a row for. They predate this phase and nothing here touches them.

Findings: `MEAS-09`, the hybrid worker count and the resume miss that has come
with it since the sweep harness was written.

No sweep was run. The rows above are single configurations produced to fill this
section and none of them is written to `experiments/results/`.

### Phase A7: surface the dispersion that is already collected

Done, in two commits: the sweep matrix, then the generator and its test. No
solver, backend or numeric path is touched, and no measurement is taken. What
changes is what the generator is allowed to say about measurements that already
exist.

**Commit 1, the sweep matrix.** `meta.repetitions` goes from 5 to 15. The two
convergence blocks keep their explicit `repetitions: 1`, because an iteration
count is deterministic and repeating it measures nothing; every other block is a
timing block and takes the new default. `reduction_cost` goes from 200
iterations to 3000 and `schedule_cost` from 100 to 1100, so that no timed run in
either block is shorter than a second. The arithmetic is in each block's `why`:
the fastest configuration sets the length, `200 x 1.000 / 0.0723 = 2767` rounded
up to 3000, and `100 x 1.000 / 0.0935 = 1070` rounded up to 1100.

**The sweep cost estimate.** Computed from the committed medians at
`cd57032941a8`, the one untimed warm up the driver runs before every
configuration, the new repetition counts and the new iteration counts. The
committed matrix is 425 configurations and this phase adds none.

```text
block                       cfgs  sum median  iter x  reps  runs    seconds
convergence_counts            36      166.20    1.00     1     2     332.40
convergence_counts_large       6       33.75    1.00     1     2      67.49
backend_cost                  90       70.50    1.00    15    16    1127.99
scaling                       66        5.57    1.00    15    16      89.16
pinning                       36        4.42    1.00    15    16      70.78
reduction_cost                 8        2.18   15.00    15    16     523.57
schedule_cost                  6        0.62   11.00    15    16     108.87
mpi_scaling                   18        4.86    1.00    15    16      77.71
device_comparison             24       31.54    1.00    15    16     504.64
dense                        135        5.25    1.00    15    16      83.99
total                        425                                    2986.60
```

`sum median` is one repetition of every configuration in the block, added up.
`runs` is `reps + 1` because `src/main.cpp` runs one untimed warm up before the
timed repetitions. That is 2986.60 seconds, 49.8 minutes, of running. On top of
it 425 process launches, 348 plain and 77 under `mpirun`, at roughly 0.3 and 1.5
seconds each, is another 220 seconds. **About 53 minutes for a full sweep**,
against 19.2 minutes for the same matrix at 5 repetitions and the old iteration
counts, a factor of 2.6.

That is inside the 60 to 90 minutes Section 13 budgets, with room for the 226
configurations Parts C and D add. It is also an upper bound in one direction:
the medians it is built from were measured before phase A1 removed the serial
state copy and before phase A5 took allocation out of the timed region, so the
real runs should be shorter than the numbers above. It is a lower bound in
another: nothing here charges problem construction at 4095 squared, which is
outside the timed region but inside the wall clock.

**Commit 2, the generator.** Spread is defined once, `(max - min) / median`, and
used by every table, every figure and both rules. Every timing table gained a
spread column; every timing figure gained min to max whiskers. Bounds on a
derived quantity, a speedup or an efficiency, are taken at the corners of its
inputs' intervals, and the convention is written into a constant, into a comment
and onto the face of every figure that draws one. `separable_effect` returns the
effect or the phrase `not separable at this precision`, deciding on the
magnitude against the larger of the two rows' spreads, and the four tables that
state a difference all go through it. Beside each of them, and beside the knee,
the generator writes a `<name>_verdicts.tex` fragment of one sentence per
comparison for `results.tex` to input in phase E4. `dispersion.tex` reports n,
the median and the max per block and deliberately no p90.

The knee keeps its two segment least squares fit and gains a bootstrap: 2000
refits at seed 20260906, each resampling the repetitions behind every worker
count. On the 1.0.0 data there is no `seconds_reps` column, so every backend
falls back to a triangular draw on the recorded minimum, median and maximum;
that is named in a column of the table, never mixed with measured repetitions
inside one backend, and the caption says what it is worth.

`--allow-dirty` is Section 7 A8 step 6's flag. Without it the generator refuses
any row whose commit stamp ends in `.dirty` and names how many it found, which
is all 425 rows of the published generation. The CI reports job now passes it,
and the Makefile passes `ASSET_FLAGS`, empty by default, so `make report` fails
by default with the reason and `make report ASSET_FLAGS=--allow-dirty` works.

**Findings.** `MEAS-11`, dispersion collected and discarded, with the per block
table recomputed from the committed data and the three headline claims that sit
inside the noise. 27 of the 56 comparisons the committed data supports now read
the phrase: 1 of 25 in `backend_cost`, 21 of 24 in `pinning`, 4 of 4 in
`reduction_cost`, 1 of 3 in `schedule_cost`.

**Gate A7.**

```text
$ python3 scripts/gen_report_assets.py 2>&1 | tail -3
gen_report_assets: refusing to build a published asset from 425 of 425 row(s)
whose commit stamp ends in .dirty. The source that produced those numbers is not
in git history. Re measure from a clean tree, or pass --allow-dirty to generate
anyway, which is for development only and produces nothing publishable.
gen_report_assets: using commit cd57032941a8.dirty, ignoring 425 row(s) from
earlier commits
$ echo $?
3
```

Wrapped for width; the refusal is one line. Exit 3, which is what the gate
requires: every committed row is dirty.

```text
$ python3 scripts/gen_report_assets.py --allow-dirty
gen_report_assets: using commit cd57032941a8.dirty, ignoring 425 row(s) from earlier commits
gen_report_assets: --allow-dirty, 425 of 425 row(s) carry a .dirty commit stamp and nothing built from them is publishable
gen_report_assets: 425 rows from experiments/results/summary.csv
  figure  assets/figures/scaling_speedup-light.png
  ...
  table   report/tables/knee.tex
  verdict report/tables/knee_verdicts.tex
  table   report/tables/dispersion.tex
gen_report_assets: done
$ echo $?
0
```

Twelve figure lines, nine table lines and five verdict lines, elided in the
middle.

```text
$ ls report/tables/
backend_cost.tex           dispersion.tex          reduction_cost.tex
backend_cost_verdicts.tex  knee.tex                reduction_cost_verdicts.tex
bandwidth.tex              knee_verdicts.tex       schedule_cost.tex
convergence.tex            pinning.tex             schedule_cost_verdicts.tex
device_comparison.tex      pinning_verdicts.tex

$ ls report/figures/
backend_cost.pdf        device_efficiency.pdf  mpi_scaling.pdf
convergence_growth.pdf  iteration_counts.pdf   scaling_speedup.pdf

$ grep -li spread report/tables/*.tex
report/tables/backend_cost.tex
report/tables/backend_cost_verdicts.tex
report/tables/device_comparison.tex
report/tables/dispersion.tex
report/tables/pinning.tex
report/tables/pinning_verdicts.tex
report/tables/reduction_cost.tex
report/tables/reduction_cost_verdicts.tex
report/tables/schedule_cost.tex
report/tables/schedule_cost_verdicts.tex

$ grep -o "percent interval" report/tables/knee.tex
percent interval
```

Every timing table carries a spread column, `dispersion.tex` and the five
verdict fragments exist, and `knee.tex` carries the interval.

```text
$ python3 benchmarks/run_sweep.py --build build --dry-run
440 configurations declared, 850 rows in the summary
0 already present at commit 3c8916ea8446.dirty, which is what a sweep would skip
410 present at 4abf914a7ea2.dirty, cd57032941a8.dirty and at no other commit,
which a sweep from this build would measure again
30 have no stored row at any commit: hybrid 15, jthread 3, mpi 3, openmp 3,
pthreads 3, serial 3
30 stored rows that no declared configuration predicts: hybrid 30
$ echo $?
0
```

The commit on the second line is whatever binary is sitting in `build/` when the
dry run is asked, not a fixed number: decision 16 puts the commit in the resume
key, so it moves with every build. The count beside it is the stable part, and
zero is the honest answer for a build nothing has been measured from. The last
three lines are phase A6's finding and are unchanged by this phase. The new
counts reach the driver: every timing configuration is launched with
`--reps 15`, `reduction_cost` with `--iterations 3000` and `schedule_cost` with
`--iterations 1100`.

```text
$ make build && make test
100% tests passed out of 13

Label Time Summary:
convergence    =   0.67 sec*proc (1 test)
cuda           =   4.13 sec*proc (1 test)
equivalence    =   1.75 sec*proc (1 test)
mpi            =   0.78 sec*proc (3 tests)
style          =   3.36 sec*proc (2 tests)
unit           =   2.22 sec*proc (5 tests)
```

Thirteen tests, one more than before: `test_gen_report_assets` is the fifth under
the `unit` label and takes 1.80 seconds.

```text
$ ruff check benchmarks scripts tests
All checks passed!

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 144 file(s) scanned

$ git status --porcelain
```

Nothing, after both commits. The regenerated figures under `report/figures/` and
tables under `report/tables/` are ignored, and the tracked PNGs under `assets/`
were restored with `git checkout -- assets/` before committing: they are rebuilt
once, at phase A8b, from the publication session.

**Figures inspected.** `assets/figures/backend_cost-light.png` and
`assets/figures/scaling_speedup-light.png`, opened as images rather than trusted
because the generator printed their names. The bar chart carries a horizontal
whisker on all six bars, the value labels moved out past the whisker cap so they
no longer collide with it, and the axis label reads
`seconds, median of 5 repetitions`, computed from the `reps` column rather than
typed. The scaling chart carries a vertical whisker on all 66 points across the
three backends, and its note line states the corner rule for derived bounds. The
scaling figure is also the picture of the knee interval: between 4 and 20 workers
the `openmp` curve is flat and its whiskers overlap every neighbouring point,
which is why the bootstrap returns 4 to 20 for a fit whose point estimate is 5.

**Not done, and why.** `results.tex` is untouched. The five verdict fragments and
`dispersion.tex` are emitted for it to input, which phase E4 does when it
rewrites the prose; wiring them in here would be that phase's work and would
leave prose quoting numbers the same phase is about to rewrite. No sweep was run:
every number in this section is either recomputed from the committed rows or the
output of a gate command.

### Phase A8a: provenance repair, interim data only

Done, in three commits: the harness and the generator, the pre registration
amendment, then the archive. The published PDFs are not rebuilt here. That is
phase A8b's, once, after the numerics freeze, and doing it twice would spend the
expensive half of the phase twice.

**Commit 1, the harness and the generator.** The sweep refuses to measure from a
tree with uncommitted changes to the sources that decide the result, asking git
the same narrowed pathspec `CMakeLists.txt cmake include src tests benchmarks
scripts Makefile` that `CMakeLists.txt` asks before it appends `.dirty` to the
stamp. It refuses separately on a binary already stamped `.dirty`, because a
tree can be clean while the binary carries a stamp from before the commit and
the stamp is what every row will claim. `--allow-dirty` lifts both and says in
its help that it is for development only.

Sessions write `manifest-<commit>-<timestamp>.json`, UTC, with the dot of a
`.dirty` stamp written as a dash in the name. `--refresh-bandwidth` updates the
manifest of the commit it refreshes and creates one only if that commit has
none. `declared`, `executed`, `skipped_already_present` and `rows_total` are
written at the top level as well as inside `counts`, from one dictionary in one
statement so they cannot drift.

The generator no longer takes `data["commit"].iloc[-1]`. It refuses on more than
one commit and names each with its row count, sorts by `measured_at` within a
generation, and selects the manifest whose name carries the commit it is
publishing. `--results-dir` on both, with `RESULTS_DIR` in the Makefile and
`sweep-interim` and `bandwidth-refresh-interim` pointing it at
`experiments/results/interim/`.

**Commit 2, the pre registration amendment.** MEAS-07 required a decision before
the publication session and this is it, dated 2026-09-06, appended to the
registration in `benchmarks/sweep_matrix.yaml` under
`preregistered.traffic_model.amended` and to the same section of
`docs/comparison_methodology.md`. The thresholds, 1.10 and 1.20, and the three
outcome sentences are exactly as registered on 2026-09-05 and are not touched.

What changes is how the statistic is taken. `pnl --bandwidth` runs five
repetitions of each triad at every worker count, alternating plain and non
temporal so a repetition's two arms see the same machine, and prints all five
figures per count for both probes beside the best of them. The driver takes `w*`
as the worker count where the plain triad's median is highest, `S` as the median
of the per repetition ratios at `w*`, and records `w*`, the ten figures, the five
ratios, `S`, the interval and the outcome under `traffic_model` in the manifest.
If the interval contains either threshold the outcome is unresolved whatever `S`
is, which is ground rule 7 applied to the denominator. One run of the amended
probe takes 27 seconds on this machine, timed with `date` either side of it.

**Commit 3, the archive.** The 425 rows at `4abf914a7ea2.dirty` moved to
`experiments/results/archive/summary-4abf914a7ea2-dirty.csv`, leaving the 425
published rows at `cd57032941a8.dirty` in `summary.csv` until A8b replaces them.
`session_manifest.json` moved to
`experiments/results/archive/manifest-cd57032941a8-dirty.json` by `git mv`, and
its `.gitignore` negation went with it in the same commit so the file is never
tracked and ignored at once. `experiments/results/archive/README.md` says what
each file is, that both generations are dirty, that the manifest records
`"declared": 8, "executed": 0, "skipped_already_present": 8` and therefore a
later eight configuration re run rather than the 440 configuration sweep, and
that `bandwidth_refreshed` is absent because the refresh never ran in the
committed session.

**The refusal, demonstrated.** An empty untracked file at
`src/scratch_provenance_probe.cpp` and nothing else:

```text
$ benchmarks/run_sweep.sh --build build --dry-run
run_sweep: refusing to measure from a tree with uncommitted changes to the
sources that decide the result. Ground rule 6: the exact source that produced a
number has to exist in git history. Commit or stash these, or pass
--allow-dirty, which is for development only.
?? src/scratch_provenance_probe.cpp
$ echo $?
2
```

The file removed, the same command exits 0 and reports the resume calculation:
6 configurations declared for `convergence_counts_large`, 0 already present at
commit `651511d59a43`, 6 present at `4abf914a7ea2.dirty` and
`cd57032941a8.dirty` and at no other commit.

**The small block.** `convergence_counts_large`, the cheapest block in the
matrix. The estimate table under phase A7 prices it at 67.49 seconds, against
70.78 for `pinning` and 77.71 for `mpi_scaling`, the next two.

```text
benchmarks/run_sweep.sh --build build \
    --results-dir experiments/results/interim --only convergence_counts_large
benchmarks/run_sweep.sh --build build \
    --results-dir experiments/results/interim --refresh-bandwidth
```

It wrote 6 rows and one manifest,
`manifest-651511d59a43-20260906T005711Z.json`, and the refresh updated that same
file rather than adding a second. Every row carries `651511d59a43` with no
`.dirty`, `executed` is 6 and `bandwidth_refreshed` is
`2026-09-06T02:59:10+0200`.

The refresh recorded a traffic model outcome of **unresolved**: `w*` of 8
workers, five ratios of 1.1172, 1.1141, 1.0323, 1.0365 and 1.0166, `S` of
1.0365, interval 1.0166 to 1.1172. The interval contains 1.10, so rule 7 fires
and the outcome is unresolved regardless of where `S` falls. That is an
observation from a six configuration block and not the publication measurement,
which is A8b's.

`python3 scripts/gen_report_assets.py --results-dir experiments/results/interim`
then ran with no `--allow-dirty`, because the rows are clean, selected
`manifest-651511d59a43-20260906T005711Z.json` by name and exited 0. From one
block it produced what one block supports: `convergence_growth` in both themes,
and `bandwidth.tex` from the manifest. The other five figures and eight tables
need `scaling`, `backend_cost`, `device_comparison`, `mpi_scaling`, `pinning`,
`reduction_cost`, `schedule_cost` and the two convergence count blocks, none of
which ran.

**The full interim sweep is the orchestrator's**, on an idle machine, with

```text
make sweep-interim && make bandwidth-refresh-interim
```

`experiments/results/interim/` should be emptied first, because the six rows
above carry commit `651511d59a43` and a sweep from a later build would leave two
generations in one file, which the generator now refuses.

**The full interim sweep, as run.** It ran from a clean tree at commit
`b6d15e659002` into `experiments/results/interim/` and wrote 425 rows, all at
that one commit and none carrying a `.dirty` stamp, in 2487 s of wall clock. The
manifest is `manifest-b6d15e659002-20260906T011231Z.json`, with `executed` 425
against `declared` 440. The 15 failures are exactly the declared inapplicable
`cg` on `dense_dd` configurations and nothing else. `bandwidth_refreshed` reads
`2026-09-06T03:54:31+0200`, the plain host triad's best is 67.528 GiB/s at 8
workers, and the `traffic_model` statistic is 1.0608 with an interval of 0.9269
to 1.1831, so the outcome is unresolved: the interval spans both thresholds.
`python3 scripts/gen_report_assets.py --results-dir experiments/results/interim`
then ran with no `--allow-dirty`, because the rows are clean, and built every
asset. The published PDFs are not rebuilt from it, which is what the
specification says about this phase; that is A8b's.

**Gate.**

```text
$ git status --porcelain
$ build/pnl --solver jacobi --backend serial --size 4 --mode fixed \
      --iterations 1 --reps 1 | awk -F, '{print $27}'
651511d59a43
$ awk -F, 'NR>1 {print $27}' experiments/results/interim/summary.csv | sort -u
651511d59a43
$ ls experiments/results/interim/manifest-*.json
experiments/results/interim/manifest-651511d59a43-20260906T005711Z.json
$ python3 -c "import json,glob; m=json.load(open(sorted(glob.glob(
      'experiments/results/interim/manifest-*.json'))[-1]));
      print(m['executed'], m.get('bandwidth_refreshed'))"
6 2026-09-06T02:59:10+0200
$ awk -F, 'NR>1 {print $27}' experiments/results/summary.csv | sort -u
cd57032941a8.dirty
$ wc -l experiments/results/archive/summary-4abf914a7ea2-dirty.csv
426 experiments/results/archive/summary-4abf914a7ea2-dirty.csv
$ python3 scripts/gen_report_assets.py --allow-dirty
gen_report_assets: --allow-dirty, 425 of 425 row(s) carry a .dirty commit stamp
and nothing built from them is publishable
gen_report_assets: no manifest for commit cd57032941a8.dirty in results, falling
back to the archived archive/manifest-cd57032941a8-dirty.json. An archived
manifest does not describe the session that measured these rows, so every
bandwidth figure below is provenance this generation does not have.
gen_report_assets: 425 rows at commit cd57032941a8.dirty from
experiments/results/summary.csv
...
gen_report_assets: done
$ make build && make test
100% tests passed out of 13
$ ruff check benchmarks scripts tests
All checks passed!
$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 148 file(s) scanned
$ clang-format --dry-run --Werror over include, src and tests
```

The column used for the commit is 27, which is where `commit` still sits in the
36 column schema.

The stamp in that transcript is `651511d59a43`, the commit the small block was
measured at, because the transcript was taken before the gate reached
`make build`. The archive commit changes no path in the dirtiness pathspec, so
nothing it touches could have altered a measurement, but `make build` reconfigures
and the stamp follows `HEAD`, so the gate rebuilds and every rerun of it prints
whatever `HEAD` is then, which is a moving hash and is deliberately not written
down here. It carries no `.dirty`, which is what the gate line asks, and that is
the whole of what it asserts. The interim rows keep the commit they were
measured at, and the
manifest beside them carries the same one, which is the property that matters:
the binary a gate happens to be run with and the binary a row was measured with
are different questions, and conflating them is how a resumed sweep silently
mixes two builds.

The generator falling back to the archived manifest is the one place the
published generation still leans on provenance it does not have. It is reachable
only under `--allow-dirty`, which has already said nothing built with it is
publishable, and it announces itself on every run. Without the fallback the
bandwidth table and the device efficiency figure would silently lose their
denominator, and `make report` and the CI reports job would fail on a missing
figure. Phase A8b removes the need for it by measuring a generation that has a
manifest of its own.

**Findings.** `PROV-03`, two generations in one summary told apart by the order
they were appended in. `PROV-04`, the committed manifest describing a different
session, with the absent `bandwidth_refreshed` as the mechanism behind the four
irreconcilable bandwidth figures of Section 4.3. Both in
`docs/ENGINEERING_LOG.md`.

**Not done, and why.** No published PDF is rebuilt and no tracked asset is
committed: the generator rewrote `assets/figures/*.png` twice during this phase
and both times they were restored with `git checkout -- assets/` before
committing. The hand typed bandwidth tables in `results.tex`, `README.md` and
`docs/comparison_methodology.md` still stand; deleting them is step 5 of A8, it
is atomic with the generator work that emits `tables/bandwidth_scaling.tex`, and
both are A8b's. No full sweep was run here.

### Phase E2: citation, licence headers and the disclosure decision

Done, in two commits. The phase has no dependency on any other, which is why it
ran between the sweep phases rather than after them.

**Commit 1, the unconditional artifacts.** `CITATION.cff` at the root, CFF 1.2.0,
`type: software`, the author from `LICENSE`, `license: MIT`, `version: 1.0.0`
matching `project()` with a comment saying E5 moves both, `date-released`
`2026-08-02` from the changelog entry for 1.0.0, and `repository-code` from
`git remote get-url origin`. It parses under `yaml.safe_load`. It carries **no
DOI**, because there is no Zenodo deposit and a DOI shaped string that resolves
to nothing would be the only unverifiable claim in the repository.

`SPDX-License-Identifier: MIT` went to the top of **63 files**: 51 C++ and CUDA
files under `include/`, `src/` and `tests/` with a `//` comment above
`#pragma once` where there is one, 9 Python and shell files under `scripts/`,
`benchmarks/` and `tests/` with a `#` comment placed after the shebang, and the
three build files `CMakeLists.txt`, `tests/CMakeLists.txt` and `Makefile`. The
version 2 specification says 47 carried no licence line; the number is 63 because
it counts the build files and every Python self test the specification's figure
appears not to. Not `.tex`, `.bib`, `.md`, `.yaml` or `.csv`. `make build && make
test` ran afterwards for the reason the task file gives, that a header line above
a shebang or a stray comment in a `.cu` file breaks quietly, and it is green.

This commit also replaces the `pending` in the phase A8a note. The full interim
sweep has run and its facts are recorded there.

**Commit 2, the process and the authorship.** `CONTRIBUTING.md` was good on
technical rules and silent on who reviews, who merges and how an argument ends.
It now says: pull requests against `main`, the owner reviews and merges, nothing
merges with a red CI, a measurement dispute is settled by a run on the target
machine with the spread recorded, and a design dispute by a new entry in
`docs/DESIGN_DECISIONS.md` that names the option it rejects. Then a short
`Authorship` section, in the owner's voice: the repository was built by its owner
driving an AI coding agent from written specifications, the specifications and
the design decisions are the owner's, the agent produced code and prose under
them, and every number comes from a run on the owner's machine. The same sentence
is the `notes` field of `CITATION.cff`, so it reaches anyone who cites the
software and never opens the repository.

**The two decisions.** Decision 22 in `docs/DESIGN_DECISIONS.md` records that the
JOSS and community apparatus is not built in 1.1.0: no `paper.md`, no
`paper.bib`, no `CODE_OF_CONDUCT.md`, no `SECURITY.md`, no issue or pull request
templates, no Zenodo deposit, and therefore no DOI. For a single author study
whose stated purpose is what the execution model costs, that set touches no
measurement and changes no number, and it can be built on the day a submission is
decided against whatever that venue asks for then. The same decision records the
authorship disclosure as deliberate rather than omitted, and leaves
`dependabot.yml` to phase B2, which owns the workflows and the pinned actions.

**Gate.**

```text
$ python3 -c "import yaml; d=yaml.safe_load(open('CITATION.cff'));
      print(d['cff-version'], d['version'], d['authors'])"
1.2.0 1.0.0 [{'family-names': 'Badejo', 'given-names': 'Olajide'}]
$ grep -L 'SPDX-License-Identifier: MIT' $(git ls-files 'include/*' 'src/*' \
      'tests/*' 'scripts/*' 'benchmarks/*' | grep -E '\.(hpp|cpp|cu|cuh|py|sh)$')
$ grep -c 'SPDX-License-Identifier: MIT' CMakeLists.txt tests/CMakeLists.txt \
      Makefile
CMakeLists.txt:1
tests/CMakeLists.txt:1
Makefile:1
$ clang-format --dry-run --Werror over include src tests
$ ruff check benchmarks scripts tests
All checks passed!
$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 165 file(s) scanned
$ make build && make test
100% tests passed out of 13
$ git status --porcelain
```

**Findings.** None. Nothing in this phase can fail in an interesting way; the
only fault it could have produced is the one the build guards against, a licence
comment placed above a shebang, and it was not produced.

**Not done, and why.** No `paper.md`, `paper.bib`, `CODE_OF_CONDUCT.md`,
`SECURITY.md`, issue or pull request templates, `dependabot.yml` or Zenodo
deposit, per decision 22. No DOI anywhere. No published PDF or tracked asset is
rebuilt here; nothing in this phase changes a number.

### Phase B1: install, export, consume

Done, in three commits. Release 1.0.0 exported no public API at all: no
`install()`, no `export()`, no package configuration file, no namespaced target
and no version header, so `find_package(pnl)` was impossible. This phase creates
that API for the first time, which is the whole argument for 1.1.0 being a minor
bump rather than a major one. The gate is that a stranger can build
`examples/poisson.cpp` against a staged install.

**Commit 1, the flag split.** `pnl_flags` was one `INTERFACE` target carrying
two unrelated kinds of flag, linked `PUBLIC` into `pnl_core`, so exporting it as
it stood would have handed every consumer `-march=native` and a warnings as
errors policy. It now carries `-ffp-contract=off` and the `cxx_std_20` compile
feature and nothing else, which is exactly what a consumer must compile with for
the bit identity claim to be true of their own translation units; the library is
about three quarters headers, so those two have to travel with the headers.
`pnl_dev_flags` carries `-Wall -Wextra -Wpedantic`, `-O3 -march=native` under
Release and `-Werror` under `PNL_WERROR`, is populated only inside
`if(PROJECT_IS_TOP_LEVEL)`, is linked `PRIVATE` into `pnl_core`, `pnl`,
`pnl_test_main`, every test binary and `test_mpi`, and is never installed.
`PNL_WERROR` now defaults to `OFF` and the Makefile's `configure` target passes
`-DPNL_WERROR=ON`, so the developer build is exactly as strict as it was and the
existing CI job, which already passes the flag explicitly, is unaffected.

`pnl_cuda` deliberately does not link `pnl_dev_flags`: every option in that
target is guarded on `$<COMPILE_LANGUAGE:CXX>` and `pnl_cuda` compiles only CUDA
sources, so it would contribute nothing to a compile line while entangling the
target with the export set. Its host warning set stays in its own `-Xcompiler`
list.

The trap in the other direction was checked in the same commit, because it is
the one that would silently change every number measured after phase A8:

```text
$ grep -m1 'main.cpp' build/compile_commands.json |
      grep -o -e '-march=native' -e '-O3' -e '-ffp-contract=off' | sort -u
-O3
-ffp-contract=off
-march=native
```

The same three appear on `equivalence/test_equivalence.cpp` and on
`src/backend/factory.cpp`, so neither the measured binary nor the suite that
asserts bit identity moved.

**Commit 2, install, export, version.** `install(TARGETS ...)` over `pnl_core`
and `pnl_flags`, plus `pnl_cuda` under `if(TARGET pnl_cuda)`;
`install(DIRECTORY include/ ...)`; `install(EXPORT pnlTargets NAMESPACE pnl::)`
with `EXPORT_NAME core` on `pnl_core` so the installed spelling is `pnl::core`,
and `add_library(pnl::core ALIAS pnl_core)` so the build tree spells it the same.
`cmake/pnlConfig.cmake.in` through `configure_package_config_file`, finding
Threads unconditionally and OpenMP, MPI and CUDAToolkit each guarded on a
`PNL_WITH_*` variable recorded at configure time, all before including
`pnlTargets.cmake` because that file names their imported targets and CMake
validates the names as it reads it. `write_basic_package_version_file` with
`SameMajorVersion`. `cmake/` did not exist before this commit.

The version header is generated from `cmake/version.hpp.in` into
`build/generated/include/pnl/version.hpp` and reached through a
`BUILD_INTERFACE` entry on `pnl_core`, never into the source tree. The narrowed
dirty check of phase A0 watches `include`, so a generated header there would
appear untracked on every configure and stamp every measured row `.dirty`, which
is finding 4.4 and `PROV-01` all over again. It defines `PNL_VERSION_MAJOR`,
`MINOR`, `PATCH`, `PNL_VERSION_STRING` and a `constexpr pnl::VERSION` triple
whose members are named `major_version` and so on rather than `major` and
`minor`, because glibc's `<sys/sysmacros.h>` defines those two as function like
macros. The version stays 1.0.0 and the header's comment says phase E5 moves it.
`pnl --version` prints it beside the commit stamp, and `test_version` asserts
that the three integers spell the string, which is the check that fails if the
substitution ever silently stops happening. `HOMEPAGE_URL` now reads
`https://github.com/Olajide-Badejo/Parallel-Numerical-Library`, from
`git remote get-url origin`; the old value named a repository that does not
exist.

Two export faults were found here and both are in the engineering log. The
guard on `pnl_cuda` is not defensive programming: with it removed, a configure
on a machine with `nvcc` fails outright with `install(EXPORT "pnlTargets" ...)
includes target "pnl_core" which requires target "pnl_cuda" that is not in any
export set`, and a contributor without a CUDA toolkit would never see it. Then
`pnl_dev_flags` hit the same rule from the other side, because a static library
records even a `PRIVATE` dependency in its link interface under `$<LINK_ONLY:>`;
wrapping it as `$<BUILD_INTERFACE:pnl_dev_flags>` leaves the build unchanged and
the install interface empty.

**Commit 3, presets and the examples.** `CMakePresets.json` at schema version 6,
with configure, build and test presets for `dev`, `dev-debug`, `asan-ubsan`,
`tsan` and `consumer`. The compiler is left to the environment on purpose: CI
pins it per matrix entry and the Makefile pins it to the publication compiler,
so a third answer here would only be a third thing to keep in step. The `tsan`
preset sets `PNL_ENABLE_OPENMP=OFF` and `PNL_ENABLE_MPI=OFF`, which is what
makes that job actionable rather than a wall of false positives from an
uninstrumented libgomp and OpenMPI, and both sanitizer presets turn CUDA off.
The Makefile is untouched by any of this and keeps working as before.

`examples/` is its own project with `project(pnl_examples LANGUAGES CXX)`,
`find_package(pnl REQUIRED CONFIG)` and one executable linking `pnl::core`. The
`LANGUAGES CXX` and nothing else is the point of the file: it is what makes it
the install test, and Section 9.7 relies on it again in 1.2.0 when the Fortran
provider arrives with a link language of its own. `examples/poisson.cpp` solves a
127 by 127 Poisson problem with conjugate gradient on the OpenMP backend where
the install has one and the serial backend otherwise, asking
`available_backends()` rather than assuming. `examples/custom_backend.cpp` is not
stubbed; the examples README carries one line saying it arrives with the
registration hook of phase B4. `make install-test` installs to `build/stage`,
configures `examples/` against nothing but `-DCMAKE_PREFIX_PATH=build/stage`,
builds it and runs it. `.gitignore` needed nothing: `build/` and `build-*/` are
unanchored, so `git check-ignore -v` confirms they already cover
`examples/build`, `examples/build-clang`, `build/stage`, `build/examples` and
`build-consumer`.

**Gate.**

```text
$ make clean && make build && make test
100% tests passed out of 14
$ grep -m1 'main.cpp' build/compile_commands.json |
      grep -o -e '-march=native' -e '-O3' -e '-ffp-contract=off' | sort -u
-O3
-ffp-contract=off
-march=native
$ build/pnl --version
pnl 1.0.0
commit 1944289038ec
$ make install-test
install-test: staged into build/stage
include/pnl/...            35 headers, pnl/version.hpp among them
lib/cmake/pnl/pnlConfig.cmake
lib/cmake/pnl/pnlConfigVersion.cmake
lib/cmake/pnl/pnlTargets-release.cmake
lib/cmake/pnl/pnlTargets.cmake
lib/libpnl_core.a
lib/libpnl_cuda.a
-- pnl_examples: building against pnl 1.0.0
pnl 1.0.0
backend           openmp
workers           4
unknowns          16129
iterations        442
relative residual 9.704e-11
$ grep -rn 'march=native' build/stage/lib/cmake/pnl/
$ grep -rn 'Werror' build/stage/lib/cmake/pnl/
$ build/pnl --solver jacobi --backend serial --size 4 --mode fixed \
      --iterations 1 --reps 1 | awk -F, '{print $27}'
1944289038ec
$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 152 file(s) scanned
$ ruff check benchmarks scripts tests
All checks passed!
$ clang-format --dry-run --Werror over include src tests
$ git status --porcelain
```

The two greps print nothing, which is the whole of what they are asked to prove:
neither the native architecture flag nor the warnings as errors policy is in the
exported package.

The two lines carrying a commit stamp are quoted from the run made on the clean
tree at this phase's second commit, because a transcript written into the tree
it describes cannot quote the hash of the commit that contains it. What matters
in them is the absence of the `.dirty` suffix: the generated version header sits
in the build tree, so the narrowed check does not see it, and that is the whole
reason it is generated where it is. Every other line above is from the gate run
on the tree this phase's third commit contains.

The consumer preset was configured into `build-consumer` and built once,
producing `pnl` and `libpnl_core.a` and no `tests` directory; the directory was
then removed, as were the three sanitizer trees the other presets were checked
with.

**Findings.** `BUILD-02`, the one flag target that carried this machine's
architecture and a warnings as errors policy into every consumer's build, with
the opposite trap checked in the same commit. `BUILD-03`, the export set
refusing a target that links a static library outside it, which bit on the first
`install(EXPORT)` and was reproduced deliberately afterwards to confirm the
mechanism. `BUILD-04`, a `PRIVATE` dependency of a static library appearing in
the exported link interface under `$<LINK_ONLY:>`, which is how the target
created to keep `-march=native` out of the install came to block the install.
All three in `docs/ENGINEERING_LOG.md`. `CONTRIBUTING.md` gains a "Public API and
compatibility" section stating which headers are public, the ABI position for a
header mostly library, and what a consumer may rely on.

**Not done, and why.** No CI work: the matrix, the sanitizer jobs and the
`install-test` step in a workflow are phase B2's, and `.github/workflows/ci.yml`
is untouched here. No `examples/custom_backend.cpp`, which needs the registry of
phase B4. `pnl_fortran` is not in the export set because it does not exist yet;
Section 9.7's treatment of its link language belongs to part C. The driver
binary is not installed, only the library, its headers and its package files.
No number in any report changes: the phase touches no measurement, and the flag
check above is the evidence for that.

### Phase B3: enforce the numerical contract in the headers

Done, in one commit. Ground rule 8 of the version 2 specification says that a
flag the numerical contract depends on gets a test that fails when the flag is
missing, and until this phase `-ffp-contract=off` had none. Removing it from
`CMakeLists.txt` left every test green, because the equivalence suite compares
backends against each other inside a single process and all of them would have
contracted identically. Phase B1 made this bigger than one repository: the
package is exported now, and the library is about three quarters headers, so
most of the arithmetic a consumer runs is compiled on the consumer's command
line rather than into `libpnl_core.a`.

**Two mechanisms, and the specification is explicit that the second is
mandatory rather than belt and braces.** `-ffast-math` defines `__FAST_MATH__`,
so a four line `#error` at the top of `include/pnl/core/types.hpp`, above every
include, catches it. `-ffp-contract` defines nothing at all, and
`-ffp-contract=fast` is GCC's default, so the dangerous setting is the one a
consumer gets by doing nothing and no header can see it. That one is caught at
runtime by `assert_no_contraction()` in the new
`include/pnl/core/contract.hpp`, which `make_backend` calls once per process
through a function local static, so every path that runs numerics passes through
it. `ConfigurationError` is new in `error.hpp` and is deliberately not a
`BackendFailure`: nothing about the request is wrong and no execution model
failed, the compile line is wrong, and a caller that catches `BackendFailure` to
fall back to serial must not swallow this.

The probe cannot be written the obvious way, and `NUM-05` records why. Comparing
a contraction sensitive expression against the same expression written to defeat
contraction fails in exactly the case it exists to catch: both sides contract, or
GCC folds both at compile time with correct rounding, and either way they agree.
The comparison has to be between a computation the compiler is forced to emit and
a hard coded literal. With `a = b = 1 + 2^-27` and `c = -1` the two step answer is
exactly `2^-26` and the fused answer is `2^-26 + 2^-54`, which is itself exactly
representable, so the two are different doubles and `==` is a legitimate test
rather than a tolerance in disguise. `volatile` on the three operands forces the
emission. The three operands and the expected result are named `constexpr` values
in the same header because release 1.2.0 runs the same probe on the Fortran side,
as assertion 4 of Section 9.8, and a number that has to be identical in two
languages should have one definition in each and no third spelling.

What the probe does not cover is stated in the header rather than left to be
discovered. `assert_no_contraction` is inline, so a consumer's translation unit
and this library's `factory.cpp` each emit a copy and the linker keeps one of
them; at `-O3` the call inside `make_backend` is inlined into the copy
`factory.cpp` compiled, which is the copy built with the flag. A consumer who
wants their own compile line checked calls the function themselves, and that is
why it is public rather than a detail of the factory.

**Three tests, and the second is the one that carries the weight.**
`test_contract` asserts the probe is quiet under the project flags and that the
constants are what their comment claims. `test_contract_detects_fma` compiles
the same source into a second executable with `-ffp-contract=fast -O2
-march=native` and asserts the probe throws; without it the probe would pass
just as happily with an empty body. It cannot go through `add_pnl_test`, because
that links `pnl_test_main`, which links `pnl_core` publicly, which links
`pnl_flags` publicly, and `pnl_flags` is where `-ffp-contract=off` comes from. A
target that must compile with contraction on can inherit none of them, so it
compiles `tests/test_main.cpp` for itself and sets the standard by hand; nothing
is left to link, because `pnl_test.hpp`, `contract.hpp` and `error.hpp` are all
header only. `test_fast_math_rejected` runs the configured compiler over
`tests/unit/fast_math_probe.cpp` twice, once with `-ffast-math` where it must
fail and name the reason and once without where it must compile cleanly. The
second compile is what stops the test passing on a typo or a missing compiler,
which an exit status alone would not.

**The fused build does fuse, and here is the evidence.** The compile line CMake
generates for that target, from `build/compile_commands.json`, carries no
`-ffp-contract=off` at all:

```text
/usr/bin/g++-15 -DPNL_CONTRACT_EXPECT_FUSED=1 -I".../tests" -I".../include"
  -O3 -DNDEBUG -std=c++20 -fPIE -Wall -Wextra -Wpedantic
  -O2 -march=native -ffp-contract=fast -o ...test_contract.cpp.o -c .../test_contract.cpp
```

Compiling that source with `-S` under those flags and under the project's flags,
and looking at the guarded comparison in each:

```text
$ g++-15 -std=c++20 -O2 -march=native -ffp-contract=fast -DNDEBUG \
      -DPNL_CONTRACT_EXPECT_FUSED=1 -I include -I tests -S tests/unit/test_contract.cpp
	vmovsd	40(%rsp), %xmm0
	vmovsd	48(%rsp), %xmm2
	vmovsd	56(%rsp), %xmm1
	vfmadd132sd	%xmm2, %xmm1, %xmm0
	vucomisd	.LC15(%rip), %xmm0

$ g++-15 -std=c++20 -O2 -march=native -ffp-contract=off -DNDEBUG \
      -I include -I tests -S tests/unit/test_contract.cpp
	vmulsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vucomisd	.LC15(%rip), %xmm0
```

`grep -c vfmadd` returns 1 on the first and 0 on the second. The test is not a
documented skip, and the fallback the specification allows for a compiler that
refuses to fuse was not needed here.

**Gate.**

```text
$ ctest --test-dir build --output-on-failure -R 'contract|fast_math'
    Start  1: test_contract
1/3 Test  #1: test_contract ....................   Passed    0.00 sec
    Start  8: test_contract_detects_fma
2/3 Test  #8: test_contract_detects_fma ........   Passed    0.00 sec
    Start 11: test_fast_math_rejected
3/3 Test #11: test_fast_math_rejected ..........   Passed    0.22 sec

100% tests passed out of 3

$ build/tests/test_contract_detects_fma
  pass  contract/the probe constants are what the comment claims
  pass  contract/the probe throws when this translation unit contracts
2 passed, 0 failed
exit 0

$ build/tests/test_contract
  pass  contract/the probe constants are what the comment claims
  pass  contract/the probe is quiet under the project flags
2 passed, 0 failed
exit 0

$ make build && make test
100% tests passed out of 17
Label Time Summary:
convergence    =   0.67 sec*proc (1 test)
cuda           =   4.41 sec*proc (1 test)
equivalence    =   1.54 sec*proc (1 test)
mpi            =   0.79 sec*proc (3 tests)
style          =   3.77 sec*proc (3 tests)
unit           =   2.66 sec*proc (8 tests)

$ make install-test
pnl 1.0.0
backend           openmp
workers           4
unknowns          16129
iterations        442
relative residual 9.704e-11

$ git diff --exit-code 1b31675 -- tests/equivalence/
$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 156 file(s) scanned
$ ruff check benchmarks scripts tests
All checks passed!
$ clang-format --dry-run --Werror over include src tests examples
$ git status --porcelain
```

The static guard demonstrated by hand, which is the row the gate table of
Section 12 names for this phase:

```text
$ g++-15 -std=c++20 -ffast-math -fsyntax-only -I include tests/unit/fast_math_probe.cpp
In file included from tests/unit/fast_math_probe.cpp:12:
include/pnl/core/types.hpp:25:2: error: #error "pnl requires IEEE arithmetic: -ffast-math voids the bit identity guarantee"
   25 | #error "pnl requires IEEE arithmetic: -ffast-math voids the bit identity guarantee"
      |  ^~~~~
exit 1

$ g++-15 -std=c++20 -fsyntax-only -I include tests/unit/fast_math_probe.cpp
exit 0
```

`make install-test` also confirms that `include/pnl/core/contract.hpp` is in the
staged install, which it has to be: it is the header a consumer calls the probe
from.

**Findings.** `NUM-05`, the flag the bit identity claim rests on having nothing
that could notice its absence, why the obvious probe does not work, and the
literal comparison that does. In `docs/ENGINEERING_LOG.md`.

**Not done, and why.** No number in any report changes: nothing measured is
touched, and the only new work on any measured path is the call into
`verify_numerical_contract()` at the top of `make_backend`, which is a function
local static read once per process and outside every timed region. The device
side of ground rule 8 already has `--fmad=false` on `pnl_cuda` but still has no
test that fails when it is missing; that is a CUDA side probe and no phase owns
it yet. The Fortran and assembler halves of the rule are parts C and D. No CI
change: the new tests are ordinary CTest tests that the existing workflow
already runs with the rest of the suite, and the compiler matrix is phase B2's.

### Phase B4: open the extension points

Done, in three commits: the registries, the interface repairs, the preconditions
and the example. Section 4.8 of the version 2 specification is a list of five
things a consumer of this library could not do, and this phase is that list.

**Commit 1, the registries.** `make_backend` was an `if` chain over five string
literals with `available_backends()` as a list beside it, and `all_solvers()` was
twelve `push_back` calls. There was no way in from outside for either, so an
execution model or a method a third party wrote could not be reached through the
library's own entry points at all. Both are registries now,
`include/pnl/backend/registry.hpp` and the rewritten
`include/pnl/solvers/registry.hpp`, and a third party writes one namespace scope
object: `BackendRegistration` or `SolverRegistration`.

Registration order is part of the contract, because `benchmarks/run_sweep.py`,
the equivalence suite and `--list` all read it, so the built ins register in
exactly the order the fixed lists held and anything registered from outside
lands after them. `build/pnl --list` is byte identical to the output at commit
`7025e8c`; the diff is quoted in the gate below.

**Where the built in backends register, which is not a detail.** They register
inside `backend_registry()` itself, and that function is defined in the new
`src/backend/builtin_backends.cpp`. `pnl_core` is a static library, so a linker
pulls an object file out of the archive only when something references a symbol
it defines: registration objects sitting alone at namespace scope define nothing
anybody names, that member would have been dropped without a diagnostic, and
`pnl --backend serial` would have failed with "not available in this build" on a
build that contains it. Putting the accessor in the same file makes the
reference unavoidable, because `make_backend_impl` in `factory.cpp` cannot look a
name up without calling it. Registering from inside the accessor rather than from
namespace scope objects also removes the static initialisation order question:
the built ins are in the registry before it is first handed to anybody, whatever
order a consumer's own registrations run in. The trap was designed around rather
than hit, so there is no engineering log entry for it;
`tests/unit/test_registry.cpp` is what would catch it, because that test asserts
the built in names from a translation unit that registers a backend of its own,
and it could only pass if the archive member is present.

**The contraction probe moved, and NUM-05 has a dated addition saying why.** B3
called `assert_no_contraction()` from `make_backend` in `factory.cpp`. The probe
checks the flags of the translation unit it is compiled into, so it checked this
library, which is the one translation unit whose flags were never in doubt, and
checked nothing in the consumer's, which is where three quarters of the
arithmetic is compiled. Since this commit rewrote `make_backend` anyway, the
public `make_backend` is now an inline function in `backend.hpp` that calls the
probe and then a non inline `detail::make_backend_impl` in `factory.cpp`. The
function local static went with it, deliberately: a static inside an inline
function is one object for the whole program, so it would have checked whichever
translation unit built the first backend and silently exempted every other one,
which is the same hole in a smaller shape. The probe now runs on every
construction, which is three volatile stores, a multiply, an add and a compare
against a call that starts a thread pool.

`test_contract_detects_fma` gained the case that proves it, and needed a CMake
change to get it: that target compiles `tests/unit/test_contract.cpp` with
`-ffp-contract=fast` and now links `pnl_core` through `$<LINK_ONLY:pnl_core>`,
which links the archive and propagates none of its compile usage requirements.
A plain link would not do, because CMake appends an inherited interface option
after a target's own and `-ffp-contract=off` would have arrived last on the
compile line and won. The compile line is quoted in the gate.

**Commit 2, the interface repairs.** Five of them, none of which changes an
iterate.

`MPI_CHECK` and `CUDA_CHECK` are `PNL_MPI_CHECK` and `PNL_CUDA_CHECK` at every
use site. A macro is not scoped by a namespace and both names are among the most
widely defined in their ecosystems, so the unprefixed spellings in two public
headers collided with whatever a consumer included second.
`pnl_cuda_launch_coloured` moved into `pnl_cuda::detail` as `launch_coloured`; it
cannot be `static` because `jacobi_sweep.cu` calls it. The namespace is
`pnl_cuda` rather than `pnl::cuda` because those files are compiled by a
different compiler across a C ABI boundary and share no type with the host
library, and putting device code inside the host library's namespace would say
otherwise. The `extern "C"` entry points of `cuda.hpp` are untouched, as decision
8 requires.

`chunking.hpp` includes `backend.hpp`. It used `Schedule` without it, so
including that one header alone was a compile error, and nothing in the suite
could see it because every other translation unit reaches `chunking.hpp` through
a backend header that has already included `backend.hpp`.
`tests/unit/chunking_alone.cpp` includes it and nothing else and is built as an
object library, so the compile is the assertion.

`Problem::suggested_relaxation()` replaces `Sor::resolve_relaxation`'s
`dynamic_cast` to `Poisson2D`. `Poisson2D` returns the same
`theory_.optimal_relaxation` member the cast used to reach, so the factor is bit
identical rather than merely equal, and a third party's own symmetric positive
definite stencil can answer for itself instead of being handed `omega = 1`
silently. Checked against an independently computed `2 / (1 + sin(pi h))`:

```text
Young optimum at n = 63, computed in Python: 1.906454701582762

sor      omega=1.906455 iterations=25 relative_residual=1.790482e-01
         fields that differ between --omega 0 and the explicit run, timing aside: none
ssor     omega=1.000000 iterations=25 relative_residual=2.758391e-02
         fields that differ: relative_residual, omega
sor_rb   omega=1.906455 iterations=25 relative_residual=1.430338e-01
         fields that differ between --omega 0 and the explicit run, timing aside: none
```

`ssor` differing is correct and is not this phase's doing: it overrides
`relaxation_factor` with its own default of one and never asks
`resolve_relaxation`, which is MEAS-04's repair.

`Backend::reduce` stays not reentrant and the interface says so now, at the class
and at the function. Decision 23 in `docs/DESIGN_DECISIONS.md` records why the
scratch array did not simply move to the stack. Two reasons. The bound the
specification offers does not cover every reducing backend: `DETERMINISTIC_CHUNKS`
is 512, which covers openmp, pthreads, jthread and the OpenMP half of hybrid at
four kilobytes each, and does not cover the MPI backend, whose reduction scratch
is one double per rank and has no relation to the chunk grid. And for the two
hand written pools the array is not what makes the object single threaded anyway:
they publish `task_n_`, `task_chunks_`, `task_body_` and `task_reducer_` as plain
members across a `std::barrier`, so a second dispatch corrupts the task the
workers are reading whether the partials are on the stack or not, and
`parallel_for`, which has no partials array at all, is exactly as unsafe. Moving
one array would have removed the visible symptom of a property that would still
have held, and a caller who read that as permission to share a backend between
threads would have got a rarer bug rather than no bug.

`backend.hpp` no longer lists a device name among the recognised backend names,
and says plainly that the CUDA path is not a `Backend`, pointing at decision 9.

**Commit 3, the preconditions and the example.** Every public `Problem` method
now checks the length of the views it is given, on both problems: `apply`,
`jacobi_sweep`, `relaxation_sweep`, `coloured_sweep`, `block_sweep`, `residual`,
`dot`, `axpy`, `xpby`, `synchronise` and `initial_state`. `exchange_halo` and
`gather_rows` check theirs, in the shared memory default as well as in the MPI
override, because the sizes a shared memory run never uses are the same sizes
that index a raw pointer under MPI. These methods write through pointers derived
from a stride the problem chose, so a view of the wrong length is a heap write
past the end rather than a wrong answer, and until now only `initial_state`
looked, which is the one method a solver never calls.

The guard is an `if` around `require(false, message)` rather than
`require(condition, message)`, because the message names the method and both
sizes and so has to be built. Building it unconditionally would be one allocation
per sweep inside the timed region, which is what MEAS-10 took out; `require`
takes a `std::string_view` for that reason and its own comment says so. The
phase A5 no allocation test is green.

The aliasing precondition of Section 9.8 assertion 7 is in:
`jacobi_sweep(backend, x, x)` throws, and so does a partial overlap, while
`dot(backend, r, r)` stays legal and has a test of its own saying why, because
`Problem::norm` calls `dot(backend, x, x)` unconditionally and every residual
evaluation of every solver goes through it. Both facts are in the comment on
`Problem::jacobi_sweep`.

`examples/custom_backend.cpp` is the extension point used the way a stranger
would use it: about a hundred lines, of which sixty are the backend, registered
against the staged install with one namespace scope object, selected by name, and
run under conjugate gradient on a 63 by 63 Poisson problem. It prints that its
iterate is bit identical to the serial backend's, all 4225 values, compared with
`==`. `make install-test` builds and runs both examples.

**The precondition cost, as an observation and not as a gate.** One Jacobi run at
4095 squared, 20 workers, 300 iterations, 5 repetitions, before and after. The
two binaries were interleaved rather than measured half an hour apart, because a
first attempt taken sequentially had the before set come out 4 percent slower
than the after set, which is a statement about machine load and not about the
code. The before binary is commit 2 built in a detached worktree with the same
compiler and flags.

```text
before-1  median 5.328397  min 4.875999  max 5.342355  relative_residual 3.262973e-02
          reps 4.882192;5.340810;4.875999;5.328397;5.342355
after-1   median 5.101904  min 5.088228  max 5.342840  relative_residual 3.262973e-02
          reps 5.342840;5.101904;5.097102;5.088228;5.164650
before-2  median 5.074258  min 4.866945  max 5.141759  relative_residual 3.262973e-02
          reps 4.866945;5.074258;5.141759;5.113296;5.065403
after-2   median 5.076881  min 4.862339  max 5.162349  relative_residual 3.262973e-02
          reps 5.072160;5.162349;4.862339;5.101813;5.076881
before-3  median 5.165832  min 5.063344  max 5.330769  relative_residual 3.262973e-02
          reps 5.165832;5.330769;5.063344;5.121301;5.304447
after-3   median 5.404406  min 4.905396  max 5.598797  relative_residual 3.262973e-02
          reps 5.404406;5.476446;4.905396;5.311677;5.598797

before medians 5.328397 5.074258 5.165832  mean of medians 5.189496
after  medians 5.101904 5.076881 5.404406  mean of medians 5.194397
```

The two means of medians differ by 4.9 milliseconds on 5.19 seconds, which is
0.09 percent. The spread inside a single set of five repetitions is up to 9.6
percent, before-1 running from 4.876 to 5.342 seconds. The checks are therefore
not observable at this size, which is the claim; the run at 4095 squared does 300
Jacobi sweeps and 300 residual evaluations per repetition, so it pays the
precondition about 3000 times against 16.8 million unknowns each. The
`relative_residual` is the same value, `3.262973e-02`, in all six runs, which is
the other half of the observation: the preconditions changed no arithmetic.

**Gate.**

```text
$ make build && make test
100% tests passed out of 19

$ ctest --test-dir build --output-on-failure -R 'registry|preconditions'
    Start 4: test_preconditions
1/2 Test #4: test_preconditions ...............   Passed    0.16 sec
    Start 5: test_registry
2/2 Test #5: test_registry ....................   Passed    0.16 sec

100% tests passed out of 2
```

`build/pnl --list` against the order at commit `7025e8c`. The two order
determining functions, `available_backends()` and `all_solvers()`, and the
`--list` printf itself are textually unchanged between `7025e8c` and the commit
this phase started from, so the output captured before the first commit is that
commit's output:

```text
$ diff list-at-7025e8c.txt list-after-B4.txt
$ echo $?
0

$ build/pnl --list
backends: serial openmp pthreads jthread mpi hybrid
solvers:
  richardson           M = I / omega
  jacobi               M = D
  gauss_seidel_f       M = D + L
  gauss_seidel_b       M = D + U
  gauss_seidel_s       M = (D + L) D^-1 (D + U)
  gauss_seidel_rb      M = D + L in red black ordering
  sor                  M = D / omega + L
  ssor                 M = (D / omega + L) (D / omega)^-1 (D / omega + U) / (2 - omega)
  sor_rb               M = D / omega + L in red black ordering
  block_jacobi         M = block diagonal of A
  block_gauss_seidel   M = block lower triangle of A
  cg                   Krylov, not a splitting
```

```text
$ make install-test
install-test: staged into build/stage
[...]
pnl 1.0.0
backend           openmp
workers           4
unknowns          16129
iterations        442
relative residual 9.704e-11

registered backends: serial openmp pthreads jthread mpi hybrid counting
25 conjugate gradient iterations on a 63 by 63 Poisson problem:
  the registered backend's iterate is bit identical to the serial one,
  all 4225 values, compared with == and not with a tolerance.

$ git diff --exit-code 1b31675 -- tests/equivalence/
$ echo $?
0

$ grep -rn 'MPI_CHECK\|CUDA_CHECK' include src | grep -v PNL_
$ echo "expect: no output"

$ grep -rn '"cuda"' include/pnl/backend/backend.hpp
$ echo "expect: no output"

$ grep -rn dynamic_cast include/pnl/solvers/
$ echo "expect: no output"

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 162 file(s) scanned

$ git status --porcelain
```

The compile line of the fused contract target, from
`build/compile_commands.json`, which is what makes the new case in
`test_contract_detects_fma` a test of anything:

```text
/usr/bin/g++-15 -DPNL_CONTRACT_EXPECT_FUSED=1 -I".../tests" -I".../include"
  -O3 -DNDEBUG -std=c++20 -fPIE -Wall -Wextra -Wpedantic
  -O2 -march=native -ffp-contract=fast -o ...unit/test_contract.cpp.o -c .../test_contract.cpp
```

No `-ffp-contract=off` anywhere on it, which is what `$<LINK_ONLY:pnl_core>`
buys, and the case passes, so `make_backend` from that translation unit throws
`ConfigurationError`.

Two of the standing gates, which are not in the phase gate but run every time:

```text
$ ruff check benchmarks scripts tests
All checks passed!

$ find include src tests examples -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \n      -o -name '*.cuh' | xargs clang-format --dry-run --Werror
$ echo $?
0
```

The `make format` target grew `examples/` for the second of those, since the
gate formats source this repository owns and `examples/` is source this
repository owns.

The first pass of the `dynamic_cast` gate failed, on a comment. The reworded
comment in `sor.hpp` that explains what the cast used to do named the construct,
and `grep -rn dynamic_cast include/pnl/solvers/` does not read comments. It says
"downcast to Poisson2D at run time" now, which is what it means anyway.

**Findings.** `NUM-06` records the relaxation factor a foreign stencil never
received: the downcast to `Poisson2D` in `Sor::resolve_relaxation`, what a
problem the library had never heard of got instead, and the virtual on `Problem`
that replaced it. `NUM-05` gained a dated addition for the probe moving into the
caller's translation unit and for why the once per process guard was dropped
rather than kept. Decision 23 in `docs/DESIGN_DECISIONS.md` records the
reentrancy choice. No new engineering log family: nothing in this phase was
found by something breaking. The static library trap that would have earned a
`BUILD` entry was designed around before it could fire, and the test that would
have caught it is in the suite either way.

**Not done, and why.** No published number changes: nothing measured is touched
and the precondition observation above is the evidence. `Backend::reduce` is
documented as not reentrant rather than made reentrant, which is decision 23 and
is the option Section 4.8 offers. The `pnl_cuda` namespace was used instead of
`pnl::cuda` for the device helper, for the reason given above. B7 owns the
golden files that would let a later phase assert the SOR family's iterates
against a committed hex dump rather than against a run in the same process; this
phase's evidence for bit identity is that the relaxation factor is the same
object it always was, plus the unit, convergence and equivalence suites.

### Phase B5: platform guards and the second and third compilers

Done, in two commits: the guards, then the platform statement. Bullets 1 and 2
of Section 8 "B5" landed in A0.6, so what is left is the part that asks whether
this tree is a Linux and GCC program that happens to compile elsewhere, or a
C++20 program with one measured platform. It was the first, in two places.

**The flag targets assumed a compiler and said nothing about it.** `pnl_flags`
carried `-ffp-contract=off` and `pnl_dev_flags` carried `-Wall -Wextra
-Wpedantic -O3 -march=native`, both unconditionally. A toolchain that does not
take those spellings would have configured happily, and for `pnl_flags` the
consequence is not a build error but a silent one: the library's central claim
is that the same arithmetic runs everywhere, and that flag is what makes it
true. Both blocks are guarded on `CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"`
now, which covers AppleClang through the match and is right for it, since it is
a clang driver. MSVC has its own branch: `/fp:strict`, which is the spelling
that disables contraction there, and `/W4` with `/WX` only under `PNL_WERROR`.
An unknown compiler gets a configure warning naming what is missing rather than
silence.

**The MSVC branch is untested and is written to be read rather than trusted.**
There is no MSVC on this machine and none in CI. It deliberately has no
optimisation or architecture line: MSVC's Release configuration already sets
`/O2`, MSVC has no equivalent of `-march=native`, only fixed `/arch:` levels,
and a build there would be measuring the baseline architecture, which must not
be mistaken for a measured run. Two things it does not fix, both stated here so
nobody reads more into it than it says: `tests/CMakeLists.txt` still spells the
contraction probe's flags for the GNU driver, and the CUDA target still passes
`-Xcompiler=-Wall,-Wextra`. So MSVC configures; a build of the full tree would
stop at those two targets.

**The affinity code is behind one macro rather than three platform tests.**
`PNL_HAVE_AFFINITY` in `include/pnl/backend/topology.hpp` is true on Linux and
false elsewhere, and it changes exactly two answers: what `core_leader_of()`
reads, and what `pin_worker()` returns. Everything else, including the four
`Pinning` policies, `cpu_for_worker()` and every backend, is unchanged. Off the
Linux path `pin_worker()` reports `not_applicable` for every policy except
`none`, which is the tri state A4 introduced, and the distinction it makes is
the point: `refused` means the operating system was asked and said no, which is
a fault a result row must not survive, and there is nothing on macOS to ask.
`probe_topology()` returns early there with a verdict that says so, instead of
timing threads it cannot hold in place and concluding that affinity was refused
on every processor.

`factory.cpp` gained a two line repair alongside it. The refusal for a `pcore`
or `ecore` policy quotes `topology.verdict`, and the probe's verdict was thrown
away whenever probing produced no per processor timings, so the message read
"this machine did not yield one: not probed", which describes the cheap path
rather than the reason. It keeps the verdict now, and the same message on the
stub path reads `per processor classification is not applicable on this
platform: it has no thread affinity interface`.

**The stub path was compiled and run here, not merely written.**
`PNL_FORCE_NO_AFFINITY` forces it on a machine that has the interfaces. It is a
test hook, documented as one in the header, and the build never sets it.

```text
$ make build test BUILD=build-stub CXX_COMPILER=g++-15 \
      CMAKE_EXTRA="-DPNL_ENABLE_CUDA=OFF -DCMAKE_CXX_FLAGS=-DPNL_FORCE_NO_AFFINITY=1"
100% tests passed out of 18

$ build-stub/pnl --topology
logical processors: 28
physical cores:     28
verdict: not probed

$ build-stub/pnl --backend jthread --workers 4 --pinning none --solver jacobi \
      --size 63 --mode fixed --iterations 25
poisson2d_rich_63,3969,jacobi,jthread,4,1,1,none,deterministic,static,fixed,25,...,not_requested,...

$ build-stub/pnl ... --pinning compact ...
pnl: backend failure: the jthread backend was asked for 'compact' pinning and worker 0
came back 'not_applicable'; a row that says it pinned must have pinned

$ build-stub/pnl ... --pinning pcore ...
pnl: backend failure: pinning policy 'pcore' needs a performance core classification and
this machine did not yield one: per processor classification is not applicable on this
platform: it has no thread affinity interface
```

Physical cores reading 28 rather than 14 is the honest answer and not a bug:
with no sysfs to read, every processor leads a core of its own, which makes
`scatter` and `compact` the same placement instead of a wrong one. The default
path, `--pinning none`, runs and produces a row. A run that asks for a policy is
refused where the backend is constructed, with a message that says why, rather
than producing a row claiming a binding it never made; that is A4's rule and
this phase did not weaken it for a platform it cannot measure.

**Three compilers, three builds, three suites.** The two alternates are CUDA
off, because nvcc's host compiler is pinned to g++-14 either way and the host
compiler is the variable under test. `CMAKE_EXTRA` is new in the Makefile's
`configure` target so that the alternates go through the same command as the
default build rather than a hand written cmake line free to drift from it; B2
can use it for the CI matrix.

```text
$ make clean && make build && make test
-- pnl: build type Release, C++ compiler GNU 15.2.0        (g++-15 15.2.0, CUDA 13.3.73, OpenMP 4.5, MPI 3.1)
100% tests passed out of 19

$ make build test BUILD=build-gcc14 CXX_COMPILER=g++-14 CMAKE_EXTRA=-DPNL_ENABLE_CUDA=OFF
-- pnl: build type Release, C++ compiler GNU 14.3.0        (OpenMP 4.5, MPI 3.1)
100% tests passed out of 18

$ make build test BUILD=build-clang CXX_COMPILER=clang++ CMAKE_EXTRA=-DPNL_ENABLE_CUDA=OFF
-- pnl: build type Release, C++ compiler Clang 21.1.8      (no OpenMP, MPI 3.1)
100% tests passed out of 18
```

The specification asks for Clang 18. There is no clang-18 on this machine;
clang 21.1.8 stands in locally and B2 pins 18 in CI. Neither GCC 14 nor clang
raised a single warning under `-Werror`, so nothing was suppressed and there is
no suppression to review.

**What clang cannot do here, exactly.** No OpenMP runtime is installed for it,
neither `libomp-dev` nor `libomp-21-dev`, and this phase does not install
packages. `find_package(OpenMP COMPONENTS CXX)` therefore fails, CMake prints
"pnl: OpenMP not found, that backend will be skipped", and that build offers
`serial pthreads jthread mpi` where the GCC builds offer `serial openmp
pthreads jthread mpi hybrid`. So under clang the `openmp` and `hybrid` backends
are not built, not tested and not swept: the equivalence suite covers three
shared memory backends instead of four, the no allocation gate skips its openmp
case, and `test_registry` expects the shorter name list. Everything else runs.
The count of ctest cases is the same 18 either way, because the openmp coverage
sits inside binaries rather than in cases of its own.

**Findings.** Two, both surfaced by the second and third compilers and both in
`tests/unit/test_no_allocation.cpp`, which is a fair summary of what a second
compiler buys. `BUILD-05`: clang at `-O3` deleted the deliberate allocation the
counting allocator's self check depends on, which the standard permits for a new
expression whose storage never escapes, so the instrument check reported it had
no evidence. A volatile store publishes the pointer and the elision is no longer
allowed. `BUILD-06`: the same file named the `openmp` backend unconditionally,
so a configuration `CMakeLists.txt` supports on purpose was a test failure; the
case is guarded now, and where it is skipped a case of the same name requires
that the backend really is absent. Neither is a defect in the library: the phase
A5 gate and every measured number are untouched, which is why they are `BUILD`
entries.

**Not done, and why.** No macOS build. There is no macOS here, so what can
honestly be claimed is that the stub path compiles, runs and reports
`not_applicable`, which is what the forced build above shows, and the README
says macOS is not measured. No MSVC build, for the same reason and stated in the
same place. The `tests/` and CUDA flag blocks were left in GNU spellings rather
than widened, because guarding them properly means deciding what
`-march=native -ffp-contract=fast` means to a compiler whose FMA behaviour
nobody here can observe, and a guess in a test that exists to detect contraction
is worse than an honest gap. Section 4.9's other half, the C++20 declaration and
`target_compile_features`, was A0.6's and is already in the tree.

### Phase B6: the correctness defects of Section 4.7

Done, in twelve commits. Sixteen rows, of which eleven were fixed here, three
had already been closed by earlier phases and were verified rather than redone,
one was closed here as the last of its kind, and one is the sanitizer
scaffolding the rest are gated by.

**The sanitizer baseline, before any fix.** Both builds were green. That is the
finding the rest of the phase rests on, and it is worth stating plainly: the
suite as it stood could not reach a single one of these defects, so a green
sanitizer run proved nothing about them.

```text
$ cmake --preset asan-ubsan && cmake --build --preset asan-ubsan -j 6
$ ctest --preset asan-ubsan -L 'unit|convergence|equivalence'
100% tests passed out of 12

$ cmake --preset tsan && cmake --build --preset tsan -j 6
$ ctest --preset tsan -L equivalence
100% tests passed out of 1
```

Section 8 "B2" predicted the second half of that for one specific row and was
right: the `pinning_failures_` race is between an increment in the constructor,
before any thread exists, and a read from a function nothing calls, so
ThreadSanitizer was never going to observe it. It was fixed in phase A4 because
it was wrong, not because a sanitizer would catch it.

**The commits, in the order they were made.**

| Commit | Row |
| --- | --- |
| `9d4d069` | `PNL_SANITIZE`, so both presets are one cache variable rather than two flag strings |
| `10e0f9e` | A worker body exception: `std::terminate` on three thread boundaries, and the jthread deadlock. CONC-03 |
| `7f2ea72` | One rank's failure hangs the job instead of failing it, in the driver and in `test_mpi`. MPI-03 |
| `4b1c792` | `n * n` in the member initialiser: an allocation before the check, and signed overflow. NUM-07 |
| `fb1c8c4` | `thomas_solve` at `n == 0` writes past the end of three objects. NUM-08 |
| `0baef81` | `fa * fb <= 0.0` accepts a bracket whose product underflows. NUM-09 |
| `a956247` | `parse()` outside the `try`, and `--reps 0` indexing an empty vector. CLI-01 |
| `2024e7f` | The shared topology reassigned under the references it handed out. CONC-04 |
| `57372c4` | Default arguments on a virtual whose override omitted them. BUILD-07 |
| `47bc9a7` | The device grid index wraps, and a refused launch read like a fault. CUDA-04 |
| `98e5592` | Dormand Prince calls a run at the step floor converged. NUM-10 |
| `9b04bbd` | The last unprefixed macro in the tree |

**Reproduced before fixed, every one.** Section 4's opening rule, and it earned
its keep twice. The jthread deadlock is only visible when the throw is on the
dispatching thread and the terminate only when it is not, so a fix aimed at one
would have left the other; and `--iterations 12abc` turned out to be accepted as
twelve and to print a full result row, which is not in Section 4.7 and was found
only because the reproduction ran the binary rather than reasoning about it. The
reproductions are quoted in the engineering log entries above.

**Three rows were already closed, and were verified rather than redone.**

- `pinning_failures_` is `std::atomic<int>` in `pthreads.hpp`, incremented with
  `fetch_add` in both places and read with `load`. Phase A4, CONC-02. In the
  jthread and OpenMP pools the same counter is a plain `int` written only by the
  constructing thread after the latch or the parallel region has closed, which
  is correct and needs no atomic.
- `chunking.hpp` includes `backend.hpp`, and `tests/unit/chunking_alone.cpp` is
  an object library that includes that one header and nothing else, so the
  compile is the assertion. Phase B4.
- `PNL_MPI_CHECK` and `PNL_CUDA_CHECK` carry their prefixes, and
  `pnl_cuda_launch_coloured` is `pnl_cuda::detail::launch_coloured`. Phase B4.
  What was left was `CUDA_OR_FAIL`, function local and undefined again a few
  lines later, which could not have collided with anything but made the rule one
  a reader has to check. It is prefixed now and a grep for an unprefixed macro
  in this tree returns nothing.

**Two things this phase added that the specification did not ask for by name.**
`StopReason::StepFloor`, because reporting the step floor as an iteration cap
would have sent a reader to the wrong place, and `PNL_CUDA_MAX_SIDE`, because
the device path had a bound it never stated and an unstated bound is one nobody
can check. Both are additive: the enumerator is appended and nothing switches
over `StopReason` but its own `to_string`, and the limit is 46338, which is a
17 GB array per side and therefore beyond anything a current device could have
run anyway.

**One thing deliberately not done.** The device kernels were not widened to 64
bit indices. The reduction kernel divides and takes a remainder per element, a
64 bit integer division on a GPU costs several times a 32 bit one, and that
kernel's time is inside the published `kernel_seconds`. Widening a measured path
to reach sizes no device can hold is the wrong trade; the bound is checked once
instead, at the entry point and in the driver. CUDA-04 records the reasoning.

**The gate.**

```text
$ make clean && make build && make test
-- pnl: results will be stamped with commit 9b04bbdd963c
-- pnl: build type Release, C++ compiler GNU 15.2.0
100% tests passed out of 26

$ cmake --preset asan-ubsan && cmake --build --preset asan-ubsan -j 6
-- pnl: instrumented with -fsanitize=address,undefined
$ ctest --preset asan-ubsan --output-on-failure -L 'unit|convergence|equivalence'
100% tests passed out of 18

$ cmake --preset tsan && cmake --build --preset tsan -j 6
-- pnl: instrumented with -fsanitize=thread
$ ctest --preset tsan --output-on-failure -L equivalence
100% tests passed out of 1

$ ctest --test-dir build --output-on-failure -R 'throwing|parse|mpi_failure|thomas|bracket|dormand|default_args'
100% tests passed out of 7

$ PNL_TEST_FAIL_RANK=1 mpirun -np 2 --oversubscribe build/tests/test_mpi
running at 2 rank(s)
  pass  mpi/the row decomposition covers the grid exactly once
rank 1 failed 'mpi/order free solvers agree with serial on the Poisson problem'
and the other ranks did not reach the end of that case within 5 seconds, so they
are waiting in a collective this rank has left. Aborting the job.
exit 1 within 5 seconds, not 900

$ ruff check benchmarks scripts tests
All checks passed!

$ clang-format --dry-run --Werror over include src tests examples
clean

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 211 file(s) scanned

$ make install-test
registered backends: serial openmp pthreads jthread mpi hybrid counting
25 conjugate gradient iterations on a 63 by 63 Poisson problem:
  the registered backend's iterate is bit identical to the serial one,
  all 4225 values, compared with == and not with a tolerance.

$ git status --porcelain
(nothing)
```

Zero sanitizer reports either side. The 12 of the baseline and the 18 here are
the same suite plus the six cases this phase added, and the equivalence label
that the ASan preset now also runs.

**The ThreadSanitizer worker counts.** `worker_counts()` in
`tests/equivalence/test_equivalence.cpp` is `{1, 2, 3}` plus each of `{4, 7, 8,
16}` that this machine has processors for, and `nproc` is 28, so 2, 4 and 8 are
all swept and the file did not have to be touched: `git diff --exit-code
1b31675 -- tests/equivalence/` is still empty. Under the TSan preset OpenMP and
MPI are off, so what is instrumented is exactly the two hand written pools,
which is what makes the report list actionable. The relay this phase added to
those pools was run under TSan directly as well, since it dispatches at 1, 2 and
8 workers and throws from both a dispatching and a spawned worker:

```text
$ build-tsan/tests/test_throwing_body
  pass  throwing_body/a body that throws surfaces on the dispatching thread
  pass  throwing_body/a reducer that throws surfaces on the dispatching thread
        pools covered: pthreads jthread
  pass  throwing_body/every pool this build has is covered
3 passed, 0 failed
```

**No published number moves.** The three device bit identity cases still assert
equality against the host, the equivalence suite is untouched and green, and the
one hot path this phase went near, the per chunk dispatch loop, gained a `try`
block whose landing pad costs nothing when nothing throws. The measured
quantities that could have moved, the device `kernel_seconds` in particular, are
the ones the 64 bit widening was rejected to protect.

### Phase B7: strengthen the test suite

Done, in seven commits. Property tests, golden files, boundary sizes, hybrid and
pinning coverage, a device contraction probe, three fuzz targets and a
performance gate, built from nothing. Four defects came out of it, three of them
in the library and one in the test framework itself.

**The commits, in the order they were made.**

| Commit | What |
| --- | --- |
| `ccfbeed` | The three defects in the test framework, and a self test of it. TEST-01 |
| `47baeee` | Property based coverage of `block_partition`, `for_chunk`, `reduction_chunk` and the padded row band map |
| `d65c801` | Twelve golden iterates, one per solver, diffed bit for bit |
| `47d7188` | Boundary sizes in the equivalence suite, and the rich right hand side at both construction sites. NUM-11 |
| `2bacc78` | Hybrid and pinning coverage, which had none. MEAS-12 |
| `cf9ffac` | The device contraction probe and its deliberately fused twin. NUM-05 addition |
| `69c576a` | Three fuzz targets and the relative performance gate. CLI-02, NUM-12 |

**What each new test covers.**

- **`tests/unit/test_framework.cpp`**, ten cases, registered first. `run_all`
  caught `Failure` and `std::exception` and nothing else, so a case that threw
  an `int` ended the process in `std::terminate` and every case after it in that
  binary went unattempted. The two vector comparisons had been written twice,
  once per suite, and neither copy treated a length mismatch as a difference:
  `first_difference` returned 0, which is the index of the first element, and
  both callers then formatted that element of a vector that may be empty;
  `worst_difference` compared the common prefix, so a vector and a truncation of
  itself were bit identical and three assertions in the distributed suite passed
  on one. All three are TEST-01.
- **`tests/unit/test_chunk_property.cpp`**, seven cases. Thousands of random
  `(n, parts, k)` draws from a printed seed over the three chunk grids,
  asserting exact coverage of `[0, n)`, pairwise disjointness and a size spread
  of at most one. Coverage is counted per index rather than summed, because a
  partition that drops one index and claims another twice sums correctly.
  `n` in `{0, 1, 511, 512, 513}` appears explicitly in every case. The padded row
  band map `[rows.begin + chunk.begin + 1, rows.begin + chunk.end]` is swept over
  ranks, workers, schedules and chunks per worker, which is Section 9.8 assertion
  11 and is what release 1.2.0 reuses. A negative control hands the harness four
  wrong partitions and one overlapping band map and requires it to refuse all
  five.
- **`tests/golden/*.hex` and `tests/unit/test_golden.cpp`**, twelve files, one
  per solver, from `Poisson2D(31, SpectrallyRich, 20260802)` over 25 fixed
  sweeps on the serial backend with the deterministic reduction. One `%a`
  formatted double per line, so the file holds the exact value and a diff shows
  which bit moved; each header line records the solver, the configuration and
  the commit the file was generated at, `47baeee2fb60`, and the test checks all
  three before it compares a number. This is the only reference in the tree that
  does not move with the code: cross backend equivalence compares every backend
  against a live serial run in the same process, so a change that breaks every
  backend identically passes it in silence. `pnl_write_golden` regenerates them
  and shares `golden_config.hpp` with the test.
- **Four boundary cases in the equivalence suite**, with a CTest entry that runs
  exactly those. The deterministic reduction across backends at `n` in
  `{510, 511, 512, 513, 514}`, which is where `reduction_chunk_count` stops
  being `n`; dense solves of order 511, 512 and 513 for the reducing solvers,
  a dense system being the one whose unknown count is the reduce length; zero
  unknowns refused by both problem constructors before any backend exists, with
  an empty range still dispatching no chunks on every backend; and one unknown
  bit identical on every backend and worker count, which is where seven of eight
  chunks are empty. Both Poisson problems in that file now ask for
  `PoissonRhs::SpectrallyRich` explicitly, as Section 9.8 requires.
- **Five hybrid cases in the distributed suite.** `worker_count()` is ranks
  times threads and agrees with the config the result row reads, which is A6 and
  MEAS-09 asserted rather than assumed; solves through `hybrid` at two threads
  per rank agree with serial to reduction tolerance and the ordered solvers
  agree bit for bit; the hybrid reduction is bit identical to the pure
  distributed one at the same rank count, which is the control that makes any
  measured difference between them the threading; and a body that throws inside
  the rank's OpenMP team surfaces on the dispatching thread with its message
  intact.
- **Seven pinning cases** on all three pools that bind. `compact` and `scatter`
  at 2 and 4 workers report exactly `bound`; `none` reports `not_requested`; a
  pinned run computes the same iterate as an unpinned one; and `pcore` is
  refused with a message naming the policy and quoting the topology verdict,
  guarded on the classification actually having failed so the file is right on a
  machine where it succeeds.
- **Three CUDA cases.** `pnl_cuda` is compiled with `--fmad=false`, which is why
  a device sweep may be asserted bit identical to the host sweep, and nothing
  could tell whether the flag was there. A one thread device kernel evaluates
  `a * b + c` from B3's constants and must return exactly `2^-26`; the same `.cu`
  file compiled with `--fmad=true` into `pnl_cuda_fused`, a target only
  `test_cuda` links, must return exactly `2^-26 + 2^-54`; and the third case
  asserts those are different doubles. The operands are arguments rather than
  constants in the `.cu` file, because literals would be folded at compile time
  with correct rounding and the probe would report the unfused answer whatever
  the flag said.
- **Three fuzz targets and two drivers.** `parse()`, `thomas_solve` and the
  bracketing root finders, each an `int fuzz_one(const uint8_t*, size_t)`. The
  dependency free driver is a CTest entry under `unit` that draws 20000 inputs
  per target from a fixed seed and prints the failing input as hexadecimal; the
  libFuzzer entry point is built only when `PNL_FUZZ=ON` and the compiler is
  clang, and is deliberately not a requirement of the default build. The
  property is the same for all three: no crash, no sanitizer report, and either
  a result or an exception derived from `pnl::Error`.
- **The performance gate**, label `perf`, excluded from `make test` with
  `-LE perf` and run by `make test-perf` and a CI step of its own.

**What the new tests found.**

| Identifier | Found by | What |
| --- | --- | --- |
| `TEST-01` | Reading `pnl_test.hpp` while writing the self test | `run_all` had no `catch (...)`; `first_difference` returned index 0 on a length mismatch and both callers then indexed an empty vector; `worst_difference` compared the common prefix, so a vector and a truncation of itself were bit identical |
| `NUM-11` | The one unknown boundary case | Conjugate gradient reaches an exactly zero residual after one iteration on a one unknown system, the recurrence then makes the search direction the zero vector, and the zero vector has zero curvature under every operator there is. The breakdown check read that as a proof that the operator is not positive definite and threw. Reachable only under a fixed iteration run shorter than the unknown count, which is why nothing had hit it |
| `MEAS-12` | The pinning cases | Every pool binds the calling thread as worker zero, which is right, and none of them ever handed the mask back. `probe_topology` reads the calling thread's mask for the processor count and spawns its probe threads from that thread, so after any pinned backend had existed the core classification saw one processor and refused `pcore` for a reason that had nothing to do with the machine |
| `CLI-02` | Writing the `parse()` fuzz target | Two paths out of `parse()` left the process by `std::exit` rather than throwing, bypassing the handler CLI-01 put there and making those paths unreachable from an in process test |
| `NUM-12` | The bracket fuzz target, draw 23 | `brent` reported an exact root with the error estimate of the bracket it happened to be found in, up to 1e199, where its own two early returns and bisection's exact root exit all report zero |

**The fuzzer's first finding was a defect in my own property, and it is recorded
because that is the more common outcome.** The bracket harness first asserted
that a converged run must have a small residual, and forty draws refuted it with
a tolerance of about 3.5e101: Brent reported convergence with an error estimate
of 1.8e101, which is correct, because `RootOptions::tolerance` bounds the
**bracket half width** and not the residual. The harness now draws a sane
tolerance on most inputs and asserts the residual only there, and the comment in
that file says so, so nobody tightens it back.

**No published number moves.** `NUM-11` is reachable only when the residual is
exactly zero and every path that reached it previously threw; `NUM-12` changes
one diagnostic field on an exit that requires an exact root; `MEAS-12` restores a
mask after a backend is destroyed, which is after every timed region that backend
was built for, and the sweep driver runs one configuration per process; `CLI-02`
changes the route a refusal takes and not the status or the sentence. The twelve
golden files were generated before the `NUM-11` fix and are reproduced bit for
bit after it, and again under the sanitizer preset, which is a Debug build
without `-march=native`.

**The perf gate ratio on this machine.** Three consecutive runs on an idle
machine, after the suite had finished:

```text
$ build/tests/test_perf
        jacobi 1023 squared, 200 iterations, median of 5:
          1 worker  0.1605 s
          4 workers 0.0283 s
          ratio     5.66, required 2.50
        jacobi 1023 squared, 200 iterations, median of 5:
          1 worker  0.1750 s
          4 workers 0.0297 s
          ratio     5.90, required 2.50
        jacobi 1023 squared, 200 iterations, median of 5:
          1 worker  0.1500 s
          4 workers 0.0279 s
          ratio     5.37, required 2.50
```

A run taken while the rest of the suite was still finishing gave 4.69, which is
the reason `make test-perf` runs the label on its own.

**The ratio is above the ideal 4, and the reason is stated rather than
celebrated.** At 1023 squared the three arrays a Jacobi sweep touches are about
25 MB, which is inside the 33 MB last level cache of this processor, so four
threads get more cache per thread than one does and the speedup is superlinear.
That is a property of this machine and this size, not a scaling result, and it
is exactly why the gate is written as a floor of 2.5 rather than as a
measurement: what it catches is a collapse to 1, which is the class of
regression finding 4.1 describes and which every correctness gate in the suite
passes happily because the answers stay bit identical.

**Under clang with the fuzzer, the address sanitizer and the undefined behaviour
sanitizer**, which is not part of the gate and was run once to confirm that the
`PNL_FUZZ=ON` path in `tests/CMakeLists.txt` is real:

```text
Ubuntu clang version 21.1.8 (6ubuntu1)
== cli ==       Done 681004 runs in 11 second(s)
== thomas ==    Done 552016 runs in 11 second(s)
== brackets ==  Done 418066 runs in 11 second(s)
```

1.65 million inputs, no crash and no sanitizer report. The build tree was
removed afterwards.

**The gate.**

```text
$ make build && make test
100% tests passed out of 32

Label Time Summary:
convergence    =   0.64 sec*proc (1 test)
cuda           =   4.70 sec*proc (1 test)
equivalence    =   7.87 sec*proc (2 tests)
mpi            =   6.08 sec*proc (4 tests)
style          =   6.64 sec*proc (3 tests)
unit           =   8.33 sec*proc (21 tests)

Total Test time (real) =  17.42 sec

$ ctest --test-dir build --output-on-failure -L perf
1/1 Test #18: test_perf ........................   Passed    1.58 sec
100% tests passed out of 1

$ ctest --test-dir build --output-on-failure -R 'golden|property|fuzz|framework|boundary'
1/5 Test  #1: test_framework ...................   Passed    0.00 sec
2/5 Test  #2: test_chunk_property ..............   Passed    0.08 sec
3/5 Test  #5: test_golden ......................   Passed    0.26 sec
4/5 Test #17: test_fuzz ........................   Passed    0.42 sec
5/5 Test #21: test_equivalence_boundary ........   Passed   10.21 sec
100% tests passed out of 5

$ ls tests/golden/ | wc -l
12

$ cmake --preset asan-ubsan && cmake --build --preset asan-ubsan -j 6
$ ctest --preset asan-ubsan --output-on-failure -L 'unit|convergence|equivalence'
100% tests passed out of 24

Label Time Summary:
convergence    =  13.63 sec*proc (1 test)
equivalence    =  47.57 sec*proc (2 tests)
unit           =  20.18 sec*proc (21 tests)

Total Test time (real) =  81.66 sec

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 191 file(s) scanned

$ ruff check benchmarks scripts tests
All checks passed!

$ clang-format --dry-run --Werror over include src tests examples
exit 0

$ git status --porcelain
```

`make install-test` is green as well, and it is the gate that would have caught
`pnl_cuda_fused` leaking into the export set: it does not, because nothing
measured, installed or exported links that object.

This phase took the suite from 26 CTest entries to 33, of which 32 run by
default and one carries the `perf` label. Seven of the eight new source files
are tests; the eighth, `tests/unit/write_golden.cpp`, is the regenerator behind
the golden files and is built but never run by CTest.

### Phase B2: make CI prove what the badge claims

Done, in three commits. The workflow went from one job on one compiler in one
configuration to eleven jobs: a style gate, a matrix of three compilers by two
configurations, two sanitizer jobs, an optionality job, a CUDA job that can now
fail, and a reports job that compares what it built against what the repository
ships.

**The commits.**

| Commit | What |
| --- | --- |
| `1e32463` | The matrix and the sanitizers, MPI at four ranks, the CUDA job made honest, the style gate pinned. CI-01, CI-03 |
| `1e8dc63` | `scripts/compare_report_text.py`, its unit test, and the reports job step that runs it. CI-02 |
| `685f522` | ccache, the apt archive cache, every action pinned by commit SHA, `.github/dependabot.yml` |

**The jobs, and what each one is there to prove.**

| Job | What it proves |
| --- | --- |
| style gates | clang-format at 20.1.7, asserted rather than assumed, over `include src tests examples`; ruff at 0.15.22; the dash rule; the dash checker's own self test |
| `{gcc-14, gcc-15, clang-18}` by `{Debug, Release}` | zero warnings with `-DPNL_WERROR=ON`; the unit, convergence, equivalence and style labels; all six backends present in the build; MPI agreement at 1, 2 and 4 ranks and the rank failure mode. The gcc-15 Release leg additionally runs the `perf` label and `make install-test` |
| address and undefined behaviour sanitizers | the `asan-ubsan` preset over unit, convergence and equivalence |
| thread sanitizer | the `tsan` preset over equivalence at 1, 2, 4 and 8 workers, with OpenMP and MPI off, and an assertion that they are off |
| OpenMP and MPI switched off | the minimal build compiles, `pnl --list` shows serial, pthreads and jthread and nothing else, and unit and equivalence pass |
| cuda compiles | the toolkit installs or the job fails, the device code compiles at sm_70, and `test_cuda` runs and skips its device cases with a printed reason |
| reports | assets regenerate from the committed summary, both PDFs build, the dash rule covers them, and their text is compared against the tracked copies |

**Three things make a skip impossible to mistake for a pass**, which is the rule
this phase was written under.

- Every `ctest` invocation carries `--no-tests=error`, and the four test presets
  in `CMakePresets.json` carry `noTestsAction: error`. A label filter that
  matches nothing used to report success over an empty set, which is CI-01 in a
  second place.
- Every build asserts the backend list afterwards. `find_package(OpenMP)` and
  `find_package(MPI)` print a status line and carry on when they fail, so a
  missing runtime silently drops a backend and the equivalence suite then
  compares what is left. The clang-18 legs install `libomp-18-dev` for exactly
  this reason, and the minimal job asserts the inverse: that the two backends it
  switched off are gone.
- The compiler is asserted by name and by `-dumpversion` against the major
  version the leg claims, so a leg cannot silently run on a different compiler.

**The equivalence suite gained one environment variable, and it is the only
source change in this phase.** `worker_counts()` in
`tests/equivalence/test_equivalence.cpp` clamps its sweep to the processors the
process may run on. A GitHub hosted runner has four, so 8 and 16 would never be
reached there and the thread sanitizer job would have covered half of what
Section 8 asks of it without saying so anywhere. `PNL_TEST_WORKERS` overrides
the list, the TSan job sets it to `1,2,4,8`, and the suite now prints the counts
it swept and where they came from on every run, green or red. Unset, empty, or
holding nothing that parses, it changes nothing, so every local run and every
other job sweeps the machine's own list. Asking eight software workers to share
four processors is a legitimate thing to require of a pool and is if anything a
harder test of one.

**The action SHA table.** Resolved with
`gh api repos/<owner>/<repo>/git/ref/tags/<tag>` from the Windows side. All
three tags point at a commit object directly, so no annotated tag dereference
was needed; the object type was checked in each case rather than assumed.

| Action | Tag | Commit SHA |
| --- | --- | --- |
| `actions/checkout` | `v7.0.1` | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `actions/cache` | `v6.1.0` | `55cc8345863c7cc4c66a329aec7e433d2d1c52a9` |
| `actions/upload-artifact` | `v7.0.1` | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` |

All three were on `@v4` before, and all three major lines have moved since that
was written, which is the argument for `dependabot.yml` rather than against
pinning: an unmaintained pin does not stay safe, it only fails later. The inputs
this workflow uses were checked against each action's `action.yml` at the tag
being pinned, because a major bump is exactly where an input disappears.

**What was verified locally, and what awaits the first push.** Nothing is pushed
until the release and GitHub Actions cannot be run from this machine, so the
rule was that every command in the workflow is run here through the same
presets, Makefile targets and scripts the job calls, and the workflow file
itself is validated for syntax.

Verified locally, by running it:

- the `asan-ubsan` preset, configure, build and the unit, convergence and
  equivalence labels;
- the `tsan` preset, configure, build, the backend list assertion the job makes,
  and the equivalence label both at the machine's own worker sweep and at the
  `1,2,4,8` the job forces;
- `ctest -R 'test_mpi_(1|2|4)rank'`;
- `make install-test`, which is what the gcc-15 Release leg runs;
- `make test`, which is the unit, convergence, equivalence, style, mpi and cuda
  labels including the new comparison test, and `make test-perf`, the `perf`
  label that leg also runs;
- the asset comparison against both tracked PDFs, and the deliberate divergence
  that proves it bites;
- `clang-format --dry-run --Werror` at 20.1.7, `ruff check`, the dash rule and
  the dash checker self test;
- the YAML of both new files parses under PyYAML.

Awaiting the first push, and listed rather than assumed:

- the matrix itself: three compilers by two configurations on `ubuntu-24.04`,
  which is not this machine;
- the `ubuntu-toolchain-r/test` PPA carrying `g++-15` for that image, and
  `clang-18` with `libomp-18-dev` finding OpenMP there;
- the CUDA job against the archive's `nvidia-cuda-toolkit` with `g++-12` as both
  the host compiler and the C++ compiler, at `sm_70`, with warnings as errors on
  a compiler two majors below the publication one;
- the virtual environment step, the pinned `pip install` of cmake, ruff and
  clang-format, and the TeX Live install;
- whether the `perf` ratio clears 2.5 on a four processor shared runner;
- the ccache and apt caches actually hitting, which is why every build leg
  prints `ccache --show-stats`;
- dependabot opening its first pull request.

**actionlint was not run, and it is not installed here.** This phase forbids
installing anything on this machine, so the mandatory check is the YAML parse,
which passes for both new files. `yamllint`, which is already installed, was run
as an extra and reports exactly one complaint on the workflow, a `hashFiles`
expression 128 characters long that cannot be wrapped without changing the cache
key, and nothing else. GitHub imposes no line length.

**The findings.**

| Identifier | What |
| --- | --- |
| `CI-01` | The CUDA job wrapped the toolkit install in a conditional and skipped both following steps when it failed, so the job reported success having compiled zero CUDA. A skipped step is not a failed step, and the two outcomes looked identical in the badge |
| `CI-02` | The reports job regenerated every asset from the committed summary, checked that the files were not empty, and never compared them against the tracked PDFs the README links to. That is the hole finding 4.3 fell through, and CI stayed green over four irreconcilable bandwidth figures for the life of release 1.0.0 |
| `CI-03` | `runs-on: ubuntu-latest`, a compiler chosen by taking the first of `g++-16 g++-15 g++-14 g++-13 g++` that existed, and every action on a mutable `@v4` tag. None of the three had yet caused a failure, which is exactly why it is recorded: a green run said the code built with some compiler on some image using whatever those actions contained that morning |

**The report comparison is red today, and that is the expected answer.** Part A
changed the tables the report is generated from and phase B5 changed the
declared standard, so the rebuilt main report differs from the tracked one at
the second numeric token, which is the C++ version in the abstract. Phase A8b
rebuilds the published PDFs from a clean generation and is what makes this
green. The debug report already agrees, at 246 numeric tokens over 17 pages,
which is the half of the demonstration that shows the script says so when it is
true.

**Proving the comparison bites.** The freshly built main report was copied aside
as a stand in for the tracked copy phase A8b will produce, one number in a
regenerated table was moved from 61.4 to 64.3, which is the exact shape of
finding 4.3, and the document was rebuilt:

```text
$ python3 scripts/compare_report_text.py report/main.pdf /tmp/pnl_b2/reference_main.pdf
compare_report_text: 1150 numeric tokens across 41 pages agree between report/main.pdf and /tmp/pnl_b2/reference_main.pdf
  volatile lines dropped: none
exit 0

$ sed -i 's/& 61\.4 & best of/\& 64.3 \& best of/' report/tables/bandwidth.tex
$ make report-only
$ python3 scripts/compare_report_text.py report/main.pdf /tmp/pnl_b2/reference_main.pdf
compare_report_text: report/main.pdf does not agree with /tmp/pnl_b2/reference_main.pdf
  numeric token 425 of the rebuilt report reads 64.3 where the tracked report reads 61.4
    rebuilt  page 22 line 24: host, all threads, plain stores                   64.3   best of workers 2:42.6 4:61.4 8:58.3 12:59.5
    tracked  page 22 line 24: host, all threads, plain stores                   61.4   best of workers 2:42.6 4:61.4 8:58.3 12:59.5
exit 1
```

The table was put back and the document returned to 1150 agreeing tokens.

**The gate.**

```text
$ python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('workflow parses')"
workflow parses
$ python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml')); print('dependabot parses')"
dependabot parses

$ cmake --preset asan-ubsan && cmake --build --preset asan-ubsan -j 6
$ ctest --preset asan-ubsan --output-on-failure -L 'unit|convergence|equivalence'
100% tests passed out of 25

Label Time Summary:
convergence    =  10.03 sec*proc (1 test)
equivalence    =  33.80 sec*proc (2 tests)
unit           =  15.13 sec*proc (22 tests)

Total Test time (real) =  59.11 sec

$ cmake --preset tsan && cmake --build --preset tsan -j 6
$ ./build-tsan/pnl --list
backends: serial pthreads jthread
$ ctest --preset tsan --output-on-failure -L equivalence
1/2 Test #20: test_equivalence .................   Passed   50.41 sec
2/2 Test #21: test_equivalence_boundary ........   Passed   43.55 sec
100% tests passed out of 2

$ PNL_TEST_WORKERS=1,2,4,8 ctest --preset tsan --no-tests=error -V
20:         worker counts swept, from PNL_TEST_WORKERS: 1 2 4 8
1/2 Test #20: test_equivalence .................   Passed   26.25 sec
21:         worker counts swept, from PNL_TEST_WORKERS: 1 2 4 8
2/2 Test #21: test_equivalence_boundary ........   Passed   22.43 sec
100% tests passed out of 2

$ ctest --test-dir build --output-on-failure --no-tests=error -R 'test_mpi_(1|2|4)rank'
3/3 Test #32: test_mpi_4rank ...................   Passed    0.29 sec
100% tests passed out of 3

$ make install-test
registered backends: serial openmp pthreads jthread mpi hybrid counting
25 conjugate gradient iterations on a 63 by 63 Poisson problem:
  the registered backend's iterate is bit identical to the serial one,
  all 4225 values, compared with == and not with a tolerance.
exit 0

$ python3 scripts/compare_report_text.py report/main.pdf assets/reports/main_report.pdf
compare_report_text: report/main.pdf does not agree with assets/reports/main_report.pdf
  numeric token 2 of the rebuilt report reads 20 where the tracked report reads 23
    rebuilt  page 2 line 3: This report describes a C++20 numerical library in which twelve iterative solvers are written
    tracked  page 2 line 3: This report describes a C++23 numerical library in which twelve iterative solvers are written
exit 1 (expected today, phase A8b makes it zero)

$ python3 scripts/compare_report_text.py report_debug/debug_report.pdf assets/reports/debug_report.pdf
compare_report_text: 246 numeric tokens across 17 pages agree between report_debug/debug_report.pdf and assets/reports/debug_report.pdf
  volatile lines dropped: none
exit 0

$ make build && make test
100% tests passed out of 33

Label Time Summary:
convergence    =   0.71 sec*proc (1 test)
cuda           =   4.07 sec*proc (1 test)
equivalence    =   6.62 sec*proc (2 tests)
mpi            =   6.04 sec*proc (4 tests)
style          =   7.35 sec*proc (3 tests)
unit           =   7.53 sec*proc (22 tests)

Total Test time (real) =  16.33 sec

$ make test-perf
1/1 Test #18: test_perf ........................   Passed    1.09 sec
100% tests passed out of 1

$ build/tests/test_perf
        jacobi 1023 squared, 200 iterations, median of 5:
          1 worker  0.1105 s
          4 workers 0.0285 s
          ratio     3.87, required 2.50

$ python3 scripts/check_no_dashes.py .
check_no_dashes: clean, 265 file(s) scanned

$ ruff check benchmarks scripts tests
All checks passed!

$ clang-format --dry-run --Werror over include src tests examples
exit 0

$ git status --porcelain
```

The `cuda` label in `make test` is what dates the README sentence: the last
local run of `test_cuda` with a device present was 2026-09-06, today, and it
passed. The `perf` ratio of 3.87 is inside the 5.4 to 5.9 band phase B7 recorded
only in the sense that both are far above the 2.5 floor; this run was taken
immediately after a sanitizer build had finished with the machine still warm,
which is exactly the effect that made B7 give the label a target of its own.

**What this phase did not do.** No performance number moved, no measured row was
touched, and the only file outside `.github/`, `scripts/`, `tests/` and the two
record documents that changed is `README.md`, which gained a paragraph saying
what continuous integration proves and the date of the last local `test_cuda`
run. The `perf` threshold is exactly what phase B7 set it to. `--allow-dirty` is
still on the asset regeneration and comes off at phase A8b, with phase E5's
checklist as the place that removes it.

### Phase E4: report and README repairs, before the publish pass

Done, in seven commits. No sweep was run, no tracked PDF was rebuilt and nothing
under `assets/` was committed: the generator rewrote `assets/figures/*.png`
during the gate and they were restored with `git checkout -- assets/` before
each commit. Both PDFs were built locally from the committed generation with
`--allow-dirty`, to prove the LaTeX compiles.

**Commit 1, every table generated.** The results chapter opened by saying
nothing in it is typed by hand, thirty lines above a hand typed table of the
host triad against worker count. `gen_report_assets.py` now emits
`report/tables/bandwidth_scaling.tex` from the same manifest `tables/bandwidth.tex`
reads, parsing the per worker figures out of each probe's detail string and
adding the spread of that probe's repetitions at each worker count where the
manifest records them; the chapter inputs it where the typed table stood. Beside
it the generator writes `report/tables/numbers.tex`, one `\newcommand` per
measured figure the prose quotes, and `main.tex` inputs that in its preamble, so
a sentence asks for a command and a figure the generator cannot derive is a
LaTeX error rather than a stale claim. The committed generation yields 20
commands. `docs/comparison_methodology.md` gained a
`<!-- generated:start -->` to `<!-- generated:end -->` region that
`gen_report_assets.py --markdown` rewrites, and `make assets` runs the generator
once each way. The verdict fragments phase A7 wrote and nothing read are now
inputted beside the tables they belong to.

Where a figure belongs to the convergence suite rather than to the sweep, the
sentence states what that suite asserts and that it passes, rather than quoting a
value the sweep cannot regenerate. That covers the Jacobi over Gauss Seidel
ratio, the growth order bounds and the discretisation order and constant.

**Commit 2, the traffic model and the stronger conclusion.** The methodology
chapter carries both candidate counts, the instrument, the rule as registered,
the amendment to the statistic with its reason, and the three outcome sentences
quoted verbatim from `benchmarks/sweep_matrix.yaml` under a "Pre registered"
heading. The generator writes `report/tables/traffic_model.tex`, which reads the
manifest's `traffic_model` key and quotes the pre registered sentence for the
outcome, reading the sentences out of the YAML rather than holding a second copy
of them. The committed manifest has no such key, so the fragment reads pending
today and phase A8b's regeneration fills it in with no hand edit.

The discussion says which way the correction cuts: under read for ownership the
host figures rise by a third and the device figures do not move, so host Jacobi
stops being the one row where the two devices are not used comparably well, and
the remaining advantage collapses to the ratio of the two measured triads, which
is the conclusion the other three kernels already gave. It is phrased
conditionally on the model. The three claims Section 4.5 found inside the noise
are restated: the thread models are not separable at this precision and the
chapter says so while quoting the verdict fragment, the reduction cost table is
presented as an upper bound rather than a price, the schedule cost sentences rest
only on the backends whose difference exceeds their spread, and the knee is
stated with its interval and with the per backend disagreement as a result.

Section 0.1.2's statement is in both documents. `docs/comparison_methodology.md`
gained a section and the report's methodology chapter gained
`\section{Comparing these numbers with release 1.0.0's}`: the values are bit
identical because `PNL_REDUCTION_ACCUMULATORS` stays at 1, and every timing
changed because the compiler, the Jacobi algorithm, the timed region and the work
unit changed. The before and after per headline number is **not** copied into
either document, deliberately. It is recorded phase by phase in this file,
against the phase that moved it and beside the gate output that measured it, and
both documents point here. A number typed into a report is a number that stops
being true at the next sweep, which is the fault this phase exists to remove; the
task file asked for the before and after to be quoted from `PROGRESS.md`, and
quoting it into prose would have reintroduced exactly the mechanism of DOC-01.

**Commit 3, related work, the bibliography and the dash carve out.**
A related work section in `discussion.tex` and `docs/RELATED_WORK.md`, citing
Intel's Conditional Numerical Reproducibility, Demmel and Nguyen on reproducible
summation (2013 and 2015), Ahrens, Demmel and Nguyen on the binning algorithm
behind ReproBLAS, Kokkos (2014 and 2022), RAJA, Ginkgo, PETSc (the 1997 paper and
the current users manual) and hypre. Eleven new entries in `refs.bib`. Every one
was checked against the publisher, the conference proceedings or the project's
own documentation before it was written; the two that have no paper of their own,
the vendor reproducibility mode and the PETSc manual, are cited as documentation
with a bib `note` saying so. Nothing was reconstructed from memory.

The section then states what this project adds, which is not the deterministic
reduction: one invariant held across seven execution models including a GPU and a
distributed backend, asserted as exact equality in a test suite, and priced with
its spread. It says plainly that no baseline is in this report and that a single
PETSc comparison is release 1.2.0's.

Ground rule 1's carve out is implemented in `scripts/check_no_dashes.py` and
recorded as an "Amended 2026-09-06" paragraph on decision 17, dated from the WSL
wall clock rather than from the date the task file suggested. The carve out is
two changes, because a page range appears in two places and only one of them was
ever checked. `.bib` files gain the ligature check `.tex` files already had and
`.bib` files never had at all, exempting a line whose field is `pages` and
nothing else, so checking one file type and exempting one field of it is a
narrower state than not checking that file type. In the compiled PDF the
allowance is anchored on the document: everything from the line reading
`Bibliography` or `References` on its own to the end of the extracted text is the
bibliography, and inside it an en dash is allowed only directly between two
digits. A document with no such heading, which is what the debug report is, gets
no allowance anywhere. `tests/style/fixtures/pages.bib` carries one legitimate
range and one double hyphen in a title, and the self test asserts exactly the
title line is reported; a second case runs the PDF scanner over synthetic
extracted text with a range before the heading, a range after it and a dash
between words after it, and asserts only the middle one is allowed. The three
existing page fields were converted from the written out form to the range form.

**Commit 4, the README.** 382 lines to 146. It carries no measured number at
all. What and why, the supported platforms paragraph phase B5 wrote, install
through `find_package` and through `FetchContent`, a quick start that compiles,
one headline figure linked from `assets/figures/`, the continuous integration
paragraph phase B2 wrote, a wall clock line reading pending until phase A8b, one
line of toolchain pointing here for versions, and a table of links.

**Commit 5, the third report.** `report-personal` is gone from the Makefile,
from `reports`, from `all` and from `clean`. `.gitignore` keeps ignoring
`report_for_me/`. The Phase 8 entry above keeps its table and gained a dated note
saying the third report is private and not published, because this file is a log.

**Commit 6, the debug report.** Every engineering log entry written since
`PROV-01` is now in `report_debug/debug_report.tex`, under its theme: the
numerics entries beside the original four, concurrency and device beside the
destructor hazard, distributed beside the gather, harness and command line beside
the resume logic, and three new chapters, provenance and measurement validity,
build and install and continuous integration, and the documents themselves. It is
a conversion of the log and not a rewrite: where the log quotes a transcript the
section states what the transcript showed, because these macros cannot carry a
verbatim block, and no entry gains a conclusion the log does not reach. The
abstract says the document now covers two bodies of work and how they differ, and
the closing chapter gains the six lessons the second half teaches. 16 pages to
42, counted with pdfinfo on the built document.

**Commit 7, the toolchain probe.** `benchmarks/run_sweep.py` filled
`session.toolchain["cxx"]` by running a literal `g++-16`, so the interim manifest
records a GCC 16 trunk snapshot as the compiler of a binary `g++-15` built, and
the publication manifest would have carried the same line into `assets/` and into
the archive README. `toolchain_versions(build)` now reads `CMAKE_CXX_COMPILER`
from `<build>/CMakeCache.txt` and runs whatever it names, falling back to
`unavailable` and a field saying why when the line is absent, never to a compiler
name. `collect_session` passes `binary.parent`, which is the build tree by
construction. For MPI, a first line carrying no digit is not a version, so the
probe falls through from `mpirun --version` to `ompi_info --version` and the
manifest records `mpi_probe`, the command that answered.

The probe re-run against `build/`, with no sweep:

```text
$ grep -m1 '^CMAKE_CXX_COMPILER:' build/CMakeCache.txt
CMAKE_CXX_COMPILER:UNINITIALIZED=g++-15
$ mpirun --version | head -1
--------------------------------------------------------------------------
$ ompi_info --version | head -1
Open MPI v5.0.10
$ python3 -c "<load run_sweep>; print(json.dumps(module.toolchain_versions(Path('build')), indent=2))"
{
  "cxx": "g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0",
  "cxx_probe": "g++-15 --version, from build/CMakeCache.txt",
  "cmake": "cmake version 4.4.0",
  "nvcc": "nvcc: NVIDIA (R) Cuda compiler driver",
  "mpi": "Open MPI v5.0.10",
  "mpi_probe": "ompi_info --version"
}
```

`cxx` reads `g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0` and `mpi` reads
`Open MPI v5.0.10`, against `g++-16 ... 16.0.1 20260322 (experimental)` and a row
of hyphens in the interim manifest. `nvcc` is left as it is: its first line
carries no version either, but that is an uninformative value rather than a wrong
one, and widening the repair to a second tool in the same commit is how a phase
stops being reviewable. The engineering log entry records it as not fixed here.

**Gate.**

```text
$ grep -n '37\.7\|62\.3\|64\.3\|61\.35' report/chapters/results.tex README.md \
      docs/comparison_methodology.md
expect: no output

$ git grep -Enw 'petsc|PETSc|Trilinos|hypre|AMGX|Eigen|Ginkgo|Kokkos|RAJA|LAPACK|OpenBLAS|MKL|cuSPARSE|cuBLAS' \
      -- '*.md' '*.tex' '*.bib' '*.yaml' '*.py' | wc -l
33
$ git grep -Enw '<the same pattern>' c10df49 -- '*.md' '*.tex' '*.bib' '*.yaml' '*.py' | wc -l
1

$ python3 scripts/gen_report_assets.py --allow-dirty \
      && python3 scripts/gen_report_assets.py --allow-dirty --markdown
...
  table   report/tables/bandwidth_scaling.tex
  verdict report/tables/traffic_model.tex
  numbers report/tables/numbers.tex, 20 command(s)
gen_report_assets: done
...
  markdown docs/comparison_methodology.md

$ make report-only && make report-debug
Output written on main.pdf (49 pages, 529728 bytes)
Output written on debug_report.pdf (42 pages)
check_no_dashes: clean, 1 file(s) scanned

$ python3 scripts/check_no_dashes.py . report/main.pdf report_debug/debug_report.pdf
check_no_dashes: clean, 271 file(s) scanned

$ python3 tests/style/check_linter.py
  pass  linter detects planted violations and ignores legitimate ones
  pass  the page range carve out is one bib field wide and one PDF region wide
2 passed, 0 failed

$ wc -l README.md
146 README.md

$ grep -rn 'report_for_me\|report-personal' Makefile README.md docs report
docs/BUILD_SPECIFICATION.md:92:  report_for_me/                  (report for me, Section 19)
docs/BUILD_SPECIFICATION.md:93:  report_for_me.tex
docs/BUILD_SPECIFICATION.md:191:Project Documentation Deliverable (LaTeX to PDF) report_for_me.pdf

$ make build && make test
100% tests passed out of 33

$ make install-test
pnl 1.0.0 / backend openmp / workers 4 / unknowns 16129 / iterations 442
the registered backend's iterate is bit identical to the serial one,
all 4225 values, compared with == and not with a tolerance.

$ ruff check benchmarks scripts tests
All checks passed!

$ clang-format --dry-run --Werror over include src tests examples
clang-format: clean

$ git status --porcelain
```

The README line count is 382 before and 146 after. The count in the task file and
in Section 4.11, 346 or 347, is the release 1.0.0 length; phases B2 and B5 added
the continuous integration and supported platforms paragraphs, both of which
survive into the 146.

The prior art grep reads 33 over the tree this entry is committed in and 28 in
the `DOC-02` verification, which is the count at the commit that fixed the
finding. The five between them are this entry, which names the libraries it says
were cited. The engineering log keeps its number rather than being updated to
match, because it records what the command returned when the repair was made.

**The one gate line that is not clean, and why.** The third report grep returns
three lines, all from `docs/BUILD_SPECIFICATION.md`. That file is the version 1
build specification. It is untracked and ignored, `git check-ignore -v` names
`.gitignore:70`, and `git ls-files docs/` does not list it, so it is a local
working copy and not part of the repository a reader clones. The protocol forbids
editing either specification, and decision 18 records why the V1 document stays
where it is. The same grep restricted to tracked files,
`git grep -n 'report_for_me\|report-personal' -- Makefile README.md docs report`,
returns nothing, which is the gate over the files that are actually published.

**Findings.** `DOC-01`, the hand typed table under the sentence that said nothing
was hand typed, with the four irreconcilable figures. `DOC-02`, no prior art and
no baseline. `PROV-05`, the manifest naming a compiler that built nothing and a
row of hyphens as the MPI version. All three in `docs/ENGINEERING_LOG.md`.
Decision 17 amended.

**What this phase leaves for A8b, and one consequence to expect.** The prose is
now regenerable end to end: the bandwidth scaling table, the numbers file and the
traffic model fragment all come from the manifest, and the methodology document's
tables come from the same run, so A8b's regeneration rewrites every measured
figure in the report and in that document with no hand edit. The traffic model
fragment reads pending until a manifest carries a `traffic_model` key.

The consequence: the continuous integration reports job compares the rebuilt PDFs
against the tracked ones under `assets/reports/` numeric token by numeric token,
and this phase changed the prose of both documents while deliberately not
rebuilding the tracked copies, which live under `assets/` and are A8b's. That
comparison diverges from this commit until A8b republishes them. It is the gate
working: the documents did change, and the job exists to say so.

### Phase A8b: the publish pass, once

Done, in two commits, and the order is the reverse of the one the task file
named. The task file's commit 1 was the data, the assets and the reports and its
commit 2 was the archive, with the instruction to keep that order only if the
tree stays consistent at each. It does not. The publication sweep left
`summary.csv` holding two generations, 425 rows at `cd57032941a8.dirty` and 425
at `fba9872e407f`, because the commit is part of a row's identity and the driver
appends rather than replaces; the generator refuses a two generation summary by
design, so the split has to happen before anything is regenerated. In the task
file's order the commit that publishes the new data is also the commit that
deletes the old rows, and the archive that receives them arrives one commit
later, so there is a commit in between where the 1.0.0 generation exists nowhere
in the tree. Archiving first costs one commit in which the same 425 rows sit in
two files and loses nothing. That is `edc9966`, and the publish commit follows
it.

No sweep, no bandwidth refresh and no `make all` was run in this phase. The
measurement is the orchestrator's publication session at the freeze commit and
nothing here re measured any part of it.

**The pre registered rule, applied.** The manifest carries the whole statistic
under `traffic_model`, computed by the driver under the rule as amended in phase
A8a: `w*` is 8, the five plain figures are 64.294, 69.529, 70.007, 62.031 and
68.743 GiB/s, the five non temporal ones are 63.679, 71.518, 54.914, 70.119 and
70.405, the five ratios are 0.9904, 1.0286, 0.7844, 1.1304 and 1.0242, the
statistic is 1.0242 and the interval is 0.7844 to 1.1304. The outcome is
**unresolved**: the statistic is below the 1.10 threshold and would select the
conservative model on its own, and the interval contains 1.10, which is the
clause the amendment added for ground rule 7 and which fires regardless of where
the statistic falls. The rule was applied by hand against the manifest before it
was believed, step by step, and every step agrees with the driver; the working is
in `MEAS-13`. The generator's `report/tables/traffic_model.tex` selects the pre
registered sentence for that outcome mechanically, reading the sentence from
`benchmarks/sweep_matrix.yaml` and the outcome from the manifest, so no hand
chose it. It already implemented the rule, so nothing was added to it, and
`scripts/gen_report_assets.py` still imports `yaml` exactly where it did before
phase E4 left it. The continuous integration matrix legs therefore need no
`pyyaml`, and `.github/workflows/ci.yml` is untouched by this phase.

The registration gains `measured_ratio: 1.0242`, `outcome: unresolved` and an
`applied` block recording the session, the matched worker count, both probes'
five figures, the five ratios, the statistic, the interval and the reason.
Nothing above it moved: not a threshold, not the selection statistic, not one of
the three outcome sentences. Those were fixed on 2026-09-05 and amended once, on
2026-09-06, before the measurement, and a pre registration that is edited after
the measurement is not one.

**Before and after.** Before is the 1.0.0 published generation at
`cd57032941a8.dirty`, now at
`experiments/results/archive/summary-cd57032941a8-dirty.csv`; after is this
generation at `fba9872e407f`. Both columns are generated by the same generator
from the two files, so what moved is the data and not the arithmetic that reads
it: the before column was produced by running
`scripts/gen_report_assets.py --allow-dirty --results-dir` at a scratch copy of
the archived generation and its archived manifest, and reading the tables that
came out. No number below is from memory or from a previously published
document.

| quantity | before, `cd57032941a8.dirty` | after, `fba9872e407f` |
| --- | --- | --- |
| host triad, plain stores | 61.4 GiB/s, best at 4 workers | 70.0 GiB/s, best at 8 workers |
| host triad, non temporal stores | pending, the probe did not exist | 71.5 GiB/s, best at 8 workers |
| device triad | 549.5 GiB/s | 550.4 GiB/s |
| device over host triad | 9.0 | 7.9 |
| Jacobi speedup at 20 workers over 1, openmp | 1.45, from 0.114588 s (spread 1.9 percent) to 0.078850 s (spread 19.1 percent) | 10.65, from 0.068041 s (spread 25.4 percent) to 0.006390 s (spread 85.9 percent) |
| Jacobi speedup at 20 workers over 1, pthreads | 1.25, from 0.116863 s (spread 11.5) to 0.093790 s (spread 10.0) | 0.61, from 0.061343 s (spread 11.4) to 0.100275 s (spread 169.5) |
| Jacobi speedup at 20 workers over 1, jthread | 1.17, from 0.111708 s (spread 14.1) to 0.095661 s (spread 8.4) | 1.84, from 0.073745 s (spread 123.9) to 0.040003 s (spread 20.3) |
| jacobi, host efficiency, declared model | 28.5 percent | 76.1 percent |
| jacobi, host efficiency, counted model | predates the column | 101.5 percent |
| jacobi, device efficiency, declared model | 95.1 percent | 98.1 percent |
| jacobi, device efficiency, counted model | predates the column | 109.9 percent |
| gauss_seidel_rb, host efficiency, declared | 46.4 percent | 43.9 percent |
| gauss_seidel_rb, host efficiency, counted | predates the column | 117.1 percent |
| gauss_seidel_rb, device efficiency, declared | 48.1 percent | 49.4 percent |
| gauss_seidel_rb, device efficiency, counted | predates the column | 120.3 percent |
| sor_rb, host efficiency, declared | 46.4 percent | 43.8 percent |
| sor_rb, host efficiency, counted | predates the column | 116.7 percent |
| sor_rb, device efficiency, declared | 48.0 percent | 49.3 percent |
| sor_rb, device efficiency, counted | predates the column | 120.7 percent |
| cg, host efficiency, declared | 20.5 percent | 19.6 percent |
| cg, host efficiency, counted | predates the column | 156.7 percent |
| cg, device efficiency, declared | 20.0 percent | 20.0 percent |
| cg, device efficiency, counted | predates the column | 152.4 percent |
| jacobi efficiency ratio, device over host | 3.34 | 1.29 |
| knee, openmp | 5 workers, interval 4 to 20, triangular fallback | 20 workers, interval 20 to 20, measured repetitions |
| knee, pthreads | 3 workers, interval 3 to 4, triangular fallback | 4 workers, interval 4 to 4, measured repetitions |
| knee, jthread | 3 workers, interval 3 to 3, triangular fallback | 9 workers, interval 3 to 17, measured repetitions |
| reduction cost, openmp | not separable, medians differ 5.5 percent against a spread of 9.4 | not separable, medians differ 4.7 percent against a spread of 99.3 |
| reduction cost, pthreads | not separable, 0.3 against 4.5 | not separable, 1.5 against 31.6 |
| reduction cost, jthread | not separable, 2.5 against 10.4 | not separable, 0.2 against 4.6 |
| reduction cost, mpi | not separable, 3.1 against 77.5 | not separable, 2.4 against 14.5 |
| schedule cost, openmp | not separable, medians differ 6.7 percent against a spread of 19.8 | dynamic 28.1 percent slower, against a wider spread of 25.7 |
| schedule cost, pthreads | dynamic 14.2 percent slower, against a wider spread of 9.1 | not separable, 3.5 against 11.3 |
| schedule cost, jthread | dynamic 12.3 percent slower, against a wider spread of 6.9 | not separable, 4.0 against 14.2 |
| thread models against serial on jacobi | openmp not separable (14.0 against 19.1), pthreads 19.2 percent faster, jthread 17.2 percent faster | openmp 59.6 percent faster, pthreads 48.5, jthread 46.9, none of the three inside its spread |
| comparisons not separable, over the four verdict fragments | 27 of 56 | 31 of 56 |
| traffic model outcome | pending, no manifest carried the key | unresolved, statistic 1.0242, interval 0.7844 to 1.1304 |

Five things in that table are worth saying out loud rather than leaving for a
reader to notice.

The Jacobi row is the phase A1 result arriving in the published sweep. The
scaling block runs at 1,046,529 unknowns, which is largely inside the 33 MiB last
level cache, and the openmp curve goes from 1.45 to 10.65 because the serial full
state copy that ran on the calling thread every iteration is gone. The pthreads
figure at 20 workers is 0.61, which is slower at twenty workers than at one, and
it is not a result: that row's spread is 169.5 percent, and ground rule 7 says an
effect smaller than its own spread is not a number. The knee table is where that
curve is read properly, with a bootstrap interval, and the dispersion table now
reports a median spread of 68.0 percent over the 66 scaling rows against 9.8
percent before. The scaling block is the noisiest block in the sweep and this
generation says so where the previous one did not have the repetitions to.

The host triad moved from 61.4 to 70.0 GiB/s and that is `PROV-04` being fixed
rather than the machine getting faster. The 1.0.0 figure was probed at the start
of a session on a machine that had just built and tested, and the refresh that
`make bandwidth-refresh` exists for never ran; this one was re probed on an idle
machine after the sweep, and the manifest records `bandwidth_refreshed` at
2026-09-06T23:11:12+0200. Every efficiency figure in the report divides by it, so
a depressed reading inflated all of them, which is why the device over host triad
ratio falls from 9.0 to 7.9 while the graphics card did not change.

Host Jacobi efficiency goes from 28.5 percent to 76.1 and the Jacobi efficiency
ratio from 3.34 to 1.29. That is the conclusion of the whole comparison
strengthening: under the declared model the host now uses its memory system
comparably well to the device on all four kernels, so the Jacobi row stops being
the one place a reader could point at and say the implementations differed rather
than the hardware. It happens under the conservative byte count, without the
correction the traffic model question was about, which is a better outcome than
needing it.

The counted column carries numbers for the first time, and every one of its
efficiencies at the streaming size exceeds one hundred percent. That is not the
data. It is `MEAS-14`: the numerator is recounted at 32 bytes per unknown per
pass and the denominator is the triad's declared figure at 24, and for the device
rows the two columns are not even on the same clock. It is recorded and not
fixed, because fixing it changes what a published table means in the commit that
publishes it, and the declared column, which is the one the report leads with and
the one every figure uses, is unaffected.

The schedule cost verdicts swapped which backends they can separate, and the
count of comparisons inside the noise went from 27 of 56 to 31 of 56. Neither is
a finding. It is what ground rule 7 looks like when the repetitions are real:
this generation carries 15 repetitions per row and the previous one carried the
minimum, median and maximum of a much thinner run, so a comparison that could be
stated before and cannot be stated now was never separable, it was only quoted
as though it were.

**The archive.** `experiments/results/archive/summary-cd57032941a8-dirty.csv` is
byte for byte the 426 lines that were in `../summary.csv` at the freeze commit,
header included; `git show fba9872e407f:experiments/results/summary.csv | cmp -
experiments/results/archive/summary-cd57032941a8-dirty.csv` is silent. The
archive README gained a table that lets a reader reproduce either generation, or
find out that they cannot: which commit, whether it is in git history at all,
which compiler and which C++ standard, which byte model, which sweeps count and
which triad every efficiency divides by. It also records what was checked rather
than assumed about the two generations agreeing. Joining them on everything that
identifies a configuration except the commit gives 410 configurations named by
both; 396 carry the same `relative_residual` and the same `iterations` to every
digit the row prints; the 14 that differ are the whole of the `reduction_cost`
block, whose fixed iteration count phase A7 raised from 200 to 3000, and the
whole of the `schedule_cost` block, raised from 100 to 1100. The 15
configurations only on the old side are the hybrid rows recording 5 workers and
the 15 only on the new side are the same configurations recording 20, which is
`MEAS-09`. The timings are superseded and the arithmetic is not.

**The wall clock, and how it is composed.** The README carries two figures and
neither is a stopwatch reading of a single command, because no single command was
run. `make all` is `setup build check-style test sweep bandwidth-refresh
reports`, and the publication session deliberately stopped before `reports`: the
generator would have refused the two generation summary and `publish_assets.py`
would have written into the tracked `assets/` before the archive split had
happened. So the figure is the sum of two measured halves.

| half | what ran | seconds |
| --- | --- | --- |
| the orchestrator's publication session, from a clean tree | `make clean setup build check-style test sweep bandwidth-refresh` | 2312 |
| this phase | `make report` then `make report-debug`, which is `make all`'s `reports` | 17 |
| total, `make clean && make all` | | 2329 |

The 2312 is the board's figure for the session that started at 20:32:40Z and
ended at 21:11:12Z, and its parts are build 44 s with a warm ccache, check style
and test 36 s with 33 of 33 passing, sweep 2195 s and bandwidth refresh 37 s. The
17 is measured here with `date +%s` either side: `make report`, which is the
generator run twice and then latexmk over `main.tex` from no `.aux` file, took
13 s, and `make report-debug` took 4 s. The second figure in the README is the
sweep session alone, taken from the manifest rather than from a clock: `started`
is 2026-09-06T22:34:09+0200 and `bandwidth_refreshed` is 23:11:12+0200, which is
2223 seconds. Nothing else numeric was added to the README.

**Provenance, checked rather than assumed.** The manifest's toolchain block reads

```json
"cxx": "g++-15 (Ubuntu 15.2.0-16ubuntu1) 15.2.0",
"cxx_probe": "g++-15 --version, from .../build/CMakeCache.txt",
"mpi": "Open MPI v5.0.10",
"mpi_probe": "ompi_info --version"
```

which is `PROV-05` fixed and holding on the session that matters: the compiler
named is the publication compiler decision 20 nominates, it was read from the
build cache of the tree that was measured rather than from a literal in the
harness, and the MPI field carries a version instead of a rule of hyphens.

**Idempotence, and what the dry run actually says.** Taken after the archive
split, with `build/pnl` as the publication session left it, which is the binary
that measured these rows and carries their commit stamp:

```text
$ python3 benchmarks/run_sweep.py --build build --dry-run
440 configurations declared, 425 rows in the summary
425 already present at commit fba9872e407f, which is what a sweep would skip
15 have no stored row at any commit: jthread 3, mpi 3, openmp 3, pthreads 3, serial 3
0 stored rows that no declared configuration predicts
```

Two things there, and neither is "nothing to run".

The 15 with no stored row are the 15 the sweep declares and the binary refuses:
conjugate gradient on the dense non symmetric problem, three configurations on
each of five backends, which exit 3 with `pnl: invalid argument: solver not
applicable`. They are declared inapplicable rather than missing, they are counted
in the manifest under `failures`, and a run would attempt them again, get the
same refusal in a few seconds and write no row. The line that would have mattered
is gone: at the freeze the same command reported 15 stored rows that no declared
configuration predicts, which were the 1.0.0 generation's hybrid rows at 5
workers, and after the split it reports none.

The second thing is larger and is worth stating plainly rather than leaving for
someone to discover. The commit is part of a row's identity and it is stamped
into the binary by `CMakeLists.txt` at configure time, from `git rev-parse
--short=12 HEAD` with `.dirty` appended when the narrowed status is not clean. So
the resume set is keyed on the commit the binary was configured at, not on the
source it was built from. `make build` in this phase's own gate, run at commit
`edc9966`, produced a binary that reports

```text
$ ./build/pnl --version
pnl 1.0.0
commit edc9966d1705.dirty
```

and a sweep driven by that binary would find no row at that commit and re measure
all 425. That is not a defect this phase introduced; it is the property Section 7
A8 of the specification builds the whole phase around when it says there is
exactly one publishable measurement session, and it is why the freeze exists. But
it means "idempotent" has a narrow meaning here that is worth being exact about:
from the tree the publication session left, with the binary it left, a rerun adds
no row and changes no published number. From this tree, after the two commits
that publish the data, a rebuild stamps the new head and a sweep would re measure
rather than resume. Phase E5 tags the release, and a stranger reproducing it
checks out the tag and re measures from that commit; the summary that comes out
carries the tag's commit and not `fba9872e407f`, and a second generation in
`summary.csv` is what the archive procedure and the generator's refusal exist for.

**Gate.** Every command below was run through `tasks/run.sh` inside WSL, in this
order, before the publish commit. The tree check is after it.

```text
$ grep -n '37\.7\|62\.3\|64\.3\|61\.35' report/chapters/results.tex README.md docs/comparison_methodology.md
grep exit 1 (1 means no match, which is the pass)

$ python3 scripts/gen_report_assets.py
  ... 12 charts, 9 tables, 5 verdict fragments, bandwidth_scaling, traffic_model
  numbers report/tables/numbers.tex, 20 command(s)
gen_report_assets: done

$ python3 scripts/compare_report_text.py report/main.pdf assets/reports/main_report.pdf
compare_report_text: 1558 numeric tokens across 50 pages agree between report/main.pdf and assets/reports/main_report.pdf
  volatile lines dropped: none

$ python3 scripts/compare_report_text.py report_debug/debug_report.pdf assets/reports/debug_report.pdf
compare_report_text: 577 numeric tokens across 43 pages agree between report_debug/debug_report.pdf and assets/reports/debug_report.pdf
  volatile lines dropped: none

$ python3 scripts/check_no_dashes.py . assets/reports/main_report.pdf assets/reports/debug_report.pdf
check_no_dashes: clean, 273 file(s) scanned

$ ruff check benchmarks scripts tests
All checks passed!

$ clang-format --dry-run --Werror over include src tests examples
clang-format clean

$ python3 tests/style/check_linter.py
  pass  linter detects planted violations and ignores legitimate ones
  pass  the page range carve out is one bib field wide and one PDF region wide
2 passed, 0 failed

$ awk -F, 'NR>1 {print $27}' experiments/results/summary.csv | sort -u
fba9872e407f

$ make build
[1/2] Building CXX object CMakeFiles/pnl.dir/src/main.cpp.o
[2/2] Linking CXX executable pnl

$ make check-style
check_no_dashes: clean, 271 file(s) scanned
2 passed, 0 failed
All checks passed!

$ make test
100% tests passed out of 33
Total Test time (real) =  18.24 sec

$ make install-test
registered backends: serial openmp pthreads jthread mpi hybrid counting
25 conjugate gradient iterations on a 63 by 63 Poisson problem:
  the registered backend's iterate is bit identical to the serial one,
  all 4225 values, compared with == and not with a tolerance.
```

`file` reports no CRLF on any file this phase wrote: `README.md`, `PROGRESS.md`,
`docs/ENGINEERING_LOG.md`, `docs/comparison_methodology.md`,
`benchmarks/sweep_matrix.yaml`, `experiments/results/summary.csv`, the archive
README, the archived CSV and the session manifest.

**Findings.** `MEAS-13`, the traffic model outcome: unresolved on this machine,
under the rule fixed before the measurement, with the working shown and the
report sentence quoted from the registration. `MEAS-14`, the counted bandwidth
column compared against a triad counted the other way, recorded and deliberately
not fixed. Both in `docs/ENGINEERING_LOG.md`.

**What this phase leaves.** The continuous integration presence list in
`.github/workflows/ci.yml` checks twelve charts, nine tables and five verdict
fragments, and does not yet check the three files phase E4 added:
`report/tables/bandwidth_scaling.tex`, `report/tables/numbers.tex` and
`report/tables/traffic_model.tex`. All three exist and are generated on every
run, and a missing one would fail the LaTeX build rather than pass silently, so
this is a gap in the presence check and not in the report. It is left for E5,
which is the phase that also removes the `--allow-dirty` flag from that job now
that a generation measured from a clean tree is committed. `MEAS-14` is the other
thing left, and it needs a phase that owns the byte model rather than a patch.
