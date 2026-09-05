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
