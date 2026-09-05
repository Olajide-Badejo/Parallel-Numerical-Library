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
