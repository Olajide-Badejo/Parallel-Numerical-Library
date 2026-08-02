# Parallel Numerical Lab: Build Specification v2 (merges 03 Numerical Methods Library and 08 Parallel Scientific Computing)

## 0. How to use this document

Paste this entire file as the opening message of a fresh Claude Code session in an empty project folder on the target machine (Section 3). It is a direct instruction to that agent. Work through the phases in Section 16 in order; never mark a phase complete without running its checks and showing output. Keep `PROGRESS.md` at the repo root with a short entry per phase. Use git from the first command, one commit per meaningful unit.

This spec supersedes the two documents it merges. The idea that makes the merge more than a stapling: one numerical library, many execution backends. Every parallelizable algorithm runs unchanged over serial, OpenMP, pthreads, a hand built `std::jthread` pool, MPI, hybrid MPI plus threads, and (for the solvers that admit it) CUDA, through one backend interface, so the benchmark isolates what each programming model costs on identical numerical work. Tool ecosystems move fast: confirm any named package or command is still current and note substitutions in `PROGRESS.md` and the engineering log.

## 1. Aim and objectives

**Aim.** Build a C++23 numerical methods library with a pluggable parallel execution backend layer, a serious iterative solver zoo with cited derivations, and an honest CPU versus GPU solver comparison whose metrics are designed so the hardware does not decide the argument.

**Objectives.**

1. Numerical core with diagnostics on every result (value, error estimate, iterations, converged flag): root finding (bisection, Newton, Brent), quadrature (adaptive Simpson, Gauss Legendre, Romberg), ODE (RK4, Dormand Prince RK45), dense LU with partial pivoting and Householder QR, plus the iterative solver zoo of Section 8.2: Richardson, Jacobi, forward Gauss Seidel, backward Gauss Seidel, symmetric Gauss Seidel, SOR, block Jacobi, block Gauss Seidel, and Conjugate Gradient, each with a brief derivation in the report backed by real citations.
2. Execution backends behind one interface (`parallel_for`, `reduce`, `barrier` semantics): serial, OpenMP, raw pthreads, `std::jthread` pool with `std::barrier`, MPI (remainder aware block decomposition), hybrid MPI plus OpenMP, and CUDA for Jacobi and red black Gauss Seidel sweeps. Identical numerics across backends is a tested invariant, not an assumption.
3. Backend cost study: the same solver on the same seeded problem across every backend, reporting time, scaling across 1 to 20 workers, and the P core versus E core knee on this heterogeneous CPU, with thread pinning as a swept variable.
4. CPU versus GPU comparison done fairly (Section 8.3): hardware independent metrics first (iteration counts, convergence rates), device normalized efficiency second (percent of each device's own measured STREAM bandwidth, since these sweeps are bandwidth bound), absolute time last and labeled machine specific.
5. Ship at professional standard: tests including cross backend equivalence, docs, CI, main report PDF with the derivations chapter, debug report PDF, all reproducible with `make all`.

## 2. Estimated runtime

**ESTIMATED RUNTIME OF THE FULL BACKEND SWEEP (9 SOLVERS X 7 BACKENDS X 3 SIZES X 5 REPS, PLUS SCALING CURVES 1 TO 20 WORKERS): 45 TO 90 MINUTES ON THE TARGET MACHINE.**

**ESTIMATED RUNTIME OF THE TEST SUITE INCLUDING CROSS BACKEND EQUIVALENCE AND CONVERGENCE ORDER CHECKS: 5 TO 12 MINUTES.**

**ESTIMATED RUNTIME OF `make all` FROM A CLEAN TREE (BUILD + TESTS + SWEEP + REPORT): 1.5 TO 2.5 HOURS.**

**ESTIMATED BUILD EFFORT FOR THE AGENT: 8 TO 12 WORKING SESSIONS.**

Estimates, not measurements. Time the first configurations, project the total, and replace these numbers in the README with measured wall clock.

## 3. Target machine and environment strategy

| Component | Spec |
|---|---|
| CPU | Intel Core i7-14700K, 8 P cores plus 12 E cores, 28 threads |
| RAM | 32 GB DDR5-5600 dual channel, about 10 GB realistically free |
| GPU | RTX 5070 12 GB (`sm_120`), used only by the CUDA backend |
| Storage | about 100 GB free |
| OS | Windows 11 Pro; everything runs inside WSL2 Ubuntu |

**Toolchain floors, verified or upgraded at Phase 0:** GCC 16.1 with C++23, CMake 4.4, OpenMPI 5.0.x (`mpicxx`, `mpirun`; the MPI 5.0 standard document is the API reference cited, and Phase 0 records which of its features the installed OpenMPI actually implements, restricting usage to those), OpenMP 6.0 as the cited specification (OpenMP ARB, November 2024) with the build asserting the `_OPENMP` version macro and documenting which newer constructs are used versus fallbacks where GCC support is partial, CUDA 13.3 for the GPU backend, Python 3 with matplotlib and pandas for plots, TeX Live with `latexmk` in WSL for the report (MiKTeX `texify` on Windows as fallback). Raise WSL to the physical core count before scaling runs: `%UserProfile%\.wslconfig` with `[wsl2]` and `processors=20`, then `wsl --shutdown`; confirm with `nproc`. Record every version in `PROGRESS.md`.

## 4. Ground rules

1. **No em dashes and no en dashes, anywhere, ever** (U+2014, U+2013), any file type. **LaTeX trap:** never type `--` or `---` in `.tex` prose (they typeset as dashes); write "1 to 20". Enforced by `scripts/check_no_dashes.py` from Phase 0, wired into `make check-style` and the report build.
2. **Reproducibility is a deliverable.** `git clone` then `make setup && make all` reproduces every number; seeds recorded; nothing hand copied.
3. **No number without a run.** Pending values read "pending". Every results row carries backend, worker count, pinning, seed, and commit hash.
4. **Honest comparison.** The report never headlines a GPU versus CPU speedup without the device normalized framing of Section 8.3 beside it.
5. **Write as the repo owner.** First person, no AI stock phrases, comments only where code cannot say it, plain present tense commits, no TODO stubs. MIT license, sole author Olajide Badejo.

## 5. Background and design choices (and why)

- **One backend interface** because the question worth answering is not "is OpenMP fast" but "what does each model cost on identical work." OpenMP, pthreads, and a `std::jthread` pool all express fork join data parallelism; the interface (`parallel_for(range, chunk, fn)`, `reduce(range, init, fn, op)`) is the minimal common denominator that leaves each backend idiomatic underneath (OpenMP pragmas; pthreads with explicit affinity via `pthread_setaffinity_np`; C++23 `std::jthread` with `std::barrier` and a work stealing free static partitioner, since uniform loops do not need stealing). Design follows the structured parallel patterns treatment in McCool, Robison, Reinders, "Structured Parallel Programming," Morgan Kaufmann 2012.
- **The solver zoo is one family tree, and the report presents it that way.** All classical stationary methods are splittings A = M minus N iterating x_{k+1} = M^{-1}(N x_k + b): Richardson (M = I over omega), Jacobi (M = D), forward Gauss Seidel (M = D + L), backward (M = D + U), symmetric Gauss Seidel (one forward then one backward sweep), SOR (M = D over omega + L, convergent for 0 < omega < 2 on SPD systems by the Ostrowski Reich theorem), and block variants (M built from diagonal blocks). CG is the non stationary contrast with its Krylov optimality. Derivations kept brief, per method, in the report with citations: Saad, "Iterative Methods for Sparse Linear Systems," 2nd ed., SIAM 2003 (splitting framework, convergence theorems, red black ordering); Young, "Iterative Solution of Large Linear Systems," Academic Press 1971 (SOR theory); Golub and Van Loan, "Matrix Computations," 4th ed., Johns Hopkins 2013; Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain," CMU 1994. Convergence is verified empirically against theory: Jacobi versus Gauss Seidel iteration count ratio on the model Poisson problem, SOR at the optimal omega for the 2D Poisson matrix (known in closed form for that model problem, Young 1971), all as tests, not prose claims.
- **Gauss Seidel on GPU requires red black coloring** because the natural ordering sweep is sequentially dependent; red black reordering makes each half sweep fully parallel at the cost of a different (typically slightly worse per iteration, still convergent) iteration count (Saad 2003, parallel implementations chapter; Hager and Wellein, "Introduction to High Performance Computing for Scientists and Engineers," CRC 2010, stencil chapter). The GPU comparison therefore pairs GPU Jacobi and GPU red black Gauss Seidel against their CPU counterparts, like against like.
- **Fair cross device methodology** (the answer to "will the hardware skew it"): (1) iterations to a fixed relative residual are hardware independent and reported first; (2) per iteration cost is bandwidth bound for these sweeps (a 5 point stencil sweep moves a countable number of bytes per unknown), so each device's achieved GB/s divided by its own measured STREAM triad bandwidth is a dimensionless efficiency comparable across devices, standard roofline reasoning (Williams, Waterman, Patterson, CACM 2009; McCalpin's STREAM methodology, IEEE TCCA 1995); (3) absolute time to solution appears last, labeled as a property of this specific CPU GPU pair. The report includes both devices' measured rooflines so a reader can transfer the analysis to other hardware.
- **Dense plus stencil problems.** Solvers run on the 2D Poisson 5 point stencil (the canonical model problem with known theory) and on seeded diagonally dominant dense systems (which exercise the block variants); CG additionally on SPD cases as its theory requires.
- ODE, quadrature, and root finding carry over from the numerical library design with their convergence order tests (Burden and Faires, "Numerical Analysis," 10th ed., Cengage; Hairer, Norsett, Wanner, "Solving Ordinary Differential Equations I," Springer, for the Dormand Prince tableau choice).

## 6. Repository layout

```text
parallel-numerical-lab/
├── README.md  LICENSE  Makefile  CHANGELOG.md  PROGRESS.md  .gitignore
├── CMakeLists.txt  CMakePresets.json  .clang-format
├── include/pnl/
│   ├── core/         types.hpp  error.hpp  diagnostics.hpp
│   ├── backend/      backend.hpp  serial.hpp  openmp.hpp  pthreads.hpp
│   │                 jthread_pool.hpp  mpi.hpp  hybrid.hpp  cuda.hpp
│   ├── solvers/      splitting.hpp  richardson.hpp  jacobi.hpp
│   │                 gauss_seidel.hpp  sor.hpp  block_solvers.hpp  cg.hpp
│   ├── numerics/     roots.hpp  quadrature.hpp  ode.hpp  lu.hpp  qr.hpp
│   ├── problems/     poisson2d.hpp  dense_generator.hpp
│   └── progress.hpp
├── src/
│   ├── backend/      pthreads_pool.cpp  jthread_pool.cpp  mpi_runtime.cpp
│   ├── cuda/         jacobi_sweep.cu  rb_gauss_seidel.cu  stream_probe.cu
│   └── main.cpp      CLI driver: solver, backend, size, workers, pinning
├── tests/
│   ├── unit/         each solver and numeric on closed form cases
│   ├── equivalence/  same solver, all backends, same seed, same answer
│   ├── convergence/  empirical orders and theory ratios (Jacobi vs GS, SOR omega)
│   └── mpi/          1, 2, 4 rank agreement to tolerance
├── benchmarks/       run_sweep.sh  sweep_matrix.yaml  plot_scaling.py
├── experiments/results/   generated; one committed canonical summary.csv
├── docs/             backends.md  solvers.md  comparison_methodology.md
│   ├── DESIGN_DECISIONS.md  ENGINEERING_LOG.md
├── report/           main.tex  refs.bib  chapters/  figures/  tables/  build/
├── report_debug/     debug_report.tex  sections/  Makefile
├── report_for_me/                  (report for me, Section 19)
|   ├── report_for_me.tex
├── scripts/          check_no_dashes.py  gen_report_assets.py
└── .github/workflows/ci.yml
```

## 7. State management and reproducibility

- `benchmarks/sweep_matrix.yaml` declares solver x backend x size x workers x pinning; `run_sweep.sh` resolves it and appends per configuration rows (median of 5 reps, compute and communication timed separately for MPI via barriered `MPI_Wtime` regions) to `experiments/results/`; a merge step assembles `summary.csv` atomically.
- **Resumable sweep:** completed (solver, backend, size, workers, pinning, commit) rows are skipped on rerun; `--force` redoes.
- Both devices' STREAM style bandwidth probes (CPU triad, GPU `stream_probe.cu`) run once per sweep session and land in the session manifest; every efficiency number in the report divides by these measured values, never by spec sheet numbers.
- All problems generated from recorded seeds; `gen_report_assets.py` is idempotent from `summary.csv`.

## 8. Component specifications

### 8.1 Backends

`backend.hpp` defines the interface; each implementation file stays idiomatic. Serial is the correctness anchor. OpenMP uses `schedule(static)` on uniform loops with the `dynamic` cost measured once and reported. pthreads implements a persistent pool with condition variable dispatch and optional affinity. The `std::jthread` pool uses `std::barrier` for sweep synchronization and `std::stop_token` shutdown. MPI does remainder aware block row distribution; the hybrid backend nests OpenMP inside ranks. CUDA implements Jacobi and red black Gauss Seidel sweeps plus the dot and norm reductions CG needs. An equivalence test suite proves every backend produces the same iterates to floating point reduction tolerance on the same seeded problem, with reduction order differences documented rather than hidden.

### 8.2 Solver zoo

Splitting solvers implemented over the generic sweep primitives so each backend gets them for free; SOR omega is a parameter with the model problem optimum tested against Young's closed form for 2D Poisson; block variants use LU on the diagonal blocks from the numerics module. CG follows Shewchuk's formulation with the two `reduce` calls per iteration exposed as the communication cost driver in MPI runs. Stopping criterion everywhere: relative residual below 1e-8 or the iteration cap, both recorded.

### 8.3 CPU versus GPU comparison protocol

Fixed problem: 2D Poisson at 1024 squared, 2048 squared, 4096 squared unknowns. Report, in this order: iterations to tolerance per method (hardware free); bytes per unknown per iteration from the implementation (counted, stated); achieved GB/s per device divided by that device's measured triad bandwidth (the fair efficiency number); absolute seconds (labeled machine pair specific). Then the honest discussion: GPU Jacobi versus CPU symmetric Gauss Seidel trades more iterations against massively higher bandwidth, and the crossover point is derivable from the measured numbers; the report derives it rather than asserting a winner.

## 9. Terminal progress reporting

Rank 0 owns the terminal. `run_sweep.sh` prints a `tqdm` style bar over configurations (percent, current config, ETA, projected total up front). Iterative solvers print an in run bar over iterations with current residual at most twice per second via `progress.hpp`; non zero ranks stay silent. `gen_report_assets.py` runs `tqdm` over figures and tables. TTY aware, plain lines in CI.

## 10. Testing requirements (phase gates)

| Level | What | Gate |
|---|---|---|
| Unit | numerics on closed forms; each solver on a hand checkable 4x4 system | Phase 1 |
| Convergence | RK4 and Simpson orders; Jacobi versus Gauss Seidel ratio and SOR optimal omega on the model problem within theory tolerance | Phase 2 |
| Equivalence | all backends, same seed, same iterates to reduction tolerance | Phases 3 to 5 |
| MPI | 1, 2, 4 rank agreement; remainder sizes covered | Phase 4 |
| CUDA | GPU sweeps match CPU serial to tolerance; red black iteration counts recorded | Phase 5 |
| Sweep | full matrix complete, summary populated, bandwidth probes present | Phase 6 |
| Style | `check_no_dashes.py` zero hits | every phase |

## 11. Style rules

C++: `snake_case` functions, files, namespaces; `PascalCase` types; `UPPER_SNAKE` constants; trailing underscore members; one solver per header; every public function documents method, convergence order, and failure exceptions. Backend files never leak their model's types through the interface. MPI calls wrapped in `MPI_CHECK`; CUDA in `CUDA_CHECK`. `-Wall -Wextra -Wpedantic` clean under `mpicxx` and nvcc. Python `ruff` clean.

## 12. Documentation set

README (what and why, quick start, headline scaling and comparison figures added last), `docs/backends.md` (interface contract, what each backend does underneath, pinning story), `docs/solvers.md` (the family tree with pointers to the report derivations), `docs/comparison_methodology.md` (Section 8.3 written for a reader who will try to misquote a speedup, so the fair framing is unmissable), DESIGN_DECISIONS, ENGINEERING_LOG, CHANGELOG, CONTRIBUTING.

## 13. Main report pipeline

Chapters: introduction; mathematical background (the splitting framework and per method derivations, brief, cited to Saad, Young, Golub and Van Loan, Shewchuk); parallelization design (backend interface, threading models compared, MPI decomposition and communication analysis); comparison methodology; results (scaling curves, P versus E core knee, backend cost table, the CPU GPU study with both rooflines); discussion; conclusion. Built by `make report`: regenerate assets from `summary.csv`, `latexmk -pdf` in WSL, dash check last. Bibliography metadata only.

## 14. Engineering log and debug report (second PDF)

Dated entries from Phase 0 as problems happen (a pthreads pool wakeup bug, a reduction order mismatch between backends, an E core pinning surprise, an MPI oversubscription artifact), each with symptom, root cause, options, fix and why, commit, verification. Converted at Phase 8 into `report_debug/debug_report.pdf` grouped by theme (numerics, concurrency, MPI, environment). Equal rank deliverable.

## 15. CI

GitHub Actions ubuntu-latest: install GCC, OpenMPI, CMake; build zero warnings; run unit, convergence, equivalence (serial, OpenMP, jthread, pthreads), and 2 rank MPI tests; dash check; report compile job builds both PDFs from the committed canonical summary. CUDA build compiles without a GPU; GPU execution tests skip cleanly and say so.

## 16. Roadmap

```mermaid
flowchart LR
    P0[P0 toolchain + wslconfig + dash lint] --> P1[P1 numerics core + serial solvers + tests]
    P1 --> P2[P2 convergence and theory ratio tests]
    P2 --> P3[P3 OpenMP + pthreads + jthread backends + equivalence]
    P3 --> P4[P4 MPI + hybrid backends]
    P4 --> P5[P5 CUDA sweeps + bandwidth probes]
    P5 --> P6[P6 full sweep + scaling + comparison study]
    P6 --> P7[P7 docs from real numbers]
    P7 --> P8[P8 report + debug report]
    P8 --> P9[P9 final QA: make clean && make all + v1.0.0]
```

Gates per Section 10. Phase 9 requires `make clean && make all` reproducing everything with zero manual steps.

## 17. GitHub publication checklist

Repo `parallel-numerical-lab` (consolidates the two older repos; pointer READMEs added to them after this ships). Description: "C++23 numerical library where every solver runs over pluggable backends (serial, OpenMP, pthreads, std::jthread, MPI, hybrid, CUDA): a nine method iterative solver zoo with cited derivations, cross backend equivalence tests, and a hardware fair CPU versus GPU study." Topics: `cpp23`, `mpi`, `openmp`, `pthreads`, `cuda`, `numerical-methods`, `iterative-solvers`, `hpc`, `parallel-computing`. MIT, sole author Olajide Badejo, attribution disabled via `.claude/settings.local.json` before first commit, history verified clean of Co-authored-by and Claude, `v1.0.0` with both PDFs attached.

## 18. Definition of done

- [ ] `make all` clean on a fresh clone; README states measured wall clock
- [ ] Zero dash characters repo wide including compiled PDFs
- [ ] All nine solvers pass unit, convergence theory, and equivalence gates on every applicable backend
- [ ] MPI agreement at 1, 2, 4 ranks; communication fraction measured and reported for CG
- [ ] CUDA sweeps verified against serial; red black iteration penalty measured and discussed
- [ ] Full sweep complete; scaling, knee, and comparison figures generated from `summary.csv` only
- [ ] Both devices' bandwidth probes present in the session manifest; every efficiency number divides by measured, not quoted, bandwidth
- [ ] Report derivations chapter complete with the Section 5 citations; both PDFs compile through the pipeline
- [ ] Zero warnings; CI green; `v1.0.0` tagged



## 19. Personal Report
Project Documentation Deliverable (LaTeX → PDF) report_for_me.pdf

**Task:** Produce a complete, standalone project documentation report for personal
reference, covering the project from start to finish. Compile it from a `.tex`
source file into a final PDF.

**Required Content**

1. **Project Overview**
   - Purpose and goals of the project
   - Scope and context

2. **Specifications**
   - Technical requirements and constraints
   - Tools, frameworks, languages, and versions used

3. **Implementation Details**
   - What was built/implemented, component by component
   - Why each part was implemented the way it was (design rationale)

4. **Chronological Steps**
   - Full step-by-step account of how the project progressed, start to finish

5. **Problems & Errors**
   - Every significant issue or error encountered
   - How each was diagnosed and solved
   - Why that particular solution was chosen over alternatives

6. **Future Work**
   - Potential upgrades or improvements
   - Features or optimizations not yet implemented, and why they'd help

**Format Requirements**

- Write the full report as a `.tex` file (proper LaTeX structure: title page,
  table of contents, sections/subsections as above)
- Compile it into a PDF
- Be as detailed and thorough as possible, this is for personal archival use,
  not a summary

**Final Output**

At the end, provide **both**:
- The `.tex` source file
- The compiled `.pdf` file