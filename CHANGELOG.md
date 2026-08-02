# Changelog

Notable changes to this project. Format follows Keep a Changelog; versions
follow semantic versioning.

## [1.0.0] 2026-08-02

First release. Consolidates two earlier projects, a numerical methods library
and a parallel scientific computing study, into one repository whose organising
idea is that neither has alone: one numerical library, many execution backends,
so a benchmark can isolate what each programming model costs on identical work.

### Numerical core

- Root finding: bisection, Newton, Brent, each with its convergence order tested.
- Quadrature: adaptive Simpson, Gauss Legendre with nodes computed by Newton
  iteration on the Legendre recurrence rather than tabulated, and Romberg.
- Ordinary differential equations: classical RK4 and adaptive Dormand Prince
  5(4) with a standard step size controller.
- Dense linear algebra: LU with partial pivoting, Householder QR, and the Thomas
  algorithm for tridiagonal systems.
- Every result carries value, error estimate, iteration count, evaluation count,
  converged flag and an explicit stop reason, so a non convergence can never be
  read as success.

### Solver zoo

- Twelve iterative solvers presented as one splitting family: Richardson,
  Jacobi, forward, backward, symmetric and red black Gauss Seidel, SOR, SSOR,
  red black SOR, block Jacobi, block Gauss Seidel, and conjugate gradient.
- Two problem families: the 2D Poisson five point stencil with closed form
  theory, and seeded dense systems in diagonally dominant and symmetric positive
  definite flavours.
- Block methods solve their diagonal blocks exactly, by the Thomas algorithm for
  the stencil and by LU for dense systems.

### Execution backends

- Seven backends behind one interface: serial, OpenMP, POSIX threads,
  `std::jthread` pool, MPI, hybrid MPI with OpenMP, and CUDA.
- **Bit identical iterates** across every shared memory backend at every worker
  count, and between the GPU sweeps and the CPU. Asserted as exact equality by
  the equivalence suite, not as a tolerance.
- Natural ordering Gauss Seidel preserves exact sequential semantics on every
  backend, including across MPI ranks through a pipelined token chain.
- Communication time is measured separately for the distributed backends.

### Measurement

- Declarative sweep matrix with nine blocks, each stating the question it
  answers.
- Resumable sweep driver with atomic merge and a session manifest recording both
  devices' measured STREAM triad bandwidth and the toolchain versions.
- Every result row carries backend, worker count, pinning, seed and the commit
  hash stamped into the binary.
- Figures and tables generated from the summary alone, idempotently.

### Documentation

- Main report, debug report and personal report, all built by `make reports`.
- `docs/comparison_methodology.md` written for a reader who will try to misquote
  a speedup.
- Engineering log kept from the first command, with symptom, root cause, options,
  fix and verification for each entry.

### Verified against theory

- Jacobi spectral radius matches cos(pi h) to 1e-3 relative.
- Jacobi to Gauss Seidel iteration ratio is 2.00 within two percent.
- Young's optimal relaxation factor confirmed as a genuine minimum.
- Optimal SOR changes the growth order from O(n squared) to O(n).
- The five point stencil is second order accurate with error constant
  pi squared over 12.
- Red black ordering penalty measured at 2.6 percent more iterations.

### Known limitations

- The performance versus efficiency core split cannot be identified from inside
  WSL2; the knee is reported from the aggregate scaling curve instead.
- MPI ranks replicate the whole grid rather than allocating a local slab. The
  communication volume is unaffected; the memory ceiling is not.
- Iteration counts at the largest grids are extrapolated from verified closed
  form rates for the methods whose counts grow like n squared.
- Every device conclusion concerns bandwidth bound stencil work.
