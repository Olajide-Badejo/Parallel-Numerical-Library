# The solver zoo

Twelve methods, eleven of which are one family. This document is the map; the
derivations with their citations are in the mathematical background chapter of
the report.

## The family tree

Every classical stationary method is a choice of **matrix splitting**. Write
A = M minus N with M invertible. Then A x = b is equivalent to M x = N x + b, and
the iteration is

```text
x_{k+1} = M^{-1} (N x_k + b) = x_k + M^{-1} r_k,     r_k = b - A x_k
```

with iteration matrix G = M^{-1} N = I minus M^{-1} A. It converges for every
starting vector exactly when the spectral radius of G is below one. Splitting
A = D + L + U into diagonal, strictly lower and strictly upper parts:

| Method | M | Header | Parallel? |
| --- | --- | --- | --- |
| Richardson | I / omega | `richardson.hpp` | fully |
| Jacobi | D | `jacobi.hpp` | fully |
| Gauss Seidel forward | D + L | `gauss_seidel.hpp` | no, sequentially dependent |
| Gauss Seidel backward | D + U | `gauss_seidel.hpp` | no |
| symmetric Gauss Seidel | forward then backward | `gauss_seidel.hpp` | no |
| red black Gauss Seidel | D + L in red black order | `gauss_seidel.hpp` | fully |
| SOR | D / omega + L | `sor.hpp` | no |
| SSOR | symmetric SOR | `sor.hpp` | no |
| red black SOR | D / omega + L, red black | `sor.hpp` | fully |
| block Jacobi | block diagonal of A | `block_solvers.hpp` | over blocks |
| block Gauss Seidel | block lower triangle of A | `block_solvers.hpp` | no |
| conjugate gradient | not a splitting | `cg.hpp` | fully, but reduction bound |

The nine that Objective 1 names are the first five plus SOR, the two block
methods and conjugate gradient. The three extras exist for good reasons: the two
red black variants because the GPU comparison needs a parallel Gauss Seidel, and
SSOR because it completes the symmetric preconditioner story.

Conjugate gradient is deliberately not in the splitting list. It is not
stationary and its optimality argument is different in kind, which is exactly why
the report presents it as the contrast that makes the family visible.

## What each one buys, measured on the model problem

On the 2D Poisson five point stencil the spectral radii are known in closed form,
and `poisson_theory()` computes them so that no test re-derives them:

| Quantity | Closed form |
| --- | --- |
| Jacobi spectral radius | cos(pi h) |
| Gauss Seidel spectral radius | cos^2(pi h) |
| Optimal SOR factor | 2 / (1 + sin(pi h)) |
| SOR spectral radius at the optimum | omega\* minus 1 |

The consequences, all of which the convergence suite checks as measurements
rather than accepting as prose:

- **Gauss Seidel needs exactly half the iterations of Jacobi.** Measured ratio
  2.00 within two percent at two grid sizes, which is Young's theorem that
  rho_GS = rho_J squared on a consistently ordered matrix.
- **Forward and backward Gauss Seidel converge at the same rate**, because on a
  symmetric operator their iteration matrices are similar.
- **Optimal SOR changes the order of the work**, from O(n^2) iterations to O(n)
  on an n by n grid. This is the single largest improvement available inside the
  stationary family, and the test asserts the growth rates differ rather than
  just that SOR is faster.
- **Young's omega\* is a genuine minimum.** Perturbing it by plus or minus 0.02
  and 0.05 costs iterations in both directions.
- **Red black ordering costs a little.** It changes the iteration matrix, so it
  converges slightly more slowly per iteration: 21938 against 21383 iterations on
  a 127 grid, a penalty of about 2.6 percent. That is the price of the
  parallelism, and it is measured.
- **Line relaxation beats point relaxation.** Solving each grid line exactly by
  the Thomas algorithm rather than each point cuts the count, and line Jacobi to
  line Gauss Seidel stands in the same ratio of two as the point case.
- **Conjugate gradient reaches O(n) without being told omega.** The condition
  number is O(h^-2), so the error bound gives O(h^-1) iterations. Measured 41 to
  93 iterations as the grid doubles from 31 to 63.

## Two traps worth knowing about

### The manufactured solution is an eigenvector

The obvious source for the Poisson problem, u = sin(pi x) sin(pi y), is exactly
the lowest eigenvector of the five point stencil. With a zero initial guess the
Krylov subspace conjugate gradient builds is therefore one dimensional, and it
converges in **one iteration at any grid size**. That is the problem being
degenerate, not the method being fast.

The same choice is *ideal* for the stationary methods, because the slowest
decaying mode of the Jacobi iteration matrix is that same lowest mode, so a
single mode source isolates the asymptotic rate and reproduces cos(pi h) to six
digits.

Both right hand sides therefore exist and each is used where it is honest:

- `PoissonRhs::ManufacturedSine` has a closed form solution. Used by the
  discretisation error tests and the stationary rate tests.
- `PoissonRhs::SpectrallyRich` is a seeded pseudo random source exciting the
  whole spectrum. Used for anything involving conjugate gradient and for the
  whole benchmark sweep, where a degenerate spectrum would flatter one method.

A dedicated test asserts the one iteration behaviour, so the trap stays recorded
rather than merely avoided.

### The relaxation default

`SolverOptions::relaxation` defaults to **zero, meaning "ask the solver"**, not
to one. Each method's natural factor is different: Young's closed form optimum
for SOR, one for SSOR, and the reciprocal of the Gershgorin bound for Richardson.
Defaulting to one handed Richardson a step eight times too large on the Poisson
operator, whose eigenvalues reach 8, and it diverged immediately. See NUM-02.

## Blocks

The block methods take a block count as a **solver parameter, never the worker
count**. That is what keeps their iterates identical whatever the backend, so the
equivalence suite can compare them bit for bit. Each problem defines its natural
block and solves it exactly:

- **2D Poisson**: one grid line per block. The diagonal block is tridiagonal and
  is solved by the Thomas algorithm, which is O(n) and needs no pivoting because
  the block is diagonally dominant. These are the classical line methods.
- **Dense**: contiguous index ranges, solved by LU with partial pivoting from
  `numerics/lu.hpp`. The factorisations are computed once in the constructor and
  reused every sweep, since the diagonal blocks never change; that is what makes
  the block methods competitive rather than merely correct.

## Which solvers apply to what

`Solver::applicable_to()` refuses rather than approximating, and
`inapplicable_reason()` says why in a sentence a user can act on:

- **Conjugate gradient** needs symmetric positive definiteness. The A
  orthogonality of its search directions and the energy norm it minimises are
  both undefined otherwise. On a non symmetric system it throws; if the operator
  turns out not to be positive definite at runtime, a non positive curvature
  `p^T A p` is a certificate of that and the solver reports it as a breakdown
  rather than continuing.
- **The red black variants** need a bipartite coupling graph. The five point
  stencil has one; a general dense system does not.
- **The CUDA path** implements the stencil only, and only the parallel methods.

## Adding a solver

1. New header under `include/pnl/solvers/`, one solver per header.
2. Derive from `Solver`. If it is stationary, express it as a sweep and hand it
   to `detail::run_stationary`, which gives you the residual test, the history,
   the progress bar and the final gather for free, and guarantees you measure
   convergence the same way as everything else.
3. Document the splitting, the convergence order, and the exceptions it throws.
4. Register it in `registry.hpp`.
5. The equivalence suite picks it up automatically and will fail if it is not bit
   identical across backends.
