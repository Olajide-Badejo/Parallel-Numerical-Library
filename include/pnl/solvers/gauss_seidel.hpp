#pragma once

/// \file gauss_seidel.hpp
/// The Gauss Seidel family: forward, backward, symmetric, and red black.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

using problems::Colour;
using problems::Sweep;

/// Forward Gauss Seidel.
///
/// Derivation. Take M = D + L, the lower triangle including the diagonal. Then
/// N = -U and the iteration is
///
///     (D + L) x_{k+1} = b - U x_k,
///
/// which is solved by forward substitution, so componentwise
///
///     x_i^{k+1} = (b_i - sum_{j<i} a_ij x_j^{k+1} - sum_{j>i} a_ij x_j^k) / a_ii.
///
/// The new iterate is used as soon as it is available, which is why the method
/// converges faster than Jacobi and why the sweep is sequentially dependent.
///
/// Convergence: for a consistently ordered matrix with property A, of which the
/// five point stencil in natural ordering is the archetype, the Gauss Seidel
/// iteration matrix has spectral radius exactly the square of the Jacobi one
/// (Young 1971, theorem 4.3). Gauss Seidel therefore needs asymptotically half
/// as many iterations as Jacobi, which the convergence tests check as a ratio
/// rather than take on trust.
///
/// Convergence order: linear, rate cos^2(pi h) on the model problem.
///
/// Parallel note. The dependence is real, not an artefact: unknown i reads
/// unknown i-1 of the same sweep. This implementation preserves exact natural
/// ordering on every backend, including across MPI ranks, where it becomes the
/// classical pipelined Gauss Seidel. The consequence is that it does not speed
/// up, and the report measures that instead of substituting a reordering the
/// caller did not ask for. The red black variant below is the reordering, made
/// explicit and costed.
class GaussSeidelForward final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "gauss_seidel_f"; }

    [[nodiscard]] std::string_view splitting() const noexcept override { return "M = D + L"; }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// One in place update per unknown in one traversal.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 1}; }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        auto sweep = [&](VectorView x, VectorView) {
            problem.relaxation_sweep(backend, x, 1.0, Sweep::Forward);
            // Natural ordering relaxation is in place, so the iterate stays in
            // the buffer it arrived in and the driver never flips.
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "gauss_seidel_f", work_unit(), workspace, sweep);
    }
};

/// Backward Gauss Seidel: M = D + U, the same iteration run in descending index
/// order. On a symmetric matrix its iteration matrix is similar to the transpose
/// of the forward one, so it has the identical spectral radius and converges at
/// the identical rate; the two differ in how they propagate information across
/// the grid, which is what makes the symmetric combination below worth having.
class GaussSeidelBackward final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "gauss_seidel_b"; }

    [[nodiscard]] std::string_view splitting() const noexcept override { return "M = D + U"; }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// The same work as the forward sweep, run in descending index order.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 1}; }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        auto sweep = [&](VectorView x, VectorView) {
            problem.relaxation_sweep(backend, x, 1.0, Sweep::Backward);
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "gauss_seidel_b", work_unit(), workspace, sweep);
    }
};

/// Symmetric Gauss Seidel: one forward sweep followed by one backward sweep.
///
/// Derivation. The composite iteration matrix is the product of the backward
/// and forward ones, and the corresponding preconditioner
/// M = (D + L) D^{-1} (D + U) is symmetric whenever A is. That symmetry is the
/// reason the method exists: it is the standard smoother inside multigrid and
/// the standard symmetric preconditioner for conjugate gradient, neither of
/// which can accept the unsymmetric single sweep.
///
/// Cost note: one iteration performs two full sweeps, so a comparison against
/// Jacobi or single sweep Gauss Seidel by iteration count alone flatters it by a
/// factor of two. The result row's `sweeps` column reads 2 and its `passes`
/// column reads 2, and `updates_per_second` is derived from `sweeps`, so the
/// report compares on equal work. Until phase A2 those columns did not exist
/// and this note described a mitigation that was not there; see MEAS-03.
class GaussSeidelSymmetric final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "gauss_seidel_s"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = (D + L) D^-1 (D + U)";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// Two updates per unknown in two traversals: a full forward sweep and then
    /// a full backward one. Not to be confused with the red black pair below,
    /// which is two traversals but one update per unknown.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {2, 2}; }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        auto sweep = [&](VectorView x, VectorView) {
            problem.relaxation_sweep(backend, x, 1.0, Sweep::Forward);
            problem.relaxation_sweep(backend, x, 1.0, Sweep::Backward);
            // Two full sweeps, both in place, so still the same buffer.
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "gauss_seidel_s", work_unit(), workspace, sweep);
    }
};

/// Red black Gauss Seidel.
///
/// Derivation. The graph of the five point stencil is bipartite: colour cell
/// (i,j) red when i+j is even and black when it is odd, and every red cell has
/// only black neighbours. Reordering the unknowns by colour puts A in the block
/// form [[D_r, C], [C^T, D_b]] with both diagonal blocks diagonal, so a sweep
/// over all red cells is a Jacobi update within that colour, and likewise for
/// black. Each half sweep is therefore fully parallel, and the pair together is
/// exactly Gauss Seidel in the red black ordering.
///
/// The cost of the reordering is a different, generally slightly larger
/// iteration count than natural ordering, because the ordering changes the
/// iteration matrix. The specification requires that penalty to be measured
/// rather than asserted, and the convergence tests record it.
///
/// This is the variant the GPU comparison uses, since natural ordering Gauss
/// Seidel has no parallelism to offer a wide device at all.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 12.4; Hager and Wellein, "Introduction to High
/// Performance Computing for Scientists and Engineers", CRC 2010, chapter 6.
class GaussSeidelRedBlack final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "gauss_seidel_rb"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = D + L in red black ordering";
    }

    [[nodiscard]] bool applicable_to(const Problem& problem) const override {
        return problem.supports_colouring();
    }

    [[nodiscard]] std::string inapplicable_reason(const Problem&) const override {
        return "red black ordering needs a bipartite coupling graph, which the five point "
               "stencil has and a general dense system does not";
    }

    /// One update per unknown, in two passes over memory. `coloured_sweep`
    /// steps `j += 2`, so each colour writes half the unknowns and red plus
    /// black is exactly one update each: this is one sweep of work, not two.
    /// The two colour passes are still two strided traversals of the whole
    /// array, which is a traffic question rather than a work question, and that
    /// is the entire reason the two fields are separate.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 2}; }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        // The reason is built only when it is needed: it is a sentence long,
        // so constructing it unconditionally allocated a string on every solve,
        // inside the timed region. See MEAS-10.
        if (!problem.supports_colouring()) require(false, inapplicable_reason(problem));
        auto sweep = [&](VectorView x, VectorView) {
            problem.coloured_sweep(backend, x, 1.0, Colour::Red);
            problem.coloured_sweep(backend, x, 1.0, Colour::Black);
            // Each colour is updated in place; the second reads what the first
            // wrote, which is what makes the pair Gauss Seidel rather than
            // Jacobi, and it is why this method has no second buffer to return.
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "gauss_seidel_rb", work_unit(), workspace, sweep);
    }
};

}  // namespace pnl::solvers
