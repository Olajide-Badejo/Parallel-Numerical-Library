#pragma once

/// \file jacobi.hpp
/// Jacobi iteration.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

/// Jacobi iteration.
///
/// Derivation. Split A = D + L + U and take M = D, so N = -(L + U) and
///
///     x_{k+1} = D^{-1} (b - (L + U) x_k),
///
/// componentwise x_i^{k+1} = (b_i - sum_{j != i} a_ij x_j^k) / a_ii. Every
/// component of the new iterate depends only on the old one, which is what
/// makes the sweep embarrassingly parallel and why its result cannot depend on
/// how the unknowns were partitioned between workers.
///
/// Convergence: guaranteed for strictly diagonally dominant A (Saad 2003,
/// theorem 4.9). On the model Poisson problem the iteration matrix has spectral
/// radius exactly cos(pi h), so the iteration count to a fixed tolerance grows
/// like h^{-2}, that is, like the number of unknowns. This is the method the
/// report uses as the baseline against which every improvement is measured, and
/// it is the method the GPU comparison of Section 8.3 runs, because being
/// perfectly parallel it is the fairest possible case for a wide device.
///
/// Convergence order: linear, rate cos(pi h) on the model problem.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 4.1; Young, "Iterative Solution of Large Linear Systems",
/// Academic Press 1971, chapter 3.
class Jacobi final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "jacobi"; }

    [[nodiscard]] std::string_view splitting() const noexcept override { return "M = D"; }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// One update per unknown in one traversal of the grid: the reference work
    /// unit every other method in the zoo is measured against.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 1}; }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        auto sweep = [&](VectorView x, VectorView work) {
            problem.jacobi_sweep(backend, x, work);
            // The new iterate lands in work, so hand work back and let the
            // driver alternate the two buffers. An earlier version swapped the
            // contents instead, which moved two full state vectors per
            // iteration on the calling thread, in serial, on every backend.
            // That copy is measurement finding MEAS-01.
            return work;
        };
        return detail::run_stationary(
            problem, backend, options, "jacobi", work_unit(), workspace, sweep);
    }
};

}  // namespace pnl::solvers
