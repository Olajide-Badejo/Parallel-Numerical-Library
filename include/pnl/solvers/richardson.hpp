#pragma once

/// \file richardson.hpp
/// Richardson iteration, the simplest member of the splitting family.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

/// Richardson iteration.
///
/// Derivation. Take M = I / omega, so N = M - A = I / omega - A and the
/// iteration x_{k+1} = M^{-1}(N x_k + b) collapses to
///
///     x_{k+1} = x_k + omega (b - A x_k) = x_k + omega r_k.
///
/// The iteration matrix is G = I - omega A. For a symmetric positive definite A
/// with eigenvalues in [lambda_min, lambda_max], the eigenvalues of G are
/// 1 - omega lambda, so the method converges precisely when
/// 0 < omega < 2 / lambda_max, and the spectral radius is minimised at
/// omega = 2 / (lambda_min + lambda_max), where it equals
/// (kappa - 1) / (kappa + 1) with kappa the spectral condition number.
///
/// Convergence order: linear, with asymptotic rate equal to that spectral
/// radius. On the model Poisson problem lambda_max is close to 8 and lambda_min
/// is O(h^2), so the optimal rate is 1 - O(h^2): the same order of slowness as
/// Jacobi, which is no accident, because with the constant diagonal 4 of the
/// five point stencil Jacobi is exactly Richardson at omega = 1/4.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 4.1.
class Richardson final : public Solver {
   public:
    [[nodiscard]] std::string_view name() const noexcept override { return "richardson"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = I / omega";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// \param options relaxation is the step omega. A non positive value asks
    ///        for the reciprocal of the problem's Gershgorin bound, which is
    ///        always strictly inside the convergence interval 0 < omega < 2 /
    ///        lambda_max because the bound is an overestimate of lambda_max.
    /// \throws InvalidArgument if omega is not positive.
    [[nodiscard]] SolveResult solve(Problem& problem, Backend& backend,
                                    const SolverOptions& options) const override {
        const Real omega =
            options.relaxation > 0.0 ? options.relaxation : safe_step(problem);
        require(omega > 0.0, "richardson needs a positive step");

        Vector residual_vector = problem.make_state();
        auto sweep = [&](VectorView x, VectorView) {
            problem.residual(backend, x, residual_vector);
            problem.axpy(backend, omega, residual_vector, x);
        };
        return detail::run_stationary(problem, backend, options, "richardson", sweep);
    }

    /// The step this solver uses when none was given.
    ///
    /// An earlier version estimated lambda_max by power iteration and diverged,
    /// because the Rayleigh quotient converges to lambda_max from below and the
    /// resulting omega therefore sat outside the convergence interval whenever
    /// the estimate had not yet converged. Gershgorin's bound is an
    /// overestimate by construction, so 1 / bound is always admissible. It is
    /// also free, deterministic, and identical on every backend, which the
    /// power iteration was not.
    [[nodiscard]] static Real safe_step(const Problem& problem) noexcept {
        const Real bound = problem.gershgorin_bound();
        return bound > 0.0 ? 1.0 / bound : 1.0;
    }
};

}  // namespace pnl::solvers
