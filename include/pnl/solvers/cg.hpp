#pragma once

/// \file cg.hpp
/// The conjugate gradient method: the non stationary contrast to the splitting
/// family.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

/// Conjugate gradient.
///
/// Derivation, following Shewchuk. For symmetric positive definite A, solving
/// A x = b is equivalent to minimising the quadratic form
/// f(x) = x^T A x / 2 - b^T x, whose gradient is A x - b, so the residual
/// r = b - A x is the direction of steepest descent. Steepest descent is slow
/// because successive steps undo each other; conjugate gradient fixes this by
/// choosing search directions that are A orthogonal, p_i^T A p_j = 0 for i != j.
/// Under that condition the exact line search along each direction never has to
/// be revisited, so after k steps the iterate is optimal over the whole Krylov
/// subspace span{r_0, A r_0, ..., A^{k-1} r_0}, and in exact arithmetic the
/// method terminates in at most n steps.
///
/// The practical recurrence, which needs only one matrix vector product and two
/// inner products per iteration:
///
///     alpha_k = (r_k^T r_k) / (p_k^T A p_k)
///     x_{k+1} = x_k + alpha_k p_k
///     r_{k+1} = r_k - alpha_k A p_k
///     beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
///     p_{k+1} = r_{k+1} + beta_k p_k
///
/// Convergence. The error in the A norm is bounded by
/// 2 ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^k times its initial value, with
/// kappa the spectral condition number. On the model Poisson problem
/// kappa = O(h^{-2}), so sqrt(kappa) = O(h^{-1}) and the iteration count to a
/// fixed tolerance is O(n) for an n by n grid: the same order as optimally
/// relaxed SOR, but reached without needing to know a relaxation factor in
/// advance. That is the reason it displaced the stationary methods in practice.
///
/// Convergence order: linear in the A norm with rate
/// (sqrt(kappa) - 1) / (sqrt(kappa) + 1), and superlinear in practice once the
/// extreme eigenvalues have been resolved.
///
/// Parallel note. The two inner products per iteration are global reductions,
/// and they are the reason conjugate gradient scales worse than Jacobi on a
/// distributed backend: each one is a synchronisation point that no amount of
/// local work can hide. The sweep driver times the reductions separately for
/// exactly this reason, and the report reports the communication fraction.
///
/// Reference: Shewchuk, "An Introduction to the Conjugate Gradient Method
/// Without the Agonizing Pain", Carnegie Mellon University 1994; Saad,
/// "Iterative Methods for Sparse Linear Systems", 2nd ed., SIAM 2003, chapter
/// 6; Golub and Van Loan, "Matrix Computations", 4th ed., Johns Hopkins 2013,
/// section 11.3.
class ConjugateGradient final : public Solver {
   public:
    [[nodiscard]] std::string_view name() const noexcept override { return "cg"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "Krylov, not a splitting";
    }

    [[nodiscard]] bool applicable_to(const Problem& problem) const override {
        return problem.is_symmetric_positive_definite();
    }

    [[nodiscard]] std::string inapplicable_reason(const Problem&) const override {
        return "conjugate gradient needs a symmetric positive definite operator; the "
               "A orthogonality of the search directions and the energy norm it minimises are "
               "both undefined otherwise";
    }

    /// \throws InvalidArgument if the problem is not symmetric positive
    ///         definite.
    /// \throws NumericalFailure if the curvature p^T A p is not positive, which
    ///         proves the operator is not positive definite whatever it was
    ///         declared to be.
    [[nodiscard]] SolveResult solve(Problem& problem, Backend& backend,
                                    const SolverOptions& options) const override {
        require(problem.is_symmetric_positive_definite(), inapplicable_reason(problem));
        require(options.check_interval >= 1, "check_interval must be at least one");

        SolveResult result;
        result.solution = problem.make_state();
        Vector r = problem.make_state();
        Vector p = problem.make_state();
        Vector ap = problem.make_state();

        const Real rhs_norm = problem.rhs_norm(backend);
        const Real scale = rhs_norm > 0.0 ? rhs_norm : 1.0;

        // r = b - A x, and with a zero initial guess p = r.
        problem.residual(backend, result.solution, r);
        std::copy(r.begin(), r.end(), p.begin());

        Real rr = problem.dot(backend, r, r);
        Real relative_residual = std::sqrt(rr) / scale;
        if (options.record_history) result.residual_history.push_back(relative_residual);

        Diagnostics diagnostics;
        diagnostics.error_estimate = relative_residual;
        const bool to_tolerance = options.mode == RunMode::ToTolerance;

        if (to_tolerance && relative_residual <= options.tolerance) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            result.diagnostics = diagnostics;
            return result;
        }

        ProgressBar bar("cg", options.max_iterations,
                        options.show_progress && backend.is_root());

        Index iteration = 0;
        for (; iteration < options.max_iterations; ++iteration) {
            problem.apply(backend, p, ap);
            const Real curvature = problem.dot(backend, p, ap);

            if (!(curvature > 0.0)) {
                // Not a numerical accident: a non positive curvature is a
                // certificate that the operator is not positive definite.
                diagnostics.reason = StopReason::Breakdown;
                diagnostics.iterations = iteration;
                diagnostics.evaluations = iteration;
                result.diagnostics = diagnostics;
                throw NumericalFailure(
                    "conjugate gradient found a search direction with curvature " +
                    std::to_string(curvature) +
                    ", which proves the operator is not positive definite");
            }

            const Real alpha = rr / curvature;
            problem.axpy(backend, alpha, p, result.solution);
            problem.axpy(backend, -alpha, ap, r);

            const Real rr_next = problem.dot(backend, r, r);
            relative_residual = std::sqrt(rr_next) / scale;
            diagnostics.error_estimate = relative_residual;
            if (options.record_history) result.residual_history.push_back(relative_residual);

            if (!std::isfinite(relative_residual)) {
                diagnostics.reason = StopReason::Diverged;
                ++iteration;
                break;
            }
            if (to_tolerance && relative_residual <= options.tolerance) {
                diagnostics.converged = true;
                diagnostics.reason = StopReason::Converged;
                ++iteration;
                break;
            }

            const Real beta = rr_next / rr;
            rr = rr_next;
            // p = r + beta p.
            problem.xpby(backend, r, beta, p);

            if (options.show_progress && backend.is_root()) {
                char detail[64];
                std::snprintf(detail, sizeof(detail), "relres=%.3e", relative_residual);
                bar.update(iteration + 1, detail);
            }
        }

        bar.finish();
        diagnostics.iterations = iteration;
        diagnostics.evaluations = iteration;
        if (options.mode == RunMode::FixedIterations) {
            diagnostics.converged = false;
            diagnostics.reason = StopReason::IterationCap;
        } else if (!diagnostics.converged && diagnostics.reason != StopReason::Diverged) {
            diagnostics.reason = StopReason::IterationCap;
        }
        result.diagnostics = diagnostics;
        return result;
    }
};

}  // namespace pnl::solvers
