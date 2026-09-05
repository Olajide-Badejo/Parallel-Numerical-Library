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

    [[nodiscard]] std::string_view splitting() const noexcept override { return "M = I / omega"; }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// One update per unknown, from the single axpy. Two streams over the
    /// array: the residual and that axpy. Before the residual was carried
    /// across iterations there were three, which is where the 1.5x of MEAS-02
    /// came from.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 2}; }

    /// The step omega, which is what M = I / omega makes of the relaxation
    /// factor. Not the SOR optimum, which is what the result row used to record
    /// for this method.
    [[nodiscard]] Real relaxation_factor(const Problem& problem,
                                         const SolverOptions& options) const override {
        return options.relaxation > 0.0 ? options.relaxation : safe_step(problem);
    }

    using Solver::solve;

    /// Richardson does not use the shared iteration driver, and it is the one
    /// method in the zoo that does not.
    ///
    /// The driver evaluates the residual after the sweep, at x_{k+1}, whenever
    /// the check interval fires. Richardson's sweep needs the residual as its
    /// step direction and used to evaluate it again at the start of the next
    /// iteration, at that same iterate. At the default check interval of one
    /// that is two applications of the operator per iteration where the method
    /// needs one, and under the byte model of Section 4.2 the residual moves 24
    /// bytes per unknown against the axpy's 24, so the measured cost was
    /// residual plus axpy plus residual, 72 against a true 48. Richardson was
    /// compared against methods that pay it once, at 1.5 times its own cost.
    /// That is MEAS-02.
    ///
    /// The repair is to carry the residual vector across iterations rather than
    /// to hand the driver a hint. A hint would have to be the residual at x_k,
    /// which the sweep has just used, and the driver would then report
    /// Richardson's residual one iteration behind every other method's, shift
    /// its iteration counts, and break the one property the shared driver
    /// exists to guarantee. So: update from the stored r_k, evaluate r_{k+1}
    /// once, test that when the check is due, and keep the vector for the next
    /// update. One evaluation per iteration, reported at x_{k+1} exactly as
    /// everywhere else, and the test itself is the driver's own through
    /// detail::apply_check.
    ///
    /// The iterates do not change. x_{k+1} = x_k + omega r_k is the same axpy
    /// over the same r_k in the same order as before; the evaluation that
    /// produces r_k has only moved from the top of iteration k to the bottom of
    /// iteration k - 1, and problem.residual is a pure function of the iterate.
    /// The unit, convergence and equivalence tests hold that claim.
    ///
    /// \param options relaxation is the step omega. A non positive value asks
    ///        for the reciprocal of the problem's Gershgorin bound, which is
    ///        always strictly inside the convergence interval 0 < omega < 2 /
    ///        lambda_max because the bound is an overestimate of lambda_max.
    /// \param workspace slot 0 is the iterate and slot 1 the carried residual.
    ///        There is no work buffer: the axpy updates in place, which is why
    ///        this method asks for two vectors where the driver asks for three.
    /// \throws InvalidArgument if omega is not positive.
    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        const Real omega = relaxation_factor(problem, options);
        require(omega > 0.0, "richardson needs a positive step");
        require(options.max_iterations >= 0, "max_iterations must not be negative");
        require(options.check_interval >= 1, "check_interval must be at least one");
        require(options.tolerance > 0.0, "tolerance must be positive");
        require(workspace.state_size() == problem.state_size(),
                "the workspace was sized for a problem with a different state size");
        require(workspace.vector_count() >= WORKSPACE_VECTORS,
                "richardson needs an iterate and a residual vector");

        SolveReport result;
        const VectorView solution = workspace.vector(WORKSPACE_ITERATE);
        const VectorView residual_vector = workspace.vector(RESIDUAL_SLOT);
        result.solution = solution;

        const Real rhs_norm = problem.rhs_norm(backend);
        const Real scale = rhs_norm > 0.0 ? rhs_norm : 1.0;

        // r_0 = b - A x_0. Every later residual is evaluated at the end of the
        // iteration that produced its iterate, so this is the only one outside
        // the loop and the count is one per iteration plus this one.
        Index evaluations = 1;
        Real relative_residual = problem.residual(backend, solution, residual_vector) / scale;
        if (options.record_history) result.residual_history.push_back(relative_residual);

        Diagnostics diagnostics;
        diagnostics.error_estimate = relative_residual;
        diagnostics.evaluations = evaluations;
        const WorkUnit unit = work_unit();
        diagnostics.sweeps = unit.sweeps;
        diagnostics.passes = unit.passes;

        if (options.mode == RunMode::ToTolerance && relative_residual <= options.tolerance) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            result.diagnostics = diagnostics;
            return result;
        }

        ProgressBar bar(
            "richardson", options.max_iterations, options.show_progress && backend.is_root());

        Index iteration = 0;
        for (; iteration < options.max_iterations; ++iteration) {
            // x_{k+1} = x_k + omega r_k, from the residual already in hand.
            problem.axpy(backend, omega, residual_vector, solution);
            // r_{k+1} = b - A x_{k+1}: at once the value the check tests and
            // the direction the next update takes.
            relative_residual = problem.residual(backend, solution, residual_vector) / scale;
            ++evaluations;

            if (detail::check_due(iteration, options)) {
                if (detail::apply_check(relative_residual, options, result, diagnostics) !=
                    detail::CheckOutcome::Continue) {
                    ++iteration;
                    break;
                }
                if (options.show_progress && backend.is_root()) {
                    char progress_detail[64];
                    std::snprintf(
                        progress_detail, sizeof(progress_detail), "relres=%.3e", relative_residual);
                    bar.update(iteration + 1, progress_detail);
                }
            }
        }

        bar.finish();

        // The iteration needs only a rank's own rows plus a halo, so the result
        // is completed once here rather than per sweep. There is no second
        // buffer and so no final copy: the axpy updates in place.
        problem.synchronise(backend, solution);

        diagnostics.iterations = iteration;
        diagnostics.evaluations = evaluations;
        detail::finalise_reason(diagnostics, options);
        result.diagnostics = diagnostics;
        return result;
    }

    /// Slot 1 of the workspace, which for this method is the carried residual
    /// rather than the driver's second iterate buffer.
    static constexpr Index RESIDUAL_SLOT = 1;

    /// Two: the iterate and that residual.
    static constexpr Index WORKSPACE_VECTORS = 2;

    [[nodiscard]] Index workspace_vectors() const noexcept override { return WORKSPACE_VECTORS; }

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
