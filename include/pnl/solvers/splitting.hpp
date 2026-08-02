#pragma once

/// \file splitting.hpp
/// The matrix splitting framework that every stationary solver in the zoo is an
/// instance of, plus the iteration driver they all share.
///
/// The family tree. Write A = M - N with M invertible. Then A x = b is
/// equivalent to M x = N x + b, and the associated iteration is
///
///     x_{k+1} = M^{-1} (N x_k + b) = x_k + M^{-1} r_k,   r_k = b - A x_k.
///
/// The iteration matrix is G = M^{-1} N = I - M^{-1} A, and the iteration
/// converges for every starting vector exactly when the spectral radius of G is
/// less than one. Every classical method is one choice of M, with A = D + L + U
/// splitting the matrix into its diagonal, strictly lower and strictly upper
/// parts:
///
///     Richardson              M = I / omega
///     Jacobi                  M = D
///     Gauss Seidel forward    M = D + L
///     Gauss Seidel backward   M = D + U
///     symmetric Gauss Seidel  one forward sweep then one backward sweep
///     SOR                     M = D / omega + L
///     block Jacobi            M = block diagonal of A
///     block Gauss Seidel      M = block lower triangle of A
///
/// Conjugate gradient is deliberately not in this list: it is not a stationary
/// method and its optimality argument is different in kind. It lives in cg.hpp
/// and the report presents it as the contrast that makes the family visible.
///
/// References: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, chapter 4; Young, "Iterative Solution of Large Linear Systems",
/// Academic Press 1971; Golub and Van Loan, "Matrix Computations", 4th ed.,
/// Johns Hopkins 2013, chapter 11.

#include <pnl/backend/backend.hpp>
#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/problems/problem.hpp>
#include <pnl/progress.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace pnl::solvers {

using backend::Backend;
using problems::Problem;

/// How the driver decides when to stop.
enum class RunMode {
    /// Iterate until the relative residual falls below the tolerance or the cap
    /// is reached. This is what produces the hardware independent iteration
    /// counts of Section 8.3.
    ToTolerance,
    /// Iterate exactly max_iterations times with no convergence test.
    ///
    /// This exists because the stationary methods need O(n^2) iterations on an
    /// n by n grid, which at 4096 squared is several million sweeps and days of
    /// compute. Per iteration cost is what the bandwidth study actually needs at
    /// those sizes, and it is measured honestly by running a fixed, stated
    /// number of sweeps. Convergence counts are measured at the sizes where
    /// they are affordable and checked against the closed form rates, which is
    /// what makes the extrapolation to large grids legitimate rather than a
    /// guess.
    FixedIterations,
};

/// Options common to every solver.
struct SolverOptions {
    Real tolerance = DEFAULT_TOLERANCE;
    Index max_iterations = DEFAULT_MAX_ITERATIONS;
    RunMode mode = RunMode::ToTolerance;

    /// Iterations between residual evaluations. One means every iteration,
    /// which makes the reported count exact. Larger values trade exactness of
    /// the count for speed and are recorded in the result row so nobody
    /// compares counts taken at different intervals.
    Index check_interval = 1;

    /// Record the relative residual at every check into the result.
    bool record_history = false;

    /// Relaxation factor, or zero to let the solver pick its own.
    ///
    /// Zero rather than one is the default on purpose. Each method's natural
    /// factor is different: SOR wants Young's closed form optimum, SSOR wants
    /// one, and Richardson wants the reciprocal of the Gershgorin bound.
    /// Defaulting to one silently handed Richardson a step eight times too
    /// large on the Poisson operator, where the eigenvalues reach 8, so the
    /// iteration matrix had spectral radius 7 and diverged. Zero means "ask the
    /// solver", which is the only default that is right for all of them.
    Real relaxation = 0.0;

    /// Block count for the block methods. Zero asks the problem for its
    /// natural block count.
    Index block_count = 0;

    /// Show an in run progress bar on the root rank.
    bool show_progress = false;
};

/// One sweep of a stationary method.
///
/// \param x    the current iterate, updated in place.
/// \param work scratch of the same length, provided so a sweep that cannot
///             update in place (Jacobi) does not allocate per iteration.
using SweepFunction = std::function<void(VectorView x, VectorView work)>;

/// Interface implemented by every solver in the zoo.
class Solver {
   public:
    Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;
    virtual ~Solver() = default;

    /// Stable identifier used in result rows, figures and the CLI.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// A one line description of the splitting, used by the CLI listing and by
    /// the generated solver documentation.
    [[nodiscard]] virtual std::string_view splitting() const noexcept = 0;

    /// Whether this solver can be applied to \p problem at all. Conjugate
    /// gradient needs symmetric positive definiteness; the red black methods
    /// need a colouring.
    [[nodiscard]] virtual bool applicable_to(const Problem& problem) const = 0;

    /// Why the solver is not applicable, for a message the user can act on.
    [[nodiscard]] virtual std::string inapplicable_reason(const Problem&) const {
        return "not applicable to this problem";
    }

    /// \throws InvalidArgument if the solver does not apply to this problem.
    /// \throws NumericalFailure on a breakdown of the underlying recurrence.
    [[nodiscard]] virtual SolveResult solve(Problem& problem, Backend& backend,
                                            const SolverOptions& options) const = 0;
};

namespace detail {

/// The iteration driver shared by every stationary method.
///
/// Keeping this in one place is what makes the comparison fair: every method
/// measures its residual the same way, stops on the same criterion, records the
/// same history, and pays the same progress reporting overhead.
[[nodiscard]] inline SolveResult run_stationary(Problem& problem, Backend& backend,
                                                const SolverOptions& options,
                                                std::string_view label,
                                                const SweepFunction& sweep) {
    require(options.max_iterations >= 0, "max_iterations must not be negative");
    require(options.check_interval >= 1, "check_interval must be at least one");
    require(options.tolerance > 0.0, "tolerance must be positive");

    SolveResult result;
    result.solution = problem.make_state();
    Vector work = problem.make_state();
    Vector residual_vector = problem.make_state();

    const Real rhs_norm = problem.rhs_norm(backend);
    // A zero right hand side makes the relative residual meaningless, so fall
    // back to the absolute residual and say so through the diagnostics.
    const Real scale = rhs_norm > 0.0 ? rhs_norm : 1.0;

    Real relative_residual =
        problem.residual(backend, result.solution, residual_vector) / scale;
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

    ProgressBar bar(std::string(label), options.max_iterations,
                    options.show_progress && backend.is_root());

    Index iteration = 0;
    for (; iteration < options.max_iterations; ++iteration) {
        sweep(result.solution, work);

        const bool check = ((iteration + 1) % options.check_interval == 0) ||
                           (iteration + 1 == options.max_iterations);
        if (check) {
            relative_residual =
                problem.residual(backend, result.solution, residual_vector) / scale;
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
            if (options.show_progress && backend.is_root()) {
                char detail[64];
                std::snprintf(detail, sizeof(detail), "relres=%.3e", relative_residual);
                bar.update(iteration + 1, detail);
            }
        }
    }

    bar.finish();

    // The iteration only ever needed a rank's own rows plus a halo. The result
    // has to be complete everywhere, so gather once, here, rather than per
    // sweep where it would swamp the communication measurement.
    problem.synchronise(backend, result.solution);

    diagnostics.iterations = iteration;
    diagnostics.evaluations = iteration;
    if (options.mode == RunMode::FixedIterations) {
        // A fixed run makes no claim about convergence; it measures cost. The
        // reason field says so rather than reporting a misleading cap hit.
        diagnostics.converged = false;
        diagnostics.reason = StopReason::IterationCap;
    } else if (!diagnostics.converged && diagnostics.reason != StopReason::Diverged) {
        diagnostics.reason = StopReason::IterationCap;
    }
    result.diagnostics = diagnostics;
    return result;
}

/// Resolve the block count, defaulting to the problem's natural choice.
[[nodiscard]] inline Index resolve_block_count(const Problem& problem,
                                               const SolverOptions& options) {
    return options.block_count > 0 ? options.block_count : problem.natural_block_count();
}

}  // namespace detail

}  // namespace pnl::solvers
