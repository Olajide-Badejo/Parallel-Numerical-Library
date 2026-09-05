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

#include <algorithm>
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
/// \param x    the current iterate.
/// \param work a second buffer of the same length, provided so a sweep that
///             cannot update in place (Jacobi) does not allocate per iteration
///             and does not have to copy its output back over \p x.
/// \return the view that holds the iterate once the sweep has run: \p x for a
///         method that updates in place, \p work for one that writes its result
///         elsewhere. The driver alternates the two buffers from that answer,
///         so no sweep ever moves a full state vector on the calling thread.
///
/// Returning the view rather than advancing a parity counter is deliberate. The
/// in place methods return \p x on every call, so their parity would never
/// advance and a counter would have to encode "this method does not flip" as a
/// separate fact. The view already carries it.
using SweepFunction = std::function<VectorView(VectorView x, VectorView work)>;

/// What one iteration of a method costs, in the two units a result row needs.
///
/// The two are separate because they disagree for the red black methods, and a
/// single number would be ambiguous for exactly those two solvers. See the
/// field documentation on Diagnostics, which is what a reader of the CSV has in
/// front of them.
struct WorkUnit {
    /// Updates per unknown per iteration.
    Index sweeps = 1;
    /// Streams over the state array per iteration.
    Index passes = 1;
};

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

    /// What one iteration of this method costs. Declared here rather than
    /// derived at the call site so that the device path, which runs the same
    /// methods through different kernels, reads the same two numbers as the
    /// host path instead of keeping a second table that can drift.
    [[nodiscard]] virtual WorkUnit work_unit() const noexcept = 0;

    /// The relaxation factor this solver will apply under \p options, or zero
    /// when the method has no relaxation factor at all.
    ///
    /// The sweep driver records this in the `omega` column and leaves the
    /// column empty when it is zero. It used to ask Sor::resolve_relaxation for
    /// every row, so a Jacobi or conjugate gradient row carried a factor that
    /// run never read, and a Richardson or SSOR row carried the SOR optimum
    /// rather than the step those methods actually take. Asking the solver is
    /// the only way to get an answer that is true of the run.
    [[nodiscard]] virtual Real relaxation_factor(const Problem&, const SolverOptions&) const {
        return 0.0;
    }

    /// \throws InvalidArgument if the solver does not apply to this problem.
    /// \throws NumericalFailure on a breakdown of the underlying recurrence.
    [[nodiscard]] virtual SolveResult solve(Problem& problem,
                                            Backend& backend,
                                            const SolverOptions& options) const = 0;
};

namespace detail {

/// Whether the residual is evaluated at the end of iteration \p iteration. The
/// last iteration always checks, so a run never reports a residual it did not
/// measure at the iterate it returns.
[[nodiscard]] inline bool check_due(Index iteration, const SolverOptions& options) noexcept {
    return ((iteration + 1) % options.check_interval == 0) ||
           (iteration + 1 == options.max_iterations);
}

/// What the shared convergence check decided.
enum class CheckOutcome {
    /// Keep iterating.
    Continue,
    /// A non finite residual appeared.
    Diverged,
    /// The tolerance was met.
    Converged,
};

/// The convergence check every stationary method shares.
///
/// Factored out rather than left inline in run_stationary because Richardson
/// does not use the driver: it carries its residual vector from one iteration
/// to the next so that the operator is applied once per iteration instead of
/// twice. Sharing the check keeps the fairness property the driver exists for,
/// that every method stops on the same criterion applied to the same quantity.
/// Threading a residual hint into the driver instead would have reported
/// Richardson's residual at x_k while every other method reports it at x_{k+1}.
///
/// \param relative_residual the residual at the iterate the caller now holds.
[[nodiscard]] inline CheckOutcome apply_check(Real relative_residual,
                                              const SolverOptions& options,
                                              SolveResult& result,
                                              Diagnostics& diagnostics) {
    diagnostics.error_estimate = relative_residual;
    if (options.record_history) result.residual_history.push_back(relative_residual);

    if (!std::isfinite(relative_residual)) {
        diagnostics.reason = StopReason::Diverged;
        return CheckOutcome::Diverged;
    }
    if (options.mode == RunMode::ToTolerance && relative_residual <= options.tolerance) {
        diagnostics.converged = true;
        diagnostics.reason = StopReason::Converged;
        return CheckOutcome::Converged;
    }
    return CheckOutcome::Continue;
}

/// The stop reason every stationary method finishes with.
inline void finalise_reason(Diagnostics& diagnostics, const SolverOptions& options) noexcept {
    if (options.mode == RunMode::FixedIterations) {
        // A fixed run makes no claim about convergence; it measures cost. The
        // reason field says so rather than reporting a misleading cap hit.
        diagnostics.converged = false;
        diagnostics.reason = StopReason::IterationCap;
    } else if (!diagnostics.converged && diagnostics.reason != StopReason::Diverged) {
        diagnostics.reason = StopReason::IterationCap;
    }
}

/// The iteration driver shared by every stationary method.
///
/// Keeping this in one place is what makes the comparison fair: every method
/// measures its residual the same way, stops on the same criterion, records the
/// same history, and pays the same progress reporting overhead.
///
/// Richardson is the one method that runs its own loop, because its sweep
/// already evaluates the residual and paying for a second evaluation here made
/// it 1.5 times more expensive than the method is. It uses check_due and
/// apply_check, so the criterion, the quantity tested and the iterate it is
/// tested at are still this driver's; only the loop around them is its own. See
/// richardson.hpp.
///
/// \param unit the per iteration work unit the caller records in its result
///        row. The driver also charges \p unit.sweeps operator applications per
///        iteration to the evaluation count, on top of the initial residual and
///        of each residual the check interval asks for.
[[nodiscard]] inline SolveResult run_stationary(Problem& problem,
                                                Backend& backend,
                                                const SolverOptions& options,
                                                std::string_view label,
                                                WorkUnit unit,
                                                const SweepFunction& sweep) {
    require(options.max_iterations >= 0, "max_iterations must not be negative");
    require(options.check_interval >= 1, "check_interval must be at least one");
    require(options.tolerance > 0.0, "tolerance must be positive");

    SolveResult result;
    result.solution = problem.make_state();
    Vector work = problem.make_state();
    Vector residual_vector = problem.make_state();

    // The iterate lives in one of two buffers and the sweep says which. An in
    // place sweep returns its first argument, so current never leaves
    // result.solution and spare is never written. Jacobi returns its second, so
    // the two views trade places every iteration and the state vector is never
    // copied. Everything below that reads the iterate must read current: after
    // an odd number of flips result.solution holds the previous one.
    VectorView current = result.solution;
    VectorView spare = work;

    const Real rhs_norm = problem.rhs_norm(backend);
    // A zero right hand side makes the relative residual meaningless, so fall
    // back to the absolute residual and say so through the diagnostics.
    const Real scale = rhs_norm > 0.0 ? rhs_norm : 1.0;

    // The initial residual is an operator application like any other and is
    // counted like any other.
    Index evaluations = 1;
    Real relative_residual = problem.residual(backend, current, residual_vector) / scale;
    if (options.record_history) result.residual_history.push_back(relative_residual);

    Diagnostics diagnostics;
    diagnostics.error_estimate = relative_residual;
    diagnostics.evaluations = evaluations;
    diagnostics.sweeps = unit.sweeps;
    diagnostics.passes = unit.passes;

    const bool to_tolerance = options.mode == RunMode::ToTolerance;
    if (to_tolerance && relative_residual <= options.tolerance) {
        diagnostics.converged = true;
        diagnostics.reason = StopReason::Converged;
        result.diagnostics = diagnostics;
        return result;
    }

    ProgressBar bar(
        std::string(label), options.max_iterations, options.show_progress && backend.is_root());

    Index iteration = 0;
    for (; iteration < options.max_iterations; ++iteration) {
        const VectorView produced = sweep(current, spare);
        evaluations += unit.sweeps;
        if (produced.data() != current.data()) {
            spare = current;
            current = produced;
        }

        if (check_due(iteration, options)) {
            // On current, not on result.solution. Reading the wrong buffer here
            // would test the previous iterate and shift every reported
            // iteration count by one.
            relative_residual = problem.residual(backend, current, residual_vector) / scale;
            ++evaluations;

            if (apply_check(relative_residual, options, result, diagnostics) !=
                CheckOutcome::Continue) {
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
    //
    // The gather goes to current, before the copy. Gathering result.solution
    // after copying into it would work too, but gathering it while the iterate
    // still sits in work would collect each rank's stale rows and hand back an
    // ungathered result on every rank.
    problem.synchronise(backend, current);

    // One copy, at the end, and only when the iterate did not land back in
    // result.solution. The test is on the data pointers: the in place solvers
    // return the same view on every call, so a parity counter would never
    // advance for them and could not distinguish the two cases.
    if (current.data() != result.solution.data()) {
        std::copy(current.begin(), current.end(), result.solution.begin());
    }

    diagnostics.iterations = iteration;
    diagnostics.evaluations = evaluations;
    finalise_reason(diagnostics, options);
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
