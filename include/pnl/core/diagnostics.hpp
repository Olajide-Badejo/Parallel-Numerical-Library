#pragma once

/// \file diagnostics.hpp
/// Every numerical routine in this library returns a value together with the
/// evidence for it. Objective 1 of the specification requires value, error
/// estimate, iteration count and converged flag on every result, so no caller
/// can accidentally consume an unconverged answer as though it were converged.

#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <cmath>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pnl {

/// Why an iteration stopped. Reported explicitly so that hitting the cap is
/// never indistinguishable from meeting the tolerance.
enum class StopReason {
    /// The convergence criterion was met.
    Converged,
    /// The iteration cap was reached first.
    IterationCap,
    /// The iteration stagnated: the update stopped changing the iterate to
    /// working precision while the residual was still above tolerance.
    Stagnated,
    /// The recurrence broke down, for example a zero curvature denominator in
    /// conjugate gradient on a matrix that is not positive definite.
    Breakdown,
    /// A non finite value appeared in the iterate or residual.
    Diverged,
};

[[nodiscard]] constexpr std::string_view to_string(StopReason reason) noexcept {
    switch (reason) {
        case StopReason::Converged: return "converged";
        case StopReason::IterationCap: return "iteration_cap";
        case StopReason::Stagnated: return "stagnated";
        case StopReason::Breakdown: return "breakdown";
        case StopReason::Diverged: return "diverged";
    }
    return "unknown";
}

/// The evidence attached to a numerical result.
struct Diagnostics {
    /// Estimated error. Its meaning is documented per routine: an interval half
    /// width for bracketing root finders, an embedded pair difference for
    /// adaptive ODE steps, a Richardson estimate for quadrature, and the
    /// relative residual norm for linear solvers.
    Real error_estimate = 0.0;

    /// Iterations, steps, or subdivisions actually performed.
    Index iterations = 0;

    /// Function or matrix vector product evaluations, which is the honest cost
    /// unit when comparing methods whose per iteration work differs.
    Index evaluations = 0;

    /// True only when StopReason::Converged.
    bool converged = false;

    StopReason reason = StopReason::IterationCap;

    /// Throw if the result did not converge. Callers that must not proceed on a
    /// bad answer call this; the sweep driver does not, because recording a non
    /// convergence is itself a result worth having.
    ///
    /// \throws ConvergenceFailure when converged is false.
    void require_converged(std::string_view what) const {
        if (!converged) {
            throw ConvergenceFailure(std::string(what) + " stopped after " +
                                     std::to_string(iterations) + " iterations with reason " +
                                     std::string(to_string(reason)) + " and error estimate " +
                                     std::to_string(error_estimate));
        }
    }
};

/// A value paired with its diagnostics.
template <typename T>
struct Result {
    T value{};
    Diagnostics diagnostics{};

    [[nodiscard]] bool converged() const noexcept { return diagnostics.converged; }

    /// \throws ConvergenceFailure when the routine did not converge.
    const T& value_or_throw(std::string_view what) const {
        diagnostics.require_converged(what);
        return value;
    }
};

/// Convenience constructor so routines read as `return make_result(x, d);`.
template <typename T>
[[nodiscard]] Result<std::decay_t<T>> make_result(T&& value, Diagnostics diagnostics) {
    return Result<std::decay_t<T>>{std::forward<T>(value), diagnostics};
}

/// The result of a linear solve, which additionally carries the residual
/// history. The history is what the convergence rate figures are fitted to, so
/// it is part of the result rather than something reconstructed later.
struct SolveResult {
    Vector solution;
    Diagnostics diagnostics;

    /// Relative residual after each iteration, index 0 being the initial guess.
    /// Recording is optional because at 4096 squared the history would dominate
    /// the working set; the sweep driver enables it only for the figures.
    std::vector<Real> residual_history;

    [[nodiscard]] bool converged() const noexcept { return diagnostics.converged; }

    /// Asymptotic convergence factor estimated from the last \p window entries
    /// of the history as the geometric mean of successive residual ratios.
    ///
    /// This is the number compared against the spectral radius predicted by
    /// theory in the convergence tests, so it is computed once, here, rather
    /// than re-derived in each test.
    ///
    /// \returns 0 when the history is too short to estimate.
    [[nodiscard]] Real convergence_factor(Index window = 20) const noexcept {
        const Index n = static_cast<Index>(residual_history.size());
        if (n < 3) return 0.0;
        const Index take = window < n - 1 ? window : n - 1;
        const Index first = n - 1 - take;
        const Real numerator = residual_history[static_cast<std::size_t>(n - 1)];
        const Real denominator = residual_history[static_cast<std::size_t>(first)];
        if (!(denominator > 0.0) || !(numerator > 0.0)) return 0.0;
        // Geometric mean of the ratios telescopes to this closed form.
        return std::pow(numerator / denominator, 1.0 / static_cast<Real>(take));
    }
};

}  // namespace pnl
