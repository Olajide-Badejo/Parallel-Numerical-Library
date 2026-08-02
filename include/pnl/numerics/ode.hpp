#pragma once

/// \file ode.hpp
/// Initial value problem integrators: classical RK4 and adaptive Dormand
/// Prince RK45.
///
/// Dormand Prince is chosen over Fehlberg because its coefficients are fitted
/// to minimise the error of the fifth order formula, so the fifth order result
/// is the one propagated (local extrapolation) and the embedded fourth order
/// formula serves only as the error estimate. Fehlberg optimises the fourth
/// order formula instead, which wastes the better solution.
///
/// Reference: Hairer, Norsett and Wanner, "Solving Ordinary Differential
/// Equations I: Nonstiff Problems", 2nd ed., Springer 1993, sections II.4 and
/// II.5, for the tableau and the step size controller; Dormand and Prince,
/// Journal of Computational and Applied Mathematics 6(1), 1980.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace pnl::numerics {

/// Right hand side of y' = f(t, y). Writes the derivative into \p dydt.
using OdeFunction = std::function<void(Real t, ConstVectorView y, VectorView dydt)>;

/// One recorded point of a solution trajectory.
struct OdeSample {
    Real t = 0.0;
    Vector y;
};

/// Controls for the adaptive integrator.
struct OdeOptions {
    Real absolute_tolerance = 1.0e-10;
    Real relative_tolerance = 1.0e-10;
    Real initial_step = 0.0;  ///< Zero asks the integrator to choose.
    Real min_step = 1.0e-14;
    Real max_step = 0.0;  ///< Zero means the whole interval.
    Index max_steps = 1000000;
    /// Record every accepted step rather than only the endpoint.
    bool record_trajectory = false;
};

/// The result of an integration.
struct OdeResult {
    Vector y;                          ///< Solution at the final time reached.
    Real t = 0.0;                      ///< Final time reached.
    std::vector<OdeSample> trajectory; ///< Populated only when requested.
    Diagnostics diagnostics;
    Index accepted_steps = 0;
    Index rejected_steps = 0;
};

/// Classical fourth order Runge Kutta with a fixed step.
///
/// Convergence order: 4, global. The global error falls by a factor of 16 when
/// the step is halved, which is the property the convergence tests measure.
/// There is no embedded estimate, so error_estimate is left at zero and the
/// diagnostics report only the step count; callers wanting an error estimate
/// use the adaptive integrator instead.
///
/// \throws InvalidArgument if the step count is not positive.
[[nodiscard]] inline OdeResult rk4(const OdeFunction& f, Real t0, ConstVectorView y0, Real t1,
                                   Index steps, bool record_trajectory = false) {
    require(steps > 0, "rk4 needs a positive step count");
    const auto n = static_cast<std::size_t>(y0.size());
    const Real h = (t1 - t0) / static_cast<Real>(steps);

    OdeResult result;
    result.y.assign(y0.begin(), y0.end());
    result.t = t0;

    Vector k1(n), k2(n), k3(n), k4(n), work(n);
    if (record_trajectory) result.trajectory.push_back({t0, result.y});

    for (Index step = 0; step < steps; ++step) {
        const Real t = t0 + static_cast<Real>(step) * h;
        Vector& y = result.y;

        f(t, y, k1);
        for (std::size_t i = 0; i < n; ++i) work[i] = y[i] + 0.5 * h * k1[i];
        f(t + 0.5 * h, work, k2);
        for (std::size_t i = 0; i < n; ++i) work[i] = y[i] + 0.5 * h * k2[i];
        f(t + 0.5 * h, work, k3);
        for (std::size_t i = 0; i < n; ++i) work[i] = y[i] + h * k3[i];
        f(t + h, work, k4);

        for (std::size_t i = 0; i < n; ++i) {
            y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        result.t = t + h;
        if (record_trajectory) result.trajectory.push_back({result.t, y});
    }

    result.accepted_steps = steps;
    result.diagnostics.iterations = steps;
    result.diagnostics.evaluations = 4 * steps;
    result.diagnostics.converged = true;
    result.diagnostics.reason = StopReason::Converged;
    result.diagnostics.error_estimate = 0.0;
    return result;
}

/// Adaptive Dormand Prince 5(4).
///
/// Convergence order: 5 for the propagated solution, with an embedded fourth
/// order formula providing the local error estimate. The step size controller
/// is the standard PI free form h_new = h * min(facmax, max(facmin, fac *
/// (tol/err)^{1/5})) with safety factor 0.9, as in Hairer, Norsett and Wanner
/// section II.4.
///
/// error_estimate is the largest scaled local error over accepted steps, so a
/// value at or below one means every step met the requested tolerance.
///
/// \throws InvalidArgument if the tolerances are not positive.
[[nodiscard]] inline OdeResult dormand_prince(const OdeFunction& f, Real t0, ConstVectorView y0,
                                              Real t1, const OdeOptions& options = {}) {
    require(options.absolute_tolerance > 0.0 && options.relative_tolerance > 0.0,
            "dormand_prince needs positive tolerances");

    // Dormand Prince 5(4) tableau.
    constexpr Real C2 = 1.0 / 5.0, C3 = 3.0 / 10.0, C4 = 4.0 / 5.0, C5 = 8.0 / 9.0;
    constexpr Real A21 = 1.0 / 5.0;
    constexpr Real A31 = 3.0 / 40.0, A32 = 9.0 / 40.0;
    constexpr Real A41 = 44.0 / 45.0, A42 = -56.0 / 15.0, A43 = 32.0 / 9.0;
    constexpr Real A51 = 19372.0 / 6561.0, A52 = -25360.0 / 2187.0, A53 = 64448.0 / 6561.0,
                   A54 = -212.0 / 729.0;
    constexpr Real A61 = 9017.0 / 3168.0, A62 = -355.0 / 33.0, A63 = 46732.0 / 5247.0,
                   A64 = 49.0 / 176.0, A65 = -5103.0 / 18656.0;
    // The fifth order weights, which are also row seven of A: the method is
    // first same as last, so k7 of one step is k1 of the next.
    constexpr Real B1 = 35.0 / 384.0, B3 = 500.0 / 1113.0, B4 = 125.0 / 192.0,
                   B5 = -2187.0 / 6784.0, B6 = 11.0 / 84.0;
    // Fourth order weights, used only for the error estimate.
    constexpr Real E1 = 5179.0 / 57600.0, E3 = 7571.0 / 16695.0, E4 = 393.0 / 640.0,
                   E5 = -92097.0 / 339200.0, E6 = 187.0 / 2100.0, E7 = 1.0 / 40.0;

    const auto n = static_cast<std::size_t>(y0.size());
    const Real span = t1 - t0;
    const Real direction = span >= 0.0 ? 1.0 : -1.0;
    const Real max_step = options.max_step > 0.0 ? options.max_step : std::abs(span);

    OdeResult result;
    result.y.assign(y0.begin(), y0.end());
    result.t = t0;
    if (options.record_trajectory) result.trajectory.push_back({t0, result.y});

    Vector k1(n), k2(n), k3(n), k4(n), k5(n), k6(n), k7(n), work(n), candidate(n);
    Index evaluations = 0;

    f(t0, result.y, k1);
    ++evaluations;

    Real h = options.initial_step > 0.0 ? options.initial_step : std::abs(span) / 100.0;
    h = std::min(h, max_step);
    h = std::max(h, options.min_step);

    Real worst_error = 0.0;
    Index step_count = 0;
    bool converged = false;
    StopReason reason = StopReason::IterationCap;

    while (step_count < options.max_steps) {
        if (std::abs(result.t - t1) <= 1.0e-14 * std::max(Real{1.0}, std::abs(t1))) {
            converged = true;
            reason = StopReason::Converged;
            break;
        }
        // Do not overshoot the endpoint.
        if (direction * (result.t + direction * h - t1) > 0.0) h = std::abs(t1 - result.t);

        const Real t = result.t;
        const Real dh = direction * h;
        const Vector& y = result.y;

        for (std::size_t i = 0; i < n; ++i) work[i] = y[i] + dh * A21 * k1[i];
        f(t + C2 * dh, work, k2);
        for (std::size_t i = 0; i < n; ++i) work[i] = y[i] + dh * (A31 * k1[i] + A32 * k2[i]);
        f(t + C3 * dh, work, k3);
        for (std::size_t i = 0; i < n; ++i) {
            work[i] = y[i] + dh * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        f(t + C4 * dh, work, k4);
        for (std::size_t i = 0; i < n; ++i) {
            work[i] = y[i] + dh * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        f(t + C5 * dh, work, k5);
        for (std::size_t i = 0; i < n; ++i) {
            work[i] = y[i] + dh * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] +
                                   A65 * k5[i]);
        }
        f(t + dh, work, k6);
        for (std::size_t i = 0; i < n; ++i) {
            candidate[i] = y[i] + dh * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] +
                                        B6 * k6[i]);
        }
        f(t + dh, candidate, k7);
        evaluations += 6;

        // Scaled error of the embedded pair, in the norm the controller uses.
        Real error = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const Real fifth = dh * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] +
                                     B6 * k6[i]);
            const Real fourth = dh * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] +
                                      E6 * k6[i] + E7 * k7[i]);
            const Real scale = options.absolute_tolerance +
                               options.relative_tolerance *
                                   std::max(std::abs(y[i]), std::abs(candidate[i]));
            const Real scaled = (fifth - fourth) / scale;
            error += scaled * scaled;
        }
        error = std::sqrt(error / static_cast<Real>(n));

        if (error <= 1.0 || h <= options.min_step) {
            result.t = t + dh;
            result.y = candidate;
            k1 = k7;  // First same as last.
            ++result.accepted_steps;
            worst_error = std::max(worst_error, error);
            if (options.record_trajectory) result.trajectory.push_back({result.t, result.y});
        } else {
            ++result.rejected_steps;
        }

        // Step size controller with the standard safety factor and clamps.
        const Real factor =
            error > 0.0 ? 0.9 * std::pow(1.0 / error, 0.2) : 5.0;
        h *= std::clamp(factor, 0.2, 5.0);
        h = std::clamp(h, options.min_step, max_step);
        ++step_count;
    }

    result.diagnostics.iterations = result.accepted_steps;
    result.diagnostics.evaluations = evaluations;
    result.diagnostics.error_estimate = worst_error;
    result.diagnostics.converged = converged;
    result.diagnostics.reason = reason;
    return result;
}

}  // namespace pnl::numerics
