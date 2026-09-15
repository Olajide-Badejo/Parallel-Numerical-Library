// SPDX-License-Identifier: MIT
#pragma once

/// \file timed_solve.hpp
/// The one timed region in this repository.
///
/// Every `seconds_median`, `seconds_min`, `seconds_max` and `seconds_reps`
/// figure in a result row is the elapsed time of the region in
/// timed_repetition() below, and the allocation gate of
/// tests/unit/test_no_allocation.cpp observes that same region through the same
/// function. Having one of them rather than one in the driver and a similar one
/// in the test is the point: a test that asserts "the timed path allocates
/// nothing" is only worth having if the path it drives is the path that is
/// timed, and the only way to make that true rather than approximately true is
/// for there to be a single place where the clock is read.
///
/// What is inside the region and what is not. Inside: the barrier that makes a
/// distributed repetition start together, the solve, and the barrier that makes
/// it end together. Outside: resetting the workspace to the problem's initial
/// state, which is a write over memory this process already owns and is the
/// price of reusing the workspace rather than allocating one per repetition.
/// See MEAS-10 and phase A5.

#include <pnl/backend/backend.hpp>
#include <pnl/core/types.hpp>
#include <pnl/problems/problem.hpp>
#include <pnl/solvers/splitting.hpp>
#include <pnl/solvers/workspace.hpp>

#include <chrono>

namespace pnl::bench {

/// Seconds on a steady clock. The only clock this repository times with.
[[nodiscard]] inline double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Called at the two ends of the timed region, with true on entry and false on
/// exit. Null on every measured path; the allocation gate is what it exists for.
using TimedRegionObserver = void (*)(bool inside);

/// One timed repetition and what the solve it timed reported.
struct TimedRepetition {
    /// Elapsed seconds of the timed region.
    double seconds = 0.0;
    /// The solve's report. Its `solution` is a view into the workspace, so it
    /// is valid until the workspace is reset or destroyed.
    solvers::SolveReport report;
};

/// Run one repetition of \p solver and time it.
///
/// \param workspace allocated once by the caller, outside its repetition loop,
///        and reset here before the clock starts. Sized by
///        Solver::make_workspace for this solver and this problem.
/// \param observer optional, invoked just inside each end of the timed region.
///
/// \throws whatever Solver::solve throws.
[[nodiscard]] inline TimedRepetition timed_repetition(const solvers::Solver& solver,
                                                      problems::Problem& problem,
                                                      backend::Backend& backend,
                                                      const solvers::SolverOptions& options,
                                                      solvers::SolverWorkspace& workspace,
                                                      TimedRegionObserver observer = nullptr) {
    // Outside the region: a repetition must start from the initial state, and
    // putting it back is a write, not an allocation.
    workspace.reset(problem);

    TimedRepetition repetition;
    const double start = now_seconds();
    if (observer != nullptr) observer(true);

    backend.barrier();
    repetition.report = solver.solve(problem, backend, options, workspace);
    backend.barrier();

    if (observer != nullptr) observer(false);
    repetition.seconds = now_seconds() - start;
    return repetition;
}

}  // namespace pnl::bench
