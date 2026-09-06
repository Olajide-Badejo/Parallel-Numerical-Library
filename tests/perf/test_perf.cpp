// SPDX-License-Identifier: MIT
/// \file test_perf.cpp
/// A relative performance gate: Jacobi on OpenMP must still scale.
///
/// **What this is for, and what it is not.** It cannot measure the target
/// hardware and does not try to. It is a ratio between two runs of the same
/// binary on the same machine in the same minute, so it survives a slow runner,
/// a shared cloud machine and a different processor, and it catches exactly one
/// thing: the class of regression finding 4.1 describes, where the parallel path
/// stops being parallel and every scaling figure in the report quietly becomes a
/// flat line. A serialised backend, a lock added to a sweep, a dispatch that
/// stopped handing work to more than one thread: all of them turn four workers
/// into one worker and all of them are invisible to a correctness suite, because
/// the answers stay bit identical. That is the point. Every other test here
/// passes just as happily on a library that got ten times slower.
///
/// **It is off by default and that is deliberate.** The label is `perf`, the
/// `test` target of the Makefile excludes it with `-LE perf`, and CI runs it in
/// a step of its own. A timing test inside `make test` is a timing test somebody
/// eventually disables, because it is the one that fails on a laptop with a
/// browser open; kept separate, it can be run where a number means something.
///
/// **The configuration, and why each part of it.**
///
///   - 1023 squared, which is 1046529 unknowns and about 8 MB per array. The
///     sweep touches five arrays, so the working set is far outside any cache on
///     this machine and the kernel is bandwidth bound, which is the regime the
///     report is about. A size that fits in cache would measure something else
///     and would scale differently.
///   - 200 fixed iterations with no convergence test, so both runs do exactly
///     the same arithmetic and the comparison is of time and nothing else.
///   - The median of five, which is what the sweep driver reports and is what
///     makes a single interrupted run harmless.
///   - The workspace is allocated once, outside the timed region, exactly as
///     phase A5 requires of every timed path. A gate that allocated a megabyte
///     inside its own measurement would be measuring the allocator.
///   - Both times and the ratio are printed whether the case passes or fails,
///     because a gate that only speaks when it fails tells a reader nothing
///     about the margin it passed by.
///
/// **The threshold is 2.5 at four workers**, against a perfect 4. It is set that
/// low on purpose: this is a bandwidth bound kernel, four threads do not get
/// four times the bandwidth on any machine, and the gate exists to catch a
/// collapse to 1 rather than to police the difference between 3.4 and 3.6.
///
/// The case skips, with a printed reason, on a machine with fewer than four
/// logical processors. There is nothing to measure there and a failure would say
/// only that the runner is small.

#include <pnl/backend/backend.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>
#include <pnl/solvers/workspace.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <pnl_test.hpp>
#include <string>
#include <vector>

using namespace pnl;

namespace {

/// Interior points per side. 1023 squared unknowns, about 8 MB per array.
constexpr Index SIDE = 1023;

/// Fixed sweeps per timed run.
constexpr Index ITERATIONS = 200;

/// Timed repetitions. The median is reported.
constexpr int REPETITIONS = 5;

/// Workers the parallel run uses, and the ratio it must reach against one.
constexpr int PARALLEL_WORKERS = 4;
constexpr double REQUIRED_SPEEDUP = 2.5;

[[nodiscard]] solvers::SolverOptions perf_options() {
    solvers::SolverOptions options;
    options.mode = solvers::RunMode::FixedIterations;
    options.max_iterations = ITERATIONS;
    options.check_interval = 1000000;
    return options;
}

/// Median wall clock seconds of REPETITIONS solves at \p workers.
///
/// The workspace is allocated once and reused, so no repetition pays for a
/// megabyte of first touch inside its own measurement. A5.
[[nodiscard]] double median_seconds(problems::Poisson2D& problem, int workers) {
    backend::Config config;
    config.workers = workers;
    config.reduction = backend::ReductionMode::Deterministic;
    auto execution = backend::make_backend("openmp", config);
    PNL_REQUIRE_MESSAGE(execution->worker_count() == workers,
                        "the openmp backend was asked for " + std::to_string(workers) +
                            " workers and reports " + std::to_string(execution->worker_count()) +
                            "; the ratio below would be between two run counts that are not the "
                            "ones it names");

    auto solver = solvers::make_solver("jacobi");
    solvers::SolverWorkspace workspace = solver->make_workspace(problem);
    const solvers::SolverOptions options = perf_options();

    // One untimed solve first: it faults in the workspace pages and warms the
    // OpenMP team, neither of which is what this measures.
    workspace.reset(problem);
    (void)solver->solve(problem, *execution, options, workspace);

    std::vector<double> seconds;
    seconds.reserve(REPETITIONS);
    for (int repetition = 0; repetition < REPETITIONS; ++repetition) {
        workspace.reset(problem);
        const auto start = std::chrono::steady_clock::now();
        const auto report = solver->solve(problem, *execution, options, workspace);
        const auto stop = std::chrono::steady_clock::now();
        PNL_REQUIRE_MESSAGE(report.diagnostics.iterations == ITERATIONS,
                            "the timed solve ran " + std::to_string(report.diagnostics.iterations) +
                                " iterations rather than the " + std::to_string(ITERATIONS) +
                                " both sides of the ratio are meant to run");
        seconds.push_back(std::chrono::duration<double>(stop - start).count());
    }

    std::sort(seconds.begin(), seconds.end());
    return seconds[seconds.size() / 2];
}

}  // namespace

PNL_TEST("perf/jacobi on openmp is at least 2.5 times faster at four workers than at one") {
    const int available = backend::available_logical_cpus();
    if (available < PARALLEL_WORKERS) {
        std::printf("        skip  this machine has %d logical processors, and the gate needs %d\n",
                    available,
                    PARALLEL_WORKERS);
        return;
    }

    problems::Poisson2D problem(SIDE, problems::PoissonRhs::SpectrallyRich);

    const double one = median_seconds(problem, 1);
    const double many = median_seconds(problem, PARALLEL_WORKERS);
    const double ratio = many > 0.0 ? one / many : 0.0;

    // Printed whether this passes or fails: the margin is the useful number and
    // a gate that speaks only on failure hides it.
    std::printf(
        "        jacobi %lld squared, %lld iterations, median of %d:\n"
        "          1 worker  %.4f s\n"
        "          %d workers %.4f s\n"
        "          ratio     %.2f, required %.2f\n",
        static_cast<long long>(SIDE),
        static_cast<long long>(ITERATIONS),
        REPETITIONS,
        one,
        PARALLEL_WORKERS,
        many,
        ratio,
        REQUIRED_SPEEDUP);

    PNL_REQUIRE_MESSAGE(
        one > 0.0 && many > 0.0,
        "one of the timed runs took no measurable time, so the ratio means nothing");
    PNL_REQUIRE_MESSAGE(
        ratio >= REQUIRED_SPEEDUP,
        "jacobi at " + std::to_string(PARALLEL_WORKERS) + " workers is only " +
            test::format(ratio) + " times faster than at one, against the required " +
            test::format(REQUIRED_SPEEDUP) +
            ". This gate cannot measure the target hardware and does not try to; what it "
            "catches is the parallel path having stopped being parallel, which every "
            "correctness test in this suite passes happily because the answers stay bit "
            "identical.");
}
