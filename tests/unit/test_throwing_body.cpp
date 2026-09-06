// SPDX-License-Identifier: MIT
/// \file test_throwing_body.cpp
/// What happens when a caller's body throws inside a parallel dispatch.
///
/// Before phase B6 the answer was one of two failures, both of them fatal and
/// neither of them an exception the caller could catch. A body that threw on a
/// spawned worker unwound out of a thread entry point, which is std::terminate
/// by definition. A body that threw on the dispatching thread of the jthread
/// pool skipped the collect barrier that every other worker was already waiting
/// on, so the process deadlocked with no output at all. Both are rows of
/// Section 4.7.
///
/// The contract these cases assert is the one the backends now document: every
/// chunk of the dispatch is attempted, the first exception to be thrown is held
/// and rethrown on the dispatching thread once the workers are back, and the
/// backend is usable afterwards. "Every chunk is attempted" is not a nicety: a
/// worker that stopped early would not reach its barrier, which is the deadlock
/// this file exists to keep closed.
///
/// The CTest entry for this binary carries a 30 second timeout rather than the
/// suite's 900, so a reintroduced deadlock is a failure within the minute
/// instead of a fifteen minute hang.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/core/error.hpp>

#include <atomic>
#include <cstdio>
#include <memory>
#include <pnl_test.hpp>
#include <string>
#include <vector>

using namespace pnl;

namespace {

/// The shared memory pools in this build. The distributed backends are not
/// here: a rank local throw is a different failure mode with a different fix,
/// and the serial backend has no thread for an exception to escape from.
[[nodiscard]] std::vector<std::string> pool_backends() {
    std::vector<std::string> names;
    for (const auto& name : backend::available_backends()) {
        if (name == "serial" || name == "mpi" || name == "hybrid") continue;
        names.push_back(name);
    }
    return names;
}

/// Worker counts. One is the path where the dispatching thread is the only
/// worker, two is the smallest pool, eight is enough that the throwing chunk is
/// several workers away from the one that will rethrow.
constexpr int WORKER_COUNTS[] = {1, 2, 8};

constexpr Index PROBLEM_SIZE = 4096;

struct Outcome {
    bool threw = false;
    std::string message;
    int chunks_run = 0;
    Index chunks_expected = 0;
};

/// Dispatch a parallel_for whose body throws from exactly one chunk.
///
/// \param victim which chunk throws, or -1 for the last one. Chunk zero is the
///        dispatching thread's own on both pools, and the last chunk belongs to
///        a spawned worker whenever there is more than one, so the two values
///        between them cover the deadlock path and the terminate path.
[[nodiscard]] Outcome throwing_dispatch(backend::Backend& execution, Index victim) {
    const backend::Config& config = execution.config();
    const Index chunks = backend::for_chunk_count(
        PROBLEM_SIZE, execution.worker_count(), config.schedule, config.chunks_per_worker);
    const Index index = victim < 0 ? chunks - 1 : victim;
    const Range bad = backend::for_chunk(
        PROBLEM_SIZE, execution.worker_count(), config.schedule, config.chunks_per_worker, index);

    Outcome outcome;
    outcome.chunks_expected = chunks;
    std::atomic<int> ran{0};
    try {
        execution.parallel_for(PROBLEM_SIZE, [&](Range range) {
            if (range.begin == bad.begin) {
                throw NumericalFailure("chunk " + std::to_string(index) + " refused to run");
            }
            ran.fetch_add(1, std::memory_order_relaxed);
        });
    } catch (const NumericalFailure& failure) {
        outcome.threw = true;
        outcome.message = failure.what();
    }
    outcome.chunks_run = ran.load(std::memory_order_relaxed);
    return outcome;
}

/// A dispatch that throws nothing, used to show the backend still works.
[[nodiscard]] Index count_chunks(backend::Backend& execution) {
    std::atomic<int> ran{0};
    execution.parallel_for(PROBLEM_SIZE,
                           [&](Range) { ran.fetch_add(1, std::memory_order_relaxed); });
    return ran.load(std::memory_order_relaxed);
}

}  // namespace

PNL_TEST("throwing_body/a body that throws surfaces on the dispatching thread") {
    for (const auto& name : pool_backends()) {
        for (const int workers : WORKER_COUNTS) {
            backend::Config config;
            config.workers = workers;
            auto execution = backend::make_backend(name, config);
            const std::string where =
                name + " at " + std::to_string(execution->worker_count()) + " workers";

            // Chunk zero is the dispatching thread's own. This is the case that
            // used to deadlock the jthread pool.
            const Outcome first = throwing_dispatch(*execution, 0);
            PNL_REQUIRE_MESSAGE(first.threw, where + " swallowed a throw from chunk zero");
            PNL_REQUIRE_MESSAGE(first.message == "numerical failure: chunk 0 refused to run",
                                where + " changed the message to '" + first.message + "'");
            PNL_REQUIRE_MESSAGE(first.chunks_run == first.chunks_expected - 1,
                                where + " ran " + std::to_string(first.chunks_run) + " of " +
                                    std::to_string(first.chunks_expected - 1) +
                                    " chunks that did not throw");

            // The last chunk belongs to a spawned worker whenever there is one.
            // This is the case that used to call std::terminate.
            const Outcome last = throwing_dispatch(*execution, -1);
            PNL_REQUIRE_MESSAGE(last.threw, where + " swallowed a throw from the last chunk");
            PNL_REQUIRE_MESSAGE(last.message == "numerical failure: chunk " +
                                                    std::to_string(last.chunks_expected - 1) +
                                                    " refused to run",
                                where + " changed the message to '" + last.message + "'");
            PNL_REQUIRE_MESSAGE(last.chunks_run == last.chunks_expected - 1,
                                where + " ran " + std::to_string(last.chunks_run) + " of " +
                                    std::to_string(last.chunks_expected - 1) +
                                    " chunks that did not throw");

            // And the pool is still a pool afterwards.
            PNL_REQUIRE_MESSAGE(count_chunks(*execution) == last.chunks_expected,
                                where + " could not dispatch again after a body threw");
        }
    }
}

PNL_TEST("throwing_body/a reducer that throws surfaces on the dispatching thread") {
    for (const auto& name : pool_backends()) {
        for (const int workers : WORKER_COUNTS) {
            backend::Config config;
            config.workers = workers;
            auto execution = backend::make_backend(name, config);
            const std::string where =
                name + " at " + std::to_string(execution->worker_count()) + " workers";

            const Index chunks = backend::reduction_chunk_count(PROBLEM_SIZE);
            // Chunk one is a spawned worker's whenever there is more than one
            // worker, since the chunk grid is dealt round robin.
            const Index index = chunks > 1 ? 1 : 0;
            const Range bad = backend::reduction_chunk(PROBLEM_SIZE, index);

            bool threw = false;
            std::string message;
            try {
                (void)execution->reduce(PROBLEM_SIZE, 0.0, [&](Range range) -> Real {
                    if (range.begin == bad.begin) {
                        throw NumericalFailure("reduction chunk refused to run");
                    }
                    return 1.0;
                });
            } catch (const NumericalFailure& failure) {
                threw = true;
                message = failure.what();
            }
            PNL_REQUIRE_MESSAGE(threw, where + " swallowed a throw from a reducer");
            PNL_REQUIRE_MESSAGE(message == "numerical failure: reduction chunk refused to run",
                                where + " changed the message to '" + message + "'");

            // Every chunk but the one that threw contributed its 1.0, so the
            // sum names how many of them ran.
            const Real total = execution->reduce(PROBLEM_SIZE, 0.0, [](Range) { return 1.0; });
            PNL_REQUIRE_MESSAGE(total == static_cast<Real>(chunks),
                                where + " could not reduce again after a reducer threw");
        }
    }
}

PNL_TEST("throwing_body/every pool this build has is covered") {
    // BUILD-06: a case that quietly covers nothing is worse than no case. This
    // one names what ran, so a build without OpenMP says so in the output
    // instead of passing the two cases above over one backend.
    const auto names = pool_backends();
    std::string listed;
    for (const auto& name : names) listed += " " + name;
    std::printf("        pools covered:%s\n", listed.c_str());
    PNL_REQUIRE_MESSAGE(names.size() >= 2,
                        "this build has fewer than the two hand written pools:" + listed);
}
