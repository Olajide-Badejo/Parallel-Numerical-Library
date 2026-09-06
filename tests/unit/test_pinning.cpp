// SPDX-License-Identifier: MIT
/// \file test_pinning.cpp
/// The thread to core binding paths, which had no test at all.
///
/// Pinning is one of the swept variables of Objective 3, so a `pinning` column
/// reading `compact` in the published tables is a claim about where the threads
/// ran. Phase A4 made that claim falsifiable, by turning a request that achieved
/// nothing into a loud failure rather than a silent one, and MEAS-08 in the
/// engineering log is the finding. Nothing then tested the mechanism, so the
/// only evidence that a compact run binds anything was that it did not throw.
///
/// Five claims are asserted here, on all three shared memory pools that pin:
///
///   0. **A backend gives the calling thread its affinity mask back.** It is
///      numbered zero because it runs first and because it is what the other
///      four rest on: every pool makes the calling thread worker zero and binds
///      it, none of them used to unbind it, and the leaked mask then made the
///      core classification below fail for a reason that had nothing to do with
///      the machine. MEAS-12, found by this file.
///   1. `compact` and `scatter` at 2 and 4 workers bind every worker, so
///      `pinning_status()` reads exactly `bound`, with no refusal count after a
///      colon. `worse_outcome` orders the four outcomes so that one worker that
///      was refused, or one policy that was not applicable, is enough to change
///      that string, which is what makes the exact comparison meaningful.
///   2. `none` reports `not_requested` and binds nothing, so the four outcomes
///      really are four and not two.
///   3. **`pcore` throws on this machine.** WSL2 is a Hyper-V guest that does
///      not pass the processor heterogeneity of the host through, so sysfs
///      reports fourteen uniform cores where the host has eight performance
///      cores and twelve efficiency cores, and `classify_cpus` cannot find two
///      speed groups. `make_backend` refuses the policy rather than running it
///      as a no operation, which is MEAS-08. The assertion is guarded on the
///      topology report actually saying the classification failed, so this file
///      is also right on a machine where it succeeds; there the same policy has
///      to construct and bind instead, and that is asserted too.
///   4. A pinned backend still computes the same answers. A binding that
///      changed a result would be a defect wearing a performance flag.
///
/// The scatter policy needs the sibling structure and compact does not, so both
/// are swept: on a machine with no `thread_siblings_list` the two collapse to
/// the same placement, and `discover_core_leaders` says so by making every
/// processor its own leader rather than by guessing.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdio>
#include <memory>
#include <pnl_test.hpp>
#include <string>
#include <vector>

using namespace pnl;

namespace {

/// The shared memory backends that pin. The serial backend has one thread and
/// never binds it, the distributed ones bind the rank's own thread and are
/// covered by the MPI suite, and this list is what is left.
[[nodiscard]] std::vector<std::string> pinning_backends() {
    std::vector<std::string> names;
    for (const auto& name : backend::available_backends()) {
        if (name == "openmp" || name == "pthreads" || name == "jthread") names.push_back(name);
    }
    return names;
}

/// Worker counts to bind at. Two and four, as the task asks, and both are
/// within the fourteen cores this guest reports.
constexpr int WORKER_COUNTS[] = {2, 4};

[[nodiscard]] backend::Config pinned_config(int workers, backend::Pinning pinning) {
    backend::Config config;
    config.workers = workers;
    config.pinning = pinning;
    return config;
}

/// True when this machine could not tell its performance cores from its
/// efficiency cores. The probe is the expensive one and runs at most once per
/// process, cached inside shared_topology.
[[nodiscard]] bool classification_failed() {
    return !backend::shared_topology(true).classification_succeeded;
}

}  // namespace

PNL_TEST("pinning/a backend gives the calling thread its affinity mask back") {
    // MEAS-12, and it is the case that found it. Every pool here makes the
    // calling thread worker zero and binds it, which is right; none of them
    // used to hand it back, which is not. The mask of the process outlives the
    // backend, so the next thing to ask this thread anything about processors
    // got one processor: probe_topology reads the calling thread's mask for the
    // processor count and spawns its probe threads from that thread, so the
    // core classification gave up with "too few processors to attempt a
    // classification" and pcore was refused for a reason that had nothing to do
    // with the machine. It is the case below that was reading that, which is
    // why this one runs first.
    //
    // This is asserted through available_logical_cpus_impl() rather than
    // available_logical_cpus(), because the latter is cached once per process
    // and so cannot see the mask change at all, which is part of why nobody
    // noticed.
    const int before = backend::available_logical_cpus_impl();
    PNL_REQUIRE_MESSAGE(before > 1,
                        "this machine reports " + std::to_string(before) +
                            " usable processor, so nothing below can distinguish a leaked "
                            "binding from the mask this process started with");

    for (const auto& backend_name : pinning_backends()) {
        for (const auto policy : {backend::Pinning::Compact, backend::Pinning::Scatter}) {
            for (int workers : WORKER_COUNTS) {
                {
                    auto execution =
                        backend::make_backend(backend_name, pinned_config(workers, policy));
                    // While it exists, the calling thread is bound: it is worker
                    // zero and it does chunk zero's work. That is the behaviour,
                    // not the defect, and asserting it is what stops a future
                    // fix from restoring the mask too early.
                    PNL_REQUIRE_MESSAGE(execution->pinning_status() == "bound",
                                        "the " + backend_name + " backend did not bind");
                }
                const int after = backend::available_logical_cpus_impl();
                PNL_REQUIRE_MESSAGE(
                    after == before,
                    "after building and destroying the " + backend_name + " backend under '" +
                        std::string(backend::to_string(policy)) + "' pinning at " +
                        std::to_string(workers) + " workers, the calling thread can run on " +
                        std::to_string(after) + " processors rather than the " +
                        std::to_string(before) + " it started with");
            }
        }
    }
}

PNL_TEST("pinning/compact and scatter bind every worker on every pool") {
    for (const auto& backend_name : pinning_backends()) {
        for (const auto policy : {backend::Pinning::Compact, backend::Pinning::Scatter}) {
            for (int workers : WORKER_COUNTS) {
                auto execution =
                    backend::make_backend(backend_name, pinned_config(workers, policy));
                const std::string status = execution->pinning_status();
                PNL_REQUIRE_MESSAGE(status == "bound",
                                    "the " + backend_name + " backend asked for '" +
                                        std::string(backend::to_string(policy)) + "' pinning at " +
                                        std::to_string(workers) +
                                        " workers and reported pinning_status '" + status +
                                        "' rather than 'bound'");
                PNL_REQUIRE_MESSAGE(execution->worker_count() == workers,
                                    "the " + backend_name + " backend reported " +
                                        std::to_string(execution->worker_count()) +
                                        " workers where " + std::to_string(workers) +
                                        " were asked for and bound");
            }
        }
    }
}

PNL_TEST("pinning/no pinning reports not_requested rather than bound") {
    for (const auto& backend_name : pinning_backends()) {
        for (int workers : WORKER_COUNTS) {
            auto execution =
                backend::make_backend(backend_name, pinned_config(workers, backend::Pinning::None));
            const std::string status = execution->pinning_status();
            PNL_REQUIRE_MESSAGE(status == "not_requested",
                                "the " + backend_name +
                                    " backend was asked for no pinning and reported '" + status +
                                    "'");
        }
    }
}

PNL_TEST("pinning/a pinned run computes the same iterate as an unpinned one") {
    // Binding a thread to a processor is a placement decision and must not be a
    // numerical one. This is the assertion that a pinning sweep measures speed
    // and nothing else.
    problems::Poisson2D problem(31, problems::PoissonRhs::SpectrallyRich);
    solvers::SolverOptions options;
    options.mode = solvers::RunMode::FixedIterations;
    options.max_iterations = 15;
    options.check_interval = 1000000;

    auto solve_on = [&](const std::string& backend_name, int workers, backend::Pinning policy) {
        auto execution = backend::make_backend(backend_name, pinned_config(workers, policy));
        auto solver = solvers::make_solver("jacobi");
        return solver->solve(problem, *execution, options).solution;
    };

    const Vector reference = solve_on("serial", 1, backend::Pinning::None);
    for (const auto& backend_name : pinning_backends()) {
        for (const auto policy : {backend::Pinning::Compact, backend::Pinning::Scatter}) {
            for (int workers : WORKER_COUNTS) {
                const Vector candidate = solve_on(backend_name, workers, policy);
                PNL_REQUIRE_MESSAGE(test::first_difference(reference, candidate) < 0,
                                    "the " + backend_name + " backend under '" +
                                        std::string(backend::to_string(policy)) + "' pinning at " +
                                        std::to_string(workers) +
                                        " workers differs from serial at " +
                                        test::describe_difference(candidate, reference));
            }
        }
    }
}

PNL_TEST("pinning/pcore is refused on a machine that cannot classify its cores") {
    // A4 and MEAS-08. Under WSL2 the classification fails, and the whole point
    // of the phase was that a policy which cannot be honoured is refused rather
    // than silently ignored. The guard is what makes this test right on both
    // kinds of machine rather than only on this one.
    const backend::TopologyReport& topology = backend::shared_topology(true);
    std::printf("        topology verdict: %s\n", topology.verdict.c_str());

    for (const auto& backend_name : pinning_backends()) {
        for (const auto policy :
             {backend::Pinning::PerformanceCores, backend::Pinning::EfficiencyCores}) {
            if (classification_failed()) {
                // make_backend refuses before any backend is constructed, so
                // the refusal is the same on every backend by construction, and
                // that is asserted rather than assumed.
                PNL_REQUIRE_THROWS(backend::make_backend(backend_name, pinned_config(4, policy)),
                                   BackendFailure);
            } else {
                // The other machine. The policy is available, so it must bind
                // rather than report anything else.
                auto execution = backend::make_backend(backend_name, pinned_config(4, policy));
                const std::string status = execution->pinning_status();
                PNL_REQUIRE_MESSAGE(status == "bound",
                                    "this machine classified its cores, so the " + backend_name +
                                        " backend under '" +
                                        std::string(backend::to_string(policy)) +
                                        "' pinning must bind, and it reported '" + status + "'");
            }
        }
    }
}

PNL_TEST("pinning/the refusal names the policy and says why") {
    // A failure that does not say what went wrong is a failure somebody will
    // work around. This is the message a user sees when they ask for pcore on
    // this machine.
    if (!classification_failed()) {
        std::printf("        skip  this machine classified its cores, so there is no refusal\n");
        return;
    }
    bool threw = false;
    try {
        auto execution =
            backend::make_backend("pthreads", pinned_config(4, backend::Pinning::PerformanceCores));
        (void)execution;
    } catch (const BackendFailure& failure) {
        threw = true;
        const std::string message = failure.what();
        std::printf("        refusal: %s\n", message.c_str());
        PNL_REQUIRE_MESSAGE(message.find("pcore") != std::string::npos,
                            "the refusal does not name the policy: " + message);
        PNL_REQUIRE_MESSAGE(message.find("classification") != std::string::npos,
                            "the refusal does not say what was missing: " + message);
        PNL_REQUIRE_MESSAGE(
            message.find(backend::shared_topology(true).verdict) != std::string::npos,
            "the refusal does not quote the topology verdict: " + message);
    }
    PNL_REQUIRE_MESSAGE(threw, "pcore was accepted on a machine that cannot classify its cores");
}

PNL_TEST("pinning/the four outcomes are four distinct spellings") {
    // pinning_status() is compared as a string above and read as a column in
    // every result row, so the four spellings have to stay four.
    PNL_REQUIRE(backend::to_string(backend::PinOutcome::NotRequested) == "not_requested");
    PNL_REQUIRE(backend::to_string(backend::PinOutcome::Bound) == "bound");
    PNL_REQUIRE(backend::to_string(backend::PinOutcome::NotApplicable) == "not_applicable");
    PNL_REQUIRE(backend::to_string(backend::PinOutcome::Refused) == "refused");

    // And the refusal count is appended only when it is not zero, which is what
    // makes `status == "bound"` above a comparison against a whole outcome
    // rather than against a prefix of one.
    PNL_REQUIRE(backend::pinning_status_text(backend::PinOutcome::Bound, 0) == "bound");
    PNL_REQUIRE(backend::pinning_status_text(backend::PinOutcome::Refused, 3) == "refused:3");

    // worse_outcome is what turns one bad worker into a bad backend.
    using backend::PinOutcome;
    PNL_REQUIRE(backend::worse_outcome(PinOutcome::Bound, PinOutcome::Refused) ==
                PinOutcome::Refused);
    PNL_REQUIRE(backend::worse_outcome(PinOutcome::Refused, PinOutcome::Bound) ==
                PinOutcome::Refused);
    PNL_REQUIRE(backend::worse_outcome(PinOutcome::Bound, PinOutcome::NotApplicable) ==
                PinOutcome::NotApplicable);
    PNL_REQUIRE(backend::worse_outcome(PinOutcome::NotRequested, PinOutcome::Bound) ==
                PinOutcome::Bound);
}
