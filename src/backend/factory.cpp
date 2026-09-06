// SPDX-License-Identifier: MIT
/// \file factory.cpp
/// Construction of backends by name, and the shared topology cache.
///
/// The full topology probe times a kernel on every logical processor and costs
/// a few seconds, so it runs at most once per process and only when something
/// actually needs it. A run with no pinning gets the cheap facts (processor
/// count and sibling structure, both of which are plain sysfs reads) and skips
/// the timing entirely.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/jthread_pool.hpp>
#include <pnl/backend/pthreads.hpp>
#include <pnl/backend/serial.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/contract.hpp>

#if defined(PNL_WITH_OPENMP)
#include <pnl/backend/openmp.hpp>
#endif

#if defined(PNL_WITH_MPI)
#include <pnl/backend/hybrid.hpp>
#include <pnl/backend/mpi.hpp>
#endif

#include <mutex>
#include <string>

namespace pnl::backend {

namespace {

std::once_flag cheap_once;
std::once_flag probe_once;
TopologyReport cached_topology;

/// Facts that cost nothing: how many processors, and which of them lead a
/// physical core.
void fill_cheap_topology() {
    cached_topology.logical_cpus = available_logical_cpus_impl();
    cached_topology.core_leaders = discover_core_leaders(cached_topology.logical_cpus);
    cached_topology.physical_cores = static_cast<int>(cached_topology.core_leaders.size());
    cached_topology.verdict = "not probed";
}

/// Run the contraction probe once per process.
///
/// A function local static gives thread safe initialisation and, more usefully,
/// the right behaviour on failure: a throw during initialisation leaves the
/// static uninitialised, so the next call runs the probe again rather than
/// treating a check that never completed as one that passed.
void verify_numerical_contract() {
    static const bool checked = [] {
        assert_no_contraction();
        return true;
    }();
    (void)checked;
}

}  // namespace

int available_logical_cpus() {
    std::call_once(cheap_once, fill_cheap_topology);
    return cached_topology.logical_cpus;
}

/// The process wide topology, probed on first use if \p need_classification.
const TopologyReport& shared_topology(bool need_classification) {
    std::call_once(cheap_once, fill_cheap_topology);
    if (need_classification) {
        std::call_once(probe_once, [] {
            TopologyReport probed = probe_topology();
            // Keep the cheap facts if probing could not improve on them.
            if (!probed.probes.empty()) cached_topology = probed;
        });
    }
    return cached_topology;
}

std::vector<std::string> available_backends() {
    std::vector<std::string> names{"serial"};
#if defined(PNL_WITH_OPENMP)
    names.emplace_back("openmp");
#endif
    names.emplace_back("pthreads");
    names.emplace_back("jthread");
#if defined(PNL_WITH_MPI)
    names.emplace_back("mpi");
#if defined(PNL_WITH_OPENMP)
    names.emplace_back("hybrid");
#endif
#endif
    return names;
}

std::unique_ptr<Backend> make_backend(std::string_view name, const Config& config) {
    // Before anything else. Every path that runs numerics constructs a backend
    // first, so this is the one funnel every caller passes through, and a build
    // whose arithmetic is not the arithmetic the library promises should fail
    // here rather than produce plausible numbers nobody can reproduce.
    verify_numerical_contract();

    // Only the two core classification policies need the timing probe.
    const bool need_classification =
        config.pinning == Pinning::PerformanceCores || config.pinning == Pinning::EfficiencyCores;
    const TopologyReport& topology = shared_topology(need_classification);

    // A policy whose classification this machine did not yield is refused here
    // rather than quietly ignored by every backend in turn. Under WSL2 that is
    // the normal answer for both core policies, and it used to produce a run
    // that pinned nothing and a row that said it had pinned. MEAS-08.
    if (need_classification && !topology.classification_succeeded) {
        throw BackendFailure("pinning policy '" + std::string(to_string(config.pinning)) +
                             "' needs a performance core classification and this machine did not "
                             "yield one: " +
                             topology.verdict);
    }

    if (name == "serial") return std::make_unique<SerialBackend>(config, topology);
    if (name == "pthreads") return std::make_unique<PthreadsBackend>(config, topology);
    if (name == "jthread") return std::make_unique<JthreadBackend>(config, topology);

#if defined(PNL_WITH_OPENMP)
    if (name == "openmp") return std::make_unique<OpenMpBackend>(config, topology);
#endif

#if defined(PNL_WITH_MPI)
    if (name == "mpi") return std::make_unique<MpiBackend>(config, topology);
#if defined(PNL_WITH_OPENMP)
    if (name == "hybrid") return std::make_unique<HybridBackend>(config, topology);
#endif
#endif

    std::string known;
    for (const auto& candidate : available_backends()) {
        if (!known.empty()) known += ", ";
        known += candidate;
    }
    throw InvalidArgument("backend '" + std::string(name) +
                          "' is not available in this build; available backends are " + known);
}

}  // namespace pnl::backend
