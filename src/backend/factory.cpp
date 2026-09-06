// SPDX-License-Identifier: MIT
/// \file factory.cpp
/// The registry lookup behind make_backend, and the shared topology cache.
///
/// The full topology probe times a kernel on every logical processor and costs
/// a few seconds, so it runs at most once per process and only when something
/// actually needs it. A run with no pinning gets the cheap facts (processor
/// count and sibling structure, both of which are plain sysfs reads) and skips
/// the timing entirely.
///
/// What is deliberately not here any more. The `if` chain over five names, the
/// list of names beside it, and the contraction probe. The first two are the
/// registry of `pnl/backend/registry.hpp`, populated by
/// `src/backend/builtin_backends.cpp`. The third moved into the inline
/// `make_backend` in `pnl/backend/backend.hpp`, so that it is compiled with the
/// consumer's flags rather than with this file's; see NUM-05.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/registry.hpp>
#include <pnl/backend/topology.hpp>

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
    return backend_registry().names();
}

namespace detail {

std::unique_ptr<Backend> make_backend_impl(std::string_view name, const Config& config) {
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

    return backend_registry().create(name, config, topology);
}

}  // namespace detail

}  // namespace pnl::backend
