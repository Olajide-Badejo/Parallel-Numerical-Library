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
#include <utility>

namespace pnl::backend {

namespace {

// Two caches rather than one, and each is written exactly once.
//
// There used to be a single `cached_topology` that the cheap pass filled and
// the probing pass then reassigned wholesale, under a second once_flag. That is
// a row of Section 4.7: shared_topology hands out a reference into it, every
// backend constructor copies a TopologyReport out of that reference, and the
// reassignment frees the vectors a copy in progress on another thread is
// reading. It is also a plain surprise on one thread, since a caller who took
// the cheap report and read its verdict later found the sentence had changed
// underneath them.
//
// Splitting the two makes each object immutable once its call_once has run.
// call_once is the happens before edge, so a reader that arrives through it can
// never see a partly written report, and a reference handed out earlier keeps
// describing what it described.
//
// The split changes no answer. The cheap report is what a run without a core
// classification needs: `cpu_for_worker` reads only logical_cpus and
// core_leaders for the none, compact and scatter policies, and the two policies
// that read `probes` are exactly the two that ask for the probing pass.
std::once_flag cheap_once;
std::once_flag probe_once;
TopologyReport cheap_topology;
TopologyReport probed_topology;

/// Facts that cost nothing: how many processors, and which of them lead a
/// physical core.
void fill_cheap_topology() {
    cheap_topology.logical_cpus = available_logical_cpus_impl();
    cheap_topology.core_leaders = discover_core_leaders(cheap_topology.logical_cpus);
    cheap_topology.physical_cores = static_cast<int>(cheap_topology.core_leaders.size());
    cheap_topology.verdict = "not probed";
}

/// The timing probe, and the cheap facts when it could not improve on them.
void fill_probed_topology() {
    std::call_once(cheap_once, fill_cheap_topology);
    TopologyReport probed = probe_topology();
    // Keep the cheap facts if probing could not improve on them, but take the
    // verdict either way: it is the only sentence that says why there is
    // nothing better, and make_backend_impl below quotes it when it refuses a
    // policy. Without this the refusal read "this machine did not yield one:
    // not probed", which describes the cheap path rather than the reason, and
    // on a platform with no affinity interface at all it was actively wrong.
    if (!probed.probes.empty()) {
        probed_topology = std::move(probed);
        return;
    }
    probed_topology = cheap_topology;
    if (!probed.verdict.empty()) probed_topology.verdict = probed.verdict;
}

}  // namespace

int available_logical_cpus() {
    std::call_once(cheap_once, fill_cheap_topology);
    return cheap_topology.logical_cpus;
}

/// The process wide topology, probed on first use if \p need_classification.
///
/// The returned reference is to an object that is written once and never again,
/// so it stays valid and stays accurate for the life of the process.
const TopologyReport& shared_topology(bool need_classification) {
    if (need_classification) {
        std::call_once(probe_once, fill_probed_topology);
        return probed_topology;
    }
    std::call_once(cheap_once, fill_cheap_topology);
    return cheap_topology;
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
