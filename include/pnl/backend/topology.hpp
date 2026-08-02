#pragma once

/// \file topology.hpp
/// CPU topology discovery and thread pinning.
///
/// What this file has to work around, stated plainly because it shapes what
/// Objective 3 can honestly claim.
///
/// The host is an Intel Core i7-14700K: 8 performance cores with two threads
/// each and 12 efficiency cores with one, 28 logical processors in total. The
/// guest is WSL2, which is a Hyper-V virtual machine, and it does not pass the
/// heterogeneity through. Inside the guest, sysfs reports 14 uniform cores of
/// two threads each, every logical processor claims the same 2048K L2 and the
/// same cpu_capacity of 1024, and the hybrid feature flag is absent from
/// /proc/cpuinfo. There is therefore no reliable way to ask the operating
/// system which logical processor is a performance core, because the operating
/// system has not been told.
///
/// Worse, sched_setaffinity inside the guest binds a thread to a guest virtual
/// processor, and the hypervisor remains free to schedule that virtual
/// processor onto any host logical processor it likes. Pinning is therefore a
/// hint about guest placement, not a guarantee about silicon.
///
/// Two consequences, both of which the report states rather than papers over:
///
///   1. classify_cpus() measures rather than asks. It times an identical
///      compute kernel on each logical processor in turn and looks for a
///      bimodal distribution. When one exists the classification is reported
///      with its separation; when the timings are unimodal the function says
///      so, and the P versus E labels are not used.
///
///   2. The knee that Objective 3 asks for is taken from the aggregate scaling
///      curve, not from per processor labels. Adding workers one at a time
///      reveals a change of slope wherever the hardware runs out of fast cores,
///      and that measurement survives virtualisation because it depends only on
///      how many cores are engaged, never on which. This is the measurement the
///      report leads with.

#include <pnl/backend/backend.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace pnl::backend {

/// What a probe was able to establish about one logical processor.
struct CpuProbe {
    int cpu = 0;
    /// Throughput relative to the fastest processor measured, in (0, 1].
    double relative_throughput = 0.0;
    /// True when this processor sits in the faster of two clearly separated
    /// groups. Meaningless unless the classification succeeded.
    bool fast_group = false;
};

/// The result of probing every logical processor.
struct TopologyReport {
    int logical_cpus = 0;
    int physical_cores = 0;
    /// Logical processors that are the first thread of their physical core.
    std::vector<int> core_leaders;
    std::vector<CpuProbe> probes;
    /// True only when the measured throughputs separated into two groups by a
    /// margin larger than the measurement noise.
    bool classification_succeeded = false;
    /// Ratio of the slow group mean to the fast group mean, or one when the
    /// classification failed.
    double group_separation = 1.0;
    /// A sentence suitable for printing and for the report.
    std::string verdict;
};

/// Logical processors this process may run on.
[[nodiscard]] inline int available_logical_cpus_impl() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) return count;
    }
#endif
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? static_cast<int>(hardware) : 1;
}

/// Read the thread sibling list of a logical processor and return the lowest
/// numbered sibling, which identifies the physical core.
[[nodiscard]] inline int core_leader_of(int cpu) {
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                             "/topology/thread_siblings_list";
    std::ifstream file(path);
    if (!file) return cpu;
    std::string contents;
    std::getline(file, contents);
    if (contents.empty()) return cpu;
    // The list looks like "6-7" or "6,7"; the first number is the leader.
    std::size_t cut = contents.find_first_of(",-");
    const std::string first = cut == std::string::npos ? contents : contents.substr(0, cut);
    try {
        return std::stoi(first);
    } catch (const std::exception&) {
        return cpu;
    }
}

/// Bind the calling thread to a single logical processor.
///
/// \returns true when the operating system accepted the request.
[[nodiscard]] inline bool pin_this_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

/// Remove any affinity restriction from the calling thread.
inline void unpin_this_thread(int logical_cpus) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = 0; cpu < logical_cpus; ++cpu) CPU_SET(static_cast<unsigned>(cpu), &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)logical_cpus;
#endif
}

/// The logical processors of a physical core, lowest first.
[[nodiscard]] inline std::vector<int> discover_core_leaders(int logical_cpus) {
    std::vector<int> leaders;
    leaders.reserve(static_cast<std::size_t>(logical_cpus));
    for (int cpu = 0; cpu < logical_cpus; ++cpu) {
        if (core_leader_of(cpu) == cpu) leaders.push_back(cpu);
    }
    if (leaders.empty()) {
        for (int cpu = 0; cpu < logical_cpus; ++cpu) leaders.push_back(cpu);
    }
    return leaders;
}

namespace detail {

/// A short, entirely compute bound kernel with a serial dependence chain, so
/// its runtime reflects core speed rather than memory bandwidth or the width of
/// the machine. Marked volatile free but written so no compiler can elide it.
[[nodiscard]] inline double timing_kernel(std::size_t iterations) {
    double a = 1.0000001;
    double b = 0.9999999;
    for (std::size_t i = 0; i < iterations; ++i) {
        a = a * b + 1.0e-12;
        b = b * a + 1.0e-12;
    }
    return a + b;
}

}  // namespace detail

/// Probe every logical processor and try to classify them into two speed
/// groups.
///
/// \param repeats how many timed runs per processor; the best is kept, since
///        the fastest observation is the one least contaminated by the
///        hypervisor scheduling something else onto the same host core.
[[nodiscard]] inline TopologyReport probe_topology(int repeats = 5) {
    TopologyReport report;
    report.logical_cpus = available_logical_cpus_impl();
    report.core_leaders = discover_core_leaders(report.logical_cpus);
    report.physical_cores = static_cast<int>(report.core_leaders.size());

    constexpr std::size_t KERNEL_ITERATIONS = 4000000;
    std::vector<double> best_seconds(static_cast<std::size_t>(report.logical_cpus), 1.0e30);

    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int cpu = 0; cpu < report.logical_cpus; ++cpu) {
            double seconds = 0.0;
            std::thread worker([&] {
                if (!pin_this_thread(cpu)) {
                    seconds = -1.0;
                    return;
                }
                // Warm up so the first timed run is not paying for migration.
                (void)detail::timing_kernel(KERNEL_ITERATIONS / 10);
                const auto start = std::chrono::steady_clock::now();
                const double sink = detail::timing_kernel(KERNEL_ITERATIONS);
                const auto stop = std::chrono::steady_clock::now();
                seconds = std::chrono::duration<double>(stop - start).count();
                // Keep the optimiser honest without perturbing the timing.
                if (sink == 12345.6789) seconds += 1.0e-30;
            });
            worker.join();
            if (seconds > 0.0) {
                best_seconds[static_cast<std::size_t>(cpu)] =
                    std::min(best_seconds[static_cast<std::size_t>(cpu)], seconds);
            }
        }
    }

    double fastest = 1.0e30;
    for (double seconds : best_seconds) {
        if (seconds < fastest && seconds > 0.0) fastest = seconds;
    }
    if (!(fastest < 1.0e30)) {
        report.verdict = "per processor probing failed: affinity was refused on every processor";
        return report;
    }

    report.probes.reserve(static_cast<std::size_t>(report.logical_cpus));
    std::vector<double> throughputs;
    for (int cpu = 0; cpu < report.logical_cpus; ++cpu) {
        const double seconds = best_seconds[static_cast<std::size_t>(cpu)];
        CpuProbe probe;
        probe.cpu = cpu;
        probe.relative_throughput = seconds < 1.0e30 ? fastest / seconds : 0.0;
        report.probes.push_back(probe);
        if (probe.relative_throughput > 0.0) throughputs.push_back(probe.relative_throughput);
    }

    // Look for two groups by splitting at the largest gap in the sorted
    // throughputs. A genuine performance versus efficiency split shows up as a
    // gap far larger than the spread inside either group.
    std::vector<double> sorted = throughputs;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() < 4) {
        report.verdict = "too few processors to attempt a classification";
        return report;
    }

    std::size_t split = 0;
    double widest_gap = 0.0;
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        const double gap = sorted[i] - sorted[i - 1];
        if (gap > widest_gap) {
            widest_gap = gap;
            split = i;
        }
    }

    // Spread within the two candidate groups, for comparison against the gap.
    const double low_spread = sorted[split - 1] - sorted.front();
    const double high_spread = sorted.back() - sorted[split];
    const double worst_spread = std::max(low_spread, high_spread);

    // A split is only believed when the gap dominates the within group spread
    // and is larger than a few percent in absolute terms. Both guards matter:
    // the first rejects noise, the second rejects a technically bimodal but
    // practically irrelevant split.
    constexpr double GAP_DOMINANCE = 2.0;
    constexpr double MINIMUM_GAP = 0.08;
    if (widest_gap > MINIMUM_GAP && widest_gap > GAP_DOMINANCE * worst_spread) {
        const double threshold = 0.5 * (sorted[split - 1] + sorted[split]);
        double fast_sum = 0.0;
        double slow_sum = 0.0;
        int fast_count = 0;
        int slow_count = 0;
        for (auto& probe : report.probes) {
            probe.fast_group = probe.relative_throughput >= threshold;
            if (probe.fast_group) {
                fast_sum += probe.relative_throughput;
                ++fast_count;
            } else {
                slow_sum += probe.relative_throughput;
                ++slow_count;
            }
        }
        report.classification_succeeded = true;
        report.group_separation =
            fast_count > 0 && slow_count > 0
                ? (slow_sum / slow_count) / (fast_sum / fast_count)
                : 1.0;
        report.verdict = "two speed groups found: " + std::to_string(fast_count) +
                         " fast and " + std::to_string(slow_count) +
                         " slow logical processors, slow group at " +
                         std::to_string(report.group_separation * 100.0) +
                         " percent of fast group throughput";
    } else {
        report.verdict =
            "no reliable performance versus efficiency split visible from inside the guest: "
            "the largest gap in per processor throughput was " + std::to_string(widest_gap) +
            " against a within group spread of " + std::to_string(worst_spread) +
            ", so the knee is taken from the aggregate scaling curve instead";
    }
    return report;
}

/// The process wide topology, probed on first use.
///
/// \param need_classification true to run the per processor timing probe, which
///        costs a few seconds; false to return only the cheap sysfs facts.
[[nodiscard]] const TopologyReport& shared_topology(bool need_classification);

/// The logical processor a worker should bind to under a given policy.
///
/// \returns the processor number, or a negative value meaning "do not pin".
[[nodiscard]] inline int cpu_for_worker(Pinning pinning, int worker, int worker_count,
                                        const TopologyReport& topology) {
    if (pinning == Pinning::None) return -1;
    const int logical = topology.logical_cpus > 0 ? topology.logical_cpus : 1;

    switch (pinning) {
        case Pinning::None:
            return -1;
        case Pinning::Compact:
            // Fill logical processors in order, so sibling threads of one
            // physical core are used before moving to the next core.
            return worker % logical;
        case Pinning::Scatter: {
            // One worker per physical core before using any sibling thread.
            const auto& leaders = topology.core_leaders;
            if (leaders.empty()) return worker % logical;
            if (worker < static_cast<int>(leaders.size())) {
                return leaders[static_cast<std::size_t>(worker)];
            }
            // More workers than cores: fall back to filling the siblings.
            return worker % logical;
        }
        case Pinning::PerformanceCores:
        case Pinning::EfficiencyCores: {
            const bool want_fast = pinning == Pinning::PerformanceCores;
            std::vector<int> pool;
            for (const auto& probe : topology.probes) {
                if (topology.classification_succeeded && probe.fast_group == want_fast) {
                    pool.push_back(probe.cpu);
                }
            }
            if (pool.empty()) return -1;
            return pool[static_cast<std::size_t>(worker) % pool.size()];
        }
    }
    (void)worker_count;
    return -1;
}

}  // namespace pnl::backend
