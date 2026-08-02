#pragma once

/// \file stream_probe.hpp
/// The host's achieved memory bandwidth, measured the same way as the device's.
///
/// Both halves of the Section 8.3 comparison divide by a bandwidth this machine
/// actually reached, never by a specification sheet figure. For that to be a
/// fair ratio the two probes have to measure the same thing, so this is the
/// same STREAM triad kernel as `stream_probe.cu`: a[i] = b[i] + q c[i], two
/// reads and one write per element, no reuse.
///
/// It runs over the execution backend rather than on one thread, because a
/// single core cannot saturate the memory system of a twenty core part and
/// dividing by a single core figure would make every parallel result look
/// superlinear.
///
/// Reference: McCalpin, "Memory Bandwidth and Machine Balance in Current High
/// Performance Computers", IEEE TCCA Newsletter, December 1995.

#include <pnl/backend/backend.hpp>
#include <pnl/core/types.hpp>

#include <chrono>
#include <vector>

namespace pnl::backend {

/// Result of a host bandwidth probe.
struct StreamResult {
    double gib_per_second = 0.0;
    Index bytes_per_array = 0;
    int repeats = 0;
    int workers = 0;
};

/// Measure the host's achieved triad bandwidth.
///
/// \param backend the execution backend, so the probe uses the same
///        parallelism as the sweeps it will normalise.
/// \param bytes_per_array size of each of the three arrays. Should comfortably
///        exceed the last level cache, or the probe measures cache rather than
///        memory; the default is 256 MiB against this machine's 33 MiB L3.
/// \param repeats timed repetitions; the best is kept, being the one least
///        contaminated by anything else running.
[[nodiscard]] inline StreamResult measure_host_triad(Backend& backend,
                                                     Index bytes_per_array = 256 * 1024 * 1024,
                                                     int repeats = 5) {
    const Index n = bytes_per_array / static_cast<Index>(sizeof(Real));
    Vector a(static_cast<std::size_t>(n));
    Vector b(static_cast<std::size_t>(n));
    Vector c(static_cast<std::size_t>(n));

    // First touch in parallel so the pages land near the worker that will use
    // them. Initialising serially would place every page on one memory domain
    // and understate the achievable bandwidth.
    backend.parallel_for(n, [&](Range chunk) {
        for (Index i = chunk.begin; i < chunk.end; ++i) {
            const auto k = static_cast<std::size_t>(i);
            a[k] = 0.0;
            b[k] = 1.0;
            c[k] = 2.0;
        }
    });

    const Real q = 3.0;
    auto triad = [&] {
        backend.parallel_for(n, [&](Range chunk) {
            Real* ap = a.data();
            const Real* bp = b.data();
            const Real* cp = c.data();
            for (Index i = chunk.begin; i < chunk.end; ++i) ap[i] = bp[i] + q * cp[i];
        });
    };

    triad();  // Untimed warm up.

    double best = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        triad();
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        if (seconds > 0.0) {
            const double moved = 3.0 * static_cast<double>(n) * sizeof(Real);
            const double gib = moved / seconds / (1024.0 * 1024.0 * 1024.0);
            if (gib > best) best = gib;
        }
    }

    // Keep the compiler from deciding the whole thing was dead.
    if (a[static_cast<std::size_t>(n / 2)] == 1.0e300) best += 1.0e-30;

    StreamResult result;
    result.gib_per_second = best;
    result.bytes_per_array = bytes_per_array;
    result.repeats = repeats;
    result.workers = backend.worker_count();
    return result;
}

}  // namespace pnl::backend
