// SPDX-License-Identifier: MIT
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
#include <cstdint>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace pnl::backend {

/// Result of a host bandwidth probe.
struct StreamResult {
    double gib_per_second = 0.0;
    Index bytes_per_array = 0;
    int repeats = 0;
    int workers = 0;
    /// True only when the loop really issued non temporal stores. The scalar
    /// fallback of measure_host_triad_nontemporal() leaves it false, because a
    /// figure from ordinary stores is the plain probe again and settles
    /// nothing about read for ownership.
    bool nontemporal = false;
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

/// The next 32 byte boundary at or after \p pointer, which is the alignment
/// _mm256_stream_pd requires of its destination.
[[nodiscard]] inline Real* align_to_32(Real* pointer) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return reinterpret_cast<Real*>((address + 31u) & ~static_cast<std::uintptr_t>(31u));
}

/// Measure the host's achieved triad bandwidth with non temporal stores.
///
/// The instrument Section 4.2 asks for. measure_host_triad() above is a plain
/// C++ loop, so the compiler emits ordinary stores into an array the loop never
/// reads. On a write allocate cache each such store misses and fetches the line
/// it is about to overwrite, which would make the loop move 32 bytes per
/// element while the figure it reports declares 24. This version writes through
/// _mm256_stream_pd, which does not fetch the line, so it moves 24 bytes per
/// element for real. Both report against the same declared 24, so the ratio of
/// the two is the ratio of the traffic they actually move: one if the plain
/// loop pays no read for ownership, four thirds if it pays one on every line.
/// Which it is, is the open question, and this ratio is what answers it.
///
/// Reported as an additional entry beside the plain probe and never as a
/// replacement for it; the rule that reads the ratio is pre registered in
/// benchmarks/sweep_matrix.yaml and applied in phase A8b.
///
/// Same arrays, sizes and repetitions as the plain probe, so the two differ in
/// the store instruction and in nothing else.
[[nodiscard]] inline StreamResult measure_host_triad_nontemporal(
    Backend& backend, Index bytes_per_array = 256 * 1024 * 1024, int repeats = 5) {
    const Index n = bytes_per_array / static_cast<Index>(sizeof(Real));
    // Four elements of headroom in each array so the base pointer can be moved
    // up to the next 32 byte boundary; the default allocator only promises 16.
    const auto span = static_cast<std::size_t>(n + 4);
    Vector a_storage(span);
    Vector b_storage(span);
    Vector c_storage(span);
    Real* a = align_to_32(a_storage.data());
    Real* b = align_to_32(b_storage.data());
    Real* c = align_to_32(c_storage.data());

    // First touch in parallel, for the reason the plain probe gives.
    backend.parallel_for(n, [&](Range chunk) {
        for (Index i = chunk.begin; i < chunk.end; ++i) {
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
    });

    const Real q = 3.0;
    bool streamed = false;
    auto triad = [&] {
#if defined(__AVX__)
        streamed = true;
        const __m256d qv = _mm256_set1_pd(q);
        backend.parallel_for(n, [&](Range chunk) {
            Index i = chunk.begin;
            // Peel to a multiple of four, so every streaming store is aligned.
            for (; i < chunk.end && (i & 3) != 0; ++i) a[i] = b[i] + q * c[i];
            for (; i + 4 <= chunk.end; i += 4) {
                const __m256d bv = _mm256_loadu_pd(b + i);
                const __m256d cv = _mm256_loadu_pd(c + i);
                _mm256_stream_pd(a + i, _mm256_add_pd(bv, _mm256_mul_pd(qv, cv)));
            }
            for (; i < chunk.end; ++i) a[i] = b[i] + q * c[i];
        });
        // One fence for the whole loop: non temporal stores are weakly ordered
        // and nothing may read the array until they have drained.
        _mm_sfence();
#else
        // No AVX in this build, so there is no non temporal store to issue.
        // The loop still runs, and the result reports itself as the fallback
        // it is rather than passing an ordinary store off as a streaming one.
        backend.parallel_for(n, [&](Range chunk) {
            for (Index i = chunk.begin; i < chunk.end; ++i) a[i] = b[i] + q * c[i];
        });
#endif
    };

    triad();  // Untimed warm up.

    double best = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        triad();
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        if (seconds > 0.0) {
            // Three arrays at eight bytes, which this loop really moves.
            const double moved = 3.0 * static_cast<double>(n) * sizeof(Real);
            const double gib = moved / seconds / (1024.0 * 1024.0 * 1024.0);
            if (gib > best) best = gib;
        }
    }

    // Keep the compiler from deciding the whole thing was dead.
    if (a[n / 2] == 1.0e300) best += 1.0e-30;

    StreamResult result;
    result.gib_per_second = best;
    result.bytes_per_array = bytes_per_array;
    result.repeats = repeats;
    result.workers = backend.worker_count();
    result.nontemporal = streamed;
    return result;
}

}  // namespace pnl::backend
