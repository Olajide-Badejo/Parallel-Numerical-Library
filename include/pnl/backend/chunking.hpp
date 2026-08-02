#pragma once

/// \file chunking.hpp
/// The partition policy shared by every backend.
///
/// Two different policies are needed and keeping them apart is the point of
/// this file.
///
/// A parallel_for wants the partition that runs fastest, which for uniform work
/// is one contiguous block per worker. Nothing about the answer depends on it.
///
/// A deterministic reduce wants a partition that does not depend on the worker
/// count at all, because floating point addition is not associative and the
/// grouping of the partials is therefore part of the answer. Fixing the chunk
/// count as a function of the problem size alone means a reduction computed on
/// one thread, on twenty threads, or on a GPU combines exactly the same
/// partials in exactly the same order, and so returns bit identical results.
/// That is what lets the equivalence suite assert exact equality rather than
/// hiding a real disagreement under a tolerance.
///
/// The distributed case is weaker and the specification says so rather than
/// pretending otherwise: rank boundaries are chosen for load balance, so they
/// do not align with the fixed chunk grid, and a reduction over four ranks
/// groups its partials differently from one over two. MPI results therefore
/// agree across rank counts to reduction tolerance, not bitwise, and the MPI
/// tests assert exactly that.

#include <pnl/core/types.hpp>

#include <algorithm>

namespace pnl::backend {

/// Chunk count used by every deterministic reduction.
///
/// Chosen large enough that twenty workers stay balanced and small enough that
/// the final ordered sum over partials is free. It is a compile time constant
/// on purpose: making it configurable would make reproducibility depend on a
/// runtime flag.
inline constexpr Index DETERMINISTIC_CHUNKS = 512;

/// Number of chunks a deterministic reduction over \p n elements uses.
/// Depends on \p n and nothing else.
[[nodiscard]] constexpr Index reduction_chunk_count(Index n) noexcept {
    if (n <= 0) return 0;
    return std::min(DETERMINISTIC_CHUNKS, n);
}

/// Range of deterministic reduction chunk \p k over [0, n).
[[nodiscard]] constexpr Range reduction_chunk(Index n, Index k) noexcept {
    return block_partition(n, reduction_chunk_count(n), k);
}

/// Number of chunks a parallel_for issues for the given policy.
[[nodiscard]] constexpr Index for_chunk_count(Index n, int workers, Schedule schedule,
                                              int chunks_per_worker) noexcept {
    if (n <= 0 || workers <= 0) return 0;
    const Index w = static_cast<Index>(workers);
    if (schedule == Schedule::Static) return std::min(w, n);
    const Index per = static_cast<Index>(std::max(chunks_per_worker, 1));
    return std::min(w * per, n);
}

/// Range of parallel_for chunk \p k.
[[nodiscard]] constexpr Range for_chunk(Index n, int workers, Schedule schedule,
                                        int chunks_per_worker, Index k) noexcept {
    return block_partition(n, for_chunk_count(n, workers, schedule, chunks_per_worker), k);
}

}  // namespace pnl::backend
