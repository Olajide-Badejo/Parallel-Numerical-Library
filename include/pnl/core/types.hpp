#pragma once

/// \file types.hpp
/// Scalar and index types shared by every module.
///
/// The index type is deliberately signed. OpenMP canonical loop form accepts
/// unsigned indices only from 3.0 onward and several compilers still generate
/// worse code for them, so a signed type keeps every backend on the same
/// footing and makes reverse sweeps (backward Gauss Seidel) expressible without
/// wraparound traps.

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace pnl {

/// Working precision for the whole library. Every solver, probe and report
/// number is double precision; no mixed precision path exists, so a single
/// alias is enough and keeps the CUDA boundary unambiguous.
using Real = double;

/// Signed index type used for all ranges and extents.
using Index = std::ptrdiff_t;

/// Non owning view of a mutable vector of unknowns.
using VectorView = std::span<Real>;

/// Non owning view of an immutable vector of unknowns.
using ConstVectorView = std::span<const Real>;

/// Owning dense vector.
using Vector = std::vector<Real>;

/// Machine epsilon for Real, named so the solvers read as mathematics.
inline constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();

/// Default relative residual target. Section 8.2 of the specification fixes
/// this for every iterative solver so cross backend and cross device counts
/// are comparable.
inline constexpr Real DEFAULT_TOLERANCE = 1.0e-8;

/// Default iteration cap. Reaching it is reported, never silently accepted.
inline constexpr Index DEFAULT_MAX_ITERATIONS = 100000;

/// A half open index range [begin, end).
struct Range {
    Index begin = 0;
    Index end = 0;

    [[nodiscard]] constexpr Index size() const noexcept { return end - begin; }
    [[nodiscard]] constexpr bool empty() const noexcept { return end <= begin; }
};

/// Split [0, n) into \p parts contiguous blocks and return block \p k.
///
/// Remainder aware: the first (n mod parts) blocks receive one extra element,
/// so block sizes differ by at most one and no element is dropped. The MPI
/// backend reuses this for row distribution, which is why it lives here rather
/// than inside a single backend.
[[nodiscard]] constexpr Range block_partition(Index n, Index parts, Index k) noexcept {
    if (parts <= 0 || k < 0 || k >= parts) return Range{0, 0};
    const Index base = n / parts;
    const Index remainder = n % parts;
    const Index begin = k * base + (k < remainder ? k : remainder);
    const Index size = base + (k < remainder ? 1 : 0);
    return Range{begin, begin + size};
}

}  // namespace pnl
