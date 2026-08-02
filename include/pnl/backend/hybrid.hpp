#pragma once

/// \file hybrid.hpp
/// MPI with OpenMP threads nested inside each rank.
///
/// This is the configuration real machines are usually run in: one rank per
/// memory domain, threads within it, so the halo exchange happens between
/// domains rather than between every core. Everything distributed is inherited
/// from MpiBackend unchanged; the only difference is that a rank spreads its
/// band across threads instead of walking it on one.
///
/// Because it inherits rather than reimplements, the halo exchange, the ordered
/// pass and the deterministic reduction are literally the same code as the pure
/// MPI backend. Any difference the sweep measures between the two is therefore
/// the threading, which is the comparison the study wants.

#include <pnl/backend/mpi.hpp>

#if defined(PNL_WITH_OPENMP)
#include <omp.h>
#endif

namespace pnl::backend {

#if defined(PNL_WITH_OPENMP)

/// Ranks with OpenMP threads inside them.
class HybridBackend final : public MpiBackend {
   public:
    explicit HybridBackend(const Config& config, const TopologyReport& topology);

    [[nodiscard]] std::string_view name() const noexcept override { return "hybrid"; }

    /// Threads inside this rank.
    [[nodiscard]] int threads_per_rank() const noexcept { return threads_; }

   protected:
    void execute_local(Index n, const RangeBody& body) override;

    Real reduce_local(Index n, const RangeReducer& reducer) override;

   private:
    int threads_ = 1;
    Vector partials_;
};

#endif  // PNL_WITH_OPENMP

}  // namespace pnl::backend
