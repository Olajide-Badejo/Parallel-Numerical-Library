#pragma once

/// \file mpi.hpp
/// Distributed backend over MPI, with remainder aware block row decomposition.
///
/// Idiomatic underneath: point to point `MPI_Sendrecv` for the halo exchange,
/// collectives for the reductions, and an explicit token chain for the
/// sequentially ordered sweeps. Every call goes through MPI_CHECK, so a failing
/// call is a diagnosed exception rather than a silently wrong answer.
///
/// Two decisions worth stating, because both cost something and both were
/// chosen deliberately.
///
/// Storage. Each rank allocates the whole grid and computes only its own row
/// band. A production code would allocate the local slab only, and this one
/// would too if memory were the constraint; here the largest case is a 4096
/// squared grid at 134 MB per vector, which fits comfortably. What matters for
/// the study is that the communication volume is identical either way: only the
/// halo rows are exchanged, which is the quantity the scaling analysis is
/// about. The memory is spent to keep the solver code identical across every
/// backend, which is the point of the whole design.
///
/// Determinism. The reduction gathers each rank's ordered partial and sums the
/// gathered values in rank order, rather than calling MPI_Allreduce and
/// accepting whatever association the library chooses. That makes results
/// reproducible across runs at a fixed rank count. It cannot make them bit
/// identical across different rank counts, because the rank boundaries are
/// chosen for load balance and so do not align with the fixed chunk grid the
/// shared memory backends use. The MPI tests therefore assert agreement to
/// reduction tolerance across rank counts, and bit identity within one, and the
/// documentation says which is which.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/error.hpp>

#include <mpi.h>

#include <string>
#include <vector>

namespace pnl::backend {

/// Turn a failing MPI call into a diagnosed BackendFailure.
///
/// \throws BackendFailure when the call does not return MPI_SUCCESS.
#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        const int pnl_mpi_status = (call);                                               \
        if (pnl_mpi_status != MPI_SUCCESS) {                                             \
            throw ::pnl::BackendFailure(::pnl::backend::describe_mpi_error(              \
                #call, pnl_mpi_status, __FILE__, __LINE__));                             \
        }                                                                                \
    } while (false)

/// Render an MPI error code with the call that produced it.
[[nodiscard]] std::string describe_mpi_error(const char* call, int status, const char* file,
                                             int line);

/// Where time went inside a distributed run, so the report can quote a
/// communication fraction rather than guess one.
struct CommunicationTiming {
    double halo_seconds = 0.0;
    double reduction_seconds = 0.0;
    double barrier_seconds = 0.0;
    double ordered_seconds = 0.0;
    Index halo_exchanges = 0;
    Index reductions = 0;

    [[nodiscard]] double total_seconds() const noexcept {
        return halo_seconds + reduction_seconds + barrier_seconds + ordered_seconds;
    }

    void reset() { *this = CommunicationTiming{}; }
};

/// Distributed execution over MPI_COMM_WORLD.
class MpiBackend : public Backend {
   public:
    explicit MpiBackend(const Config& config, const TopologyReport& topology);

    ~MpiBackend() override;

    MpiBackend(const MpiBackend&) = delete;
    MpiBackend& operator=(const MpiBackend&) = delete;
    MpiBackend(MpiBackend&&) = delete;
    MpiBackend& operator=(MpiBackend&&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override { return "mpi"; }

    [[nodiscard]] int worker_count() const noexcept override { return ranks_; }

    [[nodiscard]] int rank_count() const noexcept override { return ranks_; }

    [[nodiscard]] int rank() const noexcept override { return rank_; }

    [[nodiscard]] bool is_root() const noexcept override { return rank_ == 0; }

    [[nodiscard]] Range local_rows(Index total_rows) const noexcept override {
        return block_partition(total_rows, ranks_, rank_);
    }

    void parallel_for(Index n, const RangeBody& body) override;

    [[nodiscard]] Real reduce(Index n, Real init, const RangeReducer& reducer) override;

    void barrier() override;

    void exchange_halo(VectorView grid, Index row_stride, Index total_rows) override;

    void gather_rows(VectorView data, Range local) override;

    void run_ordered(const std::function<void()>& local_work, bool forward, VectorView data,
                     Index row_stride, Index total_rows) override;

    [[nodiscard]] const Config& config() const noexcept override { return config_; }

    [[nodiscard]] const CommunicationTiming& timing() const noexcept { return timing_; }

    void reset_timing() { timing_.reset(); }

   protected:
    /// Run the body over the local range. The hybrid backend overrides this to
    /// nest OpenMP threads inside the rank; everything else in this class is
    /// shared between the two.
    virtual void execute_local(Index n, const RangeBody& body);

    /// Reduce over the local range, returning this rank's ordered partial.
    virtual Real reduce_local(Index n, const RangeReducer& reducer);

    Config config_;
    TopologyReport topology_;
    int rank_ = 0;
    int ranks_ = 1;
    CommunicationTiming timing_;

   private:
    /// Whether MPI_Init was called by this object rather than by the caller.
    bool owns_mpi_ = false;
    /// Scratch for the gathered reduction partials, one per rank.
    Vector gathered_;
};

}  // namespace pnl::backend
