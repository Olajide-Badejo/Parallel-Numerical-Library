/// \file mpi_runtime.cpp
/// Implementation of the MPI and hybrid backends.

#include <pnl/backend/hybrid.hpp>
#include <pnl/backend/mpi.hpp>

#include <algorithm>
#include <cstring>

namespace pnl::backend {

namespace {

/// Message tags, distinct so a mismatched pair cannot be silently matched.
constexpr int TAG_HALO_DOWN = 3001;
constexpr int TAG_HALO_UP = 3002;
constexpr int TAG_ORDERED = 3003;

[[nodiscard]] double wall_time() { return MPI_Wtime(); }

}  // namespace

std::string describe_mpi_error(const char* call, int status, const char* file, int line) {
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    if (MPI_Error_string(status, text, &length) != MPI_SUCCESS) {
        std::snprintf(text, sizeof(text), "unknown error %d", status);
        length = static_cast<int>(std::strlen(text));
    }
    return std::string(call) + " failed at " + file + ":" + std::to_string(line) + ": " +
           std::string(text, static_cast<std::size_t>(length));
}

MpiBackend::MpiBackend(const Config& config, const TopologyReport& topology)
    : config_(config), topology_(topology) {
    int initialised = 0;
    MPI_CHECK(MPI_Initialized(&initialised));
    if (initialised == 0) {
        // The driver normally initialises MPI itself so it can request a thread
        // level. Initialising here as well keeps the backend usable from a test
        // binary that did not.
        int provided = 0;
        MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided));
        owns_mpi_ = true;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks_));

    // Make MPI report errors instead of aborting, so MPI_CHECK can turn them
    // into exceptions carrying the call site.
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    config_.workers = ranks_;
    gathered_.assign(static_cast<std::size_t>(ranks_), 0.0);

    if (config_.pinning != Pinning::None) {
        const int cpu = cpu_for_worker(config_.pinning, rank_, ranks_, topology_);
        if (cpu >= 0) (void)pin_this_thread(cpu);
    }
}

MpiBackend::~MpiBackend() {
    if (owns_mpi_) {
        int finalised = 0;
        if (MPI_Finalized(&finalised) == MPI_SUCCESS && finalised == 0) {
            (void)MPI_Finalize();
        }
    }
}

void MpiBackend::execute_local(Index n, const RangeBody& body) {
    // A plain rank runs its band on one thread. The hybrid backend overrides
    // this to spread the band across OpenMP threads.
    const Index chunks = for_chunk_count(n, 1, config_.schedule, config_.chunks_per_worker);
    for (Index k = 0; k < chunks; ++k) {
        body(for_chunk(n, 1, config_.schedule, config_.chunks_per_worker, k));
    }
}

Real MpiBackend::reduce_local(Index n, const RangeReducer& reducer) {
    const Index chunks = reduction_chunk_count(n);
    Real total = 0.0;
    for (Index k = 0; k < chunks; ++k) total += reducer(reduction_chunk(n, k));
    return total;
}

void MpiBackend::parallel_for(Index n, const RangeBody& body) {
    if (n <= 0) return;
    execute_local(n, body);
}

Real MpiBackend::reduce(Index n, Real init, const RangeReducer& reducer) {
    const Real local = n > 0 ? reduce_local(n, reducer) : 0.0;

    const double start = wall_time();
    Real total = init;
    if (config_.reduction == ReductionMode::Native) {
        Real sum = 0.0;
        MPI_CHECK(MPI_Allreduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        total += sum;
    } else {
        // Gather the per rank partials and sum them in rank order on every
        // rank, so all ranks agree bit for bit and repeated runs agree too.
        MPI_CHECK(MPI_Allgather(&local, 1, MPI_DOUBLE, gathered_.data(), 1, MPI_DOUBLE,
                                MPI_COMM_WORLD));
        for (int r = 0; r < ranks_; ++r) total += gathered_[static_cast<std::size_t>(r)];
    }
    timing_.reduction_seconds += wall_time() - start;
    ++timing_.reductions;
    return total;
}

void MpiBackend::barrier() {
    const double start = wall_time();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    timing_.barrier_seconds += wall_time() - start;
}

void MpiBackend::exchange_halo(VectorView grid, Index row_stride, Index total_rows) {
    if (ranks_ == 1 || total_rows <= 0 || grid.empty()) return;

    if (row_stride <= 0) {
        gather_rows(grid, local_rows(total_rows));
        return;
    }

    const Range rows = local_rows(total_rows);
    // Interior grid rows owned by this rank are [rows.begin + 1, rows.end].
    const int above = rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL;
    const int below = rank_ + 1 < ranks_ ? rank_ + 1 : MPI_PROC_NULL;
    const auto count = static_cast<int>(row_stride);

    const double start = wall_time();

    // Send my first owned row up, receive my lower halo from below.
    Real* first_owned = grid.data() + (rows.begin + 1) * row_stride;
    Real* lower_halo = grid.data() + (rows.end + 1) * row_stride;
    MPI_CHECK(MPI_Sendrecv(first_owned, count, MPI_DOUBLE, above, TAG_HALO_UP, lower_halo,
                           count, MPI_DOUBLE, below, TAG_HALO_UP, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));

    // Send my last owned row down, receive my upper halo from above.
    Real* last_owned = grid.data() + rows.end * row_stride;
    Real* upper_halo = grid.data() + rows.begin * row_stride;
    MPI_CHECK(MPI_Sendrecv(last_owned, count, MPI_DOUBLE, below, TAG_HALO_DOWN, upper_halo,
                           count, MPI_DOUBLE, above, TAG_HALO_DOWN, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));

    timing_.halo_seconds += wall_time() - start;
    ++timing_.halo_exchanges;
}

void MpiBackend::gather_rows(VectorView data, Range local) {
    if (ranks_ == 1 || data.empty()) return;

    const double start = wall_time();

    // Collect everyone's range rather than assuming it. A rank's share of the
    // blocks in a block method covers different rows than its share of the rows
    // in a point method, and gathering with the wrong offsets would corrupt the
    // vector silently.
    int mine[2] = {static_cast<int>(local.begin), static_cast<int>(local.size())};
    std::vector<int> all(static_cast<std::size_t>(2 * ranks_));
    MPI_CHECK(MPI_Allgather(mine, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD));

    std::vector<int> offsets(static_cast<std::size_t>(ranks_));
    std::vector<int> counts(static_cast<std::size_t>(ranks_));
    for (int r = 0; r < ranks_; ++r) {
        offsets[static_cast<std::size_t>(r)] = all[static_cast<std::size_t>(2 * r)];
        counts[static_cast<std::size_t>(r)] = all[static_cast<std::size_t>(2 * r + 1)];
    }

    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data.data(), counts.data(),
                             offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));

    timing_.halo_seconds += wall_time() - start;
    ++timing_.halo_exchanges;
}

void MpiBackend::run_ordered(const std::function<void()>& local_work, bool forward,
                             VectorView data, Index row_stride, Index total_rows) {
    if (ranks_ == 1) {
        local_work();
        return;
    }

    const double start = wall_time();
    const bool grid_layout = row_stride > 0 && total_rows > 0 && !data.empty();

    // Rank order for this sweep: ascending for a forward sweep, descending for
    // a backward one, which is exactly the natural ordering of the unknowns.
    const int predecessor = forward ? (rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL)
                                    : (rank_ + 1 < ranks_ ? rank_ + 1 : MPI_PROC_NULL);
    const int successor = forward ? (rank_ + 1 < ranks_ ? rank_ + 1 : MPI_PROC_NULL)
                                  : (rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL);

    if (grid_layout) {
        const Range rows = local_rows(total_rows);
        const auto count = static_cast<int>(row_stride);
        // Only the single boundary row has to travel, because a five point
        // stencil couples a row to its immediate neighbours and nothing else.
        Real* incoming = forward ? data.data() + rows.begin * row_stride
                                 : data.data() + (rows.end + 1) * row_stride;
        Real* outgoing = forward ? data.data() + rows.end * row_stride
                                 : data.data() + (rows.begin + 1) * row_stride;

        if (predecessor != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Recv(incoming, count, MPI_DOUBLE, predecessor, TAG_ORDERED,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        local_work();
        if (successor != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Send(outgoing, count, MPI_DOUBLE, successor, TAG_ORDERED,
                               MPI_COMM_WORLD));
        }
    } else if (!data.empty()) {
        // No row structure, as for a dense system, where a rank's update reads
        // every earlier unknown. The whole vector travels along the chain. That
        // is expensive by construction, and measuring how expensive is one of
        // the results: a sequentially dependent method has nothing to offer a
        // distributed machine.
        const auto count = static_cast<int>(data.size());
        if (predecessor != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Recv(data.data(), count, MPI_DOUBLE, predecessor, TAG_ORDERED,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        local_work();
        if (successor != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Send(data.data(), count, MPI_DOUBLE, successor, TAG_ORDERED,
                               MPI_COMM_WORLD));
        }
        // The last rank in the chain holds the fully updated vector. Everyone
        // needs it before the next residual evaluation.
        const int last = forward ? ranks_ - 1 : 0;
        MPI_CHECK(MPI_Bcast(data.data(), count, MPI_DOUBLE, last, MPI_COMM_WORLD));
    } else {
        local_work();
    }

    timing_.ordered_seconds += wall_time() - start;
}

// ---------------------------------------------------------------------------
// Hybrid
// ---------------------------------------------------------------------------

#if defined(PNL_WITH_OPENMP)

HybridBackend::HybridBackend(const Config& config, const TopologyReport& topology)
    : MpiBackend(config, topology) {
    threads_ = std::max(1, config.threads_per_rank);
    // Workers reported to the result row are the total across the job, which is
    // the number the scaling curve is plotted against.
    config_.workers = ranks_ * threads_;
    omp_set_num_threads(threads_);
    omp_set_max_active_levels(1);
}

void HybridBackend::execute_local(Index n, const RangeBody& body) {
    const Index chunks = for_chunk_count(n, threads_, config_.schedule, config_.chunks_per_worker);
    if (chunks <= 0) return;
    const Schedule schedule = config_.schedule;
    const int per_worker = config_.chunks_per_worker;
    const int threads = threads_;

    if (schedule == Schedule::Static) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (Index k = 0; k < chunks; ++k) {
            body(for_chunk(n, threads, Schedule::Static, per_worker, k));
        }
    } else {
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (Index k = 0; k < chunks; ++k) {
            body(for_chunk(n, threads, Schedule::Dynamic, per_worker, k));
        }
    }
}

Real HybridBackend::reduce_local(Index n, const RangeReducer& reducer) {
    const Index chunks = reduction_chunk_count(n);
    if (chunks <= 0) return 0.0;

    // Ordered within the rank for the same reason as every other backend: the
    // partials land in fixed slots and are summed in index order afterwards.
    partials_.assign(static_cast<std::size_t>(chunks), 0.0);
    Real* partials = partials_.data();
    const int threads = threads_;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (Index k = 0; k < chunks; ++k) {
        partials[k] = reducer(reduction_chunk(n, k));
    }
    Real total = 0.0;
    for (Index k = 0; k < chunks; ++k) total += partials[k];
    return total;
}

#endif  // PNL_WITH_OPENMP

}  // namespace pnl::backend
