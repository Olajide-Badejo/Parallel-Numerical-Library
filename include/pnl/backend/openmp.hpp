#pragma once

/// \file openmp.hpp
/// OpenMP backend.
///
/// Idiomatic underneath: this is the only backend that does not partition the
/// range itself. It hands the chunk grid to OpenMP's own scheduler through a
/// worksharing construct, so what the study measures is OpenMP's dispatch and
/// its barrier, not a hand written partitioner wearing an OpenMP hat.
///
/// Specification note. The build cites OpenMP 6.0 (OpenMP ARB, November 2024)
/// as the reference document, and asserts the _OPENMP version macro at compile
/// time. GCC 16 reports 202111, which is OpenMP 5.2, so the constructs used here
/// are restricted to that level: parallel for with an explicit schedule,
/// reduction, single, and the runtime routines for thread and place counts. The
/// 6.0 additions are not used, and PROGRESS.md records that as a deliberate
/// restriction rather than an oversight.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>

#include <omp.h>

#include <vector>

#if !defined(_OPENMP)
#error "openmp.hpp requires a compiler invoked with OpenMP enabled"
#endif

// OpenMP 4.5 is the floor the sweeps actually need. Asserting it means a
// silently downgraded toolchain fails the build rather than the benchmark.
static_assert(_OPENMP >= 201511, "this build requires OpenMP 4.5 or newer");

namespace pnl::backend {

/// Fork join parallelism through OpenMP worksharing.
class OpenMpBackend final : public Backend {
   public:
    explicit OpenMpBackend(const Config& config, const TopologyReport& topology)
        : config_(config), topology_(topology) {
        const int requested =
            config.workers > 0 ? config.workers : available_logical_cpus_impl();
        workers_ = std::max(1, requested);
        config_.workers = workers_;
        omp_set_num_threads(workers_);
        // Nested parallelism is off: the hybrid backend is the one place threads
        // are nested, and it configures that itself.
        omp_set_max_active_levels(1);
        apply_pinning();
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "openmp"; }

    [[nodiscard]] int worker_count() const noexcept override { return workers_; }

    void parallel_for(Index n, const RangeBody& body) override {
        const Index chunks =
            for_chunk_count(n, workers_, config_.schedule, config_.chunks_per_worker);
        if (chunks <= 0) return;

        if (config_.schedule == Schedule::Static) {
#pragma omp parallel for schedule(static) num_threads(workers_)
            for (Index k = 0; k < chunks; ++k) {
                body(for_chunk(n, workers_, Schedule::Static, config_.chunks_per_worker, k));
            }
        } else {
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers_)
            for (Index k = 0; k < chunks; ++k) {
                body(for_chunk(n, workers_, Schedule::Dynamic, config_.chunks_per_worker, k));
            }
        }
    }

    [[nodiscard]] Real reduce(Index n, Real init, const RangeReducer& reducer) override {
        const Index chunks = reduction_chunk_count(n);
        if (chunks <= 0) return init;

        if (config_.reduction == ReductionMode::Native) {
            // OpenMP's own reduction clause: fast, and its combination order is
            // whatever the runtime chooses, so results vary with thread count.
            Real total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total) num_threads(workers_)
            for (Index k = 0; k < chunks; ++k) {
                total += reducer(reduction_chunk(n, k));
            }
            return init + total;
        }

        // Deterministic: every chunk's partial lands in its own slot, and the
        // slots are summed in index order afterwards, so the answer does not
        // depend on the thread count or on which thread finished first.
        partials_.assign(static_cast<std::size_t>(chunks), 0.0);
        Real* partials = partials_.data();
#pragma omp parallel for schedule(static) num_threads(workers_)
        for (Index k = 0; k < chunks; ++k) {
            partials[k] = reducer(reduction_chunk(n, k));
        }
        Real total = init;
        for (Index k = 0; k < chunks; ++k) total += partials[k];
        return total;
    }

    void barrier() override {
        // A standalone barrier outside a parallel region is a no operation:
        // every worksharing construct above already ends with one.
    }

    [[nodiscard]] const Config& config() const noexcept override { return config_; }

   private:
    /// Bind each OpenMP thread once, from inside a parallel region so that each
    /// thread pins itself.
    void apply_pinning() {
        if (config_.pinning == Pinning::None) return;
        const Pinning policy = config_.pinning;
        const TopologyReport& topology = topology_;
        const int workers = workers_;
#pragma omp parallel num_threads(workers_)
        {
            const int thread = omp_get_thread_num();
            const int cpu = cpu_for_worker(policy, thread, workers, topology);
            if (cpu >= 0) (void)pin_this_thread(cpu);
        }
    }

    Config config_;
    TopologyReport topology_;
    int workers_ = 1;
    Vector partials_;
};

}  // namespace pnl::backend
