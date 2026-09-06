// SPDX-License-Identifier: MIT
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
/// time. What it asserts is OpenMP 4.5, and 4.5 is the whole of what this
/// backend uses: parallel for with an explicit schedule, reduction, single, and
/// the runtime routines for thread and place counts. Nothing above 4.5 is used,
/// and PROGRESS.md records that as a deliberate restriction rather than an
/// oversight.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>

#include <string>
#include <vector>

#include <omp.h>

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
        const int requested = config.workers > 0 ? config.workers : available_logical_cpus_impl();
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

    [[nodiscard]] std::string pinning_status() const override {
        return pinning_status_text(pinning_, pinning_failures_);
    }

    void parallel_for(Index n, const RangeBody& body) override {
        const Index chunks =
            for_chunk_count(n, workers_, config_.schedule, config_.chunks_per_worker);
        if (chunks <= 0) return;

        // Every chunk goes through the relay. An exception may not leave an
        // OpenMP structured block at all: the standard requires it to be caught
        // inside the region and libgomp calls std::terminate when it is not, so
        // this is what turns a throwing body into an exception the caller can
        // catch rather than a dead process. See detail::ExceptionRelay.
        if (config_.schedule == Schedule::Static) {
#pragma omp parallel for schedule(static) num_threads(workers_)
            for (Index k = 0; k < chunks; ++k) {
                relay_.capture([&] {
                    body(for_chunk(n, workers_, Schedule::Static, config_.chunks_per_worker, k));
                });
            }
        } else {
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers_)
            for (Index k = 0; k < chunks; ++k) {
                relay_.capture([&] {
                    body(for_chunk(n, workers_, Schedule::Dynamic, config_.chunks_per_worker, k));
                });
            }
        }
        relay_.rethrow();
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
                relay_.capture([&] { total += reducer(reduction_chunk(n, k)); });
            }
            relay_.rethrow();
            return init + total;
        }

        // Deterministic: every chunk's partial lands in its own slot, and the
        // slots are summed in index order afterwards, so the answer does not
        // depend on the thread count or on which thread finished first.
        partials_.assign(static_cast<std::size_t>(chunks), 0.0);
        Real* partials = partials_.data();
#pragma omp parallel for schedule(static) num_threads(workers_)
        for (Index k = 0; k < chunks; ++k) {
            relay_.capture([&] { partials[k] = reducer(reduction_chunk(n, k)); });
        }
        relay_.rethrow();
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
    /// thread pins itself, and keep what every one of them achieved.
    ///
    /// The outcome used to be dropped on the floor here, so a policy that bound
    /// nothing was indistinguishable from one that bound everything. Each
    /// thread now writes its own slot, which needs no synchronisation because
    /// no two threads touch the same one, and the region's implicit barrier
    /// makes them all readable afterwards.
    void apply_pinning() {
        if (config_.pinning == Pinning::None) return;
        const Pinning policy = config_.pinning;
        const TopologyReport& topology = topology_;
        const int workers = workers_;
        std::vector<PinOutcome> outcomes(static_cast<std::size_t>(workers),
                                         PinOutcome::NotRequested);
        PinOutcome* slots = outcomes.data();
#pragma omp parallel num_threads(workers_)
        {
            const int thread = omp_get_thread_num();
            if (thread < workers) {
                slots[thread] = pin_worker(policy, thread, workers, topology);
            }
        }

        for (int worker = 0; worker < workers; ++worker) {
            const PinOutcome outcome = outcomes[static_cast<std::size_t>(worker)];
            if (outcome == PinOutcome::Refused) ++pinning_failures_;
            pinning_ = worse_outcome(pinning_, outcome);
        }
        for (int worker = 0; worker < workers; ++worker) {
            const PinOutcome outcome = outcomes[static_cast<std::size_t>(worker)];
            if (outcome == PinOutcome::Bound) continue;
            // A slot still reading not_requested means that thread never ran,
            // so the team was smaller than the row would claim. That is the
            // "fewer than workers_ threads were pinned" half of the rule.
            if (outcome == PinOutcome::NotRequested) {
                throw BackendFailure("the openmp backend asked for " + std::to_string(workers) +
                                     " threads to bind under '" + std::string(to_string(policy)) +
                                     "' pinning and worker " + std::to_string(worker) +
                                     " never ran, so the team was smaller than the row claims");
            }
            throw BackendFailure(pinning_failure_message("openmp", policy, worker, outcome));
        }
    }

    Config config_;
    TopologyReport topology_;
    int workers_ = 1;
    PinOutcome pinning_ = PinOutcome::NotRequested;
    int pinning_failures_ = 0;
    Vector partials_;

    /// Catches what a body throws inside a worksharing region, where an
    /// exception may not cross the boundary of the structured block, and
    /// rethrows it after the region closes.
    detail::ExceptionRelay relay_;
};

}  // namespace pnl::backend
