#pragma once

/// \file pthreads.hpp
/// A persistent worker pool built directly on POSIX threads.
///
/// Idiomatic underneath: raw pthread_create, a mutex and condition variable for
/// dispatch, a counter and a second condition variable for completion, and
/// pthread_setaffinity_np for binding. Nothing from <thread> and nothing from
/// OpenMP, so what this backend measures is the cost of the classical POSIX
/// primitives against the two higher level models running the identical
/// numerical work.
///
/// The pool is persistent. Creating threads per sweep would measure
/// pthread_create, which is a well known number and not the one this study is
/// about; a persistent pool measures wakeup latency and barrier cost, which is
/// what actually differs between the three shared memory models here.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include <pthread.h>

namespace pnl::backend {

/// Fork join pool over POSIX threads with explicit affinity.
class PthreadsBackend final : public Backend {
 public:
    explicit PthreadsBackend(const Config& config, const TopologyReport& topology);

    ~PthreadsBackend() override;

    PthreadsBackend(const PthreadsBackend&) = delete;
    PthreadsBackend& operator=(const PthreadsBackend&) = delete;
    PthreadsBackend(PthreadsBackend&&) = delete;
    PthreadsBackend& operator=(PthreadsBackend&&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override { return "pthreads"; }

    [[nodiscard]] int worker_count() const noexcept override { return workers_; }

    void parallel_for(Index n, const RangeBody& body) override;

    [[nodiscard]] Real reduce(Index n, Real init, const RangeReducer& reducer) override;

    void barrier() override {}

    [[nodiscard]] const Config& config() const noexcept override { return config_; }

    [[nodiscard]] std::string pinning_status() const override {
        return pinning_status_text(pinning_, pinning_failures());
    }

    /// How many workers reported that the operating system refused to bind
    /// them. Reported through pinning_status() in the result row, because a
    /// pinning sweep whose pinning silently failed would be worse than no sweep
    /// at all.
    [[nodiscard]] int pinning_failures() const noexcept {
        return pinning_failures_.load(std::memory_order_relaxed);
    }

 private:
    struct WorkerArgument {
        PthreadsBackend* pool;
        int id;
    };

    static void* worker_entry(void* argument);

    void worker_loop(int id);

    /// Publish a task and wait for every worker to finish it.
    void run_task(Index n, Index chunks, const RangeBody* body, const RangeReducer* reducer);

    void execute_chunks(int id);

    /// Wake every worker, join the first \p joinable of them and leave the pool
    /// down. Used by the destructor and by both constructor paths that have to
    /// unwind a pool which is already up.
    void stop_workers(std::size_t joinable);

    /// Aggregate what the workers recorded, and refuse to exist when a
    /// requested pinning did not take on every one of them.
    void finish_pinning();

    Config config_;
    TopologyReport topology_;
    int workers_ = 1;

    /// One slot per worker, written once by that worker alone before it reports
    /// through pin_done_, and read by the constructor after every worker has.
    std::vector<PinOutcome> pin_outcomes_;
    PinOutcome pinning_ = PinOutcome::NotRequested;
    /// Atomic rather than mutex guarded, which is the one rule this counter now
    /// follows everywhere it is touched. It used to be incremented without the
    /// mutex in the constructor, with it in the worker, and read without it.
    /// See CONC-02.
    std::atomic<int> pinning_failures_{0};

    std::vector<pthread_t> threads_;
    std::vector<WorkerArgument> arguments_;

    pthread_mutex_t mutex_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t work_ready_ = PTHREAD_COND_INITIALIZER;
    pthread_cond_t work_done_ = PTHREAD_COND_INITIALIZER;
    /// Signalled once by each spawned worker when it has tried to bind itself.
    /// The constructor waits on it, so the pool is fully pinned before anything
    /// asks what the pinning achieved.
    pthread_cond_t pin_done_ = PTHREAD_COND_INITIALIZER;

    /// Incremented once per dispatched task. Workers wait for it to change
    /// rather than for a flag, which makes a missed wakeup impossible and
    /// removes the lost wakeup race a plain boolean would have.
    unsigned long generation_ = 0;
    int outstanding_ = 0;
    /// Spawned workers that have reported a pinning outcome. Guarded by mutex_.
    int pin_reports_ = 0;
    bool shutting_down_ = false;

    Index task_n_ = 0;
    Index task_chunks_ = 0;
    const RangeBody* task_body_ = nullptr;
    const RangeReducer* task_reducer_ = nullptr;
    Vector partials_;
};

}  // namespace pnl::backend
