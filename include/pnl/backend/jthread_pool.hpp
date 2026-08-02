#pragma once

/// \file jthread_pool.hpp
/// A persistent worker pool built from C++23 std::jthread and std::barrier.
///
/// Idiomatic underneath: no library runtime at all, only what the standard
/// gives. Workers are std::jthread, so shutdown is cooperative through
/// std::stop_token and joining is automatic in the destructor. Synchronisation
/// is a pair of std::barrier phases, which is exactly the fork join shape the
/// sweeps need and lets the standard library pick the best available wait
/// strategy for the platform.
///
/// Deliberately absent: work stealing. The loops here are uniform stencil
/// sweeps over contiguous memory, where a static partition is already balanced
/// and a stealing deque would add per task atomics that buy nothing. Section 5
/// of the specification says so, and the dynamic schedule measured on the
/// OpenMP backend gives the number that justifies it.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>

#include <atomic>
#include <barrier>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

namespace pnl::backend {

/// Fork join pool over std::jthread with std::barrier synchronisation.
class JthreadBackend final : public Backend {
   public:
    explicit JthreadBackend(const Config& config, const TopologyReport& topology)
        : config_(config), topology_(topology) {
        const int requested =
            config.workers > 0 ? config.workers : available_logical_cpus_impl();
        workers_ = std::max(1, requested);
        config_.workers = workers_;

        // Two barriers, entered in strict alternation: the first releases the
        // workers onto a task, the second collects them. Using one barrier for
        // both would let a fast worker race ahead into the next task before a
        // slow one had left the previous.
        release_ = std::make_unique<std::barrier<>>(workers_);
        collect_ = std::make_unique<std::barrier<>>(workers_);

        threads_.reserve(static_cast<std::size_t>(workers_ - 1));
        for (int id = 1; id < workers_; ++id) {
            threads_.emplace_back([this, id](std::stop_token stop) { worker_loop(id, stop); });
        }
        // Worker zero is the calling thread, which pins itself here.
        pin_worker(0);
    }

    ~JthreadBackend() override {
        // Ask the workers to stop, then release them from the barrier they are
        // waiting on so they can observe the request.
        stopping_.store(true, std::memory_order_release);
        for (auto& thread : threads_) thread.request_stop();
        if (!threads_.empty()) {
            // One final release phase so every worker wakes and sees stopping_.
            release_->arrive_and_wait();
        }
        // Join here rather than leaving it to the jthread destructors. Members
        // are destroyed in reverse declaration order, which would destroy
        // stopping_ before threads_ were joined, and a worker still reading it
        // during that window would be a use after free. Joining explicitly in
        // the destructor body removes the window entirely.
        for (auto& thread : threads_) {
            if (thread.joinable()) thread.join();
        }
    }

    JthreadBackend(const JthreadBackend&) = delete;
    JthreadBackend& operator=(const JthreadBackend&) = delete;
    JthreadBackend(JthreadBackend&&) = delete;
    JthreadBackend& operator=(JthreadBackend&&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override { return "jthread"; }

    [[nodiscard]] int worker_count() const noexcept override { return workers_; }

    void parallel_for(Index n, const RangeBody& body) override {
        const Index chunks =
            for_chunk_count(n, workers_, config_.schedule, config_.chunks_per_worker);
        if (chunks <= 0) return;
        run_task(n, chunks, &body, nullptr);
    }

    [[nodiscard]] Real reduce(Index n, Real init, const RangeReducer& reducer) override {
        const Index chunks = reduction_chunk_count(n);
        if (chunks <= 0) return init;

        partials_.assign(static_cast<std::size_t>(chunks), 0.0);
        run_task(n, chunks, nullptr, &reducer);

        // Ordered combination, regardless of which worker produced which slot.
        Real total = init;
        if (config_.reduction == ReductionMode::Native) {
            // Even the native mode combines slot by slot here, because there is
            // no library runtime to defer to; what differs is that the native
            // path is free to use pairwise summation, which is faster to no
            // measurable degree at these sizes and is left as the ordered sum
            // so the backend has one honest behaviour rather than two.
            for (Index k = 0; k < chunks; ++k) total += partials_[static_cast<std::size_t>(k)];
            return total;
        }
        for (Index k = 0; k < chunks; ++k) total += partials_[static_cast<std::size_t>(k)];
        return total;
    }

    void barrier() override {
        // Every task already begins and ends with a barrier phase.
    }

    [[nodiscard]] const Config& config() const noexcept override { return config_; }

   private:
    /// Dispatch one task to all workers and wait for it to finish.
    void run_task(Index n, Index chunks, const RangeBody* body, const RangeReducer* reducer) {
        task_n_ = n;
        task_chunks_ = chunks;
        task_body_ = body;
        task_reducer_ = reducer;

        if (workers_ == 1) {
            execute_chunks(0);
            return;
        }

        release_->arrive_and_wait();
        execute_chunks(0);
        collect_->arrive_and_wait();
    }

    /// The share of the chunk grid belonging to worker \p id.
    ///
    /// Chunks are dealt round robin rather than in contiguous blocks so that a
    /// dynamic style chunk count still spreads evenly without any atomics.
    void execute_chunks(int id) {
        const Index chunks = task_chunks_;
        const Index n = task_n_;
        if (task_body_ != nullptr) {
            const Schedule schedule = config_.schedule;
            const int per_worker = config_.chunks_per_worker;
            const int workers = workers_;
            for (Index k = id; k < chunks; k += workers) {
                (*task_body_)(for_chunk(n, workers, schedule, per_worker, k));
            }
        } else if (task_reducer_ != nullptr) {
            const int workers = workers_;
            for (Index k = id; k < chunks; k += workers) {
                partials_[static_cast<std::size_t>(k)] = (*task_reducer_)(reduction_chunk(n, k));
            }
        }
    }

    void worker_loop(int id, std::stop_token stop) {
        pin_worker(id);
        while (true) {
            release_->arrive_and_wait();
            if (stop.stop_requested() || stopping_.load(std::memory_order_acquire)) return;
            execute_chunks(id);
            collect_->arrive_and_wait();
        }
    }

    void pin_worker(int id) {
        if (config_.pinning == Pinning::None) return;
        const int cpu = cpu_for_worker(config_.pinning, id, workers_, topology_);
        if (cpu >= 0) (void)pin_this_thread(cpu);
    }

    Config config_;
    TopologyReport topology_;
    int workers_ = 1;

    std::unique_ptr<std::barrier<>> release_;
    std::unique_ptr<std::barrier<>> collect_;
    std::vector<std::jthread> threads_;
    std::atomic<bool> stopping_{false};

    // Task state, published by run_task before the release barrier and read by
    // the workers after it. The barrier provides the synchronisation, so plain
    // members are correct here and an atomic would only add cost.
    Index task_n_ = 0;
    Index task_chunks_ = 0;
    const RangeBody* task_body_ = nullptr;
    const RangeReducer* task_reducer_ = nullptr;
    Vector partials_;
};

}  // namespace pnl::backend
