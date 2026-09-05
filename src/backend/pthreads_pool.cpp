/// \file pthreads_pool.cpp
/// Implementation of the POSIX threads worker pool.

#include <pnl/backend/pthreads.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>

namespace pnl::backend {

PthreadsBackend::PthreadsBackend(const Config& config, const TopologyReport& topology)
    : config_(config), topology_(topology) {
    const int requested = config.workers > 0 ? config.workers : available_logical_cpus_impl();
    workers_ = std::max(1, requested);
    config_.workers = workers_;

    pin_outcomes_.assign(static_cast<std::size_t>(workers_), PinOutcome::NotRequested);

    // Worker zero is the calling thread; only workers 1 and above get a pthread.
    pin_outcomes_[0] = pin_worker(config_.pinning, 0, workers_, topology_);
    if (pin_outcomes_[0] == PinOutcome::Refused) {
        pinning_failures_.fetch_add(1, std::memory_order_relaxed);
    }

    const int spawned = workers_ - 1;
    threads_.resize(static_cast<std::size_t>(spawned));
    arguments_.resize(static_cast<std::size_t>(spawned));

    outstanding_ = 0;
    for (int index = 0; index < spawned; ++index) {
        arguments_[static_cast<std::size_t>(index)] = WorkerArgument{this, index + 1};
        const int status = pthread_create(&threads_[static_cast<std::size_t>(index)],
                                          nullptr,
                                          worker_entry,
                                          &arguments_[static_cast<std::size_t>(index)]);
        if (status != 0) {
            // Tear down whatever started before reporting, so a partial pool
            // never escapes the constructor.
            stop_workers(static_cast<std::size_t>(index));
            throw BackendFailure(std::string("pthread_create failed for worker ") +
                                 std::to_string(index + 1) + ": " + std::strerror(status));
        }
    }

    // Every worker binds itself from inside its own entry point, so the pool
    // has to be up before the constructor can know what the pinning achieved.
    pthread_mutex_lock(&mutex_);
    while (pin_reports_ < spawned) pthread_cond_wait(&pin_done_, &mutex_);
    pthread_mutex_unlock(&mutex_);

    finish_pinning();
}

void PthreadsBackend::stop_workers(std::size_t joinable) {
    pthread_mutex_lock(&mutex_);
    shutting_down_ = true;
    ++generation_;
    pthread_cond_broadcast(&work_ready_);
    pthread_mutex_unlock(&mutex_);

    for (std::size_t joined = 0; joined < joinable; ++joined) {
        pthread_join(threads_[joined], nullptr);
    }
}

void PthreadsBackend::finish_pinning() {
    for (const PinOutcome outcome : pin_outcomes_) pinning_ = worse_outcome(pinning_, outcome);
    if (config_.pinning == Pinning::None || pinning_ == PinOutcome::Bound) return;

    int worker = 0;
    while (worker < workers_ &&
           pin_outcomes_[static_cast<std::size_t>(worker)] == PinOutcome::Bound) {
        ++worker;
    }
    const PinOutcome outcome = pin_outcomes_[static_cast<std::size_t>(worker)];
    // The pool is already up, so it comes down before the failure leaves the
    // constructor: a destructor never runs for an object that did not finish
    // constructing.
    stop_workers(threads_.size());
    throw BackendFailure(pinning_failure_message("pthreads", config_.pinning, worker, outcome));
}

PthreadsBackend::~PthreadsBackend() {
    stop_workers(threads_.size());

    pthread_cond_destroy(&pin_done_);
    pthread_cond_destroy(&work_done_);
    pthread_cond_destroy(&work_ready_);
    pthread_mutex_destroy(&mutex_);
}

void* PthreadsBackend::worker_entry(void* argument) {
    auto* typed = static_cast<WorkerArgument*>(argument);
    typed->pool->worker_loop(typed->id);
    return nullptr;
}

void PthreadsBackend::worker_loop(int id) {
    const PinOutcome outcome = pin_worker(config_.pinning, id, workers_, topology_);
    if (outcome == PinOutcome::Refused) pinning_failures_.fetch_add(1, std::memory_order_relaxed);
    // This worker owns this slot and no other, so the write races with nothing.
    // Releasing the mutex below is what publishes it to the constructor.
    pin_outcomes_[static_cast<std::size_t>(id)] = outcome;

    pthread_mutex_lock(&mutex_);
    ++pin_reports_;
    pthread_cond_signal(&pin_done_);
    pthread_mutex_unlock(&mutex_);

    unsigned long seen = 0;
    while (true) {
        pthread_mutex_lock(&mutex_);
        // Wait for the generation counter to move. Comparing against a counter
        // rather than testing a flag means a task published while this worker
        // was still finishing the previous one cannot be missed.
        while (generation_ == seen && !shutting_down_) {
            pthread_cond_wait(&work_ready_, &mutex_);
        }
        if (shutting_down_) {
            pthread_mutex_unlock(&mutex_);
            return;
        }
        seen = generation_;
        pthread_mutex_unlock(&mutex_);

        execute_chunks(id);

        pthread_mutex_lock(&mutex_);
        if (--outstanding_ == 0) pthread_cond_signal(&work_done_);
        pthread_mutex_unlock(&mutex_);
    }
}

void PthreadsBackend::execute_chunks(int id) {
    const Index chunks = task_chunks_;
    const Index n = task_n_;
    const int workers = workers_;
    if (task_body_ != nullptr) {
        const Schedule schedule = config_.schedule;
        const int per_worker = config_.chunks_per_worker;
        for (Index k = id; k < chunks; k += workers) {
            (*task_body_)(for_chunk(n, workers, schedule, per_worker, k));
        }
    } else if (task_reducer_ != nullptr) {
        for (Index k = id; k < chunks; k += workers) {
            partials_[static_cast<std::size_t>(k)] = (*task_reducer_)(reduction_chunk(n, k));
        }
    }
}

void PthreadsBackend::run_task(Index n,
                               Index chunks,
                               const RangeBody* body,
                               const RangeReducer* reducer) {
    task_n_ = n;
    task_chunks_ = chunks;
    task_body_ = body;
    task_reducer_ = reducer;

    if (workers_ == 1) {
        execute_chunks(0);
        return;
    }

    pthread_mutex_lock(&mutex_);
    outstanding_ = workers_ - 1;
    ++generation_;
    pthread_cond_broadcast(&work_ready_);
    pthread_mutex_unlock(&mutex_);

    // The dispatching thread is worker zero and does its share too, rather than
    // idling while it waits.
    execute_chunks(0);

    pthread_mutex_lock(&mutex_);
    while (outstanding_ > 0) pthread_cond_wait(&work_done_, &mutex_);
    pthread_mutex_unlock(&mutex_);
}

void PthreadsBackend::parallel_for(Index n, const RangeBody& body) {
    const Index chunks = for_chunk_count(n, workers_, config_.schedule, config_.chunks_per_worker);
    if (chunks <= 0) return;
    run_task(n, chunks, &body, nullptr);
}

Real PthreadsBackend::reduce(Index n, Real init, const RangeReducer& reducer) {
    const Index chunks = reduction_chunk_count(n);
    if (chunks <= 0) return init;

    partials_.assign(static_cast<std::size_t>(chunks), 0.0);
    run_task(n, chunks, nullptr, &reducer);

    // Ordered combination over the fixed chunk grid, so the result matches the
    // serial backend bit for bit whatever the worker count.
    Real total = init;
    for (Index k = 0; k < chunks; ++k) total += partials_[static_cast<std::size_t>(k)];
    return total;
}

}  // namespace pnl::backend
