// SPDX-License-Identifier: MIT
#pragma once

/// \file serial.hpp
/// The correctness anchor. Every equivalence test compares against this
/// backend, and it is the only one with no concurrency of any kind, so a
/// disagreement between it and anything else is a bug in the other backend.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/topology.hpp>

#include <string>

namespace pnl::backend {

/// Single threaded execution.
///
/// It still walks the same chunk grid as the parallel backends rather than
/// looping over [0, n) in one go. That is deliberate: it means the serial
/// reduction sums the identical partials in the identical order, so
/// "serial equals parallel, bit for bit" is a meaningful statement about the
/// parallel backends rather than an artefact of the serial one taking a
/// different path.
class SerialBackend final : public Backend {
 public:
    explicit SerialBackend(const Config& config) : SerialBackend(config, TopologyReport{}) {}

    /// The one thread is worker zero, and it binds itself here rather than
    /// ignoring the request. A serial row under a pinning policy then reports
    /// what its thread did instead of a status it never earned.
    SerialBackend(const Config& config, const TopologyReport& topology) : config_(config) {
        config_.workers = 1;
        pinning_ = pin_worker(config_.pinning, 0, 1, topology);
        if (pinning_ != PinOutcome::NotRequested && pinning_ != PinOutcome::Bound) {
            throw BackendFailure(pinning_failure_message("serial", config_.pinning, 0, pinning_));
        }
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "serial"; }

    [[nodiscard]] int worker_count() const noexcept override { return 1; }

    [[nodiscard]] std::string pinning_status() const override {
        return pinning_status_text(pinning_, pinning_ == PinOutcome::Refused ? 1 : 0);
    }

    void parallel_for(Index n, const RangeBody& body) override {
        const Index chunks = for_chunk_count(n, 1, config_.schedule, config_.chunks_per_worker);
        for (Index k = 0; k < chunks; ++k) {
            body(for_chunk(n, 1, config_.schedule, config_.chunks_per_worker, k));
        }
    }

    [[nodiscard]] Real reduce(Index n, Real init, const RangeReducer& reducer) override {
        const Index chunks = reduction_chunk_count(n);
        Real total = init;
        for (Index k = 0; k < chunks; ++k) {
            total += reducer(reduction_chunk(n, k));
        }
        return total;
    }

    void barrier() override {}

    [[nodiscard]] const Config& config() const noexcept override { return config_; }

 private:
    Config config_;
    PinOutcome pinning_ = PinOutcome::NotRequested;
};

}  // namespace pnl::backend
