// SPDX-License-Identifier: MIT
/// \file test_default_args.cpp
/// One call, one meaning, whichever static type it is made through.
///
/// Section 4.7 records a default argument on a virtual function whose only
/// override omitted it. A default argument is chosen from the static type of
/// the call expression and never from the dynamic one, so
/// `Backend::run_ordered` had three trailing parameters through a `Backend&`
/// and, for the same object, none through an `MpiBackend&`: the call that
/// compiled through the base did not compile through the derived at all.
///
/// The repair is the non virtual interface: the defaults live once, on a non
/// virtual wrapper, and what a backend overrides is an implementation with no
/// defaults on it. These cases assert the property that repair buys, which is
/// that the value a body observes does not depend on the type of the reference
/// the caller happened to be holding.
///
/// The backend here is defined in this translation unit rather than borrowed
/// from the library, because what is under test is the shape of the interface
/// and any override exercises it. The distributed override of the same function
/// is covered by tests/mpi/test_mpi.cpp, which can call it through both types
/// only because this change landed.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>

#include <pnl_test.hpp>
#include <string>

using namespace pnl;

namespace {

/// Records the arguments its ordered sweep was given.
class RecordingBackend final : public backend::Backend {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "recording"; }

    [[nodiscard]] int worker_count() const noexcept override { return 1; }

    [[nodiscard]] std::string pinning_status() const override { return "not_requested"; }

    void parallel_for(Index n, const backend::RangeBody& body) override {
        const Index chunks =
            backend::for_chunk_count(n, 1, config_.schedule, config_.chunks_per_worker);
        for (Index k = 0; k < chunks; ++k) {
            body(backend::for_chunk(n, 1, config_.schedule, config_.chunks_per_worker, k));
        }
    }

    [[nodiscard]] Real reduce(Index n, Real init, const backend::RangeReducer& reducer) override {
        const Index chunks = backend::reduction_chunk_count(n);
        Real total = init;
        for (Index k = 0; k < chunks; ++k) total += reducer(backend::reduction_chunk(n, k));
        return total;
    }

    void barrier() override {}

    [[nodiscard]] const backend::Config& config() const noexcept override { return config_; }

    /// What the last ordered sweep was handed.
    struct Seen {
        bool forward = false;
        std::size_t data_size = 0;
        Index row_stride = -1;
        Index total_rows = -1;
        int calls = 0;
    };

    [[nodiscard]] const Seen& seen() const noexcept { return seen_; }

 protected:
    void run_ordered_impl(backend::OrderedWork local_work,
                          bool forward,
                          VectorView data,
                          Index row_stride,
                          Index total_rows) override {
        seen_.forward = forward;
        seen_.data_size = data.size();
        seen_.row_stride = row_stride;
        seen_.total_rows = total_rows;
        ++seen_.calls;
        local_work();
    }

 private:
    backend::Config config_;
    Seen seen_;
};

}  // namespace

PNL_TEST("default_args/an ordered sweep sees the same arguments through base and derived") {
    RecordingBackend recording;
    backend::Backend& base = recording;

    int ran = 0;
    // Two arguments through the derived type. This is the call that did not
    // compile at all before the defaults moved off the virtual function.
    recording.run_ordered([&] { ++ran; }, true);
    const RecordingBackend::Seen through_derived = recording.seen();

    // The identical call through a base reference to the same object.
    base.run_ordered([&] { ++ran; }, true);
    const RecordingBackend::Seen through_base = recording.seen();

    PNL_REQUIRE_MESSAGE(ran == 2, "the work did not run twice");
    PNL_REQUIRE_MESSAGE(through_derived.forward == through_base.forward,
                        "the forward flag differs between the two calls");
    PNL_REQUIRE_MESSAGE(through_derived.data_size == through_base.data_size,
                        "the data view differs: " + std::to_string(through_derived.data_size) +
                            " through the derived type against " +
                            std::to_string(through_base.data_size) + " through the base");
    PNL_REQUIRE_MESSAGE(through_derived.row_stride == through_base.row_stride,
                        "the row stride differs: " + std::to_string(through_derived.row_stride) +
                            " against " + std::to_string(through_base.row_stride));
    PNL_REQUIRE_MESSAGE(
        through_derived.total_rows == through_base.total_rows,
        "the total row count differs: " + std::to_string(through_derived.total_rows) + " against " +
            std::to_string(through_base.total_rows));

    // And the defaults are the ones the interface documents: an empty view and
    // two zeros, meaning a flat vector with no row structure.
    PNL_REQUIRE(through_base.data_size == 0);
    PNL_REQUIRE(through_base.row_stride == 0);
    PNL_REQUIRE(through_base.total_rows == 0);
}

PNL_TEST("default_args/an ordered sweep given every argument still receives them") {
    RecordingBackend recording;
    Vector grid(64, 0.0);
    int ran = 0;
    recording.run_ordered([&] { ++ran; }, false, grid, 8, 6);

    const RecordingBackend::Seen seen = recording.seen();
    PNL_REQUIRE(ran == 1);
    PNL_REQUIRE(seen.forward == false);
    PNL_REQUIRE(seen.data_size == 64);
    PNL_REQUIRE(seen.row_stride == 8);
    PNL_REQUIRE(seen.total_rows == 6);
}
