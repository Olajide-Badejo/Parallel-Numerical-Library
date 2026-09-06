// SPDX-License-Identifier: MIT
#pragma once

/// \file backend.hpp
/// The one interface every execution model implements.
///
/// Design note, because this is the decision the whole comparison rests on.
///
/// The callbacks are chunk level, not element level: a body receives a Range
/// and loops over it itself. Two consequences follow, both deliberate. First,
/// the std::function indirection is paid once per chunk rather than once per
/// element, so it is O(workers) per sweep and does not contaminate the timings
/// the study exists to measure. Second, the innermost loop stays a plain loop
/// over contiguous memory that the compiler can still vectorise, so each
/// backend measures its own dispatch and synchronisation cost rather than a
/// penalty this abstraction imposed.
///
/// The interface is the minimal common denominator of the models it spans, as
/// argued in McCool, Robison and Reinders, "Structured Parallel Programming",
/// Morgan Kaufmann 2012, chapters 3 and 5: a data parallel map, a reduction,
/// and a barrier. Anything richer would privilege one model. Each
/// implementation stays idiomatic underneath: OpenMP uses its own scheduler,
/// pthreads uses a condition variable pool with explicit affinity, the jthread
/// pool uses std::barrier and std::stop_token, MPI uses remainder aware block
/// decomposition.
///
/// Backends never leak their model's types through this interface: no
/// MPI_Comm, no omp_ types, no cudaStream_t appears in any signature here.
///
/// The CUDA path is not one of these. It is a separate solver reached through
/// the `extern "C"` entry points of pnl/backend/cuda.hpp, and decision 9 in
/// docs/DESIGN_DECISIONS.md says why: a device solve owns the whole iteration
/// and crosses PCIe once, where a Backend is dispatched per sweep from the host,
/// so wearing this interface would have meant one kernel launch and one
/// synchronisation per callback and a measurement of the launch overhead rather
/// than of the device. The recognised backend names below therefore do not
/// include a device one, make_backend has never answered to one and
/// available_backends() has never emitted one, and the comparison the report
/// makes between host and device is between a Backend and something that is
/// deliberately not one. The documentation used to list a device name among the
/// recognised ones anyway, which is the last item of finding 4.8.

#include <pnl/core/contract.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/function_ref.hpp>
#include <pnl/core/types.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pnl::backend {

/// How a parallel_for distributes its chunks.
enum class Schedule {
    /// One contiguous block per worker, decided up front. The right choice for
    /// the uniform stencil sweeps that dominate this library.
    Static,
    /// Chunks handed out on demand. Measured once and reported so the cost of
    /// dynamic scheduling on uniform work is a number rather than folklore.
    Dynamic,
};

/// How a reduction combines partial results.
///
/// Floating point addition is not associative, so a reduction's answer depends
/// on the order in which partials are combined. Rather than hide that behind a
/// tolerance, the library makes it a choice.
enum class ReductionMode {
    /// Partials are combined in a fixed chunk order that depends only on the
    /// problem size, never on the worker count or the arrival order of threads.
    /// Every backend therefore produces bit identical reductions, which is what
    /// lets the equivalence suite assert exact equality instead of "close
    /// enough". This is the default.
    Deterministic,
    /// The model's native reduction: an OpenMP reduction clause, an atomic
    /// accumulation, MPI_Allreduce. Faster, and not reproducible across worker
    /// counts. Available so the sweep can price determinism honestly.
    Native,
};

/// Thread to core binding policy, swept as a variable in the scaling study
/// because this is a heterogeneous CPU and the answer differs per policy.
enum class Pinning {
    /// No affinity set; the scheduler places threads.
    None,
    /// Workers bound to logical CPUs 0, 1, 2, ... in order.
    Compact,
    /// Workers bound one per physical core, skipping the sibling hyperthread.
    Scatter,
    /// Workers bound to performance cores only, as classified at runtime.
    PerformanceCores,
    /// Workers bound to efficiency cores only, as classified at runtime.
    EfficiencyCores,
};

[[nodiscard]] constexpr std::string_view to_string(Pinning pinning) noexcept {
    switch (pinning) {
        case Pinning::None:
            return "none";
        case Pinning::Compact:
            return "compact";
        case Pinning::Scatter:
            return "scatter";
        case Pinning::PerformanceCores:
            return "pcore";
        case Pinning::EfficiencyCores:
            return "ecore";
    }
    return "unknown";
}

/// What a pinning request achieved for one worker.
///
/// The outcomes are kept apart because they call for different answers. A
/// policy nobody asked for is not a fault. A policy whose classification this
/// machine did not yield is a fault in the request, and under WSL2 it is the
/// normal answer for the two core policies, which is why it used to be
/// indistinguishable from "no pinning was asked for" and silently did nothing.
/// A binding the operating system refused is a fault in the environment.
enum class PinOutcome {
    /// No pinning was asked for.
    NotRequested,
    /// The worker is bound to one logical processor.
    Bound,
    /// The policy needs a classification this machine did not yield.
    NotApplicable,
    /// The operating system refused the binding.
    Refused,
};

[[nodiscard]] constexpr std::string_view to_string(PinOutcome outcome) noexcept {
    switch (outcome) {
        case PinOutcome::NotRequested:
            return "not_requested";
        case PinOutcome::Bound:
            return "bound";
        case PinOutcome::NotApplicable:
            return "not_applicable";
        case PinOutcome::Refused:
            return "refused";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(ReductionMode mode) noexcept {
    return mode == ReductionMode::Deterministic ? "deterministic" : "native";
}

/// Body of a parallel_for: called once per chunk with the chunk's range.
///
/// A reference to the caller's callable, not a copy of it. Every backend here
/// already treated it that way, holding a bare pointer to the caller's object
/// while its workers ran; what a std::function added was one heap allocation
/// per dispatch, because these bodies capture the sweep's pointers and extents
/// and so are far too large for the small object buffer. Three dispatches per
/// Jacobi iteration meant three allocations per iteration inside the timed
/// region. See function_ref.hpp and MEAS-10.
using RangeBody = FunctionRef<void(Range)>;

/// Body of a reduce: returns this chunk's partial result.
using RangeReducer = FunctionRef<Real(Range)>;

/// Body of a run_ordered: the work one rank does when its turn comes.
using OrderedWork = FunctionRef<void()>;

/// Configuration handed to a backend factory.
struct Config {
    /// Number of workers: threads for the thread backends, ranks for MPI,
    /// ranks times threads for hybrid. Zero means "ask the system".
    int workers = 0;
    /// Threads per rank; only the hybrid backend reads this.
    int threads_per_rank = 1;
    Pinning pinning = Pinning::None;
    ReductionMode reduction = ReductionMode::Deterministic;
    Schedule schedule = Schedule::Static;
    /// Chunks per worker for Schedule::Dynamic. Ignored when Static.
    int chunks_per_worker = 8;
};

namespace detail {

/// The precondition every implementation of Backend::exchange_halo shares.
///
/// A padded grid carries a boundary ring above and below the interior rows, so
/// a row stride of s and t interior rows occupy (t + 2) * s values; a row
/// stride of zero means a flat vector with no row structure, and then the t
/// values themselves are what has to be there. The distributed implementation
/// writes the halo rows through raw pointers derived from those two numbers, so
/// a view that is short of them is a heap write past the end rather than a wrong
/// answer, and the shared memory default is where a caller's mistake would
/// otherwise go unnoticed until the day it ran under MPI.
///
/// The guard is an `if` around `require(false, ...)` because the message names
/// both sizes and building it unconditionally would allocate once per sweep
/// inside the timed region. See MEAS-10.
inline void check_halo_arguments(
    ConstVectorView grid,
    Index row_stride,
    Index total_rows,
    const std::source_location& where = std::source_location::current()) {
    if (total_rows <= 0) return;
    if (row_stride < 0) {
        require(false, "Backend::exchange_halo needs a row stride that is not negative", where);
    }
    const Index needed = row_stride > 0 ? (total_rows + 2) * row_stride : total_rows;
    if (static_cast<Index>(grid.size()) < needed) {
        require(
            false,
            pnl::detail::too_small(
                "Backend::exchange_halo", "grid", static_cast<std::size_t>(needed), grid.size()),
            where);
    }
}

/// The precondition every implementation of Backend::gather_rows shares.
///
/// The range is the segment this process owns and is used directly as an offset
/// and a count into \p data by MPI_Allgatherv, so a range that runs past the end
/// of the view is a buffer overrun on every rank at once.
inline void check_gather_arguments(
    ConstVectorView data,
    Range local,
    const std::source_location& where = std::source_location::current()) {
    if (local.begin < 0 || local.end < local.begin) {
        require(false,
                "Backend::gather_rows needs a range whose begin is not negative and whose end "
                "is not before its begin",
                where);
    }
    if (local.end > static_cast<Index>(data.size())) {
        require(
            false,
            pnl::detail::too_small(
                "Backend::gather_rows", "data", static_cast<std::size_t>(local.end), data.size()),
            where);
    }
}

/// Holds the first exception a worker body threw and rethrows it on the
/// dispatching thread.
///
/// Every pool here runs the caller's body on threads the caller never sees, and
/// the language has no way to carry an exception across that boundary by
/// itself. Left alone it ends one of two ways, both of them fatal and neither
/// of them catchable: an exception that leaves a pthread entry point, a
/// std::jthread body or an OpenMP structured block is std::terminate by
/// definition, and a body that threw on the dispatching thread of the jthread
/// pool skipped the collect barrier every other worker was already parked on,
/// which is a deadlock rather than a crash. Those are two rows of Section 4.7.
///
/// The contract this implements, and which parallel_for and reduce document:
///
/// - **Every chunk of the dispatch is attempted.** capture() swallows the
///   throw, so the worker finishes its share and reaches its barrier. That is
///   not politeness towards the other chunks, it is the only reason the pools
///   cannot deadlock: a worker that returned early would be a worker the
///   barrier is still waiting for.
/// - **The first exception to arrive wins**, decided by an atomic exchange, and
///   later ones are dropped. "First" is by arrival rather than by chunk index,
///   which is as deterministic as concurrent throws allow; what is promised is
///   that every thread sees the same one.
/// - **rethrow() delivers it on the dispatching thread** after the join point,
///   and clears the relay, so the backend is usable for the next dispatch.
///
/// The relay belongs to one backend object, which is already documented as
/// belonging to one thread, so nothing here is shareable either.
class ExceptionRelay {
 public:
    /// Run \p work, recording the first exception it throws rather than letting
    /// it escape. Never throws, which is what makes it safe to call from a
    /// thread entry point.
    template<typename Work>
    void capture(Work&& work) noexcept {
        try {
            std::forward<Work>(work)();
        } catch (...) {
            // Two flags rather than one. claimed_ picks the winner; captured_
            // publishes what the winner wrote, and it is stored with release
            // after first_ is written so that the acquire load in rethrow()
            // orders the two. Doing it with a single exchange would order the
            // flag against a write that had not happened yet.
            if (!claimed_.exchange(true, std::memory_order_relaxed)) {
                first_ = std::current_exception();
                captured_.store(true, std::memory_order_release);
            }
        }
    }

    /// True when some worker threw during the last dispatch.
    [[nodiscard]] bool holds_exception() const noexcept {
        return captured_.load(std::memory_order_acquire);
    }

    /// Rethrow the first captured exception, if there is one, and reset.
    ///
    /// Called on the dispatching thread once every worker is back, so the
    /// barrier or the completion counter has already ordered the workers'
    /// writes against this thread; the acquire here makes the relay correct on
    /// its own terms as well, rather than only in the company it keeps.
    void rethrow() {
        if (!captured_.load(std::memory_order_acquire)) return;
        const std::exception_ptr held = first_;
        first_ = nullptr;
        captured_.store(false, std::memory_order_relaxed);
        claimed_.store(false, std::memory_order_relaxed);
        std::rethrow_exception(held);
    }

 private:
    std::atomic<bool> claimed_{false};
    std::atomic<bool> captured_{false};
    std::exception_ptr first_;
};

}  // namespace detail

/// The execution backend interface.
///
/// **A Backend object is not shareable between threads, and none of these
/// functions is reentrant.** One thread owns the object and dispatches through
/// it; the parallelism is inside, not around. The two hand written pools
/// publish the task, the chunk count and the callback as plain members and then
/// open a barrier, so a second dispatch entering while the first is in flight
/// would overwrite the task the workers are reading and would leave the barrier
/// phases out of step; the reducing backends write their partials into a member
/// scratch array indexed by chunk, so two reductions would interleave their
/// slots; and the distributed backends issue collectives on MPI_COMM_WORLD,
/// where two concurrent reductions from one process have no defined pairing at
/// all. Decision 23 in docs/DESIGN_DECISIONS.md records why the scratch array
/// was not simply moved to the stack, which would have hidden this rather than
/// fixed it. Build one backend per thread if you need several.
class Backend {
 public:
    Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;
    virtual ~Backend() = default;

    /// Stable identifier used in result rows and figures.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Workers actually in use, which may differ from what was requested if the
    /// system refused. Result rows record this value, not the request.
    [[nodiscard]] virtual int worker_count() const noexcept = 0;

    /// What the requested pinning actually achieved, for the `pinning_status`
    /// column of the result row.
    ///
    /// One of the four PinOutcome spellings, `not_requested`, `bound`,
    /// `not_applicable` or `refused`, with the number of workers the operating
    /// system refused appended after a colon when that number is not zero. It
    /// is the worst outcome any of this backend's workers recorded rather than
    /// a restatement of what was asked for, so a backend that bound three
    /// workers and was refused the fourth reports `refused:1`.
    ///
    /// Every backend that pins reports what its own threads did, not what the
    /// policy would have done. The distributed backends report what the thread
    /// inside this rank did.
    ///
    /// Only `not_requested` and `bound` can reach a result row. `make_backend`
    /// refuses a policy this machine cannot classify for, the shared memory
    /// backends throw from their constructors when a worker was refused, and
    /// the driver refuses to write a row whose `pinning` is not `none` and
    /// whose status is not `bound`. The other two values exist so that the
    /// failure carries a name.
    [[nodiscard]] virtual std::string pinning_status() const = 0;

    /// Apply \p body to a partition of [0, n).
    ///
    /// The partition covers the range exactly once with contiguous chunks. The
    /// body must be safe to run concurrently on disjoint chunks; the interface
    /// makes no other ordering promise.
    ///
    /// **A body may throw.** On the multithreaded backends the remaining chunks
    /// are still attempted, because a worker that stopped short would never
    /// reach the barrier the dispatch ends on, and the first exception thrown
    /// is then rethrown on the calling thread once every worker is back. The
    /// backend is usable afterwards. Anything a body throws travels: the type
    /// and the message arrive unchanged, since what crosses the thread boundary
    /// is a std::exception_ptr and not a description of one. The serial and
    /// distributed backends run the chunks on the calling thread, so there a
    /// throw simply propagates and the chunks after it do not run.
    ///
    /// \throws BackendFailure if the execution model reports an error, and
    ///         whatever \p body throws.
    virtual void parallel_for(Index n, const RangeBody& body) = 0;

    /// Reduce \p reducer over a partition of [0, n) starting from \p init.
    ///
    /// Combination is by addition. Under ReductionMode::Deterministic the
    /// partials are summed in ascending chunk index regardless of which worker
    /// produced them or when, so repeated runs and different worker counts
    /// agree bit for bit.
    ///
    /// Not reentrant, and not safe to call on one object from two threads. Each
    /// reducing backend writes the partials into scratch it owns and combines
    /// them afterwards, so two calls in flight on one object would interleave
    /// their slots and both would return a sum of the wrong terms. See the note
    /// on the class.
    ///
    /// A reducer that throws is treated exactly as a parallel_for body that
    /// does; see the note there.
    ///
    /// \throws BackendFailure if the execution model reports an error, and
    ///         whatever \p reducer throws.
    [[nodiscard]] virtual Real reduce(Index n, Real init, const RangeReducer& reducer) = 0;

    /// Synchronise all workers. A no operation for the serial backend; a real
    /// barrier for the thread pools; MPI_Barrier for the distributed backends.
    virtual void barrier() = 0;

    /// True on the rank that owns the terminal. Only this rank prints, per
    /// Section 9 of the specification.
    [[nodiscard]] virtual bool is_root() const noexcept { return true; }

    /// Number of distributed ranks. One for every shared memory backend.
    [[nodiscard]] virtual int rank_count() const noexcept { return 1; }

    /// This process's rank. Zero for every shared memory backend.
    [[nodiscard]] virtual int rank() const noexcept { return 0; }

    /// The rows of a distributed problem this process owns. For shared memory
    /// backends this is the whole range, which is what makes solver code
    /// identical across backends.
    [[nodiscard]] virtual Range local_rows(Index total_rows) const noexcept {
        return Range{0, total_rows};
    }

    /// Exchange halo rows with neighbours for a row decomposed grid.
    ///
    /// A no operation for shared memory backends, where neighbours are simply
    /// readable. Implemented by the MPI backends. \p row_stride is the number of
    /// values per grid row, or zero when \p grid is a flat vector with no row
    /// structure.
    ///
    /// The default checks its arguments even though it does nothing with them.
    /// A shared memory run is where a caller's sizes are exercised a million
    /// times and never inspected, and a size that is wrong there is wrong under
    /// MPI too, where it is a write past the end of the buffer.
    ///
    /// 	hrows InvalidArgument if \p grid is too short for \p row_stride and
    ///         \p total_rows.
    virtual void exchange_halo(VectorView grid, Index row_stride, Index total_rows) {
        detail::check_halo_arguments(grid, row_stride, total_rows);
    }

    /// Make a replicated flat vector consistent again.
    ///
    /// Each rank has updated only the entries in \p local and needs everyone
    /// else's. Unlike exchange_halo this makes no assumption about which rows a
    /// rank owns: the ranges are collected from the ranks themselves. That
    /// matters because a rank's share of the blocks in a block method covers a
    /// different set of rows than its share of the rows in a point method, and
    /// assuming otherwise silently gathers the wrong segments.
    ///
    /// A no operation for shared memory backends, which check the arguments
    /// anyway, for the reason exchange_halo gives.
    ///
    /// 	hrows InvalidArgument if \p local is not a range inside \p data.
    virtual void gather_rows(VectorView data, Range local) {
        detail::check_gather_arguments(data, local);
    }

    /// Run \p local_work under global sequential ordering across ranks.
    ///
    /// Natural ordering Gauss Seidel and SOR are sequentially dependent by
    /// definition: unknown k reads the already updated unknown k-1. Preserving
    /// that across a row decomposition means rank r may not start until rank
    /// r-1 has finished and handed over its boundary row, which is the
    /// classical pipelined Gauss Seidel. Shared memory backends simply call
    /// \p local_work, since their rows are already in one address space and
    /// already in order.
    ///
    /// Doing this rather than quietly substituting a red black reordering is
    /// what lets every backend produce bit identical iterates for these
    /// methods, and it makes the resulting lack of parallel speedup an honest
    /// measurement instead of a hidden change of algorithm.
    ///
    /// \param forward true to order ranks 0, 1, 2, ...; false to reverse them,
    ///        as a backward sweep requires.
    /// \param data the state the ordering applies to, so a distributed backend
    ///        can hand the updated values on with the turn. Empty for shared
    ///        memory backends, which need no transfer.
    /// \param row_stride values per grid row when \p data is a padded grid, in
    ///        which case only the single boundary row needs to travel. Zero
    ///        means \p data is a flat vector with no row structure and the
    ///        whole of it is passed along the chain.
    /// \param total_rows interior rows of the grid, ignored when row_stride is
    ///        zero.
    ///
    /// Not virtual, and that is the point of it. The defaults used to sit on a
    /// virtual function whose one override omitted them, so the three trailing
    /// arguments existed through a `Backend&` and did not exist through an
    /// `MpiBackend&`, for the same object and the same call. A default argument
    /// is chosen from the static type of the expression, never from the dynamic
    /// one, so this is not a fixable property of the override: the only way to
    /// have one answer is to have one declaration. Section 4.7 lists it.
    void run_ordered(OrderedWork local_work,
                     bool forward,
                     VectorView data = {},
                     Index row_stride = 0,
                     Index total_rows = 0) {
        run_ordered_impl(local_work, forward, data, row_stride, total_rows);
    }

    /// Configuration this backend was built with, for the result row.
    [[nodiscard]] virtual const Config& config() const noexcept = 0;

 protected:
    /// What run_ordered does, which is what a backend overrides.
    ///
    /// No default arguments here, deliberately and permanently: every caller
    /// arrives through the wrapper above, which has already filled them in.
    virtual void run_ordered_impl(OrderedWork local_work,
                                  bool /*forward*/,
                                  VectorView /*data*/,
                                  Index /*row_stride*/,
                                  Index /*total_rows*/) {
        local_work();
    }
};

/// Every registered backend name, in registration order.
///
/// That is the built in backends this build contains, which CMake decides
/// through compile definitions, followed by anything a consumer registered
/// through pnl/backend/registry.hpp. The CLI uses it to refuse an unavailable
/// backend with a clear message rather than a link error, and the sweep driver
/// and the equivalence suite iterate it, so its order is part of the contract.
[[nodiscard]] std::vector<std::string> available_backends();

namespace detail {

/// The registry lookup behind make_backend, out of line in factory.cpp.
///
/// Separated from make_backend so that the contraction probe below runs in the
/// caller's translation unit rather than in this library's. Nothing else lives
/// in here: the topology probe, the pinning classification refusal and the
/// registry lookup are the whole of it.
///
/// \throws InvalidArgument, BackendFailure as make_backend documents.
[[nodiscard]] std::unique_ptr<Backend> make_backend_impl(std::string_view name,
                                                         const Config& config);

}  // namespace detail

/// Construct a backend by name.
///
/// Recognised names: "serial", "openmp", "pthreads", "jthread", "mpi",
/// "hybrid", plus whatever this program registered itself. Which of the built
/// in names a given build actually contains is what available_backends() says.
/// There is no device name in that list; see the note at the top of this file
/// and decision 9.
///
/// Before it looks at the name, this calls pnl::assert_no_contraction() from
/// pnl/core/contract.hpp. That is the runtime half of the numerical contract:
/// `-ffp-contract=off` has no predefined macro to test for, `-ffp-contract=fast`
/// is GCC's default, and this library is mostly headers, so a build that fuses a
/// multiply and an add would otherwise produce results that are wrong in the
/// last bit with no diagnostic anywhere. Every path that runs numerics builds a
/// backend first, which is why the check lives here.
///
/// **This function is inline on purpose, and that is the whole point of it.**
/// The probe checks the flags of the translation unit it is compiled into.
/// While it was called from `make_backend` in `factory.cpp` it therefore checked
/// `factory.cpp`, which is compiled with this library's own flags and is not
/// where the risk is: three quarters of this library is headers, so a consumer's
/// arithmetic is compiled on the consumer's command line. Being inline, the
/// probe is emitted into the caller's translation unit and checks the flags that
/// actually apply to the code the caller is about to run. It runs on every call
/// rather than once behind a flag, because a function local static in an inline
/// function is one object for the whole program: it would check whichever
/// translation unit happened to construct the first backend and silently exempt
/// every other one, which is exactly the hole this arrangement exists to close.
/// The cost is three volatile stores, a multiply, an add and a compare, against
/// a call that starts a thread pool. See NUM-05 and the read of contract.hpp.
///
/// \throws InvalidArgument if the name is unknown or the backend was not
///         compiled into this build.
/// \throws BackendFailure if construction fails, for example a thread that will
///         not start or a pinning request the operating system refused.
/// \throws ConfigurationError if the calling translation unit was compiled with
///         floating point contraction enabled, which voids the bit identity
///         guarantee.
[[nodiscard]] inline std::unique_ptr<Backend> make_backend(std::string_view name,
                                                           const Config& config) {
    assert_no_contraction();
    return detail::make_backend_impl(name, config);
}

/// Number of logical CPUs visible to this process, respecting any affinity mask
/// already applied. Used to size default worker counts and to bound the sweep.
[[nodiscard]] int available_logical_cpus();

}  // namespace pnl::backend
