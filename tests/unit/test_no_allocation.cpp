// SPDX-License-Identifier: MIT
/// \file test_no_allocation.cpp
/// The gate for phase A5: the timed region allocates nothing.
///
/// This is a property of the code, not of the machine, which is the whole
/// reason it is the gate. Run to run spread on a shared WSL2 guest can move
/// either way for reasons that have nothing to do with this change, so the
/// timing is recorded in PROGRESS.md as an observation and the assertion here
/// is the thing that has to stay true.
///
/// How it works. This translation unit replaces the global operator new and
/// operator delete, so every allocation anywhere in the process, on any thread,
/// goes through the counter below. The counter is armed only between the two
/// ends of the timed region, and it is armed from inside
/// pnl::bench::timed_repetition through its observer hook, so what is counted
/// is exactly what is timed rather than something close to it.
///
/// The warm up absorbs whatever a first call legitimately has to allocate: the
/// OpenMP runtime creating its team, a backend sizing its partials array, the
/// C++ runtime's own one time setup. The assertion is about the second call and
/// every call after it.
///
/// What it found on its first run is recorded as MEAS-10 in the engineering
/// log. In short: allocating the state was the largest part of the problem but
/// nowhere near all of it. Every parallel_for, reduce and run_ordered built a
/// std::function whose capture was too large for its small object buffer, every
/// precondition built a std::string from its message literal, five of the
/// twelve solvers built a std::function sweep too large to store locally, the
/// block methods allocated a full state snapshot and a scratch vector per chunk
/// inside the sweep, and the progress bar allocated a string for the longer
/// solver names.

#include <pnl/backend/backend.hpp>
#include <pnl/bench/timed_solve.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <pnl_test.hpp>
#include <string>
#include <vector>

namespace {

/// Allocations seen while armed. Relaxed ordering throughout: the workers are
/// joined by the dispatch barrier before the count is read, so the barrier
/// supplies the ordering and the counter only has to be atomic.
std::atomic<long long> allocation_count{0};
std::atomic<bool> counting_armed{false};

void count_allocation() noexcept {
    if (counting_armed.load(std::memory_order_relaxed)) {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
}

/// The observer handed to the timed region. Arming inside it is what makes the
/// count a statement about the timed region and not about the call around it.
void observe_timed_region(bool inside) noexcept {
    counting_armed.store(inside, std::memory_order_relaxed);
}

/// Where the deliberate allocation of the self check publishes its pointer.
///
/// A new expression is one of the few things a compiler is allowed to delete
/// outright: the standard permits an implementation to omit a replaceable
/// allocation whose storage does not escape, and clang at -O3 takes that
/// permission, so the self check allocated nothing, counted nothing, and
/// reported that the instrument was broken. GCC never did it, which is why it
/// took a second compiler to find. A volatile store publishes the pointer and
/// makes the omission illegal. See BUILD-05 in docs/ENGINEERING_LOG.md.
void* volatile allocation_escape = nullptr;

}  // namespace

// The replacements. Every form the standard lets a program replace is replaced,
// so nothing slips past by taking a different overload, and each pairs with a
// deallocation that frees what its partner allocated.
//
// Every one of them is noinline. Without it GCC inlines the replacement into
// the caller, sees a pointer whose provenance is operator new handed to free,
// and reports -Wmismatched-new-delete, which with -Werror is a build failure.
// The pairing is correct, and it is the textbook one: these operators are
// malloc and free, so free is exactly the right deallocator. Keeping the bodies
// out of line leaves the caller looking at a plain call to operator delete,
// which is what it is. It also keeps the counter from being optimised away.

[[gnu::noinline]] void* operator new(std::size_t size) {
    if (size == 0) size = 1;
    void* memory = std::malloc(size);
    if (memory == nullptr) throw std::bad_alloc();
    count_allocation();
    return memory;
}

[[gnu::noinline]] void* operator new[](std::size_t size) {
    return ::operator new(size);
}

[[gnu::noinline]] void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    if (size == 0) size = 1;
    void* memory = std::malloc(size);
    if (memory != nullptr) count_allocation();
    return memory;
}

[[gnu::noinline]] void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
    return ::operator new(size, tag);
}

[[gnu::noinline]] void* operator new(std::size_t size, std::align_val_t alignment) {
    const auto align = static_cast<std::size_t>(alignment);
    if (size == 0) size = align;
    // aligned_alloc wants a size that is a multiple of the alignment.
    const std::size_t rounded = ((size + align - 1) / align) * align;
    void* memory = std::aligned_alloc(align, rounded);
    if (memory == nullptr) throw std::bad_alloc();
    count_allocation();
    return memory;
}

[[gnu::noinline]] void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

[[gnu::noinline]] void operator delete(void* memory) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete[](void* memory) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete(void* memory, const std::nothrow_t&) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete[](void* memory, const std::nothrow_t&) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete(void* memory, std::align_val_t) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete[](void* memory, std::align_val_t) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete(void* memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

[[gnu::noinline]] void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

namespace {

using namespace pnl;
using solvers::all_solver_names;
using solvers::make_solver;
using solvers::RunMode;
using solvers::SolverOptions;

/// Small enough that twelve solvers on two backends is a few seconds, large
/// enough that every sweep has real chunks to distribute.
constexpr Index GRID_SIDE = 63;

/// Enough iterations that anything allocating per iteration is counted many
/// times over, and few enough that the suite stays fast.
constexpr Index ITERATIONS = 12;

[[nodiscard]] SolverOptions timed_options() {
    SolverOptions options;
    // Fixed iterations is what the sweep measures, and it keeps the residual
    // check firing every iteration so the reduction path is exercised too.
    options.mode = RunMode::FixedIterations;
    options.max_iterations = ITERATIONS;
    options.check_interval = 1;
    options.record_history = false;
    return options;
}

/// Allocations inside one timed repetition, after a warm up on the same
/// workspace.
[[nodiscard]] long long allocations_in_timed_region(const std::string& solver_name,
                                                    const std::string& backend_name,
                                                    int workers) {
    problems::Poisson2D problem(GRID_SIDE, problems::PoissonRhs::SpectrallyRich);
    backend::Config config;
    config.workers = workers;
    auto execution = backend::make_backend(backend_name, config);
    auto solver = make_solver(solver_name);
    auto workspace = solver->make_workspace(problem);
    const SolverOptions options = timed_options();

    // The warm up, unarmed. Anything a first dispatch has to allocate, such as
    // a pool sizing its partials array or an OpenMP runtime starting its team,
    // is absorbed here.
    (void)bench::timed_repetition(*solver, problem, *execution, options, workspace);

    allocation_count.store(0, std::memory_order_relaxed);
    (void)bench::timed_repetition(
        *solver, problem, *execution, options, workspace, observe_timed_region);
    const long long counted = allocation_count.load(std::memory_order_relaxed);
    counting_armed.store(false, std::memory_order_relaxed);
    return counted;
}

void check_backend(const std::string& backend_name, int workers) {
    std::string failures;
    for (const auto& solver_name : all_solver_names()) {
        const long long counted = allocations_in_timed_region(solver_name, backend_name, workers);
        if (counted != 0) {
            if (!failures.empty()) failures += ", ";
            failures += solver_name + " allocated " + std::to_string(counted) + " times";
        }
    }
    PNL_REQUIRE_MESSAGE(failures.empty(),
                        "the timed region allocated on the " + backend_name + " backend at " +
                            std::to_string(workers) + " workers: " + failures);
}

}  // namespace

PNL_TEST("no_allocation/the timed region allocates nothing on serial") {
    check_backend("serial", 1);
}

#if defined(PNL_WITH_OPENMP)
PNL_TEST("no_allocation/the timed region allocates nothing on openmp") {
    const int workers = std::min(4, std::max(1, backend::available_logical_cpus()));
    check_backend("openmp", workers);
}
#else
// A build whose compiler cannot find an OpenMP runtime is a configuration this
// project supports: CMakeLists.txt says so, drops the backend and carries on.
// This case named "openmp" unconditionally and therefore turned that supported
// configuration into a test failure. The name is kept rather than dropped, so
// the skip is visible in the test output instead of the binary quietly holding
// one case fewer, and it asserts the reason for the skip so that a build which
// does have the backend can never take this path. See BUILD-06.
PNL_TEST("no_allocation/openmp is not in this build, so its case is skipped") {
    const std::vector<std::string> names = backend::available_backends();
    PNL_REQUIRE_MESSAGE(std::find(names.begin(), names.end(), "openmp") == names.end(),
                        "PNL_WITH_OPENMP is not defined, so the no allocation gate skipped the "
                        "openmp backend, but make_backend offers it: the gate was skipped on a "
                        "backend that is present");
}
#endif

PNL_TEST("no_allocation/the counter sees an allocation when there is one") {
    // The gate above is only evidence if the instrument works, and a counter
    // that counts nothing passes every test there is. This one allocates on
    // purpose inside the armed window and requires the count to move.
    allocation_count.store(0, std::memory_order_relaxed);
    observe_timed_region(true);
    auto* deliberate = new double[512];
    // The volatile store is what keeps this allocation alive. Without it clang
    // removes the whole new expression, which the standard allows and which
    // leaves the check asserting that a counter counted an allocation nobody
    // made. BUILD-05.
    allocation_escape = deliberate;
    deliberate[0] = 1.0;
    observe_timed_region(false);
    const long long counted = allocation_count.load(std::memory_order_relaxed);
    delete[] deliberate;
    allocation_escape = nullptr;
    PNL_REQUIRE_MESSAGE(counted >= 1,
                        "the counting allocator saw " + std::to_string(counted) +
                            " allocations where a deliberate one had just been made, so the "
                            "gate above proves nothing");
}
