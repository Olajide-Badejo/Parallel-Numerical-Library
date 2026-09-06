// SPDX-License-Identifier: MIT
/// \file builtin_backends.cpp
/// The backends this build ships, and the registry they live in.
///
/// One file, and one guard block per optional model, so adding a backend to the
/// library means adding a line here rather than editing an `if` chain in the
/// factory and a list beside it and hoping the two stay in step. A header only
/// backend needs no translation unit of its own for the same reason: its
/// registration is a line in this file, not a static object in a file that
/// exists only to hold one.
///
/// **This file also owns `backend_registry()`, and that is load bearing.**
/// `pnl_core` is a static library, and a linker pulls an object file out of an
/// archive only when something references a symbol it defines. Registration
/// objects sitting alone at namespace scope define nothing anybody names, so
/// this object file would be dropped, the registry would come up empty, and
/// `pnl --backend serial` would fail with "not available in this build" on a
/// build that contains it. Defining the accessor here makes the reference
/// unavoidable: `make_backend_impl` in factory.cpp calls `backend_registry()`,
/// so the object file is always linked and the registrations always run.
///
/// Registering from inside the accessor rather than from namespace scope
/// objects also removes the static initialisation order question. The built ins
/// are in the registry before it is first handed to anybody, whatever order a
/// consumer's own registrations run in.

#include <pnl/backend/jthread_pool.hpp>
#include <pnl/backend/pthreads.hpp>
#include <pnl/backend/registry.hpp>
#include <pnl/backend/serial.hpp>

#if defined(PNL_WITH_OPENMP)
#include <pnl/backend/openmp.hpp>
#endif

#if defined(PNL_WITH_MPI)
#include <pnl/backend/hybrid.hpp>
#include <pnl/backend/mpi.hpp>
#endif

#include <memory>

namespace pnl::backend {

namespace {

/// A factory that constructs \p BackendType from the two arguments every built
/// in backend's constructor takes.
template<typename BackendType>
[[nodiscard]] BackendFactory construct() {
    return [](const Config& config, const TopologyReport& topology) -> std::unique_ptr<Backend> {
        return std::make_unique<BackendType>(config, topology);
    };
}

/// The order here is the order `--list`, `available_backends()` and the sweep
/// driver see, and it is the order the fixed list held before this registry
/// existed. Do not sort it.
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_factory("serial", construct<SerialBackend>());
#if defined(PNL_WITH_OPENMP)
    registry.register_factory("openmp", construct<OpenMpBackend>());
#endif
    registry.register_factory("pthreads", construct<PthreadsBackend>());
    registry.register_factory("jthread", construct<JthreadBackend>());
#if defined(PNL_WITH_MPI)
    registry.register_factory("mpi", construct<MpiBackend>());
#if defined(PNL_WITH_OPENMP)
    registry.register_factory("hybrid", construct<HybridBackend>());
#endif
#endif
}

}  // namespace

BackendRegistry& backend_registry() {
    // A function local static, so the built ins are registered exactly once and
    // on first use rather than before main. Initialisation of a function local
    // static is thread safe; what follows it is not, which is why registration
    // belongs at namespace scope in the registering translation unit.
    static BackendRegistry registry = [] {
        BackendRegistry created;
        register_builtin_backends(created);
        return created;
    }();
    return registry;
}

}  // namespace pnl::backend
