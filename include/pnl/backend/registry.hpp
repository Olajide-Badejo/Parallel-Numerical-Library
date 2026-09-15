// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// The backend extension point.
///
/// Until release 1.1.0 `make_backend` was an `if` chain over five string
/// literals and `available_backends()` was a list beside it, so an execution
/// model this repository did not ship could not be reached through the
/// library's own entry point at all. That is finding 4.8 of the version 2
/// specification. A third party now writes one namespace scope object in a
/// translation unit of their own:
///
/// ```cpp
/// namespace {
/// const pnl::backend::BackendRegistration my_backend(
///     "mine", [](const pnl::backend::Config& config, const pnl::backend::TopologyReport&) {
///         return std::make_unique<MyBackend>(config);
///     });
/// }  // namespace
/// ```
///
/// and `make_backend("mine", config)` builds it, `available_backends()` names
/// it, and every solver in the zoo runs over it without a line of solver code
/// changing. `examples/custom_backend.cpp` is that program, in full.
///
/// **Order is part of the contract.** `names()` returns registration order, and
/// the built in backends are registered in exactly the order the old list wrote
/// them, because `benchmarks/run_sweep.py`, the equivalence suite and the
/// `--list` output all read it. A registration a third party adds lands after
/// all of them, wherever their static initialiser happens to run, so nothing
/// that was in the list before can move.
///
/// **Where the built ins register, and why it is not a header.** They register
/// from `src/backend/builtin_backends.cpp`, inside `backend_registry()` itself,
/// which is defined there. A static library drops an object file nothing
/// references, so registration objects sitting alone in a translation unit
/// would be linked out of `libpnl_core.a` without a diagnostic and a consumer
/// would get an empty registry. Putting the accessor in the same file makes the
/// reference unavoidable: `make_backend` cannot look a name up without calling
/// `backend_registry()`, and calling it pulls the object file in. It also
/// disposes of the static initialisation order question, because the built ins
/// are registered by the accessor rather than before `main`, so a third party's
/// namespace scope registration always finds a registry that is already
/// populated whenever it runs.
///
/// **Threading.** Registration is not thread safe against itself or against a
/// lookup. Register before the first `make_backend`, which a namespace scope
/// object does by construction. Lookup after that point is read only and
/// therefore safe from several threads.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/error.hpp>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pnl::backend {

/// What a backend factory is handed.
///
/// The two arguments are exactly what `make_backend` passed to a built in
/// constructor before this registry existed: the caller's configuration, and
/// the process wide topology report, which the factory reads when it pins and
/// ignores when it does not.
using BackendFactory =
    std::function<std::unique_ptr<Backend>(const Config& config, const TopologyReport& topology)>;

/// Name to factory, in registration order.
class BackendRegistry {
 public:
    /// Add \p factory under \p name.
    ///
    /// \throws InvalidArgument if \p name is empty or already registered. A
    ///         duplicate is refused rather than replaced: silently shadowing
    ///         "serial" would change what every equivalence test compares
    ///         against, and a registry that answers to two different objects
    ///         under one name is worse than a build that stops.
    void register_factory(std::string name, BackendFactory factory) {
        require(!name.empty(), "a backend name must not be empty");
        require(static_cast<bool>(factory), "a backend factory must not be empty");
        if (contains(name)) {
            require(false, "a backend named '" + name + "' is already registered");
        }
        entries_.push_back(Entry{std::move(name), std::move(factory)});
    }

    /// Whether \p name is registered.
    [[nodiscard]] bool contains(std::string_view name) const noexcept {
        for (const auto& entry : entries_) {
            if (entry.name == name) return true;
        }
        return false;
    }

    /// Every registered name, in registration order.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(entries_.size());
        for (const auto& entry : entries_) result.push_back(entry.name);
        return result;
    }

    /// The registered names as one comma separated string, for a message.
    [[nodiscard]] std::string name_list() const {
        std::string known;
        for (const auto& entry : entries_) {
            if (!known.empty()) known += ", ";
            known += entry.name;
        }
        return known;
    }

    /// Construct the backend registered under \p name.
    ///
    /// \throws InvalidArgument if the name is unknown or the backend was not
    ///         compiled into this build.
    /// \throws BackendFailure if the factory itself fails.
    [[nodiscard]] std::unique_ptr<Backend> create(std::string_view name,
                                                  const Config& config,
                                                  const TopologyReport& topology) const {
        for (const auto& entry : entries_) {
            if (entry.name == name) return entry.factory(config, topology);
        }
        throw InvalidArgument("backend '" + std::string(name) +
                              "' is not available in this build; available backends are " +
                              name_list());
    }

 private:
    struct Entry {
        std::string name;
        BackendFactory factory;
    };

    std::vector<Entry> entries_;
};

/// The process wide registry, with the built in backends already in it.
///
/// Defined in `src/backend/builtin_backends.cpp`, which is what keeps that
/// object file in the archive; see the note at the top of this file.
[[nodiscard]] BackendRegistry& backend_registry();

/// One line self registration, for a namespace scope object.
///
/// \throws InvalidArgument through register_factory, which at namespace scope
///         means the program terminates before main. That is the intended
///         outcome for a duplicate name: there is no correct way to continue.
struct BackendRegistration {
    BackendRegistration(std::string name, BackendFactory factory) {
        backend_registry().register_factory(std::move(name), std::move(factory));
    }
};

}  // namespace pnl::backend
