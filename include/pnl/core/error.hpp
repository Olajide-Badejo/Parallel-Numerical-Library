// SPDX-License-Identifier: MIT
#pragma once

/// \file error.hpp
/// Exception hierarchy. Every public function documents which of these it can
/// throw, per the style rules.

#include <cstddef>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pnl {

/// Base of every exception the library raises.
class Error : public std::runtime_error {
 public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

/// A caller supplied argument is outside the documented domain, for example a
/// bracketing root finder given an interval whose endpoints share a sign, or an
/// SOR relaxation factor outside (0, 2).
class InvalidArgument : public Error {
 public:
    explicit InvalidArgument(const std::string& what) : Error("invalid argument: " + what) {}
};

/// The problem itself is degenerate: a singular pivot in LU, a breakdown of the
/// conjugate gradient recurrence, a non positive definite matrix handed to CG.
class NumericalFailure : public Error {
 public:
    explicit NumericalFailure(const std::string& what) : Error("numerical failure: " + what) {}
};

/// An iteration cap was reached before the tolerance. Solvers report this
/// through the diagnostics record by default; the exception exists for callers
/// that opt into throwing, so a non convergence can never be read as success.
class ConvergenceFailure : public Error {
 public:
    explicit ConvergenceFailure(const std::string& what) : Error("no convergence: " + what) {}
};

/// A backend could not honour the request: an unavailable execution model, a
/// thread that failed to start, a pinning request the operating system refused,
/// or a failing MPI or CUDA call surfaced through PNL_MPI_CHECK or
/// PNL_CUDA_CHECK.
class BackendFailure : public Error {
 public:
    explicit BackendFailure(const std::string& what) : Error("backend failure: " + what) {}
};

/// The build itself violates a contract the library depends on: a compile flag
/// the numerical claims need is absent, or one that voids them is present.
///
/// It is separate from BackendFailure because nothing about the request is
/// wrong and no execution model failed. The answer is to fix the compile line,
/// not to retry with another backend or another size, and a caller that catches
/// BackendFailure to fall back to serial must not swallow this.
class ConfigurationError : public Error {
 public:
    explicit ConfigurationError(const std::string& what) : Error("configuration: " + what) {}
};

namespace detail {

/// Format a check failure with its origin, used by the PNL_MPI_CHECK and
/// PNL_CUDA_CHECK macros and by internal preconditions.
[[nodiscard]] inline std::string describe(std::string_view what,
                                          const std::source_location& where) {
    return std::string(what) + " at " + where.file_name() + ":" + std::to_string(where.line()) +
           " in " + where.function_name();
}

/// Message for a view that is the wrong length, naming the method, the argument
/// and both sizes.
///
/// Every one of these three is called only after the check it belongs to has
/// already failed. They return an owning string, and require() takes a view for
/// the reason its own comment gives: these preconditions sit at the top of the
/// sweeps, so a message built unconditionally would be one allocation per sweep
/// inside the timed region, which is exactly what MEAS-10 removed. The call
/// pattern is therefore an `if` around a `require(false, ...)`, never a
/// `require(condition, build_message())`.
[[nodiscard]] inline std::string wrong_size(std::string_view method,
                                            std::string_view argument,
                                            std::size_t expected,
                                            std::size_t given) {
    return std::string(method) + " needs " + std::string(argument) + " to hold " +
           std::to_string(expected) + " values and it holds " + std::to_string(given);
}

/// Message for a view that is shorter than a method needs, where longer is
/// allowed.
[[nodiscard]] inline std::string too_small(std::string_view method,
                                           std::string_view argument,
                                           std::size_t minimum,
                                           std::size_t given) {
    return std::string(method) + " needs " + std::string(argument) + " to hold at least " +
           std::to_string(minimum) + " values and it holds " + std::to_string(given);
}

/// Message for two views that share storage where the method requires them not
/// to.
[[nodiscard]] inline std::string aliased(std::string_view method,
                                         std::string_view first,
                                         std::string_view second) {
    return std::string(method) + " needs " + std::string(first) + " and " + std::string(second) +
           " to be separate buffers, and they share storage";
}

}  // namespace detail

/// Internal precondition. Throws InvalidArgument when \p condition is false.
///
/// \p what is a view rather than a string because this is called from inside
/// the sweeps, where it runs once per iteration on the path that succeeds.
/// Taking a std::string here built one from the literal on every call, which
/// is an allocation in the timed region for any message longer than the small
/// string buffer, and most of these messages are. A view of a literal costs
/// nothing and the string is built only when the check fails. See MEAS-10.
///
/// A caller whose message has to be computed must still guard the computation
/// itself: passing a function that returns a std::string evaluates it whether
/// or not the check fires.
inline void require(bool condition,
                    std::string_view what,
                    const std::source_location& where = std::source_location::current()) {
    if (!condition) throw InvalidArgument(detail::describe(what, where));
}

}  // namespace pnl
