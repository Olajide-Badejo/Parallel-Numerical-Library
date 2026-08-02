#pragma once

/// \file error.hpp
/// Exception hierarchy. Every public function documents which of these it can
/// throw, per the style rules.

#include <source_location>
#include <stdexcept>
#include <string>

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
/// or a failing MPI or CUDA call surfaced through MPI_CHECK or CUDA_CHECK.
class BackendFailure : public Error {
   public:
    explicit BackendFailure(const std::string& what) : Error("backend failure: " + what) {}
};

namespace detail {

/// Format a check failure with its origin, used by the MPI_CHECK and
/// CUDA_CHECK macros and by internal preconditions.
[[nodiscard]] inline std::string describe(const std::string& what,
                                          const std::source_location& where) {
    return what + " at " + where.file_name() + ":" + std::to_string(where.line()) + " in " +
           where.function_name();
}

}  // namespace detail

/// Internal precondition. Throws InvalidArgument when \p condition is false.
inline void require(bool condition, const std::string& what,
                    const std::source_location& where = std::source_location::current()) {
    if (!condition) throw InvalidArgument(detail::describe(what, where));
}

}  // namespace pnl
