// SPDX-License-Identifier: MIT
#pragma once

/// \file function_ref.hpp
/// A non owning reference to a callable, for parameters that are invoked during
/// the call and never stored past it.
///
/// Why this exists rather than std::function. The chunk level callbacks of the
/// backend interface are handed a lambda that captures the sweep's pointers and
/// extents, which on every backend here is forty bytes or so of captures.
/// libstdc++ stores a callable inside a std::function only when it is trivially
/// copyable and fits in sixteen bytes, so every one of those lambdas was
/// allocated on the heap, and every parallel_for, reduce and run_ordered
/// therefore performed one allocation and one free per call. A Jacobi iteration
/// makes three such calls, so the timed region paid three allocations per
/// iteration for a type erasure that outlives nothing. That is measurement
/// finding MEAS-10.
///
/// The interface promise this type encodes is the one the backends already
/// kept: the body is called before the function that received it returns, and
/// no backend stores it. The thread pools already held a bare pointer to the
/// caller's std::function for exactly that reason.
///
/// It is trivially copyable and two words wide, so passing it by value is free
/// and copying it into a worker thread is a register move.
///
/// \warning It refers to the callable, so it must not outlive it. Never store
///          one in a member, a container, or anything that survives the call.

#include <memory>
#include <type_traits>
#include <utility>

namespace pnl {

template<typename Signature>
class FunctionRef;

/// A reference to something callable as `R(Args...)`.
template<typename R, typename... Args>
class FunctionRef<R(Args...)> {
 public:
    /// Bind to \p callable. Implicit on purpose, so a call site keeps passing a
    /// lambda literal and nothing at any call site had to change.
    template<typename F>
        requires(!std::is_same_v<std::remove_cvref_t<F>, FunctionRef> &&
                 std::is_invocable_r_v<R, F&, Args...>)
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr FunctionRef(F&& callable) noexcept
        : object_(const_cast<void*>(static_cast<const void*>(std::addressof(callable)))),
          invoke_([](void* object, Args... arguments) -> R {
              using Target = std::remove_reference_t<F>;
              return (*static_cast<Target*>(object))(std::forward<Args>(arguments)...);
          }) {}

    R operator()(Args... arguments) const {
        return invoke_(object_, std::forward<Args>(arguments)...);
    }

 private:
    void* object_;
    R (*invoke_)(void*, Args...);
};

}  // namespace pnl
