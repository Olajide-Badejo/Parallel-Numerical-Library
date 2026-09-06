// SPDX-License-Identifier: MIT
#pragma once

/// \file pnl_test.hpp
/// A small assertion framework.
///
/// Deliberately dependency free. The test suite has to run in CI on a plain
/// Ubuntu image and under mpirun with several ranks, and pulling in a framework
/// would add a fetch step and a set of MPI interactions to debug for no benefit
/// at this size. What is needed is named cases, an assertion that prints both
/// values, a relative comparison for floating point, and an exit status.

#include <pnl/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pnl::test {

struct Case {
    std::string name;
    std::function<void()> body;
};

/// Thrown by the assertion macros. Caught by the runner, which turns it into a
/// failure line rather than a crash.
struct Failure {
    std::string message;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> body) {
        registry().push_back(Case{std::move(name), std::move(body)});
    }
};

/// Absolute difference, safe for infinities and NaN.
[[nodiscard]] inline bool close_absolute(Real a, Real b, Real tolerance) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= tolerance;
}

/// Relative difference, falling back to absolute near zero.
[[nodiscard]] inline bool close_relative(Real a, Real b, Real tolerance) {
    if (std::isnan(a) || std::isnan(b)) return false;
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tolerance * scale;
}

[[nodiscard]] inline std::string format(Real value) {
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

/// Called after each case with the status that case produced, 0 for a pass and
/// 1 for a failure, and returning the status the runner should act on.
///
/// It exists for the distributed runner, which combines the flag across ranks
/// here. That is what stops one rank walking into the next case while another
/// has already left the previous one: in a suite whose cases are collective,
/// a rank that carries on alone does not fail, it hangs, and Section 4.7
/// records exactly that against `test_mpi.cpp`.
using CaseHook = std::function<int(const std::string& name, int status)>;

/// Run every registered case whose name contains \p filter.
///
/// \param after_case combines the per case status, see CaseHook. When one is
///        installed the run stops at the first case that failed anywhere,
///        because for the runner that needs a hook at all, carrying on is the
///        failure mode. Without one, which is every single process binary here,
///        every case runs and the failures are counted, as before.
/// \returns the process exit status: 0 when everything passed.
[[nodiscard]] inline int run_all(std::string_view filter = {},
                                 const CaseHook& after_case = CaseHook{}) {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    for (const auto& test_case : registry()) {
        if (!filter.empty() && test_case.name.find(filter) == std::string::npos) {
            ++skipped;
            continue;
        }
        int status = 0;
        try {
            test_case.body();
            std::printf("  pass  %s\n", test_case.name.c_str());
            ++passed;
        } catch (const Failure& failure) {
            std::printf(
                "  FAIL  %s\n        %s\n", test_case.name.c_str(), failure.message.c_str());
            ++failed;
            status = 1;
        } catch (const std::exception& error) {
            std::printf("  FAIL  %s\n        unexpected exception: %s\n",
                        test_case.name.c_str(),
                        error.what());
            ++failed;
            status = 1;
        }
        if (!after_case) continue;
        if (after_case(test_case.name, status) != 0) {
            std::printf("  stop  a case failed here or on another participant\n");
            if (failed == 0) failed = 1;
            break;
        }
    }
    std::printf("%d passed, %d failed", passed, failed);
    if (skipped > 0) std::printf(", %d filtered out", skipped);
    std::printf("\n");
    return failed == 0 ? 0 : 1;
}

}  // namespace pnl::test

#define PNL_CONCAT_INNER(a, b) a##b
#define PNL_CONCAT(a, b) PNL_CONCAT_INNER(a, b)

/// Declare a test case.
#define PNL_TEST(name)                                                       \
    static void PNL_CONCAT(pnl_test_body_, __LINE__)();                      \
    static const ::pnl::test::Registrar PNL_CONCAT(pnl_test_reg_, __LINE__)( \
        name, PNL_CONCAT(pnl_test_body_, __LINE__));                         \
    static void PNL_CONCAT(pnl_test_body_, __LINE__)()

#define PNL_REQUIRE(condition)                                                           \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            throw ::pnl::test::Failure{std::string("required: ") + #condition + " at " + \
                                       __FILE__ + ":" + std::to_string(__LINE__)};       \
        }                                                                                \
    } while (false)

#define PNL_REQUIRE_MESSAGE(condition, message)                                                   \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            throw ::pnl::test::Failure{std::string(message) + "\n        (" + #condition +        \
                                       " at " + __FILE__ + ":" + std::to_string(__LINE__) + ")"}; \
        }                                                                                         \
    } while (false)

/// Exact equality, used where the design promises bit identical results.
#define PNL_REQUIRE_EXACT(a, b)                                                                   \
    do {                                                                                          \
        const ::pnl::Real pnl_lhs = (a);                                                          \
        const ::pnl::Real pnl_rhs = (b);                                                          \
        if (!(pnl_lhs == pnl_rhs)) {                                                              \
            throw ::pnl::test::Failure{std::string("expected bit identical values at ") +         \
                                       __FILE__ + ":" + std::to_string(__LINE__) + "\n        " + \
                                       #a + " = " + ::pnl::test::format(pnl_lhs) + "\n        " + \
                                       #b + " = " + ::pnl::test::format(pnl_rhs)};                \
        }                                                                                         \
    } while (false)

#define PNL_REQUIRE_CLOSE(a, b, tolerance)                                                        \
    do {                                                                                          \
        const ::pnl::Real pnl_lhs = (a);                                                          \
        const ::pnl::Real pnl_rhs = (b);                                                          \
        if (!::pnl::test::close_relative(pnl_lhs, pnl_rhs, (tolerance))) {                        \
            throw ::pnl::test::Failure{std::string("values differ by more than ") +               \
                                       ::pnl::test::format(tolerance) + " relative at " +         \
                                       __FILE__ + ":" + std::to_string(__LINE__) + "\n        " + \
                                       #a + " = " + ::pnl::test::format(pnl_lhs) + "\n        " + \
                                       #b + " = " + ::pnl::test::format(pnl_rhs)};                \
        }                                                                                         \
    } while (false)

#define PNL_REQUIRE_THROWS(expression, exception_type)                                           \
    do {                                                                                         \
        bool pnl_threw = false;                                                                  \
        try {                                                                                    \
            (void)(expression);                                                                  \
        } catch (const exception_type&) {                                                        \
            pnl_threw = true;                                                                    \
        } catch (...) {                                                                          \
            throw ::pnl::test::Failure{std::string("wrong exception type from ") + #expression + \
                                       " at " + __FILE__ + ":" + std::to_string(__LINE__)};      \
        }                                                                                        \
        if (!pnl_threw) {                                                                        \
            throw ::pnl::test::Failure{std::string("expected ") + #exception_type + " from " +   \
                                       #expression + " at " + __FILE__ + ":" +                   \
                                       std::to_string(__LINE__)};                                \
        }                                                                                        \
    } while (false)
