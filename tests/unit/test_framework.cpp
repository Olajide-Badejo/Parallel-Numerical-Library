// SPDX-License-Identifier: MIT
/// \file test_framework.cpp
/// The test framework tested by the suite it runs.
///
/// This is registered first in tests/CMakeLists.txt on purpose. Every other
/// gate in this repository is a claim made through `pnl_test.hpp`, so a defect
/// here is not one failing test, it is every passing test meaning less than it
/// says. Three such defects are listed in Section 4.7 of the version 2
/// specification and each has a case below:
///
///   1. `run_all` caught `Failure` and `std::exception` and nothing else, so a
///      case that threw an `int` killed the process in `std::terminate`. CTest
///      reported a crash with no case name, and every case after it in that
///      binary never ran.
///   2. `first_difference` returned 0 on a length mismatch, which is the index
///      of the first element, and both callers then printed that element of a
///      vector that might be empty.
///   3. `worst_difference` compared only the common prefix, so a vector and a
///      truncation of itself were a perfect match and the distributed suite's
///      `difference == 0.0` assertions passed on one.
///
/// The deliberately failing cases run through `run_cases` over a list built
/// here rather than through `run_all` over the global registry, because a case
/// cannot add to the vector it is itself being iterated out of. The FAIL lines
/// they print are expected output of a passing test and the banner below says
/// so, since a reader who sees FAIL in a green run is entitled to an
/// explanation.

#include <pnl/core/error.hpp>

#include <cstdio>
#include <limits>
#include <pnl_test.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pnl;

namespace {

/// Run a list of cases with a banner, so the deliberate failures below read as
/// deliberate in the log.
[[nodiscard]] int run_expecting_failures(const std::vector<test::Case>& cases) {
    std::printf("        (the lines below are a deliberate failure, inside a passing case)\n");
    const int status = test::run_cases(cases);
    std::printf("        (end of the deliberate failure)\n");
    return status;
}

}  // namespace

PNL_TEST("framework/a case that throws a non std::exception fails rather than terminating") {
    const std::vector<test::Case> cases{
        test::Case{"deliberate: throws an int", [] { throw 42; }},
    };
    PNL_REQUIRE_MESSAGE(run_expecting_failures(cases) == 1,
                        "a case that threw an int did not come back as a failure; if this "
                        "process is still running, the catch all in run_cases is what let it");
}

PNL_TEST("framework/a case that throws a string literal fails rather than terminating") {
    // The other spelling people reach for, and it is not a std::exception
    // either. Kept separate from the int case so the failure names which one.
    const std::vector<test::Case> cases{
        test::Case{"deliberate: throws a string literal", [] { throw "not an exception"; }},
    };
    PNL_REQUIRE(run_expecting_failures(cases) == 1);
}

PNL_TEST("framework/the runner survives a throwing case and runs the ones after it") {
    // The consequence of the defect that mattered most: everything after the
    // throwing case in that binary was never attempted, so one bad case hid an
    // unknown number of others.
    int ran_after = 0;
    const std::vector<test::Case> cases{
        test::Case{"deliberate: throws an int", [] { throw 42; }},
        test::Case{"deliberate: runs after it", [&ran_after] { ++ran_after; }},
    };
    PNL_REQUIRE(run_expecting_failures(cases) == 1);
    PNL_REQUIRE_MESSAGE(ran_after == 1,
                        "the case after a throwing one did not run, so a single bad case still "
                        "hides every case that follows it");
}

PNL_TEST("framework/a passing list passes and a failing assertion is a failure") {
    // The other direction, so the three cases above are not passing because
    // run_cases returns 1 whatever happens.
    const std::vector<test::Case> passing{
        test::Case{"deliberate: passes", [] { PNL_REQUIRE(1 + 1 == 2); }},
    };
    PNL_REQUIRE(test::run_cases(passing) == 0);

    const std::vector<test::Case> failing{
        test::Case{"deliberate: fails an assertion", [] { PNL_REQUIRE(1 + 1 == 3); }},
    };
    PNL_REQUIRE(run_expecting_failures(failing) == 1);

    const std::vector<test::Case> throwing{
        test::Case{"deliberate: throws a pnl error", [] { throw InvalidArgument("deliberate"); }},
    };
    PNL_REQUIRE(run_expecting_failures(throwing) == 1);
}

PNL_TEST("framework/the filter selects by substring and reports what it skipped") {
    int ran = 0;
    const std::vector<test::Case> cases{
        test::Case{"alpha/one", [&ran] { ++ran; }},
        test::Case{"beta/two", [&ran] { ++ran; }},
    };
    PNL_REQUIRE(test::run_cases(cases, "alpha") == 0);
    PNL_REQUIRE_MESSAGE(ran == 1, "the filter ran the wrong number of cases");
}

PNL_TEST("framework/first_difference reports a length mismatch instead of index zero") {
    const Vector three{1.0, 2.0, 3.0};
    const Vector two{1.0, 2.0};
    const Vector empty;

    // Identical is minus one, which is the only value that means "no
    // difference" and is what every caller tests for.
    PNL_REQUIRE(test::first_difference(three, three) == -1);

    // A prefix that agrees: the difference is at the first index the shorter
    // vector does not have, never at index 0.
    PNL_REQUIRE_MESSAGE(test::first_difference(three, two) == 2,
                        "a vector and a prefix of it were reported as differing at " +
                            std::to_string(test::first_difference(three, two)));
    PNL_REQUIRE(test::first_difference(two, three) == 2);

    // The case that used to read past the end. An empty vector against a non
    // empty one differs at index 0, which is an index neither caller may use,
    // so describe_difference is what they print.
    PNL_REQUIRE(test::first_difference(empty, three) == 0);
    PNL_REQUIRE(test::first_difference(empty, empty) == -1);

    // And a genuine element difference is still found where it is.
    const Vector altered{1.0, 2.5, 3.0};
    PNL_REQUIRE(test::first_difference(three, altered) == 1);
}

PNL_TEST("framework/describe_difference never indexes a vector that is too short") {
    const Vector three{1.0, 2.0, 3.0};
    const Vector empty;

    // Under the address sanitizer this case is the assertion: the old callers
    // formatted candidate[first_difference(...)] and first_difference answered
    // 0 for an empty vector, which is a read of element zero of nothing.
    const std::string mismatch = test::describe_difference(empty, three);
    PNL_REQUIRE_MESSAGE(mismatch.find("not the same length") != std::string::npos,
                        "a length mismatch was described as '" + mismatch + "'");

    PNL_REQUIRE(test::describe_difference(three, three) == "no difference");

    const Vector altered{1.0, 2.5, 3.0};
    const std::string differs = test::describe_difference(three, altered);
    PNL_REQUIRE_MESSAGE(differs.find("index 1") != std::string::npos,
                        "an element difference was described as '" + differs + "'");
}

PNL_TEST("framework/worst_difference fails a length mismatch instead of passing it") {
    const Vector three{1.0, 2.0, 3.0};
    const Vector two{1.0, 2.0};
    const Vector empty;

    PNL_REQUIRE(test::worst_difference(three, three) == 0.0);
    PNL_REQUIRE(test::worst_difference(empty, empty) == 0.0);

    // The two spellings the distributed suite uses, both of which a finite
    // answer would have passed.
    const Real mismatch = test::worst_difference(three, two);
    PNL_REQUIRE_MESSAGE(!(mismatch <= 1.0e-12),
                        "a vector and a prefix of it came back within tolerance of each other");
    PNL_REQUIRE_MESSAGE(!(mismatch == 0.0), "a vector and a prefix of it came back bit identical");
    PNL_REQUIRE(!(test::worst_difference(empty, three) <= 1.0e-12));

    const Vector altered{1.0, 2.5, 3.0};
    PNL_REQUIRE(test::worst_difference(three, altered) == 0.5);
}

PNL_TEST("framework/the comparison helpers answer false for a NaN rather than throwing") {
    // Every close_ helper is asked about a residual that may be NaN, and a NaN
    // compares false against everything, so the helpers have to decide the case
    // themselves rather than let the comparison decide it for them.
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    PNL_REQUIRE(!test::close_absolute(nan, nan, 1.0));
    PNL_REQUIRE(!test::close_relative(nan, 1.0, 1.0));
    PNL_REQUIRE(!test::close_relative(1.0, nan, 1.0));
    PNL_REQUIRE(test::close_relative(1.0, 1.0, 0.0));
}

PNL_TEST("framework/PNL_REQUIRE_THROWS refuses the wrong exception type") {
    // The macro's own contract: the right type passes, no throw fails, and the
    // wrong type fails rather than being counted as a throw.
    PNL_REQUIRE_THROWS(throw InvalidArgument("expected"), InvalidArgument);

    const std::vector<test::Case> wrong_type{
        test::Case{"deliberate: throws the wrong type",
                   [] { PNL_REQUIRE_THROWS(throw std::runtime_error("wrong"), InvalidArgument); }},
    };
    PNL_REQUIRE(run_expecting_failures(wrong_type) == 1);

    const std::vector<test::Case> no_throw{
        test::Case{"deliberate: throws nothing",
                   [] { PNL_REQUIRE_THROWS((void)0, InvalidArgument); }},
    };
    PNL_REQUIRE(run_expecting_failures(no_throw) == 1);
}
