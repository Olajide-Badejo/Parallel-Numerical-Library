// SPDX-License-Identifier: MIT
/// \file test_contract.cpp
/// The floating point contraction probe, compiled twice from this one file.
///
/// `test_contract` is built the way everything else in the suite is built, so
/// `pnl_flags` supplies `-ffp-contract=off`, and it asserts the probe stays
/// quiet. `test_contract_detects_fma` is built from this same source with
/// `-ffp-contract=fast -O2 -march=native` and links neither `pnl_flags` nor
/// `pnl_test_main`, because both carry the flag that turns contraction off, and
/// it asserts the probe throws. `PNL_CONTRACT_EXPECT_FUSED` is the only
/// difference between the two, and it selects which of the two expectations
/// applies.
///
/// The second executable is the one that matters. A probe that is never
/// observed to fire is a probe nobody knows is connected: it would pass just as
/// happily if `assert_no_contraction` had an empty body, or if the constants had
/// been chosen so that the fused and unfused results agree. Building the same
/// code the other way round is what turns "the flag is set" into "the check that
/// depends on the flag notices when it is not".
///
/// The last case is phase B4's addition and it tests the path rather than the
/// function. `make_backend` is inline, so the probe it runs is compiled into
/// whichever translation unit constructs the backend; this one is compiled with
/// contraction on, so `make_backend("serial", ...)` from here must throw. While
/// the probe was called from `factory.cpp` it could not: that file is compiled
/// with the library's flags, so it checked the library and never the caller,
/// which is the wrong half of a package that is three quarters headers.

#include <pnl/backend/backend.hpp>
#include <pnl/core/contract.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <memory>
#include <pnl_test.hpp>

PNL_TEST("contract/the probe constants are what the comment claims") {
    // Two step, with the intermediate value rounded to double before the add.
    // The volatile in the middle is what forces that rounding, so this holds in
    // both executables, including the one compiled with contraction on.
    volatile pnl::Real product = pnl::CONTRACTION_PROBE_A * pnl::CONTRACTION_PROBE_B;
    volatile pnl::Real two_step = product + pnl::CONTRACTION_PROBE_C;
    PNL_REQUIRE_EXACT(two_step, pnl::CONTRACTION_PROBE_EXPECTED);

    // The values themselves, so an edit that changes one of them and leaves the
    // comment standing fails here rather than silently weakening the probe.
    PNL_REQUIRE_EXACT(pnl::CONTRACTION_PROBE_A, 1.0 + 0x1p-27);
    PNL_REQUIRE_EXACT(pnl::CONTRACTION_PROBE_B, pnl::CONTRACTION_PROBE_A);
    PNL_REQUIRE_EXACT(pnl::CONTRACTION_PROBE_C, -1.0);
    PNL_REQUIRE_EXACT(pnl::CONTRACTION_PROBE_EXPECTED, 0x1p-26);

    // And the property the whole probe rests on: the fused answer is a different
    // double, not the same double reached by a different route, so exact
    // equality is a legitimate test rather than a tolerance in disguise.
    PNL_REQUIRE(pnl::CONTRACTION_PROBE_EXPECTED != 0x1p-26 + 0x1p-54);
}

#if defined(PNL_CONTRACT_EXPECT_FUSED)

PNL_TEST("contract/the probe throws when this translation unit contracts") {
    PNL_REQUIRE_THROWS(pnl::assert_no_contraction(), pnl::ConfigurationError);
}

PNL_TEST("contract/make_backend refuses a caller compiled with contraction") {
    // The proof that the probe moved into the caller's translation unit. This
    // file is compiled with -ffp-contract=fast and links the library, whose own
    // translation units are compiled with -ffp-contract=off, so a probe that
    // still ran inside factory.cpp would be quiet and this call would succeed.
    const pnl::backend::Config config;
    PNL_REQUIRE_THROWS(pnl::backend::make_backend("serial", config), pnl::ConfigurationError);
}

#else

PNL_TEST("contract/the probe is quiet under the project flags") {
    // No assertion beyond "this does not throw": the runner turns an escaping
    // exception into a failure with its message, which is the report wanted.
    pnl::assert_no_contraction();
}

PNL_TEST("contract/make_backend accepts a caller compiled without contraction") {
    // The other half of the pair. The same call from a translation unit that
    // was compiled correctly has to build a backend rather than refuse one,
    // otherwise the case above would be satisfied by a probe that always throws.
    const pnl::backend::Config config;
    const std::unique_ptr<pnl::backend::Backend> execution =
        pnl::backend::make_backend("serial", config);
    PNL_REQUIRE(execution->name() == "serial");
}

#endif
