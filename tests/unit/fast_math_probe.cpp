// SPDX-License-Identifier: MIT
/// \file fast_math_probe.cpp
/// A translation unit that exists in order to be rejected.
///
/// No target builds this. `test_fast_math_rejected` compiles it twice with
/// `-fsyntax-only`: once with `-ffast-math`, where the guard at the top of
/// pnl/core/types.hpp must stop the compile and name the reason, and once
/// without, where it must compile cleanly. The second half is the control. A
/// test that only checks the failing compile passes just as well when the file
/// has a typo in it, and would then be asserting nothing about the guard.

#include <pnl/core/types.hpp>

int main() {
    return pnl::EPSILON > 0.0 ? 0 : 1;
}
