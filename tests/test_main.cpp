/// \file test_main.cpp
/// Entry point shared by every test binary.
///
/// An optional first argument filters cases by substring, which is what makes
/// a failing case quick to iterate on without rebuilding.

#include <pnl_test.hpp>

#include <cstdio>
#include <string_view>

int main(int argc, char** argv) {
    const std::string_view filter = argc > 1 ? argv[1] : std::string_view{};
    if (!filter.empty()) std::printf("filter: %.*s\n", static_cast<int>(filter.size()),
                                     filter.data());
    return pnl::test::run_all(filter);
}
