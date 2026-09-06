// SPDX-License-Identifier: MIT
/// \file write_golden.cpp
/// Regenerate the committed golden files.
///
/// Not a test, and it is built but never run by CTest. It exists so that the
/// procedure for regenerating a golden file is a program in the repository
/// rather than a paragraph somebody has to follow correctly, and so that the
/// files and the test that reads them are produced from one definition of the
/// configuration, `golden_config.hpp`.
///
/// Usage, from the repository root:
///
///     build/tests/pnl_write_golden tests/golden "$(git rev-parse --short=12 HEAD)"
///
/// The commit is passed in rather than read from a compile definition, because
/// a compile definition is fixed at configure time and would happily stamp a
/// stale hash onto a file generated after several more commits.
///
/// **Running this is a deliberate numerical change and is never routine.** A
/// golden file changes when the arithmetic changes, and a diff of one is the
/// evidence that it did. See the header comment of test_golden.cpp.

#include <pnl/core/error.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdio>
#include <exception>
#include <string>

#include "golden_config.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "usage: pnl_write_golden <output-directory> <commit>\n"
                     "  for example: pnl_write_golden tests/golden "
                     "\"$(git rev-parse --short=12 HEAD)\"\n");
        return 2;
    }
    const std::string directory = argv[1];
    const std::string commit = argv[2];

    try {
        for (const auto& name : pnl::solvers::all_solver_names()) {
            const std::string path = directory + "/" + name + ".hex";
            std::FILE* out = std::fopen(path.c_str(), "wb");
            if (out == nullptr) {
                std::fprintf(stderr, "pnl_write_golden: cannot write %s\n", path.c_str());
                return 1;
            }

            const pnl::Vector solution = pnl::golden::iterate(name);
            // "wb" and an explicit newline: this file is committed and the
            // repository is LF throughout, so the C library must not be given
            // the chance to translate anything.
            std::fprintf(out,
                         "# pnl golden iterate solver=%s %s values=%zu commit=%s\n",
                         name.c_str(),
                         pnl::golden::description().c_str(),
                         solution.size(),
                         commit.c_str());
            for (const pnl::Real value : solution) {
                // %a, so the file holds the exact double and a reviewer can see
                // which bit moved. %.17g round trips too, but it round trips
                // through a decimal conversion whose correctness is a property
                // of the C library rather than of the format.
                std::fprintf(out, "%a\n", value);
            }
            if (std::fclose(out) != 0) {
                std::fprintf(stderr, "pnl_write_golden: cannot close %s\n", path.c_str());
                return 1;
            }
            std::printf("wrote %s, %zu values\n", path.c_str(), solution.size());
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pnl_write_golden: %s\n", error.what());
        return 1;
    }
    return 0;
}
