// SPDX-License-Identifier: MIT
/// \file test_preconditions.cpp
/// Every public Problem entry point checks the length of the views it is given,
/// and jacobi_sweep checks that its input and its output are separate buffers.
///
/// Both halves matter for the same reason. These methods index the caller's
/// views by a stride the problem chose and write through raw pointers into them,
/// so a view of the wrong length is a heap write past the end and one buffer
/// passed twice is a different algorithm, silently. Section 9.8 assertion 7 asks
/// for the aliasing guard before the Fortran kernels of release 1.2.0 exist,
/// because a Fortran dummy argument may not alias another that is defined and
/// the failure would then be undefined behaviour that appears only at -O3 and
/// only sometimes.
///
/// The messages are checked as well as the throws. A precondition that fires
/// without naming the method and the two sizes sends the reader to a debugger,
/// which is most of the value of having one.

#include <pnl/backend/backend.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>

#include <cstddef>
#include <memory>
#include <pnl_test.hpp>
#include <string>
#include <vector>

namespace {

using namespace pnl;

constexpr Index SIDE = 15;
constexpr Index DENSE_ORDER = 16;

[[nodiscard]] std::unique_ptr<backend::Backend> serial() {
    backend::Config config;
    config.workers = 1;
    return backend::make_backend("serial", config);
}

/// Run \p call, require that it throws InvalidArgument, and require that the
/// message names \p method and both sizes.
template<typename Call>
void expect_size_failure(Call&& call,
                         const std::string& method,
                         const std::string& expected,
                         const std::string& given) {
    bool threw = false;
    try {
        call();
    } catch (const InvalidArgument& error) {
        threw = true;
        const std::string what = error.what();
        PNL_REQUIRE_MESSAGE(what.find(method) != std::string::npos,
                            "the message does not name the method: " + what);
        PNL_REQUIRE_MESSAGE(what.find(expected) != std::string::npos,
                            "the message does not name the size it needed: " + what);
        PNL_REQUIRE_MESSAGE(what.find(given) != std::string::npos,
                            "the message does not name the size it was given: " + what);
    }
    PNL_REQUIRE_MESSAGE(threw, method + " accepted a wrongly sized view");
}

}  // namespace

PNL_TEST("preconditions/a short view is refused by every Poisson entry point") {
    problems::Poisson2D problem(SIDE, problems::PoissonRhs::SpectrallyRich);
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector good = problem.make_state();
    Vector other = problem.make_state();
    Vector short_view(static_cast<std::size_t>(problem.state_size() - 1), 0.0);

    const std::string expected = std::to_string(problem.state_size());
    const std::string given = std::to_string(problem.state_size() - 1);

    expect_size_failure(
        [&] { problem.initial_state(short_view); }, "Poisson2D::initial_state", expected, given);
    expect_size_failure(
        [&] { problem.apply(*execution, short_view, other); }, "Poisson2D::apply", expected, given);
    expect_size_failure(
        [&] { problem.apply(*execution, good, short_view); }, "Poisson2D::apply", expected, given);
    expect_size_failure([&] { problem.jacobi_sweep(*execution, short_view, other); },
                        "Poisson2D::jacobi_sweep",
                        expected,
                        given);
    expect_size_failure([&] { problem.jacobi_sweep(*execution, good, short_view); },
                        "Poisson2D::jacobi_sweep",
                        expected,
                        given);
    expect_size_failure(
        [&] { problem.relaxation_sweep(*execution, short_view, 1.0, problems::Sweep::Forward); },
        "Poisson2D::relaxation_sweep",
        expected,
        given);
    expect_size_failure(
        [&] { problem.coloured_sweep(*execution, short_view, 1.0, problems::Colour::Red); },
        "Poisson2D::coloured_sweep",
        expected,
        given);
    expect_size_failure(
        [&] {
            problem.block_sweep(*execution, short_view, problem.natural_block_count(), true, other);
        },
        "Poisson2D::block_sweep",
        expected,
        given);
    expect_size_failure([&] { problem.residual(*execution, short_view, other); },
                        "Poisson2D::residual",
                        expected,
                        given);
    expect_size_failure([&] { problem.residual(*execution, good, short_view); },
                        "Poisson2D::residual",
                        expected,
                        given);
    expect_size_failure([&] { (void)problem.dot(*execution, short_view, good); },
                        "Poisson2D::dot",
                        expected,
                        given);
    expect_size_failure([&] { (void)problem.dot(*execution, good, short_view); },
                        "Poisson2D::dot",
                        expected,
                        given);
    expect_size_failure([&] { problem.axpy(*execution, 1.0, short_view, other); },
                        "Poisson2D::axpy",
                        expected,
                        given);
    expect_size_failure([&] { problem.axpy(*execution, 1.0, good, short_view); },
                        "Poisson2D::axpy",
                        expected,
                        given);
    expect_size_failure([&] { problem.xpby(*execution, short_view, 1.0, other); },
                        "Poisson2D::xpby",
                        expected,
                        given);
    expect_size_failure([&] { problem.xpby(*execution, good, 1.0, short_view); },
                        "Poisson2D::xpby",
                        expected,
                        given);
    expect_size_failure([&] { problem.synchronise(*execution, short_view); },
                        "Poisson2D::synchronise",
                        expected,
                        given);
}

PNL_TEST("preconditions/a short view is refused by every dense entry point") {
    problems::DenseProblem problem(
        DENSE_ORDER, 20260906, problems::DenseKind::SymmetricPositiveDefinite, 4);
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector good = problem.make_state();
    Vector other = problem.make_state();
    Vector short_view(static_cast<std::size_t>(problem.state_size() - 1), 0.0);

    const std::string expected = std::to_string(problem.state_size());
    const std::string given = std::to_string(problem.state_size() - 1);

    expect_size_failure([&] { problem.apply(*execution, short_view, other); },
                        "DenseProblem::apply",
                        expected,
                        given);
    expect_size_failure([&] { problem.jacobi_sweep(*execution, short_view, other); },
                        "DenseProblem::jacobi_sweep",
                        expected,
                        given);
    expect_size_failure(
        [&] { problem.relaxation_sweep(*execution, short_view, 1.0, problems::Sweep::Forward); },
        "DenseProblem::relaxation_sweep",
        expected,
        given);
    expect_size_failure(
        [&] {
            problem.block_sweep(*execution, short_view, problem.natural_block_count(), true, other);
        },
        "DenseProblem::block_sweep",
        expected,
        given);
    expect_size_failure([&] { problem.residual(*execution, short_view, other); },
                        "DenseProblem::residual",
                        expected,
                        given);
    expect_size_failure([&] { (void)problem.dot(*execution, short_view, good); },
                        "DenseProblem::dot",
                        expected,
                        given);
    expect_size_failure([&] { problem.axpy(*execution, 1.0, short_view, other); },
                        "DenseProblem::axpy",
                        expected,
                        given);
    expect_size_failure([&] { problem.xpby(*execution, short_view, 1.0, other); },
                        "DenseProblem::xpby",
                        expected,
                        given);
    expect_size_failure([&] { problem.synchronise(*execution, short_view); },
                        "DenseProblem::synchronise",
                        expected,
                        given);
}

PNL_TEST("preconditions/a longer view is refused too, not only a shorter one") {
    // The check is equality, not a minimum, because a problem indexes its own
    // stride into the view and a buffer that is too long is as likely to be the
    // wrong problem's as one that is too short.
    problems::Poisson2D problem(SIDE, problems::PoissonRhs::SpectrallyRich);
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector other = problem.make_state();
    Vector long_view(static_cast<std::size_t>(problem.state_size() + 1), 0.0);
    expect_size_failure([&] { problem.jacobi_sweep(*execution, long_view, other); },
                        "Poisson2D::jacobi_sweep",
                        std::to_string(problem.state_size()),
                        std::to_string(problem.state_size() + 1));
}

PNL_TEST("preconditions/jacobi_sweep refuses one buffer passed as both arguments") {
    problems::Poisson2D problem(SIDE, problems::PoissonRhs::SpectrallyRich);
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector x = problem.make_state();

    bool threw = false;
    try {
        problem.jacobi_sweep(*execution, x, x);
    } catch (const InvalidArgument& error) {
        threw = true;
        const std::string what = error.what();
        PNL_REQUIRE_MESSAGE(what.find("Poisson2D::jacobi_sweep") != std::string::npos,
                            "the message does not name the method: " + what);
        PNL_REQUIRE_MESSAGE(what.find("share storage") != std::string::npos,
                            "the message does not say what is wrong: " + what);
    }
    PNL_REQUIRE_MESSAGE(threw, "jacobi_sweep accepted one buffer as both input and output");

    // A partial overlap is the same fault and the same check catches it. Two
    // correctly sized windows onto one buffer, sharing a single element.
    Vector wide(static_cast<std::size_t>(2 * problem.state_size() - 1), 0.0);
    const VectorView all(wide);
    const auto width = static_cast<std::size_t>(problem.state_size());
    const VectorView front = all.subspan(0, width);
    const VectorView back = all.subspan(width - 1, width);
    PNL_REQUIRE_THROWS(problem.jacobi_sweep(*execution, front, back), InvalidArgument);

    problems::DenseProblem dense(
        DENSE_ORDER, 20260906, problems::DenseKind::SymmetricPositiveDefinite, 4);
    Vector d = dense.make_state();
    PNL_REQUIRE_THROWS(dense.jacobi_sweep(*execution, d, d), InvalidArgument);
}

PNL_TEST("preconditions/dot with one buffer passed twice stays legal") {
    // The other half of the aliasing rule, and the reason it is a test rather
    // than a comment: neither argument of dot is written, so this is an ordinary
    // inner product of a vector with itself, and Problem::norm does it on every
    // residual evaluation of every solver. Anyone who "fixes" jacobi_sweep's
    // guard by applying it here breaks the whole zoo, and this fails first.
    problems::Poisson2D problem(SIDE, problems::PoissonRhs::SpectrallyRich);
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector x = problem.make_state();
    for (Index i = 1; i <= SIDE; ++i) {
        for (Index j = 1; j <= SIDE; ++j) {
            x[static_cast<std::size_t>(problem.at(i, j))] = static_cast<Real>(i + j);
        }
    }
    const Real self = problem.dot(*execution, x, x);
    PNL_REQUIRE(self > 0.0);
    // Not exact equality: norm is the square root of this, and squaring a
    // rounded square root does not return the value it came from.
    const Real norm = problem.norm(*execution, x);
    PNL_REQUIRE_CLOSE(norm * norm, self, 1.0e-14);
}

PNL_TEST("preconditions/exchange_halo and gather_rows check their arguments") {
    // The shared memory default does nothing with these numbers, and checks them
    // anyway: the same arguments under MPI are an offset and a count used to
    // write through a raw pointer, so a size that is wrong here is a buffer
    // overrun there.
    const std::unique_ptr<backend::Backend> execution = serial();
    Vector grid(100, 0.0);

    // A 10 by 10 padded grid needs (8 + 2) * 10 = 100 values, which fits.
    execution->exchange_halo(grid, 10, 8);
    // Nine interior rows would need 110.
    PNL_REQUIRE_THROWS(execution->exchange_halo(grid, 10, 9), InvalidArgument);
    // A flat vector, where the row count alone is the requirement.
    execution->exchange_halo(grid, 0, 100);
    PNL_REQUIRE_THROWS(execution->exchange_halo(grid, 0, 101), InvalidArgument);

    execution->gather_rows(grid, Range{0, 100});
    PNL_REQUIRE_THROWS(execution->gather_rows(grid, Range{0, 101}), InvalidArgument);
    PNL_REQUIRE_THROWS(execution->gather_rows(grid, Range{-1, 10}), InvalidArgument);
}
