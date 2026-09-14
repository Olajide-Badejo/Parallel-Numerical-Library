# Contributing

## The one rule that is not negotiable

**No em dashes and no en dashes, in any file, ever.** That means U+2014 and
U+2013, and it also means the LaTeX ligatures `--` and `---` in prose, which
typeset as exactly those characters, and BibTeX page ranges, which are written
`19--26` by convention. Write "1 to 20" and "19 to 26".

`make check-style` enforces this over the whole tree including compiled PDFs. It
runs as a test, so continuous integration cannot pass while the rule is broken.

## Getting set up

```bash
make setup          # reports what is missing
make build          # configure and compile
make test           # every gate
```

The library is C++20 and needs nothing above it, so the floor is roughly GCC 11
or Clang 14, plus CMake 4.4 and Ninja, and OpenMP 4.5 for the OpenMP backend.
Release 1.1.0 is published from GCC 15.2.0, which is the compiler every measured
number in the report comes from; see decision 20 in `docs/DESIGN_DECISIONS.md`.
The tree is checked with three compilers before a change lands, each building
everything and running the whole suite: GCC 15.2.0, GCC 14.3.0 and Clang, which
is Clang 18 in CI and whatever the machine has locally. OpenMP, CUDA and MPI are
all optional and detected; a build without any of them configures cleanly and
skips the corresponding backends, which is what a Clang build with no OpenMP
runtime installed does.

Supported platforms are stated in `README.md` and the pinning half of them in
`docs/backends.md`. In short: Linux x86-64 is measured, Windows is supported
through WSL2, macOS builds without CUDA and without pinning, and MSVC
configures and is untested.

## Public API and compatibility

Release 1.0.0 installed nothing. There was no `install()`, no export set, no
package configuration file, no namespaced target and no version header, so
`find_package(pnl)` was impossible and there was nothing anyone outside this
repository could have depended on. Release 1.1.0 creates that API for the first
time, which is why it is a minor bump and not a major one: you cannot break what
was never exported.

**What is public.** Every header under `include/pnl/`, plus
`pnl/version.hpp`, which is generated into the build tree and installed beside
the rest. Anything under a `detail` namespace is internal and may change in any
release; there is no other private header today, and a header that becomes
internal will say so in its own file comment rather than only here.

**What a consumer may rely on**, from 1.1.0 onward:

- The target `pnl::core`. It spells the same in an installed tree, in a
  `FetchContent` build and after `add_subdirectory`, so a consumer who changes
  how they acquire the library changes nothing else.
- `pnl/version.hpp`: `PNL_VERSION_MAJOR`, `MINOR`, `PATCH`,
  `PNL_VERSION_STRING`, and `pnl::VERSION`, all generated from the one
  `project()` call.
- The flags `pnl_flags` carries, which reach the consumer through `pnl::core`:
  `-ffp-contract=off` and the C++20 standard, and nothing else. Those two are
  not decoration. About three quarters of this library is headers, so they are
  compiled in the consumer's own translation units, and a consumer who compiles
  them with contraction enabled gets an FMA in the SOR update and loses the bit
  identity guarantee with no diagnostic.
- `pnlConfigVersion.cmake` is `SameMajorVersion`, so `find_package(pnl 1.1)` is
  satisfied by any later 1.x and never by 2.x.

**What a consumer must not rely on.** `-O3`, `-march=native`, `-Wall -Wextra
-Wpedantic` and `-Werror` are in `pnl_dev_flags`, which is private, is applied
only when this project is the top level one, and is never installed. That is
deliberate: `-march=native` in an exported interface produces binaries on a
build farm that fault on the machine they are shipped to, and an exported
`-Werror` turns a newer compiler's new diagnostic into a build failure in code
the consumer did not write. `PNL_WERROR` therefore defaults to `OFF`; the
Makefile and CI turn it on for this repository's own builds.

**The ABI position.** There is none, and the reason is the same three quarters.
A library that is mostly headers is compiled by its consumer, not linked as a
settled binary layout, so an ABI promise across minor versions would be a claim
about a boundary that barely exists. **The source API follows semantic
versioning from 1.1.0**: within a major version, code that compiled against an
earlier minor version keeps compiling. Rebuild your own code against a new
version of this one rather than dropping a new `libpnl_core.a` beside objects
built against the old headers.

One consequence is worth stating plainly because it will otherwise read as a
contradiction. `PNL_REDUCTION_ACCUMULATORS`, when it arrives, changes numerical
output by a few units in the last place without changing a single signature.
That is a behavioural break, not an API break, so it does not force a major
bump, and for a library whose headline claim is bit identical reproducibility it
is the most significant thing a release can do. Changes of that kind lead the
changelog under their own heading and always name the setting that reproduces
the previous values.

**Adding to the public API.** A new public header goes under `include/pnl/` and
is installed by the existing `install(DIRECTORY include/ ...)` rule, so nothing
needs to be listed anywhere. A new library target that `pnl_core` links
`PUBLIC` must be added to the export set in the same commit, guarded on whether
it was built. CMake refuses to export a target whose link interface names a
static library outside the set, and the failure is a configure error on the
machines that have the optional half and silence everywhere else; `BUILD-03` and
`BUILD-04` in `docs/ENGINEERING_LOG.md` are both instances of it.
`make install-test` is the gate: it stages an install and builds `examples/`,
which declares `LANGUAGES CXX` and nothing else, against nothing but that
prefix.

## The invariant you must not break

Every shared memory backend, at every worker count, must produce **bit identical**
iterates under the deterministic reduction mode. This is not an aspiration, it is
what makes the whole comparison meaningful, and `test_equivalence` asserts it as
exact equality.

If you add a backend and it does not pass, it is not finished. If you change a
solver and equivalence fails, the solver has acquired a dependence on the
partition, which is a bug in the solver and not in the test.

Practical consequences:

- Reductions must go through `Backend::reduce`, so they use the fixed chunk grid.
  Never accumulate across chunks by hand.
- Never enable fast math or fused multiply add contraction. Both licence the
  compiler to reassociate, which is precisely what the invariant forbids.
- A sweep that is order dependent must either be order independent by
  construction (Jacobi, red black) or executed in strict order through
  `run_ordered`. There is no third option that preserves the invariant.

## Adding a solver

1. New header under `include/pnl/solvers/`, one solver per header.
2. Derive from `Solver`. If it is stationary, express it as a sweep and hand it
   to `detail::run_stationary`, which gives you the residual test, the history,
   the progress bar and the final gather, and guarantees you measure convergence
   the same way as everything else.
3. Document the splitting, the convergence order, and every exception it throws.
   That is a style rule, not a suggestion.
4. Register it in `registry.hpp`.
5. Add a convergence test that checks something against closed form theory. A
   test that only asserts convergence measures almost nothing: the test that
   caught the most interesting bug in this project asserted that an iteration
   count should *grow* with the grid.

## Adding a backend

1. Implement `Backend`. Only `name`, `worker_count`, `parallel_for`, `reduce`,
   `barrier` and `config` are required; the distributed hooks have defaults that
   are correct for shared memory.
2. Keep the deterministic reduction contract: partials in fixed slots, summed in
   index order.
3. Register it in `src/backend/factory.cpp` and `available_backends()`.
4. Add it to `thread_backends()` in the equivalence suite.

## Style

C++: `snake_case` for functions, files and namespaces; `PascalCase` for types;
`UPPER_SNAKE` for constants; trailing underscore on members. One solver per
header. Every public function documents its method, its convergence order and
the exceptions it throws. MPI calls wrapped in `MPI_CHECK`, CUDA in
`CUDA_CHECK`. Clean under `-Wall -Wextra -Wpedantic -Werror`.

Python: `ruff` clean, and runnable on Python 3.11, the `target-version` in
`ruff.toml`. ruff checks syntax against that version and not library
signatures, so a keyword or a function newer than 3.11 is on the author to
notice.

Comments explain why, not what. If a comment restates the code it is noise; if
the code cannot say it, the comment earns its place. The comments that matter
most here are the ones recording why an obvious alternative was rejected.

## Measurements

- No number without a run. A value that has not been measured reads `pending`.
- Every result row carries its provenance. Do not add a column the binary does
  not stamp itself.
- If you add a sweep block, give it a `why` field saying what question it
  answers. A block that cannot state its question does not earn its runtime.
- Check the arithmetic of a new block before running it. Two of the faults in the
  engineering log were misdeclared blocks that would have run for hours.

## Reporting a problem you fixed

Add an entry to `docs/ENGINEERING_LOG.md` with symptom, root cause, options
considered, the fix and why it beat the alternatives, and how you verified it.
The debug report is built from that log and is a deliverable of equal rank to the
main report. Entries describing faults that produced *plausible* results are the
most valuable ones; an obvious crash teaches nothing.

## Process

Contributions arrive as pull requests against `main`. The owner reviews and
merges. There is no second maintainer, so a pull request that sits is waiting on
one person rather than on a queue.

Nothing is merged with a red continuous integration run. A gate failing for a
reason unrelated to the change is a reason to repair the gate first, in its own
commit, not a reason to merge past it.

Two kinds of disagreement have a settled way out.

- **About a measurement.** It is settled by a run on the target machine with the
  spread recorded, not by an argument about what ought to be faster. Whoever
  makes the claim runs it, and the numbers go into `PROGRESS.md` beside the
  configuration that produced them.
- **About design.** It is settled by a new entry in `docs/DESIGN_DECISIONS.md`
  that states the option it rejects and why. A decision whose entry cannot name
  what it turned down has not been made yet.

## Authorship

This repository was built by its owner driving an AI coding agent from written
specifications. The specifications and the design decisions are the owner's, the
agent produced code and prose under them, and every number here and in the
reports comes from a run on the owner's machine. The `notes` field of
`CITATION.cff` says the same thing to anyone who cites the software.
