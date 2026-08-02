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

The build needs GCC 16 for `<mdspan>` and OpenMP 5.2, CMake 4.4, and Ninja. CUDA
and MPI are optional and detected; a build without either configures cleanly and
skips the corresponding backends.

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

Python: `ruff` clean.

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
