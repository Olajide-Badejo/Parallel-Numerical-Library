# The backend layer

One interface, seven implementations, and a set of promises about what stays the
same across them.

## The interface

`include/pnl/backend/backend.hpp` declares everything a solver is allowed to
know about parallelism:

| Operation | Contract |
|---|---|
| `parallel_for(n, body)` | Apply `body` to a partition of `[0, n)` that covers it exactly once with contiguous chunks. Chunks may run concurrently. |
| `reduce(n, init, reducer)` | Sum `reducer` over a partition of `[0, n)`. Under the deterministic mode the partials are combined in a fixed order that depends on `n` alone. |
| `barrier()` | Synchronise all workers. |
| `local_rows(total)` | The rows this process owns. The whole range for every shared memory backend. |
| `exchange_halo(grid, stride, rows)` | Refresh neighbouring boundary rows. A no operation where neighbours are already readable. |
| `gather_rows(data, local)` | Make a replicated vector consistent when each rank updated only its own segment. |
| `run_ordered(work, forward, ...)` | Run `work` under global sequential ordering across ranks. |

That is the whole surface. It is the minimal common denominator of the models it
spans, following the structured parallel patterns argument in McCool, Robison and
Reinders, *Structured Parallel Programming*, Morgan Kaufmann 2012: a data
parallel map, a reduction, and a barrier. Anything richer would privilege one
model and stop the comparison being fair.

### Why callbacks are chunk level

`body` receives a `Range` and loops over it itself, rather than being called per
element. Two consequences, both deliberate:

- The `std::function` indirection is paid once per chunk, so it is O(workers) per
  sweep rather than O(unknowns). It does not contaminate the timings the study
  exists to produce.
- The innermost loop stays a plain loop over contiguous memory that the compiler
  can vectorise. Each backend therefore measures its own dispatch and
  synchronisation cost, not a penalty this abstraction imposed on all of them
  equally.

### What backends may not do

No backend leaks its model's types through the interface. There is no `MPI_Comm`,
no `omp_` type and no `cudaStream_t` in any signature. The CUDA path goes further
and crosses a C ABI boundary, because it has to: see ENV-01 in the engineering
log.

## The promise that makes the comparison mean something

**With the deterministic reduction mode, every shared memory backend at every
worker count produces bit identical iterates.** Not close. Identical.

This works because the reduction chunk grid is fixed by the problem size alone
(`DETERMINISTIC_CHUNKS = 512` in `chunking.hpp`), never by the worker count. Each
chunk's partial lands in its own slot and the slots are summed in index order
afterwards, so which worker computed which slot, and in what order they finished,
cannot affect the answer. Floating point addition is not associative; this design
removes the only thing that would have made the association vary.

The equivalence suite asserts exact equality rather than a tolerance, across
every applicable solver, both problem families, four backends and seven worker
counts.

Two documented exceptions, both stated rather than hidden:

- **Across MPI rank counts**, results agree to reduction tolerance, not bitwise.
  Rank boundaries are chosen for load balance and so do not align with the fixed
  chunk grid, which regroups the additions. Within one rank count the ordered
  allgather makes results reproducible and identical on every rank.
- **The GPU reductions** associate differently, because a tree inside a thread
  block is not an ordered sum and no compiler flag changes that. GPU *sweeps* are
  bit identical to the CPU; anything that passes through a reduction is compared
  to tolerance.

## What each backend does underneath

### serial

The correctness anchor. It still walks the same chunk grid as the parallel
backends rather than looping over `[0, n)` in one go, so "serial equals parallel,
bit for bit" is a statement about the parallel backends rather than an artefact
of the serial one taking a different path.

### openmp

The only backend that does not partition the range itself. It hands the chunk
grid to OpenMP's own worksharing construct, so what is measured is OpenMP's
scheduler and barrier, not a hand written partitioner wearing an OpenMP hat.
`schedule(static)` for uniform work; `schedule(dynamic, 1)` when asked, which is
how the dynamic scheduling cost is measured rather than assumed.

The build asserts `_OPENMP >= 201511` at compile time. GCC 16 reports 202111,
which is OpenMP 5.2. The specification cites OpenMP 6.0 as the reference
document, and the code is deliberately restricted to constructs GCC actually
implements; PROGRESS.md records that restriction.

### pthreads

Raw `pthread_create`, a mutex and condition variable for dispatch, a counter and
a second condition variable for completion, and `pthread_setaffinity_np` for
binding. Nothing from `<thread>` and nothing from OpenMP, so this backend
measures the classical POSIX primitives on identical work.

The pool is persistent. Creating threads per sweep would measure
`pthread_create`, which is a well known number and not the one this study is
about.

Workers wait on a **generation counter**, not a flag. A task published while a
worker was still finishing the previous one cannot then be missed, which is the
lost wakeup race a plain boolean would have.

### jthread

C++23 only: `std::jthread` workers, `std::stop_token` shutdown, and a pair of
`std::barrier` phases entered in strict alternation. One barrier for both release
and collect would let a fast worker race into the next task before a slow one had
left the previous.

**No work stealing, deliberately.** The loops here are uniform stencil sweeps
over contiguous memory, where a static partition is already balanced and a
stealing deque would add per task atomics that buy nothing. The dynamic schedule
measurement on the OpenMP backend is the evidence for that choice rather than the
assertion of it.

### mpi

Remainder aware block row decomposition via `block_partition`, so block sizes
differ by at most one and no row is dropped. Halo exchange is two `MPI_Sendrecv`
calls moving one row each. Every MPI call goes through `MPI_CHECK`, so a failure
is a diagnosed exception carrying its call site rather than a silently wrong
answer.

Each rank allocates the whole grid and computes only its own band. A production
code would allocate the local slab only, and this one would too if memory were
the constraint; at 4096 squared a vector is 134 MB, which fits. What matters for
the study is that the communication volume is identical either way, and the
choice keeps the solver code identical across every backend, which is the point
of the whole design.

The library advertises MPI 3.1 through `MPI_VERSION` while implementing many MPI
4.0 entry points. The build probes for symbols rather than testing the version
macro; see ENV-03.

### hybrid

Ranks with OpenMP threads inside them, which is how real machines are usually
run: one rank per memory domain, threads within it. It **inherits** from the MPI
backend rather than reimplementing, so the halo exchange, the ordered pass and
the deterministic reduction are literally the same code. Any difference the sweep
measures between `mpi` and `hybrid` is therefore the threading and nothing else.

### cuda

Not a `Backend` implementation, and that is the right call. The whole point of
running on a GPU is that state stays in device memory for the entire solve;
forcing it behind `parallel_for` would mean a host callback per chunk, which
would measure PCIe latency and nothing else. It is a separate device solve path
with the same numerics, verified against the serial backend.

It implements Jacobi, red black Gauss Seidel, red black SOR and conjugate
gradient on the 2D Poisson problem. **Natural ordering Gauss Seidel is absent on
purpose**: it is sequentially dependent and has no parallelism to offer a wide
device. That is a result, not a gap.

## The pinning story, and why it is weaker here than it looks

`Pinning` offers `none`, `compact`, `scatter`, `pcore` and `ecore`. The last two
depend on being able to tell a performance core from an efficiency core, and on
this machine, inside WSL2, **that cannot be done reliably**.

The host is an i7-14700K: 8 performance cores with two threads each, 12
efficiency cores. The guest reports 14 uniform cores of two threads, identical L2
on every processor, `cpu_capacity` 1024 everywhere, no `hybrid` flag in
`/proc/cpuinfo`, and no `cpufreq` directory at all. Worse,
`sched_setaffinity` binds a thread to a *guest virtual processor*, and the
hypervisor stays free to place that processor on any host core. A binding is a
hint, not a guarantee.

So the library does two things instead of pretending:

1. `probe_topology()` **measures**. It times an identical compute kernel on each
   logical processor and only reports a two group split when the throughputs
   separate by a margin that dominates the within group spread. Otherwise it says
   so, and the `pcore` and `ecore` policies decline to bind.
2. The knee that Objective 3 asks for is read from the **aggregate scaling
   curve**, which depends only on how many cores are engaged and not on which,
   and is therefore valid under virtualisation.

`scatter` and `compact` still bind, and are still swept, because they change
cache locality and migration behaviour even when the underlying placement is out
of the guest's hands. Measuring that they matter less here than they would on
bare metal is itself the finding.

## Adding a backend

1. Implement `Backend`. Only `name`, `worker_count`, `parallel_for`, `reduce`,
   `barrier` and `config` are required; the distributed hooks have sensible
   defaults for shared memory.
2. Keep the deterministic reduction contract: partials in fixed slots, summed in
   index order.
3. Register it in `src/backend/factory.cpp` and in `available_backends()`.
4. Add it to `thread_backends()` in the equivalence suite. If it does not pass
   bit identical equivalence against serial, it is not finished.
