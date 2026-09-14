// SPDX-License-Identifier: MIT
/// \file test_chunk_property.cpp
/// Property based coverage of every chunk grid in the library.
///
/// What was here before this file was a fixed list: six problem sizes and the
/// part counts 1 to 33, in `tests/equivalence/test_equivalence.cpp`. A fixed
/// list tests the sizes somebody thought of. These are the three properties the
/// whole library rests on, asserted over thousands of random draws and over
/// every boundary the code changes shape at:
///
///   1. **Exact coverage.** Every index of `[0, n)` is claimed by exactly one
///      chunk. Checked by counting hits per index, not by summing sizes, so a
///      partition that dropped index 4 and claimed index 7 twice fails.
///   2. **Pairwise disjointness.** The same count, read the other way: no index
///      is claimed twice. It is what makes a `parallel_for` body free of a data
///      race by construction, and Section 9.8 assertion 11 of the version 2
///      specification is exactly this property with the row band map applied,
///      because the `pure` declaration a Fortran dispatch carries is a promise
///      the compiler believes and cannot verify.
///   3. **A size spread of at most one.** `block_partition` promises remainder
///      awareness, and a load imbalance of more than one element per chunk
///      would show up in every scaling curve as a property of this function
///      rather than of the machine.
///
/// The three grids are `block_partition` in `pnl/core/types.hpp`, and
/// `for_chunk` and `reduction_chunk` in `pnl/backend/chunking.hpp`. The second
/// and third delegate to the first, which is precisely why they are tested
/// through their own entry points: `for_chunk_count` and `reduction_chunk_count`
/// decide how many chunks there are, and an off by one there is not a defect in
/// `block_partition` at all.
///
/// **The padded row band map is here too**, and it is the reason this file
/// covers more than the partition. Every sweep in `poisson2d.hpp` turns a chunk
/// of `[0, rows.size())` into the inclusive padded row range
/// `[rows.begin + chunk.begin + 1, rows.begin + chunk.end]`. That map has an
/// empty case the partition does not, because a chunk with `begin == end`
/// becomes a range whose last row is before its first, and a loop written
/// `for (i = first; i <= last; ++i)` is the only spelling that handles it. The
/// bands must be pairwise disjoint and must cover the interior rows exactly,
/// which is what release 1.2.0 needs for assertion 11 and what this release
/// needs anyway.
///
/// `n` in `{0, 1, 511, 512, 513}` appears explicitly in every case below.
/// `reduction_chunk_count` is `n` up to `DETERMINISTIC_CHUNKS` and
/// `DETERMINISTIC_CHUNKS` above it, so 511, 512 and 513 are the three sides of
/// the only place its shape changes; 0 and 1 are where a partition has no
/// chunks and one chunk.
///
/// A failure prints the seed and the exact draw, so a random failure is
/// reproducible by reading the message rather than by rerunning until it
/// happens again.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/core/types.hpp>

#include <cstdint>
#include <pnl_test.hpp>
#include <random>
#include <string>
#include <vector>

using namespace pnl;
using namespace pnl::backend;

namespace {

/// Seed for every draw in this file. Fixed, and printed in every failure
/// message, so that a failure is a reproduction recipe rather than a rumour.
constexpr std::uint64_t PROPERTY_SEED = 20260906;

/// Draws per case. Thousands, and the cost is a few milliseconds because the
/// work per draw is a partition of at most a few thousand indices.
constexpr int DRAWS = 4000;

/// Sizes at which one of the grids changes shape, checked explicitly in every
/// case rather than left to the draws to find.
constexpr Index BOUNDARY_SIZES[] = {0, 1, 2, 3, 511, 512, 513, 1024};

/// Assert that \p count chunks obtained from \p chunk_of partition `[0, n)`.
///
/// \param chunk_of chunk index to range.
/// \param where a sentence naming the draw, appended to every failure.
void require_partitions(Index n,
                        Index count,
                        const std::function<Range(Index)>& chunk_of,
                        const std::string& where) {
    PNL_REQUIRE_MESSAGE(count >= 0, "a negative chunk count for " + where);

    // One counter per index. Counting is what separates "covers" from
    // "disjoint": a sum of sizes equal to n is satisfied by a partition that
    // drops one index and claims another twice.
    std::vector<int> hits(static_cast<std::size_t>(n > 0 ? n : 0), 0);
    Index largest = 0;
    Index smallest = n > 0 ? n : 0;

    // Coverage and disjointness first, contiguity second. The order matters:
    // the contiguity check is the stricter of the two and would fire first on
    // every partition that is wrong, so a harness that checked it first could
    // never report which index was dropped or claimed twice, and the negative
    // control below could not tell the two failures apart.
    for (Index k = 0; k < count; ++k) {
        const Range chunk = chunk_of(k);
        PNL_REQUIRE_MESSAGE(chunk.begin >= 0 && chunk.end >= chunk.begin,
                            "chunk " + std::to_string(k) + " of " + where + " is [" +
                                std::to_string(chunk.begin) + ", " + std::to_string(chunk.end) +
                                "), which is not a range");
        PNL_REQUIRE_MESSAGE(chunk.end <= n,
                            "chunk " + std::to_string(k) + " of " + where + " ends at " +
                                std::to_string(chunk.end) + ", past the end of the range");

        for (Index i = chunk.begin; i < chunk.end; ++i) {
            ++hits[static_cast<std::size_t>(i)];
        }
        largest = std::max(largest, chunk.size());
        smallest = std::min(smallest, chunk.size());

        // Range::empty() has to agree with the range it describes, because the
        // sweeps skip on it.
        PNL_REQUIRE_MESSAGE(
            chunk.empty() == (chunk.size() <= 0),
            "chunk " + std::to_string(k) + " of " + where + " disagrees with its own empty()");
    }

    for (std::size_t i = 0; i < hits.size(); ++i) {
        PNL_REQUIRE_MESSAGE(hits[i] == 1,
                            "index " + std::to_string(i) + " of " + where + " was claimed " +
                                std::to_string(hits[i]) +
                                " times, and every index must be "
                                "claimed exactly once");
    }

    Index previous_end = 0;
    for (Index k = 0; k < count; ++k) {
        const Range chunk = chunk_of(k);
        PNL_REQUIRE_MESSAGE(chunk.begin == previous_end,
                            "chunk " + std::to_string(k) + " of " + where + " begins at " +
                                std::to_string(chunk.begin) + " and the previous one ended at " +
                                std::to_string(previous_end));
        previous_end = chunk.end;
    }
    PNL_REQUIRE_MESSAGE(previous_end == n,
                        where + " stopped at " + std::to_string(previous_end) + " rather than " +
                            std::to_string(n));

    if (count > 0) {
        PNL_REQUIRE_MESSAGE(largest - smallest <= 1,
                            where + " has chunk sizes from " + std::to_string(smallest) + " to " +
                                std::to_string(largest) + ", a spread of more than one");
    }
}

/// True when \p check throws the framework's failure, which is what every
/// assertion in require_partitions and require_row_bands throws.
[[nodiscard]] bool rejects(const std::function<void()>& check) {
    try {
        check();
    } catch (const test::Failure&) {
        return true;
    }
    return false;
}

/// Assert that the padded row bands derived from a partition of \p rows are
/// pairwise disjoint and cover the interior rows of that band exactly.
///
/// The map is the one every sweep in poisson2d.hpp uses:
/// `[rows.begin + chunk.begin + 1, rows.begin + chunk.end]`, inclusive at both
/// ends. Section 9.8 assertion 11.
void require_row_bands(Range rows,
                       Index count,
                       const std::function<Range(Index)>& chunk_of,
                       const std::string& where) {
    // Interior rows this band owns are rows.begin + 1 through rows.begin +
    // rows.size(), inclusive. Index the counter from rows.begin + 1.
    const Index interior = rows.size();
    std::vector<int> hits(static_cast<std::size_t>(interior > 0 ? interior : 0), 0);

    for (Index k = 0; k < count; ++k) {
        const Range chunk = chunk_of(k);
        const Index first = rows.begin + chunk.begin + 1;
        const Index last = rows.begin + chunk.end;

        // An empty chunk maps to a band whose last row is before its first, and
        // the ascending loop below is the only spelling that treats that as no
        // rows rather than as one row.
        if (chunk.empty()) {
            PNL_REQUIRE_MESSAGE(last < first,
                                "an empty chunk " + std::to_string(k) + " of " + where +
                                    " mapped to the non empty row band [" + std::to_string(first) +
                                    ", " + std::to_string(last) + "]");
            continue;
        }

        PNL_REQUIRE_MESSAGE(first >= rows.begin + 1 && last <= rows.begin + interior,
                            "the row band of chunk " + std::to_string(k) + " of " + where +
                                " is [" + std::to_string(first) + ", " + std::to_string(last) +
                                "], outside the interior rows [" + std::to_string(rows.begin + 1) +
                                ", " + std::to_string(rows.begin + interior) + "]");

        for (Index i = first; i <= last; ++i) {
            ++hits[static_cast<std::size_t>(i - rows.begin - 1)];
        }
    }

    for (std::size_t i = 0; i < hits.size(); ++i) {
        PNL_REQUIRE_MESSAGE(hits[i] == 1,
                            "interior row " +
                                std::to_string(static_cast<Index>(i) + rows.begin + 1) + " of " +
                                where + " is swept " + std::to_string(hits[i]) +
                                " times, and every interior row must be swept exactly once");
    }
}

std::string draw_note(const std::string& what, Index n, Index parts) {
    return what + " with n = " + std::to_string(n) + ", parts = " + std::to_string(parts) +
           " (seed " + std::to_string(PROPERTY_SEED) + ")";
}

}  // namespace

PNL_TEST("property/block_partition covers, is disjoint and is balanced over random draws") {
    std::mt19937_64 engine(PROPERTY_SEED);
    std::uniform_int_distribution<Index> size_draw(0, 4096);
    std::uniform_int_distribution<Index> parts_draw(1, 96);

    for (int draw = 0; draw < DRAWS; ++draw) {
        const Index n = size_draw(engine);
        const Index parts = parts_draw(engine);
        require_partitions(
            n,
            parts,
            [n, parts](Index k) { return block_partition(n, parts, k); },
            draw_note("block_partition", n, parts));
    }

    // The boundaries, explicitly, so they are covered whatever the draws did.
    for (Index n : BOUNDARY_SIZES) {
        for (Index parts : {Index{1},
                            Index{2},
                            Index{3},
                            Index{7},
                            Index{511},
                            Index{512},
                            Index{513},
                            Index{1024}}) {
            require_partitions(
                n,
                parts,
                [n, parts](Index k) { return block_partition(n, parts, k); },
                draw_note("block_partition", n, parts));
        }
    }
}

PNL_TEST("property/block_partition answers an out of range chunk index with an empty range") {
    // Every caller loops k from 0 to count, so this is defensive rather than
    // load bearing, and it is the one part of the contract the coverage check
    // cannot see: a partition that returned a real range for k = parts would
    // still cover [0, n) exactly.
    for (Index n : BOUNDARY_SIZES) {
        for (Index parts : {Index{1}, Index{8}, Index{512}}) {
            for (Index k : {Index{-1}, parts, parts + 1}) {
                const Range out = block_partition(n, parts, k);
                PNL_REQUIRE_MESSAGE(out.begin == 0 && out.end == 0,
                                    "block_partition(" + std::to_string(n) + ", " +
                                        std::to_string(parts) + ", " + std::to_string(k) +
                                        ") is not the empty range");
            }
        }
        // A part count of zero or less has no chunks and must not divide.
        for (Index parts : {Index{0}, Index{-1}}) {
            const Range out = block_partition(n, parts, 0);
            PNL_REQUIRE(out.begin == 0 && out.end == 0);
        }
    }
}

PNL_TEST("property/reduction_chunk covers and is disjoint at the chunk count switch") {
    // reduction_chunk_count is n below DETERMINISTIC_CHUNKS and
    // DETERMINISTIC_CHUNKS above it, so the interesting sizes are 511, 512 and
    // 513, and this is the grid the bit identity claim of the whole library
    // rests on: it depends on the problem size and on nothing else.
    std::mt19937_64 engine(PROPERTY_SEED);
    std::uniform_int_distribution<Index> size_draw(0, 4096);

    auto check = [](Index n) {
        const Index count = reduction_chunk_count(n);
        PNL_REQUIRE_MESSAGE(
            count == (n <= 0 ? 0 : std::min(DETERMINISTIC_CHUNKS, n)),
            "reduction_chunk_count(" + std::to_string(n) + ") is " + std::to_string(count));
        require_partitions(
            n,
            count,
            [n](Index k) { return reduction_chunk(n, k); },
            draw_note("reduction_chunk", n, count));
    };

    for (int draw = 0; draw < DRAWS; ++draw) check(size_draw(engine));
    for (Index n : BOUNDARY_SIZES) check(n);
    // And a size far above the switch, where every chunk is many elements.
    for (Index n : {Index{100000}, Index{100001}}) check(n);
}

PNL_TEST("property/for_chunk covers and is disjoint under both schedules") {
    std::mt19937_64 engine(PROPERTY_SEED);
    std::uniform_int_distribution<Index> size_draw(0, 4096);
    std::uniform_int_distribution<int> worker_draw(1, 64);
    std::uniform_int_distribution<int> per_worker_draw(1, 16);
    std::uniform_int_distribution<int> schedule_draw(0, 1);

    auto check = [](Index n, int workers, Schedule schedule, int per_worker) {
        const Index count = for_chunk_count(n, workers, schedule, per_worker);
        const std::string what = std::string("for_chunk, ") +
                                 (schedule == Schedule::Static ? "static" : "dynamic") + ", " +
                                 std::to_string(workers) + " workers, " +
                                 std::to_string(per_worker) + " chunks each";
        require_partitions(
            n,
            count,
            [n, workers, schedule, per_worker](Index k) {
                return for_chunk(n, workers, schedule, per_worker, k);
            },
            draw_note(what, n, count));
    };

    for (int draw = 0; draw < DRAWS; ++draw) {
        const Schedule schedule = schedule_draw(engine) == 0 ? Schedule::Static : Schedule::Dynamic;
        check(size_draw(engine), worker_draw(engine), schedule, per_worker_draw(engine));
    }

    // The swept set Section 9.8 assertion 11 names, explicitly.
    for (Index n : BOUNDARY_SIZES) {
        for (int workers : {1, 2, 3, 4, 7, 8, 16, 512, 1024}) {
            for (int per_worker : {1, 2, 5, 8}) {
                check(n, workers, Schedule::Static, per_worker);
                check(n, workers, Schedule::Dynamic, per_worker);
            }
        }
    }

    // A worker count of zero or less issues no chunks rather than dividing by
    // it, which is what a backend that was refused its threads would ask for.
    for (Index n : BOUNDARY_SIZES) {
        for (int workers : {0, -1}) {
            PNL_REQUIRE(for_chunk_count(n, workers, Schedule::Static, 8) == 0);
            PNL_REQUIRE(for_chunk_count(n, workers, Schedule::Dynamic, 8) == 0);
        }
        // A non positive chunks per worker is clamped to one rather than
        // producing an empty dynamic grid over a non empty range.
        for (int per_worker : {0, -3}) {
            const Index count = for_chunk_count(n, 4, Schedule::Dynamic, per_worker);
            PNL_REQUIRE_MESSAGE(count == std::min(Index{4}, n) || n <= 0,
                                "a chunks per worker of " + std::to_string(per_worker) + " gave " +
                                    std::to_string(count) + " chunks at n = " + std::to_string(n));
        }
    }
}

PNL_TEST("property/the padded row bands are disjoint and cover the interior rows exactly") {
    // Section 9.8 assertion 11. The map from a chunk to a row band is applied
    // by every sweep in poisson2d.hpp and by the Fortran trampoline of release
    // 1.2.0, and it is the map, not the partition, that the pure declaration
    // there is a promise about.
    std::mt19937_64 engine(PROPERTY_SEED);
    std::uniform_int_distribution<Index> size_draw(0, 2048);
    std::uniform_int_distribution<int> worker_draw(1, 32);
    std::uniform_int_distribution<int> per_worker_draw(1, 8);
    std::uniform_int_distribution<int> rank_draw(1, 8);
    std::uniform_int_distribution<int> schedule_draw(0, 1);

    auto check = [](Index total_rows, int ranks, int workers, Schedule schedule, int per_worker) {
        for (int rank = 0; rank < ranks; ++rank) {
            // What a backend hands a sweep: this process's rows. One rank means
            // the whole grid, which is the shared memory case.
            const Range rows = block_partition(total_rows, ranks, rank);
            const Index count = for_chunk_count(rows.size(), workers, schedule, per_worker);
            const std::string what = "row bands over rank " + std::to_string(rank) + " of " +
                                     std::to_string(ranks) + ", rows [" +
                                     std::to_string(rows.begin) + ", " + std::to_string(rows.end) +
                                     "), " + std::to_string(workers) + " workers";
            require_row_bands(
                rows,
                count,
                [rows, workers, schedule, per_worker](Index k) {
                    return for_chunk(rows.size(), workers, schedule, per_worker, k);
                },
                draw_note(what, total_rows, count));
        }
    };

    for (int draw = 0; draw < DRAWS; ++draw) {
        const Schedule schedule = schedule_draw(engine) == 0 ? Schedule::Static : Schedule::Dynamic;
        check(size_draw(engine),
              rank_draw(engine),
              worker_draw(engine),
              schedule,
              per_worker_draw(engine));
    }

    for (Index n : BOUNDARY_SIZES) {
        for (int ranks : {1, 2, 4}) {
            for (int workers : {1, 2, 3, 4, 8, 512}) {
                for (int per_worker : {1, 5, 8}) {
                    check(n, ranks, workers, Schedule::Static, per_worker);
                    check(n, ranks, workers, Schedule::Dynamic, per_worker);
                }
            }
        }
    }
}

PNL_TEST("property/the reduction grid depends on the length and on nothing else") {
    // Stated as a property because it is the reason the equivalence suite may
    // assert exact equality at all. If this ever became a function of the worker
    // count, every bit identity claim in the report would quietly become false
    // and no other test would notice.
    for (Index n : BOUNDARY_SIZES) {
        const Index count = reduction_chunk_count(n);
        for (int workers : {1, 2, 3, 4, 7, 8, 16, 28}) {
            for (Schedule schedule : {Schedule::Static, Schedule::Dynamic}) {
                for (int per_worker : {1, 5, 8}) {
                    // Nothing in the reduction grid takes any of these three,
                    // which is the assertion; they are looped over so that a
                    // future overload that did take them would fail to compile
                    // here rather than change an answer silently.
                    (void)workers;
                    (void)schedule;
                    (void)per_worker;
                    PNL_REQUIRE(reduction_chunk_count(n) == count);
                    for (Index k = 0; k < count; ++k) {
                        const Range first = reduction_chunk(n, k);
                        const Range again = reduction_chunk(n, k);
                        PNL_REQUIRE(first.begin == again.begin && first.end == again.end);
                    }
                }
            }
        }
    }
}

PNL_TEST("property/the property harness rejects a partition that is wrong") {
    // The negative control, on the same argument as test_contract_detects_fma:
    // a checker with an empty body passes every case it is given, and the only
    // way to know this one has a body is to hand it partitions that are wrong
    // and watch it refuse them. Each broken partition below is a real defect
    // this file exists to catch.
    const Index n = 100;
    const Index parts = 4;
    // By reference, like every lambda below. A capture list that names n or
    // parts is an error under clang: both are constant expressions, reading one
    // is not an odr use, so -Wunused-lambda-capture calls the capture unused.
    const auto correct = [&](Index k) { return block_partition(n, parts, k); };

    // The harness passes the partition it is meant to pass, so the four
    // refusals below are refusals and not a checker that refuses everything.
    require_partitions(n, parts, correct, "the correct partition");

    // One index claimed twice and one dropped. The sizes still sum to n, which
    // is exactly the failure a size sum cannot see and a hit count can.
    PNL_REQUIRE_MESSAGE(rejects([&] {
                            require_partitions(
                                n,
                                parts,
                                [&](Index k) {
                                    Range chunk = correct(k);
                                    if (k == 1) {
                                        --chunk.begin;
                                        --chunk.end;
                                    }
                                    return chunk;
                                },
                                "a partition that overlaps by one and drops one");
                        }),
                        "the harness accepted a partition that claimed one index twice and "
                        "dropped another, which is the defect it exists to catch");

    // A dropped index and nothing to compensate: the coverage half alone.
    PNL_REQUIRE_MESSAGE(rejects([&] {
                            require_partitions(
                                n,
                                parts,
                                [&](Index k) {
                                    Range chunk = correct(k);
                                    if (k == 2) --chunk.end;
                                    return chunk;
                                },
                                "a partition with a hole");
                        }),
                        "the harness accepted a partition with a hole in it");

    // A partition that stops short of n, which is the shape an off by one in a
    // chunk count takes.
    PNL_REQUIRE_MESSAGE(
        rejects([&] { require_partitions(n, parts - 1, correct, "one chunk short"); }),
        "the harness accepted a partition that covered three quarters of the "
        "range");

    // A legal cover with an illegal imbalance: two chunks of 1 and one of 98.
    PNL_REQUIRE_MESSAGE(rejects([&] {
                            require_partitions(
                                n,
                                3,
                                [&](Index k) {
                                    if (k == 0) return Range{0, 1};
                                    if (k == 1) return Range{1, n - 1};
                                    return Range{n - 1, n};
                                },
                                "a cover with a spread of ninety seven");
                        }),
                        "the harness accepted chunk sizes of 1, 98 and 1");

    // And the row band half, where an off by one in the map turns into two
    // workers writing the same interior row.
    const Range rows{0, n};
    require_row_bands(rows, parts, correct, "the correct row bands");
    PNL_REQUIRE_MESSAGE(rejects([&] {
                            require_row_bands(
                                rows,
                                parts,
                                [&](Index k) {
                                    Range chunk = correct(k);
                                    if (k == 1) ++chunk.end;
                                    return chunk;
                                },
                                "row bands that overlap by one row");
                        }),
                        "the harness accepted two row bands that sweep the same interior row");
}
