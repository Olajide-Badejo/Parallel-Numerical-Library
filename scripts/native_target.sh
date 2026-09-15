#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# What -march=native resolves to, for one compiler on this machine.
#
#   scripts/native_target.sh COMPILER FILE
#
# -march=native is resolved by the compiler driver from the processor it runs
# on, so one command line compiles for different instruction sets on different
# processors. A compiler cache hashes the command line as it is spelled, so a
# cache carried from one machine to another serves objects the second processor
# may not be able to run. Continuous integration does exactly that: each job
# restores a ccache filled by an earlier run on whatever processor that runner
# had, and in run 34926377048 two jobs died on an illegal instruction. That is
# CI-07 in docs/ENGINEERING_LOG.md.
#
# The compiler's predefined macros are its own account of what it resolved: one
# for every instruction set extension it enabled, and the architecture and the
# tuning it chose. Two machines on which they agree get code either can run. So
# FILE receives them, sorted, and stdout gets the first sixteen hex digits of
# their SHA-256. The workflow puts that signature in every compiler cache key
# and hands FILE to ccache as an extra file to hash. The macros -march=native
# added to the compiler's defaults, and the processor they were resolved on, go
# to stderr for the log.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: native_target.sh COMPILER FILE" >&2
    exit 2
fi
compiler=$1
file=$2

"$compiler" -march=native -E -dM -x c++ /dev/null | LC_ALL=C sort >"$file"
if [ ! -s "$file" ]; then
    echo "native_target: $compiler defined no macros under -march=native" >&2
    exit 1
fi

defaults=$("$compiler" -E -dM -x c++ /dev/null | LC_ALL=C sort)
added=$(LC_ALL=C comm -13 <(printf '%s\n' "$defaults") "$file" |
    sed -n 's/^#define \([A-Za-z0-9_]*\).*/\1/p' | tr '\n' ' ')
processor=
if [ -r /proc/cpuinfo ]; then
    processor=$(sed -n '/^model name/{s/^model name[[:space:]]*: //p;q}' /proc/cpuinfo)
fi
echo "native_target: $compiler on ${processor:-a processor /proc/cpuinfo does not name}" >&2
echo "native_target: -march=native defines ${added:-nothing beyond the defaults}" >&2
sha256sum "$file" | cut -c1-16
