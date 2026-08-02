#!/usr/bin/env bash
#
# Entry point for the benchmark sweep.
#
# The resolution of sweep_matrix.yaml, the resume logic and the atomic merge
# live in run_sweep.py, because doing them in shell would mean parsing YAML with
# sed and that is how sweeps end up silently skipping configurations. This
# script is the stable command line the Makefile and the documentation refer to.
#
# Usage:
#   ./run_sweep.sh                      run everything not already recorded
#   ./run_sweep.sh --only scaling       run one block
#   ./run_sweep.sh --force              redo rows that already exist
#   ./run_sweep.sh --dry-run            print the commands and stop

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/.." && pwd)"
build="${PNL_BUILD_DIR:-${root}/build}"

if [ ! -x "${build}/pnl" ]; then
    echo "run_sweep: ${build}/pnl not found." >&2
    echo "Build first:  cmake --build \"${build}\" -j 6" >&2
    exit 2
fi

# Keep the thread backends from fighting the sweep's own worker counts. The
# binary sets its own thread count per run; leaving OMP_NUM_THREADS set in the
# environment would silently override the scaling curve.
unset OMP_NUM_THREADS || true

exec python3 "${here}/run_sweep.py" --build "${build}" "$@"
