# SPDX-License-Identifier: MIT
# Parallel Numerical Library.
#
#   make setup     check the toolchain and report what is missing
#   make build     configure and compile
#   make test      run every gate from Section 10, except the perf label
#   make test-perf run the relative performance gate on its own
#   make install-test   stage an install and build examples/ against it
#   make sweep     run the benchmark matrix into experiments/results
#   make sweep-interim        the same, into experiments/results/interim
#   make bandwidth-refresh-interim   re-probe into experiments/results/interim
#   make assets    regenerate figures and tables from summary.csv
#   make report    build the main PDF
#   make reports   build all three PDFs
#   make all       everything above, in order
#   make clean     remove build trees and generated report assets
#
# Note on paths. This repository may live under a directory whose name contains
# spaces, which GNU make cannot express as a target or prerequisite. Every rule
# here is therefore phony and every path is quoted inside the recipe, where the
# shell handles it correctly. Do not add a file based rule with an absolute
# path; it will break silently on such a checkout.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ROOT    := $(CURDIR)
BUILD   ?= build
JOBS    ?= 6
CXX_COMPILER ?= g++-15
BUILD_TYPE   ?= Release

# The .wslconfig on the target machine budgets 12 GB to the guest and its own
# comment warns that link steps are memory hungry, so the default job count is
# deliberately below the core count.
CMAKE   ?= cmake
CTEST   ?= ctest
PYTHON  ?= python3

.PHONY: all setup build configure install-test test test-perf test-quick sweep sweep-force \
        sweep-interim bandwidth-refresh-interim assets \
        report report-only report-debug report-personal reports check-style \
        format bandwidth bandwidth-refresh topology clean distclean help

help:
	@sed -n '2,22p' Makefile | sed 's/^# \{0,1\}//'

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
setup:
	@echo "== toolchain =="
	@missing=0; \
	for tool in $(CXX_COMPILER) cmake ninja mpirun $(PYTHON) latexmk; do \
	    if command -v $$tool >/dev/null 2>&1; then \
	        printf "  %-12s %s\n" "$$tool" "$$($$tool --version 2>&1 | head -1)"; \
	    else \
	        printf "  %-12s MISSING\n" "$$tool"; missing=1; \
	    fi; \
	done; \
	if command -v nvcc >/dev/null 2>&1; then \
	    printf "  %-12s %s\n" nvcc "$$(nvcc --version | tail -2 | head -1)"; \
	else \
	    printf "  %-12s absent, the CUDA backend will be skipped\n" nvcc; \
	fi; \
	printf "  %-12s " python-modules; \
	$(PYTHON) -c "import yaml, matplotlib, pandas; print('yaml, matplotlib, pandas present')" \
	    || { echo "MISSING: pip install pyyaml matplotlib pandas"; missing=1; }; \
	if [ $$missing -ne 0 ]; then \
	    echo; echo "setup: something above is missing. See the Environment section of README.md."; \
	    exit 1; \
	fi
	@echo "setup: toolchain complete"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# -DPNL_WERROR=ON because the option now defaults to OFF. The default belongs to
# a consumer, who must never inherit a warnings as errors policy from a library;
# the developer build is the one that has to stay strict, and this is where the
# developer build is defined.
#
# CMAKE_EXTRA is appended to the configure line and is empty by default. It is
# how a second compiler or a CI job turns a feature off without a hand written
# cmake command that would then be free to drift from this one:
#
#     make build test BUILD=build-clang CXX_COMPILER=clang++ \
#          CMAKE_EXTRA=-DPNL_ENABLE_CUDA=OFF
#
# It is deliberately unquoted, so several options separate with spaces.
CMAKE_EXTRA ?=

configure:
	@$(CMAKE) -S "$(ROOT)" -B "$(ROOT)/$(BUILD)" -G Ninja \
	    -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DPNL_WERROR=ON $(CMAKE_EXTRA)

build: configure
	@$(CMAKE) --build "$(ROOT)/$(BUILD)" -j $(JOBS)

# The install test, and the phase B1 gate. It installs to a staging prefix under
# the build tree, then configures examples/ against nothing but that prefix and
# runs what comes out. examples/ declares LANGUAGES CXX and knows nothing about
# this repository, so it fails when the exported package demands something a
# stranger does not have, which is the only failure mode that never shows up in
# a build tree consumer. CI runs this too, from phase B2.
STAGE ?= $(BUILD)/stage

install-test: build
	@rm -rf "$(ROOT)/$(STAGE)" "$(ROOT)/$(BUILD)/examples"
	@$(CMAKE) --install "$(ROOT)/$(BUILD)" --prefix "$(ROOT)/$(STAGE)" >/dev/null
	@echo "install-test: staged into $(STAGE)"
	@cd "$(ROOT)/$(STAGE)" && find . -type f | sed 's|^\./||' | sort
	@echo
	@$(CMAKE) -S "$(ROOT)/examples" -B "$(ROOT)/$(BUILD)/examples" -G Ninja \
	    -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DCMAKE_PREFIX_PATH="$(ROOT)/$(STAGE)"
	@$(CMAKE) --build "$(ROOT)/$(BUILD)/examples" -j $(JOBS)
	@echo
	@"$(ROOT)/$(BUILD)/examples/poisson"
	@echo
	@"$(ROOT)/$(BUILD)/examples/custom_backend"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# -LE perf excludes the relative performance gate, which is a timing test and
# has to be run where a number means something rather than on a laptop with a
# browser open. It is not optional, only separate:
#
#     ctest --test-dir build --output-on-failure -L perf
#
# runs it, and CI has a step that does. A timing test inside the default run is
# the one that eventually gets disabled, and a disabled gate is worse than a
# separate one.
test: build
	@cd "$(ROOT)/$(BUILD)" && $(CTEST) --output-on-failure -j 2 -LE perf

# The performance gate on its own, serially, because a ratio measured while the
# rest of the suite is running is a ratio about the machine's spare capacity.
test-perf: build
	@cd "$(ROOT)/$(BUILD)" && $(CTEST) --output-on-failure -L perf

# Unit and style only, for a fast inner loop.
test-quick: build
	@cd "$(ROOT)/$(BUILD)" && $(CTEST) --output-on-failure -L "unit|style"

check-style:
	@$(PYTHON) "$(ROOT)/scripts/check_no_dashes.py" "$(ROOT)"
	@$(PYTHON) "$(ROOT)/tests/style/check_linter.py"
	@if command -v ruff >/dev/null 2>&1; then \
	    ruff check "$(ROOT)/benchmarks" "$(ROOT)/scripts" "$(ROOT)/tests"; \
	else \
	    echo "check-style: ruff not installed, skipping the Python lint"; \
	fi

# examples/ is in the list because it is source this repository owns and the
# phase B4 gate runs clang-format over it.
format:
	@find "$(ROOT)/include" "$(ROOT)/src" "$(ROOT)/tests" "$(ROOT)/examples" \
	    \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
	    -exec clang-format -i {} +
	@echo "format: done"

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
# --migrate on both, so that a summary written by an older binary gains the
# columns this one emits rather than stopping the sweep. The alternative is that
# every schema change breaks `make all` from the moment it lands until a full re
# measurement finishes, which is hours, and leaves the repository unable to build
# a report in between.
#
# RESULTS_DIR is where the summary and this session's manifest land. The interim
# targets below point it at experiments/results/interim, which the results ignore
# block excludes: a sweep run to prove the pipeline is clean must not be able to
# overwrite the generation the report is built from.
RESULTS_DIR ?= $(ROOT)/experiments/results

sweep: build
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)" \
	    --results-dir "$(RESULTS_DIR)" --migrate

sweep-force: build
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)" \
	    --results-dir "$(RESULTS_DIR)" --migrate --force

sweep-interim:
	@$(MAKE) --no-print-directory sweep RESULTS_DIR="$(ROOT)/experiments/results/interim"

bandwidth-refresh-interim:
	@$(MAKE) --no-print-directory bandwidth-refresh \
	    RESULTS_DIR="$(ROOT)/experiments/results/interim"

bandwidth: build
	@"$(ROOT)/$(BUILD)/pnl" --bandwidth --backend openmp

# Re-probe both devices and update the manifest, running no configurations.
#
# This runs after the sweep rather than before it, and the ordering matters. The
# sweep driver probes at the start of its session, which is immediately after a
# build and a test run, so the machine is still busy and the host figure comes
# out low: 39.8 GiB/s measured that way against 60.7 on an idle machine. Every
# host efficiency number in the report divides by that figure, so a depressed
# reading would inflate all of them. Re-probing once the sweep has finished is
# the only point in the pipeline where the machine is reliably quiet.
bandwidth-refresh: build
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)" \
	    --results-dir "$(RESULTS_DIR)" --refresh-bandwidth

topology: build
	@"$(ROOT)/$(BUILD)/pnl" --topology

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
# Empty by default, so ground rule 6 holds: the generator refuses to build an
# asset from a row whose commit stamp ends in .dirty, and `make report` fails
# with the reason. Every row in the committed summary is dirty until phase A8a
# re measures from a clean tree, so a developer who wants the PDF before then
# asks for it in as many words:
#
#     make report ASSET_FLAGS=--allow-dirty
#
# and gets a report the generator has already said on stdout is not publishable.
ASSET_FLAGS ?=

# Two runs, because there are two outputs and they are not the same document.
# The first writes the report's figures, tables and the commands its prose
# quotes; the second rewrites the generated region of
# docs/comparison_methodology.md, which is canonical for the comparison and used
# to keep its tables in step with the report by hand. They did not stay in step,
# which is finding 4.3.
assets:
	@$(PYTHON) "$(ROOT)/scripts/gen_report_assets.py" $(ASSET_FLAGS)
	@$(PYTHON) "$(ROOT)/scripts/gen_report_assets.py" $(ASSET_FLAGS) --markdown

report: assets
	@$(MAKE) --no-print-directory report-only

report-only:
	@cd "$(ROOT)/report" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	@$(PYTHON) "$(ROOT)/scripts/check_no_dashes.py" "$(ROOT)/report/main.pdf"
	@echo "report: $(ROOT)/report/main.pdf"

report-debug:
	@cd "$(ROOT)/report_debug" && latexmk -pdf -interaction=nonstopmode -halt-on-error debug_report.tex
	@$(PYTHON) "$(ROOT)/scripts/check_no_dashes.py" "$(ROOT)/report_debug/debug_report.pdf"
	@echo "report-debug: $(ROOT)/report_debug/debug_report.pdf"

# An optional third report kept outside the repository. The target is a no
# operation when its directory is absent, so a fresh clone builds cleanly.
report-personal:
	@if [ -d "$(ROOT)/report_for_me" ]; then \
	    cd "$(ROOT)/report_for_me" && \
	    latexmk -pdf -interaction=nonstopmode -halt-on-error report_for_me.tex && \
	    $(PYTHON) "$(ROOT)/scripts/check_no_dashes.py" "$(ROOT)/report_for_me/report_for_me.pdf" && \
	    echo "report-personal: $(ROOT)/report_for_me/report_for_me.pdf"; \
	else \
	    echo "report-personal: not present, skipping"; \
	fi

reports: report report-debug report-personal
	@$(PYTHON) "$(ROOT)/scripts/publish_assets.py"

# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------
all: setup build check-style test sweep bandwidth-refresh reports
	@echo
	@echo "all: complete."
	@echo "  summary   experiments/results/summary.csv"
	@echo "  manifest  experiments/results/manifest-<commit>-<timestamp>.json"
	@echo "  reports   report/main.pdf, report_debug/debug_report.pdf, report_for_me/report_for_me.pdf"

clean:
	@rm -rf "$(ROOT)/$(BUILD)"
	@rm -rf "$(ROOT)/report/build" "$(ROOT)/report_debug/build" "$(ROOT)/report_for_me/build"
	@rm -f "$(ROOT)/report"/*.aux "$(ROOT)/report"/*.log "$(ROOT)/report"/*.out \
	       "$(ROOT)/report"/*.toc "$(ROOT)/report"/*.fdb_latexmk "$(ROOT)/report"/*.fls \
	       "$(ROOT)/report"/*.bbl "$(ROOT)/report"/*.blg "$(ROOT)/report"/*.pdf
	@rm -f "$(ROOT)/report_debug"/*.aux "$(ROOT)/report_debug"/*.log "$(ROOT)/report_debug"/*.out \
	       "$(ROOT)/report_debug"/*.toc "$(ROOT)/report_debug"/*.fdb_latexmk \
	       "$(ROOT)/report_debug"/*.fls "$(ROOT)/report_debug"/*.pdf
	@rm -f "$(ROOT)/report_for_me"/*.aux "$(ROOT)/report_for_me"/*.log "$(ROOT)/report_for_me"/*.out \
	       "$(ROOT)/report_for_me"/*.toc "$(ROOT)/report_for_me"/*.fdb_latexmk \
	       "$(ROOT)/report_for_me"/*.fls "$(ROOT)/report_for_me"/*.pdf
	@rm -f "$(ROOT)/report/figures"/*.pdf "$(ROOT)/report/tables"/*.tex
	@echo "clean: done. Measured results under experiments/results are kept;"
	@echo "       remove them by hand if you really mean to discard the sweep."

# Also discards the measurements. Separate from clean on purpose: an hour of
# sweep should not disappear because someone wanted a fresh build.
distclean: clean
	@rm -f "$(ROOT)/experiments/results"/*.csv "$(ROOT)/experiments/results"/*.json
	@rm -rf "$(ROOT)/experiments/results/interim"
	@echo "distclean: measurements removed too. experiments/results/archive is kept;"
	@echo "           it holds superseded generations and removing it loses history."
