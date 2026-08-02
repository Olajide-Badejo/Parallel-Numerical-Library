# Parallel Numerical Library.
#
#   make setup     check the toolchain and report what is missing
#   make build     configure and compile
#   make test      run every gate from Section 10
#   make sweep     run the benchmark matrix into experiments/results
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
CXX_COMPILER ?= g++-16
BUILD_TYPE   ?= Release

# The .wslconfig on the target machine budgets 12 GB to the guest and its own
# comment warns that link steps are memory hungry, so the default job count is
# deliberately below the core count.
CMAKE   ?= cmake
CTEST   ?= ctest
PYTHON  ?= python3

.PHONY: all setup build configure test test-quick sweep sweep-force assets \
        report report-only report-debug report-personal reports check-style \
        format bandwidth bandwidth-refresh topology clean distclean help

help:
	@sed -n '2,20p' Makefile | sed 's/^# \{0,1\}//'

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
	    echo; echo "setup: something above is missing. See docs/BUILD_SPECIFICATION.md section 3."; \
	    exit 1; \
	fi
	@echo "setup: toolchain complete"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
configure:
	@$(CMAKE) -S "$(ROOT)" -B "$(ROOT)/$(BUILD)" -G Ninja \
	    -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	@$(CMAKE) --build "$(ROOT)/$(BUILD)" -j $(JOBS)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: build
	@cd "$(ROOT)/$(BUILD)" && $(CTEST) --output-on-failure -j 2

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

format:
	@find "$(ROOT)/include" "$(ROOT)/src" "$(ROOT)/tests" \
	    \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
	    -exec clang-format -i {} +
	@echo "format: done"

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
sweep: build
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)"

sweep-force: build
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)" --force

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
	@"$(ROOT)/benchmarks/run_sweep.sh" --build "$(ROOT)/$(BUILD)" --refresh-bandwidth

topology: build
	@"$(ROOT)/$(BUILD)/pnl" --topology

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
assets:
	@$(PYTHON) "$(ROOT)/scripts/gen_report_assets.py"

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

report-personal:
	@cd "$(ROOT)/report_for_me" && latexmk -pdf -interaction=nonstopmode -halt-on-error report_for_me.tex
	@$(PYTHON) "$(ROOT)/scripts/check_no_dashes.py" "$(ROOT)/report_for_me/report_for_me.pdf"
	@echo "report-personal: $(ROOT)/report_for_me/report_for_me.pdf"

reports: report report-debug report-personal

# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------
all: setup build check-style test sweep bandwidth-refresh reports
	@echo
	@echo "all: complete."
	@echo "  summary   experiments/results/summary.csv"
	@echo "  manifest  experiments/results/session_manifest.json"
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
	@echo "distclean: measurements removed too"
