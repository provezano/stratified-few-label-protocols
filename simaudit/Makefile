.PHONY: help requirements requirements-dev format format-check lint typecheck test test-fast crap verify mutation mutation-results bench quality clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################
PACKAGE    = src/simaudit
TESTS      = tests
TOOLS      = tools
UV         = uv
PY         = $(UV) run
# One BLAS/numba thread per process: parallelism comes from the job executor,
# and single-threaded BLAS keeps results bit-for-bit reproducible.
export VECLIB_MAXIMUM_THREADS = 1
export OMP_NUM_THREADS        = 1
export OPENBLAS_NUM_THREADS   = 1
export NUMBA_NUM_THREADS      = 1

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install runtime dependencies
requirements:
	$(UV) sync

## Install runtime + development dependencies
requirements-dev:
	$(UV) sync --extra dev

## Format files using black
format:
	$(PY) black $(PACKAGE) $(TESTS) $(TOOLS)

## Check formatting without modifying files
format-check:
	$(PY) black --check --diff $(PACKAGE) $(TESTS) $(TOOLS)

## Lint with flake8 (docstrings + cognitive complexity <= 15)
lint:
	$(PY) flake8 $(PACKAGE) $(TESTS) $(TOOLS)

## Static type check (mypy --strict)
typecheck:
	$(PY) mypy $(PACKAGE) $(TOOLS)

## Run the test suite with branch coverage (fails under 90%)
test:
	$(PY) pytest $(TESTS)

## Run tests without coverage (faster local loop)
test-fast:
	$(PY) pytest $(TESTS) -q --no-cov -p no:cacheprovider

## Fail if any function has a CRAP score above 30 (needs coverage.json from `test`)
crap:
	$(PY) python $(TOOLS)/crap.py --coverage coverage.json --source $(PACKAGE) --fail-above 30 --warn-above 15

## Everything the quality gate checks, in order
verify: format-check lint typecheck test crap

## Mutation testing on the core modules (slow; not part of verify)
mutation:
	NUMBA_DISABLE_JIT=1 $(PY) mutmut run

## Show surviving mutants
mutation-results:
	$(PY) mutmut results

## Performance benchmarks
bench:
	$(PY) pytest $(TESTS) -m bench --no-cov -q

## Run pre-commit against all files
quality:
	$(PY) pre-commit run --all-files

## Delete compiled files and caches
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .hypothesis htmlcov coverage.xml coverage.json .coverage mutants

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Show this help
help:
	@awk '/^## /{desc=substr($$0,4); next} /^[a-zA-Z_-]+:/{if(desc){split($$1,t,":"); printf "  \033[36m%-18s\033[0m %s\n", t[1], desc; desc=""}}' $(MAKEFILE_LIST)
