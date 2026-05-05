# https://github.com/casey/just
alias t := test
alias tdd := test-watch

set quiet

export UV_CACHE_DIR := ".uv_cache"

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# Sync dev dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Sync dev dependencies, then reinstall native bindings removed by uv sync
dev-sync-native: dev-sync native-deps

# Sync production dependencies (excludes dev dependencies)
prod-sync:
	uv sync --all-extras --no-dev --cache-dir .uv_cache

# Install pre commit hooks
install-hooks:
	uv run pre-commit install

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint +TARGETS='':
	uv run --no-sync ruff check --fix --show-fixes {{TARGETS}}
	uv run --no-sync ty check {{TARGETS}}

# Run tests using pytest
test:
	@echo "🧪 Running Unit & Integration tests..."
	uv run pytest --verbose --color=yes src
test-watch:
	@echo "🔍 Watching for changes and running tests..."
	uv run ptw .
test-dora:
	@echo "Building and running the pipeline..."
	dora build ./pipeline/dataflow.yml --uv
	@echo "🔍 Running Dora pipeline tests..."
	uv run pytest --verbose --color=yes pipeline
test-regression:
	@echo "🧪 Running Regression tests..."
	uv run pytest -m regression --verbose --color=yes
test-full: test test-dora
	@echo "🎉 All tests passed!"

# Run all checks: format, lint, and test
validate: format lint test

# Clear the cache for the Euroc dataset(Currently only for the Euroc dataset)
clear-ds-cache:
	rm -rf datasets/euroc_v_01_easy/cache


install-gtsam +FLAGS='-q':
	bash scripts/install_gtsam.sh {{FLAGS}}

install-pydbow3 +FLAGS='':
	bash scripts/install_pydbow3.sh {{FLAGS}}

# Install local native Python bindings that are not tracked by uv.lock
native-deps:
	just install-gtsam
	just install-pydbow3

pipeline-new-node node_name:
	@echo "Creating a new node in the pipeline..."
	cd pipeline && dora new --kind node {{node_name}} --lang python
	@echo "Node created successfully!"
