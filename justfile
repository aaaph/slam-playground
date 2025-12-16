# https://github.com/casey/just
alias t := test
alias tdd := test-watch

set quiet

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# Sync dev dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

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
lint:
	uv run ruff check --fix --show-fixes
	uv run ty check

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
test-full: test test-dora
	@echo "🎉 All tests passed!"

# Run all checks: format, lint, and test
validate: format lint test

# Clear the cache for the Euroc dataset(Currently only for the Euroc dataset)
clear-ds-cache:
	rm -rf datasets/euroc_v_01_easy/cache


install-gtsam +FLAGS='-q':
	bash scripts/install_gtsam.sh {{FLAGS}}
