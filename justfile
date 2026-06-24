# https://github.com/casey/just
alias t := test
alias tdd := test-watch
alias ds := dataset
alias art := artifact
alias ev := eval

set quiet

export UV_CACHE_DIR := ".uv_cache"

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# Sync dev dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Sync dev dependencies, then reinstall third-party native bindings removed by uv sync
dev-sync-native: dev-sync install-3rdparty

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
	LOG_LEVEL=WARNING uv run pytest --verbose --color=yes --log-cli-level=WARNING src
test-smoke:
	@echo "💨 Running CLI smoke tests..."
	uv run pytest -m smoke --verbose --color=yes tests/smoke
test-watch:
	@echo "🔍 Watching for changes and running tests..."
	uv run ptw .
test-regression:
	@echo "🧪 Running Regression tests..."
	uv run pytest -m regression --verbose --color=yes
test-full: test test-smoke
	@echo "🎉 All tests passed!"

# Resolve or run pipeline commands through the Python CLI
pipeline +ARGS:
	uv run python -m pipeline.cli pipeline {{ARGS}}

# Resolve pipeline profiles through the Python CLI
profile +ARGS:
	uv run python -m pipeline.cli profile {{ARGS}}

# Manage supported dataset manifests
dataset +ARGS:
	uv run dataset {{ARGS}}

# Manage local research artifact manifests
artifact +ARGS:
	uv run artifact {{ARGS}}

# Export and evaluate pipeline run results
eval +ARGS:
	uv run python -m evaluation.cli {{ARGS}}

# Alias for artifact manifest management
artifacts +ARGS:
	uv run artifacts {{ARGS}}

# Manage generated pipeline logs and run artifacts
logs command:
	@if [ "{{command}}" = "clear" ]; then \
		mkdir -p pipeline/out; \
		find pipeline/out -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
	else \
		echo "Usage: just logs clear" >&2; \
		exit 2; \
	fi

# Run all checks: format, lint, and test
validate: format lint test

# Clear the cache for the Euroc dataset(Currently only for the Euroc dataset)
clear-ds-cache:
	rm -rf datasets/euroc_mh_01/cache

install-pydbow3 +FLAGS='':
	bash scripts/install_pydbow3.sh {{FLAGS}}

# Install local third-party native Python bindings that are not tracked by uv.lock
install-3rdparty:
	just install-pydbow3

pipeline-new-node node_name:
	@echo "Creating a new node in the pipeline..."
	cd pipeline && dora new --kind node {{node_name}} --lang python
	@echo "Node created successfully!"
