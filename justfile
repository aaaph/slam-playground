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
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package python_repo_template

# Run tests using pytest
test:
	uv run pytest --verbose --color=yes

test-watch:
	uv run ptw .

# Run all checks: format, lint, and test
validate: format lint test
