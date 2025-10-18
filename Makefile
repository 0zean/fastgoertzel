# Makefile

# Install dependencies
install:
	@echo "🚀 Creating virtual environment using uv"
	uv sync --frozen
	uv run pip install numpy pytest ruff fastgoertzel

# Format code with Black and isort
format:
	uv run ruff format --check ./tests ./benchmarks
	uv run ruff check --select I ./tests ./benchmarks

# Linting and formatting with ruff
lint:
	uv run ruff check ./tests ./benchmarks

# Run tests
test:
	uv run pytest ./tests/

# Run all checks
check: format lint test

.PHONY: install format lint test check