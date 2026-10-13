# Developer shortcuts. CI runs the same commands (see .github/workflows/ci.yml).

.PHONY: install build test lint format fix bench latency wheel sdist clean check

# Create the virtual environment, build the extension in place, install everything (editable).
install:
	uv sync --locked

# Rebuild the extension after editing C++ (editable install picks it up on next import).
build:
	uv sync --locked --reinstall-package fastgoertzel

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

# Apply lint fixes and formatting.
fix:
	uv run ruff check --fix .
	uv run ruff format .

# Full benchmark, writes a machine-readable report.
bench:
	uv run python benchmarks/benchmark.py --json bench.json

# Per-call latency distribution only (the part relevant to real-time use).
latency:
	uv run python benchmarks/benchmark.py --latency-only

wheel:
	uv run python -m build --wheel -o dist/

sdist:
	uv run python -m build --sdist -o dist/

clean:
	rm -rf build dist python/fastgoertzel.egg-info python/fastgoertzel/*.pyd python/fastgoertzel/*.so

check: lint format test
