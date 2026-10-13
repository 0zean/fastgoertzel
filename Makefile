# Developer shortcuts. CI runs the same commands (see .github/workflows/ci.yml).

.PHONY: install build test lint format fix typecheck precommit bench latency kernel-bench wheel sdist clean check

# Create the virtual environment and build + install the extension (editable).
install:
	uv sync --locked

# Rebuild the extension after editing C++ or CMake files.
build:
	uv sync --locked --reinstall-package fastgoertzel

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

# Apply lint fixes and formatting (Python via ruff, C++ via clang-format through pre-commit).
fix:
	uv run ruff check --fix .
	uv run ruff format .
	uv run pre-commit run clang-format --all-files

# mypy on the Python sources plus stubtest, which checks _core.pyi against the built module.
typecheck:
	uv run mypy
	uv run python -m mypy.stubtest fastgoertzel._core

precommit:
	uv run pre-commit run --all-files

# Full Python benchmark, writes a machine-readable report.
bench:
	uv run python benchmarks/benchmark.py --json bench.json

# Per-call latency distribution only (the part relevant to real-time use).
latency:
	uv run python benchmarks/benchmark.py --latency-only

# Standalone C++ kernel micro-benchmark (no Python involved).
kernel-bench:
	cmake -S . -B build/bench -DFASTGOERTZEL_BUILD_PYTHON=OFF -DFASTGOERTZEL_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build build/bench --config Release
	@echo "binary is under build/bench (Release/kernel_bench.exe on MSVC, kernel_bench elsewhere)"

wheel:
	uv run python -m build --wheel -o dist/

sdist:
	uv run python -m build --sdist -o dist/

clean:
	rm -rf build dist src/fastgoertzel/*.pyd src/fastgoertzel/*.so

check: lint format typecheck test
