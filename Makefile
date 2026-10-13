# Developer shortcuts. CI runs the same commands (see .github/workflows/ci.yml).

.PHONY: install build test lint format fix typecheck precommit bench latency cpp-test tidy readme-tables kernel-bench docs wheel sdist clean check

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
	uv run python -m mypy.stubtest fastgoertzel._core --allowlist tests/stubtest_allowlist.txt

precommit:
	uv run pre-commit run --all-files

# Full Python benchmark, writes a machine-readable report.
bench:
	uv run python benchmarks/benchmark.py --json bench.json

# Per-call latency distribution only (the part relevant to real-time use).
latency:
	uv run python benchmarks/benchmark.py --latency-only

# C++ unit tests for the header-only core (doctest is fetched by CMake).
cpp-test:
	cmake -S . -B build/cpp -DFASTGOERTZEL_BUILD_PYTHON=OFF -DFASTGOERTZEL_BUILD_TESTS=ON -DFASTGOERTZEL_WERROR=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build build/cpp --config Release
	ctest --test-dir build/cpp --build-config Release --output-on-failure

# Static analysis of the C++ sources (needs clang-tidy and run-clang-tidy on PATH).
tidy:
	cmake -S . -B build/tidy -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DFASTGOERTZEL_OPENMP=OFF -Dpybind11_DIR="$$(uv run python -m pybind11 --cmakedir)"
	run-clang-tidy -p build/tidy -quiet 'cpp/.*\.cpp$$'

# Re-render the README benchmark tables from a JSON report produced by `make bench`.
readme-tables:
	uv run python benchmarks/render_tables.py bench.json --update README.md

# Documentation site (Sphinx, warnings are errors as in CI). Output: docs/_build/html/index.html
docs:
	uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

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
