# Contributing

## Development setup

Requirements: Python 3.10+, a C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+), and
[uv](https://docs.astral.sh/uv/). CMake and Ninja are fetched automatically if missing.

```bash
uv sync                         # environment + editable build of the extension
uv run pre-commit install       # ruff, clang-format and hygiene hooks on commit
make check                      # lint, format check, mypy + stubtest, tests
```

After editing C++ or CMake files, rebuild with `make build`. Build options go through
`CMAKE_ARGS`, for example `CMAKE_ARGS="-DFASTGOERTZEL_WERROR=ON" make build` to compile with
warnings as errors the way CI does.

Other targets: `make cpp-test` (doctest suite for the header-only core), `make tidy`
(clang-tidy, needs it on PATH), `make bench` / `make latency` (benchmarks),
`make readme-tables` (re-render the README tables from `bench.json`), `make kernel-bench`
(standalone C++ kernel micro-benchmark).

## Layout

| Path | Contents |
|---|---|
| `include/fastgoertzel/goertzel.hpp` | Header-only core: recurrence, finishes, streaming filter. No Python. |
| `cpp/lane_kernels.inl` | Hot loops; compiled twice (`kernels_baseline.cpp`, `kernels_avx2.cpp`). |
| `cpp/kernels.cpp` | Runtime selection of the kernel build (`FASTGOERTZEL_KERNEL` override). |
| `cpp/bindings.cpp` | pybind11 layer: validation, work decomposition, OpenMP, GIL. |
| `src/fastgoertzel/` | Python package, type stubs (`_core.pyi`), `py.typed`. |
| `tests/` | pytest suite; `tests/cpp/` doctest suite. |
| `benchmarks/` | Benchmark harness, table renderer, C++ micro-benchmark. |

Rules that the tests enforce and that pull requests must keep:

- No input may abort the interpreter: validate before releasing the GIL and before any
  OpenMP region; nothing inside a parallel region may throw.
- Results must not depend on the thread count or on which kernel build is selected beyond
  rounding (`tests/test_kernels.py` checks both).
- Anything new in the hot loops must be reachable only through `kernels::Api`, so the two
  instruction-set builds cannot be merged by the linker.
- Every public function has a stub in `_core.pyi`; `stubtest` runs in CI.

## Benchmarks and performance claims

Run `make bench`, which writes `bench.json`, and `make readme-tables` to refresh the README
tables from it. Quote a speed-up only together with its baseline, the thread count, and the
machine; the harness prints all three. `tests/test_performance.py` guards against large
regressions with ratio checks that are machine independent.

## Release checklist

1. Update `CHANGELOG.md`: move the unreleased notes under the new version and date.
2. Bump the version in `pyproject.toml` with bumpver, which also tags:
   `uv run bumpver update --tag final` (dev → release) or `--patch` / `--minor` / `--major`.
3. Push the branch, open a pull request, wait for CI (matrix, sanitizers, C++ tests,
   clang-tidy, wheel smoke) to pass, merge.
4. Push the tag (`git push upstream vX.Y.Z`). The publish workflow builds the sdist and all
   wheels, runs the test suite inside each wheel build and once more under QEMU with a
   pre-AVX CPU, then uploads with PyPI trusted publishing. Pushing the tag yourself, rather
   than creating a GitHub Release first, avoids triggering the workflow twice.
5. Create the GitHub Release from the existing tag with the CHANGELOG section as notes.
