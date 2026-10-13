# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0.dev0] - Unreleased

Work in progress on the 2.0 line. Phase 1 (build system and layout) is complete; the
algorithmic changes (exact non-integer-bin frequencies, streaming filter, `fs` keyword,
faster kernels) follow in later phases and will be listed here as they land.

### Changed
- Build system moved from setuptools to scikit-build-core + CMake. `setup.py` and
  `MANIFEST.in` are gone; compiler flags, OpenMP detection, and warnings-as-errors live in
  `CMakeLists.txt`. Build options are passed via `CMAKE_ARGS` (see README).
  `FASTGOERTZEL_NATIVE=1` becomes `CMAKE_ARGS=-DFASTGOERTZEL_NATIVE=ON`.
- Repository layout: Python package in `src/fastgoertzel/`, header-only C++ core in
  `include/fastgoertzel/goertzel.hpp` (CMake target `fastgoertzel::core`), bindings in
  `cpp/bindings.cpp`, C++ micro-benchmark in `benchmarks/cpp/`.
- The private extension module is renamed from `fastgoertzel._fastgoertzel_core` to
  `fastgoertzel._core`. Import from `fastgoertzel` as before.
- The module declares itself safe for free-threaded CPython (`py::mod_gil_not_used`); it
  holds no mutable global state.
- Warnings raised to `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`
  (`/W4 /permissive-` on MSVC) and treated as errors in CI.

### Added
- `.clang-format`, `.editorconfig`, and a pre-commit configuration (ruff, clang-format,
  whitespace and line-ending hygiene); CI runs the hooks.
- mypy (strict) and `stubtest` in CI, so `_core.pyi` is checked against the built module.

## [1.0.3] - 2026-10-13

Patch release: crash safety, portable wheels, correct metadata. Results for valid inputs
are unchanged (within 1e-12 on x86-64; FMA contraction may change the last bit on arm64).

### Fixed
- `goertzel_sliding_batch` terminated the Python process for any out-of-range frequency
  or `window_size == 0` (exception thrown inside an OpenMP region). All validation now
  happens before the GIL is released and before any parallel region.
- `goertzel_batch` accepted negative, NaN, infinite, or `>= 1` frequencies and returned
  garbage (undefined behaviour in a `size_t` cast). It now raises `ValueError`.
- `goertzel_batch` and `goertzel_sliding_batch` accepted 2-D inputs and empty signals
  silently. They now raise `ValueError`, and `window_size` must be in `[1, len(x)]`.
- The phase assertion in the test suite was one-sided and documented the wrong phase
  convention; phase is cosine-referenced (as `numpy.fft`), and the tests now say so.
- The dead `goertzel_avx2` function, which computed wrong results, and the unused
  `AlignedAllocator` have been removed.
- License metadata said MIT on PyPI while shipping GPLv3 text. The project is GPLv3+
  and the metadata now says so.

### Changed
- Build flags: removed `-march=native`, `-mavx2`, `/arch:AVX2`, `-ffast-math`, `/fp:fast`,
  `/Qpar`, `-funroll-loops`, `-ftree-vectorize`, `-flto`. Published wheels now run on any
  x86-64 or arm64 CPU, and importing the module no longer changes the process's
  floating-point environment. FMA contraction remains enabled. Set `FASTGOERTZEL_NATIVE=1`
  when building from source for your own machine.
- `goertzel` now raises `ValueError` (was `RuntimeError`) for a non-1-D input.
- `goertzel_sliding_batch` now releases the GIL like the other functions.
- OpenMP parallel regions are skipped for small inputs (fewer than 65536
  sample-frequency products) to avoid thread start-up overhead.
- Version is single-sourced from `pyproject.toml`.
- NumPy is no longer a build dependency, and the runtime requirement is relaxed from
  `numpy>=2.2.6` to `numpy>=1.21.2`.
- `docs/pypi.md` removed; `README.md` is the single long description.
- Benchmark script rewritten with named baselines, thread-pinned runs, per-call latency
  percentiles, and JSON output.

### Added
- Source distribution is now published alongside wheels.
- Windows wheels bundle the OpenMP runtime (`vcomp140.dll`) via delvewheel.
- Linux aarch64 wheels, macOS x86_64 wheels.
- CI runs on Linux, macOS, and Windows for Python 3.10 to 3.13, plus an ASan/UBSan job
  and a cibuildwheel smoke job; release wheels are checked for AVX code before upload and
  uploaded with PyPI trusted publishing.
- README roadmap for 2.0 (exact non-integer-bin frequencies, streaming filter API,
  sampling-rate support, faster kernels).

## [1.0.2] - 2025-12-28

Previous release; see git history.
