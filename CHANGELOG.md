# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0.dev0] - Unreleased

Work in progress on the 2.0 line. Phases 1 (build system and layout), 2 (algorithm and
API) and 3 (kernel speed-ups) are complete.

### Performance (Phase 3)
Measured on the reference machine (i9-14900KF, MSVC, AVX2+FMA kernel); see README.
- One frequency: 2.5 billion samples/s single-threaded, up from 0.65 in 1.0.3 (3.9x).
  Signals of 4,096 samples or more run as up to eight independent blocks whose results
  are combined with exact twiddles, which removes the recurrence's latency bound.
- Batch: up to eight frequencies share one pass over the signal; fewer frequencies are
  padded with signal blocks so eight chains always run. N = 1e6, M = 32: 12.6 ms on one
  thread (was 48.8 ms), 1.06 ms on 32 threads (was 2.25 ms).
- Sliding windows: sliding-DFT update with an exact re-synchronisation every 256
  windows; cost per hop no longer depends on the window length. N = 1e5, W = 256, M = 8:
  11.8 ms on one thread (was 362 ms), 2.5 ms on 32 threads (was 17 ms). Step sizes of
  `window_size / 4` or more fall back to exact per-window evaluation.
- Runtime dispatch: the hot loops are compiled for baseline x86-64 and for AVX2+FMA and
  selected at import. Wheels stay portable; `FASTGOERTZEL_KERNEL=baseline|avx2` overrides
  the choice and `fastgoertzel._core._kernel` reports it. The release workflow now runs
  the test suite under QEMU with a pre-AVX CPU model instead of grepping for AVX
  instructions.
- Results are independent of the thread count: the lane decomposition depends only on the
  input sizes, and combination order is fixed.

### Breaking: frequencies are now evaluated exactly
In 1.x every function rounded `frequency * len(x)` *down* to an integer DFT bin, so
`goertzel(x, 0.0099)` on 100 samples silently returned the DC bin, and about 6 % of exact
`k/N` inputs landed on bin `k-1` through floating-point rounding. 2.0 uses the generalized
Goertzel algorithm: the DTFT is evaluated at exactly the frequency you pass.

Migration: results are unchanged whenever `frequency * len(x)` was an exact integer. If you
relied on the old bin behaviour, call `goertzel_bin(x, k)` with the integer bin, which is
identical to `numpy.fft.fft(x)[k]`. Amplitudes at DC and Nyquist are now the true sinusoid
amplitude (1.x reported twice the value there).

### Added
- `fs=` keyword on every function and on `GoertzelFilter`: frequencies in hertz (or any
  unit of `fs`), converted as `frequency / fs`. Addresses the long-standing README TODO.
- `GoertzelFilter(frequency, *, fs=None)`: stateful streaming filter with `push()` (scalars
  or arrays, any chunking), `result()`, `result_complex()`, `reset()`, `n_samples`.
  Addresses the "IIR" README TODO: feed samples as they arrive, read the spectrum any time.
- `goertzel_complex()` and `goertzel_batch_complex()` return the unscaled complex DTFT
  value, skipping the `sqrt`/`atan2` of the amplitude form.
- `goertzel_bin(x, k)`: integer-bin form of the final step, the 1.x computation with an
  explicit bin.
- `scale=` keyword: `'amplitude'` (default, DC/Nyquist aware), `'magnitude'` (`|X|`),
  `'power'` (`|X|**2`, no square root).
- `threads=` keyword on the batch functions: OpenMP thread count for that call.
- `step=` keyword on `goertzel_sliding_batch` for strided windows.
- The final twiddle of the generalized form is reduced modulo one cycle with an
  error-free product, so accuracy does not degrade for long signals or high frequencies.

### Changed
- The recurrence is written as `coeff*s1 + (x - s2)`, which shortens the dependency chain
  and lets the compiler contract to one FMA where available. Single-frequency throughput
  on the reference machine rose from about 650 to about 790 million samples per second.
- Keyword-only arguments after the positional ones; `goertzel(x, f, 8000)` is a
  `TypeError`, write `goertzel(x, f, fs=8000)`.
- Build system moved from setuptools to scikit-build-core + CMake. `setup.py` and
  `MANIFEST.in` are gone; compiler flags, OpenMP detection, and warnings-as-errors live in
  `CMakeLists.txt`. Build options are passed via `CMAKE_ARGS` (see README).
  `FASTGOERTZEL_NATIVE=1` becomes `CMAKE_ARGS=-DFASTGOERTZEL_NATIVE=ON`.
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
