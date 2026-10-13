# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0.dev0] - Unreleased

The 2.0 line: frequencies are evaluated exactly, the sample rate and a streaming filter
are built in, the kernels are three to thirty times faster, and the build system, test
suite, and documentation are new. See the
[migration guide](https://0zean.github.io/fastgoertzel/migration.html) for upgrading.

### Breaking
- **Frequencies are evaluated exactly.** In 1.x every function rounded
  `frequency * len(x)` *down* to an integer DFT bin, so `goertzel(x, 0.0099)` on 100
  samples silently returned the DC bin, and about 6 % of exact `k/N` inputs landed on
  bin `k-1` through floating-point rounding. 2.0 uses the generalized Goertzel
  algorithm: the DTFT is evaluated at exactly the frequency you pass. Results are
  unchanged whenever `frequency * len(x)` was an exact integer. If you relied on the old
  bin behaviour, call `goertzel_bin(x, k)` with the integer bin, which is identical to
  `numpy.fft.fft(x)[k]`.
- Amplitudes at DC and Nyquist are now the true sinusoid amplitude (`|X|/N`); 1.x
  reported twice that value there.
- Arguments after the positional ones are keyword-only: `goertzel(x, f, 8000)` is a
  `TypeError`, write `goertzel(x, f, fs=8000)`.
- The private extension module is renamed from `fastgoertzel._fastgoertzel_core` to
  `fastgoertzel._core`. Import from `fastgoertzel` as before.
- Build system moved from setuptools to scikit-build-core + CMake. `setup.py` and
  `MANIFEST.in` are gone; compiler flags, OpenMP detection, and warnings-as-errors live in
  `CMakeLists.txt`. Build options are passed via `CMAKE_ARGS` (see README):
  `FASTGOERTZEL_NATIVE=1` becomes `CMAKE_ARGS=-DFASTGOERTZEL_NATIVE=ON`.

### Added
- `fs=` keyword on every function and on `GoertzelFilter`: frequencies in hertz (or any
  unit of `fs`), converted as `frequency / fs`. Closes the long-standing README TODO.
- `GoertzelFilter(frequency, *, fs=None)`: stateful streaming filter with `push()`
  (scalars or arrays, any chunking), `result()`, `result_complex()`, `reset()`,
  `n_samples`. Closes the "IIR" README TODO: feed samples as they arrive, read the
  spectrum at any time. Chunks of 4,096 samples or more go through the block-parallel
  kernel and are folded in with the shift theorem.
- `goertzel_complex()`, `goertzel_batch_complex()` and `goertzel_sliding_batch_complex()`
  return the unscaled complex DTFT value, skipping the `sqrt`/`atan2` of the amplitude
  form (which dominates the per-window cost of the sliding form).
- `goertzel_bin(x, k)`: integer-bin form of the final step, the 1.x computation with an
  explicit bin and exact integer twiddles.
- `scale=` keyword: `'amplitude'` (default, DC/Nyquist aware), `'magnitude'` (`|X|`),
  `'power'` (`|X|**2`, no square root).
- `threads=` keyword on the batch and sliding functions: OpenMP thread count for that
  call; `threads=1` forces serial execution.
- `step=` keyword on the sliding functions for strided windows.
- Type stubs (`_core.pyi`, `py.typed`), checked against the compiled module by stubtest
  in CI.
- Header-only C++ core, `include/fastgoertzel/goertzel.hpp`, usable without Python
  (CMake target `fastgoertzel::core`), with its own doctest suite.
- Documentation site at <https://0zean.github.io/fastgoertzel/>: usage guide,
  conventions, API reference generated from the docstrings, performance methodology,
  migration guide. The docstring examples and the README code blocks run in the test
  suite.
- Python 3.14 in CI and in the wheel matrix. The module declares itself safe for
  free-threaded CPython (`py::mod_gil_not_used`); it holds no mutable global state.
- macOS x86_64 wheels are now built and tested on an Intel runner instead of being
  cross-built untested.

### Performance
Measured on the reference machine (i9-14900KF, MSVC, AVX2+FMA kernel); see the README
tables, which are rendered from one recorded benchmark run.
- One frequency: 2.5 billion samples/s single-threaded, up from 0.65 in 1.0.3 (3.9x).
  Signals of 4,096 samples or more run as up to eight independent blocks whose results
  are combined with exact twiddles, which removes the recurrence's latency bound. The
  recurrence itself is written as `coeff*s1 + (x - s2)`, which contracts to one FMA.
- Batch: up to eight frequencies share one pass over the signal; fewer frequencies are
  padded with signal blocks so eight chains always run. N = 1e6, M = 32: 12.6 ms on one
  thread (was 48.8 ms), about 1 ms on 32 threads (was 2.25 ms). On one thread a batch
  costs the same per frequency as single calls; its gains are OpenMP and fewer memory
  passes for out-of-cache signals.
- Sliding windows: sliding-DFT update with an exact re-synchronisation every 256
  windows; cost per hop no longer depends on the window length. N = 1e5, W = 256, M = 8:
  11.8 ms on one thread (was 362 ms), 2.5 ms on 32 threads (was 17 ms). Step sizes of
  `window_size / 4` or more fall back to exact per-window evaluation.
- Runtime dispatch: the hot loops are compiled for baseline x86-64 and for AVX2+FMA and
  selected at import. Wheels stay portable; `FASTGOERTZEL_KERNEL=baseline|avx2` overrides
  the choice and `fastgoertzel._core._kernel` reports it. The release workflow runs the
  test suite under QEMU with a pre-AVX CPU model to prove the baseline path.
- Results are independent of the thread count and of the kernel build beyond rounding:
  the work decomposition depends only on the input sizes and the combination order is
  fixed.
- The final twiddle of the generalized form is reduced modulo one cycle with an
  error-free product, so accuracy does not degrade for long signals or high frequencies.

### Changed
- Repository layout: Python package in `src/fastgoertzel/`, header-only C++ core in
  `include/fastgoertzel/`, bindings and kernels in `cpp/`, C++ micro-benchmark in
  `benchmarks/cpp/`, Sphinx sources in `docs/`.
- Warnings raised to `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`
  (`/W4 /permissive-` on MSVC) and treated as errors in CI.

### Development
- Test suite: reference tests against `numpy.fft` and direct DTFT sums, Hypothesis
  property tests, subprocess crash guards, thread-count and kernel-build invariance,
  streaming and sliding equivalence, machine-independent performance ratio guards,
  docstring and README example tests, and a C++ doctest suite.
- CI: 3 OS x Python 3.10 to 3.14 plus macOS x86_64, each also run with the baseline
  kernel forced; ASan/UBSan; C++ tests on 3 OSes; clang-tidy; cibuildwheel smoke builds;
  Sphinx build with warnings as errors; pre-commit (ruff, clang-format, hygiene);
  mypy strict and stubtest.
- Tooling: `.clang-format`, `.clang-tidy`, `.editorconfig`, pre-commit configuration,
  `Makefile` shortcuts, `CONTRIBUTING.md` with the invariants the tests enforce and the
  release checklist, `benchmarks/render_tables.py` to render the README tables from a
  benchmark JSON report.

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
