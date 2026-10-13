# fastgoertzel revamp plan

Written 2026-10-12 against `dev` at commit `9e1b78e` (version 1.0.2). Every finding below was reproduced on a local MSVC 19.44 build (Windows 11, Python 3.11.9, NumPy 2.4.6, Intel i9-14900KF). Linux and macOS flags were reviewed by reading `setup.py` and the published PyPI file list; they could not be built here (no WSL, no Docker), so the Linux-specific items say what CI must verify.

The plan is split into a **patch release (1.0.3)** that removes the crash and silent-garbage paths without changing the API, and a **2.0.0** track that fixes the semantics, the build, and the performance story properly. Each step names the files it touches and the test that proves it is done.

---

## 0. What the review found

### Verified correct, no change needed

- The scalar Goertzel kernel in `src/goertzel.cpp` (`GoertzelAlgorithm::compute`) is a correct implementation of the integer-bin Goertzel algorithm. For every bin `k` of a random length-1000 signal it matches `numpy.fft.fft` to 3e-13 in amplitude and 1e-11 in phase.
- The phase convention is the DFT convention (cosine-referenced): `cos(w n + p)` returns `p`. This matches `numpy.fft` and the README example.
- `goertzel_batch` and `goertzel_sliding_batch` produce exactly the same numbers as repeated `goertzel` calls (4e-16 max difference per window).
- `goertzel` releases the GIL and two Python threads get a 2.0x speed-up.
- The README usage example runs and prints matching Goertzel and FFT values.
- pybind11 `forcecast | c_style` correctly accepts lists, integer arrays, float32 arrays, and non-contiguous slices, copying as needed.
- The wheel builds cleanly at `/W4` apart from one macro-redefinition warning.
- OpenMP scaling of `goertzel_batch` is near-linear up to 8 threads (51 ms, 26 ms, 13 ms, 6.9 ms for 1, 2, 4, 8 threads at N = 1e6, M = 32).

### Issues, by severity

**BREAKING** (crashes the interpreter or makes the package uninstallable or unloadable)

1. `goertzel_sliding_batch` terminates the Python process (exit code 0xC0000409, no exception) for any out-of-range frequency or for `window_size = 0`. Cause: `GoertzelAlgorithm::compute` throws `std::invalid_argument` inside the `#pragma omp parallel for` region in `src/python_bindings.cpp:136-144`; an exception escaping an OpenMP region is undefined behaviour and MSVC fail-fasts. Reproduced with `OMP_NUM_THREADS=1` as well.
2. No sdist is published for 1.0.x (PyPI has 21 wheel files and no `.tar.gz` for 1.0.2). `pip install fastgoertzel` fails outright on every platform without a wheel: macOS x86_64, Linux aarch64, PyPy, Python 3.14, and any future Python.
3. Linux wheels are compiled with `-march=native -mavx2` (`setup.py:52-53`). The manylinux wheel is tuned to whichever CPU the GitHub runner happened to have (AMD Zen 3 or Intel Ice Lake with AVX-512), and will raise `SIGILL` on any machine lacking those instructions. The i686 wheel is built with the same flags. Windows wheels are built with `/arch:AVX2` and will crash with an illegal-instruction fault on CPUs without AVX2 (Pentium, Celeron, pre-2013 Core). The scalar kernel does not need AVX2 at all.

**SEVERE** (silently wrong results, or misleading users)

4. `goertzel_batch` performs no validation. Negative, NaN, infinite, or >= 1 frequencies are cast to `size_t` (undefined behaviour) and produce plausible-looking garbage without an error. An empty signal returns NaN. `goertzel` raises `ValueError` for the same inputs, so the two functions disagree.
5. `goertzel_sliding_batch` does no dimension checks. A 2-D signal or 2-D frequency array is silently interpreted via `shape[0]` and returns wrong-shaped, meaningless output. `window_size > len(x) + 1` underflows `size_t` and surfaces as a NumPy "negative dimensions" error instead of a `ValueError`.
6. Frequency is truncated to a bin: `k = static_cast<size_t>(frequency * length)`. The docstring says "normalized frequency (0 to 1)" and never mentions quantization. Consequences measured: `sin(2*pi*0.0099*n)` with N = 100 returns amplitude 0.0013 instead of 1.01 (it evaluated DC); 2750 of the 44100 exact bin frequencies `k/44100` truncate to bin `k-1` because `(k/N)*N` rounds below `k`; the same frequency maps to different bins in `goertzel_sliding_batch` (quantized to `window_size`) and `goertzel_batch` (quantized to `len(x)`).
7. The test suite's phase assertion is one-sided: `assert phase - phase_expected < 1e-10` (`tests/test_goertzel.py:28`). The library returns -pi/4 for a sine with phase +pi/4 (correctly, since phase is cosine-referenced), and the test passes only because -1.57 < 1e-10. The test therefore documents the wrong convention and would not catch a sign error.
8. The dead `goertzel_avx2` function (`src/goertzel.cpp:38-98`) is mathematically wrong. It runs the recurrence on four interleaved sub-sequences with the full-rate coefficient and sums the lane states. Emulated in NumPy it returns amplitude 0.0 for a unit sine. It is never called, so users are not affected today, but it is compiled into every build and is one edit away from being wired in.
9. License metadata is contradictory: `LICENSE` is GPLv3, `setup.py` classifies GPLv3+, `pyproject.toml` says MIT, and PyPI displays MIT while shipping the GPL text in the wheel.
10. `-ffast-math` on GCC links `crtfastmath.o` into the shared object, which sets FTZ/DAZ in MXCSR at load time for the whole Python process (GCC bug 55522, fixed only in GCC 13). The cp310 manylinux2014 wheels are built with GCC 10. Importing fastgoertzel can change the denormal behaviour of NumPy and every other extension in the process.
11. The "~75 times faster than native python" claim is measured against a Python loop that indexes a NumPy array element by element (`test.py`). Against an idiomatic pure-Python loop over a list the measured ratio is 10 to 15x; against a NumPy single-bin DFT it is 5 to 8x; against `np.fft.rfft` picking one bin it is 2.4 to 7.7x. `goertzel_sliding_batch` is 4x slower than NumPy on one thread and only wins because of OpenMP.
12. Published Windows wheels depend on `VCOMP140.DLL`, which is not bundled (cibuildwheel has no `delvewheel` repair step). Machines without the VC++ redistributable get `ImportError: DLL load failed`.

**MINOR**

13. Amplitude is always `2|X|/N`, so DC and Nyquist bins report twice the true amplitude (`ones(100)` at f = 0 returns 2.0). Not documented.
14. Numerical error grows with N near DC/Nyquist: relative amplitude error 2e-5 at N = 1e6, k = 1 and phase error 1e-4 rad at N = 1e7, k = 7, versus 2e-10 at interior bins. Expected for Goertzel (poles on the unit circle near +-1) but not documented or tested.
15. `goertzel_sliding_batch` does not release the GIL, unlike the other two functions.
16. `goertzel` raises `RuntimeError` for a 2-D input and `ValueError` for a bad frequency; `goertzel_batch` raises `RuntimeError` for a scalar frequency. Error types should be consistent (`ValueError` for values, `TypeError` for types).
17. Version is hard-coded in four places (`setup.py`, `pyproject.toml`, `__init__.py`, `python_bindings.cpp`). `setup.py` also passes `-DVERSION_INFO=1.0.2` unquoted, which the `#define` in the `.cpp` then overrides with warning C4005.
18. `setup.py` metadata is dead: with `[project]` in `pyproject.toml`, setuptools ignores `install_requires`, `extras_require`, `long_description` (so `docs/pypi.md` is never used; the wheel README is `README.md`), `classifiers`, and `license` from `setup.py` and warns about each.
19. NumPy is a build requirement and `np.get_include()` is passed to the compiler, but pybind11's `numpy.h` does not use the NumPy C API or headers. The runtime pin `numpy>=2.2.6` excludes every NumPy 1.x user for no technical reason.
20. `include/fastgoertzel/goertzel.hpp` is a byte-identical copy of `src/goertzel.hpp`, and `.gitignore` ignores `include/`, so new files there are silently untracked.
21. `AlignedAllocator` is dead code that does not compile on MinGW (`posix_memalign` is not declared; the call is non-dependent so it is looked up at definition time), and `<cstdlib>` is not included for `free`/`posix_memalign`.
22. The "optimizations" in the kernel do nothing measurable: manual 2x and 4x unrolling of a loop-carried recurrence cannot be reordered, `__builtin_prefetch` of sequential data is redundant with the hardware prefetcher, `/Qpar` and `-ftree-vectorize` cannot vectorise a recurrence, and `-funroll-loops` is implied by `-O3`. Measured throughput (614 MS/s) is consistent with a pure latency-bound scalar loop.
23. CI only builds on Linux with Python 3.11, never exercises the Windows/macOS flag branches, and sets `PYTHONPATH=$GITHUB_WORKSPACE`, which does nothing because the package lives in `python/`. `make install` runs `uv sync --frozen` against a stale lock file (it still records fastgoertzel 0.1.0). Ruff runs only on `tests/` and `benchmarks/`, not on `python/` or `setup.py` (which has two unused imports).
24. `pyproject.toml` `[tool.cibuildwheel]` lists `cp312-*` twice, has no `test-command`, and `publish.yml` overrides it anyway via `CIBW_BUILD` and publishes with a long-lived API token instead of trusted publishing. `macos-latest` is arm64-only, so no x86_64 macOS wheel is produced.
25. `project.license = {text = "MIT"}` is a deprecated TOML table form; setuptools warns it will become an error.

**NIT**

26. `src/goertzel.hpp` uses tabs, `src/goertzel.cpp` uses spaces; `PI` is defined three times (class constant plus two function-local copies); unused includes `<cstring>`, `<memory>`, `<pybind11/stl.h>`; `compute_coefficient` is declared `static inline` in the header but defined out-of-line and never used; `utils::is_valid_*` are never used; the module docstring's autosummary omits `goertzel_sliding_batch`.
27. `.pyi` types `x` as `NDArray[np.float64]` while pybind11 accepts any `ArrayLike`; `__version__` is declared in the stub for the core module but users read it from the package.
28. `test.py` in the repo root is an untracked scratch file with an unused `pandas` import.
29. `Makefile` target `format` only checks formatting; the `install` target builds a wheel and installs it, so tests never run against an in-place build.

---

## 1. Principles for the revamp

1. **Correctness is gated by a reference.** Every kernel, every optimisation, and every build-flag change must pass the same test matrix against `numpy.fft` and a direct DTFT sum before it is merged.
2. **Never crash the interpreter.** All validation happens at the Python boundary, before any parallel region. No exception may be thrown inside an OpenMP region.
3. **Wheels run everywhere they are installed.** Baseline x86-64 (SSE2) / arm64 code only, no `-march=native`, no `/arch:AVX2`, no fast-math. SIMD kernels, if added, are selected at runtime.
4. **One source of truth** for version, README, and license.
5. **Honest performance.** Every published number names its baseline, thread count, and machine, and is regenerated from `benchmarks/benchmark.py` output.
6. **Semantic versioning.** 1.0.3 fixes crashes and metadata without changing results. 2.0.0 changes frequency semantics and is documented as breaking.

---

## 2. Phase 0: patch release 1.0.3 (no API change)

**Status (2026-10-12): implemented on `dev`.** Sections 2.1 to 2.6 are done locally and verified on Windows/MSVC: 73 tests pass, every invalid input raises `ValueError` in a subprocess guard, the wheel metadata reads `GPL-3.0-or-later` and `numpy>=1.21.2`, and the sdist builds. Still to verify in CI because they cannot run on this machine: the Linux/macOS matrix, the ASan/UBSan job, delvewheel bundling of `vcomp140.dll`, the aarch64 QEMU build, the ISA-policy scan, and the trusted-publishing step (which also needs the PyPI publisher configured for this repository and a `pypi` environment in GitHub). `benchmarks/benchmark.py` gained a per-call latency section (`--latency-only`).

Goal: remove every BREAKING item and the silent-garbage paths. Results for valid inputs are bit-identical to 1.0.2.

### 2.1 Validation and crash safety (`src/python_bindings.cpp`)

- Extract a validation-free kernel:
  ```cpp
  struct BinCoeffs { double cos_w, sin_w, coeff; };
  inline BinCoeffs bin_coeffs(std::size_t k, std::size_t n) noexcept;
  inline std::pair<double,double> goertzel_kernel(const double* x, std::size_t n, const BinCoeffs& c) noexcept;
  ```
  and make `GoertzelAlgorithm::compute` call it after validating.
- In `goertzel_batch` and `goertzel_sliding_batch`, before releasing the GIL and before any `#pragma omp`:
  - `x.ndim == 1`, `frequencies.ndim == 1`, `len(x) > 0`;
  - `1 <= window_size <= len(x)` (sliding);
  - every frequency finite and in `[0, 1)`;
  - raise `py::value_error` with the same message text as `goertzel`.
- Precompute `BinCoeffs` per frequency outside the parallel region; the loop body becomes `noexcept`.
- Release the GIL in `goertzel_sliding_batch` (copy frequencies into a `std::vector<double>` first).
- Test (`tests/test_validation.py`): parametrize every invalid input for all three functions, assert `ValueError`; run each in a `subprocess` guard once so a future regression to process-abort shows up as a failed test instead of a dead pytest run.

### 2.2 Remove dead and wrong code

- Delete `goertzel_avx2`, `AlignedAllocator`, `compute_coefficient`, `namespace utils`, `include/fastgoertzel/goertzel.hpp`, and the unused includes.
- Remove `#define VERSION_INFO` from `python_bindings.cpp`; in `setup.py` pass `define_macros=[("VERSION_INFO", f'"{__version__}"')]` so the quoted string reaches the compiler, and keep the `#ifdef` fallback.
- Fix `.gitignore`: drop `include/` and `*.json` (the latter hides benchmark reports and `CppProperties.json`).

### 2.3 Fix the tests that pass for the wrong reason

- `tests/test_goertzel.py::test_pure_sine_wave`: `assert abs(phase - (phase_expected - np.pi/2)) < 1e-10`, and add a cosine case asserting `phase == phase_expected`.
- Add `test_matches_numpy_fft`: for N in `{7, 31, 32, 100, 255, 256, 1000, 1024, 4097}` compare `goertzel_batch(x, k/N for exact k)` against `np.fft.fft` for all bins, amplitude and wrapped phase, tolerance 1e-9.
- Add `test_dc_nyquist_amplitude_documented` recording the current 2x behaviour with `pytest.mark.xfail(strict=True, reason="fixed in 2.0")`.
- Add `test_sliding_matches_batch` (every window) and `test_threads_safe` (ThreadPoolExecutor calling all three functions concurrently and comparing to serial results).

### 2.4 Build flags (`setup.py`)

Replace the three flag blocks with:

| Platform | Keep | Remove | Why |
|---|---|---|---|
| MSVC | `/O2`, `/openmp:experimental` (or `/openmp:llvm`) | `/arch:AVX2`, `/fp:fast`, `/Qpar` | AVX2 is a SIGILL risk and buys nothing for a scalar recurrence; `/fp:fast` changes results; `/Qpar` cannot parallelise a recurrence and conflicts with OpenMP |
| GCC/Clang Linux | `-O3`, `-fopenmp`, `-Wall -Wextra -Wpedantic` | `-march=native`, `-mavx2`, `-ffast-math`, `-funroll-loops`, `-ftree-vectorize`, `-flto` | portability, FTZ/DAZ side effect, redundancy; LTO is already added by `pybind11.setup_helpers` when supported |
| Apple Clang | `-O3`, `-Wall -Wextra -Wpedantic` | `-ffast-math`, `-funroll-loops`, `-ftree-vectorize`, `-flto` | same |

Add `-ffp-contract=fast` (GCC/Clang) and `/fp:contract` (MSVC 17.4+) so the compiler may fuse `coeff * s_prev + x` into an FMA; this is value-safe in the sense required here (it only improves accuracy) and is where the kernel's actual speed comes from. Allow an opt-in `FASTGOERTZEL_NATIVE=1` environment variable that appends `-march=native` for people building from source for their own machine.

OpenMP policy for 1.0.3: keep it on Linux and Windows; on Windows add `delvewheel` as the cibuildwheel repair command so `vcomp140.dll` is bundled. macOS stays single-threaded until 2.0 (document it).

### 2.5 Packaging metadata

- License decision (made 2026-10-12): **GPLv3**. Set `license = "GPL-3.0-or-later"` and `license-files = ["LICENSE"]` in `pyproject.toml`, keep the classifier `License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)`, remove `{text = "MIT"}`, and add the SPDX header `// SPDX-License-Identifier: GPL-3.0-or-later` to every source file. PyPI must show GPLv3 on the next upload.
- Delete the duplicated metadata from `setup.py` (`install_requires`, `extras_require`, `long_description`, `classifiers`, `description`, `author`, `url`); keep only `ext_modules`, `cmdclass`, `packages`, `package_dir`.
- Relax `dependencies` to `numpy>=1.21` and drop numpy from `[build-system] requires` and from `include_dirs`. Verify by building in a clean venv without numpy installed.
- Replace `readme = "README.md"` duplication: delete `docs/pypi.md` and keep a single README.
- Fix `[tool.cibuildwheel]`: remove the duplicate `cp312-*`, add `test-requires = ["pytest", "numpy"]`, `test-command = "pytest {project}/tests -q"`, `[tool.cibuildwheel.windows] repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel}"` with `before-build = "pip install delvewheel"`.
- Run `uv lock` so the lock file matches the project and switch CI to `uv sync --locked`.

### 2.6 Release workflow (`.github/workflows/publish.yml`)

- Trigger on `push: tags: ['v*']` (and keep `release: published`).
- Jobs: `build-sdist` (`python -m build --sdist`, upload artifact), `build-wheels` (one job per OS, `CIBW_BUILD` taken from `pyproject.toml`, `CIBW_ARCHS_MACOS: "x86_64 arm64"`, `CIBW_ARCHS_LINUX: "x86_64 aarch64"` with QEMU), `publish` (`pypa/gh-action-pypi-publish` with `permissions: id-token: write` and a PyPI trusted publisher; delete the API-token secret).
- Add a post-build step on Linux that fails if any wheel contains AVX-512 or AVX2 code when the ISA policy says it must not:
  ```bash
  unzip -o wheelhouse/*manylinux*x86_64.whl '*.so' -d /tmp/w && objdump -d /tmp/w/**/*.so | grep -cE '(zmm|ymm)' && exit 1 || true
  ```

Acceptance for 1.0.3: all tests pass on Linux, macOS, Windows for 3.10 to 3.13; `pip install fastgoertzel==1.0.3` succeeds from the sdist on a platform without a wheel; `python -c "import fastgoertzel"` works on a Windows VM without the VC++ redistributable; no `SIGILL` on a `qemu -cpu Nehalem` run of the Linux wheel's test suite.

---

## 3. Phase 1: build system and repository layout (2.0 track)

### 3.1 Build system

Recommended: **scikit-build-core + CMake**, which is the current standard for pybind11 projects and makes OpenMP, per-compiler flags, and future SIMD dispatch declarative. If staying on setuptools is preferred, apply 2.4 and 2.5 and skip this subsection; everything else in the plan is independent of the choice.

`pyproject.toml`:
```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"

[project]
name = "fastgoertzel"
dynamic = ["version"]
requires-python = ">=3.10"
dependencies = ["numpy>=1.21"]
license = "GPL-3.0-or-later"   # or MIT, see 2.5
license-files = ["LICENSE"]

[tool.scikit-build]
minimum-version = "0.10"
wheel.packages = ["src/fastgoertzel"]
metadata.version.provider = "scikit_build_core.metadata.regex"
metadata.version.input = "src/fastgoertzel/_version.py"
cmake.define.FASTGOERTZEL_OPENMP = "ON"
```

`CMakeLists.txt` (sketch):
```cmake
cmake_minimum_required(VERSION 3.18)
project(fastgoertzel LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FASTGOERTZEL_OPENMP "Build with OpenMP" ON)
option(FASTGOERTZEL_NATIVE "Tune for the build machine (never for wheels)" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core cpp/bindings.cpp)
target_include_directories(_core PRIVATE include)
target_compile_options(_core PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /fp:contract>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -ffp-contract=fast>)
if(FASTGOERTZEL_NATIVE AND NOT MSVC)
  target_compile_options(_core PRIVATE -march=native)
endif()
if(FASTGOERTZEL_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE FASTGOERTZEL_HAS_OPENMP=1)
  endif()
endif()
install(TARGETS _core DESTINATION fastgoertzel)
```
On macOS, `cibuildwheel` `before-all = "brew install libomp"` plus `CMAKE_PREFIX_PATH=$(brew --prefix libomp)` gives OpenMP with Apple Clang; `delocate` bundles `libomp.dylib`.

### 3.2 Layout

```
src/fastgoertzel/__init__.py, _version.py, _core.pyi, py.typed
include/fastgoertzel/goertzel.hpp        # header-only C++ core, usable without Python
cpp/bindings.cpp                         # pybind11 only: argument handling, validation, threading
tests/                                   # pytest
benchmarks/benchmark.py                  # the harness (done)
docs/                                    # mkdocs or sphinx; README is the single front page
CMakeLists.txt, pyproject.toml, LICENSE, README.md, CHANGELOG.md, PLAN.md
```

### 3.3 Tooling

- `.clang-format` (LLVM base, 4-space indent, 100 columns) and `clang-format` in pre-commit; `.editorconfig` to stop the tab/space mix.
- `ruff` over the whole repo (`ruff check .` and `ruff format --check .`), not just `tests/` and `benchmarks/`.
- `mypy` plus `python -m mypy.stubtest fastgoertzel` to keep `_core.pyi` honest.
- `pre-commit` config with ruff, clang-format, end-of-file, trailing-whitespace, `check-toml`.

---

## 4. Phase 2: core algorithm and public API (2.0.0)

### 4.1 C++ core (`include/fastgoertzel/goertzel.hpp`, header-only, `namespace fastgoertzel`)

```cpp
struct Coefficients { double omega, cos_w, sin_w, coeff; };
inline Coefficients coefficients(double omega) noexcept;              // coeff = 2 cos(omega)

struct Complex { double re, im; };

// Runs s[n] = x[n] + coeff*s[n-1] - s[n-2]; returns (s1, s2). No validation, noexcept.
inline std::pair<double,double> recurrence(const double* x, std::size_t n, double coeff) noexcept;

// Integer-bin finish, valid when omega = 2*pi*k/n:  X = e^{j w} s1 - s2
inline Complex finish_bin(double s1, double s2, const Coefficients& c) noexcept;

// Generalized finish, valid for any omega:  X = e^{-j w (n-1)} (s1 - e^{-j w} s2)
inline Complex finish_general(double s1, double s2, std::size_t n, const Coefficients& c) noexcept;

inline Complex dft_bin(const double* x, std::size_t n, std::size_t k) noexcept;
inline Complex dtft(const double* x, std::size_t n, double normalized_frequency) noexcept;
```

Derivation to keep in the header comment: the IIR stage has impulse response `sin((n+1)w)/sin w`, the FIR stage `1 - e^{-jw} z^{-1}` makes the cascade `e^{jwn}`, so `y[N-1] = e^{jw(N-1)} X(w)`. For integer bins `e^{jw(N-1)} = e^{-jw}`, giving the familiar `X = e^{jw} s1 - s2`; for any other `w` the explicit factor `e^{-jw(N-1)}` is required, which is exactly what the current code lacks for non-bin frequencies.

Normalisation helper:
```cpp
// scale: Amplitude -> 2|X|/n for interior bins, |X|/n for DC and Nyquist; Complex -> raw X
inline double amplitude(Complex X, std::size_t n, bool dc_or_nyquist) noexcept;
```

Accuracy: document the measured error growth near DC/Nyquist (2e-5 relative at N = 1e6, k = 1). Offer a `block_size` parameter that splits the signal into B blocks, runs the recurrence per block, and combines with twiddles `e^{-jw*offset_b}`; this bounds the error to O(block_size * eps) and also provides the independent chains needed for 4.3 P2.

### 4.2 Python API (2.0.0)

```python
def goertzel(x, frequency, *, fs=None, scale="amplitude") -> tuple[float, float]
def goertzel_complex(x, frequency, *, fs=None) -> complex
def goertzel_batch(x, frequencies, *, fs=None, scale="amplitude", threads=None) -> NDArray  # (M, 2)
def goertzel_sliding_batch(x, window, frequencies, *, step=1, fs=None, scale="amplitude", threads=None) -> NDArray  # (W, M, 2)
def goertzel_bin(x, k) -> tuple[float, float]          # explicit integer bin, the 1.x behaviour

class GoertzelFilter:
    def __init__(self, frequency, *, fs=None): ...
    def push(self, x) -> None                          # accepts scalars or arrays, keeps s1, s2, n
    def result(self, scale="amplitude") -> tuple[float, float]
    def reset(self) -> None
    n_samples: int
```

Semantics, all documented in the docstrings and README:

- `frequency` is evaluated exactly (generalized Goertzel). No truncation, no rounding. `frequency * fs` must be finite and `0 <= frequency < 1` after normalisation; values above 0.5 alias and are allowed but documented.
- `fs` converts Hz to cycles/sample: `f_norm = frequency / fs`. It is accepted by every function and by `GoertzelFilter`. This is the README "sampling rate" TODO; it is only correct once the generalized Goertzel exists, which is why it is sequenced here.
- `goertzel_bin` keeps the 1.x bin behaviour for users who want DFT-exact bins; `goertzel(x, k/N)` and `goertzel_bin(x, k)` agree to 1e-12 (test).
- `scale="amplitude"` applies the DC/Nyquist-aware normalisation; `scale="complex"` returns `X` unscaled (for `goertzel_batch` the array dtype becomes `complex128`, shape `(M,)`).
- Validation at the boundary: `ValueError` for ranges and dimensions, `TypeError` for types (pybind11's default). `RuntimeError` is never raised for user input.
- Every function releases the GIL. `threads=None` means the OpenMP default; `threads=1` forces serial; the parallel region is entered only when `n_samples * n_frequencies > 65536` to avoid thread start-up costs on small inputs.
- Migration note for 1.x users: results change for any `frequency` that was not an exact bin. Provide `goertzel_bin` and an entry in `CHANGELOG.md` with the exact rule.

### 4.3 Performance work (each item gated by the test matrix and by a benchmark JSON diff)

Measured 2026-10-12 with a standalone C++ harness (MSVC 19.44, /O2, N = 2^20 random samples, one core, min of 30 runs; relative error against a direct DFT sum). "AVX2 build" means `/arch:AVX2` so `std::fma` becomes a `vfmadd` instruction; "baseline" is plain x64.

| Kernel formulation | AVX2 build | baseline x64 | rel. error |
|---|---:|---:|---:|
| A: shipped `s0 = x + coeff*s1 - s2` | 634 MS/s | 666 MS/s | 9.4e-11 |
| B: `t = x - s2; s0 = fma(coeff, s1, t)` (one FMA on the chain) | 1140 MS/s (1.8x) | 1200 MS/s (1.8x) | 9.3e-11 |
| C: 4 independent contiguous blocks + twiddle combine | 4539 MS/s (7.2x) | 2665 MS/s (4.0x) | 3.6e-11 |
| C: 8 blocks | 5243 MS/s (8.3x) | 2616 MS/s (3.9x) | 3.0e-11 |
| D: 8 frequencies per pass (SoA state) vs 8 passes of B | 1.35 ms vs 7.0 ms (5.2x) | 3.77 ms vs 7.0 ms (1.9x) | identical to 8xB |

The shipped loop compiles to `vmulsd, vaddsd, vsubsd` on the critical path (verified in the disassembly of the 1.0.2 wheel; MSVC does not contract it to an FMA even with `/fp:fast`). No intrinsics are involved in any row; the harness is `kernel_bench.cpp` in the review scratchpad and should be moved to `benchmarks/cpp/` under Phase 1.

- **P1, batch: multiple frequencies per pass.** Keep `M_group = 8` frequency states in SoA arrays and update all of them per sample; the inner loop over the group has no loop-carried dependency and vectorises. The signal is read once per group instead of once per frequency. Measured 5.2x on one core (row D above); OpenMP then parallelises over groups, and the combination with P2 (blocks x frequencies) is the full 2-D tiling.
- **P2, single frequency: break the dependency chain.** The measured 614 MS/s (about 9 cycles per sample) is the latency of one serial `mul, add, sub` chain. Step one is formulation B (1.8x, a two-line change). Step two is formulation C: split the signal into 4 to 8 contiguous blocks, run independent recurrences, finish each block with `finish_general`, and combine `X = sum_b e^{-jw*offset_b} X_b`. Measured 7 to 8x with FMA, 4x without, and the error *drops* because each chain is shorter. 8 blocks at 5.2 GS/s is 42 GB/s of input, which is L3-bandwidth territory on the test machine; beyond that, only reading the data fewer times (P1) helps. The polyphase form the dead AVX2 code was reaching for (lanes carry `x[4n+j]`, `coeff' = 2cos(4w)`, twiddles `e^{-jwj}`) is an equivalent alternative. If intrinsics are ever used they must sit behind runtime dispatch (`__builtin_cpu_supports("avx2")`, `IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)`); the FMA-vs-baseline column shows why a separate `-mfma` translation unit with dispatch is worth having.
- **P3, sliding: O(1) per step.** The current implementation recomputes each window from scratch, O(W * M) per step, and is 4x slower than NumPy on one thread. Use the sliding DFT update `X_{w+1} = e^{jw} (X_w - x[w] + x[w+W])` per frequency, and re-synchronise with a full Goertzel every R windows (R = 1024 by default, parameter exposed) to bound drift. Expected for N = 1e5, W = 256, M = 8: from 362 ms single-threaded to a few ms. Add `step` so users can request every k-th window. Test: agreement with the brute-force implementation to 1e-9 relative over 1e5 windows including a `float32`-magnitude signal with a large DC offset (worst case for drift).
- **P4, cleanup.** Remove manual unrolling and `__builtin_prefetch`; keep `-O3` and FMA contraction. Confirm with the benchmark that nothing regresses.

Ceiling to document: one serial recurrence is one FMA plus one subtract on the critical path, so about 5 cycles per sample at best; with P2 the kernel becomes throughput-bound and the 8-byte-per-sample memory stream (about 5 GB/s at 614 MS/s) is still far from DRAM bandwidth, so there is real headroom.

---

## 5. Phase 3: testing and quality

Test matrix (`tests/`):

| File | What it proves |
|---|---|
| `test_reference.py` | all three functions vs `np.fft.fft` for every bin, N in `{1, 2, 3, 7, 31, 32, 100, 255, 256, 1000, 1024, 4097, 65536}`; generalized vs direct DTFT sum for 200 random non-bin frequencies; `goertzel_bin` vs `goertzel(k/N)` |
| `test_conventions.py` | cosine/sine phase, DC and Nyquist amplitude, aliasing above 0.5, `fs` round trip (440 Hz at 44100) |
| `test_validation.py` | every invalid input raises the right exception type; subprocess guard proves no process abort |
| `test_inputs.py` | lists, ints, float32, complex (warning), non-contiguous, 0-d and 2-d rejection, huge N smoke |
| `test_threads.py` | `ThreadPoolExecutor` concurrency equals serial; `threads=1` equals default |
| `test_accuracy.py` | error bounds at large N near DC/Nyquist (documented tolerances), block-reset improves them |
| `test_streaming.py` | `GoertzelFilter.push` in chunks equals one-shot result |
| `test_sliding.py` | O(1) sliding equals brute force, including drift stress and `step > 1` |
| `test_stubs.py` | `mypy.stubtest` passes |

CI quality gates:
- Extension compiled with warnings as errors (`/W4 /WX`, `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`).
- A Linux job building with `-fsanitize=address,undefined -fno-omit-frame-pointer` and running pytest under `LD_PRELOAD=$(gcc -print-file-name=libasan.so)`; this would have caught items 1, 4, and 5.
- `clang-tidy` with `bugprone-*`, `performance-*`, `modernize-*` on `cpp/` and `include/`.
- Optional: Hypothesis property tests (`frequency` in `[0, 1)`, N up to 4096) against the DTFT sum.

---

## 6. Phase 4: benchmarking methodology and truthful claims

`benchmarks/benchmark.py` has been replaced with a harness that implements the following rules; keep them when editing it.

1. **Name the baseline.** Every ratio is "fastgoertzel vs X" where X is one of: idiomatic pure-Python loop over a list (not a NumPy array indexed per element), NumPy two-real-dot single-bin DFT, `np.fft.rfft` one bin, `np.fft.rfft` all bins (for batch), `sliding_window_view` + `rfft` (for sliding). Add `scipy.signal` references if SciPy is installed.
2. **Pin and report threads.** The harness records `OMP_NUM_THREADS`, runs a thread sweep in subprocesses (`--threads`), and the README must never show a multi-threaded fastgoertzel number against a single-threaded baseline without saying so.
3. **Minimum of repeats, with median shown.** `timeit.autorange` picks the inner count; the minimum of 7 rounds is the headline, the median is kept in the JSON to expose noise.
4. **Record the machine.** CPU brand and flags, core count, OS, Python, NumPy, compiler, fastgoertzel version go into the JSON and the README table caption.
5. **Regenerate, don't hand-edit.** The README performance table is produced from the JSON (`python benchmarks/benchmark.py --json bench.json`, then a small `benchmarks/render_readme_table.py` to be written) on each release and states the date and commit.
6. **Regression guard in CI**, not absolute numbers: `pytest-benchmark --benchmark-autosave --benchmark-compare --benchmark-compare-fail=min:30%` on a fixed runner, or `asv`. Shared CI runners are too noisy for absolute claims; they are good enough to catch a 2x regression.

Measured on 2026-10-12 (i9-14900KF, 32 threads, Windows 11, MSVC 19.44, Python 3.11.9, NumPy 2.4.6, 1.0.2):

| Function, case | fastgoertzel | Baseline | Ratio |
|---|---:|---|---:|
| `goertzel`, N = 1e6 | 1.63 ms | pure-Python list loop 24 ms | 15x |
| `goertzel`, N = 1e6 | 1.63 ms | NumPy 2 real dots 9.4 ms | 5.8x |
| `goertzel`, N = 1e6 | 1.63 ms | `np.fft.rfft` one bin 8.6 ms | 5.3x |
| `goertzel`, N = 1e5 | 163 us | "native python" loop over ndarray 10.9 ms | 67x (the source of the old 75x claim) |
| `goertzel_batch`, N = 1e6, M = 32, 32 threads | 2.3 ms | same on 1 thread 51 ms | 22x (pure OpenMP) |
| `goertzel_batch`, N = 1e6, M = 64, 32 threads | 4.7 ms | `np.fft.rfft` all bins 9.5 ms | 2.0x |
| `goertzel_sliding_batch`, N = 1e5, W = 256, M = 8, 32 threads | 16.8 ms | NumPy windows + rfft 87 ms | 5.2x |
| `goertzel_sliding_batch`, same, 1 thread | 362 ms | NumPy windows + rfft 87 ms | 0.24x |

The honest one-line claim for the README is: "10 to 15x faster than a pure-Python loop and 5 to 8x faster than a vectorised NumPy single-bin DFT for one frequency; batch and sliding functions scale with core count." The README has been updated to say this.

---

## 7. Phase 5: CI/CD

`ci.yml`:
- Matrix: `ubuntu-latest`, `windows-latest`, `macos-13` (x86_64), `macos-latest` (arm64) x Python 3.10, 3.11, 3.12, 3.13, 3.14.
- Steps: `uv sync --locked --group dev`, `uv pip install -e . --no-build-isolation -Ccmake.define.CMAKE_BUILD_TYPE=Release` (or `pip install -e .[test]`), `ruff check .`, `ruff format --check .`, `mypy` + `stubtest`, `pytest -q`, `python benchmarks/benchmark.py --quick --no-python` (smoke only).
- Separate jobs: sanitizers (Linux, GCC), warnings-as-errors is implied by the CMake flags, `cibuildwheel` for one target per OS on pull requests so wheel breakage is caught before a tag.
- Remove the `PYTHONPATH` step and the `make install` indirection.

`publish.yml`: as in 2.6, plus `CIBW_ARCHS_LINUX: "x86_64 aarch64"` (QEMU via `docker/setup-qemu-action`), `CIBW_TEST_SKIP: "*-macosx_arm64"` only where cross-testing is impossible, and the ISA policy scan. Consider `cp312-abi3` wheels (pybind11 >= 2.13 supports the stable ABI on 3.12+) to shrink the matrix to cp310, cp311, cp312-abi3.

Release process (`CONTRIBUTING.md`): bump `_version.py` (single file, `bumpver` config reduced to it), update `CHANGELOG.md`, regenerate the README performance table, tag `vX.Y.Z`, CI publishes. Keep `MANIFEST.in` out of it; scikit-build-core uses git-tracked files for the sdist.

---

## 8. Phase 6: documentation

- README: what the library computes (DTFT at a frequency, DFT bin via `goertzel_bin`), conventions (phase, amplitude scaling, aliasing, `fs`), when Goertzel beats an FFT (few frequencies, streaming, sliding), performance table generated from the benchmark, platform notes (OpenMP, `threads`, `OMP_NUM_THREADS`), build-from-source notes (`FASTGOERTZEL_NATIVE`, `FASTGOERTZEL_OPENMP`).
- API reference from docstrings (mkdocs-material + mkdocstrings, or Sphinx + numpydoc). The pybind11 module docstring's autosummary must list all functions and the class.
- `CHANGELOG.md` (Keep a Changelog) starting with 1.0.3 and a 2.0.0 "Breaking" section containing the frequency-semantics change and migration rule.
- Delete `docs/pypi.md` (PyPI renders `README.md`).

---

## 9. The two README TODOs

**"Add IIR and k-th FFT implementation of Goertzel."** The Goertzel algorithm *is* a second-order IIR filter whose output after N samples is the k-th DFT coefficient, and that is what `goertzel` already computes; the README now marks this item as done with that explanation. What the item was presumably reaching for is split into two concrete deliverables in this plan: the generalized (non-integer bin) Goertzel in 4.1 and 4.2, and the stateful `GoertzelFilter` streaming API in 4.2, which is the "IIR" usage pattern (feed samples as they arrive, read the result at any time, reset).

**"Add support for sampling rate."** Specified in 4.2 as the `fs` keyword on every function and on `GoertzelFilter`: `f_norm = frequency / fs`. It is intentionally not implemented on the 1.x code because, with bin truncation, `goertzel(x, 440, fs=44100)` on 4096 samples would evaluate bin 40 (430.7 Hz) and report it as 440 Hz, which would make the sampling-rate feature actively misleading. It lands in 2.0.0 together with the generalized kernel, with the round-trip test in `test_conventions.py`.

---

## 10. Milestones and acceptance

| Milestone | Contents | Done when |
|---|---|---|
| 1.0.3 | Sections 2.1 to 2.6 | No input can abort the interpreter; sdist on PyPI; wheels run on a non-AVX2 CPU; license metadata consistent; Windows wheel imports without VC++ redistributable; all new tests green on 3 OSes |
| 2.0.0-alpha | Sections 3, 4.1, 4.2 (without P1 to P3) | New API passes the Section 5 matrix; `goertzel(x, k/N) == goertzel_bin(x, k)`; `fs` round trip exact; stubtest clean; sanitizer job clean |
| 2.0.0 | 4.3 P1 to P4, Section 6 regression guard, Section 7 and 8 | Benchmark JSON shows no regression versus 1.0.3 for any case and the expected gains for batch and sliding; README table regenerated from JSON; CHANGELOG migration notes |
| 2.1 | SIMD kernels behind runtime dispatch (optional) | Bit-for-bit agreement with the scalar path within 1e-12 on the matrix; ISA scan in CI proves baseline code outside dispatched functions |
