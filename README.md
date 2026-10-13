![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/dark%20logo.png#gh-light-mode-only)
![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/light%20logo.png#gh-dark-mode-only)

<!-- start here -->

fastgoertzel ![GitHub Actions](https://github.com/0zean/fastgoertzel/actions/workflows/ci.yml/badge.svg) [![PyPI](https://img.shields.io/pypi/v/fastgoertzel.svg)](https://pypi.org/project/fastgoertzel/) [![Docs](https://img.shields.io/badge/docs-0zean.github.io-blue.svg)](https://0zean.github.io/fastgoertzel/)
============

The Goertzel algorithm in C++ with Python bindings: evaluate the spectrum of a real signal at the frequencies you care about, exactly, without computing a full FFT. One frequency, many frequencies, sliding windows, or a streaming filter fed sample by sample.

Documentation, with a usage guide, the API reference, and the migration guide from 1.x: <https://0zean.github.io/fastgoertzel/>.

## API

| Call | Returns |
|---|---|
| `goertzel(x, frequency, *, fs=None, scale="amplitude")` | `(value, phase)` at exactly `frequency` |
| `goertzel_complex(x, frequency, *, fs=None)` | complex `X = sum x[n] exp(-2j*pi*f*n)` |
| `goertzel_bin(x, k, *, scale=...)` | `(value, phase)` of DFT bin `k`, same as `np.fft.fft(x)[k]` |
| `goertzel_batch(x, frequencies, *, fs=None, scale=..., threads=None)` | array `(M, 2)` of `(value, phase)` |
| `goertzel_batch_complex(x, frequencies, *, fs=None, threads=None)` | complex array `(M,)` |
| `goertzel_sliding_batch(x, window_size, frequencies, *, step=1, fs=None, scale=..., threads=None)` | array `(n_windows, M, 2)` |
| `goertzel_sliding_batch_complex(x, window_size, frequencies, *, step=1, fs=None, threads=None)` | complex array `(n_windows, M)` |
| `GoertzelFilter(frequency, *, fs=None)` | object with `push(x)`, `result(scale=...)`, `result_complex()`, `reset()`, `n_samples` |

Conventions, the same for every call:

- **Frequency** is in cycles per sample, `0 <= f < 1`, or in the unit of `fs` when `fs` is given (`f = frequency / fs`). It is evaluated exactly; nothing is rounded to a DFT bin. Frequencies above half the sample rate alias onto their mirror image. If you want integer-bin behaviour, use `goertzel_bin`.
- **Phase** is cosine-referenced, as in `numpy.fft`: `cos(2*pi*f*n + p)` reports `p`; a sine reports `p - pi/2`.
- **Scale.** `"amplitude"` (default) is `2|X|/N`, the amplitude of a sinusoid at that frequency, with `|X|/N` at DC and Nyquist where the two spectral lines coincide. `"magnitude"` is `|X|` and `"power"` is `|X|**2` (no square root), both unnormalized. The `*_complex` functions return `X` itself.
- **Validation.** Every call raises `ValueError` for an empty or non-1-D signal, a frequency outside its range, a non-positive `fs`, a `window_size` outside `[1, len(x)]`, or `step`/`threads` below 1. Nothing you pass can crash the interpreter.
- **Threads.** `goertzel_batch` and `goertzel_sliding_batch` use OpenMP when the build has it (Linux and Windows wheels; macOS wheels are single-threaded) and the work is at least 65,536 sample-frequency products. `threads=` overrides `OMP_NUM_THREADS` per call; `threads=1` forces serial execution, which gives the most predictable latency. The GIL is released in every call.
- **Accuracy.** Rounding error of the recurrence grows roughly linearly with the signal length and is largest near DC and Nyquist. Measured against a direct DTFT sum: about 2e-10 relative at N = 1e6 for interior frequencies, about 2e-5 for the first bin. Use float64 input; float32 is converted but offers no speed advantage.

Upgrading from 1.x: results are unchanged whenever `frequency * len(x)` was an exact integer. 1.x rounded that product *down* to a bin and reported twice the true amplitude at DC and Nyquist; see the [migration guide](https://0zean.github.io/fastgoertzel/migration.html).

## Installation

From PyPI (wheels for Linux, macOS, and Windows, plus a source distribution):
```bash
pip install fastgoertzel
```

From source (needs a C++17 compiler; CMake and Ninja are fetched automatically if missing):
```bash
git clone https://github.com/0zean/fastgoertzel.git
cd fastgoertzel
pip install .
```

Published wheels are compiled for baseline x86-64 and arm64 so that they run on any CPU. The build is driven by `CMakeLists.txt` through scikit-build-core, and its options are passed with the `CMAKE_ARGS` environment variable:

| Option | Default | Effect |
|---|---|---|
| `-DFASTGOERTZEL_NATIVE=ON` | OFF | Tune for the build machine (`-march=native`). Never used for wheels. |
| `-DFASTGOERTZEL_OPENMP=OFF` | ON | Single-threaded build even when OpenMP is available. |
| `-DOpenMP_ROOT=$(brew --prefix libomp)` | unset | Enable OpenMP on macOS after `brew install libomp`; Apple's compiler has none, so macOS wheels are single-threaded. |
| `-DFASTGOERTZEL_WERROR=ON` | OFF | Warnings as errors (CI uses this). |

For example `CMAKE_ARGS="-DFASTGOERTZEL_NATIVE=ON" pip install .`.

The C++ core is header-only (`include/fastgoertzel/goertzel.hpp`) and can be used without Python; the CMake target is `fastgoertzel::core`.

For development, `uv sync` creates the environment and builds the extension; `make check` runs lint, format check, type check, and tests, and `uv run pre-commit install` enables the ruff and clang-format hooks. `make docs` builds the documentation site and `make kernel-bench` the standalone C++ kernel micro-benchmark. See `CONTRIBUTING.md`.

## Usage

```python
import numpy as np
import fastgoertzel as fg

fs = 44100.0
t = np.arange(4096)
x = 0.8 * np.cos(2 * np.pi * 440 / fs * t + 0.3) + 0.1 * np.random.default_rng(0).standard_normal(t.size)

# One frequency, in hertz. 440 Hz is not a DFT bin of a 4096-sample window; it is evaluated exactly.
amp, phase = fg.goertzel(x, 440, fs=fs)
print(f"440 Hz: amplitude {amp:.3f}, phase {phase:.3f}")        # ~0.8, ~0.3

# Several frequencies at once (OpenMP-parallel for large inputs); power avoids the square root.
res = fg.goertzel_batch(x, [430, 440, 450], fs=fs, scale="power")
print(res[:, 0])                                                 # power at each frequency

# Sliding windows: 1024-sample windows every 256 samples, two frequencies.
win = fg.goertzel_sliding_batch(x, 1024, [440, 880], step=256, fs=fs)
print(win.shape)                                                 # (13, 2, 2)

# Streaming: feed chunks as they arrive, read the spectrum whenever you like.
flt = fg.GoertzelFilter(440, fs=fs)
for chunk in np.array_split(x, 16):
    flt.push(chunk)
    amp, phase = flt.result()                                    # over everything pushed so far
print(flt.n_samples, round(amp, 3))

# Integer DFT bin, identical to numpy.fft.fft(x)[k].
amp_k, phase_k = fg.goertzel_bin(x, 41)
X = np.fft.fft(x)[41]
assert np.isclose(amp_k, abs(X) * 2 / x.size) and np.isclose(phase_k, np.angle(X))
```

When to use it instead of an FFT: the Goertzel algorithm costs about `2 * N` flops per frequency against roughly `2.5 * N * log2(N)` for a full FFT, so it wins when you need fewer than about `log2(N)` frequencies, when the frequencies are not DFT bins, when samples arrive as a stream, or when you want the result for a window without allocating a spectrum. For a dense spectrum, use `numpy.fft`.

## Performance

The tables below are generated from `benchmarks/benchmark.py --json` by `benchmarks/render_tables.py` (`make bench` then `make readme-tables`), so they always describe one real run on one named machine. Numbers on other machines will differ; re-run the script and quote your own together with the baseline and thread count. How to read them:

- **One frequency** is single-threaded by design. The Goertzel recurrence is a dependency chain that costs a few cycles per sample however the loop is written; signals of 4,096 samples or more are therefore split into up to eight independent blocks whose results are combined exactly, which keeps the FMA units busy. The hot loops are compiled for baseline x86-64 and for AVX2+FMA and the build is chosen at import, so the wheel still runs on any CPU (`FASTGOERTZEL_KERNEL=baseline` forces the portable build; `fastgoertzel._core._kernel` reports the choice).
- **Several frequencies.** On one thread a batch costs the same per frequency as a single call: both run eight chains and are compute-bound once the signal is in cache. What the batch adds is OpenMP over lane groups, and one pass over the data per eight frequencies when the signal is larger than cache. Against `np.fft.rfft`, which computes every bin at once, the single-threaded crossover is between 8 and 32 frequencies at N = 1e6; Goertzel is the tool for few frequencies, arbitrary (non-bin) frequencies, streaming, and sliding windows.
- **Sliding windows** use the sliding-DFT update (constant cost per hop, re-synchronised exactly every 256 windows), so the window length no longer matters. Version 1.0.x recomputed every window and took 362 ms single-threaded on the N = 100,000, W = 256, M = 8 row; what remains per window in the (value, phase) form is mostly `sqrt` and `atan2`, which `goertzel_sliding_batch_complex` skips.
- **Per-call latency** is what a real-time loop cares about; Python call overhead is included. Calls that enter an OpenMP region carry a heavier tail: pass `threads=1` or keep the work below the 65,536 sample-frequency threshold for predictable latency on small inputs.

<!-- benchmark-tables:start -->
Measured with `python benchmarks/benchmark.py` on Intel(R) Core(TM) i9-14900KF (32 logical cores), Windows-10-10.0.26300-SP0, Python 3.11.9, NumPy 2.3.4, fastgoertzel 2.0.0.dev0, avx2 kernel. Each time is the minimum of the timing rounds; ratios are baseline time over fastgoertzel time.

**One frequency**, `goertzel(x, f)`, single-threaded:

| N | fastgoertzel | pure-Python loop | NumPy 2x real dot | `np.fft.rfft`, one bin |
|---:|---:|---:|---:|---:|
| 100 | 0.6 us | 2.9 us (5.2x) | 3.1 us (5.5x) | 4.0 us (7.2x) |
| 1,000 | 1.7 us | 22.7 us (13.6x) | 10.0 us (6.0x) | 7.0 us (4.2x) |
| 10,000 | 4.3 us | 230.5 us (53.1x) | 76.5 us (17.6x) | 41.0 us (9.4x) |
| 100,000 | 38.4 us | 2.40 ms (62.5x) | 932.6 us (24.3x) | 451.8 us (11.8x) |
| 1,000,000 | 388.4 us | 24.45 ms (62.9x) | 10.55 ms (27.2x) | 8.53 ms (21.9x) |
| 10,000,000 | 4.32 ms | not run | 119.00 ms (27.5x) | 152.66 ms (35.3x) |

**Several frequencies**, `goertzel_batch(x, freqs)` (OpenMP default threads):

| N | M | fastgoertzel | M x `goertzel` | `np.fft.rfft` (all bins) | M x NumPy dot |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 8 | 4.2 us | 14.0 us (3.4x) | 6.1 us (1.5x) | 91.3 us (22.0x) |
| 100,000 | 8 | 36.0 us | 308.0 us (8.6x) | 432.2 us (12.0x) | 7.39 ms (205.6x) |
| 1,000,000 | 8 | 243.7 us | 3.05 ms (12.5x) | 8.25 ms (33.9x) | 84.68 ms (347.5x) |
| 1,000,000 | 64 | 1.67 ms | 24.51 ms (14.7x) | 9.89 ms (5.9x) | 631.05 ms (378.1x) |
| 10,000,000 | 8 | 2.55 ms | 32.66 ms (12.8x) | 146.18 ms (57.3x) | 993.77 ms (389.3x) |

**Sliding windows**, `goertzel_sliding_batch(x, W, freqs)` (OpenMP default threads):

| N | W | M | fastgoertzel | NumPy `sliding_window_view` + `rfft` |
|---:|---:|---:|---:|---:|
| 10,000 | 256 | 8 | 87.1 us | 7.98 ms (91.6x) |
| 10,000 | 256 | 64 | 2.79 ms | 18.99 ms (6.8x) |
| 100,000 | 256 | 8 | 2.28 ms | 88.76 ms (39.0x) |
| 100,000 | 1024 | 8 | 2.26 ms | 312.32 ms (137.9x) |

**OpenMP scaling**, batch N = 1e6, M = 32 and sliding N = 1e5, W = 256, M = 8:

| OMP_NUM_THREADS | batch | sliding |
|---:|---:|---:|
| 1 | 12.02 ms | 11.42 ms |
| 2 | 6.08 ms | 7.39 ms |
| 4 | 3.14 ms | 5.28 ms |
| 8 | 1.72 ms | 3.79 ms |
| 16 | 1.25 ms | 2.53 ms |
| default | 937.6 us | 2.38 ms |

**Per-call latency**, 100,000 calls each, Python overhead included:

| Call | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|
| goertzel        N=64 | 0.5 us | 0.6 us | 1.1 us | 2.4 us | 114 us |
| goertzel        N=256 | 0.8 us | 0.8 us | 1.4 us | 3.9 us | 385 us |
| goertzel        N=1024 | 0.9 us | 1.0 us | 1.6 us | 5.1 us | 733 us |
| goertzel        N=4096 | 2.2 us | 2.2 us | 3.4 us | 17.3 us | 464 us |
| batch           N=256    M=8 (serial) | 2.0 us | 2.2 us | 3.3 us | 12.8 us | 200 us |
| batch           N=65536  M=8 (OpenMP) | 24.7 us | 31.9 us | 73.6 us | 305.3 us | 363 us |
| sliding         N=1024 W=256 M=4 | 35.9 us | 37.3 us | 73.7 us | 146.9 us | 752 us |
| np.fft.rfft     N=256 (reference) | 3.5 us | 3.9 us | 5.6 us | 36.7 us | 324 us |
| NumPy 2x dot    N=256 (reference) | 4.8 us | 5.1 us | 7.4 us | 44.1 us | 388 us |
<!-- benchmark-tables:end -->
