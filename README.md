![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/dark%20logo.png#gh-light-mode-only)
![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/light%20logo.png#gh-dark-mode-only)

<!-- start here -->

fastgoertzel ![GitHub Actions](https://github.com/0zean/fastgoertzel/actions/workflows/ci.yml/badge.svg)
============

The Goertzel algorithm in C++ with Python bindings: evaluate the spectrum of a real signal at the frequencies you care about, exactly, without computing a full FFT. One frequency, many frequencies, sliding windows, or a streaming filter fed sample by sample.

## API

| Call | Returns |
|---|---|
| `goertzel(x, frequency, *, fs=None, scale="amplitude")` | `(value, phase)` at exactly `frequency` |
| `goertzel_complex(x, frequency, *, fs=None)` | complex `X = sum x[n] exp(-2j*pi*f*n)` |
| `goertzel_bin(x, k, *, scale=...)` | `(value, phase)` of DFT bin `k`, same as `np.fft.fft(x)[k]` |
| `goertzel_batch(x, frequencies, *, fs=None, scale=..., threads=None)` | array `(M, 2)` of `(value, phase)` |
| `goertzel_batch_complex(x, frequencies, *, fs=None, threads=None)` | complex array `(M,)` |
| `goertzel_sliding_batch(x, window_size, frequencies, *, step=1, fs=None, scale=..., threads=None)` | array `(n_windows, M, 2)` |
| `GoertzelFilter(frequency, *, fs=None)` | object with `push(x)`, `result(scale=...)`, `result_complex()`, `reset()`, `n_samples` |

Conventions, the same for every call:

- **Frequency** is in cycles per sample, `0 <= f < 1`, or in the unit of `fs` when `fs` is given (`f = frequency / fs`). It is evaluated exactly; nothing is rounded to a DFT bin. Frequencies above half the sample rate alias onto their mirror image. If you want integer-bin behaviour, use `goertzel_bin`.
- **Phase** is cosine-referenced, as in `numpy.fft`: `cos(2*pi*f*n + p)` reports `p`; a sine reports `p - pi/2`.
- **Scale.** `"amplitude"` (default) is `2|X|/N`, the amplitude of a sinusoid at that frequency, with `|X|/N` at DC and Nyquist where the two spectral lines coincide. `"magnitude"` is `|X|` and `"power"` is `|X|**2` (no square root), both unnormalized. The `*_complex` functions return `X` itself.
- **Validation.** Every call raises `ValueError` for an empty or non-1-D signal, a frequency outside its range, a non-positive `fs`, a `window_size` outside `[1, len(x)]`, or `step`/`threads` below 1. Nothing you pass can crash the interpreter.
- **Threads.** `goertzel_batch` and `goertzel_sliding_batch` use OpenMP when the build has it (Linux and Windows wheels; macOS wheels are single-threaded) and the work is at least 65,536 sample-frequency products. `threads=` overrides `OMP_NUM_THREADS` per call; `threads=1` forces serial execution, which gives the most predictable latency. The GIL is released in every call.
- **Accuracy.** Rounding error of the recurrence grows roughly linearly with the signal length and is largest near DC and Nyquist. Measured against a direct DTFT sum: about 2e-10 relative at N = 1e6 for interior frequencies, about 2e-5 for the first bin. Use float64 input; float32 is converted but offers no speed advantage.

Upgrading from 1.x: results are unchanged whenever `frequency * len(x)` was an exact integer. 1.x rounded that product *down* to a bin and reported twice the true amplitude at DC and Nyquist; see `CHANGELOG.md` for the migration note.

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

For development, `uv sync` creates the environment and builds the extension; `make check` runs lint, format check, type check, and tests, and `uv run pre-commit install` enables the ruff and clang-format hooks. `make kernel-bench` builds the standalone C++ kernel micro-benchmark.

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

Measured with `python benchmarks/benchmark.py` on an Intel Core i9-14900KF (8P + 16E cores, 32 threads), Windows 11, Python 3.11.9, NumPy 2.3.4, MSVC 19.44 build of fastgoertzel 2.0.0.dev0 with the AVX2+FMA kernel selected. Each time is the minimum of 7 rounds. Numbers on other machines will differ; re-run the script and quote your own, together with the baseline and thread count.

**One frequency**, `goertzel(x, f)`, single-threaded by design:

| N | fastgoertzel | pure-Python loop | NumPy 2x real dot | `np.fft.rfft`, one bin |
|---:|---:|---:|---:|---:|
| 1,000 | 1.7 us | 24 us (14x) | 10 us (5.8x) | 7.5 us (4.3x) |
| 100,000 | 40 us | 2.3 ms (59x) | 0.94 ms (24x) | 0.45 ms (11x) |
| 1,000,000 | 0.39 ms | 25 ms (64x) | 10.4 ms (27x) | 8.5 ms (22x) |
| 10,000,000 | 4.2 ms | not run | 113 ms (27x) | 127 ms (31x) |

That is about 2.5 billion samples per second on one core. The Goertzel recurrence is a dependency chain that costs a few cycles per sample however the loop is written; signals of 4,096 samples or more are therefore split into up to eight independent blocks whose results are combined exactly, which keeps the FMA units busy. The hot loops are compiled twice, for baseline x86-64 and for AVX2+FMA, and the build is chosen at import time, so the wheel still runs on any CPU. `FASTGOERTZEL_KERNEL=baseline` forces the portable build; `fastgoertzel._core._kernel` reports which one is active.

**Several frequencies**, `goertzel_batch`: up to eight frequencies share one pass over the signal, so the cost per frequency falls well below a single call, and OpenMP distributes lane groups over cores.

| N | M | fastgoertzel, 32 threads | fastgoertzel, 1 thread | `np.fft.rfft` (all bins) |
|---:|---:|---:|---:|---:|
| 1,000,000 | 8 | 0.27 ms | 1.6 ms | 8.4 ms (31x / 5x) |
| 1,000,000 | 32 | 1.06 ms | 12.6 ms | 8.5 ms (8x / 0.7x) |
| 1,000,000 | 64 | 1.9 ms | 25 ms | 9.6 ms (5x / 0.4x) |

The last column shows the honest crossover: with one thread, an FFT that computes all bins at once wins somewhere between 8 and 32 frequencies for N = 1e6. Goertzel is the tool for few frequencies, arbitrary (non-bin) frequencies, streaming, and sliding windows.

**Sliding windows**, `goertzel_sliding_batch`: consecutive windows are produced with the sliding-DFT update (constant cost per hop, re-synchronised exactly every 256 windows), so the window length no longer matters.

| N | W | M | fastgoertzel, 32 threads | fastgoertzel, 1 thread | NumPy `sliding_window_view` + `rfft` |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 256 | 8 | 2.5 ms | 11.8 ms | 87 ms (35x / 7x) |
| 100,000 | 1024 | 8 | 2.4 ms | about 12 ms | 321 ms (133x / 27x) |

Version 1.0.x recomputed every window from scratch and took 362 ms single-threaded on the first row. What remains per window is mostly the `sqrt` and `atan2` of the (value, phase) output.

**Per-call latency** matters more than throughput in a real-time loop. The same script reports percentiles over 100,000 individual calls (`--latency-only`), Python call overhead included:

| Call | p50 | p99 | p99.9 |
|---|---:|---:|---:|
| `goertzel`, 64 samples | 0.6 us | 1.2 us | 8.5 us |
| `goertzel`, 1,024 samples | 1.0 us | 1.9 us | 23 us |
| `goertzel`, 4,096 samples | 2.2 us | 3.7 us | 52 us |
| `goertzel_batch`, 256 samples, 8 frequencies (serial) | 2.1 us | 3.5 us | 34 us |
| `goertzel_batch`, 65,536 samples, 8 frequencies (OpenMP) | 30 us | 99 us | 300 us |
| `np.fft.rfft`, 256 samples (reference) | 3.5 us | 8.3 us | 61 us |

Calls that enter an OpenMP region carry a heavier tail. For predictable latency on small inputs pass `threads=1` or keep the work below the 65,536 sample-frequency threshold at which the parallel region is entered.
