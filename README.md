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

Measured with `python benchmarks/benchmark.py` on an Intel Core i9-14900KF (8P + 16E cores, 32 threads), Windows 11, Python 3.11.9, NumPy 2.4.6, MSVC 19.44 build of fastgoertzel 1.0.2. Each time is the minimum of 7 rounds. Numbers on other machines will differ; re-run the script and quote your own, together with the baseline and thread count.

Single frequency, `goertzel(x, 0.1)`:

| N | fastgoertzel | pure-Python loop | NumPy 2x real dot | `np.fft.rfft`, one bin |
|---:|---:|---:|---:|---:|
| 1,000 | 2.3 us | 24 us (10x) | 11 us (4.6x) | 7.3 us (3.1x) |
| 100,000 | 163 us | 2.4 ms (15x) | 0.94 ms (5.8x) | 0.44 ms (2.7x) |
| 1,000,000 | 1.63 ms | 24 ms (15x) | 9.4 ms (5.8x) | 8.6 ms (5.3x) |
| 10,000,000 | 16.3 ms | not run | 116 ms (7.1x) | 126 ms (7.7x) |

The single-frequency path ran at about 600 million samples per second on one core in 1.0.x; the 2.0 reformulation of the recurrence (`coeff*s1 + (x - s2)`, one multiply-add on the dependency chain instead of three operations) raises that to about 790 on the same machine with no SIMD involved. The path is bound by the latency of that chain, so AVX2 flags and loop unrolling do not change it; breaking the chain into independent blocks does, and is the next planned step.

Per-call latency matters more than throughput in a real-time loop. The same script reports percentiles over 100,000 individual calls (`--latency-only`). On the machine above, one `goertzel` call on 64 samples has a p50 of 0.6 us and a p99 of 0.8 us, and on 1,024 samples a p50 of 2.0 us and a p99 of 3.5 us, Python call overhead included. Calls that enter an OpenMP region have a much heavier tail: `goertzel_sliding_batch` on 1,024 samples with a 256-sample window and 4 frequencies has a p50 of 59 us but a maximum of about 6 ms. Pin `OMP_NUM_THREADS=1` for predictable latency on small inputs.

`goertzel_batch` and `goertzel_sliding_batch` parallelize over frequencies and windows with OpenMP, so their speed-ups scale with core count. On this 32-thread machine `goertzel_batch` with N = 1,000,000 and 32 frequencies takes 2.3 ms, against 51 ms on one thread. `goertzel_sliding_batch` with N = 100,000, window 256 and 8 frequencies takes 17 ms, 5x faster than NumPy's `sliding_window_view` plus `rfft`, but 362 ms on one thread, which is 4x slower than NumPy. Pin `OMP_NUM_THREADS` when comparing.
