![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/dark%20logo.png#gh-light-mode-only)
![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/light%20logo.png#gh-dark-mode-only)

<!-- start here -->

fastgoertzel ![GitHub Actions](https://github.com/0zean/fastgoertzel/actions/workflows/ci.yml/badge.svg)
============

A Python implementation of the Goertzel algorithm built using `C++` for improved run time and efficiency on large datasets and loops.


## Status and roadmap

- [x] ~~Improved speed.~~ (Significantly increased speed by using numpy arrays).
- [x] Benchmarking for speed comparison. See [Performance](#performance) for measured, reproducible numbers. The earlier "~75x faster than native Python" figure was measured against a Python loop that indexes a NumPy array one element at a time; against an idiomatic pure-Python loop the speed-up is 10-15x, and against a vectorised NumPy single-bin DFT it is 5-8x.
- [x] Batch processing for multiple frequencies (`goertzel_batch`, `goertzel_sliding_batch`).
- [x] ~~Add IIR and k-th FFT implementation of Goertzel.~~ This is what the library already does: `goertzel` runs the second-order IIR Goertzel filter and evaluates the k-th DFT bin, with `k = floor(frequency * len(x))`. The two things that are genuinely missing are listed separately below.
- [ ] Generalized Goertzel for non-integer bins: evaluate the DTFT at exactly `frequency` instead of at the nearest-below DFT bin. Planned for 2.0.
- [ ] Streaming (stateful) filter API that keeps the IIR state between calls, for sample-by-sample or chunked input. Planned for 2.0.
- [ ] Sampling-rate support (`fs=` keyword, frequencies in Hz). Deliberately deferred until the generalized Goertzel lands: with the current bin truncation, asking for 440 Hz at `fs=44100` with 4096 samples would silently evaluate bin 40, which is 430.7 Hz.

## Important semantics of the current release

- `frequency` is a normalized frequency in cycles per sample, in `[0, 1)`. It is quantized to a DFT bin: `k = floor(frequency * len(x))`. A frequency that does not fall exactly on a bin is evaluated at the bin below it, and `frequency < 1/len(x)` is evaluated at DC. Pass `k / len(x)` and check `int((k / N) * N) == k` if an exact bin is required; floating-point rounding makes a few percent of `k / N` values truncate to `k - 1` for non-power-of-two `N`.
- Phase is cosine-referenced, as with `numpy.fft`: `cos(w*n + p)` gives phase `p`, and `sin(w*n + p)` gives `p - pi/2`.
- Amplitude is `2 * |X[k]| / N`, which recovers the amplitude of a sinusoid at any interior bin. At DC (`k = 0`) and Nyquist (`k = N/2`) this is twice the true amplitude.
- All three functions validate their inputs and raise `ValueError` for an empty or non-1-D signal, a frequency outside `[0, 1)`, or a `window_size` outside `[1, len(x)]`. Nothing you pass can crash the interpreter (fixed in 1.0.3; earlier releases aborted the process on an invalid frequency in `goertzel_sliding_batch`).
- `goertzel_batch` and `goertzel_sliding_batch` use OpenMP threads when the build supports it (Linux and Windows wheels; macOS wheels are single-threaded) and the work is large enough. Set `OMP_NUM_THREADS` to control the thread count. The GIL is released in every function.

## Installation

From PyPI (wheels for Linux, macOS, and Windows, plus a source distribution):
```bash
pip install fastgoertzel
```

From source (needs a C++17 compiler):
```bash
git clone https://github.com/0zean/fastgoertzel.git
cd fastgoertzel
pip install .
```

Published wheels are compiled for baseline x86-64 and arm64 so that they run on any CPU. When building from source for your own machine you can opt in to CPU-specific tuning with `FASTGOERTZEL_NATIVE=1 pip install .` (not Windows). On macOS, Apple's compiler has no OpenMP; to enable threading install `libomp` with Homebrew and set `FASTGOERTZEL_OPENMP_FLAGS` as described at the top of `setup.py`.

For development, `uv sync` creates the environment and builds the extension in place; `make check` runs lint, format check, and tests.

## Usage
```python
import numpy as np
import pandas as pd

import fastgoertzel as fg


def wave(amp, freq, phase, x):
    return amp * np.sin(2*np.pi * freq * x + phase)


x = np.arange(0, 512)
y = wave(1, 1/128, 0, x)

amp, phase = fg.goertzel(y, 1/128)
print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')

# Compared to max amplitude FFT output 
ft = np.fft.fft(y)
FFT = pd.DataFrame()
FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
FFT['phase'] = np.arctan2(ft.imag, ft.real)

max_ = FFT.iloc[FFT['amp'].idxmax()]
print(f'FFT amp: {max_["amp"]:.4f}, '
        f'phase: {max_["phase"]:.4f}, '
        f'freq: {max_["freq"]:.4f}')

```

## Performance

Measured with `python benchmarks/benchmark.py` on an Intel Core i9-14900KF (8P + 16E cores, 32 threads), Windows 11, Python 3.11.9, NumPy 2.4.6, MSVC 19.44 build of fastgoertzel 1.0.2. Each time is the minimum of 7 rounds. Numbers on other machines will differ; re-run the script and quote your own, together with the baseline and thread count.

Single frequency, `goertzel(x, 0.1)`:

| N | fastgoertzel | pure-Python loop | NumPy 2x real dot | `np.fft.rfft`, one bin |
|---:|---:|---:|---:|---:|
| 1,000 | 2.3 us | 24 us (10x) | 11 us (4.6x) | 7.3 us (3.1x) |
| 100,000 | 163 us | 2.4 ms (15x) | 0.94 ms (5.8x) | 0.44 ms (2.7x) |
| 1,000,000 | 1.63 ms | 24 ms (15x) | 9.4 ms (5.8x) | 8.6 ms (5.3x) |
| 10,000,000 | 16.3 ms | not run | 116 ms (7.1x) | 126 ms (7.7x) |

The single-frequency path runs at about 600 million samples per second on one core. It is bound by the latency of the Goertzel recurrence, so AVX2 and loop unrolling do not change it.

Per-call latency matters more than throughput in a real-time loop. The same script reports percentiles over 100,000 individual calls (`--latency-only`). On the machine above, one `goertzel` call on 64 samples has a p50 of 0.6 us and a p99 of 0.8 us, and on 1,024 samples a p50 of 2.0 us and a p99 of 3.5 us, Python call overhead included. Calls that enter an OpenMP region have a much heavier tail: `goertzel_sliding_batch` on 1,024 samples with a 256-sample window and 4 frequencies has a p50 of 59 us but a maximum of about 6 ms. Pin `OMP_NUM_THREADS=1` for predictable latency on small inputs.

`goertzel_batch` and `goertzel_sliding_batch` parallelize over frequencies and windows with OpenMP, so their speed-ups scale with core count. On this 32-thread machine `goertzel_batch` with N = 1,000,000 and 32 frequencies takes 2.3 ms, against 51 ms on one thread. `goertzel_sliding_batch` with N = 100,000, window 256 and 8 frequencies takes 17 ms, 5x faster than NumPy's `sliding_window_view` plus `rfft`, but 362 ms on one thread, which is 4x slower than NumPy. Pin `OMP_NUM_THREADS` when comparing.