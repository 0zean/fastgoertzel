# Usage

Every function takes a real 1-D signal `x` (anything NumPy converts to a contiguous
float64 array) and one or more frequencies, and returns the discrete-time Fourier
transform

$$X(f) = \sum_{n=0}^{N-1} x[n]\, e^{-2\pi j f n}$$

at exactly those frequencies, either as the complex value `X` or as a `(value, phase)`
pair. The conventions below hold for every call.

## Conventions

**Frequency.** In cycles per sample, `0 <= f < 1`, or in the unit of `fs` when `fs` is
given (`f = frequency / fs`, so `fs=44100` makes `frequency` hertz). The frequency is
evaluated exactly; nothing is rounded to a DFT bin. Frequencies above half the sample
rate alias onto their mirror image, as the DTFT of a real signal does. For integer-bin
behaviour use {func}`~fastgoertzel.goertzel_bin`.

**Phase.** Cosine-referenced, as in `numpy.fft`: a signal `cos(2*pi*f*n + p)` reports
phase `p`; a sine reports `p - pi/2`.

**Scale.** `scale="amplitude"` (default) returns `2|X|/N`, the amplitude of a real
sinusoid at that frequency, with `|X|/N` at DC and Nyquist where the two spectral lines
coincide. `"magnitude"` returns `|X|` and `"power"` returns `|X|**2` (no square root),
both unnormalised. The `*_complex` functions return `X` itself.

**Validation.** Every call raises `ValueError` for an empty or non-1-D signal, a
frequency outside its range or not finite, a non-positive `fs`, a `window_size` outside
`[1, len(x)]`, or `step` or `threads` below 1. Validation happens before the GIL is
released and before any parallel region, so no input can crash the interpreter.

**Determinism.** Results do not depend on the thread count or on which kernel build
(baseline or AVX2) the CPU selected, beyond rounding at the 1e-12 level. The work
decomposition depends only on the input sizes, and partial results are combined in a
fixed order.

## One frequency

```python
import numpy as np
import fastgoertzel as fg

fs = 44100.0
t = np.arange(4096)
x = 0.8 * np.cos(2 * np.pi * 440 / fs * t + 0.3)

amp, phase = fg.goertzel(x, 440, fs=fs)              # (0.8, 0.3), up to leakage
X = fg.goertzel_complex(x, 440, fs=fs)               # complex X(f), unscaled
power, _ = fg.goertzel(x, 440 / fs, scale="power")   # cycles per sample, |X|**2
```

`goertzel_complex` skips the `sqrt` and `atan2` of the amplitude form and is the right
call when you combine results yourself (averaging windows, computing cross-spectra).

"Up to leakage": 440 Hz over 4096 samples at 44.1 kHz is 40.86 cycles, not a whole
number, so the negative-frequency image of the cosine contributes a few tenths of a
percent to `X(440)`. That is a property of the finite window, not of the algorithm; an
FFT zero-padded finely enough to land on 440 Hz reports the same number. Use a whole
number of cycles, or apply a window function to `x` beforehand, when it matters.

## Integer DFT bins

```python
amp_k, phase_k = fg.goertzel_bin(x, 41)              # bin 41 of a 4096-point DFT
X = np.fft.fft(x)[41]
assert np.isclose(amp_k, 2 * abs(X) / x.size) and np.isclose(phase_k, np.angle(X))
```

`goertzel_bin(x, k)` equals `goertzel(x, k / len(x))` but computes the twiddles in exact
integer arithmetic, so it matches `numpy.fft.fft(x)[k]` as closely as the recurrence
allows. It is also what versions 1.x computed for every call.

## Several frequencies

```python
res = fg.goertzel_batch(x, [430, 440, 450], fs=fs, scale="power")   # shape (3, 2)
Xs = fg.goertzel_batch_complex(x, [430, 440, 450], fs=fs)           # shape (3,), complex
```

Row `i` equals the single-frequency call for `frequencies[i]`. Up to eight frequencies
share one pass over the signal, and the frequency groups are distributed over OpenMP
threads when the build has OpenMP and `len(x) * len(frequencies)` is at least 65,536.
`threads=` sets the thread count for that call; `threads=1` forces serial execution.

Against `np.fft.rfft`, which computes every bin at once, the single-threaded crossover is
between 8 and 32 frequencies at a million samples. The Goertzel algorithm is the tool
for few frequencies, non-bin frequencies, streaming, and sliding windows; for a dense
spectrum use `numpy.fft`.

## Sliding windows

```python
win = fg.goertzel_sliding_batch(x, 1024, [440, 880], step=256, fs=fs)           # (13, 2, 2)
winX = fg.goertzel_sliding_batch_complex(x, 1024, [440, 880], step=256, fs=fs)  # (13, 2)
```

Window `w` covers `x[w*step : w*step + window_size]` and there are
`(len(x) - window_size) // step + 1` windows. When `4 * step < window_size`, consecutive
windows are obtained with the sliding-DFT update, which costs a constant amount per hop
regardless of the window length; every 256 windows the value is recomputed from scratch
so rounding error cannot accumulate. For larger steps each window is computed directly.
The complex variant skips the per-window `sqrt` and `atan2`, which dominate the cost of
the `(value, phase)` form once the sliding update has made the window length irrelevant.

## Streaming

```python
flt = fg.GoertzelFilter(440, fs=fs)
for chunk in np.array_split(x, 16):
    flt.push(chunk)                      # arrays of any length, or scalars
    amp, phase = flt.result()            # over everything pushed so far
X = flt.result_complex()
assert flt.n_samples == 4096
flt.reset()
```

`GoertzelFilter` is the Goertzel algorithm used as what it is, a second-order IIR filter:
samples enter as they arrive and the spectrum at the chosen frequency is available at any
time without buffering. `result()` equals `goertzel` applied to the concatenation of
everything pushed, whatever the chunking. Chunks of 4,096 samples or more go through the
same block-parallel kernel as the one-shot functions. An instance holds a few words of
state and no buffers. Do not push to one instance from two threads at once.

## Threads and latency

- The single-frequency functions and `GoertzelFilter` never start threads; their latency
  is a function of `len(x)` only, around a microsecond for short signals.
- `goertzel_batch` and the sliding functions enter an OpenMP region when the work is at
  least 65,536 sample-frequency products. Entering a parallel region costs tens of
  microseconds and has a heavier tail. For predictable latency on small inputs pass
  `threads=1`, keep the work below the threshold, or set `OMP_NUM_THREADS=1`.
- The GIL is released in every call, so Python threads can run calls concurrently.
- `fastgoertzel._core._has_openmp` and `fastgoertzel._core._kernel` report the build.

## Accuracy

Rounding error of the recurrence grows roughly linearly with the signal length and is
largest near DC and Nyquist, where the filter's poles approach the real axis of the unit
circle. Measured against a direct DTFT sum in float64: about 2e-10 relative error at one
million samples for interior frequencies and about 2e-5 for the first bin. Signals of
4,096 samples or more are split into blocks whose results are combined exactly, which
also shortens the error growth. Use float64 input; float32 is converted and offers no
speed advantage.

## When to use it instead of an FFT

The Goertzel algorithm costs about `2 * N` flops per frequency against roughly
`2.5 * N * log2(N)` for a full FFT, so it wins when you need fewer than about `log2(N)`
frequencies, when the frequencies are not DFT bins, when samples arrive as a stream, or
when you want the result for a window without allocating a spectrum. The
[Performance](performance.md) page has measured numbers against NumPy for each case.
