# Migrating from 1.x

Version 2.0 changes what a frequency means, adds the sample rate and the streaming
filter that the 1.x README promised, and replaces the build system. Most 1.x code runs
unchanged; this page lists what does not and what to do about it.

## Frequencies are evaluated exactly

In 1.x every function rounded `frequency * len(x)` *down* to an integer DFT bin, so
`goertzel(x, 0.0099)` on 100 samples silently returned the DC bin, and about 6 % of
exact `k / N` inputs landed on bin `k - 1` through floating-point rounding. 2.0 evaluates
the DTFT at exactly the frequency you pass.

| You wrote in 1.x | Result in 2.0 | What to do |
|---|---|---|
| `goertzel(x, k / len(x))` with the product an exact integer | unchanged | nothing |
| `goertzel(x, f)` with `f * len(x)` not an integer | the true value at `f`, no longer bin `floor(f * N)` | nothing, unless you relied on the rounding |
| code that relied on the rounding to a bin | different value | call `goertzel_bin(x, k)` with the integer bin; it is identical to `numpy.fft.fft(x)[k]` |
| `goertzel(x, 0)` or `goertzel(x, 0.5)` | half the 1.x value | 1.x reported twice the true amplitude at DC and Nyquist; 2.0 reports the sinusoid's amplitude. Use `scale="magnitude"` for raw `abs(X)`. |

## Signature changes

| 1.x | 2.0 |
|---|---|
| `goertzel(x, f)` | same; `fs=` and `scale=` are keyword-only additions |
| `goertzel_batch(x, freqs)` | same; `fs=`, `scale=`, `threads=` keyword-only |
| `goertzel_sliding_batch(x, W, freqs)` | same; `step=`, `fs=`, `scale=`, `threads=` keyword-only |
| positional extras such as `goertzel(x, f, 8000)` | `TypeError`; write `goertzel(x, f, fs=8000)` |
| `RuntimeError` for a non-1-D input | `ValueError` (since 1.0.3) |
| `fastgoertzel._fastgoertzel_core` | `fastgoertzel._core` (private; import from `fastgoertzel`) |

New in 2.0, all additive: `goertzel_complex`, `goertzel_bin`, `goertzel_batch_complex`,
`goertzel_sliding_batch_complex`, `GoertzelFilter`, the `fs=` keyword everywhere, type
stubs (`py.typed`).

## Building from source

| 1.x | 2.0 |
|---|---|
| setuptools, `setup.py`, `MANIFEST.in` | scikit-build-core, `CMakeLists.txt` |
| `FASTGOERTZEL_NATIVE=1 pip install .` | `CMAKE_ARGS="-DFASTGOERTZEL_NATIVE=ON" pip install .` |
| NumPy needed at build time | not needed; runtime requirement `numpy>=1.21.2` |
| Python 3.10 to 3.13 | 3.10 to 3.14 |

Wheels built by 1.0.2 and earlier contained `-march=native` code; since 1.0.3 they run on
any x86-64 or arm64 CPU, and 2.0 adds the AVX2+FMA build selected at import, so there is
no longer a reason to build from source for speed on x86-64.

## Numerical differences

Where 1.x and 2.0 compute the same quantity (exact-bin frequencies away from DC and
Nyquist), results agree to about 1e-12 relative. 2.0 splits long signals into blocks and
uses a different recurrence formulation, so the last bits differ; both are closer to the
exact sum than 1.x was for long signals, because the block split shortens the error
growth of the recurrence.
