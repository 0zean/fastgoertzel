# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast Goertzel algorithm: C++ kernel with Python bindings.

Evaluates the DTFT of a real signal at chosen frequencies, exactly (no rounding to DFT
bins), with optional sample-rate input, batch and sliding-window variants, and a
streaming filter.

Functions
---------
goertzel(x, frequency, *, fs=None, scale="amplitude")
    Amplitude and phase at one frequency.
goertzel_complex(x, frequency, *, fs=None)
    Unscaled complex DTFT value at one frequency.
goertzel_bin(x, k, *, scale="amplitude")
    DFT bin ``k`` (integer-bin form, the same quantity as ``numpy.fft.fft(x)[k]``).
goertzel_batch(x, frequencies, *, fs=None, scale="amplitude", threads=None)
    Several frequencies in one call, OpenMP-parallel where available.
goertzel_batch_complex(x, frequencies, *, fs=None, threads=None)
    Complex variant of ``goertzel_batch``.
goertzel_sliding_batch(x, window_size, frequencies, *, step=1, fs=None, scale=..., threads=None)
    Every (strided) sliding window of ``x`` at several frequencies.
goertzel_sliding_batch_complex(x, window_size, frequencies, *, step=1, fs=None, threads=None)
    Complex variant of ``goertzel_sliding_batch`` (no ``sqrt``/``atan2`` per window).

Classes
-------
GoertzelFilter(frequency, *, fs=None)
    Stateful filter: ``push`` samples as they arrive, ``result()`` at any time.

See the README for conventions (phase, amplitude scaling, aliasing) and accuracy notes.
"""

from ._core import (
    GoertzelFilter,
    __version__,
    goertzel,
    goertzel_batch,
    goertzel_batch_complex,
    goertzel_bin,
    goertzel_complex,
    goertzel_sliding_batch,
    goertzel_sliding_batch_complex,
)

__all__ = [
    "GoertzelFilter",
    "__version__",
    "goertzel",
    "goertzel_batch",
    "goertzel_batch_complex",
    "goertzel_bin",
    "goertzel_complex",
    "goertzel_sliding_batch",
    "goertzel_sliding_batch_complex",
]
