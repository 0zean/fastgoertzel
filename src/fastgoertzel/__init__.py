# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast Goertzel algorithm: C++ kernel with Python bindings.

Functions
---------
goertzel(x, frequency)
    Amplitude and phase of one DFT bin.
goertzel_batch(x, frequencies)
    Several frequencies in one call, parallelized with OpenMP where available.
goertzel_sliding_batch(x, window_size, frequencies)
    Every sliding window of ``x`` at several frequencies.

See the README for the frequency-to-bin mapping and the phase/amplitude conventions.
"""

from ._core import (
    __version__,
    goertzel,
    goertzel_batch,
    goertzel_sliding_batch,
)

__all__ = ["__version__", "goertzel", "goertzel_batch", "goertzel_sliding_batch"]
