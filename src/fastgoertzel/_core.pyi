# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str
_has_openmp: bool

def goertzel(x: ArrayLike, frequency: float) -> tuple[float, float]:
    """Amplitude and phase of DFT bin ``floor(frequency * len(x))`` of ``x``.

    ``x`` is converted to a contiguous 1-D float64 array. ``frequency`` must be in
    ``[0, 1)``. Phase is cosine-referenced; amplitude is ``2|X|/N``.
    Raises ``ValueError`` for an empty or non-1-D ``x`` or an out-of-range frequency.
    """
    ...

def goertzel_batch(x: ArrayLike, frequencies: ArrayLike) -> NDArray[np.float64]:
    """Same as ``goertzel`` for every entry of ``frequencies``.

    Returns an array of shape ``(len(frequencies), 2)`` with amplitude in column 0 and
    phase in column 1. Raises ``ValueError`` on invalid input.
    """
    ...

def goertzel_sliding_batch(
    x: ArrayLike,
    window_size: int,
    frequencies: ArrayLike,
) -> NDArray[np.float64]:
    """``goertzel_batch`` over every window ``x[w : w + window_size]``.

    Returns an array of shape ``(len(x) - window_size + 1, len(frequencies), 2)``.
    ``window_size`` must be in ``[1, len(x)]``. Raises ``ValueError`` on invalid input.
    """
    ...
