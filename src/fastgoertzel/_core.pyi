# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str
_has_openmp: bool
# "avx2" or "baseline": the hot-loop build selected at import (FASTGOERTZEL_KERNEL overrides).
_kernel: str

_Scale = Literal["amplitude", "magnitude", "power"]

def goertzel(
    x: ArrayLike,
    frequency: float,
    *,
    fs: float | None = None,
    scale: _Scale = "amplitude",
) -> tuple[float, float]:
    """Amplitude and phase of ``x`` at exactly ``frequency`` (generalized Goertzel).

    ``frequency`` is in cycles per sample in ``[0, 1)``, or in the unit of ``fs`` in
    ``[0, fs)`` when ``fs`` is given. ``scale`` selects ``2|X|/N`` (``|X|/N`` at DC and
    Nyquist), ``|X|``, or ``|X|**2``. Phase is cosine-referenced as in ``numpy.fft``.
    Raises ``ValueError`` on invalid input.
    """
    ...

def goertzel_complex(
    x: ArrayLike,
    frequency: float,
    *,
    fs: float | None = None,
) -> complex:
    """Unscaled complex DTFT value ``sum x[n] exp(-2j*pi*f*n)`` at ``frequency``."""
    ...

def goertzel_bin(
    x: ArrayLike,
    k: int,
    *,
    scale: _Scale = "amplitude",
) -> tuple[float, float]:
    """Amplitude and phase of DFT bin ``k`` (``0 <= k < len(x)``), as ``numpy.fft.fft(x)[k]``."""
    ...

def goertzel_batch(
    x: ArrayLike,
    frequencies: ArrayLike,
    *,
    fs: float | None = None,
    scale: _Scale = "amplitude",
    threads: int | None = None,
) -> NDArray[np.float64]:
    """``goertzel`` for every frequency; shape ``(len(frequencies), 2)`` of (value, phase).

    ``threads`` sets the OpenMP thread count for this call (``None`` = OpenMP default).
    """
    ...

def goertzel_batch_complex(
    x: ArrayLike,
    frequencies: ArrayLike,
    *,
    fs: float | None = None,
    threads: int | None = None,
) -> NDArray[np.complex128]:
    """``goertzel_complex`` for every frequency; shape ``(len(frequencies),)``."""
    ...

def goertzel_sliding_batch(
    x: ArrayLike,
    window_size: int,
    frequencies: ArrayLike,
    *,
    step: int = 1,
    fs: float | None = None,
    scale: _Scale = "amplitude",
    threads: int | None = None,
) -> NDArray[np.float64]:
    """``goertzel_batch`` over windows ``x[w*step : w*step + window_size]``.

    Returns shape ``((len(x) - window_size) // step + 1, len(frequencies), 2)``.
    ``window_size`` must be in ``[1, len(x)]`` and ``step >= 1``.
    """
    ...

def goertzel_sliding_batch_complex(
    x: ArrayLike,
    window_size: int,
    frequencies: ArrayLike,
    *,
    step: int = 1,
    fs: float | None = None,
    threads: int | None = None,
) -> NDArray[np.complex128]:
    """``goertzel_sliding_batch`` returning the complex DTFT of each window.

    Shape ``((len(x) - window_size) // step + 1, len(frequencies))``; no ``sqrt``/``atan2``.
    """
    ...

class GoertzelFilter:
    """Stateful Goertzel filter for streaming input.

    ``push`` scalars or 1-D arrays in any chunking; ``result()`` equals ``goertzel`` over
    everything pushed so far. Not safe for concurrent ``push`` on one instance.
    """

    def __init__(self, frequency: float, *, fs: float | None = None) -> None: ...
    def push(self, x: ArrayLike) -> None: ...
    def result(self, *, scale: _Scale = "amplitude") -> tuple[float, float]: ...
    def result_complex(self) -> complex: ...
    def reset(self) -> None: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def frequency(self) -> float: ...
    @property
    def fs(self) -> float | None: ...
    @property
    def normalized_frequency(self) -> float: ...
