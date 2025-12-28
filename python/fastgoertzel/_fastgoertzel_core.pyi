from typing import Tuple

import numpy as np
from numpy.typing import NDArray

def goertzel(x: NDArray[np.float64], frequency: float) -> Tuple[float, float]:
    """
    Compute amplitude and phase using the Goertzel algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        1-dimensional array of input samples
    frequency : float
        Normalized frequency to detect (0 to 1, where 1 = sampling rate)

    Returns
    -------
    tuple[float, float]
        (amplitude, phase) where phase is in radians

    Examples
    --------
    >>> import numpy as np
    >>> import fastgoertzel
    >>>
    >>> # Generate test signal
    >>> t = np.arange(0, 512)
    >>> freq = 1/128  # Normalized frequency
    >>> signal = np.sin(2 * np.pi * freq * t)
    >>>
    >>> # Detect frequency
    >>> amp, phase = fastgoertzel.goertzel(signal, freq)
    >>> print(f"Amplitude: {amp:.4f}, Phase: {phase:.4f}")

    Notes
    -----
    The Goertzel algorithm is particularly efficient when detecting
    a small number of frequencies compared to computing a full FFT.
    """
    ...

def goertzel_batch(
    x: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Process multiple frequencies in a single call.

    Parameters
    ----------
    x : numpy.ndarray
        1-dimensional array of input samples
    frequencies : numpy.ndarray
        1-dimensional array of normalized frequencies

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_frequencies, 2) with amplitudes and phases
    """
    ...

def goertzel_sliding_batch(
    x: NDArray[np.float64],
    window_size: int,
    frequencies: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Process sliding windows of data for multiple frequencies.

    Parameters
    ----------
    x : numpy.ndarray
        1-dimensional array of input samples
    window_size : int
        Size of the sliding window
    frequencies : numpy.ndarray
        1-dimensional array of normalized frequencies

    Returns
    -------
    numpy.ndarray
        Array of shape (n_windows, n_frequencies, 2) with amplitudes and phases
    """
    ...

__version__: str
