"""Fast Goertzel Algorithm Implementation."""

import numpy as np

from ._fastgoertzel_core import goertzel, goertzel_batch, goertzel_sliding_batch

__version__ = "1.0.2"


def _hz_to_normalized(frequencies_hz, sample_rate):
    """Convert frequencies in Hz to normalized frequencies."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)

    if np.any(frequencies < 0):
        raise ValueError("frequencies_hz must be non-negative")

    return frequencies / sample_rate


def goertzel_hz(x, frequency_hz, sample_rate):
    """Compute amplitude and phase at a frequency given in Hz.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional input signal.
    frequency_hz : float
        Target frequency in Hz.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    tuple[float, float]
        Amplitude and phase in radians.
    """
    frequency = float(_hz_to_normalized(frequency_hz, sample_rate))
    return goertzel(x, frequency)


def goertzel_batch_hz(x, frequencies_hz, sample_rate):
    """Compute amplitude and phase for multiple frequencies given in Hz.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional input signal.
    frequencies_hz : numpy.ndarray
        One-dimensional array of target frequencies in Hz.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frequencies, 2)``.
        Column 0 contains amplitudes and column 1 contains phases in radians.
    """
    frequencies = _hz_to_normalized(frequencies_hz, sample_rate)
    return goertzel_batch(x, frequencies)


def goertzel_sliding_batch_hz(x, window_size, frequencies_hz, sample_rate):
    """Compute sliding-window Goertzel results for frequencies given in Hz.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional input signal.
    window_size : int
        Size of each sliding window in samples.
    frequencies_hz : numpy.ndarray
        One-dimensional array of target frequencies in Hz.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_windows, n_frequencies, 2)``.
        The last dimension contains amplitude and phase in radians.
    """
    frequencies = _hz_to_normalized(frequencies_hz, sample_rate)
    return goertzel_sliding_batch(x, window_size, frequencies)


__all__ = [
    "goertzel",
    "goertzel_batch",
    "goertzel_sliding_batch",
    "goertzel_hz",
    "goertzel_batch_hz",
    "goertzel_sliding_batch_hz",
]