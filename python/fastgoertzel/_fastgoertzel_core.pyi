from typing import Tuple

import numpy as np
from numpy.typing import NDArray

def goertzel(x: NDArray[np.float64], frequency: float) -> Tuple[float, float]:
    """
    Compute amplitude and phase using Goertzel algorithm.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input signal array
    frequency : float
        Normalized frequency (0 to 1)

    Returns
    -------
    Tuple[float, float]
        (amplitude, phase) tuple
    """
    ...

def goertzel_batch(
    x: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Process multiple frequencies in batch.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input signal array
    frequencies : NDArray[np.float64]
        Array of normalized frequencies

    Returns
    -------
    NDArray[np.float64]
        2D array of shape (n_frequencies, 2)
    """
    ...

__version__: str
