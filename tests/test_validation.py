"""Every invalid input must raise a Python exception. None may abort the interpreter.

Before 1.0.3, out-of-range frequencies in ``goertzel_sliding_batch`` threw inside an
OpenMP region and terminated the process, and ``goertzel_batch`` accepted anything.
"""

import subprocess
import sys

import numpy as np
import pytest

import fastgoertzel

X = np.arange(64, dtype=float)
BAD_FREQS = [1.0, 1.5, -0.1, -5.0, float("nan"), float("inf"), -float("inf"), 1e30]


class TestGoertzel:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="Frequency"):
            fastgoertzel.goertzel(X, freq)

    def test_two_dimensional_input(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            fastgoertzel.goertzel(X.reshape(8, 8), 0.1)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            fastgoertzel.goertzel(np.array([]), 0.1)

    def test_wrong_types(self):
        with pytest.raises(TypeError):
            fastgoertzel.goertzel(X, "0.1")
        with pytest.raises(TypeError):
            fastgoertzel.goertzel(X)


class TestGoertzelBatch:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="Frequency"):
            fastgoertzel.goertzel_batch(X, np.array([0.1, freq]))

    def test_two_dimensional_signal(self):
        with pytest.raises(ValueError, match="x must be a 1-dimensional"):
            fastgoertzel.goertzel_batch(X.reshape(8, 8), np.array([0.1]))

    def test_two_dimensional_frequencies(self):
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fastgoertzel.goertzel_batch(X, np.array([[0.1, 0.2]]))

    def test_scalar_frequencies(self):
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fastgoertzel.goertzel_batch(X, 0.1)

    def test_empty_signal(self):
        with pytest.raises(ValueError, match="at least one"):
            fastgoertzel.goertzel_batch(np.array([]), np.array([0.1]))


class TestGoertzelSlidingBatch:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="Frequency"):
            fastgoertzel.goertzel_sliding_batch(X, 8, np.array([0.1, freq]))

    @pytest.mark.parametrize("window", [0, 65, 1000])
    def test_bad_window(self, window):
        with pytest.raises(ValueError, match="window_size"):
            fastgoertzel.goertzel_sliding_batch(X, window, np.array([0.1]))

    def test_negative_or_float_window(self):
        with pytest.raises(TypeError):
            fastgoertzel.goertzel_sliding_batch(X, -1, np.array([0.1]))
        with pytest.raises(TypeError):
            fastgoertzel.goertzel_sliding_batch(X, 2.0, np.array([0.1]))

    def test_two_dimensional_inputs(self):
        with pytest.raises(ValueError, match="x must be a 1-dimensional"):
            fastgoertzel.goertzel_sliding_batch(X.reshape(8, 8), 4, np.array([0.1]))
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fastgoertzel.goertzel_sliding_batch(X, 4, np.array([[0.1]]))

    def test_empty_signal(self):
        with pytest.raises(ValueError, match="at least one"):
            fastgoertzel.goertzel_sliding_batch(np.array([]), 1, np.array([0.1]))


def test_invalid_inputs_never_abort_the_interpreter():
    """Runs the worst offenders in a child process; a crash would be a non-zero exit."""
    code = """
import numpy as np, fastgoertzel as fg
x = np.zeros(100)
cases = [
    lambda: fg.goertzel_sliding_batch(x, 10, np.array([0.1, 1.5])),
    lambda: fg.goertzel_sliding_batch(x, 0, np.array([0.1])),
    lambda: fg.goertzel_sliding_batch(x, 101, np.array([0.1])),
    lambda: fg.goertzel_batch(x, np.array([-1.0, np.nan, np.inf, 2.0])),
    lambda: fg.goertzel_batch(np.array([]), np.array([0.1])),
]
for case in cases:
    try:
        case()
    except ValueError:
        continue
    raise SystemExit("expected ValueError")
print("ok")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
