"""Every invalid input must raise a Python exception. None may abort the interpreter.

Before 1.0.3, out-of-range frequencies in ``goertzel_sliding_batch`` threw inside an
OpenMP region and terminated the process, and ``goertzel_batch`` accepted anything.
"""

import subprocess
import sys

import numpy as np
import pytest

import fastgoertzel as fg

X = np.arange(64, dtype=float)
BAD_FREQS = [1.0, 1.5, -0.1, -5.0, float("nan"), float("inf"), -float("inf"), 1e30]
BAD_FS = [0.0, -1.0, float("nan"), float("inf")]


class TestGoertzel:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel(X, freq)
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel_complex(X, freq)

    def test_frequency_at_or_above_fs(self):
        with pytest.raises(ValueError, match=r"\[0, fs\)"):
            fg.goertzel(X, 8000, fs=8000)
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel(X, -1, fs=8000)

    @pytest.mark.parametrize("fs", BAD_FS)
    def test_bad_fs(self, fs):
        with pytest.raises(ValueError, match="fs"):
            fg.goertzel(X, 0.1, fs=fs)

    def test_bad_scale(self):
        with pytest.raises(ValueError, match="scale"):
            fg.goertzel(X, 0.1, scale="db")

    def test_two_dimensional_input(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            fg.goertzel(X.reshape(8, 8), 0.1)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            fg.goertzel(np.array([]), 0.1)

    def test_wrong_types(self):
        with pytest.raises(TypeError):
            fg.goertzel(X, "0.1")
        with pytest.raises(TypeError):
            fg.goertzel(X)
        with pytest.raises(TypeError):
            fg.goertzel(X, 0.1, 8000)  # fs is keyword-only


class TestGoertzelBin:
    @pytest.mark.parametrize("k", [-1, 64, 1000])
    def test_bad_bin(self, k):
        with pytest.raises(ValueError, match=r"k must be in \[0, len\(x\)\)"):
            fg.goertzel_bin(X, k)

    def test_non_integer_bin(self):
        with pytest.raises(TypeError):
            fg.goertzel_bin(X, 2.5)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            fg.goertzel_bin(np.array([]), 0)


class TestGoertzelBatch:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel_batch(X, np.array([0.1, freq]))
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel_batch_complex(X, np.array([0.1, freq]))

    def test_two_dimensional_signal(self):
        with pytest.raises(ValueError, match="x must be a 1-dimensional"):
            fg.goertzel_batch(X.reshape(8, 8), np.array([0.1]))

    def test_two_dimensional_frequencies(self):
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fg.goertzel_batch(X, np.array([[0.1, 0.2]]))

    def test_scalar_frequencies(self):
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fg.goertzel_batch(X, 0.1)

    def test_empty_signal(self):
        with pytest.raises(ValueError, match="at least one"):
            fg.goertzel_batch(np.array([]), np.array([0.1]))

    @pytest.mark.parametrize("threads", [0, -1])
    def test_bad_threads(self, threads):
        with pytest.raises(ValueError, match="threads"):
            fg.goertzel_batch(X, np.array([0.1]), threads=threads)

    def test_bad_fs_and_scale(self):
        with pytest.raises(ValueError, match="fs"):
            fg.goertzel_batch(X, [0.1], fs=0)
        with pytest.raises(ValueError, match="scale"):
            fg.goertzel_batch(X, [0.1], scale="nope")


class TestGoertzelSlidingBatch:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="frequency"):
            fg.goertzel_sliding_batch(X, 8, np.array([0.1, freq]))

    @pytest.mark.parametrize("window", [0, -1, 65, 1000])
    def test_bad_window(self, window):
        with pytest.raises(ValueError, match="window_size"):
            fg.goertzel_sliding_batch(X, window, np.array([0.1]))

    @pytest.mark.parametrize("step", [0, -3])
    def test_bad_step(self, step):
        with pytest.raises(ValueError, match="step"):
            fg.goertzel_sliding_batch(X, 8, np.array([0.1]), step=step)

    def test_float_window(self):
        with pytest.raises(TypeError):
            fg.goertzel_sliding_batch(X, 2.0, np.array([0.1]))

    def test_two_dimensional_inputs(self):
        with pytest.raises(ValueError, match="x must be a 1-dimensional"):
            fg.goertzel_sliding_batch(X.reshape(8, 8), 4, np.array([0.1]))
        with pytest.raises(ValueError, match="frequencies must be a 1-dimensional"):
            fg.goertzel_sliding_batch(X, 4, np.array([[0.1]]))

    def test_empty_signal(self):
        with pytest.raises(ValueError, match="at least one"):
            fg.goertzel_sliding_batch(np.array([]), 1, np.array([0.1]))


class TestGoertzelFilter:
    @pytest.mark.parametrize("freq", BAD_FREQS)
    def test_bad_frequency(self, freq):
        with pytest.raises(ValueError, match="frequency"):
            fg.GoertzelFilter(freq)

    @pytest.mark.parametrize("fs", BAD_FS)
    def test_bad_fs(self, fs):
        with pytest.raises(ValueError, match="fs"):
            fg.GoertzelFilter(100.0, fs=fs)

    def test_result_before_push(self):
        flt = fg.GoertzelFilter(0.1)
        with pytest.raises(ValueError, match="no samples"):
            flt.result()
        with pytest.raises(ValueError, match="no samples"):
            flt.result_complex()

    def test_push_two_dimensional(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            fg.GoertzelFilter(0.1).push(X.reshape(8, 8))

    def test_bad_scale(self):
        flt = fg.GoertzelFilter(0.1)
        flt.push(1.0)
        with pytest.raises(ValueError, match="scale"):
            flt.result(scale="log")


def test_invalid_inputs_never_abort_the_interpreter():
    """Runs the worst offenders in a child process; a crash would be a non-zero exit."""
    code = """
import numpy as np, fastgoertzel as fg
x = np.zeros(100)
cases = [
    lambda: fg.goertzel_sliding_batch(x, 10, np.array([0.1, 1.5])),
    lambda: fg.goertzel_sliding_batch(x, 0, np.array([0.1])),
    lambda: fg.goertzel_sliding_batch(x, 101, np.array([0.1])),
    lambda: fg.goertzel_sliding_batch(x, 10, np.array([0.1]), step=0),
    lambda: fg.goertzel_batch(x, np.array([-1.0, np.nan, np.inf, 2.0])),
    lambda: fg.goertzel_batch(np.array([]), np.array([0.1])),
    lambda: fg.goertzel_batch(x, np.array([0.1]), threads=0),
    lambda: fg.goertzel_bin(x, 100),
    lambda: fg.GoertzelFilter(0.1).result(),
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
