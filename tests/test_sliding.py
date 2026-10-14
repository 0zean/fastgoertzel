"""goertzel_sliding_batch_complex: the complex-output sliding variant."""

import numpy as np
import pytest

import fastgoertzel as fg


def dtft(x: np.ndarray, f: float) -> complex:
    n = np.arange(len(x))
    return complex(np.sum(x * np.exp(-2j * np.pi * f * n)))


@pytest.mark.parametrize("window,step", [(16, 1), (300, 1), (300, 5), (2048, 1), (2048, 600)])
def test_complex_sliding_matches_direct_windows(window, step):
    rng = np.random.default_rng(window * 7 + step)
    n = window + step * 600
    x = rng.standard_normal(n) - 1.5
    freqs = np.array([0.0, 0.5, 0.3333, 0.07])
    res = fg.goertzel_sliding_batch_complex(x, window, freqs, step=step)
    n_windows = (n - window) // step + 1
    assert res.shape == (n_windows, len(freqs))
    assert res.dtype == np.complex128
    for w in (0, 1, 255, 256, 257, n_windows // 2, n_windows - 1):
        seg = x[w * step : w * step + window]
        expected = np.array([dtft(seg, f) for f in freqs])
        np.testing.assert_allclose(res[w], expected, rtol=1e-9, atol=1e-8)


def test_complex_and_scaled_variants_agree():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(5_000)
    freqs = [0.11, 0.22, 0.33]
    c = fg.goertzel_sliding_batch_complex(x, 256, freqs, step=3)
    s = fg.goertzel_sliding_batch(x, 256, freqs, step=3, scale="magnitude")
    np.testing.assert_allclose(np.abs(c), s[..., 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.angle(c), s[..., 1], atol=1e-12)
    c2 = fg.goertzel_sliding_batch_complex(x, 256, freqs, step=3, threads=1)
    np.testing.assert_array_equal(c, c2)


def test_complex_sliding_with_fs_and_validation():
    x = np.cos(2 * np.pi * 50 / 1000 * np.arange(3_000))
    c = fg.goertzel_sliding_batch_complex(x, 1000, [50, 100], fs=1000)
    np.testing.assert_allclose(np.abs(c[:, 0]) * 2 / 1000, 1.0, atol=1e-9)
    np.testing.assert_allclose(np.abs(c[:, 1]), 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="window_size"):
        fg.goertzel_sliding_batch_complex(x, 0, [0.1])
    with pytest.raises(ValueError, match="step"):
        fg.goertzel_sliding_batch_complex(x, 10, [0.1], step=0)
    with pytest.raises(ValueError, match="frequency"):
        fg.goertzel_sliding_batch_complex(x, 10, [1.5])
    with pytest.raises(TypeError):
        fg.goertzel_sliding_batch_complex(x, 10, [0.1], "amplitude")
