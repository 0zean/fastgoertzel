"""Performance regression guards.

Absolute timings are meaningless on shared CI runners, so every check here is a *ratio*
measured in the same process moments apart: fastgoertzel against a NumPy formulation of
the same computation, or one fastgoertzel path against another. The thresholds sit far
below the measured ratios (see README) so that only a real regression trips them, for
example losing the block decomposition, the multi-frequency pass, or the sliding update.
"""

import os
import timeit

import numpy as np
import pytest

import fastgoertzel as fg
from fastgoertzel import _core


def best_of(fn, repeat=5, number=None) -> float:
    t = timeit.Timer(fn)
    if number is None:
        number, _ = t.autorange()
    return min(t.repeat(repeat=repeat, number=number)) / number


def numpy_two_dots(x: np.ndarray, f: float) -> float:
    ang = (2.0 * np.pi * f) * np.arange(len(x))
    re = np.dot(x, np.cos(ang))
    im = -np.dot(x, np.sin(ang))
    return float(np.hypot(re, im))


def test_dispatch_selects_avx2_when_the_cpu_has_it():
    """numpy detects CPU features with CPUID at runtime (also correct under QEMU)."""
    if os.environ.get("FASTGOERTZEL_KERNEL"):
        pytest.skip("kernel forced through the environment")
    features = getattr(np._core._multiarray_umath, "__cpu_features__", None)
    if not features or "AVX2" not in features or "FMA3" not in features:
        pytest.skip("cannot determine CPU features")
    if features["AVX2"] and features["FMA3"]:
        assert _core._kernel == "avx2"
    else:
        assert _core._kernel == "baseline"


def test_single_frequency_is_much_faster_than_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1_000_000)
    f = 0.1
    t_fg = best_of(lambda: fg.goertzel(x, f))
    t_np = best_of(lambda: numpy_two_dots(x, f), repeat=3)
    # Measured 24-27x with the AVX2 kernel, ~13x with the baseline kernel.
    assert t_np / t_fg > 5.0, f"goertzel {t_fg * 1e6:.0f} us vs numpy {t_np * 1e6:.0f} us"


def test_batch_is_never_slower_than_single_calls():
    """On one thread a batch costs the same per frequency as a single call (both run eight
    chains and are compute-bound once the signal is in cache); the batch must not be slower.
    Its advantages, OpenMP and fewer passes over out-of-cache data, are machine dependent
    and are reported by the benchmark rather than asserted here."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1_000_000)
    freqs = np.linspace(0.05, 0.45, 8)
    t_batch = best_of(lambda: fg.goertzel_batch(x, freqs, threads=1))
    t_singles = best_of(lambda: [fg.goertzel(x, f) for f in freqs])
    assert t_singles / t_batch > 0.5, (
        f"batch {t_batch * 1e3:.2f} ms vs singles {t_singles * 1e3:.2f} ms"
    )


def test_sliding_update_is_independent_of_window_length():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(60_000)
    freqs = np.linspace(0.05, 0.45, 4)
    t_small = best_of(lambda: fg.goertzel_sliding_batch_complex(x, 256, freqs, threads=1))
    t_large = best_of(lambda: fg.goertzel_sliding_batch_complex(x, 4096, freqs, threads=1))
    # Recomputing each window would make the 4096 case ~16x slower than the 256 case.
    assert t_large / t_small < 3.0, f"W=256 {t_small * 1e3:.2f} ms, W=4096 {t_large * 1e3:.2f} ms"


def test_sliding_beats_numpy_windows_plus_rfft_single_threaded():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(50_000)
    window = 256
    freqs = np.round(np.linspace(0.05, 0.45, 8) * window) / window
    bins = np.rint(freqs * window).astype(int)

    def numpy_sliding():
        v = np.lib.stride_tricks.sliding_window_view(x, window)
        return np.fft.rfft(v, axis=1)[:, bins]

    t_fg = best_of(lambda: fg.goertzel_sliding_batch_complex(x, window, freqs, threads=1), repeat=3)
    t_np = best_of(numpy_sliding, repeat=3)
    # Measured ~7x (value/phase form) and more for the complex form; 1.0.x was 0.25x.
    assert t_np / t_fg > 2.0, f"sliding {t_fg * 1e3:.2f} ms vs numpy {t_np * 1e3:.2f} ms"
