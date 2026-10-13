"""Kernel-level properties: lane decomposition, runtime dispatch, and thread independence.

The hot loops are compiled twice (baseline and AVX2+FMA) and selected at import. Every
decomposition (blocks, lane groups, sliding chunks) must give the same answer as the
straightforward definition, and the answer must not depend on which build ran or on how
many threads were used.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import fastgoertzel as fg
from fastgoertzel import _core


def dtft(x: np.ndarray, f: float) -> complex:
    n = np.arange(len(x))
    return complex(np.sum(x * np.exp(-2j * np.pi * f * n)))


def test_kernel_name_is_reported():
    assert _core._kernel in ("avx2", "baseline")


@pytest.mark.parametrize("n", [1, 7, 511, 512, 513, 4095, 4096, 4097, 8200, 70_001])
def test_block_decomposition_matches_direct_sum(n):
    """Lengths around the block threshold (512) and multiples of eight lanes."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    for f in (0.0, 0.5, 0.1234, 0.9876):
        expected = dtft(x, f)
        assert fg.goertzel_complex(x, f) == pytest.approx(expected, rel=1e-9, abs=1e-9 * np.sqrt(n))
    if n > 1:
        # Nearest-to-DC frequency: the recurrence is worst conditioned here (documented).
        f = 1 / n
        assert fg.goertzel_complex(x, f) == pytest.approx(dtft(x, f), rel=1e-6, abs=1e-6)
    X = np.fft.fft(x)
    for k in sorted({0, 1, n // 2, n - 1} & set(range(n))):
        amp, phase = fg.goertzel_bin(x, k)
        edge = k == 0 or 2 * k == n
        tol = 1e-6 if k in (1, n - 1) else 1e-9
        assert amp == pytest.approx(abs(X[k]) * (1 if edge else 2) / n, rel=tol, abs=tol)


@pytest.mark.parametrize("m", list(range(1, 19)) + [31, 32, 33, 64])
def test_lane_grouping_for_every_frequency_count(m):
    """m frequencies are tiled into lanes of 8 with blocks filling the remainder."""
    rng = np.random.default_rng(1000 + m)
    x = rng.standard_normal(20_000)
    freqs = rng.uniform(0, 1, size=m)
    got = fg.goertzel_batch_complex(x, freqs)
    expected = np.array([dtft(x, f) for f in freqs])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-7)
    # And the serial, non-blocked single calls agree with the batch.
    singles = np.array([fg.goertzel_complex(x, f) for f in freqs])
    np.testing.assert_allclose(got, singles, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2, 3, 7])
def test_results_do_not_depend_on_thread_count(threads):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300_000)
    freqs = rng.uniform(0, 1, size=5)
    ref_batch = fg.goertzel_batch_complex(x, freqs, threads=1)
    np.testing.assert_array_equal(fg.goertzel_batch_complex(x, freqs, threads=threads), ref_batch)
    ref_slide = fg.goertzel_sliding_batch(x[:30_000], 1024, freqs, step=3, threads=1)
    np.testing.assert_array_equal(
        fg.goertzel_sliding_batch(x[:30_000], 1024, freqs, step=3, threads=threads), ref_slide
    )


class TestSlidingDFT:
    @pytest.mark.parametrize("window,step", [(64, 1), (1000, 1), (1000, 7), (4096, 1), (4096, 100)])
    def test_matches_exact_windows_including_across_chunks(self, window, step):
        """Compare against per-window exact evaluation at windows spanning several chunks."""
        rng = np.random.default_rng(window + step)
        n = window + step * 700  # ~700 windows: several 256-window chunks
        x = rng.standard_normal(n) + 3.0  # DC offset stresses the cancellation in the update
        freqs = np.array([0.0, 0.5, 0.01, 0.123456, 0.49])
        res = fg.goertzel_sliding_batch(x, window, freqs, step=step, scale="magnitude")
        n_windows = (n - window) // step + 1
        assert res.shape == (n_windows, len(freqs), 2)
        for w in (0, 1, 2, 254, 255, 256, 257, 511, 512, 513, n_windows - 2, n_windows - 1):
            seg = x[w * step : w * step + window]
            expected = np.array([dtft(seg, f) for f in freqs])
            np.testing.assert_allclose(res[w, :, 0], np.abs(expected), rtol=1e-9, atol=1e-8)
            big = np.abs(expected) > 1e-6
            d = np.angle(np.exp(1j * (res[w, big, 1] - np.angle(expected[big]))))
            assert np.all(np.abs(d) < 1e-7)

    def test_step_larger_than_window_uses_exact_path(self):
        rng = np.random.default_rng(77)
        x = rng.standard_normal(10_000)
        res = fg.goertzel_sliding_batch(x, 32, [0.1, 0.2], step=40)
        for w in range(res.shape[0]):
            seg = x[w * 40 : w * 40 + 32]
            np.testing.assert_allclose(res[w], fg.goertzel_batch(seg, [0.1, 0.2]), atol=1e-12)


def test_baseline_and_dispatched_kernels_agree():
    """Run the same computations with FASTGOERTZEL_KERNEL=baseline in a child process."""
    code = r"""
import json, numpy as np, fastgoertzel as fg
from fastgoertzel import _core
rng = np.random.default_rng(123)
x = rng.standard_normal(100_003)
freqs = rng.uniform(0, 1, size=11)
out = {
    "kernel": _core._kernel,
    "single": [fg.goertzel_complex(x, f) for f in freqs],
    "bin": [fg.goertzel_bin(x, k) for k in (0, 5, 50_001, 100_002)],
    "batch": fg.goertzel_batch_complex(x, freqs).tolist(),
    "sliding": fg.goertzel_sliding_batch(x[:20_000], 512, freqs[:3], step=2).ravel().tolist(),
}
print(json.dumps(out, default=lambda z: [z.real, z.imag]))
"""
    env = dict(os.environ, FASTGOERTZEL_KERNEL="baseline")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    base = json.loads(proc.stdout)
    assert base["kernel"] == "baseline"

    env = dict(os.environ)
    env.pop("FASTGOERTZEL_KERNEL", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    auto = json.loads(proc.stdout)
    assert auto["kernel"] in ("avx2", "baseline")  # whatever this CPU supports

    def c(v):
        return np.array(v, dtype=float).reshape(-1, 2) @ np.array([1, 1j])

    np.testing.assert_allclose(c(auto["single"]), c(base["single"]), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(np.array(auto["bin"]), np.array(base["bin"]), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(c(auto["batch"]), c(base["batch"]), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(auto["sliding"], base["sliding"], rtol=1e-9, atol=1e-7)


def test_streaming_filter_matches_blocked_one_shot():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50_000)
    flt = fg.GoertzelFilter(0.31415)
    for part in np.array_split(x, 13):
        flt.push(part)
    assert flt.result_complex() == pytest.approx(fg.goertzel_complex(x, 0.31415), rel=1e-9)
