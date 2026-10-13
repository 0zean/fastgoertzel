"""Thread safety: all functions release the GIL and must be safe to call concurrently."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import fastgoertzel


def test_concurrent_calls_match_serial_results():
    rng = np.random.default_rng(11)
    signals = [rng.standard_normal(20_000) for _ in range(8)]
    freqs = np.linspace(0.01, 0.49, 6)

    def work(i: int):
        x = signals[i]
        return (
            fastgoertzel.goertzel(x, 0.1),
            fastgoertzel.goertzel_batch(x, freqs),
            fastgoertzel.goertzel_sliding_batch(x[:2_000], 128, freqs[:2]),
        )

    serial = [work(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(work, range(8)))

    for (s_single, s_batch, s_slide), (p_single, p_batch, p_slide) in zip(serial, parallel):
        assert s_single == p_single
        np.testing.assert_array_equal(s_batch, p_batch)
        np.testing.assert_array_equal(s_slide, p_slide)
