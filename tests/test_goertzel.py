"""Numerical correctness of fastgoertzel against direct DFT/DTFT sums and documented conventions."""

import numpy as np
import pytest

import fastgoertzel as fg


def wrapped(delta: np.ndarray) -> np.ndarray:
    """Absolute phase difference wrapped to [0, pi]."""
    return np.abs(np.angle(np.exp(1j * np.asarray(delta))))


def dtft(x: np.ndarray, f: float) -> complex:
    """Direct O(N) evaluation of X(f) = sum x[n] exp(-2j pi f n)."""
    n = np.arange(len(x))
    return complex(np.sum(x * np.exp(-2j * np.pi * f * n)))


def assert_rows_close(got: np.ndarray, expected: np.ndarray) -> None:
    """Compare (value, phase) rows: values to 1e-9 relative, phases where the value is not ~0."""
    np.testing.assert_allclose(got[..., 0], expected[..., 0], rtol=1e-9, atol=1e-9)
    significant = expected[..., 0] > 1e-6
    assert np.all(wrapped(got[..., 1][significant] - expected[..., 1][significant]) < 1e-7)


class TestConventions:
    def test_sine_amplitude_and_phase(self):
        t = np.arange(512)
        amp_expected, phase_expected, f = 2.5, np.pi / 4, 1 / 128
        amp, phase = fg.goertzel(amp_expected * np.sin(2 * np.pi * f * t + phase_expected), f)
        assert amp == pytest.approx(amp_expected, abs=1e-10)
        # Phase is cosine-referenced: sin(wt + p) == cos(wt + p - pi/2).
        assert phase == pytest.approx(phase_expected - np.pi / 2, abs=1e-10)

    def test_cosine_phase_is_recovered_directly(self):
        t = np.arange(512)
        amp, phase = fg.goertzel(2.5 * np.cos(2 * np.pi * t / 128 + np.pi / 4), 1 / 128)
        assert amp == pytest.approx(2.5, abs=1e-10)
        assert phase == pytest.approx(np.pi / 4, abs=1e-10)

    def test_multiple_frequencies(self):
        t = np.arange(1024)
        freqs, amps = [1 / 256, 1 / 128, 1 / 64], [1.0, 2.0, 0.5]
        x = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, freqs))
        for a, f in zip(amps, freqs):
            assert fg.goertzel(x, f)[0] == pytest.approx(a, abs=1e-9)

    def test_zero_signal(self):
        amp, phase = fg.goertzel(np.zeros(100), 0.1)
        assert amp == 0.0
        assert np.isfinite(phase)

    def test_single_sample(self):
        # One sample at DC: X = 3, amplitude scale 1/N at DC -> 3.
        amp, phase = fg.goertzel(np.array([3.0]), 0.0)
        assert amp == pytest.approx(3.0)
        assert phase == 0.0
        assert fg.goertzel(np.array([3.0]), 0.37)[0] == pytest.approx(6.0)

    def test_dc_and_nyquist_amplitude_are_true_amplitudes(self):
        assert fg.goertzel(np.ones(100), 0.0)[0] == pytest.approx(1.0)
        assert fg.goertzel(np.cos(np.pi * np.arange(100)), 0.5)[0] == pytest.approx(1.0)
        assert fg.goertzel(3 + np.sin(2 * np.pi * np.arange(100) / 10), 0.0)[0] == pytest.approx(3)
        assert fg.goertzel_bin(np.ones(100), 0)[0] == pytest.approx(1.0)
        assert fg.goertzel_bin(np.cos(np.pi * np.arange(100)), 50)[0] == pytest.approx(1.0)

    def test_frequency_is_evaluated_exactly_not_truncated_to_a_bin(self):
        # 1.x evaluated this at DC (floor(0.0099 * 100) == 0). Now it is the DTFT at 0.0099.
        x = np.sin(2 * np.pi * 0.0099 * np.arange(100))
        X = dtft(x, 0.0099)
        amp, phase = fg.goertzel(x, 0.0099)
        assert amp == pytest.approx(abs(X) * 2 / 100, rel=1e-10)
        assert phase == pytest.approx(np.angle(X), abs=1e-10)
        assert amp > 0.9  # and not the ~0.001 the DC bin reports

    def test_fs_round_trip(self):
        fs, hz, n = 44100.0, 440.0, 4410  # exactly 44 cycles
        t = np.arange(n)
        x = 0.7 * np.cos(2 * np.pi * hz / fs * t + 0.3)
        amp, phase = fg.goertzel(x, hz, fs=fs)
        assert amp == pytest.approx(0.7, abs=1e-9)
        assert phase == pytest.approx(0.3, abs=1e-9)
        assert fg.goertzel(x, hz, fs=fs) == fg.goertzel(x, hz / fs)
        # Non-integer cycle count is evaluated exactly too.
        X = dtft(x, 441.5 / fs)
        assert fg.goertzel_complex(x, 441.5, fs=fs) == pytest.approx(X, rel=1e-9)

    def test_scale_options(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        f = 0.123
        X = fg.goertzel_complex(x, f)
        amp, p1 = fg.goertzel(x, f, scale="amplitude")
        mag, p2 = fg.goertzel(x, f, scale="magnitude")
        pw, p3 = fg.goertzel(x, f, scale="power")
        assert mag == pytest.approx(abs(X))
        assert amp == pytest.approx(abs(X) * 2 / 300)
        assert pw == pytest.approx(abs(X) ** 2)
        assert p1 == p2 == p3 == pytest.approx(np.angle(X))
        # DC uses 1/N, not 2/N, for amplitude only.
        assert fg.goertzel(x, 0.0, scale="amplitude")[0] == pytest.approx(abs(x.sum()) / 300)
        assert fg.goertzel(x, 0.0, scale="magnitude")[0] == pytest.approx(abs(x.sum()))

    def test_aliasing_above_half_sample_rate(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(256)
        hi = fg.goertzel_complex(x, 0.7)
        lo = fg.goertzel_complex(x, 0.3)
        assert hi == pytest.approx(np.conj(lo), rel=1e-9)


class TestAgainstReference:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 31, 32, 100, 255, 256, 1000, 1024, 4097])
    def test_every_bin_matches_fft(self, n):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        X = np.fft.fft(x)
        expected_amp = np.abs(X) * 2 / n
        expected_amp[0] /= 2
        if n % 2 == 0:
            expected_amp[n // 2] /= 2

        # Exact frequencies k/N through the generalized path ...
        res = fg.goertzel_batch(x, np.arange(n) / n)
        np.testing.assert_allclose(res[:, 0], expected_amp, rtol=1e-9, atol=1e-9)
        nz = np.abs(X) > 1e-9
        assert np.all(wrapped(res[nz, 1] - np.angle(X[nz])) < 1e-7)
        # ... the complex variant ...
        np.testing.assert_allclose(fg.goertzel_batch_complex(x, np.arange(n) / n), X, atol=1e-9)
        # ... and the integer-bin path.
        for k in range(n):
            amp, phase = fg.goertzel_bin(x, k)
            assert amp == pytest.approx(expected_amp[k], rel=1e-9, abs=1e-9)
            if abs(X[k]) > 1e-9:
                assert wrapped(phase - np.angle(X[k])) < 1e-7

    @pytest.mark.parametrize("n", [1, 2, 10, 97, 1000, 4096])
    def test_generalized_matches_direct_dtft_at_random_frequencies(self, n):
        rng = np.random.default_rng(100 + n)
        x = rng.standard_normal(n)
        freqs = rng.uniform(0.0, 1.0, size=50)
        freqs[:3] = [0.0, 0.5, 1.0 - 1e-12]
        expected = np.array([dtft(x, f) for f in freqs])
        got = fg.goertzel_batch_complex(x, freqs)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9 * np.sqrt(n))
        for f, e in zip(freqs[:5], expected[:5]):
            assert fg.goertzel_complex(x, f) == pytest.approx(e, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("n", [5, 64, 1000, 4097])
    def test_generalized_and_bin_paths_agree_on_bins(self, n):
        rng = np.random.default_rng(200 + n)
        x = rng.standard_normal(n)
        for k in (0, 1, n // 3, n // 2, n - 1):
            a1, p1 = fg.goertzel(x, k / n)
            a2, p2 = fg.goertzel_bin(x, k)
            assert a1 == pytest.approx(a2, rel=1e-10, abs=1e-12)
            if a2 > 1e-9:
                assert wrapped(p1 - p2) < 1e-9

    def test_large_signal_is_finite(self):
        rng = np.random.default_rng(1)
        amp, phase = fg.goertzel(rng.standard_normal(100_000) * 1e6, 0.1)
        assert np.isfinite(amp) and np.isfinite(phase)

    def test_accuracy_at_large_n(self):
        n = 1_000_000
        rng = np.random.default_rng(2)
        x = rng.standard_normal(n)
        # Interior frequency: tight. Near DC: documented looser bound.
        for f, tol in ((0.1234567, 1e-8), (1 / n, 1e-3)):
            X = dtft(x, f)
            assert fg.goertzel_complex(x, f) == pytest.approx(X, rel=tol)


class TestBatchAndSliding:
    def test_batch_matches_single_calls(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(512)
        freqs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.999, 0.123456])
        res = fg.goertzel_batch(x, freqs)
        assert res.shape == (len(freqs), 2)
        for i, f in enumerate(freqs):
            np.testing.assert_allclose(res[i], fg.goertzel(x, f), rtol=0, atol=1e-12)
        for scale in ("magnitude", "power"):
            res = fg.goertzel_batch(x, freqs, scale=scale)
            for i, f in enumerate(freqs):
                np.testing.assert_allclose(res[i], fg.goertzel(x, f, scale=scale), atol=1e-9)

    def test_batch_above_parallel_threshold_and_thread_option(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(20_000)
        freqs = np.linspace(0.01, 0.49, 16)
        default = fg.goertzel_batch(x, freqs)
        serial = fg.goertzel_batch(x, freqs, threads=1)
        two = fg.goertzel_batch(x, freqs, threads=2)
        np.testing.assert_array_equal(default, serial)
        np.testing.assert_array_equal(default, two)
        # The batch and the single call use different block decompositions, so they agree
        # to rounding (relative ~N*eps), not bit for bit.
        for i, f in enumerate(freqs):
            assert_rows_close(default[i], np.array(fg.goertzel(x, f)))

    def test_batch_with_fs(self):
        x = np.cos(2 * np.pi * 1000 / 8000 * np.arange(800))
        res = fg.goertzel_batch(x, [500, 1000, 1500], fs=8000)
        assert res[1, 0] == pytest.approx(1.0, abs=1e-9)
        assert res[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_batch_empty_frequencies(self):
        assert fg.goertzel_batch(np.ones(8), np.array([])).shape == (0, 2)
        assert fg.goertzel_batch_complex(np.ones(8), np.array([])).shape == (0,)

    @pytest.mark.parametrize("window", [1, 7, 50, 200])
    def test_sliding_matches_batch_per_window(self, window):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(200)
        freqs = np.array([0.0, 0.1, 0.25, 0.49, 0.5])
        res = fg.goertzel_sliding_batch(x, window, freqs)
        n_windows = len(x) - window + 1
        assert res.shape == (n_windows, len(freqs), 2)
        for w in range(n_windows):
            assert_rows_close(res[w], fg.goertzel_batch(x[w : w + window], freqs))

    @pytest.mark.parametrize("step", [1, 2, 3, 50, 151])
    def test_sliding_step(self, step):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(200)
        freqs = np.array([0.1, 0.2])
        window = 50
        res = fg.goertzel_sliding_batch(x, window, freqs, step=step)
        expected_windows = (len(x) - window) // step + 1
        assert res.shape[0] == expected_windows
        for w in range(expected_windows):
            start = w * step
            assert_rows_close(res[w], fg.goertzel_batch(x[start : start + window], freqs))

    def test_sliding_above_parallel_threshold(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(5_000)
        freqs = np.array([0.1, 0.2])
        res = fg.goertzel_sliding_batch(x, 256, freqs)
        serial = fg.goertzel_sliding_batch(x, 256, freqs, threads=1)
        np.testing.assert_array_equal(res, serial)
        for w in (0, 1234, 4744):
            assert_rows_close(res[w], fg.goertzel_batch(x[w : w + 256], freqs))

    def test_sliding_window_equal_to_length(self):
        x = np.arange(16, dtype=float)
        res = fg.goertzel_sliding_batch(x, 16, np.array([0.25]))
        assert res.shape == (1, 1, 2)
        np.testing.assert_allclose(res[0, 0], fg.goertzel(x, 0.25), atol=1e-12)


class TestInputConversion:
    def test_accepts_list_ints_float32_and_strided(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(64)
        ref = fg.goertzel(x, 0.1)
        assert fg.goertzel(list(x), 0.1) == pytest.approx(ref)
        assert fg.goertzel(x.astype(np.float32), 0.1) == pytest.approx(ref, rel=1e-5)
        assert fg.goertzel(x[::2], 0.1) == pytest.approx(
            fg.goertzel(np.ascontiguousarray(x[::2]), 0.1)
        )
        ints = np.arange(64)
        assert fg.goertzel(ints, 0.1) == pytest.approx(fg.goertzel(ints.astype(float), 0.1))

    def test_frequencies_accept_lists_and_ints(self):
        x = np.sin(np.arange(128) * 0.3)
        assert fg.goertzel_batch(x, [0.1, 0.2]).shape == (2, 2)
        assert fg.goertzel_batch(x, [0]).shape == (1, 2)
        assert fg.goertzel(x, 0) == fg.goertzel(x, 0.0)

    def test_nan_input_propagates(self):
        amp, phase = fg.goertzel(np.array([1.0, np.nan, 2.0]), 0.0)
        assert np.isnan(amp) and np.isnan(phase)

    def test_version_is_exposed(self):
        assert isinstance(fg.__version__, str)
        assert fg.__version__ != "dev"
