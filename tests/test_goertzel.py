"""Numerical correctness of fastgoertzel against NumPy's FFT and known conventions."""

import numpy as np
import pytest

import fastgoertzel


def wave(amp: float, freq: float, phase: float, x: np.ndarray) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * x + phase)


def wrapped(delta: np.ndarray) -> np.ndarray:
    """Absolute phase difference wrapped to [0, pi]."""
    return np.abs(np.angle(np.exp(1j * delta)))


def exact_bin_frequency(k: int, n: int) -> float:
    """A frequency that floor(f * n) maps to bin k regardless of float rounding."""
    return (k + 0.5) / n


class TestConventions:
    def test_sine_amplitude_and_phase(self):
        n_samples = 512
        frequency = 1 / 128
        t = np.arange(n_samples)
        amplitude_expected = 2.5
        phase_expected = np.pi / 4

        signal = wave(amp=amplitude_expected, freq=frequency, phase=phase_expected, x=t)
        amp, phase = fastgoertzel.goertzel(signal, frequency)

        assert abs(amp - amplitude_expected) < 1e-10
        # Phase is cosine-referenced: sin(wt + p) == cos(wt + p - pi/2).
        assert abs(phase - (phase_expected - np.pi / 2)) < 1e-10

    def test_cosine_phase_is_recovered_directly(self):
        t = np.arange(512)
        phase_expected = np.pi / 4
        amp, phase = fastgoertzel.goertzel(
            2.5 * np.cos(2 * np.pi * t / 128 + phase_expected), 1 / 128
        )
        assert abs(amp - 2.5) < 1e-10
        assert abs(phase - phase_expected) < 1e-10

    def test_multiple_frequencies(self):
        n_samples = 1024
        t = np.arange(n_samples)
        freqs = [1 / 256, 1 / 128, 1 / 64]
        amps = [1.0, 2.0, 0.5]
        signal = sum(wave(amp=a, freq=f, phase=0, x=t) for a, f in zip(amps, freqs))

        for expected_amp, freq in zip(amps, freqs):
            amp, _ = fastgoertzel.goertzel(signal, freq)
            assert abs(amp - expected_amp) < 1e-9

    def test_zero_signal(self):
        amp, phase = fastgoertzel.goertzel(np.zeros(100), 0.1)
        assert amp == 0.0
        assert np.isfinite(phase)

    def test_single_sample(self):
        amp, phase = fastgoertzel.goertzel(np.array([3.0]), 0.0)
        # One sample: X[0] = 3, amplitude = 2 * 3 / 1.
        assert amp == pytest.approx(6.0)
        assert phase == 0.0

    @pytest.mark.xfail(strict=True, reason="DC/Nyquist read 2x the true amplitude until 2.0")
    def test_dc_amplitude_is_true_amplitude(self):
        amp, _ = fastgoertzel.goertzel(np.ones(100), 0.0)
        assert amp == pytest.approx(1.0)

    @pytest.mark.xfail(strict=True, reason="DC/Nyquist read 2x the true amplitude until 2.0")
    def test_nyquist_amplitude_is_true_amplitude(self):
        amp, _ = fastgoertzel.goertzel(np.cos(np.pi * np.arange(100)), 0.5)
        assert amp == pytest.approx(1.0)

    def test_frequency_is_truncated_to_a_bin(self):
        # Documented 1.x behaviour: 0.0099 * 100 -> bin 0 (DC), not the DTFT at 0.0099.
        t = np.arange(100)
        amp_dc, _ = fastgoertzel.goertzel(np.sin(2 * np.pi * 0.0099 * t), 0.0099)
        amp_bin0, _ = fastgoertzel.goertzel(np.sin(2 * np.pi * 0.0099 * t), 0.0)
        assert amp_dc == pytest.approx(amp_bin0)


class TestAgainstNumpyFFT:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 31, 32, 100, 255, 256, 1000, 1024, 4097])
    def test_every_bin_matches_fft(self, n):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        X = np.fft.fft(x)
        expected_amp = np.abs(X) / (n / 2)
        expected_phase = np.angle(X)

        freqs = np.array([exact_bin_frequency(k, n) for k in range(n)])
        result = fastgoertzel.goertzel_batch(x, freqs)

        assert result.shape == (n, 2)
        np.testing.assert_allclose(result[:, 0], expected_amp, rtol=1e-9, atol=1e-9)
        nonzero = expected_amp > 1e-9
        assert np.all(wrapped(result[nonzero, 1] - expected_phase[nonzero]) < 1e-7)

    def test_single_call_matches_fft(self):
        n = 1000
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        X = np.fft.fft(x)
        for k in (0, 1, 17, 250, 499, 500, 777, 999):
            amp, phase = fastgoertzel.goertzel(x, exact_bin_frequency(k, n))
            assert amp == pytest.approx(abs(X[k]) / (n / 2), rel=1e-9, abs=1e-9)
            assert wrapped(phase - np.angle(X[k])) < 1e-7

    def test_large_signal_is_finite(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(100_000) * 1e6
        amp, phase = fastgoertzel.goertzel(signal, 0.1)
        assert np.isfinite(amp)
        assert np.isfinite(phase)

    def test_accuracy_near_dc_at_large_n_is_bounded(self):
        # Goertzel is worst conditioned near DC; document the bound rather than hide it.
        n = 1_000_000
        rng = np.random.default_rng(2)
        x = rng.standard_normal(n)
        k = 1
        X = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n) / n))
        amp, _ = fastgoertzel.goertzel(x, exact_bin_frequency(k, n))
        assert abs(amp - abs(X) / (n / 2)) / (abs(X) / (n / 2)) < 1e-3


class TestBatchAndSliding:
    def test_batch_matches_single_calls(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(512)
        frequencies = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.999])
        results = fastgoertzel.goertzel_batch(signal, frequencies)
        assert results.shape == (len(frequencies), 2)
        for i, freq in enumerate(frequencies):
            amp, phase = fastgoertzel.goertzel(signal, freq)
            assert results[i, 0] == pytest.approx(amp, abs=1e-12)
            assert results[i, 1] == pytest.approx(phase, abs=1e-12)

    def test_batch_above_parallel_threshold_matches_single_calls(self):
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(20_000)
        frequencies = np.linspace(0.01, 0.49, 16)
        results = fastgoertzel.goertzel_batch(signal, frequencies)
        for i, freq in enumerate(frequencies):
            amp, phase = fastgoertzel.goertzel(signal, freq)
            assert results[i, 0] == pytest.approx(amp, abs=1e-12)
            assert results[i, 1] == pytest.approx(phase, abs=1e-12)

    def test_batch_empty_frequencies(self):
        assert fastgoertzel.goertzel_batch(np.ones(8), np.array([])).shape == (0, 2)

    @pytest.mark.parametrize("window", [1, 7, 50, 200])
    def test_sliding_matches_batch_per_window(self, window):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(200)
        freqs = np.array([0.0, 0.1, 0.25, 0.49])
        result = fastgoertzel.goertzel_sliding_batch(x, window, freqs)
        n_windows = len(x) - window + 1
        assert result.shape == (n_windows, len(freqs), 2)
        for w in range(n_windows):
            expected = fastgoertzel.goertzel_batch(x[w : w + window], freqs)
            np.testing.assert_allclose(result[w], expected, rtol=0, atol=1e-12)

    def test_sliding_above_parallel_threshold(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(5_000)
        freqs = np.array([0.1, 0.2])
        result = fastgoertzel.goertzel_sliding_batch(x, 256, freqs)
        for w in (0, 1234, 4744):
            expected = fastgoertzel.goertzel_batch(x[w : w + 256], freqs)
            np.testing.assert_allclose(result[w], expected, rtol=0, atol=1e-12)

    def test_sliding_window_equal_to_length(self):
        x = np.arange(16, dtype=float)
        result = fastgoertzel.goertzel_sliding_batch(x, 16, np.array([0.25]))
        assert result.shape == (1, 1, 2)
        np.testing.assert_allclose(result[0, 0], fastgoertzel.goertzel(x, 0.25), atol=1e-12)


class TestInputConversion:
    def test_accepts_list_ints_float32_and_strided(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(64)
        ref = fastgoertzel.goertzel(x, 0.1)
        assert fastgoertzel.goertzel(list(x), 0.1) == pytest.approx(ref)
        assert fastgoertzel.goertzel(x.astype(np.float32), 0.1) == pytest.approx(ref, rel=1e-5)
        assert fastgoertzel.goertzel(x[::2], 0.1) == pytest.approx(
            fastgoertzel.goertzel(np.ascontiguousarray(x[::2]), 0.1)
        )
        ints = np.arange(64)
        assert fastgoertzel.goertzel(ints, 0.1) == pytest.approx(
            fastgoertzel.goertzel(ints.astype(float), 0.1)
        )

    def test_batch_accepts_list_of_frequencies(self):
        x = np.sin(np.arange(128) * 0.3)
        out = fastgoertzel.goertzel_batch(x, [0.1, 0.2])
        assert out.shape == (2, 2)

    def test_nan_input_propagates(self):
        amp, phase = fastgoertzel.goertzel(np.array([1.0, np.nan, 2.0]), 0.0)
        assert np.isnan(amp)
        assert np.isnan(phase)

    def test_version_is_exposed(self):
        assert isinstance(fastgoertzel.__version__, str)
        assert fastgoertzel.__version__ != "dev"
