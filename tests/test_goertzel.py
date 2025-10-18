import fastgoertzel
import numpy as np
import pytest


def wave(amp: float, freq: float, phase: float, x: np.ndarray) -> np.ndarray:
    signal = amp * np.sin(2 * np.pi * freq * x + phase)
    return signal


class TestGoertzel:
    def test_pure_sine_wave(self):
        """Test detection of a pure sine wave."""
        # Generate test signal
        n_samples = 512
        frequency = 1 / 128  # Normalized frequency
        t = np.arange(n_samples)
        amplitude_expected = 2.5
        phase_expected = np.pi / 4

        signal = wave(amp=amplitude_expected, freq=frequency, phase=phase_expected, x=t)

        # Run Goertzel algorithm
        amp, phase = fastgoertzel.goertzel(signal, frequency)

        # Check results (with tolerance for floating-point)
        assert np.abs(amp - amplitude_expected) < 1e-10
        assert phase - phase_expected < 1e-10

    def test_multiple_frequencies(self):
        """Test signal with multiple frequency components."""
        n_samples = 1024
        t = np.arange(n_samples)

        # Create composite signal
        freqs = [1 / 256, 1 / 128, 1 / 64]
        amps = [1.0, 2.0, 0.5]
        signal = sum([wave(amp=a, freq=f, phase=0, x=t) for a, f in zip(amps, freqs)])

        # Test each frequency
        for expected_amp, freq in zip(amps, freqs):
            amp, _ = fastgoertzel.goertzel(np.asarray(signal, dtype=np.float64), freq)
            assert np.abs(amp - expected_amp) < 0.1

    def test_batch_processing(self):
        """Test batch frequency processing."""
        n_samples = 512
        signal = np.random.randn(n_samples)
        frequencies = np.array([0.1, 0.2, 0.3, 0.4])

        # Batch processing
        results = fastgoertzel.goertzel_batch(signal, frequencies)

        # Verify against individual calls
        for i, freq in enumerate(frequencies):
            amp, phase = fastgoertzel.goertzel(signal, freq)
            assert np.abs(results[i, 0] - amp) < 1e-10
            assert np.abs(results[i, 1] - phase) < 1e-10

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zeros
        signal = np.zeros(100)
        amp, phase = fastgoertzel.goertzel(signal, 0.1)
        assert amp == 0.0

        # Test with single sample (should work)
        signal = np.array([1.0])
        amp, phase = fastgoertzel.goertzel(signal, 0.0)

        # Test invalid frequency
        with pytest.raises(ValueError):
            fastgoertzel.goertzel(np.array([1, 2, 3]), 1.5)

    def test_numerical_stability(self):
        """Test numerical stability with large signals."""
        signal = np.random.randn(100000) * 1e6
        amp, phase = fastgoertzel.goertzel(signal, 0.1)
        assert np.isfinite(amp)
        assert np.isfinite(phase)
