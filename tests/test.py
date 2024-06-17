import pytest

import fastgoertzel as G
import numpy as np
import pandas as pd


def wave(amp, freq, phase, x):
    return amp * np.sin(2*np.pi * freq * x + phase)


x = np.arange(0, 512)
y = wave(1, 1/128, 0, x)

amp, phase = G.goertzel(y, 1/128)
print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')

# Compared to max amplitude FFT output 
ft = np.fft.fft(y)
FFT = pd.DataFrame()
FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
FFT['phase'] = np.arctan2(ft.imag, ft.real)

max_ = FFT.iloc[FFT['amp'].idxmax()]
print(f'FFT amp: {max_["amp"]:.4f}, '
        f'phase: {max_["phase"]:.4f}, '
        f'freq: {max_["freq"]:.4f}')


@pytest.fixture
def large_wave_array():
    x = np.arange(0, 512)
    y = wave(1, 1/128, 0, x)
    return y


@pytest.fixture
def expected_values():
    return {
        "amp": 1.0000,
        "phase": -1.5708
    }


def test_goertzel(large_wave_array, expected_values):
    amp, phase = G.goertzel(large_wave_array, 1/128)
    assert pytest.approx(amp, rel=1e-4) == expected_values["amp"], f"Expected amplitude {expected_values['amp']}, but got {amp}"
    assert pytest.approx(phase, rel=1e-4) == expected_values["phase"], f"Expected phase {expected_values['phase']}, but got {phase}"
