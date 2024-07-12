import pytest

import fastgoertzel as G
import numpy as np


def wave(amp, freq, phase, x):
    return amp * np.sin(2*np.pi * freq * x + phase)


@pytest.fixture
def wave_array():
    x = np.arange(0, 512)
    return wave(1, 1/128, 0, x)


@pytest.fixture
def expected_values():
    return {
        "amp": 1.0000,
        "phase": -1.5708
    }


def test_goertzel(wave_array, expected_values):
    amp, phase = G.goertzel(wave_array, 1/128)
    assert pytest.approx(amp, rel=1e-4) == expected_values["amp"], f"Expected amplitude {expected_values['amp']}, but got {amp}"
    assert pytest.approx(phase, rel=1e-4) == expected_values["phase"], f"Expected phase {expected_values['phase']}, but got {phase}"
