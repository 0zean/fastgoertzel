"""Property-based tests: random signals, lengths, frequencies and chunkings against the
direct DTFT definition and against each other."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import fastgoertzel as fg

SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

signals = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=3000),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
frequencies = st.floats(min_value=0.0, max_value=1.0, exclude_max=True, allow_nan=False)


def dtft(x: np.ndarray, f: float) -> complex:
    n = np.arange(len(x))
    return complex(np.sum(x * np.exp(-2j * np.pi * f * n)))


def tolerance(x: np.ndarray) -> float:
    # Rounding of the recurrence grows ~linearly with N; scale the absolute tolerance by the
    # signal's l1 norm, which bounds |X|.
    return 1e-9 * (1.0 + np.sum(np.abs(x))) * max(1.0, len(x) / 1000)


@SETTINGS
@given(x=signals, f=frequencies)
def test_matches_direct_dtft(x, f):
    expected = dtft(x, f)
    got = fg.goertzel_complex(x, f)
    assert abs(got - expected) <= tolerance(x)


@SETTINGS
@given(x=signals, fs=st.lists(frequencies, min_size=0, max_size=12))
def test_batch_equals_single_calls(x, fs):
    batch = fg.goertzel_batch_complex(x, fs)
    singles = np.array([fg.goertzel_complex(x, f) for f in fs], dtype=np.complex128)
    tol = tolerance(x)
    assert batch.shape == (len(fs),)
    assert np.all(np.abs(batch - singles) <= tol)


@SETTINGS
@given(x=signals, f=frequencies, n_chunks=st.integers(min_value=1, max_value=16))
def test_streaming_equals_one_shot(x, f, n_chunks):
    flt = fg.GoertzelFilter(f)
    for part in np.array_split(x, min(n_chunks, len(x))):
        if len(part) == 1:
            flt.push(float(part[0]))
        else:
            flt.push(part)
    assert flt.n_samples == len(x)
    assert abs(flt.result_complex() - fg.goertzel_complex(x, f)) <= tolerance(x)


@SETTINGS
@given(
    x=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=1500),
        elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    ),
    f=frequencies,
    data=st.data(),
)
def test_sliding_equals_exact_windows(x, f, data):
    window = data.draw(st.integers(min_value=1, max_value=len(x)))
    step = data.draw(st.integers(min_value=1, max_value=max(1, len(x) // 2)))
    res = fg.goertzel_sliding_batch_complex(x, window, [f], step=step)
    n_windows = (len(x) - window) // step + 1
    assert res.shape == (n_windows, 1)
    tol = tolerance(x)
    for w in range(n_windows):
        seg = x[w * step : w * step + window]
        assert abs(res[w, 0] - dtft(seg, f)) <= tol


@SETTINGS
@given(x=signals, k=st.integers(min_value=0, max_value=10_000))
def test_bin_matches_fft(x, k):
    n = len(x)
    k = k % n
    X = np.fft.fft(x)[k]
    amp, phase = fg.goertzel_bin(x, k, scale="magnitude")
    assert amp == pytest.approx(abs(X), abs=tolerance(x))
    if abs(X) > 1e-6 * (1 + np.sum(np.abs(x))):
        assert abs(np.angle(np.exp(1j * (phase - np.angle(X))))) < 1e-6
