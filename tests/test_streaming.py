"""GoertzelFilter: chunked streaming must equal the one-shot result."""

import numpy as np
import pytest

import fastgoertzel as fg


@pytest.mark.parametrize("chunks", [1, 2, 7, 64, 1000])
def test_chunked_push_matches_one_shot(chunks):
    rng = np.random.default_rng(chunks)
    x = rng.standard_normal(1000)
    f = 0.1234
    flt = fg.GoertzelFilter(f)
    for part in np.array_split(x, chunks):
        flt.push(part)
    assert flt.n_samples == 1000
    assert flt.result() == pytest.approx(fg.goertzel(x, f), rel=1e-12, abs=1e-12)
    assert flt.result_complex() == pytest.approx(fg.goertzel_complex(x, f), rel=1e-12)
    for scale in ("magnitude", "power"):
        assert flt.result(scale=scale) == pytest.approx(fg.goertzel(x, f, scale=scale), rel=1e-12)


def test_scalar_pushes_and_intermediate_results():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    flt = fg.GoertzelFilter(0.3)
    for i, v in enumerate(x, start=1):
        flt.push(float(v))
        assert flt.n_samples == i
        # The result after i samples is goertzel over the first i samples.
        assert flt.result_complex() == pytest.approx(fg.goertzel_complex(x[:i], 0.3), rel=1e-11)


def test_reset_and_reuse():
    flt = fg.GoertzelFilter(0.25)
    flt.push(np.ones(10))
    first = flt.result()
    flt.reset()
    assert flt.n_samples == 0
    with pytest.raises(ValueError):
        flt.result()
    flt.push(np.ones(10))
    assert flt.result() == first


def test_properties_and_fs():
    flt = fg.GoertzelFilter(440, fs=44100)
    assert flt.frequency == 440
    assert flt.fs == 44100
    assert flt.normalized_frequency == pytest.approx(440 / 44100)
    assert fg.GoertzelFilter(0.1).fs is None
    t = np.arange(4410)
    flt.push(np.cos(2 * np.pi * 440 / 44100 * t + 0.2))
    amp, phase = flt.result()
    assert amp == pytest.approx(1.0, abs=1e-9)
    assert phase == pytest.approx(0.2, abs=1e-9)
    assert "GoertzelFilter(frequency=440" in repr(flt)
    assert "n_samples=4410" in repr(flt)


def test_dc_and_nyquist_scaling_in_filter():
    flt = fg.GoertzelFilter(0.0)
    flt.push(np.full(20, 2.5))
    assert flt.result()[0] == pytest.approx(2.5)
    flt = fg.GoertzelFilter(0.5)
    flt.push(1.5 * np.cos(np.pi * np.arange(20)))
    assert flt.result()[0] == pytest.approx(1.5)


def test_accepts_lists_and_integer_arrays():
    flt = fg.GoertzelFilter(0.2)
    flt.push([1, 2, 3])
    flt.push(np.arange(4, 8))
    assert flt.n_samples == 7
    assert flt.result_complex() == pytest.approx(fg.goertzel_complex(np.arange(1, 8), 0.2))


def test_many_independent_filters_are_cheap_and_isolated():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(256)
    freqs = np.linspace(0.01, 0.49, 32)
    filters = [fg.GoertzelFilter(f) for f in freqs]
    for flt in filters:
        flt.push(x)
    got = np.array([flt.result_complex() for flt in filters])
    np.testing.assert_allclose(got, fg.goertzel_batch_complex(x, freqs), rtol=1e-11)


@pytest.mark.parametrize("chunks", [1, 3, 10])
def test_large_chunks_take_the_blocked_path_and_match(chunks):
    """Chunks of 4096 samples or more are folded in via the shift theorem; must equal one-shot."""
    rng = np.random.default_rng(300 + chunks)
    x = rng.standard_normal(100_000)
    f = 0.2718
    flt = fg.GoertzelFilter(f)
    for part in np.array_split(x, chunks):
        flt.push(part)
    assert flt.n_samples == 100_000
    assert flt.result_complex() == pytest.approx(fg.goertzel_complex(x, f), rel=1e-9)


def test_mixed_small_and_large_pushes_match_one_shot():
    rng = np.random.default_rng(301)
    x = rng.standard_normal(30_000)
    f = 0.05
    flt = fg.GoertzelFilter(f)
    cuts = [0, 1, 2, 5_000, 5_001, 9_100, 9_101, 20_000, 29_999, 30_000]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a == 1:
            flt.push(float(x[a]))
        else:
            flt.push(x[a:b])
        # Intermediate results are valid at every point, whichever path was taken.
        assert flt.result_complex() == pytest.approx(fg.goertzel_complex(x[:b], f), rel=1e-9)
    assert flt.n_samples == 30_000
    assert flt.result() == pytest.approx(fg.goertzel(x, f), rel=1e-9)
