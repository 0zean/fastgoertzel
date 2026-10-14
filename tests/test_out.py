"""The ``out=`` parameter: results land in the caller's array, nothing else is allocated,
and every mismatch is rejected before any computation."""

import numpy as np
import pytest

import fastgoertzel as fg

X = np.random.default_rng(3).standard_normal(5000)
F = [0.05, 0.1, 0.25, 0.5]


def calls():
    # (function, kwargs, dtype, expected shape) for the four array-returning functions.
    return [
        (fg.goertzel_batch, dict(frequencies=F), np.float64, (4, 2)),
        (fg.goertzel_batch_complex, dict(frequencies=F), np.complex128, (4,)),
        (
            fg.goertzel_sliding_batch,
            dict(window_size=256, frequencies=F, step=3),
            np.float64,
            ((5000 - 256) // 3 + 1, 4, 2),
        ),
        (
            fg.goertzel_sliding_batch_complex,
            dict(window_size=256, frequencies=F, step=3),
            np.complex128,
            ((5000 - 256) // 3 + 1, 4),
        ),
    ]


@pytest.mark.parametrize("fn, kw, dtype, shape", calls(), ids=lambda v: getattr(v, "__name__", ""))
def test_out_is_filled_and_returned(fn, kw, dtype, shape):
    expected = fn(X, **kw)
    out = np.full(shape, np.nan, dtype=dtype)
    returned = fn(X, out=out, **kw)
    assert returned is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("fn, kw, dtype, shape", calls(), ids=lambda v: getattr(v, "__name__", ""))
def test_out_reused_across_calls(fn, kw, dtype, shape):
    out = np.empty(shape, dtype=dtype)
    for seed in range(3):
        x = np.random.default_rng(seed).standard_normal(5000)
        fn(x, out=out, **kw)
        np.testing.assert_array_equal(out, fn(x, **kw))


@pytest.mark.parametrize("fn, kw, dtype, shape", calls(), ids=lambda v: getattr(v, "__name__", ""))
class TestRejected:
    def test_wrong_dtype(self, fn, kw, dtype, shape):
        other = np.float32 if dtype is np.float64 else np.complex64
        with pytest.raises(ValueError, match="out must be"):
            fn(X, out=np.empty(shape, dtype=other), **kw)

    def test_wrong_shape(self, fn, kw, dtype, shape):
        with pytest.raises(ValueError, match="shape"):
            fn(X, out=np.empty(tuple(s + 1 for s in shape), dtype=dtype), **kw)
        with pytest.raises(ValueError, match="shape"):
            fn(X, out=np.empty(shape + (1,), dtype=dtype), **kw)

    def test_not_contiguous(self, fn, kw, dtype, shape):
        big = np.empty(tuple(2 * s for s in shape), dtype=dtype)
        view = big[tuple(slice(None, None, 2) for _ in shape)]
        assert view.shape == shape and not view.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            fn(X, out=view, **kw)

    def test_read_only(self, fn, kw, dtype, shape):
        out = np.empty(shape, dtype=dtype)
        out.flags.writeable = False
        with pytest.raises(ValueError, match="read-only"):
            fn(X, out=out, **kw)

    def test_not_an_array(self, fn, kw, dtype, shape):
        with pytest.raises(TypeError, match="ndarray"):
            fn(X, out=[0.0] * int(np.prod(shape)), **kw)


def test_out_overlapping_x_is_rejected():
    x = np.zeros(64)
    freqs = [0.1, 0.2]
    # A float64 view of x's own buffer with the result shape.
    out = x[:4].reshape(2, 2)
    with pytest.raises(ValueError, match="share memory"):
        fg.goertzel_batch(x, freqs, out=out)
    # Same buffer seen as complex128.
    with pytest.raises(ValueError, match="share memory"):
        fg.goertzel_batch_complex(x, freqs, out=x[:4].view(np.complex128))


def test_out_from_a_converted_x_is_fine():
    # x is float32, so the kernel reads a converted copy; the user's float64 out is unrelated.
    x = np.random.default_rng(0).standard_normal(300).astype(np.float32)
    out = np.empty((2, 2))
    fg.goertzel_batch(x, [0.1, 0.2], out=out)
    np.testing.assert_allclose(out, fg.goertzel_batch(x.astype(np.float64), [0.1, 0.2]))
