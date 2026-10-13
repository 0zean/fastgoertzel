# API reference

All public names are importable from `fastgoertzel`. The signatures below are those of
the shipped type stubs (`fastgoertzel/_core.pyi`), which CI checks against the compiled
module. `ArrayLike` is anything NumPy converts to a 1-D float64 array; `str` arguments
named `scale` accept `"amplitude"`, `"magnitude"` or `"power"`.

```{eval-rst}
.. currentmodule:: fastgoertzel

One frequency
-------------

.. autofunction:: goertzel
.. autofunction:: goertzel_complex
.. autofunction:: goertzel_bin

Several frequencies
-------------------

.. autofunction:: goertzel_batch
.. autofunction:: goertzel_batch_complex

Sliding windows
---------------

.. autofunction:: goertzel_sliding_batch
.. autofunction:: goertzel_sliding_batch_complex

Streaming
---------

.. autoclass:: GoertzelFilter(frequency, *, fs=None)
   :members: push, result, result_complex, reset, n_samples, frequency, fs, normalized_frequency

Module attributes
-----------------

.. py:data:: __version__
   :type: str

   The package version.

.. py:data:: _core._has_openmp
   :type: bool

   Whether this build parallelises the batch and sliding functions with OpenMP.

.. py:data:: _core._kernel
   :type: str

   The hot-loop build selected at import, ``"avx2"`` or ``"baseline"``. The environment
   variable ``FASTGOERTZEL_KERNEL=baseline|avx2|auto`` overrides the choice.
```
