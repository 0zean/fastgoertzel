// SPDX-License-Identifier: GPL-3.0-or-later
//
// pybind11 layer for fastgoertzel.
//
// Responsibilities of this file, and only this file:
//   * argument conversion (any array-like -> contiguous float64),
//   * validation, raising ValueError *before* the GIL is released and before any
//     OpenMP region is entered (an exception escaping an OpenMP region terminates
//     the process),
//   * releasing the GIL for the numeric work,
//   * optional OpenMP parallelism for the batch functions.
// The numeric kernel lives in include/fastgoertzel/goertzel.hpp and is noexcept.

#include "fastgoertzel/goertzel.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define FASTGOERTZEL_STRINGIFY_IMPL(x) #x
#define FASTGOERTZEL_STRINGIFY(x) FASTGOERTZEL_STRINGIFY_IMPL(x)

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Below this much work (samples * frequencies) an OpenMP region costs more than it saves.
constexpr std::size_t kParallelThreshold = std::size_t{1} << 16;

// ---- validation -------------------------------------------------------------------

void require_1d(const Array& a, const char* name) {
    if (a.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be a 1-dimensional array, got " +
                              std::to_string(a.ndim()) + " dimension(s)");
    }
}

void require_frequency(double f) {
    if (!fastgoertzel::is_valid_frequency(f)) {
        throw py::value_error("Frequency must be in [0, 1) range (got " + std::to_string(f) + ")");
    }
}

void require_nonempty(std::size_t n, const char* name) {
    if (n == 0) {
        throw py::value_error(std::string(name) + " must contain at least one element");
    }
}

// Validates every frequency and precomputes the per-bin constants for a window of `n`.
std::vector<fastgoertzel::BinCoefficients> coefficients_for(const Array& frequencies,
                                                            std::size_t n) {
    require_1d(frequencies, "frequencies");
    const auto f = frequencies.unchecked<1>();
    std::vector<fastgoertzel::BinCoefficients> out;
    out.reserve(static_cast<std::size_t>(f.shape(0)));
    for (py::ssize_t i = 0; i < f.shape(0); ++i) {
        require_frequency(f(i));
        out.push_back(fastgoertzel::bin_coefficients(fastgoertzel::frequency_to_bin(f(i), n), n));
    }
    return out;
}

// ---- functions --------------------------------------------------------------------

py::tuple goertzel(const Array& x, double frequency) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_frequency(frequency);

    const auto c = fastgoertzel::bin_coefficients(fastgoertzel::frequency_to_bin(frequency, n), n);
    const double* data = x.data();
    fastgoertzel::Result r{};
    {
        py::gil_scoped_release release;
        r = fastgoertzel::goertzel_bin(data, n, c);
    }
    return py::make_tuple(r.amplitude, r.phase);
}

py::array_t<double> goertzel_batch(const Array& x, const Array& frequencies) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    const auto coeffs = coefficients_for(frequencies, n);
    const auto m = coeffs.size();

    py::array_t<double> result({static_cast<py::ssize_t>(m), py::ssize_t{2}});
    auto out = result.mutable_unchecked<2>();
    const double* data = x.data();
    const bool parallel = n * m >= kParallelThreshold;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(m); ++i) {
        const auto r = fastgoertzel::goertzel_bin(data, n, coeffs[static_cast<std::size_t>(i)]);
        out(i, 0) = r.amplitude;
        out(i, 1) = r.phase;
    }
    (void)parallel;
    return result;
}

py::array_t<double> goertzel_sliding_batch(const Array& x, std::size_t window_size,
                                           const Array& frequencies) {
    require_1d(x, "x");
    const auto total = static_cast<std::size_t>(x.shape(0));
    require_nonempty(total, "x");
    if (window_size == 0 || window_size > total) {
        throw py::value_error("window_size must be in [1, len(x)] (got " +
                              std::to_string(window_size) +
                              " for len(x) = " + std::to_string(total) + ")");
    }
    const auto coeffs = coefficients_for(frequencies, window_size);
    const auto m = coeffs.size();
    const std::size_t n_windows = total - window_size + 1;

    py::array_t<double> result(
        {static_cast<py::ssize_t>(n_windows), static_cast<py::ssize_t>(m), py::ssize_t{2}});
    auto out = result.mutable_unchecked<3>();
    const double* data = x.data();
    const bool parallel = n_windows * window_size * m >= kParallelThreshold;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (std::ptrdiff_t w = 0; w < static_cast<std::ptrdiff_t>(n_windows); ++w) {
        const double* window = data + w;
        for (std::size_t f = 0; f < m; ++f) {
            const auto r = fastgoertzel::goertzel_bin(window, window_size, coeffs[f]);
            out(w, static_cast<py::ssize_t>(f), 0) = r.amplitude;
            out(w, static_cast<py::ssize_t>(f), 1) = r.phase;
        }
    }
    (void)parallel;
    return result;
}

}  // namespace

// The module keeps no mutable global state and every function only reads its inputs, so it
// is safe under free-threaded CPython (the flag is a no-op on GIL-enabled builds).
PYBIND11_MODULE(_core, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        Fast Goertzel Algorithm Implementation
        ---------------------------------------

        High-performance Goertzel algorithm for frequency detection
        implemented in C++ with Python bindings.

        .. currentmodule:: fastgoertzel

        .. autosummary::
           :toctree: _generate

           goertzel
           goertzel_batch
           goertzel_sliding_batch
    )pbdoc";

    m.def("goertzel", &goertzel, py::arg("x"), py::arg("frequency"),
          R"pbdoc(
        Compute amplitude and phase of one DFT bin using the Goertzel algorithm.

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples. Converted to a contiguous float64 array.
        frequency : float
            Normalized frequency in cycles per sample, in ``[0, 1)``. It is quantized
            to the DFT bin ``k = floor(frequency * len(x))``.

        Returns
        -------
        tuple[float, float]
            ``(amplitude, phase)``. ``amplitude`` is ``2 * |X[k]| / len(x)`` (twice the
            true value at DC and Nyquist). ``phase`` is in radians and cosine-referenced:
            ``cos(w n + p)`` yields ``p``.

        Raises
        ------
        ValueError
            If ``x`` is not 1-dimensional or is empty, or ``frequency`` is outside ``[0, 1)``.

        Examples
        --------
        >>> import numpy as np
        >>> import fastgoertzel
        >>> t = np.arange(512)
        >>> amp, phase = fastgoertzel.goertzel(np.cos(2 * np.pi * t / 128), 1 / 128)
        >>> round(amp, 6), round(phase, 6)
        (1.0, 0.0)
        )pbdoc");

    m.def("goertzel_batch", &goertzel_batch, py::arg("x"), py::arg("frequencies"),
          R"pbdoc(
        Compute amplitude and phase for several frequencies in one call.

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples.
        frequencies : array_like
            1-dimensional sequence of normalized frequencies, each in ``[0, 1)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(frequencies), 2)``; column 0 is amplitude, column 1 phase.
            Identical to calling :func:`goertzel` once per frequency.

        Notes
        -----
        Frequencies are processed in parallel with OpenMP when the build supports it and
        ``len(x) * len(frequencies) >= 65536``. Control the thread count with
        ``OMP_NUM_THREADS``. The GIL is released during computation.
        )pbdoc");

    m.def("goertzel_sliding_batch", &goertzel_sliding_batch, py::arg("x"), py::arg("window_size"),
          py::arg("frequencies"),
          R"pbdoc(
        Compute amplitude and phase for every sliding window and every frequency.

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples.
        window_size : int
            Length of each window, in ``[1, len(x)]``. Window ``w`` covers
            ``x[w : w + window_size]``; there are ``len(x) - window_size + 1`` windows.
        frequencies : array_like
            1-dimensional sequence of normalized frequencies, each in ``[0, 1)``. They
            are quantized to bins of the *window*, ``k = floor(frequency * window_size)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_windows, len(frequencies), 2)``.

        Notes
        -----
        Each window is computed from scratch (cost ``window_size`` per window and
        frequency). Windows are processed in parallel with OpenMP when available and the
        total work is at least 65536 sample-frequency products. The GIL is released.
        )pbdoc");

#if defined(VERSION_INFO)
    m.attr("__version__") = FASTGOERTZEL_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
#if defined(_OPENMP)
    m.attr("_has_openmp") = true;
#else
    m.attr("_has_openmp") = false;
#endif
}
