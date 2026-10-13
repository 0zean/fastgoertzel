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

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define FASTGOERTZEL_STRINGIFY_IMPL(x) #x
#define FASTGOERTZEL_STRINGIFY(x) FASTGOERTZEL_STRINGIFY_IMPL(x)

namespace py = pybind11;
namespace fgz = fastgoertzel;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using OptDouble = std::optional<double>;
using OptInt = std::optional<int>;

// Below this much work (samples * frequencies) an OpenMP region costs more than it saves.
constexpr std::size_t kParallelThreshold = std::size_t{1} << 16;

// ---- scale ------------------------------------------------------------------------

enum class Scale { Amplitude, Magnitude, Power };

Scale parse_scale(const std::string& s) {
    if (s == "amplitude") return Scale::Amplitude;
    if (s == "magnitude") return Scale::Magnitude;
    if (s == "power") return Scale::Power;
    throw py::value_error("scale must be 'amplitude', 'magnitude' or 'power' (got '" + s + "')");
}

// Value reported for X given the scale; `dc_or_nyquist` only matters for Amplitude.
double scaled(fgz::Complex X, std::size_t n, bool dc_or_nyquist, Scale scale) noexcept {
    switch (scale) {
        case Scale::Amplitude:
            return fgz::magnitude(X) * fgz::amplitude_scale(n, dc_or_nyquist);
        case Scale::Magnitude:
            return fgz::magnitude(X);
        case Scale::Power:
            return fgz::power(X);
    }
    return 0.0;  // unreachable
}

// ---- validation -------------------------------------------------------------------

void require_1d(const Array& a, const char* name) {
    if (a.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be a 1-dimensional array, got " +
                              std::to_string(a.ndim()) + " dimension(s)");
    }
}

void require_nonempty(std::size_t n, const char* name) {
    if (n == 0) {
        throw py::value_error(std::string(name) + " must contain at least one element");
    }
}

void require_fs(const OptDouble& fs) {
    if (fs && !(std::isfinite(*fs) && *fs > 0.0)) {
        throw py::value_error("fs must be a positive finite number (got " + std::to_string(*fs) +
                              ")");
    }
}

// Hz (when fs is given) or cycles/sample -> validated cycles/sample.
double resolve_frequency(double frequency, const OptDouble& fs) {
    if (!std::isfinite(frequency)) {
        throw py::value_error("frequency must be finite (got " + std::to_string(frequency) + ")");
    }
    const double f = fs ? frequency / *fs : frequency;
    if (!fgz::is_valid_frequency(f)) {
        if (fs) {
            throw py::value_error("frequency must be in [0, fs) (got " +
                                  std::to_string(frequency) + " with fs = " +
                                  std::to_string(*fs) + ")");
        }
        throw py::value_error("frequency must be in [0, 1) (got " + std::to_string(frequency) +
                              ")");
    }
    return f;
}

std::vector<double> resolve_frequencies(const Array& frequencies, const OptDouble& fs) {
    require_1d(frequencies, "frequencies");
    const auto f = frequencies.unchecked<1>();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(f.shape(0)));
    for (py::ssize_t i = 0; i < f.shape(0); ++i) {
        out.push_back(resolve_frequency(f(i), fs));
    }
    return out;
}

std::vector<fgz::Coefficients> coefficients_for(const std::vector<double>& freqs) {
    std::vector<fgz::Coefficients> out;
    out.reserve(freqs.size());
    for (double f : freqs) out.push_back(fgz::coefficients(fgz::omega_from_frequency(f)));
    return out;
}

// threads=None -> 0 (OpenMP default); otherwise a positive count.
int resolve_threads(const OptInt& threads) {
    if (!threads) return 0;
    if (*threads < 1) {
        throw py::value_error("threads must be a positive integer (got " +
                              std::to_string(*threads) + ")");
    }
    return *threads;
}

#if defined(_OPENMP)
int omp_threads(int requested) noexcept { return requested > 0 ? requested : omp_get_max_threads(); }
#endif

py::ssize_t checked_ssize(std::size_t v) { return static_cast<py::ssize_t>(v); }

// ---- one frequency ----------------------------------------------------------------

fgz::Complex compute_dtft(const Array& x, double f) {
    const double* data = x.data();
    const auto n = static_cast<std::size_t>(x.shape(0));
    py::gil_scoped_release release;
    return fgz::dtft(data, n, f);
}

py::tuple goertzel(const Array& x, double frequency, const OptDouble& fs,
                   const std::string& scale) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const Scale sc = parse_scale(scale);
    const double f = resolve_frequency(frequency, fs);
    const fgz::Complex X = compute_dtft(x, f);
    return py::make_tuple(scaled(X, n, fgz::is_dc_or_nyquist(f), sc), fgz::phase(X));
}

std::complex<double> goertzel_complex(const Array& x, double frequency, const OptDouble& fs) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const double f = resolve_frequency(frequency, fs);
    const fgz::Complex X = compute_dtft(x, f);
    return {X.re, X.im};
}

py::tuple goertzel_bin(const Array& x, py::ssize_t k, const std::string& scale) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    const Scale sc = parse_scale(scale);
    if (k < 0 || static_cast<std::size_t>(k) >= n) {
        throw py::value_error("k must be in [0, len(x)) (got " + std::to_string(k) +
                              " for len(x) = " + std::to_string(n) + ")");
    }
    const auto kk = static_cast<std::size_t>(k);
    const double* data = x.data();
    fgz::Complex X{};
    {
        py::gil_scoped_release release;
        X = fgz::dft_bin(data, n, kk);
    }
    const bool edge = (kk == 0) || (2 * kk == n);
    return py::make_tuple(scaled(X, n, edge, sc), fgz::phase(X));
}

// ---- several frequencies ----------------------------------------------------------

py::array_t<double> goertzel_batch(const Array& x, const Array& frequencies, const OptDouble& fs,
                                   const std::string& scale, const OptInt& threads) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const Scale sc = parse_scale(scale);
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::size_t m = freqs.size();

    py::array_t<double> result({checked_ssize(m), py::ssize_t{2}});
    auto out = result.mutable_unchecked<2>();
    const double* data = x.data();
    const bool parallel = n * m >= kParallelThreshold;
    (void)parallel;
    (void)nthreads;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel) num_threads(omp_threads(nthreads))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(m); ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const fgz::Complex X =
            fgz::finish_general(fgz::recurrence(data, n, coeffs[idx].coeff), n, coeffs[idx],
                                freqs[idx]);
        out(i, 0) = scaled(X, n, fgz::is_dc_or_nyquist(freqs[idx]), sc);
        out(i, 1) = fgz::phase(X);
    }
    return result;
}

py::array_t<std::complex<double>> goertzel_batch_complex(const Array& x, const Array& frequencies,
                                                         const OptDouble& fs,
                                                         const OptInt& threads) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::size_t m = freqs.size();

    py::array_t<std::complex<double>> result(checked_ssize(m));
    auto out = result.mutable_unchecked<1>();
    const double* data = x.data();
    const bool parallel = n * m >= kParallelThreshold;
    (void)parallel;
    (void)nthreads;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel) num_threads(omp_threads(nthreads))
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(m); ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const fgz::Complex X =
            fgz::finish_general(fgz::recurrence(data, n, coeffs[idx].coeff), n, coeffs[idx],
                                freqs[idx]);
        out(i) = std::complex<double>(X.re, X.im);
    }
    return result;
}

py::array_t<double> goertzel_sliding_batch(const Array& x, py::ssize_t window_size,
                                           const Array& frequencies, py::ssize_t step,
                                           const OptDouble& fs, const std::string& scale,
                                           const OptInt& threads) {
    require_1d(x, "x");
    const auto total = static_cast<std::size_t>(x.shape(0));
    require_nonempty(total, "x");
    if (window_size < 1 || static_cast<std::size_t>(window_size) > total) {
        throw py::value_error("window_size must be in [1, len(x)] (got " +
                              std::to_string(window_size) +
                              " for len(x) = " + std::to_string(total) + ")");
    }
    if (step < 1) {
        throw py::value_error("step must be a positive integer (got " + std::to_string(step) + ")");
    }
    require_fs(fs);
    const Scale sc = parse_scale(scale);
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::size_t m = freqs.size();
    const auto w = static_cast<std::size_t>(window_size);
    const auto st = static_cast<std::size_t>(step);
    const std::size_t n_windows = (total - w) / st + 1;

    std::vector<bool> edge(m);
    for (std::size_t i = 0; i < m; ++i) edge[i] = fgz::is_dc_or_nyquist(freqs[i]);

    py::array_t<double> result({checked_ssize(n_windows), checked_ssize(m), py::ssize_t{2}});
    auto out = result.mutable_unchecked<3>();
    const double* data = x.data();
    const bool parallel = n_windows * w * m >= kParallelThreshold;
    (void)parallel;
    (void)nthreads;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel) num_threads(omp_threads(nthreads))
#endif
    for (std::ptrdiff_t wi = 0; wi < static_cast<std::ptrdiff_t>(n_windows); ++wi) {
        const double* window = data + static_cast<std::size_t>(wi) * st;
        for (std::size_t fi = 0; fi < m; ++fi) {
            const fgz::Complex X = fgz::finish_general(
                fgz::recurrence(window, w, coeffs[fi].coeff), w, coeffs[fi], freqs[fi]);
            out(wi, static_cast<py::ssize_t>(fi), 0) = scaled(X, w, edge[fi], sc);
            out(wi, static_cast<py::ssize_t>(fi), 1) = fgz::phase(X);
        }
    }
    return result;
}

// ---- streaming filter -------------------------------------------------------------

struct PyFilter {
    fgz::StreamingGoertzel core;
    double frequency_as_given;
    OptDouble fs;

    PyFilter(double frequency, OptDouble sample_rate)
        : core(resolve_frequency(frequency, (require_fs(sample_rate), sample_rate))),
          frequency_as_given(frequency),
          fs(sample_rate) {}

    void push(const Array& x) {
        if (x.ndim() > 1) {
            throw py::value_error("x must be a scalar or a 1-dimensional array, got " +
                                  std::to_string(x.ndim()) + " dimension(s)");
        }
        const auto n = static_cast<std::size_t>(x.size());
        const double* data = x.data();
        py::gil_scoped_release release;
        core.push(data, n);
    }

    fgz::Complex result_or_throw() const {
        if (core.count() == 0) {
            throw py::value_error("no samples have been pushed");
        }
        return core.result();
    }

    py::tuple result(const std::string& scale) const {
        const Scale sc = parse_scale(scale);
        const fgz::Complex X = result_or_throw();
        return py::make_tuple(
            scaled(X, core.count(), fgz::is_dc_or_nyquist(core.frequency()), sc), fgz::phase(X));
    }

    std::complex<double> result_complex() const {
        const fgz::Complex X = result_or_throw();
        return {X.re, X.im};
    }
};

}  // namespace

// The module keeps no mutable global state and every function only reads its inputs, so it
// is safe under free-threaded CPython (the flag is a no-op on GIL-enabled builds).
PYBIND11_MODULE(_core, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        Fast Goertzel Algorithm Implementation
        ---------------------------------------

        Evaluates the discrete-time Fourier transform of a real signal at chosen
        frequencies with the (generalized) Goertzel algorithm, in C++.

        .. currentmodule:: fastgoertzel

        .. autosummary::
           :toctree: _generate

           goertzel
           goertzel_complex
           goertzel_bin
           goertzel_batch
           goertzel_batch_complex
           goertzel_sliding_batch
           GoertzelFilter
    )pbdoc";

    m.def("goertzel", &goertzel, py::arg("x"), py::arg("frequency"), py::kw_only(),
          py::arg("fs") = py::none(), py::arg("scale") = "amplitude",
          R"pbdoc(
        Amplitude and phase of ``x`` at exactly ``frequency`` (generalized Goertzel).

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples. Converted to a contiguous float64 array.
        frequency : float
            Frequency to evaluate. In cycles per sample, ``[0, 1)``, unless ``fs`` is
            given, in which case it is in the same unit as ``fs`` and must be in
            ``[0, fs)``. Any value is evaluated exactly; nothing is rounded to a DFT bin.
            Values above half the sample rate alias onto their mirror image.
        fs : float, optional
            Sample rate. ``frequency / fs`` is the normalized frequency.
        scale : {'amplitude', 'magnitude', 'power'}
            ``'amplitude'`` (default) returns ``2|X|/N``, the amplitude of a sinusoid at
            that frequency (``|X|/N`` at DC and Nyquist). ``'magnitude'`` returns ``|X|``
            and ``'power'`` returns ``|X|**2``, both without normalization.

        Returns
        -------
        tuple[float, float]
            ``(value, phase)``. ``phase`` is in radians, cosine-referenced as in
            ``numpy.fft``: ``cos(2*pi*f*n + p)`` reports ``p``.

        Raises
        ------
        ValueError
            Empty or non-1-D ``x``; ``frequency`` outside its range or not finite;
            ``fs`` not positive and finite; unknown ``scale``.

        Examples
        --------
        >>> import numpy as np, fastgoertzel
        >>> t = np.arange(4096)
        >>> x = np.cos(2 * np.pi * 440 / 44100 * t + 0.25)
        >>> amp, phase = fastgoertzel.goertzel(x, 440, fs=44100)
        >>> round(amp, 3), round(phase, 3)
        (1.0, 0.25)
        )pbdoc");

    m.def("goertzel_complex", &goertzel_complex, py::arg("x"), py::arg("frequency"),
          py::kw_only(), py::arg("fs") = py::none(),
          R"pbdoc(
        Unscaled complex DTFT value ``X = sum x[n] exp(-2j*pi*f*n)`` at ``frequency``.

        Same arguments and validation as :func:`goertzel`. Use this when you need the
        complex value or want to avoid the ``sqrt``/``atan2`` of the amplitude form.
        )pbdoc");

    m.def("goertzel_bin", &goertzel_bin, py::arg("x"), py::arg("k"), py::kw_only(),
          py::arg("scale") = "amplitude",
          R"pbdoc(
        Amplitude and phase of DFT bin ``k`` of ``x`` (``k`` in ``[0, len(x))``).

        Equivalent to ``goertzel(x, k / len(x))`` but uses the integer-bin form of the
        final step, exactly as ``numpy.fft.fft(x)[k]``. This is what versions 1.x
        computed for every call after truncating ``frequency * len(x)`` to an integer.
        )pbdoc");

    m.def("goertzel_batch", &goertzel_batch, py::arg("x"), py::arg("frequencies"), py::kw_only(),
          py::arg("fs") = py::none(), py::arg("scale") = "amplitude",
          py::arg("threads") = py::none(),
          R"pbdoc(
        :func:`goertzel` for every entry of ``frequencies`` in one call.

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples.
        frequencies : array_like
            1-dimensional sequence of frequencies (cycles per sample, or in the unit of
            ``fs`` when given).
        fs, scale
            As in :func:`goertzel`.
        threads : int, optional
            OpenMP thread count for this call. ``None`` uses the OpenMP default
            (``OMP_NUM_THREADS``); ``1`` forces serial execution. Ignored in builds
            without OpenMP.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(frequencies), 2)``: column 0 is the scaled value, column 1 the
            phase. Row ``i`` equals ``goertzel(x, frequencies[i], ...)``.

        Notes
        -----
        The parallel region is entered only when ``len(x) * len(frequencies) >= 65536``.
        The GIL is released during computation.
        )pbdoc");

    m.def("goertzel_batch_complex", &goertzel_batch_complex, py::arg("x"), py::arg("frequencies"),
          py::kw_only(), py::arg("fs") = py::none(), py::arg("threads") = py::none(),
          R"pbdoc(
        :func:`goertzel_complex` for every entry of ``frequencies``; returns a complex128
        array of shape ``(len(frequencies),)``.
        )pbdoc");

    m.def("goertzel_sliding_batch", &goertzel_sliding_batch, py::arg("x"), py::arg("window_size"),
          py::arg("frequencies"), py::kw_only(), py::arg("step") = 1, py::arg("fs") = py::none(),
          py::arg("scale") = "amplitude", py::arg("threads") = py::none(),
          R"pbdoc(
        :func:`goertzel_batch` over sliding windows of ``x``.

        Parameters
        ----------
        x : array_like
            1-dimensional sequence of samples.
        window_size : int
            Length of each window, in ``[1, len(x)]``.
        frequencies : array_like
            1-dimensional sequence of frequencies.
        step : int
            Hop between consecutive windows (default 1). Window ``w`` covers
            ``x[w*step : w*step + window_size]`` and there are
            ``(len(x) - window_size) // step + 1`` windows.
        fs, scale, threads
            As in :func:`goertzel_batch`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_windows, len(frequencies), 2)``.

        Notes
        -----
        Each window is evaluated from scratch (cost ``window_size`` per window and
        frequency). Windows are distributed over OpenMP threads when the total work is at
        least 65536 sample-frequency products. The GIL is released.
        )pbdoc");

    py::class_<PyFilter>(m, "GoertzelFilter",
                         R"pbdoc(
        Stateful Goertzel filter for streaming input.

        Create it for one frequency, ``push`` samples as they arrive (scalars or arrays,
        any chunking), and read ``result()`` at any time; it equals ``goertzel`` applied
        to everything pushed so far. ``reset()`` clears the state. Instances are cheap and
        hold no buffers. Do not ``push`` to the same instance from two threads at once.

        Examples
        --------
        >>> import numpy as np, fastgoertzel
        >>> flt = fastgoertzel.GoertzelFilter(440, fs=44100)
        >>> for chunk in np.array_split(np.cos(2 * np.pi * 440 / 44100 * np.arange(4410)), 7):
        ...     flt.push(chunk)
        >>> amp, phase = flt.result()
        >>> round(amp, 6), flt.n_samples
        (1.0, 4410)
        )pbdoc")
        .def(py::init<double, OptDouble>(), py::arg("frequency"), py::kw_only(),
             py::arg("fs") = py::none())
        .def("push", &PyFilter::push, py::arg("x"),
             "Feed a scalar or a 1-dimensional array of samples.")
        .def("result", &PyFilter::result, py::kw_only(), py::arg("scale") = "amplitude",
             "``(value, phase)`` over all samples pushed so far; see :func:`goertzel` for "
             "``scale``. Raises ``ValueError`` before the first push.")
        .def("result_complex", &PyFilter::result_complex,
             "Unscaled complex DTFT value over all samples pushed so far.")
        .def("reset", [](PyFilter& self) { self.core.reset(); }, "Forget all pushed samples.")
        .def_property_readonly(
            "n_samples", [](const PyFilter& self) { return self.core.count(); },
            "Number of samples pushed since construction or the last reset.")
        .def_property_readonly(
            "frequency", [](const PyFilter& self) { return self.frequency_as_given; },
            "Frequency as passed to the constructor.")
        .def_property_readonly(
            "fs", [](const PyFilter& self) { return self.fs; }, "Sample rate, or ``None``.")
        .def_property_readonly(
            "normalized_frequency", [](const PyFilter& self) { return self.core.frequency(); },
            "Frequency in cycles per sample.")
        .def("__repr__", [](const PyFilter& self) {
            std::string r = "GoertzelFilter(frequency=" + std::to_string(self.frequency_as_given);
            if (self.fs) r += ", fs=" + std::to_string(*self.fs);
            return r + ", n_samples=" + std::to_string(self.core.count()) + ")";
        });

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
