#include "goertzel.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Wrapper function for NumPy arrays
std::tuple<double, double> goertzel_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    double frequency
) {
    py::buffer_info buf = x.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be a 1-dimensional array");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    size_t length = buf.shape[0];

    py::gil_scoped_release release; // <-- Release GIL during computation
    return fastgoertzel::GoertzelAlgorithm::compute(ptr, length, frequency);
}

py::array_t<double> goertzel_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> frequencies
) {
    py::buffer_info x_buf = x.request();
    py::buffer_info freq_buf = frequencies.request();

    if (x_buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }
    if (freq_buf.ndim != 1) {
        throw std::runtime_error("Frequencies must be 1-dimensional");
    }

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* freq_ptr = static_cast<double*>(freq_buf.ptr);
    size_t x_length = x_buf.shape[0];
    size_t num_freqs = freq_buf.shape[0];

    py::array_t<double> result(std::vector<pybind11::ssize_t>{
        static_cast<pybind11::ssize_t>(num_freqs),
        static_cast<pybind11::ssize_t>(2)
    });
    auto result_rw = result.mutable_unchecked<2>();

    // Precompute coefficients for all frequencies (reduces redundant work)
    std::vector<double> cos_omegas(num_freqs), sin_omegas(num_freqs), coeffs(num_freqs);
    static constexpr double PI = 3.141592653589793238462643383279502884;
    for (size_t i = 0; i < num_freqs; ++i) {
        size_t k = static_cast<size_t>(freq_ptr[i] * x_length);
        double omega = 2.0 * PI * k / static_cast<double>(x_length);
        cos_omegas[i] = std::cos(omega);
        sin_omegas[i] = std::sin(omega);
        coeffs[i] = 2.0 * cos_omegas[i];
    }

    py::gil_scoped_release release; // Release GIL during batch computation

    // Parallelize across frequencies
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(num_freqs); ++i) {
        // Use the precomputed coefficients for each frequency
        double s_prev = 0.0, s_prev2 = 0.0;
        double coeff = coeffs[i];
        double cos_omega = cos_omegas[i];
        double sin_omega = sin_omegas[i];

        // Loop unrolling for speed
        size_t j = 0;
        for (; j + 3 < x_length; j += 4) {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&x_ptr[j + 16], 0, 1);
#endif
            double s_curr = x_ptr[j] + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_curr;

            s_curr = x_ptr[j + 1] + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_curr;

            s_curr = x_ptr[j + 2] + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_curr;

            s_curr = x_ptr[j + 3] + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_curr;
        }
        for (; j < x_length; ++j) {
            double s_curr = x_ptr[j] + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_curr;
        }

        double real_part = cos_omega * s_prev - s_prev2;
        double imag_part = sin_omega * s_prev;
        double amplitude = std::sqrt(real_part * real_part + imag_part * imag_part) / (static_cast<double>(x_length) / 2.0);
        double phase = std::atan2(imag_part, real_part);

        result_rw(i, 0) = amplitude;
        result_rw(i, 1) = phase;
    }

    return result;
}

py::array_t<double> goertzel_sliding_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    size_t window_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> frequencies
) {
    py::buffer_info x_buf = x.request();
    py::buffer_info freq_buf = frequencies.request();

    size_t total = x_buf.shape[0];
    size_t n_windows = total - window_size + 1;
    size_t n_freqs = freq_buf.shape[0];

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* freq_ptr = static_cast<double*>(freq_buf.ptr);

    py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_windows), static_cast<py::ssize_t>(n_freqs), 2});
    auto result_rw = result.mutable_unchecked<3>();

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t w = 0; w < static_cast<ptrdiff_t>(n_windows); ++w) {
        const double* window_ptr = x_ptr + w;
        for (size_t f = 0; f < n_freqs; ++f) {
            auto tup = fastgoertzel::GoertzelAlgorithm::compute(window_ptr, window_size, freq_ptr[f]);
            result_rw(w, f, 0) = std::get<0>(tup); // amplitude
            result_rw(w, f, 1) = std::get<1>(tup); // phase
        }
    }
    return result;
}


PYBIND11_MODULE(_fastgoertzel_core, m) {
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
    )pbdoc";

    // Main function
    m.def("goertzel", &goertzel_numpy,
        py::arg("x"),
        py::arg("frequency"),
        R"pbdoc(
        Compute amplitude and phase using the Goertzel algorithm.
        
        Parameters
        ----------
        x : numpy.ndarray
            1-dimensional array of input samples
        frequency : float
            Normalized frequency to detect (0 to 1, where 1 = sampling rate)
            
        Returns
        -------
        tuple[float, float]
            (amplitude, phase) where phase is in radians
            
        Examples
        --------
        >>> import numpy as np
        >>> import fastgoertzel
        >>> 
        >>> # Generate test signal
        >>> t = np.arange(0, 512)
        >>> freq = 1/128  # Normalized frequency
        >>> signal = np.sin(2 * np.pi * freq * t)
        >>> 
        >>> # Detect frequency
        >>> amp, phase = fastgoertzel.goertzel(signal, freq)
        >>> print(f"Amplitude: {amp:.4f}, Phase: {phase:.4f}")
        
        Notes
        -----
        The Goertzel algorithm is particularly efficient when detecting
        a small number of frequencies compared to computing a full FFT.
        )pbdoc");

    // Batch processing function
    m.def("goertzel_batch", &goertzel_batch,
        py::arg("x"),
        py::arg("frequencies"),
        R"pbdoc(
        Process multiple frequencies in a single call.
        
        Parameters
        ----------
        x : numpy.ndarray
            1-dimensional array of input samples
        frequencies : numpy.ndarray
            1-dimensional array of normalized frequencies
            
        Returns
        -------
        numpy.ndarray
            2D array of shape (n_frequencies, 2) with amplitudes and phases
        )pbdoc");
    
    // Sliding window batch processing function
    m.def("goertzel_sliding_batch", &goertzel_sliding_batch,
        py::arg("x"),
        py::arg("window_size"),
        py::arg("frequencies"),
        R"pbdoc(
        Compute Goertzel for multiple sliding windows and frequencies in parallel.

        Parameters
        ----------
        x : numpy.ndarray
            1-dimensional array of input samples
        window_size : int
            Size of each sliding window
        frequencies : numpy.ndarray
            1-dimensional array of normalized frequencies
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_windows, n_frequencies, 2) with amplitudes and phases

        )pbdoc");

    // Add version info
    #define VERSION_INFO "1.0.2"
    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
