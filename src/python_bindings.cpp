#include "goertzel.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Wrapper function for NumPy arrays
std::tuple<double, double> goertzel_numpy(
	py::array_t<double, py::array::c_style | py::array::forcecast> x,
	double frequency
) {
	// Request buffer info
	py::buffer_info buf = x.request();

	// Validate input
	if (buf.ndim != 1) {
		throw std::runtime_error("Input must be a 1-dimensional array");
	}

	// Get pointer to data
	double* ptr = static_cast<double*>(buf.ptr);
	size_t length = buf.shape[0];

	// Call the core algorithm
	return fastgoertzel::GoertzelAlgorithm::compute(ptr, length, frequency);
}

// Batch processing for multiple frequencies
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

	// Allocate output array: [num_freqs, 2] for amplitude and phase
	py::array_t<double> result(std::vector<pybind11::ssize_t>{
        static_cast<pybind11::ssize_t>(num_freqs),
        static_cast<pybind11::ssize_t>(2)
    });
	double* result_ptr = static_cast<double*>(result.mutable_unchecked<2>().mutable_data(0, 0));

	// Process each frequency
	for (size_t i = 0; i < num_freqs; ++i) {
		auto [amp, phase] = fastgoertzel::GoertzelAlgorithm::compute(
			x_ptr, x_length, freq_ptr[i]
		);
		result_ptr[i * 2] = amp;
		result_ptr[i * 2 + 1] = phase;
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

    // Add version info
    #define VERSION_INFO "0.1.0"
    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
