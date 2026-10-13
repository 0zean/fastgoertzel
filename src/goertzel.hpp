// SPDX-License-Identifier: GPL-3.0-or-later
//
// fastgoertzel core: integer-bin Goertzel algorithm.
//
// This header is self-contained and has no Python dependency. The functions here
// perform no validation and never throw; callers (the pybind11 layer) validate
// inputs *before* entering any parallel region.
//
// Derivation. With s[n] = x[n] + 2cos(w) s[n-1] - s[n-2], s[-1] = s[-2] = 0, and
// w = 2*pi*k/N for integer k, the DFT coefficient is
//     X[k] = e^{jw} s[N-1] - s[N-2],
// i.e. real = cos(w) s1 - s2, imag = sin(w) s1. This is the cosine-referenced
// (numpy.fft) phase convention. The amplitude is scaled by 2/N so that an interior
// bin recovers the amplitude of a sinusoid; DC and Nyquist bins read 2x (documented).

#ifndef FASTGOERTZEL_GOERTZEL_HPP
#define FASTGOERTZEL_GOERTZEL_HPP

#include <cmath>
#include <cstddef>

namespace fastgoertzel {

constexpr double kPi = 3.141592653589793238462643383279502884;

/// Per-frequency constants of the recurrence and the final step.
struct BinCoefficients {
    double cos_w;
    double sin_w;
    double coeff;  // 2 * cos_w
};

/// Amplitude/phase pair returned to Python.
struct Result {
    double amplitude;
    double phase;
};

/// A frequency is valid when it is finite and in [0, 1). NaN compares false on both
/// tests and is therefore rejected.
inline bool is_valid_frequency(double frequency) noexcept {
    return frequency >= 0.0 && frequency < 1.0;
}

/// 1.x bin mapping: k = floor(frequency * n). Requires is_valid_frequency(frequency),
/// so the product is in [0, n) and the cast is well defined.
inline std::size_t frequency_to_bin(double frequency, std::size_t n) noexcept {
    return static_cast<std::size_t>(frequency * static_cast<double>(n));
}

inline BinCoefficients bin_coefficients(std::size_t k, std::size_t n) noexcept {
    const double omega = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n);
    const double c = std::cos(omega);
    return BinCoefficients{c, std::sin(omega), 2.0 * c};
}

/// Goertzel over n samples at the bin described by `c`. Requires n >= 1.
inline Result goertzel_bin(const double* x, std::size_t n, const BinCoefficients& c) noexcept {
    double s1 = 0.0;
    double s2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double s0 = x[i] + c.coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    const double re = c.cos_w * s1 - s2;
    const double im = c.sin_w * s1;
    const double amplitude = std::sqrt(re * re + im * im) / (static_cast<double>(n) / 2.0);
    return Result{amplitude, std::atan2(im, re)};
}

}  // namespace fastgoertzel

#endif  // FASTGOERTZEL_GOERTZEL_HPP
