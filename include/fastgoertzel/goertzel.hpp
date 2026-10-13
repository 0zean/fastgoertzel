// SPDX-License-Identifier: GPL-3.0-or-later
//
// fastgoertzel core: the Goertzel algorithm for one DTFT value, header-only, no Python.
//
// Nothing here validates or throws; callers (the pybind11 layer) validate inputs before
// entering any parallel region.
//
// Derivation
// ----------
// The second-order IIR recurrence
//     s[i] = x[i] + 2 cos(w) s[i-1] - s[i-2],      s[-1] = s[-2] = 0,
// followed by the FIR step y[i] = s[i] - e^{-jw} s[i-1] has the overall impulse response
// e^{jwi}, so after n samples
//     y[n-1] = e^{jw(n-1)} X(w),      X(w) = sum_{i<n} x[i] e^{-jwi}.
// Hence, for ANY w (the generalized Goertzel algorithm),
//     X(w) = e^{-jw(n-1)} (s1 - e^{-jw} s2)                                   (finish_general)
// and when w = 2 pi k / n for integer k, e^{-jw(n-1)} = e^{jw} and this reduces to the
// textbook form
//     X[k] = e^{jw} s1 - s2,   i.e. re = cos(w) s1 - s2, im = sin(w) s1.       (finish_bin)
// Both give the numpy.fft convention: a cosine with phase p reports phase p.
//
// Amplitude scaling: |X| * 2/n recovers the amplitude of a real sinusoid at an interior
// frequency; at DC and Nyquist the two spectral lines coincide and the factor is 1/n.
//
// Accuracy: rounding error of the recurrence grows roughly linearly with n and is worst
// for w near 0 or pi (poles near +-1). Measured: ~2e-10 relative at n = 1e6 for interior
// bins, ~2e-5 for bin 1. Use float64 only.

#ifndef FASTGOERTZEL_GOERTZEL_HPP
#define FASTGOERTZEL_GOERTZEL_HPP

#include <cmath>
#include <cstddef>

namespace fastgoertzel {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct Complex {
    double re;
    double im;
};

/// Per-frequency constants of the recurrence and the final step.
struct Coefficients {
    double omega;  // radians per sample
    double cos_w;
    double sin_w;
    double coeff;  // 2 cos(omega)
};

/// Recurrence state (s[i-1], s[i-2]).
struct State {
    double s1 = 0.0;
    double s2 = 0.0;
};

inline Coefficients coefficients(double omega) noexcept {
    const double c = std::cos(omega);
    return Coefficients{omega, c, std::sin(omega), 2.0 * c};
}

/// Normalised frequency (cycles per sample) to radians per sample.
inline double omega_from_frequency(double cycles_per_sample) noexcept {
    return kTwoPi * cycles_per_sample;
}

/// Bin k of an n-point DFT to radians per sample.
inline double omega_from_bin(std::size_t k, std::size_t n) noexcept {
    return kTwoPi * static_cast<double>(k) / static_cast<double>(n);
}

/// A normalised frequency is valid when it is finite and in [0, 1). NaN fails both tests.
inline bool is_valid_frequency(double cycles_per_sample) noexcept {
    return cycles_per_sample >= 0.0 && cycles_per_sample < 1.0;
}

/// At DC and Nyquist a real sinusoid has a single spectral line (see amplitude_scale).
inline bool is_dc_or_nyquist(double cycles_per_sample) noexcept {
    return cycles_per_sample == 0.0 || cycles_per_sample == 0.5;
}

/// Advances the recurrence over n samples starting from `s`.
///
/// The update is written as coeff*s1 + (x - s2) so that the subtraction leaves the
/// loop-carried dependency chain; with -ffp-contract=fast the remainder contracts to one
/// FMA on targets that have it (arm64, x86 with -mfma) and is a plain multiply-add elsewhere.
inline State recurrence(const double* x, std::size_t n, double coeff, State s = State{}) noexcept {
    double s1 = s.s1;
    double s2 = s.s2;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = x[i] - s2;
        const double s0 = coeff * s1 + t;
        s2 = s1;
        s1 = s0;
    }
    return State{s1, s2};
}

/// X[k] for an integer bin. Valid only when c.omega == 2 pi k / n and n samples were fed.
inline Complex finish_bin(State s, const Coefficients& c) noexcept {
    return Complex{c.cos_w * s.s1 - s.s2, c.sin_w * s.s1};
}

/// Fractional part of f*m in [0, 1), using an error-free product so the result keeps full
/// precision even when f*m is in the millions of cycles.
inline double fractional_cycles(double f, double m) noexcept {
    const double p = f * m;
    const double e = std::fma(f, m, -p);  // exact rounding error of the product
    double frac = (p - std::floor(p)) + e;
    frac -= std::floor(frac);
    return frac;
}

/// Multiplies v by e^{-j 2 pi cycles}; `cycles` should already be reduced to [0, 1).
inline Complex rotate_by_cycles(Complex v, double cycles) noexcept {
    const double ang = -kTwoPi * cycles;
    const double tr = std::cos(ang);
    const double ti = std::sin(ang);
    return Complex{v.re * tr - v.im * ti, v.re * ti + v.im * tr};
}

/// Generalized final step given the twiddle in cycles: X = e^{-j 2 pi cycles} (s1 - e^{-jw} s2),
/// where `cycles` is f*(n-1) reduced modulo 1 for a chain of n samples (see finish_general).
inline Complex finish_with_cycles(State s, const Coefficients& c, double cycles) noexcept {
    return rotate_by_cycles(Complex{s.s1 - c.cos_w * s.s2, c.sin_w * s.s2}, cycles);
}

/// X(omega) after n >= 1 samples, for any omega: e^{-j omega (n-1)} (s1 - e^{-j omega} s2).
/// `cycles_per_sample` must equal c.omega / (2 pi); it is passed separately so the final
/// twiddle can be reduced modulo one cycle exactly instead of through a huge angle.
inline Complex finish_general(State s, std::size_t n, const Coefficients& c,
                              double cycles_per_sample) noexcept {
    return finish_with_cycles(s, c,
                              fractional_cycles(cycles_per_sample, static_cast<double>(n - 1)));
}

/// One-shot DFT bin k of x[0..n). Requires n >= 1 and k < n.
inline Complex dft_bin(const double* x, std::size_t n, std::size_t k) noexcept {
    const Coefficients c = coefficients(omega_from_bin(k, n));
    return finish_bin(recurrence(x, n, c.coeff), c);
}

/// One-shot DTFT of x[0..n) at an arbitrary normalised frequency. Requires n >= 1.
inline Complex dtft(const double* x, std::size_t n, double cycles_per_sample) noexcept {
    const Coefficients c = coefficients(omega_from_frequency(cycles_per_sample));
    return finish_general(recurrence(x, n, c.coeff), n, c, cycles_per_sample);
}

inline double power(Complex X) noexcept { return X.re * X.re + X.im * X.im; }
inline double magnitude(Complex X) noexcept { return std::sqrt(power(X)); }
inline double phase(Complex X) noexcept { return std::atan2(X.im, X.re); }

/// Factor that turns |X| into the amplitude of the real sinusoid at that frequency.
inline double amplitude_scale(std::size_t n, bool dc_or_nyquist) noexcept {
    return (dc_or_nyquist ? 1.0 : 2.0) / static_cast<double>(n);
}

/// Stateful Goertzel filter: feed samples as they arrive, read X at any time.
/// Not safe for concurrent push() on the same instance.
class StreamingGoertzel {
public:
    explicit StreamingGoertzel(double cycles_per_sample) noexcept
        : f_(cycles_per_sample), c_(coefficients(omega_from_frequency(cycles_per_sample))) {}

    void push(const double* x, std::size_t n) noexcept {
        s_ = recurrence(x, n, c_.coeff, s_);
        n_ += n;
    }
    void push(double x) noexcept { push(&x, 1); }

    /// X at the filter frequency over every sample pushed so far (0 when nothing was pushed).
    Complex result() const noexcept {
        return n_ == 0 ? Complex{0.0, 0.0} : finish_general(s_, n_, c_, f_);
    }

    void reset() noexcept {
        s_ = State{};
        n_ = 0;
    }

    std::size_t count() const noexcept { return n_; }
    double frequency() const noexcept { return f_; }
    const Coefficients& coeffs() const noexcept { return c_; }

private:
    double f_;
    Coefficients c_;
    State s_{};
    std::size_t n_ = 0;
};

}  // namespace fastgoertzel

#endif  // FASTGOERTZEL_GOERTZEL_HPP
