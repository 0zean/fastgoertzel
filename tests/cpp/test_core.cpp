// SPDX-License-Identifier: GPL-3.0-or-later
//
// Unit tests for the header-only core (include/fastgoertzel/goertzel.hpp), independent of
// Python. Built with -DFASTGOERTZEL_BUILD_TESTS=ON; see CMakeLists.txt.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "fastgoertzel/goertzel.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

namespace fgz = fastgoertzel;

namespace {

std::vector<double> random_signal(std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd;
    std::vector<double> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

std::complex<double> direct_dtft(const std::vector<double>& x, double f) {
    std::complex<double> acc{0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        acc += x[i] * std::polar(1.0, -fgz::kTwoPi * f * static_cast<double>(i));
    }
    return acc;
}

std::complex<double> as_std(fgz::Complex c) { return {c.re, c.im}; }

bool close(std::complex<double> a, std::complex<double> b, double tol) {
    return std::abs(a - b) <= tol;
}

}  // namespace

TEST_CASE("coefficients and frequency validation") {
    const auto c = fgz::coefficients(fgz::omega_from_frequency(0.25));
    CHECK(c.cos_w == doctest::Approx(0.0).epsilon(1e-15));
    CHECK(c.sin_w == doctest::Approx(1.0));
    CHECK(c.coeff == doctest::Approx(0.0).epsilon(1e-15));
    CHECK(fgz::omega_from_bin(3, 12) == doctest::Approx(fgz::omega_from_frequency(0.25)));

    CHECK(fgz::is_valid_frequency(0.0));
    CHECK(fgz::is_valid_frequency(0.999999));
    CHECK_FALSE(fgz::is_valid_frequency(1.0));
    CHECK_FALSE(fgz::is_valid_frequency(-1e-300));
    CHECK_FALSE(fgz::is_valid_frequency(std::nan("")));
    CHECK_FALSE(fgz::is_valid_frequency(INFINITY));
    CHECK(fgz::is_dc_or_nyquist(0.0));
    CHECK(fgz::is_dc_or_nyquist(0.5));
    CHECK_FALSE(fgz::is_dc_or_nyquist(0.25));
}

TEST_CASE("fractional_cycles keeps precision for large products") {
    // 0.1 * 3 would be 0.30000000000000004 in plain arithmetic; the error-free product and
    // reduction must give the fractional part of the exact product within ~1 ulp.
    CHECK(fgz::fractional_cycles(0.5, 7.0) == doctest::Approx(0.5).epsilon(1e-15));
    CHECK(fgz::fractional_cycles(0.25, 4.0) == doctest::Approx(0.0).epsilon(1e-15));
    // f = k/n exactly representable: k*(n-1)/n mod 1 = (n-k)/n ... for k=1, n=8: 7/8.
    CHECK(fgz::fractional_cycles(1.0 / 8.0, 7.0) == doctest::Approx(0.875).epsilon(1e-15));
    // Large multiplier: result stays in [0, 1).
    const double r = fgz::fractional_cycles(0.123456789, 1e9);
    CHECK(r >= 0.0);
    CHECK(r < 1.0);
}

TEST_CASE("integer-bin and generalized finishes agree with the DFT") {
    for (std::size_t n :
         {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{64}, std::size_t{1000}}) {
        const auto x = random_signal(n, static_cast<unsigned>(n));
        for (std::size_t k : {std::size_t{0}, n / 3, n / 2, n - 1}) {
            const auto expected = direct_dtft(x, static_cast<double>(k) / static_cast<double>(n));
            const auto by_bin = as_std(fgz::dft_bin(x.data(), n, k));
            const auto by_dtft =
                as_std(fgz::dtft(x.data(), n, static_cast<double>(k) / static_cast<double>(n)));
            const double tol = 1e-9 * (1.0 + static_cast<double>(n));
            CHECK(close(by_bin, expected, tol));
            CHECK(close(by_dtft, expected, tol));
        }
    }
}

TEST_CASE("generalized Goertzel at arbitrary frequencies") {
    const auto x = random_signal(513, 42);
    for (double f : {0.0, 0.5, 0.001, 0.123456789, 0.75, 0.999}) {
        const auto expected = direct_dtft(x, f);
        CHECK(close(as_std(fgz::dtft(x.data(), x.size(), f)), expected, 1e-8));
    }
}

TEST_CASE("amplitude scaling recovers sinusoid amplitude, with DC/Nyquist handled") {
    const std::size_t n = 400;
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 2.0 * std::cos(fgz::kTwoPi * 0.1 * static_cast<double>(i) + 0.3) + 0.7;
    }
    const auto X1 = fgz::dtft(x.data(), n, 0.1);
    CHECK(fgz::magnitude(X1) * fgz::amplitude_scale(n, false) ==
          doctest::Approx(2.0).epsilon(1e-9));
    CHECK(fgz::phase(X1) == doctest::Approx(0.3).epsilon(1e-9));
    const auto X0 = fgz::dtft(x.data(), n, 0.0);
    CHECK(fgz::magnitude(X0) * fgz::amplitude_scale(n, true) == doctest::Approx(0.7).epsilon(1e-9));
    CHECK(fgz::power(X1) == doctest::Approx(fgz::magnitude(X1) * fgz::magnitude(X1)));
}

TEST_CASE("recurrence can be resumed from a saved state") {
    const auto x = random_signal(1000, 7);
    const auto c = fgz::coefficients(fgz::omega_from_frequency(0.3));
    const fgz::State whole = fgz::recurrence(x.data(), x.size(), c.coeff);
    fgz::State split = fgz::recurrence(x.data(), 400, c.coeff);
    split = fgz::recurrence(x.data() + 400, 600, c.coeff, split);
    CHECK(split.s1 == doctest::Approx(whole.s1));
    CHECK(split.s2 == doctest::Approx(whole.s2));
}

TEST_CASE("StreamingGoertzel: push, absorb and reset") {
    const auto x = random_signal(5000, 11);
    const double f = 0.2718;
    const auto expected = direct_dtft(x, f);

    SUBCASE("sample by sample") {
        fgz::StreamingGoertzel s(f);
        for (double v : x) s.push(v);
        CHECK(s.count() == x.size());
        CHECK(close(as_std(s.result()), expected, 1e-8));
    }
    SUBCASE("chunks through push") {
        fgz::StreamingGoertzel s(f);
        s.push(x.data(), 1234);
        s.push(x.data() + 1234, 5000 - 1234);
        CHECK(close(as_std(s.result()), expected, 1e-8));
    }
    SUBCASE("mixed push and absorb (shift theorem)") {
        fgz::StreamingGoertzel s(f);
        s.push(x.data(), 100);
        const fgz::Complex mid = fgz::dtft(x.data() + 100, 3000, f);
        s.absorb(mid, 3000);
        s.push(x.data() + 3100, 1900);
        CHECK(s.count() == 5000);
        CHECK(close(as_std(s.result()), expected, 1e-8));
        s.reset();
        CHECK(s.count() == 0);
        CHECK(s.result().re == 0.0);
        CHECK(s.result().im == 0.0);
    }
    SUBCASE("empty filter") {
        fgz::StreamingGoertzel s(f);
        CHECK(s.count() == 0);
        CHECK(s.result().re == 0.0);
    }
}
