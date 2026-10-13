// Micro-benchmark of Goertzel kernel formulations. Build with and without /arch:AVX2.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using std::complex;
static const double PI = 3.141592653589793238462643383279502884;

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

// A: the shipped formulation  s0 = x + coeff*s1 - s2  (mul -> add -> sub on the critical path)
NOINLINE complex<double> kernelA(const double* x, size_t n, double omega) {
    const double coeff = 2.0 * std::cos(omega);
    double s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double s0 = x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return complex<double>(std::cos(omega) * s1 - s2, std::sin(omega) * s1);
}

// B: same recurrence, but (x - s2) is computed off the critical path and the chain is one FMA
NOINLINE complex<double> kernelB(const double* x, size_t n, double omega) {
    const double coeff = 2.0 * std::cos(omega);
    double s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = x[i] - s2;
        double s0 = std::fma(coeff, s1, t);
        s2 = s1;
        s1 = s0;
    }
    return complex<double>(std::cos(omega) * s1 - s2, std::sin(omega) * s1);
}

// C: NB independent contiguous blocks (breaks the dependency chain), generalized finish per block,
//    combined with block twiddles. Exact for any omega, not only integer bins.
template <int NB>
NOINLINE complex<double> kernelC(const double* x, size_t n, double omega) {
    const size_t len = n / NB;
    const double coeff = 2.0 * std::cos(omega);
    double s1[NB], s2[NB];
    for (int b = 0; b < NB; ++b)
        s1[b] = s2[b] = 0.0;
    for (size_t i = 0; i < len; ++i) {
        for (int b = 0; b < NB; ++b) {
            double t = x[b * len + i] - s2[b];
            double s0 = std::fma(coeff, s1[b], t);
            s2[b] = s1[b];
            s1[b] = s0;
        }
    }
    const complex<double> e_minus_jw = std::polar(1.0, -omega);
    const complex<double> finish = std::polar(1.0, -omega * static_cast<double>(len - 1));
    complex<double> X = 0.0;
    for (int b = 0; b < NB; ++b) {
        complex<double> Xb = finish * (s1[b] - e_minus_jw * s2[b]);
        X += std::polar(1.0, -omega * static_cast<double>(b * len)) * Xb;
    }
    for (size_t i = len * NB; i < n; ++i)
        X += x[i] * std::polar(1.0, -omega * static_cast<double>(i));
    return X;
}

// D: M frequencies per pass (structure-of-arrays state), the batch formulation
template <int M>
NOINLINE void kernelD(const double* x, size_t n, const double* omegas, complex<double>* out) {
    double coeff[M], s1[M], s2[M];
    for (int m = 0; m < M; ++m) {
        coeff[m] = 2.0 * std::cos(omegas[m]);
        s1[m] = s2[m] = 0.0;
    }
    for (size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        for (int m = 0; m < M; ++m) {
            double t = xi - s2[m];
            double s0 = std::fma(coeff[m], s1[m], t);
            s2[m] = s1[m];
            s1[m] = s0;
        }
    }
    for (int m = 0; m < M; ++m)
        out[m] = complex<double>(std::cos(omegas[m]) * s1[m] - s2[m], std::sin(omegas[m]) * s1[m]);
}

template <class F>
double time_it(F f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main() {
    const size_t n = 1 << 20;  // 1,048,576 samples
    std::vector<double> x(n);
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nd;
    for (auto& v : x)
        v = nd(rng);
    const size_t k = 104857;  // ~0.1 cycles/sample, integer bin
    const double omega = 2.0 * PI * static_cast<double>(k) / static_cast<double>(n);

    // reference DFT bin
    complex<double> ref = 0.0;
    for (size_t i = 0; i < n; ++i)
        ref += x[i] * std::polar(1.0, -omega * static_cast<double>(i));

    volatile double sink = 0.0;
    auto report = [&](const char* name, complex<double> X, double sec) {
        double err = std::abs(X - ref) / std::abs(ref);
        std::printf("%-28s %7.3f ms  %6.1f MS/s  %5.2f ns/sample  rel err %.1e\n", name, sec * 1e3,
                    n / sec / 1e6, sec / n * 1e9, err);
    };
    const int reps = 30;
    complex<double> X;
    double t;

    t = time_it(
        [&] {
            X = kernelA(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("A shipped (mul,add,sub)", X, t);
    t = time_it(
        [&] {
            X = kernelB(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("B single FMA chain", X, t);
    t = time_it(
        [&] {
            X = kernelC<2>(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("C 2 blocks", X, t);
    t = time_it(
        [&] {
            X = kernelC<4>(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("C 4 blocks", X, t);
    t = time_it(
        [&] {
            X = kernelC<8>(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("C 8 blocks", X, t);
    t = time_it(
        [&] {
            X = kernelC<16>(x.data(), n, omega);
            sink += X.real();
        },
        reps);
    report("C 16 blocks", X, t);

    // batch: 8 frequencies, compare "8 x kernelB" against one pass with kernelD<8>
    double omegas[8];
    complex<double> outs[8];
    for (int m = 0; m < 8; ++m)
        omegas[m] = 2.0 * PI * static_cast<double>(1000 * (m + 1)) / static_cast<double>(n);
    t = time_it(
        [&] {
            for (int m = 0; m < 8; ++m) {
                outs[m] = kernelB(x.data(), n, omegas[m]);
            }
            sink += outs[7].real();
        },
        reps);
    std::printf("%-28s %7.3f ms  (8 passes over x)\n", "batch: 8 x kernelB", t * 1e3);
    complex<double> outs2[8];
    t = time_it(
        [&] {
            kernelD<8>(x.data(), n, omegas, outs2);
            sink += outs2[7].real();
        },
        reps);
    double maxrel = 0.0;
    for (int m = 0; m < 8; ++m)
        maxrel = std::max(maxrel, std::abs(outs2[m] - outs[m]) / std::abs(outs[m]));
    std::printf("%-28s %7.3f ms  (1 pass, SoA)  max rel diff vs 8xB %.1e\n",
                "batch: kernelD<8> one pass", t * 1e3, maxrel);
    std::printf("(sink %g)\n", (double)sink);
    return 0;
}
