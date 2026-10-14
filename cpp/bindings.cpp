// SPDX-License-Identifier: GPL-3.0-or-later
//
// pybind11 layer for fastgoertzel.
//
// Responsibilities of this file:
//   * argument conversion (any array-like -> contiguous float64),
//   * validation, raising ValueError *before* the GIL is released and before any
//     OpenMP region is entered (an exception escaping an OpenMP region terminates
//     the process),
//   * work decomposition: (frequency, block) lanes for the lane kernel, (frequency, chunk)
//     items for the sliding DFT, OpenMP over items, deterministic combination,
//   * releasing the GIL for the numeric work.
// Hot loops live in lane_kernels.inl and are reached through kernels::active_api().

#include "fastgoertzel/goertzel.hpp"
#include "kernels_api.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
namespace fgk = fastgoertzel::kernels;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using OptDouble = std::optional<double>;
using OptInt = std::optional<int>;

// Below this much work (samples * frequencies) an OpenMP region costs more than it saves.
constexpr std::size_t kParallelThreshold = std::size_t{1} << 16;
// Blocks shorter than this are not worth a lane of their own.
constexpr std::size_t kMinBlock = 512;
// Sliding DFT: windows per chunk between exact re-synchronisations.
constexpr std::size_t kSlidingChunk = 256;
// Lane groups a large batch is cut into, independent of the actual thread count.
constexpr std::size_t kPlanParallelism = 32;

// ---- scale ------------------------------------------------------------------------

enum class Scale { Amplitude, Magnitude, Power };

Scale parse_scale(const std::string& s) {
    if (s == "amplitude") return Scale::Amplitude;
    if (s == "magnitude") return Scale::Magnitude;
    if (s == "power") return Scale::Power;
    throw py::value_error("scale must be 'amplitude', 'magnitude' or 'power' (got '" + s + "')");
}

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

// Decimal text for messages; std::to_string prints six fixed decimals and turns 1e-9 into
// "0.000000".
std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}

void require_1d(const Array& a, const char* name) {
    if (a.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be a 1-dimensional array, got " +
                              std::to_string(a.ndim()) + " dimension(s)");
    }
}

void require_nonempty(std::size_t n, const char* name) {
    if (n == 0) throw py::value_error(std::string(name) + " must contain at least one element");
}

void require_fs(const OptDouble& fs) {
    if (fs && !(std::isfinite(*fs) && *fs > 0.0)) {
        throw py::value_error("fs must be a positive finite number (got " + num(*fs) + ")");
    }
}

// Hz (when fs is given) or cycles/sample -> validated cycles/sample.
double resolve_frequency(double frequency, const OptDouble& fs) {
    if (!std::isfinite(frequency)) {
        throw py::value_error("frequency must be finite (got " + num(frequency) + ")");
    }
    const double f = fs ? frequency / *fs : frequency;
    if (!fgz::is_valid_frequency(f)) {
        if (fs) {
            throw py::value_error("frequency must be in [0, fs) (got " + num(frequency) +
                                  " with fs = " + num(*fs) + ")");
        }
        throw py::value_error("frequency must be in [0, 1) (got " + num(frequency) + ")");
    }
    return f;
}

std::vector<double> resolve_frequencies(const Array& frequencies, const OptDouble& fs) {
    require_1d(frequencies, "frequencies");
    const auto f = frequencies.unchecked<1>();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(f.shape(0)));
    for (py::ssize_t i = 0; i < f.shape(0); ++i) out.push_back(resolve_frequency(f(i), fs));
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

int effective_threads(int requested) noexcept {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

py::ssize_t checked_ssize(std::size_t v) { return static_cast<py::ssize_t>(v); }

// ---- preallocated output ----------------------------------------------------------

template <typename It>
std::string shape_string(It first, It last) {
    std::string s = "(";
    for (It it = first; it != last; ++it) {
        if (it != first) s += ", ";
        s += std::to_string(*it);
    }
    return s + (last - first == 1 ? ",)" : ")");
}

// A fresh array of `shape`, or the caller's `out`, which must match exactly: the kernels
// write straight into it, so there is no casting, no copying, and no allocation. Overlap
// with `x` is rejected because the kernels read `x` while writing the output.
template <typename T, std::size_t Dim>
py::array_t<T> resolve_out(const py::object& out, const std::array<py::ssize_t, Dim>& shape,
                           const char* dtype_name, const Array& x) {
    if (out.is_none()) {
        return py::array_t<T>(std::vector<py::ssize_t>(shape.begin(), shape.end()));
    }
    if (!py::isinstance<py::array>(out)) {
        throw py::type_error("out must be a numpy.ndarray or None");
    }
    const auto arr = py::reinterpret_borrow<py::array>(out);
    bool ok = py::isinstance<py::array_t<T, py::array::c_style>>(out) && arr.writeable() &&
              static_cast<std::size_t>(arr.ndim()) == Dim;
    for (std::size_t d = 0; ok && d < Dim; ++d) ok = arr.shape(checked_ssize(d)) == shape[d];
    if (!ok) {
        throw py::value_error(std::string("out must be a writeable C-contiguous ") + dtype_name +
                              " array of shape " + shape_string(shape.begin(), shape.end()) +
                              " (got dtype " + std::string(py::str(arr.dtype())) + ", shape " +
                              shape_string(arr.shape(), arr.shape() + arr.ndim()) +
                              (arr.writeable() ? "" : ", read-only") +
                              ((arr.flags() & py::array::c_style) ? "" : ", not C-contiguous") +
                              ")");
    }
    const auto* ob = static_cast<const char*>(arr.data());
    const auto* xb = reinterpret_cast<const char*>(x.data());
    if (ob < xb + x.nbytes() && xb < ob + arr.nbytes()) {
        throw py::value_error("out must not share memory with x");
    }
    return py::reinterpret_borrow<py::array_t<T>>(out);
}

// ---- twiddle policies -------------------------------------------------------------
// cycles(m) returns f*m reduced modulo one cycle, where f is the frequency in cycles/sample.

struct FreqTwiddle {
    double f;
    double cycles(std::size_t m) const noexcept {
        return fgz::fractional_cycles(f, static_cast<double>(m));
    }
};

// Integer bin k of an n-point DFT: k*m/n reduced modulo 1 in exact integer arithmetic.
struct BinTwiddle {
    std::size_t k;
    std::size_t n;
    double cycles(std::size_t m) const noexcept {
        if (static_cast<double>(n) < 4294967296.0) {
            const std::uint64_t prod =
                static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(m % n);
            return static_cast<double>(prod % static_cast<std::uint64_t>(n)) /
                   static_cast<double>(n);
        }
        return fgz::fractional_cycles(static_cast<double>(k) / static_cast<double>(n),
                                      static_cast<double>(m));
    }
};

// ---- lane planning ----------------------------------------------------------------
// m frequencies over n samples are split into m * blocks lanes (lane = frequency f, block b
// covering samples [b*base_len, ...)). Blocks only exist to create independent chains when
// there are fewer than `want_lanes` frequencies; every block is at least kMinBlock samples.

struct Plan {
    std::size_t n;
    std::size_t m;
    std::size_t blocks;
    std::size_t base_len;
    std::size_t lanes;
    std::size_t items;  // groups of up to kMaxLanes lanes
};

Plan make_plan(std::size_t n, std::size_t m, std::size_t want_lanes) noexcept {
    std::size_t blocks = 1;
    if (m < want_lanes) {
        blocks = (want_lanes + m - 1) / m;
        blocks = std::min(blocks, std::max<std::size_t>(1, n / kMinBlock));
    }
    const std::size_t lanes = m * blocks;
    const auto L = static_cast<std::size_t>(fgk::kMaxLanes);
    return Plan{n, m, blocks, n / blocks, lanes, (lanes + L - 1) / L};
}

struct LaneGeom {
    std::size_t f;
    std::size_t offset;
    std::size_t len;
};

LaneGeom lane_geometry(const Plan& p, std::size_t lane) noexcept {
    const std::size_t f = lane / p.blocks;
    const std::size_t b = lane % p.blocks;
    const std::size_t offset = b * p.base_len;
    const std::size_t len = (b + 1 == p.blocks) ? p.n - offset : p.base_len;
    return LaneGeom{f, offset, len};
}

void run_item(const fgk::Api& api, const double* x, const Plan& p, const fgz::Coefficients* coeffs,
              std::size_t item, fgz::State* states) noexcept {
    const auto L = static_cast<std::size_t>(fgk::kMaxLanes);
    fgk::LaneInput in[fgk::kMaxLanes];
    int count = 0;
    const std::size_t first = item * L;
    const std::size_t last = std::min(first + L, p.lanes);
    for (std::size_t lane = first; lane < last; ++lane) {
        const LaneGeom g = lane_geometry(p, lane);
        in[count++] = fgk::LaneInput{x + g.offset, g.len, coeffs[g.f].coeff, 0.0, 0.0};
    }
    api.run_lanes(in, count, states + first);
}

// X_f = sum over blocks of e^{-jw offset_b} * finish(block b). Deterministic order.
template <class Twiddle>
fgz::Complex combine(const Plan& p, std::size_t f, const fgz::State* states,
                     const fgz::Coefficients& c, const Twiddle& tw) noexcept {
    fgz::Complex X{0.0, 0.0};
    for (std::size_t b = 0; b < p.blocks; ++b) {
        const std::size_t lane = f * p.blocks + b;
        const LaneGeom g = lane_geometry(p, lane);
        fgz::Complex Xb = fgz::finish_with_cycles(states[lane], c, tw.cycles(g.len - 1));
        if (g.offset != 0) Xb = fgz::rotate_by_cycles(Xb, tw.cycles(g.offset));
        X.re += Xb.re;
        X.im += Xb.im;
    }
    return X;
}

// One frequency, serial, all lanes used as blocks.
template <class Twiddle>
fgz::Complex dtft_one(const fgk::Api& api, const double* x, std::size_t n,
                      const fgz::Coefficients& c, const Twiddle& tw) noexcept {
    const Plan p = make_plan(n, 1, static_cast<std::size_t>(fgk::kMaxLanes));
    fgz::State states[fgk::kMaxLanes];
    run_item(api, x, p, &c, 0, states);
    return combine(p, 0, states, c, tw);
}

// Several frequencies; OpenMP over lane groups when the work is large enough.
void dtft_many(const double* x, std::size_t n, const std::vector<fgz::Coefficients>& coeffs,
               const std::vector<FreqTwiddle>& tws, fgz::Complex* out, int threads) noexcept {
    const fgk::Api& api = fgk::active_api();
    const std::size_t m = coeffs.size();
    if (m == 0) return;
    const bool big = n * m >= kParallelThreshold;
    const int T = big ? effective_threads(threads) : 1;
    // The decomposition must not depend on the thread count, or results would differ in
    // their last bits between machines and `threads=` settings. Large inputs are always cut
    // into enough lanes for kPlanParallelism groups; OpenMP then distributes the groups.
    const std::size_t want_lanes =
        static_cast<std::size_t>(fgk::kMaxLanes) * (big ? kPlanParallelism : std::size_t{1});
    const Plan p = make_plan(n, m, want_lanes);
    std::vector<fgz::State> states(p.items * static_cast<std::size_t>(fgk::kMaxLanes));
    const bool parallel = big && p.items > 1;
    (void)parallel;
    (void)T;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel) num_threads(T)
#endif
    for (std::ptrdiff_t item = 0; item < static_cast<std::ptrdiff_t>(p.items); ++item) {
        run_item(api, x, p, coeffs.data(), static_cast<std::size_t>(item), states.data());
    }
    for (std::size_t f = 0; f < m; ++f) out[f] = combine(p, f, states.data(), coeffs[f], tws[f]);
}

std::vector<FreqTwiddle> twiddles_for(const std::vector<double>& freqs) {
    std::vector<FreqTwiddle> out;
    out.reserve(freqs.size());
    for (double f : freqs) out.push_back(FreqTwiddle{f});
    return out;
}

// ---- sliding windows --------------------------------------------------------------

struct SlidingSpec {
    std::size_t total;
    std::size_t window;
    std::size_t step;
    std::size_t n_windows;
};

template <class Writer>
void sliding_many(const double* x, const SlidingSpec& sp,
                  const std::vector<fgz::Coefficients>& coeffs, const std::vector<double>& freqs,
                  int threads, Writer write) noexcept {
    const fgk::Api& api = fgk::active_api();
    const std::size_t m = coeffs.size();
    if (m == 0 || sp.n_windows == 0) return;

    // Hop cost: `step` updates of ~8 flops against ~2*window flops for an exact window.
    const bool use_updates = sp.step * 4 < sp.window;
    std::vector<fgk::SlidingConsts> consts(m);
    for (std::size_t f = 0; f < m; ++f) {
        const fgz::Complex tail = fgz::rotate_by_cycles(
            fgz::Complex{1.0, 0.0},
            fgz::fractional_cycles(freqs[f], static_cast<double>(sp.window - 1)));
        consts[f] = fgk::SlidingConsts{coeffs[f].cos_w, coeffs[f].sin_w, tail.re, tail.im};
    }

    const std::size_t R = std::min(sp.n_windows, kSlidingChunk);
    const std::size_t n_chunks = (sp.n_windows + R - 1) / R;
    const std::size_t items = n_chunks * m;
    const std::size_t work = use_updates ? sp.n_windows * m * sp.step * 4 + n_chunks * m * sp.window
                                         : sp.n_windows * m * sp.window;
    const bool parallel = work >= kParallelThreshold && items > 1;
    const int T = parallel ? effective_threads(threads) : 1;
    (void)parallel;
    (void)T;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel) num_threads(T)
#endif
    for (std::ptrdiff_t it = 0; it < static_cast<std::ptrdiff_t>(items); ++it) {
        const std::size_t f = static_cast<std::size_t>(it) / n_chunks;
        const std::size_t chunk = static_cast<std::size_t>(it) % n_chunks;
        const std::size_t w0 = chunk * R;
        const std::size_t count = std::min(R, sp.n_windows - w0);
        const FreqTwiddle tw{freqs[f]};
        const double* start = x + w0 * sp.step;
        const fgz::Complex y0 = dtft_one(api, start, sp.window, coeffs[f], tw);
        if (use_updates) {
            fgz::Complex buf[kSlidingChunk];  // count <= R <= kSlidingChunk; no heap traffic
            api.sliding_updates(start, sp.window, sp.step, count, y0, consts[f], buf);
            for (std::size_t w = 0; w < count; ++w) write(w0 + w, f, buf[w]);
        } else {
            write(w0, f, y0);
            for (std::size_t w = 1; w < count; ++w) {
                write(w0 + w, f, dtft_one(api, start + w * sp.step, sp.window, coeffs[f], tw));
            }
        }
    }
}

// ---- one frequency ----------------------------------------------------------------

fgz::Complex compute_dtft(const Array& x, double f) {
    const double* data = x.data();
    const auto n = static_cast<std::size_t>(x.shape(0));
    const fgz::Coefficients c = fgz::coefficients(fgz::omega_from_frequency(f));
    py::gil_scoped_release release;
    return dtft_one(fgk::active_api(), data, n, c, FreqTwiddle{f});
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
    const fgz::Coefficients c = fgz::coefficients(fgz::omega_from_bin(kk, n));
    const double* data = x.data();
    fgz::Complex X{};
    {
        py::gil_scoped_release release;
        X = dtft_one(fgk::active_api(), data, n, c, BinTwiddle{kk, n});
    }
    const bool edge = (kk == 0) || (2 * kk == n);
    return py::make_tuple(scaled(X, n, edge, sc), fgz::phase(X));
}

// ---- several frequencies ----------------------------------------------------------

py::array_t<double> goertzel_batch(const Array& x, const Array& frequencies, const OptDouble& fs,
                                   const std::string& scale, const OptInt& threads,
                                   const py::object& out) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const Scale sc = parse_scale(scale);
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::vector<FreqTwiddle> tws = twiddles_for(freqs);
    const std::size_t m = freqs.size();

    py::array_t<double> result = resolve_out<double, 2>(out, {checked_ssize(m), 2}, "float64", x);
    auto dst = result.mutable_unchecked<2>();
    const double* data = x.data();
    std::vector<fgz::Complex> X(m);
    {
        py::gil_scoped_release release;
        dtft_many(data, n, coeffs, tws, X.data(), nthreads);
    }
    for (std::size_t i = 0; i < m; ++i) {
        dst(checked_ssize(i), 0) = scaled(X[i], n, fgz::is_dc_or_nyquist(freqs[i]), sc);
        dst(checked_ssize(i), 1) = fgz::phase(X[i]);
    }
    return result;
}

py::array_t<std::complex<double>> goertzel_batch_complex(const Array& x, const Array& frequencies,
                                                         const OptDouble& fs, const OptInt& threads,
                                                         const py::object& out) {
    require_1d(x, "x");
    const auto n = static_cast<std::size_t>(x.shape(0));
    require_nonempty(n, "x");
    require_fs(fs);
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::vector<FreqTwiddle> tws = twiddles_for(freqs);
    const std::size_t m = freqs.size();

    py::array_t<std::complex<double>> result =
        resolve_out<std::complex<double>, 1>(out, {checked_ssize(m)}, "complex128", x);
    auto dst = result.mutable_unchecked<1>();
    const double* data = x.data();
    std::vector<fgz::Complex> X(m);
    {
        py::gil_scoped_release release;
        dtft_many(data, n, coeffs, tws, X.data(), nthreads);
    }
    for (std::size_t i = 0; i < m; ++i)
        dst(checked_ssize(i)) = std::complex<double>(X[i].re, X[i].im);
    return result;
}

py::array_t<double> goertzel_sliding_batch(const Array& x, py::ssize_t window_size,
                                           const Array& frequencies, py::ssize_t step,
                                           const OptDouble& fs, const std::string& scale,
                                           const OptInt& threads, const py::object& out) {
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
    const SlidingSpec sp{total, w, st, (total - w) / st + 1};

    std::vector<bool> edge(m);
    for (std::size_t i = 0; i < m; ++i) edge[i] = fgz::is_dc_or_nyquist(freqs[i]);

    py::array_t<double> result = resolve_out<double, 3>(
        out, {checked_ssize(sp.n_windows), checked_ssize(m), 2}, "float64", x);
    auto dst = result.mutable_unchecked<3>();
    const double* data = x.data();
    {
        py::gil_scoped_release release;
        sliding_many(data, sp, coeffs, freqs, nthreads,
                     [&](std::size_t wi, std::size_t fi, fgz::Complex Y) noexcept {
                         dst(checked_ssize(wi), checked_ssize(fi), 0) = scaled(Y, w, edge[fi], sc);
                         dst(checked_ssize(wi), checked_ssize(fi), 1) = fgz::phase(Y);
                     });
    }
    return result;
}

py::array_t<std::complex<double>> goertzel_sliding_batch_complex(
    const Array& x, py::ssize_t window_size, const Array& frequencies, py::ssize_t step,
    const OptDouble& fs, const OptInt& threads, const py::object& out) {
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
    const int nthreads = resolve_threads(threads);
    const std::vector<double> freqs = resolve_frequencies(frequencies, fs);
    const std::vector<fgz::Coefficients> coeffs = coefficients_for(freqs);
    const std::size_t m = freqs.size();
    const auto w = static_cast<std::size_t>(window_size);
    const auto st = static_cast<std::size_t>(step);
    const SlidingSpec sp{total, w, st, (total - w) / st + 1};

    py::array_t<std::complex<double>> result = resolve_out<std::complex<double>, 2>(
        out, {checked_ssize(sp.n_windows), checked_ssize(m)}, "complex128", x);
    auto dst = result.mutable_unchecked<2>();
    const double* data = x.data();
    {
        py::gil_scoped_release release;
        sliding_many(data, sp, coeffs, freqs, nthreads,
                     [&](std::size_t wi, std::size_t fi, fgz::Complex Y) noexcept {
                         dst(checked_ssize(wi), checked_ssize(fi)) =
                             std::complex<double>(Y.re, Y.im);
                     });
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
        // Large chunks go through the block-parallel lane kernel and are folded in with
        // the shift theorem; small ones extend the open recurrence chain (no trig per push).
        constexpr std::size_t kBlockedPushMin = 4096;
        if (n >= kBlockedPushMin) {
            const fgz::Complex chunk =
                dtft_one(fgk::active_api(), data, n, core.coeffs(), FreqTwiddle{core.frequency()});
            core.absorb(chunk, n);
        } else {
            core.push(data, n);
        }
    }

    fgz::Complex result_or_throw() const {
        if (core.count() == 0) throw py::value_error("no samples have been pushed");
        return core.result();
    }

    py::tuple result(const std::string& scale) const {
        const Scale sc = parse_scale(scale);
        const fgz::Complex X = result_or_throw();
        return py::make_tuple(scaled(X, core.count(), fgz::is_dc_or_nyquist(core.frequency()), sc),
                              fgz::phase(X));
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
        Compiled core of fastgoertzel; import the public names from ``fastgoertzel``.

        Evaluates the discrete-time Fourier transform of a real signal at chosen
        frequencies with the (generalized) Goertzel algorithm, in C++.

        Functions: goertzel, goertzel_complex, goertzel_bin, goertzel_batch,
        goertzel_batch_complex, goertzel_sliding_batch, goertzel_sliding_batch_complex.
        Class: GoertzelFilter. Attributes: __version__, _has_openmp, _kernel.
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

        Notes
        -----
        Single-threaded by design (predictable latency). Signals of 1024 samples or more
        are split into independent blocks of at least 512 samples (up to eight) so that
        the recurrence's dependency chain does not limit throughput; the blocks are
        combined exactly.

        Examples
        --------
        >>> import numpy as np, fastgoertzel
        >>> t = np.arange(4410)  # 44 whole cycles of 440 Hz at 44.1 kHz
        >>> x = np.cos(2 * np.pi * 440 / 44100 * t + 0.25)
        >>> amp, phase = fastgoertzel.goertzel(x, 440, fs=44100)
        >>> round(amp, 3), round(phase, 3)
        (1.0, 0.25)
        )pbdoc");

    m.def("goertzel_complex", &goertzel_complex, py::arg("x"), py::arg("frequency"), py::kw_only(),
          py::arg("fs") = py::none(),
          R"pbdoc(
        Unscaled complex DTFT value ``X = sum x[n] exp(-2j*pi*f*n)`` at ``frequency``.

        Same arguments and validation as :func:`goertzel`. Use this when you need the
        complex value or want to avoid the ``sqrt``/``atan2`` of the amplitude form.
        )pbdoc");

    m.def("goertzel_bin", &goertzel_bin, py::arg("x"), py::arg("k"), py::kw_only(),
          py::arg("scale") = "amplitude",
          R"pbdoc(
        Amplitude and phase of DFT bin ``k`` of ``x`` (``k`` in ``[0, len(x))``).

        Equivalent to ``goertzel(x, k / len(x))`` but with the twiddles computed in exact
        integer arithmetic, so it matches ``numpy.fft.fft(x)[k]`` as closely as the
        recurrence allows. This is what versions 1.x computed for every call after
        truncating ``frequency * len(x)`` to an integer.
        )pbdoc");

    m.def("goertzel_batch", &goertzel_batch, py::arg("x"), py::arg("frequencies"), py::kw_only(),
          py::arg("fs") = py::none(), py::arg("scale") = "amplitude",
          py::arg("threads") = py::none(), py::arg("out") = py::none(),
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
            without OpenMP. Results do not depend on the thread count.
        out : numpy.ndarray, optional
            Preallocated result array: float64, C-contiguous, writeable, shape
            ``(len(frequencies), 2)``, not sharing memory with ``x``. The call then
            allocates nothing and returns ``out``.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(frequencies), 2)``: column 0 is the scaled value, column 1 the
            phase. Row ``i`` is ``goertzel(x, frequencies[i], ...)`` up to rounding.

        Notes
        -----
        Up to eight frequencies share one pass over the signal; fewer frequencies are
        padded with signal blocks of at least 512 samples so that up to eight independent
        chains run, and large batches use more lanes still. The parallel region is
        entered only when ``len(x) * len(frequencies) >= 65536``.
        The GIL is released during computation.
        )pbdoc");

    m.def("goertzel_batch_complex", &goertzel_batch_complex, py::arg("x"), py::arg("frequencies"),
          py::kw_only(), py::arg("fs") = py::none(), py::arg("threads") = py::none(),
          py::arg("out") = py::none(),
          R"pbdoc(
        :func:`goertzel_complex` for every entry of ``frequencies``; returns a complex128
        array of shape ``(len(frequencies),)``. ``out`` may be a preallocated complex128
        array of that shape, as in :func:`goertzel_batch`.
        )pbdoc");

    m.def("goertzel_sliding_batch", &goertzel_sliding_batch, py::arg("x"), py::arg("window_size"),
          py::arg("frequencies"), py::kw_only(), py::arg("step") = 1, py::arg("fs") = py::none(),
          py::arg("scale") = "amplitude", py::arg("threads") = py::none(),
          py::arg("out") = py::none(),
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
        out : numpy.ndarray, optional
            Preallocated result array: float64, C-contiguous, writeable, shape
            ``(n_windows, len(frequencies), 2)``, not sharing memory with ``x``. With a
            reused ``out`` the call allocates nothing; see Notes.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_windows, len(frequencies), 2)``.

        Notes
        -----
        When ``4 * step < window_size`` consecutive windows are obtained with the sliding
        DFT update, which costs a constant amount per hop regardless of ``window_size``;
        every 256 windows the value is recomputed exactly, so rounding error cannot
        accumulate. Otherwise each window is computed from scratch. Work is distributed
        over OpenMP threads by (frequency, 256-window chunk); results do not depend on the
        thread count. The GIL is released.

        The result array is large (``n_windows * len(frequencies) * 16`` bytes) and, when
        freshly allocated, is page-faulted in by the writing threads, which serialises in
        the kernel and bounds the thread scaling. In a loop that calls this repeatedly,
        pass the same ``out`` each time.
        )pbdoc");

    m.def("goertzel_sliding_batch_complex", &goertzel_sliding_batch_complex, py::arg("x"),
          py::arg("window_size"), py::arg("frequencies"), py::kw_only(), py::arg("step") = 1,
          py::arg("fs") = py::none(), py::arg("threads") = py::none(), py::arg("out") = py::none(),
          R"pbdoc(
        :func:`goertzel_sliding_batch` returning the unscaled complex DTFT of every window.

        Same arguments as :func:`goertzel_sliding_batch` without ``scale``. Returns a
        complex128 array of shape ``(n_windows, len(frequencies))``, or ``out`` if given. Skips the ``sqrt`` and
        ``atan2`` per window, which dominate the cost of the (value, phase) form once the
        sliding update has made the window length irrelevant.
        )pbdoc");

    py::class_<PyFilter>(m, "GoertzelFilter",
                         R"pbdoc(
        Stateful Goertzel filter for streaming input.

        Create it for one frequency, ``push`` samples as they arrive (scalars or arrays,
        any chunking), and read ``result()`` at any time; it is ``goertzel`` applied to
        everything pushed so far (equal to rounding). ``reset()`` clears the state. Instances are cheap and
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
        .def(
            "reset", [](PyFilter& self) { self.core.reset(); }, "Forget all pushed samples.")
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
            std::string r = "GoertzelFilter(frequency=" + num(self.frequency_as_given);
            if (self.fs) r += ", fs=" + num(*self.fs);
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
    m.attr("_kernel") = fgk::active_api().name;
}
