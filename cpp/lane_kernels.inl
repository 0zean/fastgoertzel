// SPDX-License-Identifier: GPL-3.0-or-later
//
// Hot loops. This file is #included INSIDE an architecture namespace by kernels_baseline.cpp
// and kernels_avx2.cpp, which are compiled with different instruction-set flags. Rules:
//   * everything here has internal linkage (anonymous namespace), so the two copies can
//     never be merged by the linker;
//   * nothing here calls inline functions from goertzel.hpp (an emitted copy of such a
//     function could be merged across TUs and carry the wrong instruction set);
//   * no validation; callers guarantee count in [1, kMaxLanes] and in-range indices.

namespace {

using fastgoertzel::Complex;
using fastgoertzel::State;
using fastgoertzel::kernels::LaneInput;
using fastgoertzel::kernels::SlidingConsts;

// L independent recurrences advanced in lockstep: L dependency chains interleave, so the
// FMA units are kept busy instead of waiting ~4 cycles per sample on one chain. Lanes may
// have different lengths; the common prefix runs in lockstep and the tails run alone.
template <std::size_t L>
void run_lanes_t(const LaneInput* lanes, State* out) noexcept {
    double s1[L];
    double s2[L];
    double cf[L];
    const double* xp[L];
    std::size_t common = lanes[0].len;
    for (std::size_t l = 0; l < L; ++l) {
        s1[l] = lanes[l].s1;
        s2[l] = lanes[l].s2;
        cf[l] = lanes[l].coeff;
        xp[l] = lanes[l].x;
        if (lanes[l].len < common) common = lanes[l].len;
    }
    for (std::size_t i = 0; i < common; ++i) {
        for (std::size_t l = 0; l < L; ++l) {
            const double t = xp[l][i] - s2[l];
            const double s0 = cf[l] * s1[l] + t;
            s2[l] = s1[l];
            s1[l] = s0;
        }
    }
    for (std::size_t l = 0; l < L; ++l) {
        double a = s1[l];
        double b = s2[l];
        const double c = cf[l];
        const double* x = xp[l];
        for (std::size_t i = common; i < lanes[l].len; ++i) {
            const double t = x[i] - b;
            const double s0 = c * a + t;
            b = a;
            a = s0;
        }
        out[l] = State{a, b};
    }
}

void run_lanes(const LaneInput* lanes, int count, State* out) noexcept {
    switch (count) {
        case 8:
            run_lanes_t<8>(lanes, out);
            break;
        case 7:
            run_lanes_t<7>(lanes, out);
            break;
        case 6:
            run_lanes_t<6>(lanes, out);
            break;
        case 5:
            run_lanes_t<5>(lanes, out);
            break;
        case 4:
            run_lanes_t<4>(lanes, out);
            break;
        case 3:
            run_lanes_t<3>(lanes, out);
            break;
        case 2:
            run_lanes_t<2>(lanes, out);
            break;
        case 1:
            run_lanes_t<1>(lanes, out);
            break;
        default:
            break;
    }
}

// Sliding DFT: each hop of one sample costs one complex rotation and two real updates,
// independent of the window length. Exactness is restored by the caller every chunk.
void sliding_updates(const double* x, std::size_t window, std::size_t step, std::size_t count,
                     Complex y0, const SlidingConsts& k, Complex* out) noexcept {
    double yr = y0.re;
    double yi = y0.im;
    out[0] = y0;
    std::size_t pos = 0;
    for (std::size_t w = 1; w < count; ++w) {
        for (std::size_t s = 0; s < step; ++s, ++pos) {
            const double leaving = x[pos];
            const double entering = x[pos + window];
            const double ar = yr - leaving;
            const double ai = yi;
            yr = k.rot_re * ar - k.rot_im * ai + entering * k.tail_re;
            yi = k.rot_re * ai + k.rot_im * ar + entering * k.tail_im;
        }
        out[w] = Complex{yr, yi};
    }
}

}  // namespace
