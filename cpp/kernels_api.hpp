// SPDX-License-Identifier: GPL-3.0-or-later
//
// Interface between the pybind11 layer and the hot loops.
//
// The hot loops (lane_kernels.inl) are compiled once per instruction set, each in its own
// translation unit and namespace, and reached only through the function pointers in `Api`.
// Everything that is not a hot loop (validation, planning, twiddles, combination) is
// compiled once, in the baseline TU, so no ISA-specific code can leak into it.

#ifndef FASTGOERTZEL_KERNELS_API_HPP
#define FASTGOERTZEL_KERNELS_API_HPP

#include "fastgoertzel/goertzel.hpp"

#include <cstddef>

namespace fastgoertzel::kernels {

/// Hard upper bound on chains run in lockstep by one call to Api::run_lanes.
constexpr int kMaxLanes = 8;

/// One independent Goertzel chain: `len` contiguous samples at `x`, recurrence coefficient
/// `coeff`, starting from state (s1, s2).
struct LaneInput {
    const double* x;
    std::size_t len;
    double coeff;
    double s1;
    double s2;
};

/// Constants for the sliding-DFT update at one frequency:
///   Y_{w+1} = rot * (Y_w - x[w]) + x[w+W] * tail,   rot = e^{+jw}, tail = e^{-jw(W-1)}.
struct SlidingConsts {
    double rot_re;
    double rot_im;
    double tail_re;
    double tail_im;
};

struct Api {
    const char* name;

    /// Runs `count` (1..kMaxLanes) lanes and writes the final (s1, s2) of lane i to out[i].
    void (*run_lanes)(const LaneInput* lanes, int count, State* out) noexcept;

    /// Starting from `y0` (the window-relative DTFT of the window beginning at x[0]),
    /// produces the DTFT of the next `count - 1` windows spaced `step` samples apart.
    /// out[0] = y0, out[i] = value for the window starting at x[i * step].
    void (*sliding_updates)(const double* x, std::size_t window, std::size_t step,
                            std::size_t count, Complex y0, const SlidingConsts& k,
                            Complex* out) noexcept;
};

/// The implementation selected for this process (CPU detection at first use, overridable
/// with the environment variable FASTGOERTZEL_KERNEL=baseline|avx2|auto).
const Api& active_api() noexcept;

}  // namespace fastgoertzel::kernels

#endif  // FASTGOERTZEL_KERNELS_API_HPP
