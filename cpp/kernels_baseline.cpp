// SPDX-License-Identifier: GPL-3.0-or-later
//
// Baseline instruction-set build of the hot loops (whatever the compiler's default target
// is: SSE2 on x86-64, NEON on arm64, or -march=native when FASTGOERTZEL_NATIVE is set).

#include "kernels_api.hpp"

#include <cstddef>

namespace fastgoertzel::arch::baseline {

#include "lane_kernels.inl"

// `extern` first: a const object at namespace scope has internal linkage by default.
extern const kernels::Api api;
const kernels::Api api{"baseline", &run_lanes, &sliding_updates};

}  // namespace fastgoertzel::arch::baseline
