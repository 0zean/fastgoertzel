// SPDX-License-Identifier: GPL-3.0-or-later
//
// AVX2 + FMA build of the hot loops. CMake compiles this file with -mavx2 -mfma (GCC/Clang)
// or /arch:AVX2 (MSVC) on x86 targets only; kernels.cpp selects it at runtime when the CPU
// and OS support those instructions. Nothing in this TU may be called except through the
// `api` function pointers.

#include "kernels_api.hpp"

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

namespace fastgoertzel::arch::avx2 {

#include "lane_kernels.inl"

// `extern` first: a const object at namespace scope has internal linkage by default.
extern const kernels::Api api;
const kernels::Api api{"avx2", &run_lanes, &sliding_updates};

}  // namespace fastgoertzel::arch::avx2

#else

namespace fastgoertzel::arch::avx2 {
// Not an x86 target: the dispatcher never references this symbol, but keep the TU non-empty
// so the build is uniform.
const int api_unavailable = 1;
}  // namespace fastgoertzel::arch::avx2

#endif
