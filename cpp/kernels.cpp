// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime selection of the hot-loop implementation.

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS  // std::getenv is fine here; MSVC's C4996 is noise
#endif

#include "kernels_api.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fastgoertzel::arch::baseline {
extern const kernels::Api api;
}
#if defined(FASTGOERTZEL_HAVE_AVX2_TU)
namespace fastgoertzel::arch::avx2 {
extern const kernels::Api api;
}
#endif

namespace fastgoertzel::kernels {
namespace {

bool cpu_has_avx2_fma() noexcept {
#if !defined(FASTGOERTZEL_HAVE_AVX2_TU)
    return false;
#elif defined(_MSC_VER)
    // Checks CPUID and OS (XSAVE) support; FALSE on Windows versions that predate the flag.
    return IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

const Api* select() noexcept {
    const char* env = std::getenv("FASTGOERTZEL_KERNEL");
    const bool force_baseline = env != nullptr && std::strcmp(env, "baseline") == 0;
    const bool force_avx2 = env != nullptr && std::strcmp(env, "avx2") == 0;
#if defined(FASTGOERTZEL_HAVE_AVX2_TU)
    if (force_avx2) return &arch::avx2::api;
    if (!force_baseline && cpu_has_avx2_fma()) return &arch::avx2::api;
#else
    (void)force_avx2;
    (void)force_baseline;
    (void)cpu_has_avx2_fma;
#endif
    return &arch::baseline::api;
}

}  // namespace

const Api& active_api() noexcept {
    static const Api* const chosen = select();  // thread-safe one-time initialisation
    return *chosen;
}

}  // namespace fastgoertzel::kernels
