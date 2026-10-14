// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime selection of the hot-loop implementation.

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS  // std::getenv is fine here; MSVC's C4996 is noise
#endif

#include "kernels_api.hpp"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

std::string lowercase(const char* s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// FASTGOERTZEL_KERNEL=auto|baseline|avx2 (case-insensitive; unset or empty means auto).
// A value that is not one of these, or avx2 on a machine that cannot run it, is an error
// rather than a silent fallback: this is a tuning knob, and a typo must not go unnoticed,
// while an unconditional avx2 would die with an illegal-instruction signal later.
const Api* select() {
    const char* env = std::getenv("FASTGOERTZEL_KERNEL");
    const std::string mode = (env == nullptr || *env == '\0') ? "auto" : lowercase(env);
    if (mode != "auto" && mode != "baseline" && mode != "avx2") {
        throw std::runtime_error("FASTGOERTZEL_KERNEL must be 'auto', 'baseline' or 'avx2' (got '" +
                                 std::string(env) + "')");
    }
    const bool have_avx2 = cpu_has_avx2_fma();
#if defined(FASTGOERTZEL_HAVE_AVX2_TU)
    if (mode == "avx2") {
        if (!have_avx2) {
            throw std::runtime_error(
                "FASTGOERTZEL_KERNEL=avx2 requested, but this CPU or operating system does not "
                "support AVX2 and FMA");
        }
        return &arch::avx2::api;
    }
    if (mode == "auto" && have_avx2) return &arch::avx2::api;
#else
    (void)have_avx2;
    if (mode == "avx2") {
        throw std::runtime_error(
            "FASTGOERTZEL_KERNEL=avx2 requested, but this build has no AVX2 kernel (not x86)");
    }
#endif
    return &arch::baseline::api;
}

}  // namespace

const Api& active_api() {
    // Thread-safe one-time initialisation. The first call happens during module import
    // (the `_kernel` attribute), where an exception becomes a Python ImportError-time
    // RuntimeError; later calls cannot throw.
    static const Api* const chosen = select();
    return *chosen;
}

}  // namespace fastgoertzel::kernels
