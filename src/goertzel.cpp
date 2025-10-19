#include "goertzel.hpp"
#include <cstring>
#include <stdexcept>
#include <memory>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fastgoertzel {

// --- Custom Aligned Allocator Example ---
template<typename T, std::size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) throw std::bad_alloc();
#endif
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

// --- AVX2-optimized Goertzel implementation ---
#if defined(__AVX2__)
std::tuple<double, double> goertzel_avx2(
    const double* data,
    size_t length,
    double frequency
) {
    // Only use AVX2 for large enough arrays
    if (length < 32) {
        // Fallback to scalar for small arrays
        return GoertzelAlgorithm::compute(data, length, frequency);
    }

    const size_t k = static_cast<size_t>(frequency * length);
    static constexpr double PI = 3.141592653589793238462643383279502884;
    const double omega = 2.0 * PI * k / static_cast<double>(length);
    const double cos_omega = std::cos(omega);
    const double sin_omega = std::sin(omega);
    const double coeff = 2.0 * cos_omega;

    // State vectors (initialize to zero)
    __m256d s_prev = _mm256_setzero_pd();
    __m256d s_prev2 = _mm256_setzero_pd();

    size_t i = 0;
    // Process 4 samples at a time
    for (; i + 4 <= length; i += 4) {
        __m256d x = _mm256_loadu_pd(&data[i]);
        __m256d s_curr = _mm256_add_pd(x, _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(coeff), s_prev), s_prev2));
        s_prev2 = s_prev;
        s_prev = s_curr;
    }

    // Horizontal reduction: sum up the last two states
    alignas(32) double s_prev_arr[4], s_prev2_arr[4];
    _mm256_store_pd(s_prev_arr, s_prev);
    _mm256_store_pd(s_prev2_arr, s_prev2);

    double s_prev_sum = 0.0, s_prev2_sum = 0.0;
    for (int j = 0; j < 4; ++j) {
        s_prev_sum += s_prev_arr[j];
        s_prev2_sum += s_prev2_arr[j];
    }

    // Handle remaining samples (scalar)
    double s_prev_scalar = 0.0, s_prev2_scalar = 0.0;
    for (; i < length; ++i) {
        double s_curr = data[i] + coeff * s_prev_scalar - s_prev2_scalar;
        s_prev2_scalar = s_prev_scalar;
        s_prev_scalar = s_curr;
    }

    s_prev_sum += s_prev_scalar;
    s_prev2_sum += s_prev2_scalar;

    // Final calculation
    const double real_part = cos_omega * s_prev_sum - s_prev2_sum;
    const double imag_part = sin_omega * s_prev_sum;
    const double amplitude = std::sqrt(real_part * real_part + imag_part * imag_part) / (static_cast<double>(length) / 2.0);
    const double phase = std::atan2(imag_part, real_part);

    return std::make_tuple(amplitude, phase);
}
#endif

std::tuple<double, double> GoertzelAlgorithm::compute(
    const double* data,
    size_t length,
    double frequency
) {
    // Validate inputs
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (length == 0) {
        throw std::invalid_argument("Data length must be positive");
    }
    if (frequency < 0.0 || frequency >= 1.0) {
        throw std::invalid_argument("Frequency must be in [0, 1) range");
    }

    const size_t k = static_cast<size_t>(frequency * length);
    const double omega = 2.0 * PI * k / static_cast<double>(length);
    const double cos_omega = std::cos(omega);
    const double sin_omega = std::sin(omega);
    const double coeff = 2.0 * cos_omega;

    double s_prev = 0.0;
    double s_prev2 = 0.0;

    // Loop unrolling for speed (example: unroll by 2)
    size_t i = 0;
    for (; i + 1 < length; i += 2) {
        double s_curr = data[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s_curr;

        s_curr = data[i + 1] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s_curr;
    }
    // Handle last sample if length is odd
    for (; i < length; ++i) {
        double s_curr = data[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s_curr;
    }

    const double real_part = cos_omega * s_prev - s_prev2;
    const double imag_part = sin_omega * s_prev;
    const double amplitude = std::sqrt(real_part * real_part + imag_part * imag_part) / (static_cast<double>(length) / 2.0);
    const double phase = std::atan2(imag_part, real_part);

    return std::make_tuple(amplitude, phase);
}

std::tuple<double, double> GoertzelAlgorithm::compute(
    const std::vector<double>& data,
    double frequency
) {
    // Example usage of aligned allocator for future extensibility
    // std::vector<double, AlignedAllocator<double>> aligned_data(data.begin(), data.end());
    // return compute(aligned_data.data(), aligned_data.size(), frequency);
    return compute(data.data(), data.size(), frequency);
}

double GoertzelAlgorithm::compute_coefficient(double frequency, size_t length) {
    const size_t k = static_cast<size_t>(frequency * length);
    const double omega = 2.0 * PI * k / static_cast<double>(length);
    return 2.0 * std::cos(omega);
}

namespace utils {

bool is_valid_frequency(double freq) {
    return freq >= 0.0 && freq < 1.0;
}

bool is_valid_data_length(size_t length) {
    return length > 0 && length <= 1e9; // Reasonable upper bound
}

} // namespace utils
} // namespace fastgoertzel