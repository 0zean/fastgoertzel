#include "goertzel.hpp"
#include <cstring>
#include <stdexcept>

namespace fastgoertzel {

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

	// Compute the target bin
	const size_t k = static_cast<size_t>(frequency * length);

	// Precompute constants
	const double omega = 2.0 * PI * k / static_cast<double>(length);
	const double cos_omega = std::cos(omega);
	const double sin_omega = std::sin(omega);
	const double coeff = 2.0 * cos_omega;

	// Goerztel recursion
	double s_prev = 0.0;
	double s_prev2 = 0.0;

	// Main loop - optimized for auto-vectorization
	for (size_t i = 0; i < length; i++) {
		const double s_curr = data[i] + coeff * s_prev - s_prev2;
		s_prev2 = s_prev;
		s_prev = s_curr;
	}

	// Compute final values
	const double real_part = cos_omega * s_prev - s_prev2;
	const double imag_part = sin_omega * s_prev;

	// Calculate amplitude and phase
	const double amplitude = std::sqrt(real_part * real_part + imag_part * imag_part) / (static_cast<double>(length) / 2.0);
	const double phase = std::atan2(imag_part, real_part);

	return std::make_tuple(amplitude, phase);
}

std::tuple<double, double> GoertzelAlgorithm::compute(
	const std::vector<double>& data,
	double frequency
) {
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