#ifndef FASTGOERTZEL_GOERTZEL_HPP
#define FASTGOERTZEL_GOERTZEL_HPP

#include <cmath>
#include <tuple>
#include <vector>
#include <cstddef>

namespace fastgoertzel {

// Core algorithm class
class GoertzelAlgorithm {
public:
	// Main computation function
	static std::tuple<double, double> compute(
		const double* data,
		size_t length,
		double frequency
	);

	// Overload for std::vector
	static std::tuple<double, double> compute(
		const std::vector<double>& data,
		double frequency
	);

private:
	// Constants
	static constexpr double PI = 3.141592653589793238462643383279502884;

	// Helper function
	static inline double compute_coefficient(double frequency, size_t length);
};

// Utility functions for validation
namespace utils {
	bool is_valid_frequency(double freq);
	bools is_valid_data_length(size_t length);
}

} // namespace fastgoertzel

#endif // FASTGOERTZEL_GOERTZEL_HPP
