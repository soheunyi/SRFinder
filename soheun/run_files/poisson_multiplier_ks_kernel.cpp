#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Compute centered weighted-KS multiplier statistics for a batch of supplied
// Poisson count arrays.  Python owns random-number generation so reference and
// compiled implementations consume identical replicates.
extern "C" int poisson_multiplier_ks_batch(
    std::size_t support_size,
    std::size_t n3,
    std::size_t n4,
    std::size_t replicates,
    const std::int64_t* support_index_3,
    const double* normalized_weight_3,
    const double* cdf_3,
    const std::int64_t* support_index_4,
    const double* normalized_weight_4,
    const double* cdf_4,
    const std::int64_t* counts_3,
    const std::int64_t* counts_4,
    int direct_normalized,
    double* output) {
  if (!support_size || !n3 || !n4 || !replicates) return 1;
  if (direct_normalized != 0 && direct_normalized != 1) return 2;

  std::vector<double> mass3(support_size);
  std::vector<double> mass4(support_size);
  for (std::size_t b = 0; b < replicates; ++b) {
    std::fill(mass3.begin(), mass3.end(), 0.0);
    std::fill(mass4.begin(), mass4.end(), 0.0);
    double total3 = 0.0;
    double total4 = 0.0;

    const std::int64_t* current3 = counts_3 + b * n3;
    const std::int64_t* current4 = counts_4 + b * n4;
    for (std::size_t i = 0; i < n3; ++i) {
      if (current3[i] < 0) return 3;
      const double multiplier = direct_normalized
          ? static_cast<double>(current3[i])
          : static_cast<double>(current3[i] - 1);
      const double contribution = multiplier * normalized_weight_3[i];
      const std::int64_t index = support_index_3[i];
      if (index < 0 || static_cast<std::size_t>(index) >= support_size) return 4;
      mass3[static_cast<std::size_t>(index)] += contribution;
      total3 += contribution;
    }
    for (std::size_t i = 0; i < n4; ++i) {
      if (current4[i] < 0) return 3;
      const double multiplier = direct_normalized
          ? static_cast<double>(current4[i])
          : static_cast<double>(current4[i] - 1);
      const double contribution = multiplier * normalized_weight_4[i];
      const std::int64_t index = support_index_4[i];
      if (index < 0 || static_cast<std::size_t>(index) >= support_size) return 4;
      mass4[static_cast<std::size_t>(index)] += contribution;
      total4 += contribution;
    }

    if (direct_normalized && (!(total3 > 0.0) || !(total4 > 0.0))) {
      output[b] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    double cumulative3 = 0.0;
    double cumulative4 = 0.0;
    double maximum = 0.0;
    for (std::size_t j = 0; j < support_size; ++j) {
      cumulative3 += mass3[j];
      cumulative4 += mass4[j];
      const double process3 = direct_normalized
          ? cumulative3 / total3 - cdf_3[j]
          : cumulative3 - cdf_3[j] * total3;
      const double process4 = direct_normalized
          ? cumulative4 / total4 - cdf_4[j]
          : cumulative4 - cdf_4[j] * total4;
      maximum = std::max(maximum, std::abs(process3 - process4));
    }
    output[b] = maximum;
  }
  return 0;
}
