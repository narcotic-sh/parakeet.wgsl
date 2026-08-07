// This exact weighted 1D k-means implementation is derived from kmeans1d's
// MIT-licensed _core.cpp bundled with the pinned coremltools 9.0 dependency.
//
// Copyright (c) 2019 Daniel Steinberg
// Copyright © 2023 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "kmeans1d_sorted_unique.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

namespace {

using Index = std::uint32_t;

class WeightedCostCalculator {
 public:
  WeightedCostCalculator(
      const double* values,
      const double* weights,
      Index n)
      : cumulative_weight_(static_cast<std::size_t>(n) + 1),
        cumulative_sum_(static_cast<std::size_t>(n) + 1),
        cumulative_sum_squared_(static_cast<std::size_t>(n) + 1) {
    cumulative_weight_[0] = 0.0;
    cumulative_sum_[0] = 0.0;
    cumulative_sum_squared_[0] = 0.0;
    for (Index i = 0; i < n; ++i) {
      const double x = values[i];
      const double w = weights[i];
      cumulative_weight_[static_cast<std::size_t>(i) + 1] =
          w + cumulative_weight_[i];
      cumulative_sum_[static_cast<std::size_t>(i) + 1] =
          w * x + cumulative_sum_[i];
      cumulative_sum_squared_[static_cast<std::size_t>(i) + 1] =
          w * x * x + cumulative_sum_squared_[i];
    }
  }

  inline double weight(Index i, Index j) const {
    return i <= j
        ? cumulative_weight_[static_cast<std::size_t>(j) + 1]
            - cumulative_weight_[i]
        : 0.0;
  }

  inline double calculate(Index i, Index j) const {
    if (j < i) {
      return 0.0;
    }
    const double w = weight(i, j);
    const double sum =
        cumulative_sum_[static_cast<std::size_t>(j) + 1]
        - cumulative_sum_[i];
    const double mean = sum / w;
    double result =
        cumulative_sum_squared_[static_cast<std::size_t>(j) + 1]
        - cumulative_sum_squared_[i];
    result += w * (mean * mean);
    result -= (2 * mean) * sum;
    return result;
  }

 private:
  std::vector<double> cumulative_weight_;
  std::vector<double> cumulative_sum_;
  std::vector<double> cumulative_sum_squared_;
};

template <typename Lookup>
void smawk_recursive(
    const std::vector<Index>& rows,
    const std::vector<Index>& columns,
    const Lookup& lookup,
    std::vector<Index>* result,
    std::vector<Index>* column_positions) {
  if (rows.empty()) {
    return;
  }

  std::vector<Index> reduced_columns;
  reduced_columns.reserve(rows.size());
  for (const Index column : columns) {
    while (true) {
      if (reduced_columns.empty()) {
        break;
      }
      const Index row = rows[reduced_columns.size() - 1];
      // Keep the earlier column on an equal cost, matching coremltools 9.
      if (lookup(row, column) >= lookup(row, reduced_columns.back())) {
        break;
      }
      reduced_columns.pop_back();
    }
    if (reduced_columns.size() < rows.size()) {
      reduced_columns.push_back(column);
    }
  }

  std::vector<Index> odd_rows;
  odd_rows.reserve(rows.size() / 2);
  for (std::size_t i = 1; i < rows.size(); i += 2) {
    odd_rows.push_back(rows[i]);
  }
  smawk_recursive(
      odd_rows,
      reduced_columns,
      lookup,
      result,
      column_positions);

  for (Index position = 0;
       position < static_cast<Index>(reduced_columns.size());
       ++position) {
    (*column_positions)[reduced_columns[position]] = position;
  }

  Index start = 0;
  for (std::size_t r = 0; r < rows.size(); r += 2) {
    const Index row = rows[r];
    Index stop = static_cast<Index>(reduced_columns.size() - 1);
    if (r < rows.size() - 1) {
      stop = (*column_positions)[(*result)[rows[r + 1]]];
    }
    Index argmin = reduced_columns[start];
    double minimum = lookup(row, argmin);
    for (Index c = start + 1; c <= stop; ++c) {
      const double value = lookup(row, reduced_columns[c]);
      // Keep the first argmin on an equal cost, matching coremltools 9.
      if (c == start || value < minimum) {
        argmin = reduced_columns[c];
        minimum = value;
      }
    }
    (*result)[row] = argmin;
    start = stop;
  }
}

template <typename Lookup>
std::vector<Index> smawk(
    Index number_of_rows,
    Index number_of_columns,
    const Lookup& lookup) {
  std::vector<Index> result(number_of_rows);
  std::vector<Index> rows(number_of_rows);
  std::iota(rows.begin(), rows.end(), Index{0});
  std::vector<Index> columns(number_of_columns);
  std::iota(columns.begin(), columns.end(), Index{0});
  std::vector<Index> column_positions(number_of_columns);
  smawk_recursive(rows, columns, lookup, &result, &column_positions);
  return result;
}

int cluster_sorted_unique_weighted(
    const double* values,
    const double* weights,
    Index n,
    Index k,
    Index* clusters,
    double* centroids) {
  if (values == nullptr
      || weights == nullptr
      || clusters == nullptr
      || centroids == nullptr
      || n == 0
      || k == 0
      || k > n) {
    return PARAKEET_KMEANS1D_INVALID_ARGUMENT;
  }

  const std::size_t n_size = n;
  const std::size_t k_size = k;
  if (k_size > std::numeric_limits<std::size_t>::max() / n_size) {
    return PARAKEET_KMEANS1D_INVALID_ARGUMENT;
  }

  const WeightedCostCalculator cost_calculator(values, weights, n);
  std::vector<double> previous_costs(n);
  std::vector<double> current_costs(n);
  std::vector<Index> backpointers(k_size * n_size);

  for (Index i = 0; i < n; ++i) {
    previous_costs[i] = cost_calculator.calculate(0, i);
    backpointers[i] = 0;
  }

  for (Index cluster = 1; cluster < k; ++cluster) {
    const auto lookup =
        [&previous_costs, &cost_calculator](Index i, Index j) -> double {
      // j is unsigned. At j == 0, j - 1 wraps and selects i, exactly as in
      // coremltools' unsigned-long implementation.
      const Index column = i < j - 1 ? i : j - 1;
      return previous_costs[column] + cost_calculator.calculate(j, i);
    };
    const std::vector<Index> row_argmins = smawk(n, n, lookup);
    const std::size_t backpointer_row =
        static_cast<std::size_t>(cluster) * n_size;
    for (Index i = 0; i < n; ++i) {
      const Index argmin = row_argmins[i];
      const double minimum = lookup(i, argmin);
      current_costs[i] = minimum;
      backpointers[backpointer_row + i] = argmin;
    }
    previous_costs.swap(current_costs);
  }

  Index t = n;
  Index cluster = k - 1;
  Index last_element = n - 1;
  do {
    const Index previous_t = t;
    t = backpointers[
        static_cast<std::size_t>(cluster) * n_size + last_element];
    double centroid = 0.0;
    for (Index i = t; i < previous_t; ++i) {
      clusters[i] = cluster;
      // Preserve the reference helper's incremental weighted-mean operation
      // order so the returned f64 centroid bits remain identical.
      centroid += (
          (values[i] - centroid)
          * cost_calculator.weight(i, i)
          / cost_calculator.weight(t, i)
      );
    }
    centroids[cluster] = centroid;
    cluster -= 1;
    last_element = t - 1;
  } while (t > 0);

  return PARAKEET_KMEANS1D_OK;
}

}  // namespace

extern "C" {

int parakeet_kmeans1d_sorted_unique_weighted_f64(
    const double* values,
    const double* weights,
    std::uint32_t n,
    std::uint32_t k,
    std::uint32_t* clusters,
    double* centroids) noexcept {
  try {
    return cluster_sorted_unique_weighted(
        values,
        weights,
        n,
        k,
        clusters,
        centroids);
  } catch (const std::bad_alloc&) {
    return PARAKEET_KMEANS1D_ALLOCATION_FAILURE;
  } catch (...) {
    return PARAKEET_KMEANS1D_INTERNAL_ERROR;
  }
}

}  // extern "C"
