#ifndef PARAKEET_KMEANS1D_SORTED_UNIQUE_H_
#define PARAKEET_KMEANS1D_SORTED_UNIQUE_H_

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define PARAKEET_KMEANS1D_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define PARAKEET_KMEANS1D_API __attribute__((visibility("default")))
#else
#define PARAKEET_KMEANS1D_API
#endif

#ifdef __cplusplus
extern "C" {
#define PARAKEET_KMEANS1D_NOEXCEPT noexcept
#else
#define PARAKEET_KMEANS1D_NOEXCEPT
#endif

enum ParakeetKmeans1dStatus {
  PARAKEET_KMEANS1D_OK = 0,
  PARAKEET_KMEANS1D_INVALID_ARGUMENT = 1,
  PARAKEET_KMEANS1D_ALLOCATION_FAILURE = 2,
  PARAKEET_KMEANS1D_INTERNAL_ERROR = 3,
};

// Exact weighted one-dimensional k-means for an already-sorted unique input.
//
// Preconditions:
// - values contains n strictly increasing, finite doubles;
// - weights contains n finite, strictly positive doubles;
// - 0 < k <= n <= UINT32_MAX;
// - clusters has room for n uint32_t values;
// - centroids has room for k doubles.
//
// Cluster IDs correspond directly to the sorted input order. The function does
// not validate the O(n) sorted/finite/positive preconditions so the converter
// can validate once and call without a redundant pass.
PARAKEET_KMEANS1D_API int
parakeet_kmeans1d_sorted_unique_weighted_f64(
    const double* values,
    const double* weights,
    uint32_t n,
    uint32_t k,
    uint32_t* clusters,
    double* centroids) PARAKEET_KMEANS1D_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef PARAKEET_KMEANS1D_NOEXCEPT

#endif  // PARAKEET_KMEANS1D_SORTED_UNIQUE_H_
