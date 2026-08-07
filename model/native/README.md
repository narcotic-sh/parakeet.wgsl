# Native exact weighted 1D k-means

`kmeans1d_sorted_unique.cpp` is the converter-only native implementation of
the exact weighted one-dimensional k-means operation. It is derived from the
MIT-licensed `_core.cpp` bundled with pinned `coremltools==9.0`; the complete
attribution and license grant are retained in the source.

The ABI accepts sorted unique f64 values and f64 occurrence weights directly.
The FP32 converter supplies sorted unique FP16-rounded values to derive the
same stable assignments as the FP16 package, then separately refits FP32
centroids from the original checkpoint values. Omitting the reference
helper's sort, sorted-value/weight copies, undo map, and de-sort is exact
because the supplied unique values are already strictly increasing. SMAWK
comparisons, centroid arithmetic, and tie-breaking order are preserved.

The dynamic-programming cost matrix uses two rolling f64 rows. Backpointers
remain available for exact reconstruction but use `uint32_t`, which covers the
converter's input cardinalities. For `n` unique values and `k` clusters, the
two dominant allocations are `16n` bytes for D and `4kn` bytes for T, rather
than `16kn` bytes for the reference helper's full f64 D plus unsigned-long T.

## Build contract

`convert.py` builds the helper automatically for FP32 conversion. On arm64
macOS the command has this shape (deliberately without fast-math):

```bash
clang++ \
  -O3 \
  -DNDEBUG \
  -std=c++17 \
  -fvisibility=hidden \
  -dynamiclib \
  native/kmeans1d_sorted_unique.cpp \
  -o /tmp/libparakeet_kmeans1d.dylib
```

Linux uses the same semantic flags plus `-fPIC -shared`. The exported symbol
is `parakeet_kmeans1d_sorted_unique_weighted_f64`; its stable declaration and
status codes are in `kmeans1d_sorted_unique.h`.

The converter hashes this source and header together with the platform, a
versioned build recipe, compile flags, resolved `clang++` path, and complete
compiler version output. It atomically caches the resulting build-specific
library under `model/cache/native/`. Spawned FP32 layer tasks receive that
exact path instead of discovering or rebuilding a library themselves. Local
libraries are generated cache products and must not be committed.

## Exactness evidence

The retained implementation was checked against
`coremltools._deps._kmeans1d.cluster` on 511 deterministic, randomized, and
representative FP32-derived cases, matching every assignment and returned f64
centroid bit. It also matched real multi-million-value encoder fits and their
packed output byte-for-byte.

The current v2 stable-assignment/refit FP32 package has manifest SHA-256
`28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b`.
This identity is enforced by the converter and covered by the tests in
`../tests/`.

## Worker sizing

FP32 layer packing accepts `--jobs 0` for automatic selection and explicit
values from 1 through 8. Automatic selection reserves 4 GiB for the OS and
parent, budgets 1.5 GiB per native k-means worker, and caps at eight workers,
the CPU count, or the 24 layer tasks:

```text
workers = clamp(floor((physical_bytes - 4 GiB) / 1.5 GiB), 1, 8)
```

If physical memory cannot be determined, automatic selection uses one worker.
Use `--jobs 1` when other memory-heavy work is active. Subsampling, decoder
packing, manifest assembly, and hashing remain outside the layer-worker pool.
