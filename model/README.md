# Reproducing the model packages

This directory contains the complete conversion tool for the two
`parakeet.wgsl` browser model packages. It downloads the pinned official
[`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
NeMo archive, validates its exact identity and structure, and writes the
deterministic FP16 and FP32 packages used by `parakeet.wgsl`.

Generated weights are deliberately not committed to this repository. The
canonical hosted packages can be verified against the identities below, and
the converter refuses to install output whose manifest does not reproduce the
corresponding canonical SHA-256.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) and CPython 3.13
- a platform supported by the locked Python dependencies
- `clang++` for FP32 conversion (the converter builds its exact weighted
  one-dimensional k-means helper locally)
- enough disk and memory for a 2.47 GB source archive and the selected output

The environment is locked for Python 3.13. Its direct dependencies are exact
pins: `coremltools==9.0`, `huggingface-hub==1.24.0`, and `numpy==2.4.6`.
Always use `uv run --frozen`; changing a dependency or the lockfile can change
converter behavior and therefore requires regenerating and re-verifying both
canonical packages.

Plan for at least 8 GB of free disk for the download cache, both generated
packages, and transactional staging. The final FP16 and FP32 directories are
about 406 MB and 441 MB respectively, but a target is staged in full before it
replaces any existing package. The checkpoint payload is read through
read-only memory maps rather than materialized as one Python object.

FP16 conversion is serial. FP32 conversion defaults to bounded process
parallelism: it reserves 4 GiB for the OS and parent, budgets 1.5 GiB per
worker, and uses at most eight workers, the CPU count, or the 24 layer tasks.
A 16 GiB, eight-CPU machine therefore selects eight workers. Use `--jobs 1`
when memory pressure matters or when other memory-heavy work is active.

## Generate the packages

Run these commands from this directory:

```bash
# The established default: FP16 only, written to files/.
uv run --frozen --project . --python 3.13 python3 convert.py

# FP32 only, written to files-fp32/.
uv run --frozen --project . --python 3.13 \
  python3 convert.py --precision fp32

# Reproduce both canonical directories from the same validated checkpoint.
uv run --frozen --project . --python 3.13 \
  python3 convert.py --precision both
```

`--cache-dir` relocates downloader and native-helper cache state.
`--output-dir` relocates a single-precision package, but cannot be combined
with `--precision both`. `--jobs 0` is automatic FP32 sizing; explicit values
from 1 through 8 are accepted and bounded by the available CPU count.

Every package is generated in a hidden sibling staging directory. The
converter writes and hashes the complete package, requires the canonical
manifest identity, validates the source archive again, and only then replaces
the target directory as one unit. An existing nonempty target must already
contain `manifest.json`, and symbolic-link targets are rejected.

## Pinned source identity

The source repository revision is permanently fixed to:

```text
nvidia/parakeet-tdt-0.6b-v2
ae9ad07059c7c739ffaf932226a8fe64ae2620b0
```

The converter accepts no revision override and validates all of the following:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `parakeet-tdt-0.6b-v2.nemo` | 2,472,222,720 | `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c` |
| embedded `model_weights.ckpt` | 2,471,920,514 | `fb1cbe2765fb80c02c0f874b6ea2c2bd24048c3618c092584fd086c6366e48a9` |

It also validates the complete tar and PyTorch ZIP inventories, every embedded
asset, the restricted pickle/storage contract, and all 725 source tensor names
and shapes. It neither imports PyTorch or NeMo nor extracts the checkpoint.

## Canonical output identities

Each package has 797 tensors in 27 binary shards. “Payload files” below means
the 32 files listed and individually hashed by `manifest.json`; the manifest
itself is the 33rd file in the directory.

| Precision | Format | Manifest SHA-256 | Manifest bytes | Payload files | Payload bytes | Binary bytes | Directory bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | `parakeet-webgpu-v1` | `11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65` | 176,861 | 32 | 405,363,502 | 405,063,936 | 405,540,363 |
| FP32 | `parakeet-webgpu-fp32-v1` | `28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b` | 177,045 | 32 | 440,817,198 | 440,517,632 | 440,994,243 |

The FP32 package contains 601 FP32 tensors and 196 uint32 tensors, with no
FP16 tensors. The FP16 and FP32 identities are independent; never compare one
precision's manifest to the other precision's hash.

## Hosted packages and self-hosting

The canonical packages are available at these immutable manifest URLs:

```text
FP16: https://parakeet-wgsl-models.narcotic.sh/v1/fp16/11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65/manifest.json
FP32: https://parakeet-wgsl-models.narcotic.sh/v1/fp32/28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b/manifest.json
```

Each manifest lists 32 payload files, and the runtime resolves every payload
filename relative to that manifest's URL. To self-host or mirror a package,
copy its `manifest.json` and all 32 listed files into one directory without
changing their names or bytes, then pass the new manifest URL through the
library's `modelUrls` option. Serve responses whose decoded bodies are the
exact stored bytes; browser-negotiated HTTP content coding is valid. For a
different origin, allow cross-origin `GET` requests; use HTTPS in production.
The current runtime accepts only these canonical package identities. Never
replace bytes in place; a future package revision requires a new URL and a
corresponding library release.

On macOS, verify freshly generated or downloaded manifests with:

```bash
shasum -a 256 files/manifest.json files-fp32/manifest.json
```

Matching the canonical manifest SHA-256 authenticates the manifest containing
the expected byte length and SHA-256 of every payload file. Re-hash every
payload and check the complete package contract with:

```bash
PARAKEET_VERIFY_PACKAGE_HASHES=1 \
  uv run --frozen --project . --python 3.13 \
  python3 -m unittest discover -s tests -v
```

For the slower source-to-package packing audit as well, retain the pinned NeMo
archive in the converter cache and run:

```bash
PARAKEET_VERIFY_PACKAGE_HASHES=1 PARAKEET_VERIFY_SOURCE_PACKING=1 \
  uv run --frozen --project . --python 3.13 \
  python3 -m unittest discover -s tests -v
```

The ordinary focused test suite does not download or convert the model:

```bash
uv run --frozen --project . --python 3.13 \
  python3 -m unittest discover -s tests -v
```

## Weight preparation and package contents

The converter preserves the complete model architecture; no layers are
removed. It reads the pinned official FP32 checkpoint through restricted
memory-mapped tensor views and writes operation-native shards for the browser
runtime. Along the way, it transposes and pads matrices, packs convolution
kernels, permutes subsampling columns, concatenates Q/K/V projections, folds
BatchNorm parameters, combines LSTM weights and biases, and precomputes the
fixed relative-position projections.

The size reduction comes primarily from weight palettization, a form of
weight-only quantization. Of the checkpoint's 617,908,374 inference tensor
values, 583,794,688 (94.5%) belong to 196 large matrices. Exact weighted
one-dimensional k-means clusters their values into small lookup tables, and
each weight is stored as a bit-packed table index. Of those matrices, 172 use
5-bit indices into 32-value palettes, while the 24 self-attention output
projections use 6-bit indices into 64-value palettes.

FP16 clusters FP16-rounded source values and stores FP16 palettes. FP32 derives
the same assignments from FP16-rounded values, refits each centroid against
the original FP32 values, and stores FP32 palettes. This avoids independently
changing assignments in ways known to disrupt TDT decoding. The packed indices
occupy 368,017,408 bytes in either package. Together with the remaining
runtime tensors and assets, this reduces the 2.47 GB source archive to
approximately 405 MB for FP16 or 441 MB for FP32.

WebGPU expansion shaders reconstruct matrices into bounded, reusable scratch
as they are needed rather than retaining a fully expanded copy of the model.

The use of k-means palettization for Parakeet's encoder was informed by
FluidInference's [Möbius](https://github.com/FluidInference/mobius)
model-conversion work. The mixed 5/6-bit topology, packed binary format, and
FP32 assignment scheme are specific to `parakeet.wgsl`.

Each output directory contains:

- `preprocessor.bin`
- `encoder-subsampling.bin`
- `encoder-layer-00.bin` through `encoder-layer-23.bin`
- `decoder.bin`
- the exact embedded `model_config.yaml`, `tokenizer.model`,
  `tokenizer.vocab`, and `vocab.txt`
- generated `tokenizer.json`
- `manifest.json`, with the size and SHA-256 of every payload file

The generated tokenizer preserves the official 1,024 SentencePiece pieces at
IDs 0–1023, marks `<unk>` at ID 0, and adds only `<blank>` at ID 1024. It does
not synthesize a pad token or enable byte fallback.

The exact FP32 helper in `native/` is derived from the MIT-licensed kmeans1d
implementation bundled with `coremltools==9.0`. Its source retains the full
copyright and permission notice. Built libraries are cache products and are
not committed.

## Generated files and licenses

The repository's `model/.gitignore` excludes `cache/`, `files/`,
`files-fp32/`, virtual environments, bytecode, temporary package directories,
and local native libraries. Do not force-add any of those outputs. The npm
package also excludes them; browser weights are distributed separately from
the software bundle.

The `parakeet.wgsl` source code and converter are licensed under the root MIT
license. NVIDIA's Parakeet model and its derived weight packages are licensed
under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: **Parakeet-TDT 0.6B v2 by NVIDIA**, source repository linked at
the top of this document.

The FP16 and FP32 browser packages are modified derivatives of the official
model: tensors are selected, reordered, folded, padded, precision-converted,
and palette-compressed, while the official tokenizer and configuration assets
are retained.
See the root `THIRD_PARTY_LICENSES` for consolidated attribution and the
complete CC BY 4.0 terms; the root MIT `LICENSE` does not replace the model's
license.
