# Native and WebAssembly audio components

This file records the production architecture and build details for the
package's native audio components.

The package ships two independent native audio components:

| Component | Purpose |
| --- | --- |
| `wasm/parakeet-fbank.wasm` | Streaming SIMD filter-bank extraction and RMS accounting |
| `wasm/parakeet-wgsl-ffmpeg-core.{js,wasm}` | Lazy pull-driven audio decoding, downmixing, resampling, and PCM16 output |

Package consumers never compile either component. The npm tarball contains the
reviewed, prebuilt artifacts and has no install-time native build step.

## SIMD frontend

`fbank/parakeet_fbank.cpp` is the native frontend implementation. Emscripten
compiles it into the fixed-memory SIMD artifact at
`wasm/parakeet-fbank.wasm`.

One artifact supports the repository's fixed frontend timeline:

- 16 kHz mono float PCM decoded from the canonical PCM16 WAV
- 240,000 source samples and hop 160
- 128 feature bins and 1,501 physical / 1,500 valid frames
- logical batches of at most 40 windows
- reusable input/output arenas and overlap-aware frame reuse
- exact ordered-Float32 RMS bits for up to 187 complete 80 ms frames

The physical input arena is sized to the fixed 240,000-sample capacity.
Primary inference reads expose at most 238,080 logical samples plus any real
context/backfill, while standalone repair reads expose at most 239,360 real
samples; the wrapper right-pads the remaining physical tail. Frame preparation
and raw-frame reuse are compile-time-specialized for hop 160. The RMS result
uses a 748-byte scratch inside the existing 43-page heap. TypeScript consumes
it synchronously into one persistent high-bit histogram; the complete audio is
never retained as RMS values.

Build the frontend with:

```bash
pnpm build:wasm
```

`build-wasm.sh` prints the selected compiler version to stdout but does not
persist it beside the artifact or pin a particular Emscripten release. The
checked-in artifact remains the reviewed production input; after rebuilding
it, run the frontend tests and production build before committing it. The
build produces a standalone Wasm module with SIMD enabled, no filesystem, no
allocator, no memory growth, and no JavaScript glue. The TypeScript wrapper
validates the compiled constants during initialization.

## FFmpeg audio-decoding core

The JavaScript/Wasm core is built separately from the
[`narcotic-sh/ffmpeg.wasm`](https://github.com/narcotic-sh/ffmpeg.wasm) fork at
commit `d29b31252b46d05887fcf9603849ff0784bce91e`. It pins FFmpeg `n5.1.4` at
commit `4729204c17f756e186d622060088371d10b34f7e`, disables GPL, nonfree, and
version-3-only components, and retains only the demuxers, decoders, filters,
resampler, and the PCM WAV encoder used by the one-shot equivalence oracle.

The core is single-threaded and uses no `SharedArrayBuffer`, so it does not add
a COOP/COEP or cross-origin-isolation requirement. It is loaded only for a
noncanonical source. WORKERFS reads the original `Blob` without copying it
wholesale into MEMFS. One persistent FFmpeg decoder/filter/resampler session
then exposes bounded pulls of exact 16 kHz mono signed PCM16 immediately before
the WAV muxer. The package worker writes each pull to a closed immutable OPFS
`.s16le` segment and posts its disk-backed `File` snapshot to the inference
worker. Concatenating every pull is byte-identical to the data payload from the
pinned one-shot reference command; pull boundaries never reset the
resampler or alter timestamps, priming, edit lists, or final drain behavior.
After open, the core also exposes the selected audio stream's advisory
duration rescaled to 16 kHz. The runtime uses it only to seed progress and
reconciles it against the exact decoded sample count at EOF.

The first pull contains at most 8,275,201 samples, the strict-lookahead
boundary for forty stable primary windows. Later pulls contain at most
8,243,200 samples (forty primary strides), bounding live decoder output to
about 16.6 MB while keeping complete B40 inference graphs supplied. The
decoder worker runs ahead to EOF, and its immutable disk segments remain until
global energy selection and adaptive repair finish.

Rebuilding requires Emscripten **6.0.3 exactly**. The target rejects another
compiler version before modifying the build tree. From the pinned fork, run:

```bash
make parakeet-wgsl-audio
```

Copy
`packages/core-parakeet-wgsl/dist/esm/parakeet-wgsl-ffmpeg-core.{js,wasm}`
into this directory's `wasm/` folder. The target records the complete
configuration and link command. The immutable, content-addressed
[corresponding-source archive](https://github.com/narcotic-sh/ffmpeg.wasm/releases/download/parakeet-wgsl-ffmpeg-core-v1/parakeet-wgsl-ffmpeg-core-8703a1a66936cf39a10ef94490e9e594dc602e4d094fa7b0325d59fee0488fa5-source.tar.xz)
provides the exact source and relinking materials required by the LGPL. It is
11,349,996 bytes with SHA-256
`d0d4fad96cf8f849160f81b5a89f081eb137d2695b21cd54200654fe7a69e706`.

Current reviewed artifacts:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `parakeet-wgsl-ffmpeg-core.js` | 82,159 | `ca0d9083e37ae0b016f6763db2e541b718077d70c5c99f8b09817741c25578ee` |
| `parakeet-wgsl-ffmpeg-core.wasm` | 1,599,767 | `8703a1a66936cf39a10ef94490e9e594dc602e4d094fa7b0325d59fee0488fa5` |

The ffmpeg.wasm MIT and FFmpeg LGPL-2.1 license texts, corresponding-source
location, and relinking information are consolidated in the repository's root
`THIRD_PARTY_LICENSES`. Changing the toolchain, core bindings, enabled
FFmpeg components, memory limit, or exported functions can change emitted
bytes and runtime behavior; rebuild both paired artifacts and run
incremental-vs-one-shot byte equivalence plus browser package checks before
accepting them.
