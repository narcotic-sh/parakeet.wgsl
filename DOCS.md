# parakeet.wgsl Documentation

### `checkSupport()`

```ts
import { checkSupport } from "parakeet.wgsl";

const support = await checkSupport();
```

Performs a network-free browser and GPU preflight. It requests a
high-performance WebGPU adapter and checks the required limits, but does not
create a GPU device, compile shaders, load WebAssembly, or download a model.

It also probes Origin Private File System access for optional audio decoding.
Failure of that probe does not make the browser unsupported for an already
canonical WAV.

#### Returns

```ts
interface ParakeetSupport {
  supported: boolean;
  errors: readonly string[];
  warnings: readonly string[];
  executionProfile: ParakeetExecutionProfile | null;
  adapter: ParakeetAdapterSummary | null;
  audioDecoding: {
    available: boolean;
    reason?: string;
  };
}

type ParakeetModelPrecision = "fp16" | "fp32";
type ParakeetKernelBackend = "subgroups" | "portable";

interface ParakeetExecutionProfile {
  precision: ParakeetModelPrecision;
  kernelBackend: ParakeetKernelBackend;
  requiredFeatures: readonly ("shader-f16" | "subgroups")[];
}

interface ParakeetAdapterSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter?: boolean;
}
```

`supported` covers the secure-context, WebGPU-adapter, buffer, binding, and
workgroup-storage requirements. `audioDecoding.available` is independent and
only describes the optional path for compressed or noncanonical recordings.
A fallback adapter is reported as a warning.

The execution profile is selected automatically:

- `shader-f16` selects the FP16 model; its absence selects FP32.
- `subgroups` selects the subgroup kernels only when both reported subgroup
  bounds are exactly 32 lanes; otherwise the portable kernels are selected.

The inference worker repeats the authoritative selection during
initialization. A successful preflight cannot guarantee that later device
creation or shader compilation will succeed.

### `createTranscriber()`

```ts
import { createTranscriber } from "parakeet.wgsl";

const transcriber = createTranscriber(options);
```

Creates a reusable transcription controller synchronously. Its dedicated
module worker is created immediately, so a bundler may load that small worker
asset. Construction does not request WebGPU, load FFmpeg, or download a model.

#### Options

```ts
interface ParakeetModelUrls {
  fp16: string | URL;
  fp32: string | URL;
}

interface CreateTranscriberOptions {
  modelUrls?: Partial<ParakeetModelUrls>;
  onLoadProgress?: (progress: ModelLoadProgress) => void;
  wasmUrl?: string | URL;
  workerFactory?: () => Worker;
}
```

- `modelUrls`: Override either or both canonical manifest locations with an
  exact mirror. An omitted precision keeps its default URL.
- `onLoadProgress`: Receive events from initialization that loads the model.
- `wasmUrl`: Override only the streaming filter-bank WebAssembly URL for an
  unusual bundler or Content Security Policy.
- `workerFactory`: Advanced override for constructing the WebGPU inference
  worker. It is invoked once during `createTranscriber()`.

Empty URLs throw a `TypeError` synchronously. There is no public precision or
kernel-backend option; both are capability-derived.

`createTranscriber()` returns this public controller surface:

```ts
interface ParakeetTranscriber {
  readonly initialized: boolean;
  readonly diagnostics: Readonly<Record<string, unknown>> | null;
  loadCachedModel(): Promise<boolean>;
  transcribe(
    audio: Blob,
    options?: TranscribeOptions,
  ): Promise<TranscriptionResult>;
  cancel(): boolean;
  dispose(): void;
}
```

### `transcriber.loadCachedModel()`

```ts
const loaded = await transcriber.loadCachedModel();
```

Loads and compiles the capability-selected model only when every required
object is already valid in Cache Storage.

When this method starts initialization, it prohibits model network access. It
returns `true` after the model is GPU-resident and `false` when Cache Storage is
unavailable or any selected-package object is missing, incomplete, or corrupt.
It may still load package-owned worker and filter-bank WebAssembly assets.

Concurrent calls share one initialization task. A concurrent `transcribe()`
also joins that task and, after a cache miss, starts the ordinary
network-enabled initialization without restarting healthy work. Conversely,
if a transcription has already started network-enabled initialization,
`loadCachedModel()` joins that existing task and it may use the network.

Standalone cache warmup has no `AbortSignal`. Call `dispose()` to terminate it.

### `transcriber.transcribe()`

```ts
const result = await transcriber.transcribe(audio, options);
```

Transcribes one `File` or `Blob`. The same controller can be reused for later
recordings, but it accepts only one active transcription at a time.

#### Parameters

- `audio`: The source `File` or `Blob`.
- `options.sourceName`: Container-identifying file name for an unnamed `Blob`.
  A `File`'s own non-empty name always takes precedence. Without either, the
  source name defaults to `audio`.
- `options.onProgress`: Overall transcription progress.
- `options.onPartialResult`: Full, replaceable transcript snapshots.
- `options.onPacedTranscript`: Presentation-paced text and word splices.
- `options.signal`: An `AbortSignal` for the complete operation.

```ts
interface TranscribeOptions {
  sourceName?: string;
  onProgress?: (progress: TranscriptionProgress) => void;
  onPartialResult?: (snapshot: TranscriptionSnapshot) => void;
  onPacedTranscript?: (update: PacedTranscriptUpdate) => void;
  signal?: AbortSignal;
}
```

For example, an unnamed recording blob can provide its container extension:

```ts
const result = await transcriber.transcribe(recordingBlob, {
  sourceName: "interview.webm",
});
```

The first ordinary call initializes WebGPU, selects one model precision,
streams that package from Cache Storage or the network, compiles the graph, and
then transcribes. Later calls reuse the GPU-resident model.

#### Returns

A `Promise<TranscriptionResult>` containing the final transcript, timestamped
tokens and words, and metrics. See [Transcription result](#transcription-result)
below.

### `transcriber.cancel()`

```ts
const requested = transcriber.cancel();
```

Requests cancellation of the active transcription and returns `true`, or
returns `false` when there is no active transcription. Cancellation covers
audio preparation and decoding, model initialization started by that job,
retry backoff, and inference. The transcription promise rejects with an error
whose `name` is `AbortError`.

The controller remains reusable after cancellation. A transcription that only
joined an independently started `loadCachedModel()` warmup does not cancel that
shared warmup.

### `transcriber.dispose()`

```ts
transcriber.dispose();
```

Permanently terminates the controller's worker, releases its GPU resources,
and rejects pending work. It is idempotent. A disposed controller cannot be
used again.

### Controller state

```ts
transcriber.initialized;
transcriber.diagnostics;
```

- `initialized` is `true` after the selected model is GPU-resident and the
  inference pipelines are ready.
- `diagnostics` is `null` before initialization and typed as a read-only
  diagnostic record afterward. Its contents are intended for inspection
  rather than as a stable application-data schema.

## Audio input and decoding

A canonical input is a valid RIFF/WAVE file containing format-code-1, 16 kHz,
mono, signed little-endian PCM16 with consistent block alignment and byte rate.
It bypasses FFmpeg and OPFS completely and streams directly into the inference
frontend.

All other inputs use the packaged, lazy FFmpeg decoder. The reduced core
supports:

- MP3;
- AAC or ALAC in M4A, MP4, or CAF;
- Opus or Vorbis in WebM, Matroska, or Ogg;
- FLAC in its native container or Ogg;
- supported PCM variants in WAV, AIFF, AIFF-C, or CAF; and
- WavPack (`.wv`).

Raw ADTS AAC (`.aac`), AC-3, AMR, MPEG-TS, and APE are not supported. When a
container has multiple audio streams, the first one is used.

The decoder reads the source `Blob` through WORKERFS, downmixes and resamples
it to exact 16 kHz mono PCM16, and writes bounded immutable segments to OPFS.
The source must open successfully before the large model fetch begins.
Inference then starts as soon as enough PCM is available instead of waiting
for the complete recording. The temporary PCM remains on disk through final
transcript repair and is removed after success, failure, cancellation, or
disposal.

This path needs approximately 115.2 MB of temporary origin storage per hour of
audio (about 109.9 MiB). There is no full-output in-memory fallback. The
single-threaded decoder does not use `SharedArrayBuffer`, so it requires neither
cross-origin isolation nor COOP/COEP headers.

### `requiresAudioDecoding()`

```ts
import { requiresAudioDecoding } from "parakeet.wgsl";

const needsDecoder = await requiresAudioDecoding(audio);
```

Returns `false` only when the `File` or `Blob` passes the strict canonical-WAV
parser described above. It performs bounded header reads without fetching
FFmpeg, accessing OPFS, or using the network.

A `true` result means the FFmpeg path will be attempted; it does not guarantee
that arbitrary input data is a format supported by the reduced core. A
non-`Blob` argument rejects with `TypeError`.

## Model loading progress

Pass `onLoadProgress` to `createTranscriber()`:

```ts
const transcriber = createTranscriber({
  onLoadProgress(progress) {
    console.log(progress.phase, progress.fraction);
  },
});
```

```ts
type ModelLoadPhase =
  | "webgpu"
  | "wasm"
  | "manifest"
  | "weights"
  | "tokenizer"
  | "cache"
  | "retry"
  | "ready"
  | "pipelines";

interface ModelLoadProgress {
  phase: ModelLoadPhase;
  fraction: number;
  loadedBytes?: number;
  totalBytes?: number;
  message?: string;
  attempt?: number;
  maxAttempts?: number;
}
```

`fraction` is local to the current phase, not one overall initialization
percentage. Byte counts are present for package transfers. Cache-recovery and
retry events may instead provide a message and attempt information.

The callback belongs to the controller's initialization rather than to an
individual transcription. Throwing from it cancels the affected initialization
and rejects the operation.

## Transcription progress

Pass `onProgress` to `transcribe()`:

```ts
const result = await transcriber.transcribe(file, {
  onProgress(progress) {
    console.log(progress.phase, Math.round(progress.fraction * 100));
  },
});
```

```ts
type TranscriptionPhase = "transcribing" | "done";

interface TranscriptionProgress {
  phase: TranscriptionPhase;
  fraction: number;
  metrics: {
    audioDecoding: AudioDecodingMetrics;
    audioDurationSeconds: number;
    processedAudioSeconds: number;
    elapsedMs: number;
    audioReadMs: number;
    inferenceMs: number;
    frontendMs: number;
    encoderMs: number;
    decoderMs: number;
    speedFactor: number;
    completedWindows: number;
    totalWindows: number;
    repairGapProbes: number;
    repairWindowDecodes: number;
    repairRounds: number;
    recoveredTokens: number;
  };
}
```

`fraction` is globally monotonic across audio opening, primary inference,
stitching, and adaptive repair. It can plateau when newly discovered work
expands the estimate, but it never moves backward. It remains below `1` while
work is active and reaches exactly `1` only with `phase: "done"`.

For a decoded input, the first event can be `transcribing` at zero while the
audio pipeline opens. There is no separate decoding phase. `totalWindows` is
zero until decoded EOF makes the final primary plan exact;
`completedWindows` always counts only primary windows whose output has been
decoded and merged. A metrics-only refresh can repeat the same fraction.

The terminal progress event is delivered before the transcription promise
settles. Throwing from `onProgress` cancels the operation and rejects its
promise.

## Live transcript callbacks

### `onPartialResult`

```ts
interface TranscriptionSnapshot {
  revision: number;
  text: string;
  tokens: readonly TranscriptionToken[];
  words: readonly TranscriptionWord[];
  audioDurationSeconds: number;
  processedAudioSeconds: number;
}
```

Receives an authoritative snapshot after each completed primary inference
batch and after a repair round inserts tokens. Each snapshot contains the full
provisional transcript.

Snapshots are replaceable, not append-only. A later overlapping window can
revise the newest seam, and repair can insert words at an earlier timestamp.
Use `revision` to identify newer snapshots and replace the previous state.
There is no provisional `isFinal` flag; await `transcribe()` for the final
result.

### `onPacedTranscript`

Provides a presentation-paced stream derived from the same real snapshots.
It does not change inference or raw-snapshot cadence. Pending primary words are
revealed when primary inference finishes; repair corrections and final
reconciliation are immediate. The final update is delivered before the
transcription promise settles.

```ts
interface PacedTranscriptTextSplice {
  startOffset: number;
  deleteCount: number;
  insertText: string;
}

interface PacedTranscriptWord extends TranscriptionWord {
  startOffset: number;
  endOffset: number;
}

interface PacedTranscriptWordSplice {
  startIndex: number;
  deleteCount: number;
  insertWords: readonly PacedTranscriptWord[];
}

interface PacedTranscriptUpdate {
  revision: number;
  sourceRevision: number;
  textSplice: PacedTranscriptTextSplice;
  wordSplice: PacedTranscriptWordSplice;
  audioDurationSeconds: number;
  processedAudioSeconds: number;
  isFinal: boolean;
}
```

`revision` belongs to the paced presentation stream. Several paced updates can
share one nondecreasing `sourceRevision`. Text splice offsets are UTF-16 code
units in the text before the update. Inserted word offsets are half-open UTF-16
ranges in the text after the update.

The package avoids transcript compilation and callback traffic for callback
types that are not supplied. Throwing from either live callback cancels the
operation and rejects its promise.

## Transcription result

```ts
interface TranscriptionResult {
  text: string;
  tokens: readonly TranscriptionToken[];
  words: readonly TranscriptionWord[];
  metrics: TranscriptionMetrics;
}

interface TranscriptionToken {
  tokenId: number;
  startSeconds: number;
  endSeconds: number;
}

interface TranscriptionWord {
  text: string;
  startSeconds: number;
  endSeconds: number;
}
```

Every result includes all four fields; there is no text-only result mode. All
timestamps are seconds on the original source-audio timeline.

### Metrics

Final metrics contain every progress metric plus:

```ts
interface TranscriptionMetrics {
  audioDecoding: AudioDecodingMetrics;
  audioDurationSeconds: number;
  processedAudioSeconds: number;
  elapsedMs: number;
  audioReadMs: number;
  inferenceMs: number;
  frontendMs: number;
  encoderMs: number;
  decoderMs: number;
  speedFactor: number;
  completedWindows: number;
  totalWindows: number;
  repairGapProbes: number;
  repairWindowDecodes: number;
  repairRounds: number;
  recoveredTokens: number;
  stitchingMs: number;
  repairMs: number;
  repairPrefetchWindowDecodes: number;
  repairPrefetchCacheHits: number;
  repairPrefetchUnusedDecodes: number;
  batchSubmissionTimeline: readonly TranscriptionBatchTimelineRecord[];
  batchSubmissionTimelineOmitted: number;
  phaseTimings: TranscriptionPhaseTimings;
  totalMs: number;
}

interface AudioDecodingMetrics {
  applied: boolean;
  wallMs: number;
  activeMs: number;
  inputByteLength: number;
  outputByteLength: number;
  overlapMs: number;
  pcmWaitMs: number;
  pcmStarvationMs: number;
  pcmStarvationCount: number;
}
```

- `speedFactor` is processed audio seconds divided by elapsed wall seconds.
- `completedWindows` and `totalWindows` are exact in the final result.
- `repairMs` is only the sequential post-primary repair tail. Repair work
  co-batched with another required graph is intentionally excluded.
- `audioDecoding.wallMs` covers decoder-worker startup through decoded EOF, or
  bounded canonical-WAV inspection when decoding was not applied.
- `audioDecoding.overlapMs` is decoder time concurrent with model loading or
  transcription.
- `pcmWaitMs` includes every inference-scheduler wait for decoded PCM;
  `pcmStarvationMs` and `pcmStarvationCount` isolate waits left uncovered after
  submitted inference work finishes.
- `totalMs` is transcription wall time after the model is ready. Model loading
  is excluded. Decoder wall time is reported separately and is not added
  wholesale when it overlaps transcription; genuine PCM waits naturally remain
  inside `totalMs`.

The bounded batch timeline uses this shape:

```ts
type TranscriptionBatchClass = "primary" | "mixed" | "repair";

interface TranscriptionBatchTimelineRecord {
  submissionIndex: number;
  batchClass: TranscriptionBatchClass;
  activeRows: number;
  primaryRows: number;
  repairRows: number;
  officialRepairRows: number;
  finalPrimaryPrefetchRows: number;
  crossRoundPrefetchRows: number;
  preparationStartedOffsetMs: number;
  submissionStartedOffsetMs: number;
  completionOffsetMs: number;
  preparationMs: number;
  submissionToCompletionMs: number;
  frontendMs: number;
  gpuIntervalMs: number;
}

interface TranscriptionPhaseTimings {
  trailingSpeechScanMs: number;
  adaptiveThresholdScanMs: number;
  gapEnergyScanMs: number;
  incrementalStitchMs: number;
  prefixStitchMs: number;
  finalStitchMs: number;
  repairPlanningMs: number;
  repairSplicingMs: number;
  partialResultMs: number;
  tokenDecodeMs: number;
}
```

Timeline offsets are relative to transcription start. `gpuIntervalMs` is the
submit-to-readback wall interval, not a GPU timestamp-query result, and can
overlap another logical inference slot. Only a bounded prefix is retained;
`batchSubmissionTimelineOmitted` reports additional submissions.

## Paced transcript DOM renderer

### `PacedTranscriptDomRenderer`

```ts
import {
  PacedTranscriptDomRenderer,
  createTranscriber,
} from "parakeet.wgsl";

const container = document.querySelector<HTMLElement>("#transcript")!;
const renderer = new PacedTranscriptDomRenderer(container);
const transcriber = createTranscriber();

await transcriber.transcribe(file, {
  onPacedTranscript: renderer.applyUpdate,
});
```

The renderer applies a paced update stream to one stable text node without
creating an element for every word. It exclusively owns the supplied
container's children until disposed.

```ts
interface PacedTranscriptDomRendererOptions {
  autoFollow?: boolean;
}

interface PacedTranscriptDomWordHit {
  word: PacedTranscriptWord;
  clientRects: readonly DOMRect[];
}

new PacedTranscriptDomRenderer(container, {
  autoFollow: true,
});
```

`autoFollow` defaults to `true`. It follows the growing transcript until the
user scrolls away, using native scroll anchoring after initial activation.

#### Properties

- `text`: Currently rendered text.
- `wordRanges`: Live array of timestamped UTF-16 ranges. Treat it as immutable.
- `revision`: Latest applied paced revision.
- `isFinal`: Whether the final update has been applied.

#### Methods

- `applyUpdate(update)`: Arrow-bound callback suitable for
  `onPacedTranscript`. Revisions must be contiguous.
- `wordAtTextOffset(offset)`: Return the word covering a UTF-16 offset.
- `domRangeForWord(word)`: Create a DOM `Range` for a current word object.
  Stale or foreign objects return `undefined`, even when their values match.
- `wordHitAtPoint(clientX, clientY)`: Return `{ word, clientRects }` for a word
  at viewport coordinates.
- `jumpToLatest()`: Resume following at the end of the transcript.
- `reset()`: Start a new paced revision stream while preserving the owned DOM
  nodes.
- `dispose()`: Stop observation and make the renderer unusable while leaving
  the last rendered text visible.

### `pacedTranscriptWordAtOffset()`

```ts
const word = pacedTranscriptWordAtOffset(words, textOffset);
```

Searches sorted `PacedTranscriptWord` entries by their half-open UTF-16 ranges.
Returns `undefined` for whitespace, invalid offsets, or an empty list.

## Model cache

### `getModelCacheInfo()`

```ts
const info = await getModelCacheInfo();
```

```ts
interface ModelCacheInfo {
  supported: boolean;
  assetCount: number;
  sizeBytes: number;
}
```

Reports the aggregate objects stored in parakeet.wgsl-managed Cache Storage
namespaces for the current origin. When Cache Storage is unavailable, it
returns `supported: false` with zero counts.

### `deleteCachedModels()`

```ts
await deleteCachedModels();
```

Deletes every parakeet.wgsl-managed model cache for the current origin. A model
already uploaded to a live controller's GPU buffers remains available until
that controller is disposed.

Normal model loads verify every object by URL, byte length, and SHA-256 while
streaming it to the GPU loader and cache. A cache or quota failure is nonfatal
and falls back to a verified network stream. A damaged cached object is evicted
and replaced.

Transient request and response-body failures, plus HTTP 408, 425, 429, and 5xx
responses, retry up to four total network attempts with abortable exponential
backoff, jitter, and bounded `Retry-After` handling. Deterministic 4xx, schema,
overlong-body, and integrity failures do not retry.

## Model hosting

### `DEFAULT_MODEL_URLS`

```ts
import { DEFAULT_MODEL_URLS } from "parakeet.wgsl";

console.log(DEFAULT_MODEL_URLS.fp16);
console.log(DEFAULT_MODEL_URLS.fp32);
```

Contains the immutable default manifest URLs. Only the precision selected from
the WebGPU adapter is requested.

To self-host, mirror the complete canonical package without changing its bytes
and override one or both manifest URLs:

```ts
const transcriber = createTranscriber({
  modelUrls: {
    fp16: "https://cdn.example/parakeet/fp16/manifest.json",
    fp32: "https://cdn.example/parakeet/fp32/manifest.json",
  },
});
```

Child objects are resolved relative to each manifest. The runtime still
requires the canonical manifest hash, package format, source identity, file
lengths, and object hashes; an override is a mirror, not a custom-model API.
Cross-origin hosting must allow CORS-enabled browser `GET` requests over HTTPS.

See [model/README.md](https://github.com/narcotic-sh/parakeet.wgsl/blob/main/model/README.md)
for exact URLs, hashes, package contents, conversion, and mirroring
instructions.

## Errors

Invalid API arguments reject or throw `TypeError`. Cancellation rejects with an
error whose `name` is `AbortError`. Runtime and worker failures commonly use:

```ts
import { ParakeetError } from "parakeet.wgsl";

try {
  await transcriber.transcribe(file);
} catch (error) {
  if (error instanceof ParakeetError) {
    console.error(error.code, error.message);
  }
}
```

`ParakeetError.code` is a string rather than an exhaustive public union.
Common controller codes include `BUSY`, `TRANSCRIBER_DISPOSED`, and
`WORKER_ERROR`; capability initialization can report codes such as
`INSECURE_CONTEXT`, `WEBGPU_UNAVAILABLE`, `ADAPTER_UNAVAILABLE`,
`FEATURE_UNAVAILABLE`, and `LIMIT_UNAVAILABLE`. Use `checkSupport()` before
initialization instead of relying on capability exceptions for normal control
flow.

A second concurrent `transcribe()` call rejects with `BUSY`. Callback
exceptions are considered operation failures: throwing from a load, progress,
partial-result, or paced-transcript callback cancels the affected work and
rejects its promise, generally with the thrown value.

## Bundling and deployment

parakeet.wgsl is an ESM-only, browser-only package with no production
dependencies. It requires a secure context, WebGPU, module workers, and
WebAssembly SIMD. Cache Storage is optional; OPFS is required only for the
FFmpeg decoding path.

The published entry references four emitted assets through relocatable
`new URL(..., import.meta.url)` expressions:

- the WebGPU inference worker;
- the streaming audio-decoder worker, including the FFmpeg JavaScript glue;
- the filter-bank WebAssembly module; and
- the FFmpeg WebAssembly module.

Vite 8 is verified to discover and relocate this graph automatically. Other
asset-aware ESM bundlers must preserve the same worker and WebAssembly assets.
A custom deployment must keep every emitted `dist/assets` file addressable
relative to the emitted package entry rather than deploying `dist/index.js`
alone.

A restrictive Content Security Policy must permit module workers, WebAssembly,
and connections to the configured model origin. The packaged FFmpeg core is
single-threaded and does not require `SharedArrayBuffer`, cross-origin
isolation, or COOP/COEP headers.
