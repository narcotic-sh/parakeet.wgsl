/// <reference lib="webworker" />

import {
  STREAMING_DECODER_INITIAL_SAMPLES,
  STREAMING_DECODER_STEADY_SAMPLES,
  type StartStreamingDecodeRequest,
} from "./streaming-decoder-protocol";
import {
  AUDIO_DECODING_SCRATCH_ROOT,
  cleanupStaleAudioDecodingScratchData,
} from "./audio-scratch";
import createFFmpegCoreModule from "../cpp/wasm/parakeet-wgsl-ffmpeg-core.js";

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

interface FFmpegFileSystem {
  readonly filesystems: { readonly WORKERFS: unknown };
  mkdir(path: string): void;
  mount(
    type: unknown,
    options: {
      readonly blobs: readonly {
        readonly name: string;
        readonly data: Blob;
      }[];
    },
    mountpoint: string,
  ): void;
}

interface IncrementalDecodeResult {
  readonly samples: Int16Array<ArrayBuffer>;
  readonly sampleCount: number;
  readonly done: boolean;
  readonly decodeMs: number;
  readonly totalDecodeMs: number;
}

interface IncrementalAudioDecoder {
  readonly estimatedSampleCount: number | null;
  decodeNext(maxSamples: number): IncrementalDecodeResult;
  close(): void;
  cancel(): void;
}

interface IncrementalFFmpegCore {
  readonly FS: FFmpegFileSystem;
  createIncrementalAudioDecoder(inputPath: string): IncrementalAudioDecoder;
}

type IncrementalFFmpegCoreFactory = (options: {
  readonly wasmURL: string;
}) => Promise<IncrementalFFmpegCore>;

let started = false;

workerScope.addEventListener(
  "message",
  (event: MessageEvent<StartStreamingDecodeRequest>) => {
    if (event.data.type !== "start-streaming-decode" || started) return;
    started = true;
    void start(event.data);
  },
);

async function start(request: StartStreamingDecodeRequest): Promise<void> {
  const decoderStartedAtMs = monotonicEpochNow();
  const wallStartedAt = performance.now();
  let decoder: IncrementalAudioDecoder | undefined;
  let ready = false;
  try {
    if (typeof createFFmpegCoreModule !== "function") {
      throw new Error("The packaged FFmpeg core has no module factory.");
    }
    const createCore =
      createFFmpegCoreModule as IncrementalFFmpegCoreFactory;
    const core = await createCore({ wasmURL: request.ffmpegCoreWasm });
    core.FS.mkdir("/input");
    const mountedInputName = inputName(request.fileName);
    core.FS.mount(
      core.FS.filesystems.WORKERFS,
      { blobs: [{ name: mountedInputName, data: request.audio }] },
      "/input",
    );
    decoder = core.createIncrementalAudioDecoder(`/input/${mountedInputName}`);

    workerScope.postMessage({
      type: "streaming-decoder-ready",
      requestId: request.requestId,
      estimatedSampleCount: decoder.estimatedSampleCount,
    });
    ready = true;
    await decodeToImmutableSegments(
      request,
      decoder,
      decoderStartedAtMs,
      wallStartedAt,
    );
  } catch (error) {
    try {
      decoder?.cancel();
    } catch {
      // Preserve the decode/open failure.
    }
    if (ready) {
      request.port.postMessage({
        type: "pcm-failed",
        message: errorMessage(error),
      });
    } else {
      workerScope.postMessage({
        type: "streaming-decoder-open-failed",
        requestId: request.requestId,
        message: errorMessage(error),
      });
    }
    await removeJob(request.scratchDirectory);
  } finally {
    try {
      decoder?.close();
    } catch {
      // The primary result is already reported.
    }
    request.port.close();
    workerScope.close();
  }
}

async function decodeToImmutableSegments(
  request: StartStreamingDecodeRequest,
  decoder: IncrementalAudioDecoder,
  decoderStartedAtMs: number,
  wallStartedAt: number,
): Promise<void> {
  const storageRoot = await navigator.storage.getDirectory();
  await cleanupStaleAudioDecodingScratchData();
  const scratchRoot = await storageRoot.getDirectoryHandle(
    AUDIO_DECODING_SCRATCH_ROOT,
    { create: true },
  );
  const jobDirectory = await scratchRoot.getDirectoryHandle(
    request.scratchDirectory,
    { create: true },
  );
  let segmentIndex = 0;
  let totalSamples = 0;
  let decoderActiveMs = 0;

  while (true) {
    const maxSamples =
      segmentIndex === 0
        ? STREAMING_DECODER_INITIAL_SAMPLES
        : STREAMING_DECODER_STEADY_SAMPLES;
    const decoded = decoder.decodeNext(maxSamples);
    validateDecodeResult(decoded, maxSamples, decoderActiveMs);
    decoderActiveMs = decoded.totalDecodeMs;

    if (decoded.sampleCount > 0) {
      const file = await writeImmutableSegment(
        jobDirectory,
        segmentIndex,
        decoded.samples,
      );
      request.port.postMessage({
        type: "pcm-segment",
        segmentIndex,
        startSample: totalSamples,
        sampleCount: decoded.sampleCount,
        file,
      });
      totalSamples += decoded.sampleCount;
      segmentIndex += 1;
    }
    if (decoded.done) break;
  }

  const decoderEndedAtMs = monotonicEpochNow();
  request.port.postMessage({
    type: "pcm-end",
    totalSamples,
    outputByteLength: totalSamples * Int16Array.BYTES_PER_ELEMENT,
    decoderWallMs: Math.max(0, performance.now() - wallStartedAt),
    decoderActiveMs,
    decoderStartedAtMs,
    decoderEndedAtMs,
  });
}

async function writeImmutableSegment(
  directory: FileSystemDirectoryHandle,
  index: number,
  samples: Int16Array<ArrayBuffer>,
): Promise<File> {
  const name = `pcm-${index.toString().padStart(8, "0")}.s16le`;
  const fileHandle = await directory.getFileHandle(name, { create: true });
  const access = await fileHandle.createSyncAccessHandle();
  try {
    access.truncate(0);
    const bytes = new Uint8Array(
      samples.buffer,
      samples.byteOffset,
      samples.byteLength,
    );
    let written = 0;
    while (written < bytes.byteLength) {
      const count = access.write(bytes.subarray(written), { at: written });
      if (!Number.isSafeInteger(count) || count <= 0) {
        throw new Error("OPFS produced a short PCM segment write");
      }
      written += count;
    }
    access.flush();
  } finally {
    access.close();
  }
  const file = await fileHandle.getFile();
  if (file.size !== samples.byteLength) {
    throw new Error("Immutable OPFS PCM segment has the wrong size");
  }
  return file;
}

function validateDecodeResult(
  result: IncrementalDecodeResult,
  maxSamples: number,
  previousTotalDecodeMs: number,
): void {
  if (
    !(result.samples instanceof Int16Array) ||
    result.samples.byteOffset !== 0 ||
    result.samples.byteLength !== result.samples.buffer.byteLength ||
    !Number.isSafeInteger(result.sampleCount) ||
    result.sampleCount < 0 ||
    result.sampleCount > maxSamples ||
    result.samples.length !== result.sampleCount ||
    typeof result.done !== "boolean" ||
    !Number.isFinite(result.decodeMs) ||
    result.decodeMs < 0 ||
    !Number.isFinite(result.totalDecodeMs) ||
    result.totalDecodeMs < previousTotalDecodeMs ||
    result.totalDecodeMs < result.decodeMs ||
    (result.sampleCount === 0 && !result.done)
  ) {
    throw new Error("Incremental FFmpeg decoder returned an invalid step");
  }
}

function inputName(fileName: string): string {
  const extension = /\.([a-z0-9]{1,10})$/i.exec(fileName)?.[0];
  return `source${extension?.toLowerCase() ?? ".audio"}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Audio decoding failed.";
}

function monotonicEpochNow(): number {
  return performance.timeOrigin + performance.now();
}

async function removeJob(name: string): Promise<void> {
  try {
    const storageRoot = await navigator.storage.getDirectory();
    const scratchRoot = await storageRoot.getDirectoryHandle(
      AUDIO_DECODING_SCRATCH_ROOT,
    );
    await scratchRoot.removeEntry(name, { recursive: true });
  } catch {
    // Parent-side settlement makes another best-effort attempt.
  }
}
