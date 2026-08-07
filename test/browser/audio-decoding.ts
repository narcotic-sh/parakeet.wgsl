import {
  createTranscriber,
  requiresAudioDecoding,
  type AudioDecodingMetrics,
  type TranscriptionMetrics,
} from "parakeet.wgsl";

const EXPECTED_OUTPUT_SHA256 =
  "b897d88ed738f97cbedafa082321a90948687417d347ecd53d6c64bacabffbdb";
const EXPECTED_SAMPLE_COUNT = 1_600;
const FAKE_TRANSCRIPTION_WALL_MS = 17;
const SCRATCH_ROOT = "parakeet-wgsl-audio-decoding";

interface IterableFileSystemDirectoryHandle extends FileSystemDirectoryHandle {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
}

interface WorkerInspection {
  readonly outputByteLength: number;
  readonly outputSha256: string;
  readonly sampleCount: number;
  readonly segmentCount: number;
  readonly audioDecoding: AudioDecodingMetrics;
}

interface AcceptanceReport {
  readonly ok: boolean;
  readonly inputByteLength?: number;
  readonly outputByteLength?: number;
  readonly outputSha256?: string;
  readonly sampleCount?: number;
  readonly segmentCount?: number;
  readonly workerMessageSequence?: readonly string[];
  readonly audioDecoding?: AudioDecodingMetrics;
  readonly cleanupRemainingEntries?: readonly string[];
  readonly fakeWorkerTerminated?: boolean;
  readonly error?: string;
}

const resultElement = document.querySelector<HTMLPreElement>("#result");
if (resultElement === null) throw new Error("Missing acceptance result element");

void runAcceptance().then(
  (report) => {
    resultElement.textContent = JSON.stringify(report);
  },
  (error: unknown) => {
    const report: AcceptanceReport = {
      ok: false,
      error: error instanceof Error ? error.stack ?? error.message : String(error),
    };
    resultElement.textContent = JSON.stringify(report);
  },
);

async function runAcceptance(): Promise<AcceptanceReport> {
  assert(typeof navigator.storage?.getDirectory === "function", "OPFS unavailable");
  const input = stereoPcm16Wav8Khz(800);
  assert(
    await requiresAudioDecoding(input),
    "Public input inspection accepted the noncanonical fixture",
  );

  const fakeWorker = new FakeInferenceWorker(input.size);
  const transcriber = createTranscriber({
    modelUrls: {
      fp16: "https://not-requested.invalid/fp16.json",
      fp32: "https://not-requested.invalid/fp32.json",
    },
    wasmUrl: "https://not-requested.invalid/fbank.wasm",
    workerFactory: () => fakeWorker as unknown as Worker,
  });

  try {
    const result = await transcriber.transcribe(input);
    assert(result.text === "", "Fake inference result changed in transit");
    assert(result.metrics.audioDecoding.applied, "Result lost conversion metrics");
    assert(
      result.metrics.totalMs === FAKE_TRANSCRIPTION_WALL_MS,
      "Controller mixed decoder wall time into transcription wall time",
    );
    const inspection = fakeWorker.inspection;
    assert(inspection !== undefined, "Fake inference worker saw no transcription");
    assert(
      fakeWorker.messageSequence.join(",") ===
        "prepare-stream,initialize,transcribe-stream",
      `Unexpected controller sequence: ${fakeWorker.messageSequence.join(",")}`,
    );

    const cleanupRemainingEntries = await scratchEntryNames();
    assert(
      cleanupRemainingEntries.length === 0,
      `Scratch cleanup left: ${cleanupRemainingEntries.join(", ")}`,
    );

    transcriber.dispose();
    assert(fakeWorker.terminated, "dispose() did not terminate the inference worker");
    return {
      ok: true,
      inputByteLength: input.size,
      outputByteLength: inspection.outputByteLength,
      outputSha256: inspection.outputSha256,
      sampleCount: inspection.sampleCount,
      segmentCount: inspection.segmentCount,
      workerMessageSequence: fakeWorker.messageSequence,
      audioDecoding: inspection.audioDecoding,
      cleanupRemainingEntries,
      fakeWorkerTerminated: fakeWorker.terminated,
    };
  } finally {
    transcriber.dispose();
  }
}

type MessageListener = (event: MessageEvent<unknown>) => void;
type ErrorListener = (event: ErrorEvent) => void;

class FakeInferenceWorker {
  readonly #messageListeners = new Set<MessageListener>();
  readonly #errorListeners = new Set<ErrorListener>();
  readonly #expectedInputByteLength: number;
  readonly messageSequence: string[] = [];
  #streamJobId: number | undefined;
  #streamInspection: Promise<WorkerInspection> | undefined;
  inspection: WorkerInspection | undefined;
  terminated = false;

  constructor(expectedInputByteLength: number) {
    this.#expectedInputByteLength = expectedInputByteLength;
  }

  postMessage(message: unknown): void {
    if (!isRecord(message) || typeof message.type !== "string") {
      throw new Error("Public controller posted an invalid worker message");
    }
    this.messageSequence.push(message.type);
    if (message.type === "prepare-stream") {
      const jobId = nonNegativeInteger(message.jobId, "jobId");
      assert(this.#streamInspection === undefined, "Stream was prepared twice");
      assert(message.port instanceof MessagePort, "Stream message has no port");
      assert(
        message.inputByteLength === this.#expectedInputByteLength,
        "Stream message reported the wrong input byte count",
      );
      assert(
        message.estimatedSampleCount === EXPECTED_SAMPLE_COUNT,
        "Stream message lost the advisory decoded-sample estimate",
      );
      this.#streamJobId = jobId;
      this.#streamInspection = inspectStreamingPcm(
        message.port,
        this.#expectedInputByteLength,
      );
      return;
    }
    if (message.type === "initialize") {
      const requestId = nonNegativeInteger(message.requestId, "requestId");
      queueMicrotask(() => {
        this.emitMessage({
          type: "ready",
          requestId,
          diagnostics: { acceptanceFakeWorker: true },
        });
      });
      return;
    }
    if (message.type === "transcribe-stream") {
      const jobId = nonNegativeInteger(message.jobId, "jobId");
      assert(jobId === this.#streamJobId, "Streaming job ID changed");
      assert(
        this.#streamInspection !== undefined,
        "Streaming transcription arrived before preparation",
      );
      void this.#streamInspection.then(
        (inspection) => {
          this.inspection = inspection;
          this.emitMessage({
            type: "result",
            jobId,
            result: {
              text: "",
              tokens: [],
              words: [],
              metrics: emptyMetrics(inspection.audioDecoding),
            },
          });
        },
        (error: unknown) => {
          this.emitMessage({
            type: "error",
            jobId,
            error: {
              name: error instanceof Error ? error.name : "Error",
              message: error instanceof Error ? error.message : String(error),
              code: "ACCEPTANCE_INSPECTION_FAILED",
            },
          });
        },
      );
      return;
    }
    if (message.type === "discard-stream") return;
    if (message.type === "cancel") {
      const jobId = nonNegativeInteger(message.jobId, "jobId");
      queueMicrotask(() => this.emitMessage({ type: "cancelled", jobId }));
      return;
    }
    if (message.type !== "cancel" && message.type !== "cancel-initialization") {
      throw new Error(`Unexpected worker message ${message.type}`);
    }
  }

  addEventListener(
    type: "message" | "error",
    listener: MessageListener | ErrorListener,
  ): void {
    if (type === "message") this.#messageListeners.add(listener as MessageListener);
    else this.#errorListeners.add(listener as ErrorListener);
  }

  removeEventListener(
    type: "message" | "error",
    listener: MessageListener | ErrorListener,
  ): void {
    if (type === "message") this.#messageListeners.delete(listener as MessageListener);
    else this.#errorListeners.delete(listener as ErrorListener);
  }

  terminate(): void {
    this.terminated = true;
  }

  private emitMessage(data: unknown): void {
    for (const listener of this.#messageListeners) {
      listener({ data } as MessageEvent<unknown>);
    }
  }
}

function inspectStreamingPcm(
  port: MessagePort,
  inputByteLength: number,
): Promise<WorkerInspection> {
  return new Promise<WorkerInspection>((resolve, reject) => {
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let consumeTail: Promise<void> = Promise.resolve();
    let segmentCount = 0;
    let totalSamples = 0;
    let terminal = false;

    const fail = (error: unknown): void => {
      if (terminal) return;
      terminal = true;
      port.close();
      reject(error);
    };
    const handleMessage = (event: MessageEvent<unknown>): void => {
      if (terminal) return;
      const message = event.data;
      if (!isRecord(message) || typeof message.type !== "string") {
        fail(new Error("Decoder sent an invalid streaming PCM message"));
        return;
      }
      if (message.type === "pcm-failed") {
        fail(new Error(String(message.message ?? "Audio decoding failed")));
        return;
      }
      if (message.type === "pcm-segment") {
        const segmentIndex = nonNegativeInteger(
          message.segmentIndex,
          "segmentIndex",
        );
        const startSample = nonNegativeInteger(message.startSample, "startSample");
        const sampleCount = nonNegativeInteger(message.sampleCount, "sampleCount");
        assert(sampleCount > 0, "Decoder emitted an empty PCM segment");
        assert(message.file instanceof File, "PCM segment has no OPFS File");
        assert(segmentIndex === segmentCount, "PCM segment index is not contiguous");
        assert(startSample === totalSamples, "PCM sample timeline is not contiguous");
        assert(
          message.file.size === sampleCount * Int16Array.BYTES_PER_ELEMENT,
          "PCM segment File has the wrong size",
        );
        const file = message.file;
        consumeTail = consumeTail.then(async () => {
          const bytes = new Uint8Array(await file.arrayBuffer());
          assert(bytes.byteLength === file.size, "PCM segment produced a short read");
          chunks.push(bytes);
        });
        segmentCount += 1;
        totalSamples += sampleCount;
        return;
      }
      if (message.type !== "pcm-end") {
        fail(new Error(`Unexpected decoder message ${message.type}`));
        return;
      }

      const endedSamples = nonNegativeInteger(message.totalSamples, "totalSamples");
      const outputByteLength = nonNegativeInteger(
        message.outputByteLength,
        "outputByteLength",
      );
      assert(endedSamples === totalSamples, "PCM EOF sample count changed");
      assert(
        outputByteLength === totalSamples * Int16Array.BYTES_PER_ELEMENT,
        "PCM EOF byte count changed",
      );
      assertNonNegativeFinite(message.decoderWallMs, "decoderWallMs");
      assertNonNegativeFinite(message.decoderActiveMs, "decoderActiveMs");
      assertNonNegativeFinite(message.decoderStartedAtMs, "decoderStartedAtMs");
      assertNonNegativeFinite(message.decoderEndedAtMs, "decoderEndedAtMs");
      assert(
        message.decoderEndedAtMs >= message.decoderStartedAtMs,
        "Decoder monotonic timestamps moved backward",
      );
      terminal = true;
      port.close();
      void consumeTail.then(async () => {
        const output = new Uint8Array(outputByteLength);
        let offset = 0;
        for (const chunk of chunks) {
          output.set(chunk, offset);
          offset += chunk.byteLength;
        }
        assert(offset === output.byteLength, "PCM segment bytes did not reach EOF");
        const outputSha256 = await sha256(output);
        assert(
          outputSha256 === EXPECTED_OUTPUT_SHA256,
          `Output SHA-256 changed: ${outputSha256}`,
        );
        assert(totalSamples === EXPECTED_SAMPLE_COUNT, "Decoded sample count changed");
        const audioDecoding = parseAudioDecodingMetrics({
          applied: true,
          wallMs: message.decoderWallMs,
          activeMs: message.decoderActiveMs,
          inputByteLength,
          outputByteLength,
          overlapMs: 0,
          pcmWaitMs: 0,
          pcmStarvationMs: 0,
          pcmStarvationCount: 0,
        });
        resolve({
          outputByteLength,
          outputSha256,
          sampleCount: totalSamples,
          segmentCount,
          audioDecoding,
        });
      }, fail);
    };

    port.addEventListener("message", handleMessage);
    port.start();
  });
}

function emptyMetrics(
  audioDecoding: AudioDecodingMetrics,
): TranscriptionMetrics {
  return {
    audioDecoding,
    audioDurationSeconds: 0.1,
    processedAudioSeconds: 0.1,
    elapsedMs: FAKE_TRANSCRIPTION_WALL_MS,
    audioReadMs: 0,
    inferenceMs: 0,
    frontendMs: 0,
    encoderMs: 0,
    decoderMs: 0,
    speedFactor: 0,
    completedWindows: 1,
    totalWindows: 1,
    repairGapProbes: 0,
    repairWindowDecodes: 0,
    repairRounds: 0,
    recoveredTokens: 0,
    stitchingMs: 0,
    repairMs: 0,
    repairPrefetchWindowDecodes: 0,
    repairPrefetchCacheHits: 0,
    repairPrefetchUnusedDecodes: 0,
    batchSubmissionTimeline: [],
    batchSubmissionTimelineOmitted: 0,
    phaseTimings: {
      trailingSpeechScanMs: 0,
      adaptiveThresholdScanMs: 0,
      gapEnergyScanMs: 0,
      incrementalStitchMs: 0,
      prefixStitchMs: 0,
      finalStitchMs: 0,
      repairPlanningMs: 0,
      repairSplicingMs: 0,
      partialResultMs: 0,
      tokenDecodeMs: 0,
    },
    totalMs: FAKE_TRANSCRIPTION_WALL_MS,
  };
}

function parseAudioDecodingMetrics(value: unknown): AudioDecodingMetrics {
  assert(isRecord(value), "Worker message has no audio-decoding metrics");
  assert(typeof value.applied === "boolean", "Invalid applied metric");
  assertNonNegativeFinite(value.wallMs, "wallMs");
  assertNonNegativeFinite(value.activeMs, "activeMs");
  const inputByteLength = nonNegativeInteger(
    value.inputByteLength,
    "inputByteLength",
  );
  const outputByteLength = nonNegativeInteger(
    value.outputByteLength,
    "outputByteLength",
  );
  assertNonNegativeFinite(value.overlapMs, "overlapMs");
  assertNonNegativeFinite(value.pcmWaitMs, "pcmWaitMs");
  assertNonNegativeFinite(value.pcmStarvationMs, "pcmStarvationMs");
  const pcmStarvationCount = nonNegativeInteger(
    value.pcmStarvationCount,
    "pcmStarvationCount",
  );
  return {
    applied: value.applied,
    wallMs: value.wallMs,
    activeMs: value.activeMs,
    inputByteLength,
    outputByteLength,
    overlapMs: value.overlapMs,
    pcmWaitMs: value.pcmWaitMs,
    pcmStarvationMs: value.pcmStarvationMs,
    pcmStarvationCount,
  };
}

function stereoPcm16Wav8Khz(frameCount: number): File {
  const channels = 2;
  const sampleRate = 8_000;
  const bytesPerFrame = channels * 2;
  const dataByteLength = frameCount * bytesPerFrame;
  const bytes = new Uint8Array(44 + dataByteLength);
  const view = new DataView(bytes.buffer);
  writeFourCc(bytes, 0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  writeFourCc(bytes, 8, "WAVE");
  writeFourCc(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerFrame, true);
  view.setUint16(32, bytesPerFrame, true);
  view.setUint16(34, 16, true);
  writeFourCc(bytes, 36, "data");
  view.setUint32(40, dataByteLength, true);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const left = ((frame * 97 + 23) % 20_001) - 10_000;
    const right = ((frame * 53 + 1_019) % 16_001) - 8_000;
    view.setInt16(44 + frame * bytesPerFrame, left, true);
    view.setInt16(46 + frame * bytesPerFrame, right, true);
  }
  return new File([bytes], "stereo-8khz.wav", { type: "audio/wav" });
}

async function scratchEntryNames(): Promise<readonly string[]> {
  const storageRoot = await navigator.storage.getDirectory();
  let scratchRoot: FileSystemDirectoryHandle;
  try {
    scratchRoot = await storageRoot.getDirectoryHandle(SCRATCH_ROOT);
  } catch (error) {
    if (error instanceof DOMException && error.name === "NotFoundError") return [];
    throw error;
  }
  const names: string[] = [];
  for await (const [name] of (
    scratchRoot as IterableFileSystemDirectoryHandle
  ).entries()) {
    names.push(name);
  }
  return names.sort();
}

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function writeFourCc(bytes: Uint8Array, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    bytes[offset + index] = value.charCodeAt(index);
  }
}

function nonNegativeInteger(value: unknown, label: string): number {
  assert(
    typeof value === "number" && Number.isSafeInteger(value) && value >= 0,
    `${label} is not a non-negative integer`,
  );
  return value;
}

function assertNonNegativeFinite(value: unknown, label: string): asserts value is number {
  assert(
    typeof value === "number" && Number.isFinite(value) && value >= 0,
    `${label} is not a non-negative finite number`,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
