import { describe, expect, it } from "vitest";

import {
  isParakeetClientMessage,
  isParakeetWorkerMessage,
  serializeWorkerError,
} from "../../src/runtime/protocol";

const canonicalAudioDecoding = {
  applied: false,
  wallMs: 0,
  activeMs: 0,
  inputByteLength: 44,
  outputByteLength: 44,
  overlapMs: 0,
  pcmWaitMs: 0,
  pcmStarvationMs: 0,
  pcmStarvationCount: 0,
} as const;

describe("Parakeet worker shell protocol", () => {
  it("accepts fixed initialization and Blob transcription messages", () => {
    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 1,
        reportProgress: true,
        modelSource: "cache-only",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
        },
      }),
    ).toBe(true);

    const channel = new MessageChannel();
    expect(
      isParakeetClientMessage({
        type: "prepare-stream",
        jobId: 7,
        port: channel.port1,
        inputByteLength: 123,
        estimatedSampleCount: 456,
      }),
    ).toBe(true);
    expect(
      isParakeetClientMessage({
        type: "prepare-stream",
        jobId: 8,
        port: channel.port2,
        inputByteLength: 123,
        estimatedSampleCount: null,
      }),
    ).toBe(true);
    channel.port1.close();
    channel.port2.close();
    expect(
      isParakeetClientMessage({
        type: "cancel-initialization",
        requestId: 1,
      }),
    ).toBe(true);
    expect(
      isParakeetWorkerMessage({
        type: "initialization-cancelled",
        requestId: 1,
      }),
    ).toBe(true);
    expect(
      isParakeetClientMessage({
        type: "transcribe",
        jobId: 7,
        audio: new Blob(["wav"]),
        audioDecoding: canonicalAudioDecoding,
        reportProgress: true,
        reportPartialResults: true,
        reportPacedTranscriptBoundary: true,
      }),
    ).toBe(true);
  });

  it("rejects malformed envelopes and any non-contract configuration", () => {
    expect(
      isParakeetClientMessage({
        type: "transcribe",
        jobId: -1,
        audio: new Blob(),
        reportProgress: false,
        reportPartialResults: false,
        reportPacedTranscriptBoundary: false,
      }),
    ).toBe(false);
    const channel = new MessageChannel();
    expect(
      isParakeetClientMessage({
        type: "prepare-stream",
        jobId: 7,
        port: channel.port1,
        inputByteLength: 123,
        estimatedSampleCount: -1,
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "prepare-stream",
        jobId: 7,
        port: channel.port2,
        inputByteLength: 123,
      }),
    ).toBe(false);
    channel.port1.close();
    channel.port2.close();
    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 1,
        reportProgress: false,
        modelSource: "cache-or-network",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: 42,
        },
      }),
    ).toBe(false);

    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 2,
        reportProgress: false,
        modelSource: "cache-or-network",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
          unexpected: true,
        },
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 3,
        modelSource: "cache-or-network",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
        },
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 4,
        reportProgress: false,
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
        },
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "initialize",
        requestId: 5,
        reportProgress: false,
        modelSource: "network-only",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
        },
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "transcribe",
        jobId: 7,
        audio: new Blob(),
        reportProgress: false,
        reportPartialResults: false,
      }),
    ).toBe(false);
    expect(
      isParakeetClientMessage({
        type: "transcribe",
        jobId: 7,
        audio: new Blob(),
        reportProgress: false,
        reportPartialResults: false,
        reportPacedTranscriptBoundary: true,
      }),
    ).toBe(false);
    expect(isParakeetClientMessage({ type: "dispose" })).toBe(false);
  });

  it("recognizes provisional timestamped transcript snapshots", () => {
    expect(
      isParakeetWorkerMessage({
        type: "partial-result",
        jobId: 7,
        snapshot: {
          revision: 2,
          text: "hello world",
          tokens: [
            { tokenId: 4, startSeconds: 0, endSeconds: 0.08 },
          ],
          words: [
            { text: "hello", startSeconds: 0, endSeconds: 0.08 },
          ],
          audioDurationSeconds: 60,
          processedAudioSeconds: 12.88,
        },
      }),
    ).toBe(true);
    expect(
      isParakeetWorkerMessage({
        type: "paced-transcript-flush",
        jobId: 7,
      }),
    ).toBe(true);
    expect(
      isParakeetWorkerMessage({
        type: "paced-transcript-flush",
        jobId: -1,
      }),
    ).toBe(false);
  });

  it("recognizes scoped errors and preserves engine error codes", () => {
    const error = new Error("GPU adapter unavailable") as Error & {
      code: string;
    };
    error.name = "GpuInitializationError";
    error.code = "WEBGPU_UNAVAILABLE";

    const serialized = serializeWorkerError(error);
    expect(serialized).toMatchObject({
      name: "GpuInitializationError",
      message: "GPU adapter unavailable",
      code: "WEBGPU_UNAVAILABLE",
    });
    expect(
      isParakeetWorkerMessage({
        type: "error",
        requestId: 3,
        error: serialized,
      }),
    ).toBe(true);
  });
});
