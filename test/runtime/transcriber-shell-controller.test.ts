import { describe, expect, it, vi } from "vitest";

import { createTranscriber } from "../../src/api";

import {
  ParakeetTranscriber,
  type ParakeetLoadOptions,
  type PacedTranscriptUpdate,
  type ParakeetWorkerLike,
} from "../../src/runtime/transcriber";
import type {
  ParakeetClientMessage,
  ParakeetWorkerMessage,
  TranscriptionMetrics,
  TranscriptionResult,
  TranscriptionSnapshot,
} from "../../src/runtime/protocol";

const audioDecodingAssets = {
  ffmpegCoreWasm: "/parakeet-wgsl-ffmpeg-core.wasm",
} as const;

describe("ParakeetTranscriber worker controller", () => {
  it("rejects unsupported input before starting lazy model initialization", async () => {
    const worker = new FakeWorker();
    vi.stubGlobal("navigator", {});
    try {
      const transcriber = createTranscriber({
        workerFactory: () => worker as unknown as Worker,
      });

      await expect(
        transcriber.transcribe(new Blob(["compressed"]), {
          sourceName: "recording.mp3",
        }),
      ).rejects.toThrow(/Origin Private File System/);
      expect(worker.messages).toEqual([]);
      transcriber.dispose();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it.each([
    ["startup error", "error", "decoder startup failed"],
    ["postMessage exception", "throw", "decoder post failed"],
  ] as const)(
    "rejects a decoder %s before model initialization and removes opening listeners",
    async (_description, behavior, expectedMessage) => {
      const inferenceWorker = new FakeWorker();
      let decoderWorker: FakeAudioDecoderWorker | undefined;
      vi.stubGlobal("navigator", {
        storage: {
          getDirectory: async () => ({
            getDirectoryHandle: async () => ({
              removeEntry: async () => undefined,
            }),
          }),
        },
      });
      vi.stubGlobal(
        "Worker",
        class {
          constructor() {
            decoderWorker = new FakeAudioDecoderWorker(behavior);
            return decoderWorker as unknown as object;
          }
        },
      );

      try {
        const transcriber = createTranscriber({
          workerFactory: () => inferenceWorker as unknown as Worker,
        });
        await expect(
          transcriber.transcribe(new Blob(["compressed"]), {
            sourceName: "recording.mp3",
          }),
        ).rejects.toThrow(expectedMessage);
        expect(inferenceWorker.messages).toEqual([]);
        expect(decoderWorker?.terminated).toBe(true);
        expect(decoderWorker?.listenerCount).toBe(0);
        transcriber.dispose();
      } finally {
        vi.unstubAllGlobals();
      }
    },
  );

  it("prepares streaming decode before initialization and becomes terminal before scratch cleanup finishes", async () => {
    const inferenceWorker = new FakeWorker();
    let releaseCleanup!: () => void;
    const cleanup = new Promise<void>((resolve) => {
      releaseCleanup = resolve;
    });
    const removeEntry = vi.fn(() => cleanup);
    vi.stubGlobal("navigator", {
      storage: {
        getDirectory: async () => ({
          getDirectoryHandle: async () => ({ removeEntry }),
        }),
      },
    });
    vi.stubGlobal(
      "Worker",
      class {
        readonly listeners = new Set<
          (event: MessageEvent<unknown>) => void
        >();

        postMessage(message: unknown): void {
          const request = message as {
            readonly type: string;
            readonly requestId: string;
          };
          expect(request.type).toBe("start-streaming-decode");
          queueMicrotask(() => {
            for (const listener of this.listeners) {
              listener({
                data: {
                  type: "streaming-decoder-ready",
                  requestId: request.requestId,
                  estimatedSampleCount: 32_000,
                },
              } as MessageEvent<unknown>);
            }
          });
        }

        addEventListener(
          type: "message" | "error",
          listener: (event: MessageEvent<unknown>) => void,
        ): void {
          if (type === "message") this.listeners.add(listener);
        }

        removeEventListener(
          type: "message" | "error",
          listener: (event: MessageEvent<unknown>) => void,
        ): void {
          if (type === "message") this.listeners.delete(listener);
        }

        terminate(): void {}
      },
    );

    try {
      const transcriber = createTranscriber({
        workerFactory: () => inferenceWorker as unknown as Worker,
      });
      const pending = transcriber.transcribe(new Blob(["compressed"]), {
        sourceName: "recording.mp3",
      });
      await waitForMessageCount(inferenceWorker, "prepare-stream", 1);
      await waitForMessageCount(inferenceWorker, "initialize", 1);
      expect(
        inferenceWorker.messages.map(({ type }) => type).slice(0, 2),
      ).toEqual(["prepare-stream", "initialize"]);
      inferenceWorker.emit({
        type: "ready",
        requestId: 1,
        diagnostics: {},
      });
      await waitForMessageCount(inferenceWorker, "transcribe-stream", 1);
      inferenceWorker.emit({
        type: "result",
        jobId: 1,
        result: transcriptionResult(),
      });

      expect(transcriber.cancel()).toBe(false);
      await vi.waitFor(() => expect(removeEntry).toHaveBeenCalledOnce());
      releaseCleanup();
      await expect(pending).resolves.toEqual(transcriptionResult());
      const prepared = inferenceWorker.messages.find(
        (message) => message.type === "prepare-stream",
      );
      if (prepared?.type === "prepare-stream") {
        expect(prepared.estimatedSampleCount).toBe(32_000);
        prepared.port.close();
      }
      transcriber.dispose();
    } finally {
      releaseCleanup();
      vi.unstubAllGlobals();
    }
  });

  it("defers initialization until transcribe and then reuses the loaded engine", async () => {
    const worker = new FakeWorker();
    const transcriber = createTranscriber({
      modelUrls: {
        fp16: "https://example.test/fp16/manifest.json",
        fp32: "https://example.test/fp32/manifest.json",
      },
      wasmUrl: "https://example.test/parakeet-fbank.wasm",
      workerFactory: () => worker as unknown as Worker,
    });

    expect(worker.messages).toEqual([]);
    expect(transcriber.initialized).toBe(false);
    expect(transcriber.diagnostics).toBeNull();

    const first = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "initialize", 1);
    expect(worker.messages).toEqual([
      {
        type: "initialize",
        requestId: 1,
        modelSource: "cache-or-network",
        reportProgress: false,
        configuration: {
          fp16ModelUrl: "https://example.test/fp16/manifest.json",
          fp32ModelUrl: "https://example.test/fp32/manifest.json",
          wasmUrl: "https://example.test/parakeet-fbank.wasm",
        },
      },
    ]);
    worker.emit({
      type: "ready",
      requestId: 1,
      diagnostics: { precision: "fp16" },
    });
    await waitForMessageCount(worker, "transcribe", 1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      jobId: 1,
    });
    worker.emit({ type: "result", jobId: 1, result: transcriptionResult() });
    await expect(first).resolves.toEqual(transcriptionResult());

    const second = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "transcribe", 2);
    expect(worker.messages.filter(({ type }) => type === "initialize")).toHaveLength(1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      jobId: 2,
    });
    worker.emit({ type: "result", jobId: 2, result: transcriptionResult() });
    await expect(second).resolves.toEqual(transcriptionResult());
    transcriber.dispose();
  });

  it("coalesces cached warmup and a concurrent transcription", async () => {
    const worker = new FakeWorker();
    const transcriber = createTranscriber({
      workerFactory: () => worker as unknown as Worker,
    });

    const firstWarmup = transcriber.loadCachedModel();
    const secondWarmup = transcriber.loadCachedModel();
    expect(worker.messages).toHaveLength(1);
    expect(worker.messages[0]).toMatchObject({
      type: "initialize",
      requestId: 1,
      modelSource: "cache-only",
      reportProgress: false,
    });

    const transcription = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "initialize", 1);
    expect(
      worker.messages.filter(({ type }) => type === "initialize"),
    ).toHaveLength(1);
    expect(
      worker.messages.filter(({ type }) => type === "transcribe"),
    ).toHaveLength(0);

    worker.emit({
      type: "ready",
      requestId: 1,
      diagnostics: { precision: "fp16" },
    });
    await expect(firstWarmup).resolves.toBe(true);
    await expect(secondWarmup).resolves.toBe(true);
    await waitForMessageCount(worker, "transcribe", 1);
    worker.emit({
      type: "result",
      jobId: 1,
      result: transcriptionResult(),
    });
    await expect(transcription).resolves.toEqual(transcriptionResult());
    transcriber.dispose();
  });

  it("falls through from a shared cache miss to one user-authorized load", async () => {
    const worker = new FakeWorker();
    const transcriber = createTranscriber({
      workerFactory: () => worker as unknown as Worker,
    });
    const warmup = transcriber.loadCachedModel();
    const transcription = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "initialize", 1);

    worker.emit({
      type: "error",
      requestId: 1,
      error: {
        name: "ModelNotCachedError",
        message: "The selected model is not completely cached",
        code: "MODEL_NOT_CACHED",
      },
    });
    await expect(warmup).resolves.toBe(false);
    await waitForMessageCount(worker, "initialize", 2);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "initialize",
      requestId: 2,
      modelSource: "cache-or-network",
    });

    worker.emit({ type: "ready", requestId: 2, diagnostics: {} });
    await waitForMessageCount(worker, "transcribe", 1);
    worker.emit({
      type: "result",
      jobId: 1,
      result: transcriptionResult(),
    });
    await expect(transcription).resolves.toEqual(transcriptionResult());
    transcriber.dispose();
  });

  it("does not cancel a cached warmup when its joining job aborts", async () => {
    const worker = new FakeWorker();
    const controller = new AbortController();
    const transcriber = createTranscriber({
      workerFactory: () => worker as unknown as Worker,
    });
    const warmup = transcriber.loadCachedModel();
    const transcription = transcriber.transcribe(makeWaveBlob(), {
      signal: controller.signal,
    });
    const rejection = expect(transcription).rejects.toMatchObject({
      name: "AbortError",
    });
    await waitForMessageCount(worker, "initialize", 1);
    controller.abort();
    await rejection;
    expect(
      worker.messages.filter(
        ({ type }) => type === "cancel-initialization",
      ),
    ).toHaveLength(0);

    worker.emit({ type: "ready", requestId: 1, diagnostics: {} });
    await expect(warmup).resolves.toBe(true);
    expect(
      worker.messages.filter(({ type }) => type === "transcribe"),
    ).toHaveLength(0);
    transcriber.dispose();
  });

  it("aborts lazy model loading and can initialize again on the next call", async () => {
    const worker = new FakeWorker();
    const controller = new AbortController();
    const transcriber = createTranscriber({
      wasmUrl: "https://example.test/parakeet-fbank.wasm",
      workerFactory: () => worker as unknown as Worker,
    });
    const first = transcriber.transcribe(makeWaveBlob(), {
      signal: controller.signal,
    });
    const firstRejection = expect(first).rejects.toMatchObject({
      name: "AbortError",
    });
    await waitForMessageCount(worker, "initialize", 1);
    controller.abort();
    expect(worker.messages.at(-1)).toEqual({
      type: "cancel-initialization",
      requestId: 1,
    });
    worker.emit({ type: "initialization-cancelled", requestId: 1 });
    await firstRejection;

    const second = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "initialize", 2);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "initialize",
      requestId: 2,
    });
    worker.emit({ type: "ready", requestId: 2, diagnostics: {} });
    await waitForMessageCount(worker, "transcribe", 1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      jobId: 2,
    });
    worker.emit({ type: "result", jobId: 2, result: transcriptionResult() });
    await expect(second).resolves.toEqual(transcriptionResult());
    transcriber.dispose();
  });

  it("initializes, reports progress, and structured-clones the original Blob", async () => {
    const worker = new FakeWorker();
    const loadProgress: number[] = [];
    const creating = createWithWorker(
      {
        ...audioDecodingAssets,
        fp16ModelUrl: new URL("https://example.test/model/manifest.json"),
        fp32ModelUrl: new URL(
          "https://example.test/model-fp32/manifest.json",
        ),
        wasmUrl: "/parakeet-fbank.wasm",
        onLoadProgress: (progress) => loadProgress.push(progress.fraction),
      },
      worker,
    );

    expect(worker.messages[0]).toEqual({
      type: "initialize",
      requestId: 1,
      modelSource: "cache-or-network",
      reportProgress: true,
      configuration: {
        fp16ModelUrl: "https://example.test/model/manifest.json",
        fp32ModelUrl: "https://example.test/model-fp32/manifest.json",
        wasmUrl: "/parakeet-fbank.wasm",
      },
    });
    worker.emit({
      type: "load-progress",
      requestId: 1,
      progress: { phase: "weights", fraction: 0.5 },
    });
    worker.emit({
      type: "ready",
      requestId: 1,
      diagnostics: { adapter: "test-gpu" },
    });

    const transcriber = await creating;
    expect(loadProgress).toEqual([0.5]);
    expect(transcriber.diagnostics).toEqual({ adapter: "test-gpu" });

    const audio = makeWaveBlob();
    const arrayBuffer = vi.spyOn(audio, "arrayBuffer");
    const progressFractions: number[] = [];
    const transcription = transcriber.transcribe(audio, {
      onProgress: (progress) => progressFractions.push(progress.fraction),
    });

    expect(arrayBuffer).not.toHaveBeenCalled();
    await waitForMessageCount(worker, "transcribe", 1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      jobId: 1,
      audio,
      audioDecoding: {
        applied: false,
        inputByteLength: 46,
        outputByteLength: 2,
      },
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    worker.emit({
      type: "progress",
      jobId: 1,
      progress: {
        phase: "transcribing",
        fraction: 0.25,
        metrics: partialMetrics(),
      },
    });
    const expected = transcriptionResult();
    worker.emit({ type: "result", jobId: 1, result: expected });

    await expect(transcription).resolves.toEqual(expected);
    expect(progressFractions).toEqual([0.25]);
    expect(arrayBuffer).not.toHaveBeenCalled();
    const sentMessageCount = worker.messages.length;
    transcriber.dispose();
    expect(worker.messages).toHaveLength(sentMessageCount);
    expect(worker.terminated).toBe(true);
    expect(worker.listenerCount).toBe(0);
  });

  it("delivers replaceable snapshots only for the active job", async () => {
    const worker = new FakeWorker();
    const transcriber = await readyTranscriber(worker);
    const snapshots: TranscriptionSnapshot[] = [];
    const transcription = transcriber.transcribe(makeWaveBlob(), {
      onPartialResult: (snapshot) => snapshots.push(snapshot),
    });
    await waitForMessageCount(worker, "transcribe", 1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      jobId: 1,
      reportProgress: false,
      reportPartialResults: true,
      reportPacedTranscriptBoundary: false,
    });

    const snapshot = transcriptionSnapshot();
    worker.emit({
      type: "partial-result",
      jobId: 2,
      snapshot: { ...snapshot, revision: 99 },
    });
    worker.emit({ type: "partial-result", jobId: 1, snapshot });
    expect(snapshots).toEqual([snapshot]);

    const expected = transcriptionResult();
    worker.emit({ type: "result", jobId: 1, result: expected });
    await expect(transcription).resolves.toEqual(expected);
    worker.emit({
      type: "partial-result",
      jobId: 1,
      snapshot: { ...snapshot, revision: 2 },
    });
    expect(snapshots).toEqual([snapshot]);
    transcriber.dispose();
  });

  it("offers an opt-in paced track with immediate final fallback", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      const transcriber = await readyTranscriber(worker);
      const updates: PacedTranscriptUpdate[] = [];
      const transcription = transcriber.transcribe(makeWaveBlob(), {
        onPacedTranscript: (update) => updates.push(update),
      });
      await waitForMessageCount(worker, "transcribe", 1);
      expect(worker.messages.at(-1)).toMatchObject({
        type: "transcribe",
        jobId: 1,
        reportProgress: false,
        reportPartialResults: true,
        reportPacedTranscriptBoundary: true,
      });

      const snapshot = transcriptionSnapshot();
      worker.emit({ type: "partial-result", jobId: 1, snapshot });
      const expected = transcriptionResult();
      worker.emit({ type: "result", jobId: 1, result: expected });

      await expect(transcription).resolves.toEqual(expected);
      expect(updates.at(-1)).toMatchObject({
        isFinal: true,
        textSplice: {
          startOffset: 0,
          deleteCount: 0,
          insertText: "hello",
        },
      });
      expect(updates.flatMap(({ wordSplice }) => wordSplice.insertWords))
        .toEqual([
          {
            text: "hello",
            startSeconds: 0,
            endSeconds: 0.08,
            startOffset: 0,
            endOffset: 5,
          },
        ]);
      transcriber.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it("flushes primary output at the boundary and applies repair immediately", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      const transcriber = await readyTranscriber(worker);
      const updates: PacedTranscriptUpdate[] = [];
      const transcription = transcriber.transcribe(makeWaveBlob(), {
        onPacedTranscript: (update) => updates.push(update),
      });
      await waitForMessageCount(worker, "transcribe", 1);
      worker.emit({
        type: "partial-result",
        jobId: 1,
        snapshot: transcriptionSnapshot(),
      });
      expect(updates).toHaveLength(0);
      worker.emit({
        type: "paced-transcript-flush",
        jobId: 2,
      });
      expect(updates).toHaveLength(0);
      worker.emit({
        type: "paced-transcript-flush",
        jobId: 1,
      });
      expect(updates.at(-1)).toMatchObject({
        isFinal: false,
        textSplice: {
          startOffset: 0,
          deleteCount: 0,
          insertText: "hello",
        },
      });

      const repairedSnapshot: TranscriptionSnapshot = {
        ...transcriptionSnapshot(),
        revision: 2,
        text: "hello repaired",
        words: [
          { text: "hello", startSeconds: 0, endSeconds: 0.08 },
          { text: "repaired", startSeconds: 0.1, endSeconds: 0.18 },
        ],
      };
      worker.emit({
        type: "partial-result",
        jobId: 1,
        snapshot: repairedSnapshot,
      });
      expect(
        updates.flatMap(({ wordSplice }) => wordSplice.insertWords)
          .map(({ text }) => text),
      ).toEqual(["hello", "repaired"]);

      const expected: TranscriptionResult = {
        ...transcriptionResult(),
        text: repairedSnapshot.text,
        words: repairedSnapshot.words,
      };
      worker.emit({
        type: "result",
        jobId: 1,
        result: expected,
      });
      await expect(transcription).resolves.toEqual(expected);
      expect(updates.at(-1)?.isFinal).toBe(true);
      expect(updates.at(-1)?.textSplice.insertText).toBe("");
      expect(transcriber.cancel()).toBe(false);
      transcriber.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops paced output immediately while worker cancellation is pending", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      const transcriber = await readyTranscriber(worker);
      const controller = new AbortController();
      const updates: PacedTranscriptUpdate[] = [];
      const transcription = transcriber.transcribe(makeWaveBlob(), {
        signal: controller.signal,
        onPacedTranscript: (update) => updates.push(update),
      });
      const rejection = expect(transcription).rejects.toMatchObject({
        name: "AbortError",
      });
      await waitForMessageCount(worker, "transcribe", 1);
      worker.emit({
        type: "partial-result",
        jobId: 1,
        snapshot: transcriptionSnapshot(),
      });

      controller.abort();
      expect(worker.messages.at(-1)).toEqual({
        type: "cancel",
        jobId: 1,
      });
      await vi.advanceTimersByTimeAsync(500);
      expect(updates).toHaveLength(0);

      worker.emit({ type: "cancelled", jobId: 1 });
      await rejection;
      transcriber.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it("turns a paced subscriber exception into a cancelled job", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      const transcriber = await readyTranscriber(worker);
      const callbackError = new Error("paced consumer failed");
      const transcription = transcriber.transcribe(makeWaveBlob(), {
        onPacedTranscript: () => {
          throw callbackError;
        },
      });
      const rejection = expect(transcription).rejects.toBe(callbackError);
      await waitForMessageCount(worker, "transcribe", 1);
      worker.emit({
        type: "partial-result",
        jobId: 1,
        snapshot: transcriptionSnapshot(),
      });

      await vi.advanceTimersByTimeAsync(200);
      expect(worker.messages.at(-1)).toEqual({
        type: "cancel",
        jobId: 1,
      });
      worker.emit({ type: "cancelled", jobId: 1 });
      await rejection;
      transcriber.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it("disables worker traffic when no progress subscriber exists", async () => {
    const worker = new FakeWorker();
    const creating = createWithWorker(
      {
        ...audioDecodingAssets,
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
      worker,
    );
    expect(worker.messages[0]).toMatchObject({
      type: "initialize",
      reportProgress: false,
    });
    worker.emit({
      type: "ready",
      requestId: 1,
      diagnostics: {},
    });
    const transcriber = await creating;

    const transcription = transcriber.transcribe(makeWaveBlob());
    await waitForMessageCount(worker, "transcribe", 1);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "transcribe",
      reportProgress: false,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    const expected = transcriptionResult();
    worker.emit({ type: "result", jobId: 1, result: expected });
    await expect(transcription).resolves.toEqual(expected);
    transcriber.dispose();
  });

  it("enforces one active job and maps worker cancellation to AbortError", async () => {
    const worker = new FakeWorker();
    const transcriber = await readyTranscriber(worker);
    const controller = new AbortController();
    const first = transcriber.transcribe(makeWaveBlob(), {
      signal: controller.signal,
    });
    const firstRejection = expect(first).rejects.toMatchObject({
      name: "AbortError",
    });

    await expect(
      transcriber.transcribe(makeWaveBlob()),
    ).rejects.toMatchObject({ code: "BUSY" });

    await waitForMessageCount(worker, "transcribe", 1);
    controller.abort();
    expect(worker.messages.at(-1)).toEqual({ type: "cancel", jobId: 1 });
    worker.emit({ type: "cancelled", jobId: 1 });
    await firstRejection;
    expect(transcriber.cancel()).toBe(false);
    transcriber.dispose();
  });

  it("surfaces initialization errors with their worker error code", async () => {
    const worker = new FakeWorker();
    const creating = createWithWorker(
      {
        ...audioDecodingAssets,
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
      worker,
    );
    worker.emit({
      type: "error",
      requestId: 1,
      error: {
        name: "ParakeetRuntimeError",
        message: "Raw WebGPU backend is not connected",
        code: "ENGINE_NOT_INITIALIZED",
      },
    });

    await expect(creating).rejects.toMatchObject({
      message: "Raw WebGPU backend is not connected",
      code: "ENGINE_NOT_INITIALIZED",
    });
    expect(worker.terminated).toBe(true);
  });

  it("makes a fatal worker error terminal and rejects active work", async () => {
    const worker = new FakeWorker();
    const transcriber = await readyTranscriber(worker);
    const crash = new Error("worker crashed");
    const transcription = transcriber.transcribe(makeWaveBlob());
    const rejection = expect(transcription).rejects.toBe(crash);
    await waitForMessageCount(worker, "transcribe", 1);

    worker.emitError(crash);

    await rejection;
    expect(worker.terminated).toBe(true);
    expect(worker.listenerCount).toBe(0);
    await expect(
      transcriber.transcribe(makeWaveBlob()),
    ).rejects.toMatchObject({ code: "TRANSCRIBER_DISPOSED" });
    expect(worker.messages.filter((message) => message.type === "transcribe"))
      .toHaveLength(1);
  });
});

class FakeAudioDecoderWorker {
  terminated = false;

  private readonly messageListeners = new Set<
    (event: MessageEvent<unknown>) => void
  >();
  private readonly errorListeners = new Set<
    (event: ErrorEvent) => void
  >();

  constructor(private readonly behavior: "error" | "throw") {}

  get listenerCount(): number {
    return this.messageListeners.size + this.errorListeners.size;
  }

  postMessage(): void {
    if (this.behavior === "throw") {
      throw new Error("decoder post failed");
    }
    queueMicrotask(() => {
      const error = new Error("decoder startup failed");
      const event = {
        error,
        message: error.message,
      } as ErrorEvent;
      for (const listener of this.errorListeners) listener(event);
    });
  }

  addEventListener(
    type: "message" | "error",
    listener:
      | ((event: MessageEvent<unknown>) => void)
      | ((event: ErrorEvent) => void),
  ): void {
    if (type === "message") {
      this.messageListeners.add(
        listener as (event: MessageEvent<unknown>) => void,
      );
    } else {
      this.errorListeners.add(listener as (event: ErrorEvent) => void);
    }
  }

  removeEventListener(
    type: "message" | "error",
    listener:
      | ((event: MessageEvent<unknown>) => void)
      | ((event: ErrorEvent) => void),
  ): void {
    if (type === "message") {
      this.messageListeners.delete(
        listener as (event: MessageEvent<unknown>) => void,
      );
    } else {
      this.errorListeners.delete(listener as (event: ErrorEvent) => void);
    }
  }

  terminate(): void {
    this.terminated = true;
  }
}

class FakeWorker implements ParakeetWorkerLike {
  readonly messages: ParakeetClientMessage[] = [];
  terminated = false;

  private readonly messageListeners = new Set<
    (event: MessageEvent<unknown>) => void
  >();
  private readonly errorListeners = new Set<(event: ErrorEvent) => void>();

  postMessage(message: ParakeetClientMessage): void {
    this.messages.push(message);
  }

  get listenerCount(): number {
    return this.messageListeners.size + this.errorListeners.size;
  }

  terminate(): void {
    this.terminated = true;
  }

  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  addEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
  addEventListener(
    type: "message" | "error",
    listener:
      | ((event: MessageEvent<unknown>) => void)
      | ((event: ErrorEvent) => void),
  ): void {
    if (type === "message") {
      this.messageListeners.add(
        listener as (event: MessageEvent<unknown>) => void,
      );
    } else {
      this.errorListeners.add(listener as (event: ErrorEvent) => void);
    }
  }

  removeEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  removeEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
  removeEventListener(
    type: "message" | "error",
    listener:
      | ((event: MessageEvent<unknown>) => void)
      | ((event: ErrorEvent) => void),
  ): void {
    if (type === "message") {
      this.messageListeners.delete(
        listener as (event: MessageEvent<unknown>) => void,
      );
    } else {
      this.errorListeners.delete(listener as (event: ErrorEvent) => void);
    }
  }

  emit(message: ParakeetWorkerMessage): void {
    const event = new MessageEvent("message", { data: message });
    for (const listener of this.messageListeners) listener(event);
  }

  emitError(error: Error): void {
    const event = {
      error,
      message: error.message,
    } as ErrorEvent;
    for (const listener of this.errorListeners) listener(event);
  }
}

async function readyTranscriber(
  worker: FakeWorker,
): Promise<ParakeetTranscriber> {
  const creating = createWithWorker(
    {
      ...audioDecodingAssets,
      fp16ModelUrl: "/model",
      fp32ModelUrl: "/model-fp32",
      wasmUrl: "/wasm",
    },
    worker,
  );
  worker.emit({
    type: "ready",
    requestId: 1,
    diagnostics: {},
  });
  return creating;
}

function createWithWorker(
  options: ParakeetLoadOptions,
  worker: FakeWorker,
): Promise<ParakeetTranscriber> {
  class WorkerStub {
    constructor(url: URL, workerOptions: WorkerOptions) {
      expect(url.pathname).toMatch(/\/worker-entry\.ts$/);
      expect(workerOptions).toEqual({
        type: "module",
        name: "parakeet-wgsl",
      });
      return worker as unknown as WorkerStub;
    }
  }
  vi.stubGlobal("Worker", WorkerStub);
  try {
    return ParakeetTranscriber.create(options);
  } finally {
    vi.unstubAllGlobals();
  }
}

async function waitForMessageCount(
  worker: FakeWorker,
  type: ParakeetClientMessage["type"],
  count: number,
): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (worker.messages.filter((message) => message.type === type).length >= count) {
      return;
    }
    await Promise.resolve();
  }
  expect(worker.messages.filter((message) => message.type === type)).toHaveLength(
    count,
  );
}

function partialMetrics(): TranscriptionMetrics {
  return {
    audioDecoding: {
      applied: false,
      wallMs: 0,
      activeMs: 0,
      inputByteLength: 44,
      outputByteLength: 44,
      overlapMs: 0,
      pcmWaitMs: 0,
      pcmStarvationMs: 0,
      pcmStarvationCount: 0,
    },
    audioDurationSeconds: 1,
    processedAudioSeconds: 0.25,
    elapsedMs: 10,
    audioReadMs: 1,
    inferenceMs: 8,
    frontendMs: 1,
    encoderMs: 5,
    decoderMs: 2,
    speedFactor: 25,
    completedWindows: 1,
    totalWindows: 4,
    repairGapProbes: 0,
    repairWindowDecodes: 0,
    repairRounds: 0,
    recoveredTokens: 0,
    stitchingMs: 0,
    repairMs: 0,
    repairPrefetchWindowDecodes: 0,
    repairPrefetchCacheHits: 0,
    repairPrefetchUnusedDecodes: 0,
    batchSubmissionTimeline: [
      {
        submissionIndex: 0,
        batchClass: "primary",
        activeRows: 1,
        primaryRows: 1,
        repairRows: 0,
        officialRepairRows: 0,
        finalPrimaryPrefetchRows: 0,
        crossRoundPrefetchRows: 0,
        preparationStartedOffsetMs: 0,
        submissionStartedOffsetMs: 1,
        completionOffsetMs: 9,
        preparationMs: 1,
        submissionToCompletionMs: 8,
        frontendMs: 1,
        gpuIntervalMs: 7,
      },
    ],
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
    totalMs: 10,
  };
}

function transcriptionResult(): TranscriptionResult {
  return {
    text: "hello",
    tokens: [{ tokenId: 4, startSeconds: 0, endSeconds: 0.08 }],
    words: [{ text: "hello", startSeconds: 0, endSeconds: 0.08 }],
    metrics: partialMetrics(),
  };
}

function transcriptionSnapshot(): TranscriptionSnapshot {
  return {
    revision: 1,
    text: "hello",
    tokens: [{ tokenId: 4, startSeconds: 0, endSeconds: 0.08 }],
    words: [{ text: "hello", startSeconds: 0, endSeconds: 0.08 }],
    audioDurationSeconds: 60,
    processedAudioSeconds: 12.88,
  };
}

function makeWaveBlob(): Blob {
  const bytes = new Uint8Array(46);
  const view = new DataView(bytes.buffer);
  writeFourCc(bytes, 0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  writeFourCc(bytes, 8, "WAVE");
  writeFourCc(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16_000, true);
  view.setUint32(28, 32_000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeFourCc(bytes, 36, "data");
  view.setUint32(40, 2, true);
  return new Blob([bytes]);
}

function writeFourCc(
  target: Uint8Array,
  offset: number,
  value: string,
): void {
  for (let index = 0; index < value.length; index += 1) {
    target[offset + index] = value.charCodeAt(index);
  }
}
