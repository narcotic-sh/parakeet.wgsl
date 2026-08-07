import { describe, expect, it } from "vitest";

import type {
  ParakeetWorkerMessage,
  TranscriptionProgress,
} from "../../src/runtime/protocol";
import {
  ParakeetWorkerRuntime,
  type EngineBatchExecutionProgress,
  type ParakeetInferenceEngine,
  type ParakeetInferenceEngineFactory,
} from "../../src/runtime/worker";
import {
  PARAKEET_PRIMARY_VISIBLE_SAMPLES,
  PARAKEET_WINDOW_STRIDE_SAMPLES,
} from "../../src/runtime/chunking";

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

const convertedAudioDecoding = {
  applied: true,
  wallMs: 10_000,
  activeMs: 29,
  inputByteLength: 12_345,
  outputByteLength: 520_044,
  overlapMs: 0,
  pcmWaitMs: 0,
  pcmStarvationMs: 0,
  pcmStarvationCount: 0,
} as const;

describe("Parakeet dedicated worker runtime", () => {
  it("forwards the model source and remains reusable after a cache miss", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const modelSources: string[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async (_configuration, context) => {
        modelSources.push(context.modelSource);
        if (context.modelSource === "cache-only") {
          throw Object.assign(
            new Error("The selected model package is not completely cached"),
            { code: "MODEL_NOT_CACHED" },
          );
        }
        return testEngine({
          diagnostics: { fbankHopSamples: 160 },
        });
      },
    });

    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-only",
      configuration: {
        fp16ModelUrl: "/fp16/manifest.json",
        fp32ModelUrl: "/fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      },
    });
    expect(messages).toContainEqual({
      type: "error",
      requestId: 1,
      error: expect.objectContaining({
        message: "The selected model package is not completely cached",
        code: "MODEL_NOT_CACHED",
      }),
    });

    await runtime.handleMessage({
      type: "initialize",
      requestId: 2,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/fp16/manifest.json",
        fp32ModelUrl: "/fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      },
    });

    expect(modelSources).toEqual(["cache-only", "cache-or-network"]);
    expect(messages).toContainEqual(
      expect.objectContaining({ type: "ready", requestId: 2 }),
    );
  });

  it("aborts model initialization in place and remains reusable", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let factoryCalls = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async (_configuration, context) => {
        factoryCalls += 1;
        if (factoryCalls === 1) {
          await new Promise<never>((_resolve, reject) => {
            context.signal?.addEventListener(
              "abort",
              () => reject(context.signal?.reason),
              { once: true },
            );
          });
        }
        return testEngine({
          diagnostics: { fbankHopSamples: 160 },
        });
      },
    });
    const first = runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/fp16/manifest.json",
        fp32ModelUrl: "/fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      },
    });
    await Promise.resolve();
    await runtime.handleMessage({
      type: "cancel-initialization",
      requestId: 1,
    });
    await first;
    expect(messages).toContainEqual({
      type: "initialization-cancelled",
      requestId: 1,
    });

    await runtime.handleMessage({
      type: "initialize",
      requestId: 2,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/fp16/manifest.json",
        fp32ModelUrl: "/fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      },
    });
    expect(factoryCalls).toBe(2);
    expect(messages).toContainEqual(
      expect.objectContaining({ type: "ready", requestId: 2 }),
    );
  });

  it("streams fixed windows through an injected engine and reports stitched metrics", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const sampleLengths: number[] = [];
    const sourceStarts: number[] = [];
    const batchSizes: number[] = [];
    const factory: ParakeetInferenceEngineFactory = async (
      configuration,
      context,
    ) => {
      expect(configuration).toEqual({
        fp16ModelUrl: "/model/manifest.json",
        fp32ModelUrl: "/model-fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      });
      context.onProgress?.({ phase: "weights", fraction: 0.5 });
      return testEngine({
        diagnostics: {
          backend: "injected-test-engine",
          fbankHopSamples: 160,
        },
        preferredBatchSize: 2,
        async transcribeWindows(windows) {
          batchSizes.push(windows.length);
          return windows.map((window) => {
            sampleLengths.push(window.samples.length);
            sourceStarts.push(window.sourceStartSample);
            if (window.index === 0) {
              return {
                tokens: [
                  {
                    tokenId: 1,
                    startSample: 1_280,
                    endSample: 2_560,
                    durationFrames: 1,
                  },
                  {
                    tokenId: 2,
                    startSample: 217_600,
                    endSample: 218_880,
                    durationFrames: 1,
                  },
                ],
                timings: { frontendMs: 1, encoderMs: 2, decoderMs: 3 },
              };
            }
            return {
              tokens: [
                {
                  tokenId: 2,
                  startSample: 217_600,
                  endSample: 218_880,
                  durationFrames: 1,
                },
                {
                  tokenId: 3,
                  startSample: 249_600,
                  endSample: 250_880,
                  durationFrames: 1,
                },
              ],
              timings: { frontendMs: 1, encoderMs: 2, decoderMs: 3 },
            };
          });
        },
        decodeTokenIds(tokenIds) {
          return tokenIds.join(" ");
        },
      });
    };
    let tick = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: factory,
      now: () => {
        tick += 1;
        return tick;
      },
    });

    await runtime.handleMessage({
      type: "initialize",
      requestId: 10,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model/manifest.json",
        fp32ModelUrl: "/model-fp32/manifest.json",
        wasmUrl: "/fbank.wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 20,
      audio: makeWaveBlob(260_000),
      audioDecoding: convertedAudioDecoding,
      reportProgress: true,
      reportPartialResults: true,
      reportPacedTranscriptBoundary: true,
    });

    expect(batchSizes).toEqual([2]);
    expect(sampleLengths).toEqual([240_000, 240_000]);
    expect(sourceStarts).toEqual([0, 23_040]);
    expect(messages).toContainEqual({
      type: "load-progress",
      requestId: 10,
      progress: { phase: "weights", fraction: 0.5 },
    });
    expect(messages).toContainEqual({
      type: "ready",
      requestId: 10,
      diagnostics: {
        backend: "injected-test-engine",
        windowSeconds: 15,
        windowSamples: 240_000,
        primaryVisibleSamples: 238_080,
        primaryContextSamples: 1_280,
        windowStrideSamples: 206_080,
        overlapSeconds: 2,
        overlapSamples: 32_000,
        repairVisibleSamples: 239_360,
        seamGapRepairMinFrames: 15,
        seamGapRepairMaxProbes: 32,
        seamGapRepairMaxRounds: 3,
        fbankHopSamples: 160,
        encoderFrameSamples: 1_280,
      },
    });

    const transcribingProgress = expectIntegratedProgress(messages);
    expect(
      transcribingProgress.map(
        ({ metrics: { completedWindows } }) => completedWindows,
      ),
    ).toContain(2);
    const partialResults = messages.filter(
      (message) => message.type === "partial-result",
    );
    expect(partialResults).toHaveLength(1);
    expect(partialResults[0]).toMatchObject({
      type: "partial-result",
      jobId: 20,
      snapshot: {
        revision: 1,
        text: "1 2 3",
        processedAudioSeconds: 16.25,
        audioDurationSeconds: 16.25,
        words: [
          { text: "1", startSeconds: 0.08, endSeconds: 0.16 },
          { text: "2", startSeconds: 13.6, endSeconds: 13.68 },
          { text: "3", startSeconds: 15.6, endSeconds: 15.68 },
        ],
      },
    });
    const finalPrimaryPartialIndex = messages.findIndex(
      (message) => message.type === "partial-result",
    );
    const pacedFlushIndex = messages.findIndex(
      (message) => message.type === "paced-transcript-flush",
    );
    const postFlushProgressIndex = messages.findIndex(
      (message, index) =>
        index > pacedFlushIndex &&
        message.type === "progress" &&
        message.progress.phase === "transcribing",
    );
    expect(finalPrimaryPartialIndex).toBeGreaterThanOrEqual(0);
    expect(pacedFlushIndex).toBeGreaterThan(finalPrimaryPartialIndex);
    expect(postFlushProgressIndex).toBeGreaterThan(pacedFlushIndex);

    const result = messages.find((message) => message.type === "result");
    expect(result?.type).toBe("result");
    if (result?.type !== "result") throw new Error("missing result");
    expect(result.result.text).toBe("1 2 3");
    expect(result.result.tokens.map((token) => token.tokenId)).toEqual([
      1, 2, 3,
    ]);
    expect(result.result.words.map((word) => word.text)).toEqual([
      "1",
      "2",
      "3",
    ]);
    expect(result.result.metrics).toMatchObject({
      audioDecoding: convertedAudioDecoding,
      audioDurationSeconds: 16.25,
      processedAudioSeconds: 16.25,
      completedWindows: 2,
      totalWindows: 2,
      frontendMs: 2,
      encoderMs: 4,
      decoderMs: 6,
    });
    expect(result.result.metrics.batchSubmissionTimeline).toHaveLength(1);
    expect(result.result.metrics.batchSubmissionTimeline[0]).toMatchObject({
      submissionIndex: 0,
      batchClass: "primary",
      activeRows: 2,
      primaryRows: 2,
      repairRows: 0,
      officialRepairRows: 0,
      finalPrimaryPrefetchRows: 0,
      crossRoundPrefetchRows: 0,
      frontendMs: 2,
      gpuIntervalMs: 10,
    });
    const batch = result.result.metrics.batchSubmissionTimeline[0]!;
    expect(batch.preparationStartedOffsetMs).toBeGreaterThanOrEqual(0);
    expect(batch.preparationStartedOffsetMs).toBeLessThan(
      convertedAudioDecoding.wallMs,
    );
    expect(batch.preparationStartedOffsetMs).toBeLessThanOrEqual(
      batch.submissionStartedOffsetMs,
    );
    expect(batch.submissionStartedOffsetMs).toBeLessThanOrEqual(
      batch.completionOffsetMs,
    );
    expect(result.result.metrics.batchSubmissionTimelineOmitted).toBe(0);
    expect(result.result.metrics.phaseTimings).toEqual({
      trailingSpeechScanMs: expect.any(Number),
      adaptiveThresholdScanMs: expect.any(Number),
      gapEnergyScanMs: expect.any(Number),
      incrementalStitchMs: expect.any(Number),
      prefixStitchMs: expect.any(Number),
      finalStitchMs: expect.any(Number),
      repairPlanningMs: expect.any(Number),
      repairSplicingMs: expect.any(Number),
      partialResultMs: expect.any(Number),
      tokenDecodeMs: expect.any(Number),
    });
    expect(
      result.result.metrics.phaseTimings.incrementalStitchMs +
        result.result.metrics.phaseTimings.finalStitchMs +
        result.result.metrics.phaseTimings.tokenDecodeMs,
    ).toBe(result.result.metrics.stitchingMs);
    expect(result.result.metrics.totalMs).toBeGreaterThan(0);
    expect(result.result.metrics.totalMs).toBeLessThan(
      convertedAudioDecoding.wallMs,
    );
    expect(result.result.metrics.speedFactor).toBeGreaterThan(0);
  });

  it("reports monotonic sub-batch fractions without advancing completed windows", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const executionMilestones = [
      { completedWorkUnits: 1, totalWorkUnits: 4 },
      { completedWorkUnits: 2, totalWorkUnits: 4 },
      { completedWorkUnits: 3, totalWorkUnits: 4 },
      { completedWorkUnits: 4, totalWorkUnits: 4 },
    ] as const satisfies readonly EngineBatchExecutionProgress[];
    let batchCalls = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 2,
        maxInFlightBatches: 1,
        async transcribeWindows(
          windows,
          _signal,
          onExecutionProgress,
        ) {
          batchCalls += 1;
          expect(onExecutionProgress).toBeTypeOf("function");
          for (const milestone of executionMilestones) {
            onExecutionProgress?.(milestone);
          }
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(batchCalls).toBe(2);
    const progress = messages.flatMap((message) =>
      message.type === "progress" &&
      message.progress.phase === "transcribing"
        ? [message.progress]
        : [],
    );
    const fractions = progress.map(({ fraction }) => fraction);
    const allProgress = expectIntegratedProgress(messages);
    expect(fractions[0]).toBe(0);
    expect(fractions.at(-1)).toBeLessThan(1);
    expect(
      fractions.filter(
        (fraction, index) =>
          index > 0 && fraction > fractions[index - 1]!,
      ).length,
    ).toBeGreaterThanOrEqual(8);
    expect(allProgress).toEqual(progress);
    const completedWindowCounts = progress.map(
      ({ metrics: { completedWindows } }) => completedWindows,
    );
    expect(completedWindowCounts[0]).toBe(0);
    expect(completedWindowCounts.at(-1)).toBe(3);
    expect(completedWindowCounts).toContain(2);
    expect(
      completedWindowCounts.every(
        (completedWindows, index) =>
          Number.isSafeInteger(completedWindows) &&
          completedWindows >= 0 &&
          completedWindows <= 3 &&
          (index === 0 ||
            completedWindows >= completedWindowCounts[index - 1]!),
      ),
    ).toBe(true);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 3, totalWindows: 3 },
      },
    });
  });

  it("reserves full completion for done on a single window without repair", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        maxInFlightBatches: 1,
        async transcribeWindows(
          windows,
          _signal,
          onExecutionProgress,
        ) {
          for (
            let completedWorkUnits = 1;
            completedWorkUnits <= 4;
            completedWorkUnits += 1
          ) {
            onExecutionProgress?.({
              completedWorkUnits,
              totalWorkUnits: 4,
            });
          }
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    const progress = expectIntegratedProgress(messages);
    expect(progress.at(-1)?.fraction).toBeLessThan(1);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: {
          completedWindows: 1,
          totalWindows: 1,
          repairGapProbes: 0,
          repairWindowDecodes: 0,
          repairRounds: 0,
          recoveredTokens: 0,
        },
      },
    });
  });

  it.each([
    [
      "regressing completion",
      [
        { completedWorkUnits: 2, totalWorkUnits: 4 },
        { completedWorkUnits: 1, totalWorkUnits: 4 },
      ],
    ],
    [
      "changing totals",
      [
        { completedWorkUnits: 1, totalWorkUnits: 4 },
        { completedWorkUnits: 2, totalWorkUnits: 5 },
      ],
    ],
  ] as const)(
    "rejects %s in engine batch progress",
    async (_description, executionMilestones) => {
      const messages: ParakeetWorkerMessage[] = [];
      const runtime = new ParakeetWorkerRuntime({
        postMessage: (message) => messages.push(message),
        engineFactory: async () => testEngine({
          diagnostics: { fbankHopSamples: 160 },
          maxInFlightBatches: 1,
          async transcribeWindows(
            windows,
            _signal,
            onExecutionProgress,
          ) {
            for (const milestone of executionMilestones) {
              onExecutionProgress?.(milestone);
            }
            return windows.map(() => ({ tokens: [] }));
          },
        }),
      });
      await runtime.handleMessage({
        type: "initialize",
        requestId: 1,
        reportProgress: true,
        modelSource: "cache-or-network",
        configuration: {
          fp16ModelUrl: "/model",
          fp32ModelUrl: "/model-fp32",
          wasmUrl: "/wasm",
        },
      });
      await runtime.handleMessage({
        type: "transcribe",
        jobId: 1,
        audio: makeWaveBlob(16_000),
        audioDecoding: canonicalAudioDecoding,
        reportProgress: true,
        reportPartialResults: false,
        reportPacedTranscriptBoundary: false,
      });

      expect(messages).toContainEqual(
        expect.objectContaining({
          type: "error",
          jobId: 1,
          error: expect.objectContaining({
            code: "ENGINE_PROGRESS_MISMATCH",
          }),
        }),
      );
      expect(
        messages.some((message) => message.type === "result"),
      ).toBe(false);
    },
  );

  it("rejects concurrent work and keeps cancellation responsive", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let enteredResolve: (() => void) | undefined;
    const entered = new Promise<void>((resolve) => {
      enteredResolve = resolve;
    });
    const engine = testEngine({
      diagnostics: { fbankHopSamples: 160 },
      preferredBatchSize: 2,
      transcribeWindows(_windows, signal) {
        enteredResolve?.();
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("cancelled", "AbortError")),
            { once: true },
          );
        });
      },
      decodeTokenIds: () => "",
    });
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => engine,
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const first = runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    await entered;
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 2,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    expect(messages).toContainEqual(
      expect.objectContaining({
        type: "error",
        jobId: 2,
        error: expect.objectContaining({ code: "BUSY" }),
      }),
    );

    await runtime.handleMessage({ type: "cancel", jobId: 1 });
    await first;
    expect(messages).toContainEqual({ type: "cancelled", jobId: 1 });
  });

  it("never reads more than the engine's preferred batch of audio windows", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchSizes: number[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 2,
        async transcribeWindows(windows) {
          batchSizes.push(windows.length);
          expect(windows.every((window) => window.samples.length === 240_000)).toBe(
            true,
          );
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(batchSizes).toEqual([2, 1]);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 3, totalWindows: 3 },
      },
    });
  });

  it("bounds the retained submission timeline for arbitrarily long inputs", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let inferenceCalls = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 1,
        maxInFlightBatches: 1,
        async transcribeWindows(windows) {
          inferenceCalls += 1;
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(160 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(inferenceCalls).toBeGreaterThan(11);
    const result = messages.find((message) => message.type === "result");
    if (result?.type !== "result") throw new Error("missing result");
    expect(result.result.metrics.batchSubmissionTimeline).toHaveLength(11);
    expect(
      result.result.metrics.batchSubmissionTimeline.map(
        ({ submissionIndex }) => submissionIndex,
      ),
    ).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(result.result.metrics.batchSubmissionTimelineOmitted).toBe(
      inferenceCalls - 11,
    );
  });

  it("runs gap-start then centered seam repair and inserts only recovered tokens", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchKinds: string[] = [];
    const milestoneFractions: number[][] = [];
    let inferenceCall = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 40,
        async transcribeWindows(
          windows,
          _signal,
          onExecutionProgress,
        ) {
          inferenceCall += 1;
          const batchMilestones: number[] = [];
          for (
            let completedWorkUnits = 1;
            completedWorkUnits <= 4;
            completedWorkUnits += 1
          ) {
            onExecutionProgress?.({
              completedWorkUnits,
              totalWorkUnits: 4,
            });
            batchMilestones.push(latestTranscribingFraction(messages));
          }
          milestoneFractions.push(batchMilestones);
          if (inferenceCall === 1) {
            batchKinds.push(`primary:${windows.length}`);
            return windows.map((_window, index) => ({
              tokens:
                index === 0
                  ? [
                      frameToken(1, 10),
                      // Sixteen token-empty, speech-filled encoder frames are
                      // 1.28 seconds: below the former repair threshold.
                      frameToken(2, 27),
                    ]
                  : [],
            }));
          }
          if (inferenceCall === 2) {
            batchKinds.push(`gap-start:${windows.length}`);
            return windows.map(() => ({ tokens: [] }));
          }
          batchKinds.push(`centered:${windows.length}`);
          return windows.map(() => ({
            tokens: [frameToken(3, 20)],
          }));
        },
        tokenPiece(tokenId) {
          return {
            1: "▁left",
            2: "▁right",
            3: "▁recovered",
          }[tokenId];
        },
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000, 1_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: true,
      reportPacedTranscriptBoundary: false,
    });

    expect(batchKinds).toEqual([
      "primary:3",
      "gap-start:1",
      "centered:1",
    ]);
    const result = messages.find((message) => message.type === "result");
    expect(result).toMatchObject({
      type: "result",
      result: {
        text: "1 3 2",
        metrics: {
          completedWindows: 3,
          totalWindows: 3,
          repairGapProbes: 1,
          repairWindowDecodes: 2,
          repairRounds: 1,
          recoveredTokens: 1,
        },
      },
    });
    expectIntegratedProgress(messages);
    expect(milestoneFractions).toHaveLength(3);
    for (const batchMilestones of milestoneFractions) {
      expect(batchMilestones).toHaveLength(4);
      expect(
        batchMilestones.every(
          (fraction, index) =>
            index === 0 || fraction >= batchMilestones[index - 1]!,
        ),
      ).toBe(true);
      expect(new Set(batchMilestones).size).toBeGreaterThan(1);
    }
    expect(
      milestoneFractions.flat().every(
        (fraction, index, all) =>
          index === 0 || fraction >= all[index - 1]!,
      ),
    ).toBe(true);
    const partialResults = messages.filter(
      (message) => message.type === "partial-result",
    );
    expect(partialResults.map((message) => message.snapshot.revision)).toEqual([
      1, 2,
    ]);
    expect(
      partialResults.at(-1)?.snapshot.tokens.map((token) => token.tokenId),
    ).toEqual([1, 3, 2]);
  });

  it("prefetches an exact next-round gap-start decode in centered spare rows", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batches: Array<{
      readonly kind: string;
      readonly sourceStartFrames: readonly number[];
    }> = [];
    let inferenceCall = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 40,
        async transcribeWindows(windows) {
          inferenceCall += 1;
          const sourceStartFrames = windows.map(
            (window) => window.sourceStartSample / 1_280,
          );
          if (inferenceCall === 1) {
            batches.push({ kind: "primary", sourceStartFrames });
            return windows.map((_window, index) => ({
              tokens:
                index === 0
                  ? [
                      frameToken(1, 100),
                      frameToken(2, 149),
                      frameToken(3, 184),
                    ]
                  : [],
            }));
          }
          if (inferenceCall === 2) {
            batches.push({ kind: "gap-start", sourceStartFrames });
            expect(sourceStartFrames).toEqual([101, 150]);
            return windows.map((window) => ({
              tokens:
                window.sourceStartSample / 1_280 === 101
                  ? [frameToken(4, 125)]
                  : [],
            }));
          }
          if (inferenceCall === 3) {
            batches.push({
              kind: "centered+prefetch",
              sourceStartFrames,
            });
            expect(sourceStartFrames).toEqual([73, 126]);
            return windows.map((window) => ({
              tokens:
                window.sourceStartSample / 1_280 === 73
                  ? [frameToken(5, 170, 4)]
                  : [frameToken(6, 137, 2)],
            }));
          }
          throw new Error("unexpected fourth inference batch");
        },
        tokenPiece: (tokenId) => `▁token-${tokenId}`,
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000, 1_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(batches).toEqual([
      {
        kind: "primary",
        sourceStartFrames: [0, 160, 239],
      },
      {
        kind: "gap-start",
        sourceStartFrames: [101, 150],
      },
      {
        kind: "centered+prefetch",
        sourceStartFrames: [73, 126],
      },
    ]);
    expect(inferenceCall).toBe(3);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        text: "1 4 6 2 5 3",
        tokens: [
          { tokenId: 1 },
          { tokenId: 4 },
          { tokenId: 6 },
          { tokenId: 2 },
          { tokenId: 5 },
          { tokenId: 3 },
        ],
        metrics: {
          completedWindows: 3,
          totalWindows: 3,
          repairGapProbes: 3,
          repairWindowDecodes: 4,
          repairRounds: 2,
          recoveredTokens: 3,
          repairPrefetchWindowDecodes: 1,
          repairPrefetchCacheHits: 1,
          repairPrefetchUnusedDecodes: 0,
        },
      },
    });
    const result = messages.find((message) => message.type === "result");
    if (result?.type !== "result") throw new Error("missing result");
    expect(
      result.result.metrics.batchSubmissionTimeline.map((batch) => ({
        batchClass: batch.batchClass,
        activeRows: batch.activeRows,
        primaryRows: batch.primaryRows,
        officialRepairRows: batch.officialRepairRows,
        finalPrimaryPrefetchRows: batch.finalPrimaryPrefetchRows,
        crossRoundPrefetchRows: batch.crossRoundPrefetchRows,
      })),
    ).toEqual([
      {
        batchClass: "primary",
        activeRows: 3,
        primaryRows: 3,
        officialRepairRows: 0,
        finalPrimaryPrefetchRows: 0,
        crossRoundPrefetchRows: 0,
      },
      {
        batchClass: "repair",
        activeRows: 2,
        primaryRows: 0,
        officialRepairRows: 2,
        finalPrimaryPrefetchRows: 0,
        crossRoundPrefetchRows: 0,
      },
      {
        batchClass: "repair",
        activeRows: 2,
        primaryRows: 0,
        officialRepairRows: 1,
        finalPrimaryPrefetchRows: 0,
        crossRoundPrefetchRows: 1,
      },
    ]);
    expectIntegratedProgress(messages);
  });

  it("does not launch prefetch when centered repair is already cached", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let inferenceCall = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 40,
        async transcribeWindows(windows) {
          inferenceCall += 1;
          const sourceStartFrames = windows.map(
            (window) => window.sourceStartSample / 1_280,
          );
          if (inferenceCall === 1) {
            return windows.map((_window, index) => ({
              tokens:
                index === 0
                  ? [frameToken(1, 100), frameToken(2, 149)]
                  : index === 1
                    ? [frameToken(3, 240)]
                    : [],
            }));
          }
          if (inferenceCall === 2) {
            expect(sourceStartFrames).toEqual([101, 150]);
            return windows.map((window) => ({
              tokens:
                window.sourceStartSample / 1_280 === 101
                  ? [frameToken(4, 125)]
                  : [],
            }));
          }
          if (inferenceCall === 3) {
            expect(sourceStartFrames).toEqual([126]);
            const progressSoFar = messages.flatMap((message) =>
              message.type === "progress" ? [message.progress] : [],
            );
            expect(
              progressSoFar
                .slice(1)
                .every(({ phase }) => phase === "transcribing"),
            ).toBe(true);
            expect(progressSoFar.at(-1)?.fraction).toBeLessThan(1);
            return [{ tokens: [frameToken(5, 137, 2)] }];
          }
          throw new Error("unexpected inference batch");
        },
        tokenPiece: (tokenId) => `▁token-${tokenId}`,
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000, 1_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(inferenceCall).toBe(3);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        text: "1 4 5 2 3",
        metrics: {
          repairGapProbes: 3,
          repairWindowDecodes: 3,
          repairRounds: 2,
          recoveredTokens: 2,
          repairPrefetchWindowDecodes: 0,
          repairPrefetchCacheHits: 0,
          repairPrefetchUnusedDecodes: 0,
        },
      },
    });
    expectIntegratedProgress(messages);
  });

  it("fills a partial final primary batch with exact first-round repair", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchIndices: number[][] = [];
    let inferenceCall = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 4,
        async transcribeWindows(windows) {
          inferenceCall += 1;
          batchIndices.push(windows.map(({ index }) => index));
          if (inferenceCall === 1) {
            return windows.map((_window, index) => ({
              tokens:
                index === 0
                  ? [frameToken(1, 100), frameToken(2, 130)]
                  : [],
            }));
          }
          if (inferenceCall === 2) {
            return windows.map(() => ({ tokens: [] }));
          }
          if (inferenceCall === 3) {
            expect(windows).toHaveLength(2);
            return windows.map((window) => ({
              tokens:
                window.index === 8
                  ? []
                  : [frameToken(3, 117, 2)],
            }));
          }
          throw new Error("repair should hit the final-batch cache");
        },
        tokenPiece: (tokenId) => `▁token-${tokenId}`,
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(110 * 16_000, 1_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(batchIndices).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9],
    ]);
    expect(inferenceCall).toBe(3);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        text: "1 3 2",
        metrics: {
          completedWindows: 9,
          totalWindows: 9,
          repairGapProbes: 1,
          repairWindowDecodes: 1,
          repairRounds: 1,
          recoveredTokens: 1,
          repairPrefetchWindowDecodes: 1,
          repairPrefetchCacheHits: 1,
          repairPrefetchUnusedDecodes: 0,
        },
      },
    });
    const result = messages.find((message) => message.type === "result");
    if (result?.type !== "result") throw new Error("missing result");
    expect(result.result.metrics.batchSubmissionTimeline.at(-1)).toMatchObject({
      submissionIndex: 2,
      batchClass: "mixed",
      activeRows: 2,
      primaryRows: 1,
      repairRows: 1,
      officialRepairRows: 0,
      finalPrimaryPrefetchRows: 1,
      crossRoundPrefetchRows: 0,
    });
    expectIntegratedProgress(messages);
  });

  it("never inserts a final-batch prefetch rejected by the official merge", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchIndices: number[][] = [];
    let inferenceCall = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 4,
        async transcribeWindows(windows) {
          inferenceCall += 1;
          batchIndices.push(windows.map(({ index }) => index));
          if (inferenceCall === 1) {
            return windows.map(() => ({ tokens: [] }));
          }
          if (inferenceCall === 2) {
            return windows.map((_window, index) => ({
              tokens:
                index === 3
                  ? [frameToken(1, 1_250), frameToken(2, 1_300)]
                  : [],
            }));
          }
          if (inferenceCall === 3) {
            return windows.map((window) => ({
              tokens:
                window.index === 8
                  ? [frameToken(3, 1_290)]
                  : [frameToken(4, 1_260)],
            }));
          }
          throw new Error("false prefetch must not trigger more work");
        },
        tokenPiece: (tokenId) => `▁token-${tokenId}`,
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(110 * 16_000, 1_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(batchIndices).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9],
    ]);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        text: "1",
        metrics: {
          completedWindows: 9,
          totalWindows: 9,
          repairGapProbes: 0,
          repairWindowDecodes: 1,
          repairRounds: 0,
          recoveredTokens: 0,
          repairMs: 0,
          repairPrefetchWindowDecodes: 1,
          repairPrefetchCacheHits: 0,
          repairPrefetchUnusedDecodes: 1,
        },
      },
    });
    expectIntegratedProgress(messages);
  });

  it("pipelines a declared second in-flight engine batch", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let calls = 0;
    let resolveFirst:
      | ((results: readonly { readonly tokens: readonly [] }[]) => void)
      | undefined;
    const firstResult = new Promise<
      readonly { readonly tokens: readonly [] }[]
    >((resolve) => {
      resolveFirst = resolve;
    });
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 2,
        maxInFlightBatches: 2,
        transcribeWindows(windows) {
          calls += 1;
          if (calls === 1) return firstResult;
          expect(calls).toBe(2);
          resolveFirst?.([
            { tokens: [] },
            { tokens: [] },
          ]);
          return Promise.resolve(windows.map(() => ({ tokens: [] })));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(calls).toBe(2);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 3, totalWindows: 3 },
      },
    });
  });

  it("releases a decoder-failed prepared stream before accepting the next job", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({}),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const failedChannel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 1,
      port: failedChannel.port1,
      inputByteLength: 10,
      estimatedSampleCount: null,
    });
    failedChannel.port2.postMessage({
      type: "pcm-failed",
      message: "decoder failed before transcription",
    });
    await waitUntil(
      () =>
        messages.filter(
          (message) => message.type === "error" && message.jobId === 1,
        ).length === 1,
    );
    await runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 1,
      reportProgress: false,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    failedChannel.port2.close();

    const nextChannel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 2,
      port: nextChannel.port1,
      inputByteLength: 20,
      estimatedSampleCount: null,
    });
    const totalSamples = 16_000;
    nextChannel.port2.postMessage(
      pcmSegment(0, 0, totalSamples, 1_000),
    );
    nextChannel.port2.postMessage(pcmEnd(totalSamples));
    await runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 2,
      reportProgress: false,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    nextChannel.port2.close();

    expect(
      messages.filter(
        (message) => message.type === "error" && message.jobId === 1,
      ),
    ).toHaveLength(1);
    expect(
      messages.some(
        (message) => message.type === "error" && message.jobId === 2,
      ),
    ).toBe(false);
    expect(messages).toContainEqual(
      expect.objectContaining({ type: "result", jobId: 2 }),
    );
  });

  it("preserves two in-flight B40 batches across streamed PCM feed messages and executes the EOF tail once", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchIndices: number[][] = [];
    let resolveFirst:
      | ((results: readonly { readonly tokens: readonly [] }[]) => void)
      | undefined;
    const firstResult = new Promise<
      readonly { readonly tokens: readonly [] }[]
    >((resolve) => {
      resolveFirst = resolve;
    });
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () =>
        testEngine({
          diagnostics: { fbankHopSamples: 160 },
          preferredBatchSize: 40,
          maxInFlightBatches: 2,
          transcribeWindows(windows) {
            batchIndices.push(windows.map(({ index }) => index));
            if (batchIndices.length === 1) return firstResult;
            return Promise.resolve(
              windows.map(() => ({ tokens: [] })),
            );
          },
        }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const channel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 1,
      port: channel.port1,
      inputByteLength: 123,
      estimatedSampleCount: null,
    });
    const transcription = runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 1,
      reportProgress: true,
      reportPartialResults: true,
      reportPacedTranscriptBoundary: false,
    });

    const firstFortySamples =
      39 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES +
      1;
    const totalSamples =
      80 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES;
    channel.port2.postMessage(
      pcmSegment(0, 0, firstFortySamples, 1_000),
    );
    await waitUntil(() => batchIndices.length === 1);
    channel.port2.postMessage(
      pcmSegment(
        1,
        firstFortySamples,
        totalSamples - firstFortySamples,
        1_000,
      ),
    );
    await waitUntil(() => batchIndices.length === 2);

    // The second B40 reached the engine before the unresolved first B40 was
    // consumed, proving maxInFlight=2 survived the separate PCM feed turns.
    expect(batchIndices[0]).toEqual(sequence(0, 40));
    expect(batchIndices[1]).toEqual(sequence(40, 40));
    channel.port2.postMessage(pcmEnd(totalSamples));
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    resolveFirst?.(Array.from({ length: 40 }, () => ({ tokens: [] })));
    await transcription;
    channel.port2.close();

    expect(batchIndices).toEqual([
      sequence(0, 40),
      sequence(40, 40),
      [80],
    ]);
    expect(batchIndices.flat()).toEqual(sequence(0, 81));
    const partials = messages.flatMap((message) =>
      message.type === "partial-result" ? [message.snapshot] : [],
    );
    expect(partials.length).toBeGreaterThan(0);
    expect(
      partials.every(
        ({ audioDurationSeconds }) =>
          audioDurationSeconds === totalSamples / 16_000,
      ),
    ).toBe(true);
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 81, totalWindows: 81 },
      },
    });
  }, 30_000);

  it("executes an exact-multiple EOF primary batch once without replaying its stable prefix", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const batchIndices: number[][] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () =>
        testEngine({
          preferredBatchSize: 2,
          maxInFlightBatches: 2,
          async transcribeWindows(windows) {
            batchIndices.push(windows.map(({ index }) => index));
            return windows.map(() => ({ tokens: [] }));
          },
        }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const channel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 1,
      port: channel.port1,
      inputByteLength: 1,
      estimatedSampleCount: 100 * PARAKEET_WINDOW_STRIDE_SAMPLES,
    });
    const totalSamples =
      3 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES;
    channel.port2.postMessage(
      pcmSegment(0, 0, totalSamples, 1_000),
    );
    channel.port2.postMessage(pcmEnd(totalSamples));
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 1,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    channel.port2.close();

    expect(batchIndices).toEqual([
      [0, 1],
      [2, 3],
    ]);
    expect(batchIndices.flat()).toEqual(sequence(0, 4));
    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 4, totalWindows: 4 },
      },
    });
    expectIntegratedProgress(messages);
  });

  it("publishes streamed B40 progress before EOF while submitting the final graph under it", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let resolveGraph:
      | ((results: readonly { readonly tokens: readonly [] }[]) => void)
      | undefined;
    const graphResult = new Promise<
      readonly { readonly tokens: readonly [] }[]
    >((resolve) => {
      resolveGraph = resolve;
    });
    const batchIndices: number[][] = [];
    let firstGraphProgress:
      | ((progress: EngineBatchExecutionProgress) => void)
      | undefined;
    let tick = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () =>
        testEngine({
          preferredBatchSize: 40,
          maxInFlightBatches: 2,
          transcribeWindows(windows, _signal, onExecutionProgress) {
            batchIndices.push(windows.map(({ index }) => index));
            if (batchIndices.length === 1) {
              firstGraphProgress = onExecutionProgress;
              onExecutionProgress?.({
                completedWorkUnits: 1,
                totalWorkUnits: 4,
              });
              return graphResult;
            }
            return Promise.resolve(windows.map(() => ({ tokens: [] })));
          },
        }),
      now: () => {
        tick += 1;
        return tick;
      },
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const channel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 1,
      port: channel.port1,
      inputByteLength: 1,
      estimatedSampleCount: 10 * PARAKEET_WINDOW_STRIDE_SAMPLES,
    });
    const firstFortySamples =
      39 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES +
      1;
    channel.port2.postMessage(
      pcmSegment(0, 0, firstFortySamples, 1_000),
    );
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const transcription = runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 1,
      reportProgress: true,
      reportPartialResults: true,
      reportPacedTranscriptBoundary: false,
    });
    await waitUntil(() => batchIndices.length === 1);
    const preEofProgress = messages
      .flatMap((message) =>
        message.type === "progress" &&
        message.progress.phase === "transcribing"
          ? [message]
          : [],
      )
      .at(-1);
    expect(preEofProgress).toMatchObject({
      type: "progress",
      progress: {
        fraction: expect.any(Number),
        metrics: { completedWindows: 0, totalWindows: 0 },
      },
    });
    if (preEofProgress?.type !== "progress") {
      throw new Error("missing pre-EOF progress");
    }
    expect(preEofProgress.progress.fraction).toBeGreaterThan(0);
    expect(
      messages.some((message) => message.type === "partial-result"),
    ).toBe(false);

    const totalSamples =
      58 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES;
    channel.port2.postMessage(
      pcmSegment(
        1,
        firstFortySamples,
        totalSamples - firstFortySamples,
        1_000,
      ),
    );
    channel.port2.postMessage(pcmEnd(totalSamples));
    await waitUntil(() => batchIndices.length === 2);
    expect(batchIndices).toEqual([
      sequence(0, 40),
      sequence(40, 19),
    ]);
    expect(
      messages.some(
        (message) =>
          message.type === "progress" &&
          message.progress.phase === "transcribing" &&
          message.progress.metrics.completedWindows > 0,
      ),
    ).toBe(false);
    const attachedFraction = latestTranscribingFraction(messages);
    expect(attachedFraction).toBeGreaterThan(0);
    firstGraphProgress?.({
      completedWorkUnits: 2,
      totalWorkUnits: 4,
    });
    expect(latestTranscribingFraction(messages)).toBeGreaterThan(
      attachedFraction,
    );
    resolveGraph?.(Array.from({ length: 40 }, () => ({ tokens: [] })));
    await transcription;
    channel.port2.close();

    const result = messages.find((message) => message.type === "result");
    expect(result?.type).toBe("result");
    if (result?.type !== "result") throw new Error("missing result");
    expect(batchIndices.flat()).toEqual(sequence(0, 59));
    expect(result.result.metrics).toMatchObject({
      completedWindows: 59,
      totalWindows: 59,
    });
    expect(result.result.metrics.audioDecoding.pcmWaitMs).toBeGreaterThan(0);
    expect(result.result.metrics.audioDecoding).toMatchObject({
      pcmStarvationMs: 0,
      pcmStarvationCount: 0,
    });
    const partials = messages.flatMap((message) =>
      message.type === "partial-result" ? [message.snapshot] : [],
    );
    expect(
      partials.every(
        ({ audioDurationSeconds }) =>
          audioDurationSeconds === totalSamples / 16_000,
      ),
    ).toBe(true);
  }, 30_000);

  it("publishes completed decode metrics at EOF and excludes pre-transcription decode wall time from totalMs", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let tick = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({}),
      now: () => {
        tick += 1;
        return tick;
      },
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    const channel = new MessageChannel();
    await runtime.handleMessage({
      type: "prepare-stream",
      jobId: 1,
      port: channel.port1,
      inputByteLength: 77,
      estimatedSampleCount: null,
    });
    const transcription = runtime.handleMessage({
      type: "transcribe-stream",
      jobId: 1,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const totalSamples = 16_000;
    const decoderEndedAtMs = performance.timeOrigin + performance.now();
    channel.port2.postMessage(
      pcmSegment(0, 0, totalSamples, 1_000),
    );
    channel.port2.postMessage(
      pcmEnd(totalSamples, {
        decoderWallMs: 5_000,
        decoderActiveMs: 4_000,
        decoderStartedAtMs: decoderEndedAtMs - 5_000,
        decoderEndedAtMs,
      }),
    );
    await transcription;
    channel.port2.close();

    const result = messages.find((message) => message.type === "result");
    expect(result?.type).toBe("result");
    if (result?.type !== "result") throw new Error("missing result");
    const decoding = result.result.metrics.audioDecoding;
    expect(decoding).toMatchObject({
      applied: true,
      wallMs: 5_000,
      activeMs: 4_000,
      inputByteLength: 77,
      outputByteLength: totalSamples * 2,
      pcmStarvationCount: 1,
    });
    expect(decoding.pcmWaitMs).toBeGreaterThan(0);
    expect(decoding.pcmStarvationMs).toBeGreaterThan(0);
    expect(decoding.pcmStarvationMs).toBeLessThanOrEqual(
      decoding.pcmWaitMs,
    );
    expect(decoding.overlapMs).toBeGreaterThanOrEqual(0);
    expect(decoding.overlapMs).toBeLessThanOrEqual(decoding.wallMs);
    expect(result.result.metrics.totalMs).toBeGreaterThanOrEqual(
      decoding.pcmWaitMs,
    );
    expect(result.result.metrics.totalMs).toBeLessThan(
      decoding.wallMs,
    );

    const eofRefresh = messages.find(
      (message) =>
        message.type === "progress" &&
        message.progress.phase === "transcribing" &&
        message.progress.metrics.audioDecoding.applied &&
        message.progress.metrics.audioDecoding.wallMs === 5_000,
    );
    expect(eofRefresh).toBeDefined();
  });

  it("decodes and commits each PCM window through one reusable engine item", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let prepared: Float32Array<ArrayBuffer> | undefined;
    const committed: number[] = [];
    let nextRmsSample = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 16,
        prepareWindowBatch(batchSize, sampleCount) {
          expect(batchSize).toBe(3);
          expect(sampleCount).toBe(240_000);
          prepared = new Float32Array(sampleCount);
          prepared.fill(Number.NaN);
          return {
            samples: prepared,
            commit(window, batchIndex, rmsFrameRange) {
              expect(prepared?.every((sample) => sample === 0)).toBe(true);
              committed.push(batchIndex);
              if (rmsFrameRange !== undefined) {
                expect(
                  window.sourceStartSample +
                    rmsFrameRange.sampleOffset,
                ).toBe(nextRmsSample);
                nextRmsSample +=
                  rmsFrameRange.frameCount * 1_280;
              }
              prepared?.fill(Number.NaN);
              return rmsFrameRange === undefined
                ? undefined
                : new Uint32Array(rmsFrameRange.frameCount);
            },
            finish() {
              expect(committed).toEqual([0, 1, 2]);
            },
            abort() {
              throw new Error("writer should not abort");
            },
          };
        },
        async transcribeWindows(windows) {
          expect(windows).toHaveLength(3);
          expect(
            windows.every(
              (window) =>
                window.samples.buffer === prepared?.buffer &&
                window.samples.byteOffset === prepared?.byteOffset &&
                window.samples.length === 240_000,
            ),
          ).toBe(true);
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 3, totalWindows: 3 },
      },
    });
    expect(nextRmsSample).toBe(
      Math.floor((34 * 16_000) / 1_280) * 1_280,
    );
  });

  it("supports advancing transferable input views and an asynchronous finish", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const prepared = Array.from(
      { length: 3 },
      () => new Float32Array(240_000).fill(Number.NaN),
    );
    let nextIndex = 0;
    let finished = false;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 16,
        prepareWindowBatch(batchSize, sampleCount) {
          expect(batchSize).toBe(3);
          expect(sampleCount).toBe(240_000);
          return {
            get samples() {
              return prepared[nextIndex]!;
            },
            commit(_window, batchIndex) {
              expect(batchIndex).toBe(nextIndex);
              expect(
                prepared[nextIndex]!.every((sample) => sample === 0),
              ).toBe(true);
              const buffer = prepared[nextIndex]!.buffer;
              structuredClone(buffer, { transfer: [buffer] });
              nextIndex += 1;
            },
            async finish() {
              await Promise.resolve();
              expect(nextIndex).toBe(3);
              finished = true;
            },
            abort() {
              throw new Error("writer should not abort");
            },
          };
        },
        async transcribeWindows(windows) {
          expect(finished).toBe(true);
          expect(windows).toHaveLength(3);
          expect(windows.every((window) => window.samples.length === 0)).toBe(
            true,
          );
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(34 * 16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(
      messages.find((message) => message.type === "result"),
    ).toMatchObject({
      type: "result",
      result: {
        metrics: { completedWindows: 3, totalWindows: 3 },
      },
    });
  });

  it("awaits asynchronous writer cleanup after a commit failure", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let aborted = false;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        prepareWindowBatch(_batchSize, sampleCount) {
          const samples = new Float32Array(sampleCount);
          return {
            samples,
            commit() {
              throw new Error("synthetic commit failure");
            },
            finish() {
              throw new Error("writer should not finish");
            },
            async abort() {
              await Promise.resolve();
              aborted = true;
            },
          };
        },
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 7,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(aborted).toBe(true);
    expect(messages).toContainEqual(
      expect.objectContaining({
        type: "error",
        jobId: 7,
        error: expect.objectContaining({
          message: "synthetic commit failure",
        }),
      }),
    );
    expect(messages.some((message) => message.type === "result")).toBe(false);
  });

  it("resets engine streaming state before every input file", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    let resets = 0;
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 2,
        beginTranscription() {
          resets += 1;
        },
        async transcribeWindows(windows) {
          return windows.map(() => ({ tokens: [] }));
        },
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 2,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(resets).toBe(2);
    expect(
      messages.filter((message) => message.type === "result"),
    ).toHaveLength(2);
  });

  it("rejects a batch result cardinality mismatch", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 2,
        transcribeWindows: async () => [],
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 5,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: true,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(messages).toContainEqual(
      expect.objectContaining({
        type: "error",
        jobId: 5,
        error: expect.objectContaining({
          code: "ENGINE_BATCH_RESULT_MISMATCH",
        }),
      }),
    );
    expect(messages.some((message) => message.type === "result")).toBe(false);
  });

  it("skips optional load, progress, and partial traffic without subscribers", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async (_configuration, context) => {
        expect(context.onProgress).toBeUndefined();
        return testEngine({
          diagnostics: { fbankHopSamples: 160 },
          async transcribeWindows(
            windows,
            _signal,
            onExecutionProgress,
          ) {
            expect(onExecutionProgress).toBeUndefined();
            return windows.map(() => ({
              tokens: [frameToken(4, 1)],
            }));
          },
          tokenPiece: () => "▁hello",
          decodeTokenIds: () => "hello",
        });
      },
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 1,
      reportProgress: false,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });
    await runtime.handleMessage({
      type: "transcribe",
      jobId: 1,
      audio: makeWaveBlob(16_000),
      audioDecoding: canonicalAudioDecoding,
      reportProgress: false,
      reportPartialResults: false,
      reportPacedTranscriptBoundary: false,
    });

    expect(messages.map((message) => message.type)).toEqual([
      "ready",
      "result",
    ]);
    expect(messages[1]).toMatchObject({
      type: "result",
      result: {
        text: "hello",
        tokens: [
          { tokenId: 4, startSeconds: 0.08, endSeconds: 0.16 },
        ],
        words: [
          { text: "hello", startSeconds: 0.08, endSeconds: 0.16 },
        ],
        metrics: {
          phaseTimings: { partialResultMs: 0 },
        },
      },
    });
  });

  it("rejects a non-positive preferred engine batch size during loading", async () => {
    const messages: ParakeetWorkerMessage[] = [];
    const runtime = new ParakeetWorkerRuntime({
      postMessage: (message) => messages.push(message),
      engineFactory: async () => testEngine({
        diagnostics: { fbankHopSamples: 160 },
        preferredBatchSize: 0,
        transcribeWindows: async () => [],
        decodeTokenIds: () => "",
      }),
    });
    await runtime.handleMessage({
      type: "initialize",
      requestId: 8,
      reportProgress: true,
      modelSource: "cache-or-network",
      configuration: {
        fp16ModelUrl: "/model",
        fp32ModelUrl: "/model-fp32",
        wasmUrl: "/wasm",
      },
    });

    expect(messages).toContainEqual(
      expect.objectContaining({
        type: "error",
        requestId: 8,
        error: expect.objectContaining({
          code: "INVALID_ENGINE_BATCH_SIZE",
        }),
      }),
    );
  });

});

function expectIntegratedProgress(
  messages: readonly ParakeetWorkerMessage[],
): TranscriptionProgress[] {
  const indexedProgress = messages.flatMap((message, messageIndex) =>
    message.type === "progress"
      ? [{ messageIndex, progress: message.progress }]
      : [],
  );
  expect(indexedProgress.length).toBeGreaterThanOrEqual(3);
  expect(indexedProgress[0]?.progress).toMatchObject({
    phase: "transcribing",
    fraction: 0,
  });

  const done = indexedProgress.filter(
    ({ progress }) => progress.phase === "done",
  );
  expect(done).toHaveLength(1);
  expect(done[0]?.progress.fraction).toBe(1);
  expect(indexedProgress.at(-1)).toBe(done[0]);

  const beforeDone = indexedProgress.slice(0, -1);
  expect(
    beforeDone.every(
      ({ progress }) => progress.phase === "transcribing",
    ),
  ).toBe(true);
  expect(
    beforeDone.every(({ progress }) => progress.fraction < 1),
  ).toBe(true);

  const fractions = indexedProgress.map(
    ({ progress }) => progress.fraction,
  );
  expect(
    fractions.every(
      (fraction, index) =>
        Number.isFinite(fraction) &&
        fraction >= 0 &&
        fraction <= 1 &&
        (index === 0 || fraction >= fractions[index - 1]!),
    ),
  ).toBe(true);
  expect(fractions.filter((fraction) => fraction === 1)).toHaveLength(1);

  const resultIndex = messages.findIndex(
    (message) => message.type === "result",
  );
  expect(resultIndex).toBeGreaterThan(done[0]!.messageIndex);

  return beforeDone.map(({ progress }) => progress);
}

function latestTranscribingFraction(
  messages: readonly ParakeetWorkerMessage[],
): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!;
    if (
      message.type === "progress" &&
      message.progress.phase === "transcribing"
    ) {
      return message.progress.fraction;
    }
  }
  throw new Error("Missing transcribing progress");
}

function testEngine(
  overrides: Partial<ParakeetInferenceEngine>,
): ParakeetInferenceEngine {
  return {
    diagnostics: { fbankHopSamples: 160 },
    preferredBatchSize: 2,
    maxInFlightBatches: 2,
    beginTranscription() {},
    prepareWindowBatch(_batchSize, sampleCount) {
      const samples = new Float32Array(sampleCount);
      return {
        samples,
        commit() {},
        finish() {},
        abort() {},
      };
    },
    async transcribeWindows(windows) {
      return windows.map(() => ({ tokens: [] }));
    },
    decodeTokenIds: () => "",
    tokenPiece: () => undefined,
    dispose() {},
    ...overrides,
  };
}

function frameToken(
  tokenId: number,
  frame: number,
  durationFrames = 1,
) {
  return {
    tokenId,
    startSample: frame * 1_280,
    endSample: (frame + durationFrames) * 1_280,
    durationFrames,
  };
}

function makeWaveBlob(sampleCount: number, sampleValue = 0): Blob {
  const bytes = new Uint8Array(44 + sampleCount * 2);
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
  view.setUint32(40, sampleCount * 2, true);
  for (let sample = 0; sample < sampleCount; sample += 1) {
    view.setInt16(44 + sample * 2, sampleValue, true);
  }
  return new Blob([bytes.buffer]);
}

function pcmSegment(
  segmentIndex: number,
  startSample: number,
  sampleCount: number,
  sampleValue = 0,
): Record<string, unknown> {
  const samples = new Int16Array(sampleCount);
  if (sampleValue !== 0) samples.fill(sampleValue);
  return {
    type: "pcm-segment",
    segmentIndex,
    startSample,
    sampleCount,
    file: new File([samples], `segment-${segmentIndex}.s16le`),
  };
}

function pcmEnd(
  totalSamples: number,
  timings: {
    readonly decoderWallMs: number;
    readonly decoderActiveMs: number;
    readonly decoderStartedAtMs: number;
    readonly decoderEndedAtMs: number;
  } = {
    decoderWallMs: 10,
    decoderActiveMs: 7,
    decoderStartedAtMs: 100,
    decoderEndedAtMs: 110,
  },
): Record<string, unknown> {
  return {
    type: "pcm-end",
    totalSamples,
    outputByteLength: totalSamples * 2,
    ...timings,
  };
}

function sequence(start: number, length: number): number[] {
  return Array.from({ length }, (_, index) => start + index);
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("Timed out waiting for asynchronous test condition");
}

function writeFourCc(target: Uint8Array, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    target[offset + index] = value.charCodeAt(index);
  }
}
