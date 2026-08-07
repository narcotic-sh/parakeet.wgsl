import { Pcm16WavReader } from "../audio/wav-stream";
import { SegmentedPcm16Reader } from "../audio/segmented-pcm-reader";
import {
  PARAKEET_ENCODER_FRAME_SAMPLES,
  PARAKEET_FBANK_HOP_SAMPLES,
  PARAKEET_OVERLAP_SAMPLES,
  PARAKEET_OVERLAP_SECONDS,
  PARAKEET_PRIMARY_CONTEXT_SAMPLES,
  PARAKEET_PRIMARY_VISIBLE_SAMPLES,
  PARAKEET_REPAIR_VISIBLE_SAMPLES,
  PARAKEET_WINDOW_SAMPLES,
  PARAKEET_WINDOW_SECONDS,
  PARAKEET_WINDOW_STRIDE_SAMPLES,
  OnlineStatelessWindowPlanner,
  planStatelessWindows,
  type StatelessAudioWindow,
  type StatelessWindowPlan,
} from "./chunking";
import {
  SEAM_GAP_MAX_PROBES,
  SEAM_GAP_MAX_ROUNDS,
  SEAM_GAP_MIN_FRAMES,
  planSeamRepairWindow,
  seamGapAt,
  seamRepairPlanCacheKey,
  spliceRepairCandidate,
  type SeamGap,
  type SeamRepairPlacement,
} from "./seam-gap-repair";
import {
  AdaptiveSpeechRmsAccumulator,
  SEAM_GAP_MIN_SPEECH_SECONDS,
  completeAdaptiveSpeechThresholdOnly,
  findSpeechEndSample,
  speechLikeSeconds,
} from "./audio-energy";
import {
  isParakeetClientMessage,
  serializeWorkerError,
  TRANSCRIPTION_BATCH_TIMELINE_LIMIT,
  type AudioDecodingMetrics,
  type ModelLoadSource,
  type ModelLoadProgress,
  type ParakeetClientMessage,
  type ParakeetWorkerMessage,
  type PartialTranscriptionMetrics,
  type TranscriptionBatchTimelineRecord,
  type TranscriptionMetrics,
  type TranscriptionPhaseTimings,
  type TranscriptionProgress,
  type WorkerLoadConfiguration,
} from "./protocol";
import {
  IncrementalStitchAccumulator,
  type TimedToken,
} from "./stitch";
import { serializeTimedTranscript } from "./transcript";
import {
  MonotonicTranscriptionProgress,
  type TranscriptionGraphProgress,
} from "./transcription-progress";

export interface EngineStageTimings {
  readonly frontendMs?: number;
  readonly encoderMs?: number;
  readonly decoderMs?: number;
}

export interface EngineWindowResult {
  /**
   * Tokens must be sorted and expressed in global input-sample coordinates.
   * The window plan contains the offset needed to globalize model timestamps.
   */
  readonly tokens: readonly TimedToken[];
  readonly timings?: EngineStageTimings;
}

/**
 * Bounded-memory writer supplied by the engine for one logical batch.
 *
 * `samples` is the current reusable source PCM window. The shell obtains the
 * getter for every plan, decodes directly into it, and calls `commit`
 * synchronously before reusing the same bounded backing store.
 */
export interface EngineWindowBatchWriter {
  readonly samples: Float32Array<ArrayBuffer>;
  commit(
    window: StatelessWindowPlan,
    batchIndex: number,
    rmsFrameRange?: EngineRmsFrameRange,
  ): void | Uint32Array<ArrayBuffer>;
  finish(): void | Promise<void>;
  abort(): void | Promise<void>;
}

export interface EngineRmsFrameRange {
  /** Sample offset within the currently resident PCM window. */
  readonly sampleOffset: number;
  /** Number of complete ordered 80 ms RMS frames to emit. */
  readonly frameCount: number;
}

export interface EngineBatchExecutionProgress {
  readonly completedWorkUnits: number;
  readonly totalWorkUnits: number;
}

/**
 * Boundary between the streaming worker shell and the raw WebGPU backend.
 *
 * The backend owns model tensors and GPU resources. The shell owns WAV
 * parsing, fixed-size long-form windows, stitching, cancellation and metrics.
 */
export interface ParakeetInferenceEngine {
  readonly diagnostics: Readonly<
    Record<string, unknown> & { readonly fbankHopSamples: number }
  >;
  /** Maximum number of fixed audio windows the shell should keep live. */
  readonly preferredBatchSize: number;
  /** Number of ordered GPU batches the backend can safely keep in flight. */
  readonly maxInFlightBatches: number;

  /** Reset per-file streaming state before the first window is submitted. */
  beginTranscription(): void | Promise<void>;

  /**
   * Expose one reusable input item. `commit` must consume it synchronously
   * before the shell decodes the next window into the same view.
   */
  prepareWindowBatch(
    batchSize: number,
    sampleCount: number,
  ): EngineWindowBatchWriter;

  /**
   * Returns one result for each input window, in the same order.
   * A cardinality mismatch is treated as an engine contract violation.
   * Every `window.samples` view aliases the engine's reusable prepared-batch
   * storage. The engine must consume samples in `commit` and retain any
   * derived state it needs here.
   */
  transcribeWindows(
    windows: readonly StatelessAudioWindow[],
    signal: AbortSignal,
    onExecutionProgress?: (
      progress: EngineBatchExecutionProgress,
    ) => void,
  ): Promise<readonly EngineWindowResult[]>;

  decodeTokenIds(tokenIds: readonly number[]): string;
  tokenPiece(tokenId: number): string | undefined;
  dispose(): void | Promise<void>;
}

export interface EngineFactoryContext {
  readonly modelSource: ModelLoadSource;
  readonly onProgress?: (progress: ModelLoadProgress) => void;
  readonly signal?: AbortSignal;
}

export type ParakeetInferenceEngineFactory = (
  configuration: WorkerLoadConfiguration,
  context: EngineFactoryContext,
) => Promise<ParakeetInferenceEngine>;

export type ParakeetWorkerPostMessage = (
  message: ParakeetWorkerMessage,
) => void;

export interface ParakeetWorkerScope {
  postMessage(message: ParakeetWorkerMessage): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
}

export interface ParakeetWorkerRuntimeOptions {
  readonly postMessage: ParakeetWorkerPostMessage;
  readonly engineFactory: ParakeetInferenceEngineFactory;
  readonly now?: () => number;
}

interface ActiveTranscription {
  readonly jobId: number;
  readonly controller: AbortController;
}

interface ActiveInitialization {
  readonly requestId: number;
  readonly controller: AbortController;
}

interface PreparedStreamingAudio {
  readonly jobId: number;
  readonly reader: SegmentedPcm16Reader;
  readonly inputByteLength: number;
  readonly estimatedSampleCount: number | null;
  failure: unknown | undefined;
  failureReported: boolean;
}

type RepairPrefetchOrigin = "final-primary" | "cross-round";

/**
 * Testable message runtime used by the dedicated module worker.
 *
 * `handleMessage` returns the work Promise for direct tests, while an installed
 * worker intentionally does not await it so cancellation messages remain
 * responsive during inference.
 */
export class ParakeetWorkerRuntime {
  private readonly postMessage: ParakeetWorkerPostMessage;
  private readonly engineFactory: ParakeetInferenceEngineFactory;
  private readonly now: () => number;

  private engine: ParakeetInferenceEngine | undefined;
  private initialization: ActiveInitialization | undefined;
  private active: ActiveTranscription | undefined;
  private preparedStream: PreparedStreamingAudio | undefined;
  private initializationStartedAtMs: number | undefined;
  private initializationEndedAtMs: number | undefined;

  constructor(options: ParakeetWorkerRuntimeOptions) {
    this.postMessage = options.postMessage;
    this.engineFactory = options.engineFactory;
    this.now = options.now ?? defaultNow;
  }

  async handleMessage(message: unknown): Promise<void> {
    if (!isParakeetClientMessage(message)) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "INVALID_MESSAGE",
          "Worker received an invalid Parakeet protocol message",
        ),
      );
      return;
    }

    switch (message.type) {
      case "initialize":
        await this.initialize(message);
        return;
      case "transcribe":
        await this.transcribe(
          message.jobId,
          message.audio,
          message.audioDecoding,
          message.reportProgress,
          message.reportPartialResults,
          message.reportPacedTranscriptBoundary,
        );
        return;
      case "prepare-stream":
        this.prepareStream(
          message.jobId,
          message.port,
          message.inputByteLength,
          message.estimatedSampleCount,
        );
        return;
      case "transcribe-stream":
        await this.transcribeStream(
          message.jobId,
          message.reportProgress,
          message.reportPartialResults,
          message.reportPacedTranscriptBoundary,
        );
        return;
      case "discard-stream":
        this.discardStream(message.jobId);
        return;
      case "cancel-initialization":
        this.cancelInitialization(message.requestId);
        return;
      case "cancel":
        this.cancel(message.jobId);
        return;
    }
  }

  private async initialize(
    message: Extract<ParakeetClientMessage, { readonly type: "initialize" }>,
  ): Promise<void> {
    if (this.engine !== undefined || this.initialization !== undefined) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "ALREADY_INITIALIZED",
          "Parakeet worker is already initialized or loading",
        ),
        { requestId: message.requestId },
      );
      return;
    }

    const active: ActiveInitialization = {
      requestId: message.requestId,
      controller: new AbortController(),
    };
    this.initialization = active;
    this.initializationStartedAtMs = monotonicEpochNow();
    try {
      const engine = await this.engineFactory(
        message.configuration,
        {
          modelSource: message.modelSource,
          signal: active.controller.signal,
          ...(message.reportProgress
            ? {
              onProgress: (progress) => {
                if (this.initialization !== active) return;
                this.postMessage({
                  type: "load-progress",
                  requestId: message.requestId,
                  progress,
                });
              },
            }
            : {}),
        },
      );
      let installed = false;
      try {
        active.controller.signal.throwIfAborted();
        assertPositiveBatchSize(engine.preferredBatchSize);
        assertPositiveBatchSize(engine.maxInFlightBatches);
        const engineHopSamples = engine.diagnostics.fbankHopSamples;
        if (engineHopSamples !== PARAKEET_FBANK_HOP_SAMPLES) {
          throw new ParakeetRuntimeError(
            "ENGINE_TIMELINE_MISMATCH",
            `Engine reported FBank hop ${String(engineHopSamples)}; ` +
              `expected ${String(PARAKEET_FBANK_HOP_SAMPLES)}`,
          );
        }
        this.engine = engine;
        installed = true;
        this.postMessage({
          type: "ready",
          requestId: message.requestId,
          diagnostics: Object.freeze({
            ...engine.diagnostics,
            windowSeconds: PARAKEET_WINDOW_SECONDS,
            windowSamples: PARAKEET_WINDOW_SAMPLES,
            primaryVisibleSamples: PARAKEET_PRIMARY_VISIBLE_SAMPLES,
            primaryContextSamples: PARAKEET_PRIMARY_CONTEXT_SAMPLES,
            windowStrideSamples: PARAKEET_WINDOW_STRIDE_SAMPLES,
            overlapSeconds: PARAKEET_OVERLAP_SECONDS,
            overlapSamples: PARAKEET_OVERLAP_SAMPLES,
            repairVisibleSamples: PARAKEET_REPAIR_VISIBLE_SAMPLES,
            seamGapRepairMinFrames: SEAM_GAP_MIN_FRAMES,
            seamGapRepairMaxProbes: SEAM_GAP_MAX_PROBES,
            seamGapRepairMaxRounds: SEAM_GAP_MAX_ROUNDS,
            fbankHopSamples: PARAKEET_FBANK_HOP_SAMPLES,
            encoderFrameSamples: PARAKEET_ENCODER_FRAME_SAMPLES,
          }),
        });
      } finally {
        if (!installed) await engine.dispose();
      }
    } catch (error) {
      if (active.controller.signal.aborted) {
        this.postMessage({
          type: "initialization-cancelled",
          requestId: message.requestId,
        });
      } else {
        this.postRuntimeError(error, { requestId: message.requestId });
      }
    } finally {
      this.initializationEndedAtMs = monotonicEpochNow();
      if (this.initialization === active) {
        this.initialization = undefined;
      }
    }
  }

  private cancelInitialization(requestId: number): void {
    const active = this.initialization;
    if (active?.requestId !== requestId) return;
    active.controller.abort(
      new DOMException("Parakeet model loading was cancelled", "AbortError"),
    );
  }

  private prepareStream(
    jobId: number,
    port: MessagePort,
    inputByteLength: number,
    estimatedSampleCount: number | null,
  ): void {
    if (
      this.preparedStream !== undefined ||
      this.active !== undefined
    ) {
      port.close();
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "BUSY",
          "Parakeet worker already has a prepared audio stream",
        ),
        { jobId },
      );
      return;
    }
    const prepared: PreparedStreamingAudio = {
      jobId,
      reader: new SegmentedPcm16Reader(port),
      inputByteLength,
      estimatedSampleCount,
      failure: undefined,
      failureReported: false,
    };
    this.preparedStream = prepared;
    void prepared.reader.waitForEnd().catch((error: unknown) => {
      prepared.failure = error;
      if (
        this.preparedStream === prepared &&
        this.active === undefined &&
        !prepared.failureReported
      ) {
        prepared.failureReported = true;
        this.postRuntimeError(error, { jobId });
      }
    });
  }

  private discardStream(jobId: number): void {
    const prepared = this.preparedStream;
    if (prepared?.jobId !== jobId || this.active?.jobId === jobId) return;
    this.releasePreparedStream(prepared);
  }

  private releasePreparedStream(prepared: PreparedStreamingAudio): void {
    if (this.preparedStream === prepared) this.preparedStream = undefined;
    prepared.reader.dispose();
  }

  private async transcribeStream(
    jobId: number,
    reportProgress: boolean,
    reportPartialResults: boolean,
    reportPacedTranscriptBoundary: boolean,
  ): Promise<void> {
    const prepared = this.preparedStream;
    if (prepared?.jobId !== jobId) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "AUDIO_STREAM_NOT_PREPARED",
          "Parakeet streaming audio was not prepared",
        ),
        { jobId },
      );
      return;
    }
    if (prepared.failure !== undefined) {
      if (!prepared.failureReported) {
        prepared.failureReported = true;
        this.postRuntimeError(prepared.failure, { jobId });
      }
      this.releasePreparedStream(prepared);
      return;
    }
    if (this.engine === undefined) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "ENGINE_NOT_INITIALIZED",
          "Parakeet inference engine is not initialized",
        ),
        { jobId },
      );
      this.releasePreparedStream(prepared);
      return;
    }
    if (this.active !== undefined) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "BUSY",
          `Transcription job ${this.active.jobId} is already running`,
        ),
        { jobId },
      );
      this.releasePreparedStream(prepared);
      return;
    }

    const controller = new AbortController();
    const active: ActiveTranscription = { jobId, controller };
    this.active = active;
    try {
      await this.runStreamingTranscription(
        jobId,
        prepared,
        controller.signal,
        this.engine,
        reportProgress,
        reportPartialResults,
        reportPacedTranscriptBoundary,
      );
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) {
        this.postMessage({ type: "cancelled", jobId });
      } else if (!prepared.failureReported) {
        prepared.failureReported = true;
        this.postRuntimeError(error, { jobId });
      }
    } finally {
      if (this.active === active) this.active = undefined;
      this.releasePreparedStream(prepared);
    }
  }

  private async transcribe(
    jobId: number,
    audio: Blob,
    audioDecoding: AudioDecodingMetrics,
    reportProgress: boolean,
    reportPartialResults: boolean,
    reportPacedTranscriptBoundary: boolean,
  ): Promise<void> {
    if (this.engine === undefined) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "ENGINE_NOT_INITIALIZED",
          "Parakeet inference engine is not initialized",
        ),
        { jobId },
      );
      return;
    }
    if (this.active !== undefined) {
      this.postRuntimeError(
        new ParakeetRuntimeError(
          "BUSY",
          `Transcription job ${this.active.jobId} is already running`,
        ),
        { jobId },
      );
      return;
    }

    const controller = new AbortController();
    const active: ActiveTranscription = { jobId, controller };
    this.active = active;

    try {
      await this.runTranscription(
        jobId,
        audio,
        audioDecoding,
        controller.signal,
        this.engine,
        reportProgress,
        reportPartialResults,
        reportPacedTranscriptBoundary,
      );
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) {
        this.postMessage({ type: "cancelled", jobId });
      } else {
        this.postRuntimeError(error, { jobId });
      }
    } finally {
      if (this.active === active) this.active = undefined;
    }
  }

  private async runStreamingTranscription(
    jobId: number,
    streaming: PreparedStreamingAudio,
    signal: AbortSignal,
    engine: ParakeetInferenceEngine,
    reportProgress: boolean,
    reportPartialResults: boolean,
    reportPacedTranscriptBoundary: boolean,
  ): Promise<void> {
    await this.runTranscription(
      jobId,
      new Blob(),
      {
        applied: true,
        wallMs: 0,
        activeMs: 0,
        inputByteLength: streaming.inputByteLength,
        outputByteLength: 0,
        overlapMs: 0,
        pcmWaitMs: 0,
        pcmStarvationMs: 0,
        pcmStarvationCount: 0,
      },
      signal,
      engine,
      reportProgress,
      reportPartialResults,
      reportPacedTranscriptBoundary,
      streaming,
    );
  }

  private async runTranscription(
    jobId: number,
    audio: Blob,
    audioDecoding: AudioDecodingMetrics,
    signal: AbortSignal,
    engine: ParakeetInferenceEngine,
    reportProgress: boolean,
    reportPartialResults: boolean,
    reportPacedTranscriptBoundary: boolean,
    streaming?: PreparedStreamingAudio,
  ): Promise<void> {
    const startedAt = this.now();
    const transcriptionStartedAtMs = monotonicEpochNow();
    const processingElapsedMs = (): number =>
      elapsed(this.now(), startedAt);
    let audioReadMs = 0;
    let inferenceMs = 0;
    let frontendMs = 0;
    let encoderMs = 0;
    let decoderMs = 0;
    let completedWindows = 0;
    let processedAudioSeconds = 0;
    let repairGapProbes = 0;
    let repairWindowDecodes = 0;
    let repairRounds = 0;
    let recoveredTokens = 0;
    let trailingSpeechScanMs = 0;
    let adaptiveThresholdScanMs = 0;
    let gapEnergyScanMs = 0;
    let incrementalStitchMs = 0;
    let prefixStitchMs = 0;
    let finalStitchMs = 0;
    let repairPlanningMs = 0;
    let repairSplicingMs = 0;
    let partialResultMs = 0;
    let tokenDecodeMs = 0;
    let partialRevision = 0;
    const batchSubmissionTimeline: TranscriptionBatchTimelineRecord[] = [];
    let batchSubmissionTimelineOmitted = 0;
    let nextBatchSubmissionIndex = 0;
    let pcmWaitMs = 0;
    let pcmStarvationMs = 0;
    let pcmStarvationCount = 0;

    const postProgress = (progress: TranscriptionProgress): void => {
      if (!reportProgress) return;
      this.postMessage({ type: "progress", jobId, progress });
    };

    if (reportProgress) {
      postProgress({
        phase: "transcribing",
        fraction: 0,
        metrics: partialMetrics({
          audioDecoding,
          audioDurationSeconds: 0,
          processedAudioSeconds,
          elapsedMs: processingElapsedMs(),
          audioReadMs,
          inferenceMs,
          frontendMs,
          encoderMs,
          decoderMs,
          completedWindows,
          totalWindows: 0,
          repairGapProbes,
          repairWindowDecodes,
          repairRounds,
          recoveredTokens,
        }),
      });
    }

    throwIfAborted(signal);
    await engine.beginTranscription();
    throwIfAborted(signal);
    const openStartedAt = this.now();
    const reader =
      streaming === undefined
        ? await Pcm16WavReader.open(audio)
        : streaming.reader;
    audioReadMs += elapsed(this.now(), openStartedAt);
    throwIfAborted(signal);

    const completedStreamingAudioDecoding = (): AudioDecodingMetrics => {
      if (streaming === undefined) return audioDecoding;
      const decoder = streaming.reader.decoderTimings;
      if (decoder === undefined) {
        throw new ParakeetRuntimeError(
          "AUDIO_STREAM_INCOMPLETE",
          "Streaming decoder completed without final timing metadata",
        );
      }
      const initializationOverlap = intervalOverlap(
        decoder.decoderStartedAtMs,
        decoder.decoderEndedAtMs,
        this.initializationStartedAtMs,
        this.initializationEndedAtMs,
      );
      const transcriptionOverlap = intervalOverlap(
        decoder.decoderStartedAtMs,
        decoder.decoderEndedAtMs,
        transcriptionStartedAtMs,
        decoder.decoderEndedAtMs,
      );
      return {
        applied: true,
        wallMs: decoder.decoderWallMs,
        activeMs: decoder.decoderActiveMs,
        inputByteLength: streaming.inputByteLength,
        outputByteLength:
          streaming.reader.sampleCount * Int16Array.BYTES_PER_ELEMENT,
        overlapMs: Math.min(
          decoder.decoderWallMs,
          initializationOverlap + transcriptionOverlap,
        ),
        pcmWaitMs,
        pcmStarvationMs,
        pcmStarvationCount,
      };
    };

    let plans: readonly StatelessWindowPlan[] = [];
    let totalWindows = 0;
    let audioDurationSeconds = 0;
    let primaryPlanBoundary = Number.MAX_SAFE_INTEGER;
    let overallProgress: MonotonicTranscriptionProgress | undefined;
    if (streaming === undefined) {
      const speechEndStartedAt = this.now();
      const speechEndSample = await findSpeechEndSample(reader, signal);
      trailingSpeechScanMs = elapsed(this.now(), speechEndStartedAt);
      audioReadMs += trailingSpeechScanMs;
      throwIfAborted(signal);
      plans = planStatelessWindows(reader.sampleCount, speechEndSample);
      totalWindows = plans.length;
      primaryPlanBoundary = totalWindows;
      audioDurationSeconds = reader.sampleCount / reader.sampleRate;
      overallProgress = reportProgress
        ? new MonotonicTranscriptionProgress({
            totalPrimaryWindows: totalWindows,
            batchCapacity: engine.preferredBatchSize,
          })
        : undefined;
    } else if (reportProgress) {
      const estimatedPrimaryWindows =
        streaming.estimatedSampleCount === null ||
        streaming.estimatedSampleCount === 0
          ? 0
          : 1 +
            Math.max(
              0,
              Math.ceil(
                (streaming.estimatedSampleCount -
                  PARAKEET_PRIMARY_VISIBLE_SAMPLES) /
                  PARAKEET_WINDOW_STRIDE_SAMPLES,
              ),
            );
      overallProgress = new MonotonicTranscriptionProgress({
        totalPrimaryWindows: estimatedPrimaryWindows,
        batchCapacity: engine.preferredBatchSize,
        primaryWindowsSealed: false,
      });
    }
    let lastReportedOverallFraction = -1;
    const currentPartialMetrics = (): PartialTranscriptionMetrics =>
      partialMetrics({
        audioDecoding,
        audioDurationSeconds,
        processedAudioSeconds,
        elapsedMs: processingElapsedMs(),
        audioReadMs,
        inferenceMs,
        frontendMs,
        encoderMs,
        decoderMs,
        completedWindows,
        totalWindows,
        repairGapProbes,
        repairWindowDecodes,
        repairRounds,
        recoveredTokens,
      });
    const reportOverallProgress = (
      forceMetricsUpdate = false,
    ): void => {
      if (!reportProgress || overallProgress === undefined) return;
      const fraction = overallProgress.fraction;
      if (
        !forceMetricsUpdate &&
        fraction <= lastReportedOverallFraction
      ) {
        return;
      }
      lastReportedOverallFraction = Math.max(
        lastReportedOverallFraction,
        fraction,
      );
      postProgress({
        phase: "transcribing",
        fraction,
        metrics: currentPartialMetrics(),
      });
    };
    reportOverallProgress(true);
    const primaryStitcher = new IncrementalStitchAccumulator<TimedToken>({
      tokenPiece: (tokenId) => engine.tokenPiece(tokenId),
    });
    const postPartialResult = (
      tokens: readonly TimedToken[],
      snapshotProcessedAudioSeconds: number,
      partialStartedAt: number,
    ): void => {
      const transcript = serializeTimedTranscript(
        tokens,
        reader.sampleRate,
        engine,
      );
      partialRevision += 1;
      this.postMessage({
        type: "partial-result",
        jobId,
        snapshot: {
          revision: partialRevision,
          ...transcript,
          audioDurationSeconds,
          processedAudioSeconds: snapshotProcessedAudioSeconds,
        },
      });
      partialResultMs += elapsed(this.now(), partialStartedAt);
    };
    const adaptiveSpeechRms = new AdaptiveSpeechRmsAccumulator();
    let completeRmsSampleCount =
      streaming === undefined
        ? Math.floor(
            reader.sampleCount / PARAKEET_ENCODER_FRAME_SAMPLES,
          ) * PARAKEET_ENCODER_FRAME_SAMPLES
        : Number.MAX_SAFE_INTEGER;
    let nextRepairPlanIndex = totalWindows;
    let preparedSpeechThreshold: number | undefined;
    const repairDecodeCache = new Map<
      string,
      readonly TimedToken[]
    >();
    const repairPrefetchKeys = new Set<string>();
    const repairPrefetchOrigins = new Map<
      string,
      RepairPrefetchOrigin
    >();
    const usedRepairPrefetchKeys = new Set<string>();
    const gapSpeechSecondsCache = new Map<string, number>();
    interface RepairProbe {
      readonly id: number;
      readonly gap: SeamGap<TimedToken>;
    }
    interface PlannedProbe {
      readonly probe: RepairProbe;
      readonly plan: StatelessWindowPlan;
    }
    const planProbePlacement = (
      probes: readonly RepairProbe[],
      placement: SeamRepairPlacement,
    ): readonly PlannedProbe[] => {
      const startedAt = this.now();
      const planned = probes
        .map((probe) => ({
          probe,
          plan: planSeamRepairWindow(
            reader.sampleCount,
            probe.gap,
            placement,
            nextRepairPlanIndex++,
          ),
        }))
        .sort(
          (left, right) =>
            left.plan.sourceStartSample -
              right.plan.sourceStartSample ||
            left.probe.id - right.probe.id,
        );
      repairPlanningMs += elapsed(this.now(), startedAt);
      return planned;
    };
    const forecastProbePlacement = (
      probes: readonly RepairProbe[],
      placement: SeamRepairPlacement,
    ): readonly PlannedProbe[] =>
      probes.map((probe, index) => ({
        probe,
        plan: planSeamRepairWindow(
          reader.sampleCount,
          probe.gap,
          placement,
          nextRepairPlanIndex + index,
        ),
      }));
    interface ExecutedWindow {
      readonly window: StatelessAudioWindow;
      readonly decoded: EngineWindowResult;
    }
    interface BatchRows {
      readonly batchClass: "primary" | "mixed" | "repair";
      readonly activeRows: number;
      readonly primaryRows: number;
      readonly repairRows: number;
      readonly officialRepairRows: number;
      readonly finalPrimaryPrefetchRows: number;
      readonly crossRoundPrefetchRows: number;
    }
    const describeBatchRows = (
      batchPlans: readonly StatelessWindowPlan[],
    ): BatchRows => {
      let primaryRows = 0;
      let officialRepairRows = 0;
      let finalPrimaryPrefetchRows = 0;
      let crossRoundPrefetchRows = 0;
      for (const plan of batchPlans) {
        if (plan.index < primaryPlanBoundary) {
          primaryRows += 1;
          continue;
        }
        const origin = repairPrefetchOrigins.get(
          seamRepairPlanCacheKey(plan),
        );
        if (origin === "final-primary") {
          finalPrimaryPrefetchRows += 1;
        } else if (origin === "cross-round") {
          crossRoundPrefetchRows += 1;
        } else {
          officialRepairRows += 1;
        }
      }
      const repairRows =
        officialRepairRows +
        finalPrimaryPrefetchRows +
        crossRoundPrefetchRows;
      return {
        batchClass:
          primaryRows === 0
            ? "repair"
            : repairRows === 0
              ? "primary"
              : "mixed",
        activeRows: batchPlans.length,
        primaryRows,
        repairRows,
        officialRepairRows,
        finalPrimaryPrefetchRows,
        crossRoundPrefetchRows,
      };
    };
    const incrementalRmsRange = (
      plan: StatelessWindowPlan,
    ): EngineRmsFrameRange | undefined => {
      if (plan.index >= primaryPlanBoundary) return undefined;
      const nextSample = adaptiveSpeechRms.nextFrameStartSample;
      const availableEndSample = Math.min(
        plan.sourceEndSample,
        completeRmsSampleCount,
      );
      if (
        nextSample < plan.sourceStartSample ||
        nextSample + PARAKEET_ENCODER_FRAME_SAMPLES >
          availableEndSample
      ) {
        return undefined;
      }
      const frameCount = Math.floor(
        (availableEndSample - nextSample) /
          PARAKEET_ENCODER_FRAME_SAMPLES,
      );
      return {
        sampleOffset: nextSample - plan.sourceStartSample,
        frameCount,
      };
    };
    interface PendingWindowBatch {
      readonly batch: readonly StatelessAudioWindow[];
      readonly rows: BatchRows;
      readonly progress: TranscriptionGraphProgress | undefined;
      readonly submissionIndex: number;
      readonly retainTimelineRecord: boolean;
      readonly preparationStartedAt: number;
      readonly submissionStartedAt: number;
      readonly result: Promise<{
        readonly decodedBatch: readonly EngineWindowResult[];
        readonly completedAt: number;
      }>;
      readonly executedWindows: ExecutedWindow[];
      readonly onWindow: ((executed: ExecutedWindow) => void) | undefined;
      readonly onBatch: (() => void) | undefined;
    }
    const pendingWindowBatches: PendingWindowBatch[] = [];
    let runningWindowBatchCount = 0;
    let latestWindowBatchCompletionAt: number | undefined;
    const markWindowBatchCompleted = (): number => {
      const completedAt = this.now();
      runningWindowBatchCount -= 1;
      latestWindowBatchCompletionAt = Math.max(
        latestWindowBatchCompletionAt ?? completedAt,
        completedAt,
      );
      return completedAt;
    };
    const consumeWindowBatch = async (
      pending: PendingWindowBatch,
    ): Promise<void> => {
      const { decodedBatch, completedAt } = await pending.result;
      const batchWallMs = elapsed(
        completedAt,
        pending.submissionStartedAt,
      );
      throwIfAborted(signal);
      if (decodedBatch.length !== pending.batch.length) {
        throw new ParakeetRuntimeError(
          "ENGINE_BATCH_RESULT_MISMATCH",
          `Parakeet engine returned ${decodedBatch.length} results for a ` +
            `batch of ${pending.batch.length} windows`,
        );
      }

      let reportedInferenceMs = 0;
      let reportedFrontendMs = 0;
      let reportedGpuIntervalMs = 0;
      for (const decoded of decodedBatch) {
        const reportedWindowFrontendMs = nonNegativeFinite(
          decoded.timings?.frontendMs,
        );
        const reportedWindowEncoderMs = nonNegativeFinite(
          decoded.timings?.encoderMs,
        );
        const reportedWindowDecoderMs = nonNegativeFinite(
          decoded.timings?.decoderMs,
        );
        reportedFrontendMs += reportedWindowFrontendMs;
        reportedGpuIntervalMs +=
          reportedWindowEncoderMs + reportedWindowDecoderMs;
        reportedInferenceMs +=
          reportedWindowFrontendMs +
          reportedWindowEncoderMs +
          reportedWindowDecoderMs;
      }
      inferenceMs +=
        reportedInferenceMs === 0 ? batchWallMs : reportedInferenceMs;
      if (pending.retainTimelineRecord) {
        batchSubmissionTimeline.push({
          submissionIndex: pending.submissionIndex,
          ...pending.rows,
          preparationStartedOffsetMs:
            elapsed(pending.preparationStartedAt, startedAt),
          submissionStartedOffsetMs:
            elapsed(pending.submissionStartedAt, startedAt),
          completionOffsetMs:
            elapsed(completedAt, startedAt),
          preparationMs: elapsed(
            pending.submissionStartedAt,
            pending.preparationStartedAt,
          ),
          submissionToCompletionMs: batchWallMs,
          frontendMs: reportedFrontendMs,
          gpuIntervalMs: reportedGpuIntervalMs,
        });
      }

      pending.progress?.complete();
      reportOverallProgress();

      for (
        let batchIndex = 0;
        batchIndex < pending.batch.length;
        batchIndex += 1
      ) {
        const decoded = decodedBatch[batchIndex]!;
        frontendMs += nonNegativeFinite(decoded.timings?.frontendMs);
        encoderMs += nonNegativeFinite(decoded.timings?.encoderMs);
        decoderMs += nonNegativeFinite(decoded.timings?.decoderMs);
        const executed = {
          window: pending.batch[batchIndex]!,
          decoded,
        };
        pending.executedWindows.push(executed);
        pending.onWindow?.(executed);
      }
      pending.onBatch?.();
    };

    const drainWindowBatches = async (): Promise<void> => {
      while (pendingWindowBatches.length > 0) {
        await consumeWindowBatch(pendingWindowBatches.shift()!);
      }
    };

    const settleWindowBatchesAfterFailure = async (): Promise<void> => {
      await Promise.all(
        pendingWindowBatches.splice(0).map(async (pending) => {
          try {
            await pending.result;
          } catch {
            // The primary error/cancellation is reported by the outer runtime.
          }
        }),
      );
    };

    const executeWindowPlans = async (
      windowPlans: readonly StatelessWindowPlan[],
      onWindow?: (executed: ExecutedWindow) => void,
      onBatch?: () => void,
      drain = true,
    ): Promise<readonly ExecutedWindow[]> => {
      const executedWindows: ExecutedWindow[] = [];

      try {
        for (
          let batchStart = 0;
          batchStart < windowPlans.length;
          batchStart += engine.preferredBatchSize
        ) {
          const batchPlans = windowPlans.slice(
            batchStart,
            batchStart + engine.preferredBatchSize,
          );
          const preparationStartedAt = this.now();
          const windowSampleCount =
            batchPlans[0]!.windowSampleCount;
          const batchWriter = engine.prepareWindowBatch(
            batchPlans.length,
            windowSampleCount,
          );
          if (batchWriter.samples.length !== windowSampleCount) {
            throw new ParakeetRuntimeError(
              "ENGINE_INPUT_STORAGE_MISMATCH",
              `Engine prepared ${batchWriter.samples.length} samples for a ` +
                `${windowSampleCount}-sample reusable window`,
            );
          }

          const batch: StatelessAudioWindow[] = [];
          try {
            for (
              let batchIndex = 0;
              batchIndex < batchPlans.length;
              batchIndex += 1
            ) {
              throwIfAborted(signal);
              const plan = batchPlans[batchIndex]!;
              const samples = batchWriter.samples;
              if (plan.readSampleCount < windowSampleCount) {
                samples.fill(0, plan.readSampleCount);
              }
              const readStartedAt = this.now();
              const written = await reader.readSamplesInto(
                plan.sourceStartSample,
                samples,
                0,
                plan.readSampleCount,
              );
              if (written !== plan.readSampleCount) {
                throw new RangeError(
                  `PCM reader wrote ${written} samples for a ` +
                    `${plan.readSampleCount}-sample request`,
                );
              }
              audioReadMs += elapsed(this.now(), readStartedAt);
              throwIfAborted(signal);
              const rmsFrameRange = incrementalRmsRange(plan);
              const rmsBits = batchWriter.commit(
                plan,
                batchIndex,
                rmsFrameRange,
              );
              if (
                rmsFrameRange !== undefined &&
                rmsBits !== undefined
              ) {
                if (
                  !(rmsBits instanceof Uint32Array) ||
                  rmsBits.length !== rmsFrameRange.frameCount
                ) {
                  throw new ParakeetRuntimeError(
                    "ENGINE_RMS_RESULT_MISMATCH",
                    `Parakeet engine returned an invalid RMS result for ` +
                      `window ${plan.index}`,
                  );
                }
                adaptiveSpeechRms.addContiguousFrameBits(
                  plan.sourceStartSample +
                    rmsFrameRange.sampleOffset,
                  rmsBits,
                );
              }
              batch.push({ ...plan, samples });
            }
            await batchWriter.finish();
          } catch (error) {
            try {
              await batchWriter.abort();
            } catch (cleanupError) {
              throw new AggregateError(
                [error, cleanupError],
                "Prepared FBank batch failed and could not be aborted",
              );
            }
            throw error;
          }

          const submissionIndex = nextBatchSubmissionIndex;
          nextBatchSubmissionIndex += 1;
          const rows = describeBatchRows(batchPlans);
          const graphProgress = overallProgress?.beginGraph({
            primaryRows: rows.primaryRows,
            repairRows: rows.repairRows,
          });
          reportOverallProgress();
          const retainTimelineRecord =
            submissionIndex < TRANSCRIPTION_BATCH_TIMELINE_LIMIT;
          if (!retainTimelineRecord) {
            batchSubmissionTimelineOmitted += 1;
          }
          const submissionStartedAt = this.now();
          let completedExecutionWorkUnits = 0;
          let totalExecutionWorkUnits: number | undefined;
          const reportBatchExecutionProgress =
            !reportProgress
              ? undefined
              : (progress: EngineBatchExecutionProgress): void => {
                  if (
                    !Number.isSafeInteger(
                      progress.completedWorkUnits,
                    ) ||
                    !Number.isSafeInteger(
                      progress.totalWorkUnits,
                    ) ||
                    progress.totalWorkUnits < 1 ||
                    progress.completedWorkUnits < 1 ||
                    progress.completedWorkUnits >
                      progress.totalWorkUnits ||
                    progress.completedWorkUnits <=
                      completedExecutionWorkUnits ||
                    (totalExecutionWorkUnits !== undefined &&
                      progress.totalWorkUnits !==
                        totalExecutionWorkUnits)
                  ) {
                    throw new ParakeetRuntimeError(
                      "ENGINE_PROGRESS_MISMATCH",
                      "Parakeet engine returned invalid batch execution progress",
                    );
                  }
                  totalExecutionWorkUnits =
                    progress.totalWorkUnits;
                  completedExecutionWorkUnits =
                    progress.completedWorkUnits;
                  graphProgress?.update(
                    progress.completedWorkUnits,
                    progress.totalWorkUnits,
                  );
                  reportOverallProgress();
                };
          const engineResult = engine.transcribeWindows(
            batch,
            signal,
            reportBatchExecutionProgress,
          );
          runningWindowBatchCount += 1;
          const result = engineResult.then(
            (decodedBatch) => ({
              decodedBatch,
              completedAt: markWindowBatchCompleted(),
            }),
            (error: unknown) => {
              markWindowBatchCompleted();
              throw error;
            },
          );
          const pendingBatch: PendingWindowBatch = {
            batch,
            rows,
            progress: graphProgress,
            submissionIndex,
            retainTimelineRecord,
            preparationStartedAt,
            submissionStartedAt,
            result,
            executedWindows,
            onWindow,
            onBatch,
          };
          pendingWindowBatches.push(pendingBatch);
          if (pendingWindowBatches.length >= engine.maxInFlightBatches) {
            await consumeWindowBatch(pendingWindowBatches.shift()!);
          }
        }
        if (drain) await drainWindowBatches();
      } catch (error) {
        await settleWindowBatchesAfterFailure();
        throw error;
      }
      return executedWindows;
    };

    const missingRepairPlans = (
      planned: readonly PlannedProbe[],
    ): readonly StatelessWindowPlan[] => {
      const missingByKey = new Map<
        string,
        StatelessWindowPlan
      >();
      for (const { plan } of planned) {
        const key = seamRepairPlanCacheKey(plan);
        if (
          !repairDecodeCache.has(key) &&
          !missingByKey.has(key)
        ) {
          missingByKey.set(key, plan);
        }
      }
      return [...missingByKey.values()].sort(
        (left, right) =>
          left.sourceStartSample - right.sourceStartSample ||
          left.index - right.index,
      );
    };

    const executeRepairPlans = async (
      planned: readonly PlannedProbe[],
    ): Promise<void> => {
      const missing = missingRepairPlans(planned);
      if (missing.length === 0) return;

      const executed = await executeWindowPlans(missing);
      repairWindowDecodes += executed.length;
      for (const { window, decoded } of executed) {
        repairDecodeCache.set(
          seamRepairPlanCacheKey(window),
          decoded.tokens,
        );
      }
    };

    const candidatesForPlannedProbes = (
      planned: readonly PlannedProbe[],
    ): Map<number, readonly TimedToken[]> => {
      const splicingStartedAt = this.now();
      const candidates = new Map<number, readonly TimedToken[]>();
      for (const { probe, plan } of planned) {
        const key = seamRepairPlanCacheKey(plan);
        const decoded = repairDecodeCache.get(key);
        if (decoded === undefined) {
          throw new ParakeetRuntimeError(
            "ENGINE_REPAIR_RESULT_MISMATCH",
            `Missing repair result for window ${plan.index}`,
          );
        }
        if (repairPrefetchKeys.has(key)) {
          usedRepairPrefetchKeys.add(key);
        }
        candidates.set(
          probe.id,
          spliceRepairCandidate(
            decoded,
            probe.gap,
            (tokenId) => engine.tokenPiece(tokenId),
          ),
        );
      }
      repairSplicingMs += elapsed(this.now(), splicingStartedAt);
      return candidates;
    };

    const resolveGapSpeechSeconds = async (
      gap: SeamGap<TimedToken>,
      speechThreshold: number,
    ): Promise<number> => {
      const key = `${gap.startSample}:${gap.endSample}`;
      const cached = gapSpeechSecondsCache.get(key);
      if (cached !== undefined) return cached;
      const energyStartedAt = this.now();
      const speechSeconds = await speechLikeSeconds(
        reader,
        gap.startSample,
        gap.endSample,
        speechThreshold,
        signal,
      );
      const energyMs = elapsed(this.now(), energyStartedAt);
      audioReadMs += energyMs;
      gapEnergyScanMs += energyMs;
      gapSpeechSecondsCache.set(key, speechSeconds);
      return speechSeconds;
    };
    const seamGapForTokens = (
      tokens: readonly TimedToken[],
      index: number,
    ): SeamGap<TimedToken> | undefined =>
      seamGapAt(
        tokens,
        index,
        reader.sampleCount,
        (tokenId) => engine.tokenPiece(tokenId),
      );

    const recordPrimaryWindow = ({
      window,
      decoded,
    }: ExecutedWindow): void => {
      const incrementalStitchStartedAt = this.now();
      primaryStitcher.append({
        index: window.index,
        startSample: window.logicalStartSample,
        endSample: window.logicalEndSample,
        tokens: decoded.tokens,
      });
      incrementalStitchMs += elapsed(
        this.now(),
        incrementalStitchStartedAt,
      );
      completedWindows += 1;
      processedAudioSeconds =
        window.logicalEndSample / reader.sampleRate;
    };

    const reportPrimaryBatch = (): void => {
      if (!reportProgress && !reportPartialResults) return;
      if (streaming !== undefined) {
        audioDurationSeconds =
          streaming.reader.availableSampleCount / reader.sampleRate;
      }
      if (reportPartialResults) {
        const partialStartedAt = this.now();
        postPartialResult(
          primaryStitcher.snapshot().tokens,
          processedAudioSeconds,
          partialStartedAt,
        );
      }
      reportOverallProgress(true);
    };

    let primaryPlansToExecute = plans;
    if (streaming !== undefined) {
      try {
        const streamReader = streaming.reader;
        const onlinePlanner = new OnlineStatelessWindowPlanner();
        const pendingStablePlans: StatelessWindowPlan[] = [];
        const extendOnlinePrimaryProgress = (): void => {
          if (overallProgress === undefined) return;
          const knownStableWindows = onlinePlanner.plannedWindowCount;
          const currentEstimate =
            overallProgress.totalPrimaryWindowEstimate;
          const retainFutureGraphHorizon =
            !streamReader.ended &&
            knownStableWindows >= currentEstimate;
          const nextEstimate = Math.max(
            currentEstimate,
            knownStableWindows +
              (retainFutureGraphHorizon
                ? engine.preferredBatchSize
                : 0),
          );
          if (nextEstimate === currentEstimate) return;
          overallProgress.extendPrimaryWindowEstimate(nextEstimate);
          reportOverallProgress();
        };
        while (!streamReader.ended) {
          pendingStablePlans.push(
            ...onlinePlanner.takeStablePrefix(
              streamReader.availableSampleCount,
            ),
          );
          while (
            pendingStablePlans.length >= engine.preferredBatchSize
          ) {
            const batch = pendingStablePlans.splice(
              0,
              engine.preferredBatchSize,
            );
            extendOnlinePrimaryProgress();
            await executeWindowPlans(
              batch,
              recordPrimaryWindow,
              reportPrimaryBatch,
              false,
            );
          }
          if (streamReader.ended) break;

          const nextGraphLastIndex =
            onlinePlanner.plannedWindowCount +
            (engine.preferredBatchSize - pendingStablePlans.length) -
            1;
          const requiredSamples =
            nextGraphLastIndex * PARAKEET_WINDOW_STRIDE_SAMPLES +
            PARAKEET_PRIMARY_VISIBLE_SAMPLES +
            1;
          const waitStartedAt = this.now();
          await streamReader.waitForSamples(requiredSamples, signal);
          const waitEndedAt = this.now();
          pcmWaitMs += elapsed(waitEndedAt, waitStartedAt);
          if (runningWindowBatchCount === 0) {
            const uncoveredStartedAt = Math.max(
              waitStartedAt,
              latestWindowBatchCompletionAt ?? waitStartedAt,
            );
            const uncoveredMs = elapsed(waitEndedAt, uncoveredStartedAt);
            if (uncoveredMs > 0) {
              pcmStarvationMs += uncoveredMs;
              pcmStarvationCount += 1;
            }
          }
          throwIfAborted(signal);
        }

        pendingStablePlans.push(
          ...onlinePlanner.takeStablePrefix(
            streamReader.availableSampleCount,
          ),
        );
        while (
          pendingStablePlans.length >= engine.preferredBatchSize
        ) {
          const batch = pendingStablePlans.splice(
            0,
            engine.preferredBatchSize,
          );
          extendOnlinePrimaryProgress();
          await executeWindowPlans(
            batch,
            recordPrimaryWindow,
            reportPrimaryBatch,
            false,
          );
        }

        const speechEndStartedAt = this.now();
        const speechEndSample = await findSpeechEndSample(reader, signal);
        trailingSpeechScanMs = elapsed(this.now(), speechEndStartedAt);
        audioReadMs += trailingSpeechScanMs;
        throwIfAborted(signal);
        const finalized = onlinePlanner.finish(
          reader.sampleCount,
          speechEndSample,
        );
        plans = finalized.allPlans;
        totalWindows = plans.length;
        primaryPlanBoundary = totalWindows;
        audioDurationSeconds = reader.sampleCount / reader.sampleRate;
        completeRmsSampleCount =
          Math.floor(
            reader.sampleCount / PARAKEET_ENCODER_FRAME_SAMPLES,
          ) * PARAKEET_ENCODER_FRAME_SAMPLES;
        nextRepairPlanIndex = totalWindows;
        audioDecoding = completedStreamingAudioDecoding();
        primaryPlansToExecute = [
          ...pendingStablePlans,
          ...finalized.remainingPlans,
        ];
        if (overallProgress !== undefined) {
          overallProgress.sealPrimaryWindows(totalWindows);
          reportOverallProgress(true);
        }
      } catch (error) {
        await settleWindowBatchesAfterFailure();
        throw error;
      }
    }

    const finalPrimaryBatchSize =
      totalWindows % engine.preferredBatchSize;
    const canPrefetchIntoFinalPrimary =
      finalPrimaryBatchSize > 0 &&
      totalWindows >
        2 * engine.preferredBatchSize;
    if (!canPrefetchIntoFinalPrimary) {
      await executeWindowPlans(
        primaryPlansToExecute,
        recordPrimaryWindow,
        reportPrimaryBatch,
      );
    } else {
      const prefixWindowCount =
        primaryPlansToExecute.length - finalPrimaryBatchSize;
      await executeWindowPlans(
        primaryPlansToExecute.slice(0, prefixWindowCount),
        recordPrimaryWindow,
        reportPrimaryBatch,
      );

      const prefixStitchStartedAt = this.now();
      const prefixStitched = primaryStitcher.snapshot();
      prefixStitchMs += elapsed(this.now(), prefixStitchStartedAt);
      let prefetched: readonly PlannedProbe[] = [];
      if (prefixStitched.tokens.length > 1) {
        const thresholdStartedAt = this.now();
        preparedSpeechThreshold =
          await completeAdaptiveSpeechThresholdOnly(
            reader,
            adaptiveSpeechRms,
            signal,
          );
        const thresholdMs = elapsed(this.now(), thresholdStartedAt);
        audioReadMs += thresholdMs;
        adaptiveThresholdScanMs += thresholdMs;
        const prefetchProbeLimit = Math.min(
          SEAM_GAP_MAX_PROBES,
          engine.preferredBatchSize - finalPrimaryBatchSize,
        );
        const gapPlanningStartedAt = this.now();
        const gapEnergyBeforePlanning = gapEnergyScanMs;
        const prefetchedGapStarts = new Set<number>();
        const probes: RepairProbe[] = [];
        for (
          let index = 0;
          index + 1 < prefixStitched.tokens.length &&
          probes.length < prefetchProbeLimit;
          index += 1
        ) {
          const gap = seamGapForTokens(prefixStitched.tokens, index);
          if (
            gap === undefined ||
            prefetchedGapStarts.has(gap.startFrame)
          ) {
            continue;
          }
          const speechSeconds = await resolveGapSpeechSeconds(
            gap,
            preparedSpeechThreshold,
          );
          if (speechSeconds < SEAM_GAP_MIN_SPEECH_SECONDS) {
            continue;
          }
          prefetchedGapStarts.add(gap.startFrame);
          probes.push({
            id: -(probes.length + 1),
            gap,
          });
        }
        repairPlanningMs += Math.max(
          0,
          elapsed(this.now(), gapPlanningStartedAt) -
            (gapEnergyScanMs - gapEnergyBeforePlanning),
        );
        prefetched = planProbePlacement(probes, "gap-start");
      }

      const finalPrimaryPlans =
        primaryPlansToExecute.slice(prefixWindowCount);
      const physicalPrefetchPlans =
        missingRepairPlans(prefetched);
      if (overallProgress !== undefined && prefetched.length > 0) {
        const prefetchedKeys = new Set(
          physicalPrefetchPlans.map(seamRepairPlanCacheKey),
        );
        const centeredForecastRows = missingRepairPlans(
          forecastProbePlacement(
            prefetched.map(({ probe }) => probe),
            "centered",
          ),
        ).filter(
          (plan) => !prefetchedKeys.has(seamRepairPlanCacheKey(plan)),
        ).length;
        overallProgress.setEstimatedRepairWaves([
          physicalPrefetchPlans.length,
          centeredForecastRows,
        ]);
        reportOverallProgress();
      }
      for (const plan of physicalPrefetchPlans) {
        const key = seamRepairPlanCacheKey(plan);
        repairPrefetchKeys.add(key);
        repairPrefetchOrigins.set(key, "final-primary");
      }
      const combined = [
        ...finalPrimaryPlans,
        ...physicalPrefetchPlans,
      ];
      const executed = await executeWindowPlans(
        combined,
        (window) => {
          if (window.window.index < totalWindows) {
            recordPrimaryWindow(window);
          }
        },
        reportPrimaryBatch,
      );
      for (const { window, decoded } of executed) {
        if (window.index < totalWindows) continue;
        repairWindowDecodes += 1;
        repairDecodeCache.set(
          seamRepairPlanCacheKey(window),
          decoded.tokens,
        );
      }
    }

    throwIfAborted(signal);
    if (reportPacedTranscriptBoundary) {
      this.postMessage({
        type: "paced-transcript-flush",
        jobId,
      });
    }
    processedAudioSeconds = audioDurationSeconds;
    reportOverallProgress(true);

    const stitchingStartedAt = this.now();
    const stitched = primaryStitcher.snapshot();
    finalStitchMs = elapsed(this.now(), stitchingStartedAt);
    let stitchingMs = finalStitchMs;
    let repairedTokens = [...stitched.tokens];
    let repairMs = 0;

    if (completedWindows > 1 && repairedTokens.length > 1) {
      const repairStartedAt = this.now();
      reportOverallProgress(true);
      let speechThreshold = preparedSpeechThreshold;
      if (speechThreshold === undefined) {
        const thresholdStartedAt = this.now();
        speechThreshold = await completeAdaptiveSpeechThresholdOnly(
          reader,
          adaptiveSpeechRms,
          signal,
        );
        const thresholdMs = elapsed(this.now(), thresholdStartedAt);
        audioReadMs += thresholdMs;
        adaptiveThresholdScanMs += thresholdMs;
      }
      const probedGapStarts = new Set<number>();
      let nextProbeId = 0;

      const planNextRoundPrefetch = async (
        firstCandidates: ReadonlyMap<
          number,
          readonly TimedToken[]
        >,
        probes: readonly RepairProbe[],
        centeredPhysicalMissCount: number,
        round: number,
      ): Promise<readonly PlannedProbe[]> => {
        const spareBatchRows =
          engine.preferredBatchSize - centeredPhysicalMissCount;
        const remainingOfficialProbes =
          SEAM_GAP_MAX_PROBES - repairGapProbes;
        if (
          round + 1 >= SEAM_GAP_MAX_ROUNDS ||
          centeredPhysicalMissCount === 0 ||
          spareBatchRows <= 0 ||
          remainingOfficialProbes <= 0
        ) {
          return [];
        }

        const gapPlanningStartedAt = this.now();
        const gapEnergyBeforePlanning = gapEnergyScanMs;
        const successfulTokens = probes.flatMap(({ id }) => {
          const tokens = firstCandidates.get(id) ?? [];
          return tokens.length > 0 ? [...tokens] : [];
        });
        if (successfulTokens.length === 0) {
          repairPlanningMs += elapsed(
            this.now(),
            gapPlanningStartedAt,
          );
          return [];
        }
        const successfulTokenSet = new Set(successfulTokens);
        const provisionalTokens = [
          ...repairedTokens,
          ...successfulTokens,
        ];
        provisionalTokens.sort(
          (left, right) => left.startSample - right.startSample,
        );
        const speculativeGapStarts = new Set(probedGapStarts);
        const prefetchLimit = Math.min(
          spareBatchRows,
          remainingOfficialProbes,
        );
        const predicted: RepairProbe[] = [];

        for (
          let index = 0;
          index + 1 < provisionalTokens.length &&
          predicted.length < prefetchLimit;
          index += 1
        ) {
          if (
            !successfulTokenSet.has(provisionalTokens[index]!) &&
            !successfulTokenSet.has(provisionalTokens[index + 1]!)
          ) {
            continue;
          }
          const gap = seamGapForTokens(provisionalTokens, index);
          if (
            gap === undefined ||
            speculativeGapStarts.has(gap.startFrame)
          ) {
            continue;
          }
          const speechSeconds = await resolveGapSpeechSeconds(
            gap,
            speechThreshold,
          );
          if (speechSeconds < SEAM_GAP_MIN_SPEECH_SECONDS) {
            continue;
          }
          speculativeGapStarts.add(gap.startFrame);
          predicted.push({
            id: -(predicted.length + 1),
            gap,
          });
        }
        repairPlanningMs += Math.max(
          0,
          elapsed(this.now(), gapPlanningStartedAt) -
            (gapEnergyScanMs - gapEnergyBeforePlanning),
        );
        return planProbePlacement(predicted, "gap-start");
      };

      for (
        let round = 0;
        round < SEAM_GAP_MAX_ROUNDS &&
        repairGapProbes < SEAM_GAP_MAX_PROBES;
        round += 1
      ) {
        const gapPlanningStartedAt = this.now();
        const gapEnergyBeforePlanning = gapEnergyScanMs;
        const probes: RepairProbe[] = [];
        for (
          let index = 0;
          index + 1 < repairedTokens.length &&
          repairGapProbes < SEAM_GAP_MAX_PROBES;
          index += 1
        ) {
          const gap = seamGapForTokens(repairedTokens, index);
          if (
            gap === undefined ||
            probedGapStarts.has(gap.startFrame)
          ) {
            continue;
          }

          const speechSeconds = await resolveGapSpeechSeconds(
            gap,
            speechThreshold,
          );
          if (speechSeconds < SEAM_GAP_MIN_SPEECH_SECONDS) {
            continue;
          }

          probedGapStarts.add(gap.startFrame);
          repairGapProbes += 1;
          probes.push({ id: nextProbeId++, gap });
        }
        repairPlanningMs += Math.max(
          0,
          elapsed(this.now(), gapPlanningStartedAt) -
            (gapEnergyScanMs - gapEnergyBeforePlanning),
        );

        if (probes.length === 0) {
          overallProgress?.setEstimatedRepairWaves([]);
          reportOverallProgress(true);
          break;
        }
        repairRounds += 1;
        const firstPlanned = planProbePlacement(
          probes,
          "gap-start",
        );
        const canRunAnotherRound =
          round + 1 < SEAM_GAP_MAX_ROUNDS &&
          repairGapProbes < SEAM_GAP_MAX_PROBES;
        if (overallProgress !== undefined) {
          const firstPhysicalPlans =
            missingRepairPlans(firstPlanned);
          const firstPhysicalKeys = new Set(
            firstPhysicalPlans.map(seamRepairPlanCacheKey),
          );
          const possibleCenteredRows = missingRepairPlans(
            forecastProbePlacement(probes, "centered"),
          ).filter(
            (plan) =>
              !firstPhysicalKeys.has(seamRepairPlanCacheKey(plan)),
          ).length;
          overallProgress.setEstimatedRepairWaves([
            firstPhysicalPlans.length,
            possibleCenteredRows,
          ]);
          reportOverallProgress();
        }
        await executeRepairPlans(firstPlanned);
        const firstCandidates =
          candidatesForPlannedProbes(firstPlanned);
        const failed = probes.filter(
          ({ id }) => (firstCandidates.get(id)?.length ?? 0) === 0,
        );
        let centeredCandidates =
          new Map<number, readonly TimedToken[]>();
        if (failed.length > 0) {
          const centeredPlanned = planProbePlacement(
            failed,
            "centered",
          );
          const centeredPhysicalMissCount =
            missingRepairPlans(centeredPlanned).length;
          overallProgress?.setEstimatedRepairWaves([
            centeredPhysicalMissCount,
          ]);
          reportOverallProgress();
          const prefetched = await planNextRoundPrefetch(
            firstCandidates,
            probes,
            centeredPhysicalMissCount,
            round,
          );
          const centeredPhysicalKeys = new Set(
            missingRepairPlans(centeredPlanned).map((plan) =>
              seamRepairPlanCacheKey(plan),
            ),
          );
          for (const plan of missingRepairPlans(prefetched)) {
            const key = seamRepairPlanCacheKey(plan);
            if (!centeredPhysicalKeys.has(key)) {
              repairPrefetchKeys.add(key);
              repairPrefetchOrigins.set(key, "cross-round");
            }
          }
          const centeredAndPrefetched = [
            ...centeredPlanned,
            ...prefetched,
          ];
          overallProgress?.setEstimatedRepairWaves([
            missingRepairPlans(centeredAndPrefetched).length,
          ]);
          reportOverallProgress();
          await executeRepairPlans(centeredAndPrefetched);
          centeredCandidates =
            candidatesForPlannedProbes(centeredPlanned);
        } else {
          overallProgress?.setEstimatedRepairWaves([]);
          reportOverallProgress();
        }

        const insertionSelectionStartedAt = this.now();
        const inserts: TimedToken[] = [];
        for (const probe of probes) {
          const first = firstCandidates.get(probe.id) ?? [];
          const recovered =
            first.length > 0
              ? first
              : centeredCandidates.get(probe.id) ?? [];
          inserts.push(...recovered);
        }
        repairSplicingMs += elapsed(
          this.now(),
          insertionSelectionStartedAt,
        );

        if (inserts.length === 0) {
          overallProgress?.setEstimatedRepairWaves([]);
          reportOverallProgress(true);
          break;
        }
        recoveredTokens += inserts.length;
        const insertionMergeStartedAt = this.now();
        repairedTokens = [...repairedTokens, ...inserts];
        repairedTokens.sort(
          (left, right) => left.startSample - right.startSample,
        );
        repairSplicingMs += elapsed(
          this.now(),
          insertionMergeStartedAt,
        );
        if (reportPartialResults) {
          const partialStartedAt = this.now();
          postPartialResult(
            repairedTokens,
            audioDurationSeconds,
            partialStartedAt,
          );
        }
        overallProgress?.setEstimatedRepairWaves(
          canRunAnotherRound ? [1, 1] : [],
        );
        reportOverallProgress(true);
      }
      repairMs = elapsed(this.now(), repairStartedAt);
    } else {
      overallProgress?.setEstimatedRepairWaves([]);
    }
    overallProgress?.setEstimatedRepairWaves([]);
    reportOverallProgress(true);

    const decodeStartedAt = this.now();
    const transcript = serializeTimedTranscript(
      repairedTokens,
      reader.sampleRate,
      engine,
    );
    tokenDecodeMs = elapsed(this.now(), decodeStartedAt);
    stitchingMs += incrementalStitchMs + tokenDecodeMs;
    throwIfAborted(signal);

    const totalMs = processingElapsedMs();
    if (streaming !== undefined) {
      audioDecoding = completedStreamingAudioDecoding();
    }
    batchSubmissionTimeline.sort(
      (left, right) =>
        left.submissionIndex - right.submissionIndex,
    );
    const phaseTimings: TranscriptionPhaseTimings = {
      trailingSpeechScanMs,
      adaptiveThresholdScanMs,
      gapEnergyScanMs,
      incrementalStitchMs,
      prefixStitchMs,
      finalStitchMs,
      repairPlanningMs,
      repairSplicingMs,
      partialResultMs,
      tokenDecodeMs,
    };
    const metrics: TranscriptionMetrics = {
      ...partialMetrics({
        audioDecoding,
        audioDurationSeconds,
        processedAudioSeconds: audioDurationSeconds,
        elapsedMs: totalMs,
        audioReadMs,
        inferenceMs,
        frontendMs,
        encoderMs,
        decoderMs,
        completedWindows,
        totalWindows,
        repairGapProbes,
        repairWindowDecodes,
        repairRounds,
        recoveredTokens,
      }),
      stitchingMs,
      repairMs,
      repairPrefetchWindowDecodes: repairPrefetchKeys.size,
      repairPrefetchCacheHits: usedRepairPrefetchKeys.size,
      repairPrefetchUnusedDecodes:
        repairPrefetchKeys.size - usedRepairPrefetchKeys.size,
      batchSubmissionTimeline,
      batchSubmissionTimelineOmitted,
      phaseTimings,
      totalMs,
    };
    if (reportProgress) {
      postProgress({
        phase: "done",
        fraction: overallProgress!.finish(),
        metrics,
      });
    }
    this.postMessage({
      type: "result",
      jobId,
      result: {
        ...transcript,
        metrics,
      },
    });
  }

  private cancel(jobId: number): void {
    if (this.active?.jobId === jobId) this.active.controller.abort();
  }

  private postRuntimeError(
    error: unknown,
    target: { readonly requestId?: number; readonly jobId?: number } = {},
  ): void {
    this.postMessage({
      type: "error",
      ...(target.requestId === undefined
        ? {}
        : { requestId: target.requestId }),
      ...(target.jobId === undefined ? {} : { jobId: target.jobId }),
      error: serializeWorkerError(error),
    });
  }

}

export function installParakeetWorker(
  scope: ParakeetWorkerScope,
  engineFactory: ParakeetInferenceEngineFactory,
): ParakeetWorkerRuntime {
  const runtime = new ParakeetWorkerRuntime({
    postMessage: (message) => scope.postMessage(message),
    engineFactory,
  });
  scope.addEventListener("message", (event) => {
    void runtime.handleMessage(event.data);
  });
  return runtime;
}

class ParakeetRuntimeError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ParakeetRuntimeError";
    this.code = code;
  }
}

interface MetricInputs {
  readonly audioDecoding: AudioDecodingMetrics;
  readonly audioDurationSeconds: number;
  readonly processedAudioSeconds: number;
  readonly elapsedMs: number;
  readonly audioReadMs: number;
  readonly inferenceMs: number;
  readonly frontendMs: number;
  readonly encoderMs: number;
  readonly decoderMs: number;
  readonly completedWindows: number;
  readonly totalWindows: number;
  readonly repairGapProbes: number;
  readonly repairWindowDecodes: number;
  readonly repairRounds: number;
  readonly recoveredTokens: number;
}

function partialMetrics(inputs: MetricInputs): PartialTranscriptionMetrics {
  const speedFactor =
    inputs.elapsedMs <= 0
      ? 0
      : inputs.processedAudioSeconds / (inputs.elapsedMs / 1_000);
  return {
    ...inputs,
    speedFactor,
  };
}

function defaultNow(): number {
  return performance.now();
}

function elapsed(end: number, start: number): number {
  return Math.max(0, end - start);
}

function nonNegativeFinite(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) && value >= 0 ? value : 0;
}

function assertPositiveBatchSize(value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new ParakeetRuntimeError(
      "INVALID_ENGINE_BATCH_SIZE",
      "Parakeet engine preferredBatchSize must be a positive safe integer",
    );
  }
}

function throwIfAborted(signal: AbortSignal): void {
  signal.throwIfAborted();
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function monotonicEpochNow(): number {
  return performance.timeOrigin + performance.now();
}

function intervalOverlap(
  leftStart: number,
  leftEnd: number,
  rightStart: number | undefined,
  rightEnd: number | undefined,
): number {
  if (rightStart === undefined || rightEnd === undefined) return 0;
  return Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
}
