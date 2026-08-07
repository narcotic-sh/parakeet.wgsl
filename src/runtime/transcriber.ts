import {
  openStreamingAudioDecoder,
  type StreamingAudioDecoderSession,
} from "../audio/streaming-audio-decoder.js";
import { Pcm16WavReader, WavFormatError } from "../audio/wav-stream.js";
import type { AudioDecodingRuntimeAssets } from "../audio/audio-decoding-assets.js";
import {
  isParakeetWorkerMessage,
  type AudioDecodingMetrics,
  type ModelLoadProgress,
  type ModelLoadSource,
  type ParakeetClientMessage,
  type ParakeetWorkerMessage,
  type SerializedWorkerError,
  type TranscriptionSnapshot,
  type TranscriptionProgress,
  type TranscriptionResult,
} from "./protocol.js";
import {
  PacedTranscriptDelivery,
  type PacedTranscriptUpdate,
} from "./paced-transcript.js";

export type {
  AudioDecodingMetrics,
  ModelLoadPhase,
  ModelLoadProgress,
  TranscriptionMetrics,
  TranscriptionProgress,
  TranscriptionResult,
  TranscriptionSnapshot,
  TranscriptionToken,
  TranscriptionWord,
} from "./protocol.js";
export type {
  PacedTranscriptTextSplice,
  PacedTranscriptUpdate,
  PacedTranscriptWord,
  PacedTranscriptWordSplice,
} from "./paced-transcript.js";

/** @internal */
export interface ParakeetWorkerLike {
  postMessage(
    message: ParakeetClientMessage,
    transfer?: readonly Transferable[],
  ): void;
  terminate(): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  addEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  removeEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
}

/** @internal */
export interface ParakeetLoadOptions extends AudioDecodingRuntimeAssets {
  /** FP16 package manifest used on adapters exposing shader-f16. */
  readonly fp16ModelUrl: string | URL;
  /** FP32 package manifest used only on adapters without shader-f16. */
  readonly fp32ModelUrl: string | URL;
  readonly wasmUrl: string | URL;
  readonly onLoadProgress?: (progress: ModelLoadProgress) => void;
}

export interface TranscribeOptions {
  /**
   * File name used to identify an in-memory Blob's container when FFmpeg
   * decoding is needed. A File's own non-empty name takes precedence.
   */
  readonly sourceName?: string;
  /**
   * Monotonic overall progress through primary inference and adaptive repair.
   * Scheduled GPU work advances at exact completed execution milestones;
   * undiscovered repair work is estimated, so newly discovered work can pause
   * but never reverse the fraction. Only completion reports one.
   */
  readonly onProgress?: (progress: TranscriptionProgress) => void;
  /**
   * Receives replaceable, timestamped transcript snapshots after each
   * completed primary inference batch and after repair inserts new tokens.
   */
  readonly onPartialResult?: (snapshot: TranscriptionSnapshot) => void;
  /**
   * Receives the same real transcript content as a presentation-paced stream
   * of timestamped word/text splices. Inference and raw snapshot cadence are
   * unchanged. Pending primary words flush when primary inference completes;
   * repair corrections and the exact final update are then immediate. This
   * presentation track never intentionally delays the returned promise.
   */
  readonly onPacedTranscript?: (update: PacedTranscriptUpdate) => void;
  readonly signal?: AbortSignal;
}

interface PendingInitialization {
  readonly requestId: number;
  readonly modelSource: ModelLoadSource;
  readonly promise: Promise<void>;
  readonly resolve: () => void;
  readonly reject: (reason: unknown) => void;
  readonly onProgress: ((progress: ModelLoadProgress) => void) | undefined;
  cancelSent: boolean;
  failureReason: unknown | undefined;
}

interface PendingTranscription {
  readonly jobId: number;
  readonly audio: Blob;
  readonly sourceName: string;
  readonly operationController: AbortController;
  readonly resolve: (result: TranscriptionResult) => void;
  readonly reject: (reason: unknown) => void;
  readonly onProgress: ((progress: TranscriptionProgress) => void) | undefined;
  readonly onPartialResult:
    | ((snapshot: TranscriptionSnapshot) => void)
    | undefined;
  readonly pacedDelivery: PacedTranscriptDelivery | undefined;
  readonly signal: AbortSignal | undefined;
  readonly abortListener: (() => void) | undefined;
  streamingDecoder: StreamingAudioDecoderSession | undefined;
  streamPrepared: boolean;
  audioDecoding: AudioDecodingMetrics | undefined;
  initializationRequestId: number | undefined;
  preparingAudio: boolean;
  transcriptionSent: boolean;
  cancelSent: boolean;
  workerFinished: boolean;
  settling: boolean;
  failureReason: unknown | undefined;
}

/**
 * Main-thread controller for one dedicated Parakeet module worker.
 *
 * Canonical audio is posted as the original Blob/File. Other inputs are
 * decoded into immutable OPFS segments by an isolated worker and streamed to
 * inference. The controller never calls `arrayBuffer()` on a complete input.
 */
export class ParakeetTranscriber {
  private readonly worker: ParakeetWorkerLike;
  private readonly lazyLoadOptions: ParakeetLoadOptions | undefined;
  private nextRequestId = 1;
  private nextJobId = 1;
  private pendingInitialization: PendingInitialization | undefined;
  private pendingTranscription: PendingTranscription | undefined;
  private ready = false;
  private disposed = false;
  private _diagnostics: Readonly<Record<string, unknown>> | null = null;

  private readonly workerMessageListener = (
    event: MessageEvent<unknown>,
  ): void => {
    this.handleWorkerMessage(event.data);
  };

  private readonly workerErrorListener = (event: ErrorEvent): void => {
    const error =
      event.error instanceof Error
        ? event.error
        : deserializeParakeetError({
            name: "WorkerError",
            message: event.message || "Parakeet worker crashed",
            code: "WORKER_ERROR",
          });
    this.terminate(error);
  };

  private constructor(
    worker: ParakeetWorkerLike,
    lazyLoadOptions?: ParakeetLoadOptions,
  ) {
    this.worker = worker;
    this.lazyLoadOptions = lazyLoadOptions;
    worker.addEventListener("message", this.workerMessageListener);
    worker.addEventListener("error", this.workerErrorListener);
  }

  /** @internal */
  static async create(
    options: ParakeetLoadOptions,
  ): Promise<ParakeetTranscriber> {
    const transcriber = new ParakeetTranscriber(
      createDefaultParakeetWorker(),
      options,
    );

    try {
      await transcriber.startInitialization(
        options,
        "cache-or-network",
      ).promise;
      return transcriber;
    } catch (error) {
      transcriber.dispose();
      throw error;
    }
  }

  /** @internal Public consumers construct through createTranscriber(). */
  static createLazy(
    options: ParakeetLoadOptions,
    worker: ParakeetWorkerLike = createDefaultParakeetWorker(),
  ): ParakeetTranscriber {
    return new ParakeetTranscriber(worker, options);
  }

  get diagnostics(): Readonly<Record<string, unknown>> | null {
    return this._diagnostics;
  }

  get initialized(): boolean {
    return this.ready;
  }

  /**
   * Load the capability-selected model only when its complete runtime package
   * is already present in Cache Storage. This never downloads model bytes.
   * Concurrent calls and transcribe() share the same initialization task.
   *
   * @returns true when the model is GPU-resident; false on a cache miss.
   */
  async loadCachedModel(): Promise<boolean> {
    if (this.disposed) {
      throw new ParakeetError(
        "Parakeet transcriber has been disposed",
        "TRANSCRIBER_DISPOSED",
      );
    }
    if (this.ready) return true;
    const loadOptions = this.lazyLoadOptions;
    if (loadOptions === undefined) {
      throw new ParakeetError(
        "Parakeet transcriber has no model loading configuration",
        "MODEL_ASSETS_UNAVAILABLE",
      );
    }

    const initialization =
      this.pendingInitialization ??
      this.startInitialization(loadOptions, "cache-only");
    try {
      await initialization.promise;
      return true;
    } catch (error) {
      if (isModelNotCachedError(error)) return false;
      throw error;
    }
  }

  transcribe(
    audio: Blob,
    options: TranscribeOptions = {},
  ): Promise<TranscriptionResult> {
    if (this.disposed) {
      return Promise.reject(
        new ParakeetError(
          "Parakeet transcriber has been disposed",
          "TRANSCRIBER_DISPOSED",
        ),
      );
    }
    if (!(audio instanceof Blob)) {
      return Promise.reject(
        new TypeError("transcribe() expects an audio File or Blob"),
      );
    }
    if (this.pendingTranscription !== undefined) {
      return Promise.reject(
        new ParakeetError(
          `Transcription job ${this.pendingTranscription.jobId} is already running`,
          "BUSY",
        ),
      );
    }
    if (options.signal?.aborted === true) {
      return Promise.reject(abortError(options.signal.reason));
    }

    const jobId = this.nextJobId;
    this.nextJobId += 1;

    return new Promise<TranscriptionResult>((resolve, reject) => {
      let pending: PendingTranscription;
      const pacedDelivery =
        options.onPacedTranscript === undefined
          ? undefined
          : new PacedTranscriptDelivery({
              onUpdate: options.onPacedTranscript,
              onError: (error) => {
                if (this.pendingTranscription === pending) {
                  this.requestCancellation(pending, error);
                }
              },
            });
      const abortListener =
        options.signal === undefined
          ? undefined
          : () => {
              if (this.pendingTranscription === pending) {
                this.requestCancellation(
                  pending,
                  abortError(options.signal?.reason),
                );
              }
            };
      pending = {
        jobId,
        audio,
        sourceName: sourceName(audio, options),
        operationController: new AbortController(),
        resolve,
        reject,
        onProgress: options.onProgress,
        onPartialResult: options.onPartialResult,
        pacedDelivery,
        signal: options.signal,
        abortListener,
        streamingDecoder: undefined,
        streamPrepared: false,
        audioDecoding: undefined,
        initializationRequestId: undefined,
        preparingAudio: true,
        transcriptionSent: false,
        cancelSent: false,
        workerFinished: false,
        settling: false,
        failureReason: undefined,
      };
      this.pendingTranscription = pending;
      options.signal?.addEventListener("abort", abortListener!, { once: true });
      void this.prepareAndStartTranscription(pending);
    });
  }

  private async prepareAndStartTranscription(
    pending: PendingTranscription,
  ): Promise<void> {
    const loadOptions = this.lazyLoadOptions;
    if (loadOptions === undefined) {
      this.rejectTranscription(
        pending,
        new ParakeetError(
          "Parakeet transcriber has no streaming audio-decoding assets",
          "AUDIO_ASSETS_UNAVAILABLE",
        ),
      );
      return;
    }

    const preparationStartedAt = performance.now();
    try {
      try {
        const canonicalReader = await Pcm16WavReader.open(pending.audio);
        pending.operationController.signal.throwIfAborted();
        pending.audioDecoding = {
          applied: false,
          wallMs: Math.max(0, performance.now() - preparationStartedAt),
          activeMs: 0,
          inputByteLength: pending.audio.size,
          outputByteLength:
            canonicalReader.sampleCount * Int16Array.BYTES_PER_ELEMENT,
          overlapMs: 0,
          pcmWaitMs: 0,
          pcmStarvationMs: 0,
          pcmStarvationCount: 0,
        };
      } catch (error) {
        if (!(error instanceof WavFormatError)) throw error;
        this.notifyAudioDecodingStarted(
          pending,
          Math.max(0, performance.now() - preparationStartedAt),
        );
        const streamingDecoder = await openStreamingAudioDecoder(
          pending.audio,
          pending.sourceName,
          loadOptions,
          pending.operationController.signal,
          (workerError) => {
            if (this.pendingTranscription === pending && !pending.settling) {
              this.requestCancellation(pending, workerError);
            }
          },
        );
        pending.streamingDecoder = streamingDecoder;
        if (this.pendingTranscription !== pending || pending.settling) {
          await streamingDecoder.cleanup();
          return;
        }
        const port = streamingDecoder.takePort();
        this.worker.postMessage(
          {
            type: "prepare-stream",
            jobId: pending.jobId,
            port,
            inputByteLength: streamingDecoder.inputByteLength,
            estimatedSampleCount: streamingDecoder.estimatedSampleCount,
          },
          [port],
        );
        pending.streamPrepared = true;
      }
      pending.preparingAudio = false;
      if (this.pendingTranscription !== pending || pending.settling) {
        return;
      }
      pending.operationController.signal.throwIfAborted();
      if (pending.failureReason !== undefined) throw pending.failureReason;

      if (!this.ready) {
        await this.initializeForTranscription(pending, loadOptions);
        if (this.pendingTranscription !== pending || pending.settling) return;
        pending.operationController.signal.throwIfAborted();
        if (pending.failureReason !== undefined) throw pending.failureReason;
      }
      if (pending.streamingDecoder === undefined) {
        this.dispatchTranscription(pending);
      } else {
        this.dispatchStreamingTranscription(pending);
      }
    } catch (error) {
      pending.preparingAudio = false;
      if (this.pendingTranscription === pending && !pending.settling) {
        this.rejectTranscription(
          pending,
          pending.failureReason ?? error,
        );
      }
    }
  }

  private notifyAudioDecodingStarted(
    pending: PendingTranscription,
    elapsedMs: number,
  ): void {
    if (pending.onProgress === undefined || pending.settling) return;
    try {
      pending.onProgress({
        phase: "transcribing",
        fraction: 0,
        metrics: {
          audioDecoding: {
            applied: true,
            wallMs: elapsedMs,
            activeMs: 0,
            inputByteLength: pending.audio.size,
            outputByteLength: 0,
            overlapMs: 0,
            pcmWaitMs: 0,
            pcmStarvationMs: 0,
            pcmStarvationCount: 0,
          },
          audioDurationSeconds: 0,
          processedAudioSeconds: 0,
          elapsedMs,
          audioReadMs: 0,
          inferenceMs: 0,
          frontendMs: 0,
          encoderMs: 0,
          decoderMs: 0,
          speedFactor: 0,
          completedWindows: 0,
          totalWindows: 0,
          repairGapProbes: 0,
          repairWindowDecodes: 0,
          repairRounds: 0,
          recoveredTokens: 0,
        },
      });
    } catch (error) {
      this.requestCancellation(pending, error);
    }
  }

  /** Request cancellation of the current job, if any. */
  cancel(): boolean {
    const pending = this.pendingTranscription;
    if (pending === undefined) return false;
    this.requestCancellation(
      pending,
      abortError(pending.signal?.reason),
    );
    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.terminate(
      new ParakeetError(
        "Parakeet transcriber was disposed",
        "TRANSCRIBER_DISPOSED",
      ),
    );
  }

  private async initializeForTranscription(
    transcription: PendingTranscription,
    options: ParakeetLoadOptions,
  ): Promise<void> {
    while (!this.ready) {
      if (
        this.pendingTranscription !== transcription ||
        transcription.settling
      ) {
        return;
      }
      transcription.operationController.signal.throwIfAborted();
      const existing = this.pendingInitialization;
      const initialization =
        existing ??
        this.startInitialization(options, "cache-or-network");
      if (existing === undefined) {
        transcription.initializationRequestId = initialization.requestId;
      }
      try {
        await initialization.promise;
      } catch (error) {
        if (
          initialization.modelSource === "cache-only" &&
          isModelNotCachedError(error)
        ) {
          continue;
        }
        throw error;
      }
    }
  }

  private startInitialization(
    options: ParakeetLoadOptions,
    modelSource: ModelLoadSource,
  ): PendingInitialization {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    let resolve!: () => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<void>((promiseResolve, promiseReject) => {
      resolve = promiseResolve;
      reject = promiseReject;
    });
    const pending: PendingInitialization = {
      requestId,
      modelSource,
      promise,
      resolve,
      reject,
      onProgress: options.onLoadProgress,
      cancelSent: false,
      failureReason: undefined,
    };
    this.pendingInitialization = pending;
    try {
      this.worker.postMessage({
        type: "initialize",
        requestId,
        modelSource,
        reportProgress: options.onLoadProgress !== undefined,
        configuration: {
          fp16ModelUrl: String(options.fp16ModelUrl),
          fp32ModelUrl: String(options.fp32ModelUrl),
          wasmUrl: String(options.wasmUrl),
        },
      });
    } catch (error) {
      if (this.pendingInitialization === pending) {
        this.pendingInitialization = undefined;
      }
      reject(error);
    }
    return pending;
  }

  private handleWorkerMessage(value: unknown): void {
    if (!isParakeetWorkerMessage(value)) return;

    switch (value.type) {
      case "load-progress": {
        const pending = this.pendingInitialization;
        if (pending?.requestId === value.requestId) {
          try {
            pending.onProgress?.(value.progress);
          } catch (error) {
            pending.failureReason = error;
            const transcription = this.pendingTranscription;
            if (transcription !== undefined) {
              this.requestCancellation(transcription, error);
            }
            this.cancelInitialization(pending);
          }
        }
        return;
      }
      case "ready": {
        const pending = this.pendingInitialization;
        if (pending?.requestId !== value.requestId) return;
        this.pendingInitialization = undefined;
        this._diagnostics = value.diagnostics;
        this.ready = true;
        if (pending.failureReason === undefined) {
          pending.resolve();
        } else {
          pending.reject(pending.failureReason);
        }
        return;
      }
      case "initialization-cancelled": {
        const pending = this.pendingInitialization;
        if (pending?.requestId !== value.requestId) return;
        this.pendingInitialization = undefined;
        pending.reject(
          pending.failureReason ?? abortError(undefined),
        );
        return;
      }
      case "progress": {
        const pending = this.pendingTranscription;
        if (
          pending?.jobId === value.jobId &&
          !pending.workerFinished
        ) {
          try {
            pending.onProgress?.(value.progress);
          } catch (error) {
            this.requestCancellation(pending, error);
          }
        }
        return;
      }
      case "partial-result": {
        const pending = this.pendingTranscription;
        if (
          pending?.jobId === value.jobId &&
          !pending.workerFinished
        ) {
          pending.pacedDelivery?.pushSnapshot(
            value.snapshot,
            performance.now(),
          );
          try {
            pending.onPartialResult?.(value.snapshot);
          } catch (error) {
            this.requestCancellation(pending, error);
          }
        }
        return;
      }
      case "paced-transcript-flush": {
        const pending = this.pendingTranscription;
        if (
          pending?.jobId === value.jobId &&
          !pending.workerFinished
        ) {
          pending.pacedDelivery?.flushPendingAndDisablePacing();
        }
        return;
      }
      case "result": {
        const pending = this.pendingTranscription;
        if (pending?.jobId !== value.jobId) return;
        pending.workerFinished = true;
        if (pending.failureReason !== undefined) {
          this.rejectTranscription(pending, pending.failureReason);
          return;
        }
        const pacedDelivery = pending.pacedDelivery;
        if (pacedDelivery === undefined) {
          this.resolveTranscription(pending, value.result);
          return;
        }
        void pacedDelivery.finish(value.result).then(
          () => {
            if (this.pendingTranscription === pending) {
              this.resolveTranscription(pending, value.result);
            }
          },
          (error: unknown) => {
            if (this.pendingTranscription === pending) {
              this.rejectTranscription(pending, error);
            }
          },
        );
        return;
      }
      case "cancelled": {
        const pending = this.pendingTranscription;
        if (pending?.jobId !== value.jobId) return;
        this.rejectTranscription(
          pending,
          pending.failureReason ??
            abortError(pending.signal?.reason),
        );
        return;
      }
      case "error":
        this.handleProtocolError(value);
        return;
    }
  }

  private handleProtocolError(
    message: Extract<ParakeetWorkerMessage, { readonly type: "error" }>,
  ): void {
    const error = deserializeParakeetError(message.error);
    const initialization = this.pendingInitialization;
    if (
      initialization !== undefined &&
      (message.requestId === initialization.requestId ||
        (message.requestId === undefined && message.jobId === undefined))
    ) {
      this.pendingInitialization = undefined;
      initialization.reject(error);
      return;
    }

    const transcription = this.pendingTranscription;
    if (
      transcription !== undefined &&
      (message.jobId === transcription.jobId ||
        (message.requestId === undefined && message.jobId === undefined))
    ) {
      if (!transcription.transcriptionSent) {
        this.requestCancellation(transcription, error);
      } else {
        this.rejectTranscription(transcription, error);
      }
    }
  }

  private requestCancellation(
    pending: PendingTranscription,
    reason: unknown,
  ): void {
    if (pending.failureReason === undefined) {
      const failure =
        reason === undefined
          ? new Error("A transcription callback failed")
          : reason;
      pending.failureReason = failure;
      pending.pacedDelivery?.cancel(failure);
      pending.operationController.abort(failure);
    }
    if (!pending.transcriptionSent) {
      const initialization = this.pendingInitialization;
      if (
        initialization !== undefined &&
        initialization.requestId === pending.initializationRequestId
      ) {
        this.cancelInitialization(initialization);
      } else if (pending.preparingAudio) {
        // Audio inspection or decoder opening observes the operation signal
        // and rejects the preparation promise.
        return;
      } else {
        this.rejectTranscription(pending, pending.failureReason);
      }
      return;
    }
    if (pending.workerFinished) {
      this.rejectTranscription(pending, pending.failureReason);
      return;
    }
    if (pending.cancelSent) return;
    pending.cancelSent = true;
    try {
      this.worker.postMessage({ type: "cancel", jobId: pending.jobId });
    } catch (error) {
      this.rejectTranscription(pending, error);
    }
  }

  private cancelInitialization(pending: PendingInitialization): void {
    if (pending.cancelSent) return;
    pending.cancelSent = true;
    try {
      this.worker.postMessage({
        type: "cancel-initialization",
        requestId: pending.requestId,
      });
    } catch (error) {
      if (this.pendingInitialization === pending) {
        this.pendingInitialization = undefined;
      }
      pending.reject(error);
    }
  }

  private dispatchTranscription(pending: PendingTranscription): void {
    if (this.pendingTranscription !== pending) return;
    const audioDecoding = pending.audioDecoding;
    if (audioDecoding === undefined) {
      this.rejectTranscription(
        pending,
        new ParakeetError(
          "Parakeet audio preparation did not produce a canonical stream",
          "AUDIO_PREPARATION_FAILED",
        ),
      );
      return;
    }
    pending.initializationRequestId = undefined;
    pending.transcriptionSent = true;
    pending.cancelSent = false;
    try {
      // No transfer list: Blob/File is structured-cloned without eagerly
      // materializing its complete PCM payload.
      this.worker.postMessage({
        type: "transcribe",
        jobId: pending.jobId,
        audio: pending.audio,
        audioDecoding,
        reportProgress: pending.onProgress !== undefined,
        reportPartialResults:
          pending.onPartialResult !== undefined ||
          pending.pacedDelivery !== undefined,
        reportPacedTranscriptBoundary:
          pending.pacedDelivery !== undefined,
      });
    } catch (error) {
      this.rejectTranscription(pending, error);
    }
  }

  private dispatchStreamingTranscription(pending: PendingTranscription): void {
    if (this.pendingTranscription !== pending) return;
    if (!pending.streamPrepared || pending.streamingDecoder === undefined) {
      this.rejectTranscription(
        pending,
        new ParakeetError(
          "Parakeet streaming audio was not prepared",
          "AUDIO_PREPARATION_FAILED",
        ),
      );
      return;
    }
    pending.initializationRequestId = undefined;
    pending.transcriptionSent = true;
    pending.cancelSent = false;
    try {
      this.worker.postMessage({
        type: "transcribe-stream",
        jobId: pending.jobId,
        reportProgress: pending.onProgress !== undefined,
        reportPartialResults:
          pending.onPartialResult !== undefined ||
          pending.pacedDelivery !== undefined,
        reportPacedTranscriptBoundary:
          pending.pacedDelivery !== undefined,
      });
    } catch (error) {
      this.rejectTranscription(pending, error);
    }
  }

  private resolveTranscription(
    pending: PendingTranscription,
    result: TranscriptionResult,
  ): void {
    this.settleTranscription(pending, () => pending.resolve(result));
  }

  private rejectTranscription(
    pending: PendingTranscription,
    error: unknown,
  ): void {
    pending.pacedDelivery?.cancel(error);
    pending.operationController.abort(error);
    this.settleTranscription(pending, () => pending.reject(error));
  }

  private settleTranscription(
    pending: PendingTranscription,
    settle: () => void,
  ): void {
    if (pending.settling) return;
    pending.settling = true;
    pending.workerFinished = true;
    const streamingDecoder = pending.streamingDecoder;
    pending.streamingDecoder = undefined;
    // The operation is terminal before best-effort scratch cleanup begins.
    // This keeps cancel()/dispose() from reporting that they changed an
    // outcome after the worker has already produced it.
    this.finishTranscription(pending);
    void (async () => {
      try {
        if (pending.streamPrepared && !pending.transcriptionSent) {
          try {
            this.worker.postMessage({
              type: "discard-stream",
              jobId: pending.jobId,
            });
          } catch {
            // Worker failure is already the terminal reason.
          }
        }
        await streamingDecoder?.cleanup();
      } catch (error) {
        console.warn(
          "[parakeet.wgsl] Failed to clean transient decoded audio",
          error,
        );
      } finally {
        settle();
      }
    })();
  }

  private finishTranscription(pending: PendingTranscription): void {
    if (this.pendingTranscription === pending) {
      this.pendingTranscription = undefined;
    }
    if (pending.signal !== undefined && pending.abortListener !== undefined) {
      pending.signal.removeEventListener("abort", pending.abortListener);
    }
  }

  private failAll(error: unknown): void {
    const initialization = this.pendingInitialization;
    this.pendingInitialization = undefined;
    initialization?.reject(error);

    const transcription = this.pendingTranscription;
    if (transcription !== undefined) {
      this.rejectTranscription(transcription, error);
    }
  }

  private terminate(error: unknown): void {
    if (this.disposed) return;
    this.disposed = true;
    this.ready = false;
    this.worker.removeEventListener("message", this.workerMessageListener);
    this.worker.removeEventListener("error", this.workerErrorListener);
    try {
      this.worker.terminate();
    } finally {
      this.failAll(error);
    }
  }
}

/**
 * Keep Vite's required `new Worker(new URL(..., import.meta.url))` shape
 * syntactically intact so production builds bundle the worker graph instead
 * of copying the TypeScript entry as an inert asset.
 */
function createDefaultParakeetWorker(): ParakeetWorkerLike {
  return new Worker(new URL("./worker-entry.ts", import.meta.url), {
    type: "module",
    name: "parakeet-wgsl",
  }) as unknown as ParakeetWorkerLike;
}

export class ParakeetError extends Error {
  readonly code: string;

  constructor(message: string, code: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ParakeetError";
    this.code = code;
  }
}

function deserializeParakeetError(
  serialized: SerializedWorkerError,
): ParakeetError {
  const error = new ParakeetError(serialized.message, serialized.code);
  error.name = serialized.name;
  if (serialized.stack !== undefined) error.stack = serialized.stack;
  return error;
}

function abortError(reason: unknown): Error {
  if (reason instanceof Error && reason.name === "AbortError") return reason;
  return new DOMException("The operation was aborted", "AbortError");
}

function isModelNotCachedError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error as Error & { readonly code?: unknown }).code ===
      "MODEL_NOT_CACHED"
  );
}

function sourceName(audio: Blob, options: TranscribeOptions): string {
  const possibleFile = audio as Blob & { readonly name?: unknown };
  if (
    typeof possibleFile.name === "string" &&
    possibleFile.name.length > 0
  ) {
    return possibleFile.name;
  }
  return options.sourceName ?? "audio";
}
