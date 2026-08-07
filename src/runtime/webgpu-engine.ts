/// <reference types="@webgpu/types" />

import {
  PARAKEET_FBANK_MAX_BATCH,
  ParakeetFbankWasm,
  reusableFrameShiftForWindows,
  type FbankWindowDescriptor,
} from "../audio/fbank-wasm";
import { ParakeetGpuPackage } from "../model/package";
import {
  PARAKEET_FP16_MANIFEST_SHA256,
  PARAKEET_FP32_MANIFEST_SHA256,
  type ParakeetManifest,
  type ParakeetModelPrecision,
} from "../model/manifest";
import {
  ModelCacheSession,
  ModelNotCachedError,
  type ModelCacheAsset,
} from "../model-cache";
import { runModelLoadWithRetry } from "../model-load-retry";
import {
  planRuntimePackage,
  RUNTIME_BUFFER_LIMIT_BYTES,
} from "../model/runtime-plan";
import { ParakeetTokenizer } from "../model/tokenizer";
import {
  PARAKEET_ENCODER_FRAME_SAMPLES,
  type StatelessAudioWindow,
  type StatelessWindowPlan,
} from "./chunking";
import {
  globalizeTdtTokens,
  type TimedToken,
} from "./stitch";
import type { TdtEmittedToken } from "./tdt-types";
import type {
  EngineBatchExecutionProgress,
  EngineWindowBatchWriter,
  EngineWindowResult,
  ParakeetInferenceEngine,
  ParakeetInferenceEngineFactory,
} from "./worker";
import { align, type ArenaSlice } from "../webgpu/arena";
import {
  requestWebGpuContext,
  type ParakeetExecutionProfile,
  type ParakeetOptionalWebGpuFeature,
  type WebGpuCapabilityReport,
} from "../webgpu/capabilities";
import {
  PARAKEET_ENCODER_BATCH_CAPACITY,
  PARAKEET_ENCODER_FRAMES,
  PARAKEET_ENCODER_SUBMISSION_CHUNKS,
  PARAKEET_FEATURE_BINS,
  PARAKEET_FEATURE_FRAMES,
  ParakeetEncoderGraph,
  deriveEncoderValidLengths,
  planParakeetEncoderArena,
} from "../webgpu/encoder";
import {
  FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
  TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
} from "../webgpu/kernels/gemm";
import {
  PersistentTdtDecoderKernel,
  TDT_DECODER_OVERFLOW,
  type TdtDecoderDispatch,
} from "../webgpu/kernels/tdt-decoder";

export const INFERENCE_BATCH_SIZE = PARAKEET_ENCODER_BATCH_CAPACITY;
export const FEATURE_UPLOAD_MODE = "batch" as const;
/**
 * Production deliberately does not enable timestamp-query on the device.
 *
 * Chrome/Dawn on the target Metal backend produced zero query values for the
 * full Parakeet command buffer and later submissions on a reused inference
 * slot could replay stale output. Focused benchmark pages still opt into
 * timestamp-query independently.
 */
export const PRODUCTION_GPU_TIMING_SOURCE = "submit-to-readback-wall" as const;
export const PRODUCTION_OPTIONAL_WEBGPU_FEATURES =
  [] as const satisfies readonly ParakeetOptionalWebGpuFeature[];
const MAX_TOKENS_PER_WINDOW = 2_048;
const REQUIRED_WORKGROUP_STORAGE_BYTES = 32 * 1024;
const FP16_MANIFEST_BYTE_LENGTH = 176_861;
const FP32_MANIFEST_BYTE_LENGTH = 177_045;
const DECODER_STATE_CHANNELS = 4 * 640;
const DECODER_PREDICTOR_SCRATCH_CHANNELS = 5 * 640;
const WORD_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const DECODER_CONTROL_ARRAY_COUNT = 3;
export const INFERENCE_SLOT_COUNT = 2;
export const ENCODER_GRAPH_COUNT = 1;
export const INFERENCE_COMMAND_BUFFERS_PER_GRAPH =
  PARAKEET_ENCODER_SUBMISSION_CHUNKS + 1;
export const COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH =
  INFERENCE_COMMAND_BUFFERS_PER_GRAPH - 1;
export const COOPERATIVE_GPU_IDLE_MILLISECONDS = 1;
export const PRODUCTION_GPU_SUBMISSION_MODE =
  "queue-drained-subsampling-layer-projector-decoder" as const;
export const FEATURE_STAGING_BYTES =
  INFERENCE_BATCH_SIZE *
  PARAKEET_FEATURE_BINS *
  PARAKEET_FEATURE_FRAMES *
  Float32Array.BYTES_PER_ELEMENT;
export const DECODER_CONTROL_STAGING_BYTES =
  DECODER_CONTROL_ARRAY_COUNT *
  INFERENCE_BATCH_SIZE *
  WORD_BYTES;
if (INFERENCE_BATCH_SIZE > PARAKEET_FBANK_MAX_BATCH) {
  throw new Error("The fixed B40 graph exceeds the Wasm frontend capacity");
}

export function selectModelManifestUrl(
  configuration: {
    readonly fp16ModelUrl: string;
    readonly fp32ModelUrl: string;
  },
  precision: ParakeetModelPrecision,
): string {
  return precision === "fp16"
    ? configuration.fp16ModelUrl
    : configuration.fp32ModelUrl;
}

export function validateTokenizerModelContract(
  tokenizer: Pick<
    ParakeetTokenizer,
    "blankTokenId" | "vocabularySize"
  >,
  config: {
    readonly blankTokenId: number;
    readonly vocabularySize: number;
  },
): void {
  if (tokenizer.blankTokenId !== config.blankTokenId) {
    throw new Error("Tokenizer blank token does not match the model");
  }
  if (tokenizer.vocabularySize !== config.vocabularySize) {
    throw new Error("Tokenizer vocabulary size does not match the model");
  }
}

export interface DecoderArenaLayout {
  readonly batchCapacity: number;
  readonly maxTokens: number;
  readonly decodeStarts: ArenaSlice;
  readonly decodeEnds: ArenaSlice;
  readonly flushFinals: ArenaSlice;
  readonly state: ArenaSlice;
  readonly predictorScratch: ArenaSlice;
  readonly tokens: ArenaSlice;
  readonly tokenFrames: ArenaSlice;
  readonly tokenDurations: ArenaSlice;
  readonly metadata: ArenaSlice;
}

export interface DecoderReadbackCopy {
  readonly sourceByteOffset: number;
  readonly destinationByteOffset: number;
  readonly byteLength: number;
}

export interface DecoderReadbackLayout {
  readonly activeBatchSize: number;
  readonly tokens: ArenaSlice;
  readonly tokenFrames: ArenaSlice;
  readonly tokenDurations: ArenaSlice;
  readonly metadata: ArenaSlice;
  readonly copies: readonly DecoderReadbackCopy[];
  readonly readbackByteLength: number;
}

interface InferenceSlot {
  readonly decoderReadback: GPUBuffer;
  busy: boolean;
}

interface SharedInferenceCompute {
  readonly encoderGraph: ParakeetEncoderGraph;
  readonly decoderDispatch: TdtDecoderDispatch;
  readonly decoderLayout: DecoderArenaLayout;
  /**
   * Stable source for controls copied into aliased decoder scratch after the
   * encoder has finished writing that scratch and before TDT dispatch.
   */
  readonly decoderControlStaging: GPUBuffer;
}

interface PreparedFbankBatch {
  readonly inferenceSlot: InferenceSlot;
  readonly batchSize: number;
  readonly sampleCount: number;
  readonly windows: StatelessWindowPlan[];
  readonly validFeatureCounts: number[];
  readonly featureStaging: Float32Array<ArrayBuffer>;
  releaseGraphSubmission: (() => void) | undefined;
  frontendMs: number;
  finished: boolean;
}

export class ExclusiveAsyncGate {
  private tail: Promise<void> = Promise.resolve();

  async acquire(): Promise<() => void> {
    const previous = this.tail;
    let resolveLease!: () => void;
    const lease = new Promise<void>((resolve) => {
      resolveLease = resolve;
    });
    this.tail = previous.then(() => lease);
    await previous;

    let released = false;
    return () => {
      if (released) return;
      released = true;
      resolveLease();
    };
  }
}

class RawWebGpuParakeetEngine implements ParakeetInferenceEngine {
  readonly preferredBatchSize: number;
  readonly maxInFlightBatches: number;
  readonly diagnostics: Readonly<
    Record<string, unknown> & { readonly fbankHopSamples: number }
  >;

  private previousFbankWindow: FbankWindowDescriptor | undefined;
  private preparedFbankBatch: PreparedFbankBatch | undefined;
  private featureStaging: Float32Array<ArrayBuffer> | undefined;
  private readonly graphSubmissionGate = new ExclusiveAsyncGate();
  private disposed = false;
  private gpuError: Error | undefined;
  private readonly uncapturedErrorListener = (
    event: GPUUncapturedErrorEvent,
  ): void => {
    this.gpuError ??= new Error(
      `WebGPU ${event.error.constructor.name}: ${event.error.message}`,
    );
  };

  constructor(
    private readonly device: GPUDevice,
    capabilityReport: WebGpuCapabilityReport,
    private readonly model: ParakeetGpuPackage,
    private readonly fbank: ParakeetFbankWasm,
    private readonly tokenizer: ParakeetTokenizer,
    private readonly inferenceCompute: SharedInferenceCompute,
    private readonly inferenceSlots: readonly InferenceSlot[],
  ) {
    if (inferenceSlots.length !== INFERENCE_SLOT_COUNT) {
      throw new Error(
        `Raw WebGPU engine requires ${INFERENCE_SLOT_COUNT} inference slots`,
      );
    }
    const primaryGraph = inferenceCompute.encoderGraph;
    const transientPaletteParameterPoolTotalBytes =
      primaryGraph.transientPaletteParameterPoolBytes *
      ENCODER_GRAPH_COUNT;
    const transientWeightScratchBytesPerGraph =
      primaryGraph.plan.precision === "fp16"
        ? TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
        : FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES;
    const transientWeightScratchTotalBytes =
      transientWeightScratchBytesPerGraph * ENCODER_GRAPH_COUNT;
    const decoderReadbackBytesPerSlot = planDecoderReadback(
      inferenceCompute.decoderLayout,
      primaryGraph.batchSize,
    ).readbackByteLength;
    this.preferredBatchSize = primaryGraph.batchSize;
    this.maxInFlightBatches = inferenceSlots.length;
    device.addEventListener("uncapturederror", this.uncapturedErrorListener);
    void device.lost.then((info) => {
      if (!this.disposed && info.reason !== "destroyed") {
        this.gpuError ??= new Error(
          `WebGPU device lost (${info.reason}): ${info.message}`,
        );
      }
    });
    this.diagnostics = Object.freeze({
      backend: "raw-webgpu-wgsl-and-simd-wasm",
      model: model.manifest.source.repository,
      revision: model.manifest.source.revision,
      encoderIntermediateSize: model.manifest.config.intermediateSize,
      encoderWeightFormat: model.manifest.config.encoderWeightFormat,
      modelPackageFormat: model.manifest.format,
      executionPrecision: capabilityReport.executionProfile.precision,
      executionSelection:
        capabilityReport.executionProfile.precision === "fp16"
          ? "adapter-exposes-shader-f16"
          : "adapter-does-not-expose-shader-f16",
      kernelBackend: capabilityReport.executionProfile.kernelBackend,
      kernelSelection:
        capabilityReport.executionProfile.kernelBackend === "subgroups"
          ? "adapter-exposes-fixed-32-lane-subgroups"
          : "adapter-lacks-compatible-subgroups",
      ...(primaryGraph.portableF16GemmLoadEncoding === null
        ? {}
        : {
            portableF16GemmLoadEncoding:
              primaryGraph.portableF16GemmLoadEncoding,
          }),
      modelPrecision: model.precision,
      modelManifestUrl: model.manifestUrl,
      modelManifestSha256: model.manifestSha256,
      modelManifestListedBytes: model.manifestListedBytes,
      batchSize: this.preferredBatchSize,
      maxInFlightBatches: this.maxInFlightBatches,
      inferenceSlotCount: inferenceSlots.length,
      encoderGraphCount: ENCODER_GRAPH_COUNT,
      subsamplingMicrobatchSize:
        primaryGraph.plan.subsamplingMicrobatchSize,
      gemmAccumulation:
        primaryGraph.plan.precision === "fp16"
          ? "float16"
          : "float32",
      adapter: capabilityReport.adapterInfo,
      requiredWebGpuFeatures: capabilityReport.requiredFeatures,
      webGpuFeatures: capabilityReport.stockFeatures,
      modelRuntimeGpuBytes:
        model.memory.runtimeGpuBytes,
      modelPackedDownloadBytes:
        model.memory.packedDownloadBytes,
      modelRetainedGpuBytes:
        model.memory.retainedRuntimeBytes,
      modelExpandedGpuBytes:
        model.memory.expandedRuntimeBytes,
      modelLargestRuntimeShardBytes:
        model.memory.largestRuntimeShardBytes,
      modelLoaderPeakGpuBytes:
        model.memory.loaderPeakGpuBytes,
      modelRuntimeBufferCount:
        model.memory.runtimeBufferCount,
      modelTransientWeightScratchBytesPerGraph:
        transientWeightScratchBytesPerGraph,
      modelTransientWeightScratchCount:
        ENCODER_GRAPH_COUNT,
      modelTransientWeightScratchTotalBytes:
        transientWeightScratchTotalBytes,
      modelTransientWeightParameterPoolBytesPerGraph:
        primaryGraph.transientPaletteParameterPoolBytes,
      modelTransientWeightParameterPoolCount: ENCODER_GRAPH_COUNT,
      modelTransientWeightParameterPoolBufferCountPerGraph: 1,
      modelTransientWeightParameterPoolUsedSlotsPerGraph:
        primaryGraph.transientPaletteParameterPoolUsedSlots,
      modelTransientWeightParameterPoolTotalBytes:
        transientPaletteParameterPoolTotalBytes,
      modelTransientWeightMatrixCount:
        primaryGraph.transientPaletteMatrixCount,
      modelTransientWeightExpansionDispatchesPerGraph:
        primaryGraph.transientPaletteExpansionDispatches,
      modelTransientWeightExpandedBytesPerGraph:
        primaryGraph.transientPaletteExpandedBytesPerGraph,
      modelSteadyGpuBytes:
        model.memory.runtimeGpuBytes +
        transientWeightScratchTotalBytes +
        transientPaletteParameterPoolTotalBytes,
      activationArenaBytes: primaryGraph.arena.byteLength,
      activationArenaCount: ENCODER_GRAPH_COUNT,
      activationArenaBufferCount: primaryGraph.arena.bufferCount,
      activationArenaBufferBytes:
        primaryGraph.arena.bufferByteLengths,
      activationArenaLargestBufferBytes:
        primaryGraph.arena.largestBufferByteLength,
      activationArenaTotalBytes:
        primaryGraph.arena.byteLength * ENCODER_GRAPH_COUNT,
      encoderUniformParameterPoolBytesPerGraph:
        primaryGraph.uniformParameterPoolBytes,
      encoderUniformParameterPoolCount: ENCODER_GRAPH_COUNT,
      encoderUniformParameterPoolBufferCountPerGraph:
        primaryGraph.uniformParameterPoolBufferCount,
      encoderUniformParameterPoolUsedSlotsPerGraph:
        primaryGraph.uniformParameterPoolUsedSlots,
      encoderUniformParameterPoolTotalBytes:
        primaryGraph.uniformParameterPoolBytes * ENCODER_GRAPH_COUNT,
      decoderDispatchCount: ENCODER_GRAPH_COUNT,
      decoderReadbackBytesPerSlot,
      decoderReadbackCount: inferenceSlots.length,
      decoderReadbackTotalBytes:
        decoderReadbackBytesPerSlot * inferenceSlots.length,
      decoderControlStagingBytes: DECODER_CONTROL_STAGING_BYTES,
      decoderControlStagingBufferCount: 1,
      wasmHeapBytes: fbank.memoryStats.heapBytes,
      fbankMode: "local-simd-wasm",
      featureStagingBytes: FEATURE_STAGING_BYTES,
      featureStagingBufferCount: 1,
      fbankHopSamples: fbank.hopSamples,
      decoderMode: "gpu",
      tdtRecurrentState: "f32",
      featureUploadMode: FEATURE_UPLOAD_MODE,
      encoderPreset: "full",
      gpuTimingSource: PRODUCTION_GPU_TIMING_SOURCE,
      gpuSubmissionMode: PRODUCTION_GPU_SUBMISSION_MODE,
      encoderCommandBuffersPerGraph:
        PARAKEET_ENCODER_SUBMISSION_CHUNKS,
      commandBuffersPerGraph:
        INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
      queueDrainFencesPerGraph:
        COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
      cooperativeGpuTaskYieldsPerGraph:
        COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
      cooperativeGpuIdleMilliseconds:
        COOPERATIVE_GPU_IDLE_MILLISECONDS,
      maxQueuedParakeetCommandBuffers: 1,
      graphSubmissionConcurrency: 1,
      conformerLayersPerCommandBuffer: 1,
      maxTokensPerWindow: MAX_TOKENS_PER_WINDOW,
    });
    model.releaseInitializationMetadata();
  }

  beginTranscription(): void {
    this.assertAlive();
    if (this.inferenceSlots.some((slot) => slot.busy)) {
      throw new Error("Cannot begin transcription with inference in flight");
    }
    this.previousFbankWindow = undefined;
    this.preparedFbankBatch = undefined;
    this.fbank.resetReuse();
  }

  prepareWindowBatch(
    batchSize: number,
    sampleCount: number,
  ): EngineWindowBatchWriter {
    this.assertAlive();
    if (this.preparedFbankBatch !== undefined) {
      throw new Error("A prepared FBank batch is already pending");
    }
    const inferenceSlot = this.acquireInferenceSlot();
    let samples: Float32Array<ArrayBuffer>;
    try {
      samples = this.fbank.beginPreparedBatch(batchSize, sampleCount);
    } catch (error) {
      inferenceSlot.busy = false;
      throw error;
    }
    const prepared: PreparedFbankBatch = {
      inferenceSlot,
      batchSize,
      sampleCount,
      windows: [],
      validFeatureCounts: [],
      featureStaging: this.featureStagingForBatch(batchSize),
      releaseGraphSubmission: undefined,
      frontendMs: 0,
      finished: false,
    };
    this.preparedFbankBatch = prepared;
    let active = true;

    return {
      samples,
      commit: (window, batchIndex, rmsFrameRange) => {
        this.assertAlive();
        if (!active || this.preparedFbankBatch !== prepared) {
          throw new Error("Prepared FBank batch is no longer active");
        }
        if (
          batchIndex !== prepared.windows.length ||
          batchIndex >= prepared.batchSize
        ) {
          throw new RangeError(
            `Prepared FBank item index must be ${prepared.windows.length}`,
          );
        }
        if (window.windowSampleCount !== prepared.sampleCount) {
          throw new RangeError("Prepared FBank window shape changed");
        }
        const startedAt = performance.now();
        let rmsBits: Uint32Array<ArrayBuffer> | undefined;
        if (rmsFrameRange !== undefined) {
          const rmsSampleEnd =
            rmsFrameRange.sampleOffset +
            rmsFrameRange.frameCount *
              PARAKEET_ENCODER_FRAME_SAMPLES;
          if (
            !Number.isSafeInteger(rmsFrameRange.sampleOffset) ||
            rmsFrameRange.sampleOffset < 0 ||
            !Number.isSafeInteger(rmsFrameRange.frameCount) ||
            rmsFrameRange.frameCount < 1 ||
            rmsSampleEnd > window.validSampleCount
          ) {
            throw new RangeError(
              "Incremental RMS range exceeds the real PCM window",
            );
          }
          rmsBits = this.fbank.computeRmsBits(
            rmsFrameRange.sampleOffset,
            rmsFrameRange.frameCount,
          );
        }
        const current = {
          startSample: window.sourceStartSample,
          sampleCount: window.windowSampleCount,
          validSampleCount: window.validSampleCount,
        };
        const reusableFrameShift = reusableFrameShiftForWindows(
          this.previousFbankWindow,
          current,
        );
        const features = this.fbank.computePreparedItem(batchIndex, {
          reusableFrameShift,
          validSampleCount: window.validSampleCount,
        });
        if (
          features.frameCount !== PARAKEET_FEATURE_FRAMES ||
          features.binCount !== PARAKEET_FEATURE_BINS
        ) {
          throw new Error(
            "SIMD Wasm frontend returned an incompatible feature shape",
          );
        }
        prepared.featureStaging.set(
          features.data,
          batchIndex * PARAKEET_FEATURE_BINS * PARAKEET_FEATURE_FRAMES,
        );
        prepared.validFeatureCounts.push(features.validFrameCount);
        prepared.windows.push(window);
        prepared.frontendMs += performance.now() - startedAt;
        this.previousFbankWindow = current;
        return rmsBits;
      },
      finish: async () => {
        this.assertAlive();
        if (!active || this.preparedFbankBatch !== prepared) {
          throw new Error("Prepared FBank batch is no longer active");
        }
        if (prepared.windows.length !== prepared.batchSize) {
          throw new Error(
            `Prepared ${prepared.windows.length}/${prepared.batchSize} windows`,
          );
        }
        const fbankStartedAt = performance.now();
        this.fbank.finishPreparedBatch();
        prepared.frontendMs += performance.now() - fbankStartedAt;

        const releaseGraphSubmission =
          await this.graphSubmissionGate.acquire();
        try {
          this.assertAlive();
          if (!active || this.preparedFbankBatch !== prepared) {
            throw new Error("Prepared FBank batch is no longer active");
          }
          prepared.releaseGraphSubmission =
            releaseGraphSubmission;
          const uploadStartedAt = performance.now();
          this.inferenceCompute.encoderGraph.uploadFeatures(
            prepared.featureStaging,
          );
          prepared.frontendMs += performance.now() - uploadStartedAt;
          prepared.finished = true;
          active = false;
        } catch (error) {
          prepared.releaseGraphSubmission = undefined;
          releaseGraphSubmission();
          throw error;
        }
      },
      abort: () => {
        const ownsPreparedBatch =
          this.preparedFbankBatch === prepared;
        if (!active && !ownsPreparedBatch) return;
        active = false;
        if (ownsPreparedBatch) {
          this.preparedFbankBatch = undefined;
          releasePreparedGraphSubmission(prepared);
          prepared.inferenceSlot.busy = false;
          this.previousFbankWindow = undefined;
          this.fbank.resetReuse();
        }
      },
    };
  }

  async transcribeWindows(
    windows: readonly StatelessAudioWindow[],
    signal: AbortSignal,
    onExecutionProgress?: (
      progress: EngineBatchExecutionProgress,
    ) => void,
  ): Promise<readonly EngineWindowResult[]> {
    const prepared = this.preparedFbankBatch;
    this.preparedFbankBatch = undefined;
    try {
      this.assertAlive();
    } catch (error) {
      discardPreparedFbankBatch(prepared);
      throw error;
    }
    if (
      windows.length < 1 ||
      windows.length > this.preferredBatchSize
    ) {
      discardPreparedFbankBatch(prepared);
      throw new RangeError(
        `Raw WebGPU engine batch must contain 1-${this.preferredBatchSize} windows`,
      );
    }
    const frontendStartedAt = performance.now();
    if (
      prepared === undefined ||
      !prepared.finished ||
      prepared.batchSize !== windows.length ||
      prepared.windows.length !== windows.length ||
      !windows.every(
        (window, index) =>
          samePreparedWindow(window, prepared.windows[index]!),
      )
    ) {
      discardPreparedFbankBatch(prepared);
      throw new Error(
        "Raw WebGPU inference requires a complete prepared FBank batch",
      );
    }

    const slot = prepared.inferenceSlot;
    let releaseGraphSubmission =
      prepared.releaseGraphSubmission;
    prepared.releaseGraphSubmission = undefined;
    if (releaseGraphSubmission === undefined) {
      slot.busy = false;
      throw new Error(
        "Raw WebGPU inference requires graph-submission ownership",
      );
    }
    const validInputStorage = windows.every(
      (window) =>
        window.samples.length === window.windowSampleCount &&
        window.samples.byteLength ===
          window.windowSampleCount *
            Float32Array.BYTES_PER_ELEMENT,
    );
    if (!validInputStorage) {
      releaseGraphSubmission();
      releaseGraphSubmission = undefined;
      slot.busy = false;
      throw new RangeError(
        `Local FBank source windows must contain ${prepared.sampleCount} samples`,
      );
    }
    if (!slot.busy) {
      releaseGraphSubmission();
      releaseGraphSubmission = undefined;
      throw new Error("Prepared inference slot was released prematurely");
    }
    try {
      throwIfAborted(signal);
      const validFeatureLengths = prepared.validFeatureCounts.map(
        (count) => Math.max(1, count),
      );
      const validEncoderLengths =
        deriveEncoderValidLengths(validFeatureLengths);
      const frontendMs =
        prepared.frontendMs + performance.now() - frontendStartedAt;
      throwIfAborted(signal);

      this.device.pushErrorScope("validation");
      const { encoderGraph, decoderDispatch, decoderLayout } =
        this.inferenceCompute;
      const controlWords = new Uint32Array(
        DECODER_CONTROL_ARRAY_COUNT * decoderLayout.batchCapacity,
      );
      const decodeStarts = controlWords.subarray(
        0,
        decoderLayout.batchCapacity,
      );
      const decodeEnds = controlWords.subarray(
        decoderLayout.batchCapacity,
        decoderLayout.batchCapacity * 2,
      );
      const flushFinals = controlWords.subarray(
        decoderLayout.batchCapacity * 2,
      );
      for (let sequence = 0; sequence < windows.length; sequence += 1) {
        const window = windows[sequence]!;
        validateDecoderFrameRange(
          window,
          validEncoderLengths[sequence]!,
        );
        const decodeStartFrame = window.decodeStartFrame;
        const decodeEndFrame = window.decodeEndFrame;
        decodeStarts[sequence] = decodeStartFrame;
        decodeEnds[sequence] = decodeEndFrame;
        flushFinals[sequence] = window.flushFinal ? 1 : 0;
      }
      this.device.queue.writeBuffer(
        this.inferenceCompute.decoderControlStaging,
        0,
        controlWords,
      );
      const commandBuffers = [
        ...encoderGraph.encodeSubmissionChunks(validFeatureLengths),
      ];
      const decoderCommands = this.device.createCommandEncoder({
        label: "parakeet-decoder-and-readback",
      });
      encodeDecoderControlCopies(
        decoderCommands,
        this.inferenceCompute.decoderControlStaging,
        encoderGraph,
        decoderLayout,
      );
      decoderDispatch.encode(
        decoderCommands,
        undefined,
        windows.length,
      );
      const readbackLayout = planDecoderReadback(
        decoderLayout,
        windows.length,
      );
      encodeDecoderReadbackCopies(
        decoderCommands,
        encoderGraph.arena.bufferFor(decoderLayout.tokens),
        slot.decoderReadback,
        readbackLayout,
      );
      commandBuffers.push(
        decoderCommands.finish({
          label: "parakeet-decoder-and-readback",
        }),
      );
      if (
        commandBuffers.length !==
        INFERENCE_COMMAND_BUFFERS_PER_GRAPH
      ) {
        throw new Error("Inference command-buffer topology changed");
      }
      const validation = this.device.popErrorScope();

      const gpuStartedAt = performance.now();
      // Graph ownership was acquired before the shared feature upload. Keep
      // exactly one Parakeet command buffer outstanding, drain it completely,
      // then leave a real queue-empty interval before submitting the next.
      await submitCommandBuffersCooperatively(
        this.device.queue,
        commandBuffers,
        signal,
        yieldCooperativeGpuIdle,
        onExecutionProgress === undefined
          ? undefined
          : (completedWorkUnits, totalWorkUnits) => {
              onExecutionProgress({
                completedWorkUnits,
                totalWorkUnits,
              });
            },
      );
      const [validationError, words] = await Promise.all([
        validation,
        this.mapDecoderOutput(slot, readbackLayout),
      ]);
      if (validationError !== null) {
        throw new Error(`WebGPU validation failed: ${validationError.message}`);
      }
      this.throwGpuError();
      throwIfAborted(signal);
      onExecutionProgress?.({
        completedWorkUnits: commandBuffers.length,
        totalWorkUnits: commandBuffers.length,
      });
      const gpuMs = performance.now() - gpuStartedAt;
      releaseGraphSubmission();
      releaseGraphSubmission = undefined;
      throwIfAborted(signal);

      const decoderMs = 0;
      const encoderMs = gpuMs;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      throwIfAborted(signal);
      const timingDivisor = windows.length;
      const results: EngineWindowResult[] = [];

      for (let sequence = 0; sequence < windows.length; sequence += 1) {
        if (prepared.validFeatureCounts[sequence] === 0) {
          results.push({
            tokens: [],
            timings: {
              frontendMs: frontendMs / timingDivisor,
              encoderMs: encoderMs / timingDivisor,
              decoderMs: decoderMs / timingDivisor,
            },
          });
          continue;
        }
        const emitted: TdtEmittedToken[] = [];
        const count = this.readWord(
          words,
          readbackLayout,
          readbackLayout.metadata,
          sequence,
        );
        const status = this.readWord(
          words,
          readbackLayout,
          readbackLayout.metadata,
          windows.length + sequence,
        );
        if ((status & TDT_DECODER_OVERFLOW) !== 0) {
          throw new Error(
            `TDT output overflow for window ${windows[sequence]!.index}`,
          );
        }
        if (count > MAX_TOKENS_PER_WINDOW) {
          throw new Error(
            `TDT decoder returned an invalid token count ${count}`,
          );
        }
        const sequenceBase = sequence * MAX_TOKENS_PER_WINDOW;
        for (let tokenIndex = 0; tokenIndex < count; tokenIndex += 1) {
          const element = sequenceBase + tokenIndex;
          const tokenId = this.readWord(
            words,
            readbackLayout,
            readbackLayout.tokens,
            element,
          );
          const frameIndex = this.readWord(
            words,
            readbackLayout,
            readbackLayout.tokenFrames,
            element,
          );
          const durationFrames = this.readWord(
            words,
            readbackLayout,
            readbackLayout.tokenDurations,
            element,
          );
          emitted.push({
            tokenId,
            frameIndex,
            durationFrames,
          });
        }
        const tokens: readonly TimedToken[] = globalizeTdtTokens(
          emitted,
          windows[sequence]!,
        );
        results.push({
          tokens,
          timings: {
            frontendMs: frontendMs / timingDivisor,
            encoderMs: encoderMs / timingDivisor,
            decoderMs: decoderMs / timingDivisor,
          },
        });
      }
      return results;
    } finally {
      releaseGraphSubmission?.();
      if (slot.decoderReadback.mapState === "mapped") {
        slot.decoderReadback.unmap();
      }
      slot.busy = false;
    }
  }

  decodeTokenIds(tokenIds: readonly number[]): string {
    this.assertAlive();
    return this.tokenizer.decode(tokenIds);
  }

  tokenPiece(tokenId: number): string {
    this.assertAlive();
    return this.tokenizer.pieceForTokenId(tokenId);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    discardPreparedFbankBatch(this.preparedFbankBatch);
    this.preparedFbankBatch = undefined;
    this.device.removeEventListener(
      "uncapturederror",
      this.uncapturedErrorListener,
    );
    for (const slot of this.inferenceSlots) {
      if (slot.decoderReadback.mapState === "mapped") {
        slot.decoderReadback.unmap();
      }
      slot.decoderReadback.destroy();
    }
    this.inferenceCompute.decoderDispatch.destroy();
    this.inferenceCompute.decoderControlStaging.destroy();
    this.inferenceCompute.encoderGraph.destroy();
    this.model.destroy();
    this.fbank.dispose();
    this.device.destroy();
  }

  private acquireInferenceSlot(): InferenceSlot {
    const slot = this.inferenceSlots.find((candidate) => !candidate.busy);
    if (slot === undefined) {
      throw new Error(
        `Raw WebGPU engine supports at most ${INFERENCE_SLOT_COUNT} in-flight batches`,
      );
    }
    slot.busy = true;
    return slot;
  }

  private featureStagingForBatch(
    batchSize: number,
  ): Float32Array<ArrayBuffer> {
    // GPUQueue.writeBuffer snapshots the source bytes synchronously. Once a
    // prepared batch has uploaded them, frontend work for the next batch can
    // safely refill this one CPU arena while the prior GPU submission runs.
    this.featureStaging ??= new Float32Array(
      FEATURE_STAGING_BYTES / Float32Array.BYTES_PER_ELEMENT,
    );
    const elementCount =
      batchSize * PARAKEET_FEATURE_BINS * PARAKEET_FEATURE_FRAMES;
    return this.featureStaging.subarray(0, elementCount);
  }

  private async mapDecoderOutput(
    slot: InferenceSlot,
    layout: DecoderReadbackLayout,
  ): Promise<Uint32Array<ArrayBuffer>> {
    await slot.decoderReadback.mapAsync(
      GPUMapMode.READ,
      0,
      layout.readbackByteLength,
    );
    return new Uint32Array(
      slot.decoderReadback.getMappedRange(0, layout.readbackByteLength),
    );
  }

  private readWord(
    readback: Uint32Array<ArrayBuffer>,
    layout: DecoderReadbackLayout,
    slice: ArenaSlice,
    element: number,
  ): number {
    const byteOffset = slice.byteOffset + element * WORD_BYTES;
    if (
      byteOffset < slice.byteOffset ||
      byteOffset + WORD_BYTES >
        slice.byteOffset + slice.byteLength ||
      byteOffset + WORD_BYTES > layout.readbackByteLength
    ) {
      throw new RangeError(`Decoder read exceeded ${slice.label}`);
    }
    return readback[byteOffset / WORD_BYTES]!;
  }

  private assertAlive(): void {
    if (this.disposed) {
      throw new Error("Raw WebGPU Parakeet engine was disposed");
    }
    this.throwGpuError();
  }

  private throwGpuError(): void {
    if (this.gpuError !== undefined) throw this.gpuError;
  }
}

/**
 * Submits one scheduling quantum at a time. The optional completion callback
 * runs only for drained non-final buffers while the queue-empty timer is
 * already active; the caller reports the final decoder/readback buffer after
 * its existing map fence succeeds.
 */
export async function submitCommandBuffersCooperatively(
  queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone">,
  commandBuffers: readonly GPUCommandBuffer[],
  signal: AbortSignal,
  yieldQueueIdle: () => Promise<void> =
    yieldCooperativeGpuIdle,
  onCommandBufferCompleted?: (
    completedCommandBuffers: number,
    totalCommandBuffers: number,
  ) => void,
): Promise<void> {
  if (commandBuffers.length === 0) {
    throw new RangeError("At least one GPU command buffer is required");
  }
  for (
    let index = 0;
    index < commandBuffers.length;
    index += 1
  ) {
    throwIfAborted(signal);
    const commandBuffer = commandBuffers[index]!;
    queue.submit([commandBuffer]);
    if (index === commandBuffers.length - 1) break;
    await queue.onSubmittedWorkDone();
    throwIfAborted(signal);
    const idleCompletion = yieldQueueIdle();
    try {
      onCommandBufferCompleted?.(
        index + 1,
        commandBuffers.length,
      );
    } catch (error) {
      await idleCompletion;
      throw error;
    }
    await idleCompletion;
    throwIfAborted(signal);
  }
}

function yieldCooperativeGpuIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

function releasePreparedGraphSubmission(
  prepared: PreparedFbankBatch,
): void {
  const release = prepared.releaseGraphSubmission;
  if (release === undefined) return;
  prepared.releaseGraphSubmission = undefined;
  release();
}

function discardPreparedFbankBatch(
  prepared: PreparedFbankBatch | undefined,
): void {
  if (prepared === undefined) return;
  releasePreparedGraphSubmission(prepared);
  prepared.inferenceSlot.busy = false;
}

export function validateDecoderFrameRange(
  window: Pick<
    StatelessWindowPlan,
    "index" | "decodeStartFrame" | "decodeEndFrame"
  >,
  validEncoderLength: number,
): void {
  if (
    !Number.isSafeInteger(validEncoderLength) ||
    validEncoderLength < 1 ||
    validEncoderLength > PARAKEET_ENCODER_FRAMES
  ) {
    throw new RangeError("validEncoderLength is outside graph capacity");
  }
  if (
    !Number.isSafeInteger(window.decodeStartFrame) ||
    window.decodeStartFrame < 0 ||
    !Number.isSafeInteger(window.decodeEndFrame) ||
    window.decodeEndFrame <= window.decodeStartFrame ||
    window.decodeEndFrame > PARAKEET_ENCODER_FRAMES ||
    window.decodeStartFrame >= validEncoderLength
  ) {
    throw new RangeError(
      `Window ${window.index} has an invalid decoder range for its ` +
        "valid encoder output",
    );
  }
}

export const createRawWebGpuParakeetEngine: ParakeetInferenceEngineFactory =
  async (configuration, context) => {
    const reportProgress = context.onProgress;
    const requiredStorageBytes = Math.max(
      RUNTIME_BUFFER_LIMIT_BYTES,
      planParakeetEncoderArena().byteLength,
    );
    reportProgress?.({ phase: "webgpu", fraction: 0 });
    context.signal?.throwIfAborted();
    const webgpu = await requestWebGpuContext({
      optionalFeatures: PRODUCTION_OPTIONAL_WEBGPU_FEATURES,
      requiredLimits: {
        maxBufferSize: requiredStorageBytes,
        maxStorageBufferBindingSize: requiredStorageBytes,
        maxComputeWorkgroupStorageSize: REQUIRED_WORKGROUP_STORAGE_BYTES,
      },
    });
    const { device, report } = webgpu;
    if (report.requiredFeatures.some((feature) => !device.features.has(feature))) {
      device.destroy();
      throw new Error(
        "Raw Parakeet WebGPU inference did not enable its selected profile",
      );
    }
    try {
      context.signal?.throwIfAborted();
    } catch (error) {
      device.destroy();
      throw error;
    }
    reportProgress?.({ phase: "webgpu", fraction: 1 });

    let fbank: ParakeetFbankWasm | undefined;
    let model: ParakeetGpuPackage | undefined;
    let inferenceCompute: SharedInferenceCompute | undefined;
    const inferenceSlots: InferenceSlot[] = [];
    try {
      const modelUrl = selectModelManifestUrl(
        configuration,
        webgpu.executionProfile.precision,
      );
      const loadFbank = async (): Promise<void> => {
        reportProgress?.({ phase: "wasm", fraction: 0 });
        fbank = await ParakeetFbankWasm.create(
          configuration.wasmUrl,
          context.signal,
        );
        context.signal?.throwIfAborted();
        reportProgress?.({ phase: "wasm", fraction: 1 });
      };
      if (context.modelSource === "cache-or-network") {
        await loadFbank();
      }
      model = await loadSelectedModelPackage(
        device,
        modelUrl,
        webgpu.executionProfile.precision,
        context,
      );
      if (context.modelSource === "cache-only") {
        // A cold visit proves the selected runtime inventory is present before
        // fetching or compiling even the small local FBank Wasm asset.
        await loadFbank();
      }
      if (fbank === undefined) {
        throw new Error("Parakeet FBank Wasm was not initialized");
      }
      const tokenizer = ParakeetTokenizer.fromJSON(model.tokenizer);
      validateTokenizerModelContract(tokenizer, model.manifest.config);

      reportProgress?.({ phase: "pipelines", fraction: 0 });
      context.signal?.throwIfAborted();
      inferenceCompute = await createSharedInferenceCompute(
        device,
        model,
        webgpu.executionProfile,
      );
      reportProgress?.({ phase: "pipelines", fraction: 0.5 });
      for (let index = 0; index < INFERENCE_SLOT_COUNT; index += 1) {
        inferenceSlots.push(
          createInferenceSlot(device, inferenceCompute, index),
        );
        reportProgress?.({
          phase: "pipelines",
          fraction:
            0.5 + (0.5 * (index + 1)) / INFERENCE_SLOT_COUNT,
        });
      }
      context.signal?.throwIfAborted();
      return new RawWebGpuParakeetEngine(
        device,
        report,
        model,
        fbank,
        tokenizer,
        inferenceCompute,
        inferenceSlots,
      );
    } catch (error) {
      for (const slot of inferenceSlots) {
        slot.decoderReadback.destroy();
      }
      inferenceCompute?.decoderDispatch.destroy();
      inferenceCompute?.decoderControlStaging.destroy();
      inferenceCompute?.encoderGraph.destroy();
      model?.destroy();
      fbank?.dispose();
      device.destroy();
      throw error;
    }
  };

async function loadSelectedModelPackage(
  device: GPUDevice,
  manifestUrl: string,
  precision: ParakeetModelPrecision,
  context: Parameters<ParakeetInferenceEngineFactory>[1],
): Promise<ParakeetGpuPackage> {
  let activeFetch: typeof fetch | undefined;
  const cache = new ModelCacheSession(
    selectedManifestAsset(manifestUrl, precision),
    {
      cacheOnly: context.modelSource === "cache-only",
      fetch: (input, init) => {
        if (activeFetch === undefined) {
          throw new Error("The Parakeet model fetch attempt has not started");
        }
        return activeFetch(input, init);
      },
    },
  );
  let corruptionRetries = 0;
  try {
    return await runModelLoadWithRetry(
      async (attempt) => {
        activeFetch = attempt.fetch;
        return await ParakeetGpuPackage.load(device, manifestUrl, {
          expectedPrecision: precision,
          fetch: cache.fetch,
          ...(context.signal === undefined
            ? {}
            : { signal: context.signal }),
          onManifest: async (manifest) => {
            cache.registerAssets(selectedPackageAssets(manifestUrl, manifest));
            if (context.modelSource === "cache-only") {
              await cache.requireCachedAssets(
                selectedRuntimePackageAssets(manifestUrl, manifest),
              );
            }
          },
          ...(context.onProgress === undefined
            ? {}
            : {
                onProgress(progress) {
                  context.onProgress?.({
                    phase: progress.phase,
                    fraction:
                      progress.totalBytes === 0
                        ? 0
                        : progress.loadedBytes / progress.totalBytes,
                    loadedBytes: progress.loadedBytes,
                    totalBytes: progress.totalBytes,
                  });
                },
              }),
        });
      },
      {
        ...(context.signal === undefined
          ? {}
          : { signal: context.signal }),
        cleanup: async (error) => {
          await cache.cancelInFlight(error);
        },
        recover: () => {
          if (context.modelSource === "cache-only") return false;
          const recover =
            cache.consumeCacheCorruption() &&
            corruptionRetries < cache.maximumCorruptionRetries;
          if (!recover) return false;
          corruptionRetries += 1;
          context.onProgress?.({
            phase: "cache",
            fraction: 0,
            message: "Replacing a damaged cached model file",
          });
          return true;
        },
        onRetry: (event) => {
          context.onProgress?.({
            phase: "retry",
            fraction: 0,
            attempt: event.nextAttempt,
            maxAttempts: event.maxAttempts,
            message:
              `Temporary model ${event.error.phase} failure; retrying in ` +
              `${Math.max(0.1, event.delayMs / 1_000).toFixed(1)} seconds`,
          });
        },
      },
    );
  } catch (error) {
    if (
      context.modelSource === "cache-only" &&
      (error instanceof ModelNotCachedError ||
        cache.consumeCacheCorruption())
    ) {
      throw new ModelNotCachedError();
    }
    throw error;
  } finally {
    activeFetch = undefined;
    await cache.settle();
  }
}

function selectedManifestAsset(
  manifestUrl: string,
  precision: ParakeetModelPrecision,
): ModelCacheAsset {
  return {
    url: manifestUrl,
    byteLength:
      precision === "fp16"
        ? FP16_MANIFEST_BYTE_LENGTH
        : FP32_MANIFEST_BYTE_LENGTH,
    sha256:
      precision === "fp16"
        ? PARAKEET_FP16_MANIFEST_SHA256
        : PARAKEET_FP32_MANIFEST_SHA256,
  };
}

function selectedPackageAssets(
  manifestUrl: string,
  manifest: ParakeetManifest,
): ModelCacheAsset[] {
  return manifest.files.map((file) => ({
    url: new URL(file.name, absoluteUrl(manifestUrl)).href,
    byteLength: file.byteLength,
    sha256: file.sha256,
  }));
}

function selectedRuntimePackageAssets(
  manifestUrl: string,
  manifest: ParakeetManifest,
): ModelCacheAsset[] {
  const runtimePlan = planRuntimePackage(manifest);
  const tokenizer = manifest.files.find(
    (file) => file.name === "tokenizer.json",
  );
  if (tokenizer === undefined) {
    throw new Error("Selected model manifest is missing tokenizer.json");
  }
  return [...runtimePlan.shards.map(({ file }) => file), tokenizer].map(
    (file) => ({
      url: new URL(file.name, absoluteUrl(manifestUrl)).href,
      byteLength: file.byteLength,
      sha256: file.sha256,
    }),
  );
}

function absoluteUrl(url: string): URL {
  return new URL(
    url,
    globalThis.location?.href ?? "http://localhost/",
  );
}

async function createSharedInferenceCompute(
  device: GPUDevice,
  model: ParakeetGpuPackage,
  executionProfile: ParakeetExecutionProfile,
): Promise<SharedInferenceCompute> {
  let encoderGraph: ParakeetEncoderGraph | undefined;
  let decoderDispatch: TdtDecoderDispatch | undefined;
  let decoderControlStaging: GPUBuffer | undefined;
  try {
    encoderGraph = await ParakeetEncoderGraph.create(
      device,
      model,
      executionProfile,
    );
    const decoderLayout = planDecoderArena(
      encoderGraph,
      MAX_TOKENS_PER_WINDOW,
    );
    const decoderKernel = await PersistentTdtDecoderKernel.create(
      device,
      encoderGraph.arena,
      model,
      executionProfile,
    );
    decoderControlStaging = device.createBuffer({
      label: "parakeet-decoder-control-staging",
      size: DECODER_CONTROL_STAGING_BYTES,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    decoderDispatch = decoderKernel.createDispatch({
      label: "parakeet-persistent-tdt-decode",
      ...encoderGraph.decoderInput(),
      state: decoderLayout.state,
      decodeStarts: decoderLayout.decodeStarts,
      decodeEnds: decoderLayout.decodeEnds,
      flushFinals: decoderLayout.flushFinals,
      predictorScratch: decoderLayout.predictorScratch,
      tokens: decoderLayout.tokens,
      tokenFrames: decoderLayout.tokenFrames,
      tokenDurations: decoderLayout.tokenDurations,
      metadata: decoderLayout.metadata,
      maxTokens: MAX_TOKENS_PER_WINDOW,
    });
    return {
      encoderGraph,
      decoderDispatch,
      decoderLayout,
      decoderControlStaging,
    };
  } catch (error) {
    decoderDispatch?.destroy();
    decoderControlStaging?.destroy();
    encoderGraph?.destroy();
    throw error;
  }
}

function createInferenceSlot(
  device: GPUDevice,
  inferenceCompute: SharedInferenceCompute,
  index: number,
): InferenceSlot {
  return {
    decoderReadback: device.createBuffer({
      label: `parakeet-decoder-readback-${index}`,
      size: planDecoderReadback(
        inferenceCompute.decoderLayout,
        inferenceCompute.encoderGraph.batchSize,
      ).readbackByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    busy: false,
  };
}

export function planDecoderArena(
  encoder: ParakeetEncoderGraph,
  maxTokens: number,
): DecoderArenaLayout {
  const batchSize = encoder.batchSize;
  const available = encoder.plan.decoderScratch;
  let cursor = available.byteOffset;
  const allocate = (label: string, byteLength: number): ArenaSlice => {
    cursor = align(cursor, 256);
    const result = encoder.arena.slice(
      label,
      cursor,
      byteLength,
      available.bufferIndex ?? 0,
    );
    cursor += byteLength;
    return result;
  };

  const state = allocate(
    "tdt-state",
    batchSize *
      DECODER_STATE_CHANNELS *
      Float32Array.BYTES_PER_ELEMENT,
  );
  const decodeStarts = allocate(
    "tdt-decode-starts",
    batchSize * WORD_BYTES,
  );
  const decodeEnds = allocate(
    "tdt-decode-ends",
    batchSize * WORD_BYTES,
  );
  const flushFinals = allocate(
    "tdt-flush-finals",
    batchSize * WORD_BYTES,
  );
  const predictorScratch = allocate(
    "tdt-predictor-scratch",
    batchSize *
      DECODER_PREDICTOR_SCRATCH_CHANNELS *
      Float32Array.BYTES_PER_ELEMENT,
  );
  const tokens = allocate(
    "tdt-tokens",
    batchSize * maxTokens * WORD_BYTES,
  );
  const tokenFrames = allocate(
    "tdt-token-frames",
    batchSize * maxTokens * WORD_BYTES,
  );
  const tokenDurations = allocate(
    "tdt-token-durations",
    batchSize * maxTokens * WORD_BYTES,
  );
  const metadata = allocate(
    "tdt-metadata",
    batchSize * 2 * WORD_BYTES,
  );
  if (cursor > available.byteOffset + available.byteLength) {
    throw new Error("TDT scratch does not fit the post-encoder activation slot");
  }
  return {
    batchCapacity: batchSize,
    maxTokens,
    decodeStarts,
    decodeEnds,
    flushFinals,
    state,
    predictorScratch,
    tokens,
    tokenFrames,
    tokenDurations,
    metadata,
  };
}

/**
 * Restore the three small decoder-control tables only after the encoder has
 * finished using their aliased activation-arena storage.
 *
 * A queue.writeBuffer directly into these slices is incorrect: queue writes
 * execute before the submitted command buffer, so conformer/projector work
 * overwrites the controls before the decoder can consume them.
 */
export function encodeDecoderControlCopies(
  encoder: GPUCommandEncoder,
  staging: GPUBuffer,
  graph: ParakeetEncoderGraph,
  layout: DecoderArenaLayout,
): void {
  const bytesPerControl = layout.batchCapacity * WORD_BYTES;
  const controls = [
    layout.decodeStarts,
    layout.decodeEnds,
    layout.flushFinals,
  ] as const;
  for (let index = 0; index < controls.length; index += 1) {
    const destination = controls[index]!;
    if (destination.byteLength < bytesPerControl) {
      throw new RangeError(
        `${destination.label} is smaller than the decoder batch`,
      );
    }
    encoder.copyBufferToBuffer(
      staging,
      index * bytesPerControl,
      graph.arena.bufferFor(destination),
      destination.byteOffset,
      bytesPerControl,
    );
  }
}

export function planDecoderReadback(
  layout: DecoderArenaLayout,
  activeBatchSize: number,
): DecoderReadbackLayout {
  if (
    !Number.isSafeInteger(activeBatchSize) ||
    activeBatchSize <= 0 ||
    activeBatchSize > layout.batchCapacity
  ) {
    throw new RangeError(
      `Decoder active batch must be in [1, ${layout.batchCapacity}]`,
    );
  }
  const outputBytes =
    activeBatchSize * layout.maxTokens * WORD_BYTES;
  const metadataBytes = activeBatchSize * 2 * WORD_BYTES;
  let cursor = 0;
  const allocate = (label: string, byteLength: number): ArenaSlice => {
    const result = { label, byteOffset: cursor, byteLength };
    cursor += byteLength;
    return result;
  };
  const tokens = allocate("tdt-readback-tokens", outputBytes);
  const tokenFrames = allocate("tdt-readback-token-frames", outputBytes);
  const tokenDurations = allocate(
    "tdt-readback-token-durations",
    outputBytes,
  );
  const metadata = allocate("tdt-readback-metadata", metadataBytes);
  const copies: readonly DecoderReadbackCopy[] =
    activeBatchSize === layout.batchCapacity
      ? [{
          sourceByteOffset: layout.tokens.byteOffset,
          destinationByteOffset: 0,
          byteLength: cursor,
        }]
      : [
          {
            sourceByteOffset: layout.tokens.byteOffset,
            destinationByteOffset: tokens.byteOffset,
            byteLength: outputBytes,
          },
          {
            sourceByteOffset: layout.tokenFrames.byteOffset,
            destinationByteOffset: tokenFrames.byteOffset,
            byteLength: outputBytes,
          },
          {
            sourceByteOffset: layout.tokenDurations.byteOffset,
            destinationByteOffset: tokenDurations.byteOffset,
            byteLength: outputBytes,
          },
          {
            sourceByteOffset: layout.metadata.byteOffset,
            destinationByteOffset: metadata.byteOffset,
            byteLength: activeBatchSize * WORD_BYTES,
          },
          {
            sourceByteOffset:
              layout.metadata.byteOffset +
              layout.batchCapacity * WORD_BYTES,
            destinationByteOffset:
              metadata.byteOffset + activeBatchSize * WORD_BYTES,
            byteLength: activeBatchSize * WORD_BYTES,
          },
        ];
  return {
    activeBatchSize,
    tokens,
    tokenFrames,
    tokenDurations,
    metadata,
    copies,
    readbackByteLength: cursor,
  };
}

export function encodeDecoderReadbackCopies(
  encoder: GPUCommandEncoder,
  source: GPUBuffer,
  destination: GPUBuffer,
  layout: DecoderReadbackLayout,
  destinationBase = 0,
): void {
  if (
    !Number.isSafeInteger(destinationBase) ||
    destinationBase < 0 ||
    destinationBase % WORD_BYTES !== 0
  ) {
    throw new RangeError("Decoder readback destination base is invalid");
  }
  for (const copy of layout.copies) {
    encoder.copyBufferToBuffer(
      source,
      copy.sourceByteOffset,
      destination,
      destinationBase + copy.destinationByteOffset,
      copy.byteLength,
    );
  }
}

function samePreparedWindow(
  actual: StatelessWindowPlan,
  prepared: StatelessWindowPlan,
): boolean {
  return (
    actual.index === prepared.index &&
    actual.windowSampleCount === prepared.windowSampleCount &&
    actual.logicalStartSample === prepared.logicalStartSample &&
    actual.logicalEndSample === prepared.logicalEndSample &&
    actual.sourceStartSample === prepared.sourceStartSample &&
    actual.sourceEndSample === prepared.sourceEndSample &&
    actual.readSampleCount === prepared.readSampleCount &&
    actual.timestampOffsetSample === prepared.timestampOffsetSample &&
    actual.emissionStartSample === prepared.emissionStartSample &&
    actual.emissionEndSample === prepared.emissionEndSample &&
    actual.decodeStartFrame === prepared.decodeStartFrame &&
    actual.decodeEndFrame === prepared.decodeEndFrame &&
    actual.flushFinal === prepared.flushFinal &&
    actual.validSampleCount === prepared.validSampleCount
  );
}

function throwIfAborted(signal: AbortSignal): void {
  signal.throwIfAborted();
}
