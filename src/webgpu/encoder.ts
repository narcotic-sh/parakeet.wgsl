/// <reference types="@webgpu/types" />

import type { GpuTensor, ParakeetGpuPackage } from "../model/package";
import {
  PARAKEET_FP16_ENCODER_WEIGHT_FORMAT,
  PARAKEET_FFN_SOURCE_WIDTH,
  PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
} from "../model/manifest";
import {
  COMMON_TRANSIENT_PALETTE_MATRIX_COUNT,
} from "../model/runtime-plan";
import {
  align,
  type ActivationArenaBufferPlan,
  type ArenaSlice,
  GpuActivationArena,
} from "./arena";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "./capabilities";
import {
  RELATIVE_ATTENTION_PARAMETER_BYTES,
  RelativeAttentionKernel,
} from "./kernels/attention";
import {
  CONFORMER_CONV_PARAMETER_BYTES,
  FusedConformerConvKernel,
} from "./kernels/conformer-conv";
import {
  F16GemmKernel,
  F32GemmKernel,
  FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
  TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT,
  TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
  transientPaletteParameterPoolBytes,
  type Fp32GemmDescriptor,
  type F16PortableLoadEncoding,
  type GemmDescriptor,
  type PreparedPaletteWeight,
  type PreparedFp32PaletteWeight,
} from "./kernels/gemm";
import {
  LAYERNORM_PARAMETER_BYTES,
  LayerNormKernel,
  type LayerNormDispatch,
} from "./kernels/layernorm";
import {
  DepthwiseSubsamplingKernel,
  FusedSubsamplingConv0Depthwise2Kernel,
  SUBSAMPLING_DEPTHWISE_PARAMETER_BYTES,
  SUBSAMPLING_FUSED_PARAMETER_BYTES,
  SUBSAMPLING_MASK_PARAMETER_BYTES,
  SubsamplingTimeMaskKernel,
  subsamplingOutputDimension,
} from "./kernels/subsampling";
import type { EncoderDispatchShape } from "./kernels/execution-shape";
import { planUniformParameterPool } from "./uniform-parameter-pool";

export const PARAKEET_FEATURE_BINS = 128;
export const PARAKEET_FEATURE_FRAMES = 1501;
export const PARAKEET_ENCODER_FRAMES = 188;
export const PARAKEET_SUBSAMPLING_CHANNELS = 256;
export const PARAKEET_ENCODER_CHANNELS = 1024;
export const PARAKEET_INTERMEDIATE_CHANNELS = PARAKEET_FFN_SOURCE_WIDTH;
export const PARAKEET_PROJECTED_CHANNELS = 640;
export const PARAKEET_ENCODER_LAYERS = 24;
export const PARAKEET_ENCODER_BATCH_CAPACITY = 40;
export const PARAKEET_ENCODER_SUBMISSION_CHUNKS =
  PARAKEET_ENCODER_LAYERS + 2;
export const PARAKEET_TRANSIENT_PALETTE_MATRIX_COUNT =
  PARAKEET_ENCODER_LAYERS * 8 +
  COMMON_TRANSIENT_PALETTE_MATRIX_COUNT;
export const PARAKEET_TRANSIENT_PALETTE_DISPATCHES_PER_GRAPH =
  PARAKEET_TRANSIENT_PALETTE_MATRIX_COUNT;
export const PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH =
  1_167_589_376;
export const PARAKEET_FP32_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH =
  PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH * 2;
export const PARAKEET_SUBSAMPLING_MICROBATCH_SIZE = 3;

const ENCODER_LAYER_LOGICAL_DISPATCHES = 15;
const SUBSAMPLED_HEIGHT_2 = 376;
const SUBSAMPLED_WIDTH_2 = 32;
const SUBSAMPLED_HEIGHT_3 = PARAKEET_ENCODER_FRAMES;
const SUBSAMPLED_WIDTH_3 = 16;
const SUBSAMPLED_FLAT_CHANNELS =
  SUBSAMPLED_WIDTH_3 * PARAKEET_SUBSAMPLING_CHANNELS;
const SUBSAMPLING_FLATTEN_WEIGHT_BYTES =
  SUBSAMPLED_FLAT_CHANNELS *
  PARAKEET_ENCODER_CHANNELS *
  Uint16Array.BYTES_PER_ELEMENT;
const SUBSAMPLING_POINTWISE_WEIGHT_BYTES =
  PARAKEET_SUBSAMPLING_CHANNELS *
  PARAKEET_SUBSAMPLING_CHANNELS *
  Uint16Array.BYTES_PER_ELEMENT;
const SUBSAMPLING_FLATTEN_WEIGHT_OFFSET = 0;
const SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET =
  SUBSAMPLING_FLATTEN_WEIGHT_BYTES;
const SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET =
  SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET +
  SUBSAMPLING_POINTWISE_WEIGHT_BYTES;
if (
  SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET +
    SUBSAMPLING_POINTWISE_WEIGHT_BYTES !==
  TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
) {
  throw new Error(
    "Subsampling palette weights do not fill their transient scratch",
  );
}
const FP32_SUBSAMPLING_FLATTEN_WEIGHT_BYTES =
  SUBSAMPLED_FLAT_CHANNELS *
  PARAKEET_ENCODER_CHANNELS *
  Float32Array.BYTES_PER_ELEMENT;
const FP32_SUBSAMPLING_POINTWISE_WEIGHT_BYTES =
  PARAKEET_SUBSAMPLING_CHANNELS *
  PARAKEET_SUBSAMPLING_CHANNELS *
  Float32Array.BYTES_PER_ELEMENT;
const FP32_SUBSAMPLING_FLATTEN_WEIGHT_OFFSET = 0;
const FP32_SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET =
  FP32_SUBSAMPLING_FLATTEN_WEIGHT_BYTES;
const FP32_SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET =
  FP32_SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET +
  FP32_SUBSAMPLING_POINTWISE_WEIGHT_BYTES;
if (
  FP32_SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET +
    FP32_SUBSAMPLING_POINTWISE_WEIGHT_BYTES !==
  FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
) {
  throw new Error(
    "FP32 subsampling palette weights do not fill their transient scratch",
  );
}
const QKV_CHANNELS = PARAKEET_ENCODER_CHANNELS * 3;
const CONFORMER_POINTWISE_CHANNELS = PARAKEET_ENCODER_CHANNELS * 2;
const RELATIVE_POSITION_ROWS = PARAKEET_ENCODER_FRAMES * 2 - 1;
const FEATURE_ELEMENT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const FP16_ACTIVATION_ELEMENT_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const FP32_ACTIVATION_ELEMENT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const LENGTH_ELEMENT_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const STORAGE_ALIGNMENT = 256;
const LAYER_NORM_EPSILON = 1e-5;

export interface EncoderMicrobatchPlan {
  readonly batchOffset: number;
  readonly batchSize: number;
  /** Direct flatten-project destination within encoder slot zero. */
  readonly output: ArenaSlice;
}

export interface SubsamplingArenaSlices {
  /** Ping aliases for the two depthwise outputs. */
  readonly stage376: ArenaSlice;
  readonly stage188: ArenaSlice;
  /** Pong aliases for the two pointwise outputs. */
  readonly projected376: ArenaSlice;
  readonly projected188: ArenaSlice;
}

export interface ParakeetEncoderArenaPlan {
  readonly precision: ParakeetExecutionProfile["precision"];
  readonly batchSize: typeof PARAKEET_ENCODER_BATCH_CAPACITY;
  readonly rows: number;
  readonly intermediateChannels: typeof PARAKEET_INTERMEDIATE_CHANNELS;
  readonly subsamplingMicrobatchSize:
    typeof PARAKEET_SUBSAMPLING_MICROBATCH_SIZE;
  /** Sum of the fixed physical activation buffers. */
  readonly byteLength: number;
  readonly buffers: readonly ActivationArenaBufferPlan[];
  /** External f32 input, physically `[batch, 128, 1501]` feature-major. */
  readonly featureInput: ArenaSlice;
  /** Raw feature-frame lengths uploaded by the caller. */
  readonly featureLengths: ArenaSlice;
  /** Three-times stride-2 lengths consumed by every attention layer. */
  readonly encoderLengths: ArenaSlice;
  /** Two full `[batch * 188, 1024]` alternating profile-scalar slots. */
  readonly encoderSlots: readonly [ArenaSlice, ArenaSlice];
  /** One `[batch * 188, 4096]` profile-scalar temporary, reused by all layers. */
  readonly bigTemporary: ArenaSlice;
  /** FFN view of `bigTemporary`, sized to this package's physical width. */
  readonly feedForwardTemporary: ArenaSlice;
  /** QKV `[batch * 188, 3072]` in the first three quarters of `bigTemporary`. */
  readonly attentionQkv: ArenaSlice;
  /** Attention context `[batch * 188, 1024]` in the final quarter. */
  readonly attentionContext: ArenaSlice;
  /** Convolution pointwise-1 `[batch * 188, 2048]` in the first half. */
  readonly convolutionPointwise: ArenaSlice;
  /** Fused convolution output `[batch * 188, 1024]` in the third quarter. */
  readonly convolutionOutput: ArenaSlice;
  /** Final `[batch * 188, 640]` profile-scalar projected output. */
  readonly projectedOutput: ArenaSlice;
  /** Dead encoder storage reused for persistent decoder state and outputs. */
  readonly decoderScratch: ArenaSlice;
  readonly subsampling: SubsamplingArenaSlices;
  /**
   * One prebuilt command variant for every active batch end B1-B40.
   *
   * Full B3 commands end at multiples of three. The intervening entries are
   * exact B1/B2 tails, so an arbitrary active prefix never reads or writes an
   * inactive sequence.
   */
  readonly microbatches: readonly EncoderMicrobatchPlan[];
  /** Output slot after each full Conformer layer. */
  readonly layerOutputSlots: readonly number[];
}

export interface EncoderLayerArenaPlan {
  readonly input: ArenaSlice;
  /** The other dedicated full slot at layer entry. */
  readonly alternate: ArenaSlice;
  /** Residual stream after the first FFN. */
  readonly postFeedForward1: ArenaSlice;
  /** Scratch/output slot used by attention and FF2 normalization. */
  readonly moduleScratch: ArenaSlice;
  readonly bigTemporary: ArenaSlice;
  readonly feedForwardTemporary: ArenaSlice;
  readonly attentionQkv: ArenaSlice;
  readonly attentionContext: ArenaSlice;
  readonly convolutionPointwise: ArenaSlice;
  readonly convolutionOutput: ArenaSlice;
  readonly outputSlot: number;
}

export interface EncoderLayerStructure {
  readonly logicalDispatches: number;
  readonly layerNormPasses: number;
  /** Matrix passes for FF1, excluding their transient expansions. */
  readonly feedForward1MatrixPasses: number;
  /** Logical dispatches plus one expansion before each layer GEMM. */
  readonly totalComputePasses: number;
}

export interface SubsamplingDispatchStructure {
  readonly unconditionalPerMicrobatch: number;
  readonly conditionalFinalMasksPerMicrobatch: number;
}

export interface EncoderUniformParameterPoolPlan {
  readonly bufferCount: 6;
  readonly uniqueSlotCount: number;
  readonly byteLength: number;
}

export interface ParakeetEncoderDecoderInput {
  readonly encoderProjected: ArenaSlice;
  readonly encodedLengths: ArenaSlice;
  readonly batchSize: number;
  readonly frames: typeof PARAKEET_ENCODER_FRAMES;
}

export interface ParakeetEncoderExecutionPlan extends EncoderDispatchShape {
  readonly microbatchCount: number;
}

export type EncoderSubmissionChunkPlan =
  | {
      readonly kind: "subsampling";
      readonly label: "parakeet-encoder-subsampling";
      readonly layerIndex: null;
    }
  | {
      readonly kind: "conformer-layer";
      readonly label: string;
      readonly layerIndex: number;
    }
  | {
      readonly kind: "projector";
      readonly label: "parakeet-encoder-projector";
      readonly layerIndex: null;
    };

export type EncoderB1CheckpointKind =
  | "subsampling"
  | "layer"
  | "projector";

/**
 * One tightly packed slice in the diagnostic-only B1 encoder capture.
 *
 * `elementOffset` is precision-independent and therefore also indexes the
 * widened FP32 file emitted by the retained diagnostic page. `byteOffset`
 * indexes the precision-native GPU readback.
 */
export interface EncoderB1Checkpoint {
  readonly label: string;
  readonly kind: EncoderB1CheckpointKind;
  readonly layerIndex: number | null;
  readonly rows: typeof PARAKEET_ENCODER_FRAMES;
  readonly channels:
    | typeof PARAKEET_ENCODER_CHANNELS
    | typeof PARAKEET_PROJECTED_CHANNELS;
  readonly elementOffset: number;
  readonly elementCount: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface EncoderB1CheckpointPlan {
  readonly precision: ParakeetExecutionProfile["precision"];
  readonly elementBytes: 2 | 4;
  readonly elementCount: number;
  readonly byteLength: number;
  readonly widenedFloat32ByteLength: number;
  readonly checkpoints: readonly EncoderB1Checkpoint[];
}

interface ManagedDispatch {
  readonly label: string;
  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    shape?: EncoderDispatchShape,
  ): void;
  destroy(): void;
}

interface SubsamplingCommands {
  readonly batchOffset: number;
  readonly batchSize: number;
  readonly beforeFinalMaskDispatches: readonly ManagedDispatch[];
  readonly finalMaskDispatch: ManagedDispatch;
  readonly afterFinalMaskDispatches: readonly ManagedDispatch[];
}

interface SubsamplingBuild {
  readonly preparedWeights: readonly EncoderPreparedPaletteWeight[];
  readonly commands: readonly SubsamplingCommands[];
}

interface EncoderKernelSet {
  readonly gemm: EncoderGemmBackend;
  readonly layerNorm: LayerNormKernel;
  readonly attention: RelativeAttentionKernel;
  readonly conformerConv: FusedConformerConvKernel;
}

interface ManagedEncoderKernel {
  destroy(): void;
}

interface UniformPooledEncoderKernel extends ManagedEncoderKernel {
  readonly uniformParameterPoolBytes: number;
  readonly uniformParameterPoolUsedSlots: number;
}

type EncoderPreparedPaletteWeight =
  | PreparedPaletteWeight
  | PreparedFp32PaletteWeight;

type EncoderGemmDescriptor =
  Omit<GemmDescriptor, "preparedPaletteWeight"> & {
    readonly preparedPaletteWeight?: EncoderPreparedPaletteWeight;
  };

interface EncoderGemmBackend {
  readonly portableF16LoadEncoding:
    F16PortableLoadEncoding | null;
  readonly transientPaletteParameterBytes: number;
  readonly transientPaletteParameterUsedSlots: number;
  readonly transientPaletteDispatchCount: number;
  readonly transientPaletteExpandedBytes: number;
  createDispatch(descriptor: EncoderGemmDescriptor): ManagedDispatch;
  createPreparedPaletteWeight(
    descriptor: EncoderGemmDescriptor,
    scratchOffset: number,
  ): EncoderPreparedPaletteWeight;
}

function createF16GemmBackend(
  kernel: F16GemmKernel,
): EncoderGemmBackend {
  return {
    get portableF16LoadEncoding() {
      return kernel.portableLoadEncoding;
    },
    get transientPaletteParameterBytes() {
      return kernel.transientPaletteParameterBytes;
    },
    get transientPaletteParameterUsedSlots() {
      return kernel.transientPaletteParameterUsedSlots;
    },
    get transientPaletteDispatchCount() {
      return kernel.transientPaletteDispatchCount;
    },
    get transientPaletteExpandedBytes() {
      return kernel.transientPaletteExpandedBytes;
    },
    createDispatch(descriptor) {
      return kernel.createDispatch(
        descriptor as unknown as GemmDescriptor,
      );
    },
    createPreparedPaletteWeight(descriptor, scratchOffset) {
      return kernel.createPreparedPaletteWeight(
        descriptor as unknown as GemmDescriptor,
        scratchOffset,
      );
    },
  };
}

function createF32GemmBackend(
  kernel: F32GemmKernel,
): EncoderGemmBackend {
  return {
    portableF16LoadEncoding: null,
    get transientPaletteParameterBytes() {
      return kernel.transientPaletteParameterBytes;
    },
    get transientPaletteParameterUsedSlots() {
      return kernel.transientPaletteParameterUsedSlots;
    },
    get transientPaletteDispatchCount() {
      return kernel.transientPaletteDispatchCount;
    },
    get transientPaletteExpandedBytes() {
      return kernel.transientPaletteExpandedBytes;
    },
    createDispatch(descriptor) {
      return kernel.createDispatch(
        descriptor as unknown as Fp32GemmDescriptor,
      );
    },
    createPreparedPaletteWeight(descriptor, scratchOffset) {
      return kernel.createPreparedPaletteWeight(
        descriptor as unknown as Fp32GemmDescriptor,
        scratchOffset,
      );
    },
  };
}

interface TensorSpec {
  readonly dtype: "float16" | "float32" | "uint32";
  readonly logicalShape: readonly number[];
  readonly storageShape: readonly number[];
  readonly layout: string;
}

export function deriveSubsamplingLengths(
  featureLength: number,
): readonly [number, number, number] {
  requirePositiveInteger(featureLength, "featureLength");
  const first = subsamplingOutputDimension(featureLength);
  const second = subsamplingOutputDimension(first);
  const third = subsamplingOutputDimension(second);
  return [first, second, third];
}

export function deriveEncoderValidLength(featureLength: number): number {
  if (featureLength > PARAKEET_FEATURE_FRAMES) {
    throw new RangeError(
      `featureLength must not exceed ${PARAKEET_FEATURE_FRAMES}`,
    );
  }
  return deriveSubsamplingLengths(featureLength)[2];
}

export function deriveEncoderValidLengths(
  featureLengths: readonly number[],
  batchSize = featureLengths.length,
): Uint32Array<ArrayBuffer> {
  requirePositiveInteger(batchSize, "batchSize");
  if (featureLengths.length !== batchSize) {
    throw new RangeError(
      `Expected ${batchSize} feature lengths, got ${featureLengths.length}`,
    );
  }
  const result = new Uint32Array(batchSize);
  for (let index = 0; index < batchSize; index++) {
    const featureLength = featureLengths[index];
    if (featureLength === undefined) {
      throw new RangeError(`Missing feature length ${index}`);
    }
    result[index] = deriveEncoderValidLength(featureLength);
  }
  return result;
}

export function planParakeetEncoderExecution(
  batchSize: number,
): ParakeetEncoderExecutionPlan {
  if (
    !Number.isSafeInteger(batchSize) ||
    batchSize <= 0 ||
    batchSize > PARAKEET_ENCODER_BATCH_CAPACITY
  ) {
    throw new RangeError(
      `Encoder active batch must be in ` +
        `[1, ${PARAKEET_ENCODER_BATCH_CAPACITY}]`,
    );
  }
  return {
    batchSize,
    rows: batchSize * PARAKEET_ENCODER_FRAMES,
    microbatchCount: Math.ceil(
      batchSize / PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
    ),
  };
}

export function subsamplingSliceNeedsFinalMask(
  validEncoderLengths: ArrayLike<number>,
  batchOffset: number,
  batchSize: number,
): boolean {
  if (!Number.isSafeInteger(batchOffset) || batchOffset < 0) {
    throw new RangeError("Subsampling batch offset must be non-negative");
  }
  requirePositiveInteger(batchSize, "subsampling batch size");
  const batchEnd = batchOffset + batchSize;
  if (
    !Number.isSafeInteger(batchEnd) ||
    batchEnd > validEncoderLengths.length
  ) {
    throw new RangeError("Subsampling batch slice exceeds valid lengths");
  }
  let needsMask = false;
  for (let index = batchOffset; index < batchEnd; index++) {
    const validLength = validEncoderLengths[index];
    if (
      validLength === undefined ||
      !Number.isSafeInteger(validLength) ||
      validLength <= 0 ||
      validLength > PARAKEET_ENCODER_FRAMES
    ) {
      throw new RangeError(
        `Encoder valid length ${index} must be in ` +
          `[1, ${PARAKEET_ENCODER_FRAMES}]`,
      );
    }
    needsMask ||= validLength < PARAKEET_ENCODER_FRAMES;
  }
  return needsMask;
}

export function planEncoderLayerArena(
  plan: ParakeetEncoderArenaPlan,
  inputSlot: number,
): EncoderLayerArenaPlan {
  if (
    !Number.isSafeInteger(inputSlot) ||
    inputSlot < 0 ||
    inputSlot >= plan.encoderSlots.length
  ) {
    throw new RangeError("Encoder input slot is out of range");
  }
  const alternateSlot = (inputSlot + 1) % plan.encoderSlots.length;
  const outputSlot = alternateSlot;
  const moduleScratchSlot = (outputSlot + 1) % plan.encoderSlots.length;
  return {
    input: plan.encoderSlots[inputSlot]!,
    alternate: plan.encoderSlots[alternateSlot]!,
    postFeedForward1: plan.encoderSlots[outputSlot]!,
    moduleScratch: plan.encoderSlots[moduleScratchSlot]!,
    bigTemporary: plan.bigTemporary,
    feedForwardTemporary: plan.feedForwardTemporary,
    attentionQkv: plan.attentionQkv,
    attentionContext: plan.attentionContext,
    convolutionPointwise: plan.convolutionPointwise,
    convolutionOutput: plan.convolutionOutput,
    outputSlot,
  };
}

export function planEncoderLayerStructure(): EncoderLayerStructure {
  return {
    logicalDispatches: ENCODER_LAYER_LOGICAL_DISPATCHES,
    layerNormPasses: 5,
    feedForward1MatrixPasses: 2,
    totalComputePasses: 23,
  };
}

export function planEncoderSubmissionChunks():
  readonly EncoderSubmissionChunkPlan[] {
  return ENCODER_SUBMISSION_CHUNK_PLAN;
}

function createEncoderSubmissionChunkPlan():
  readonly EncoderSubmissionChunkPlan[] {
  const chunks: EncoderSubmissionChunkPlan[] = [
    {
      kind: "subsampling",
      label: "parakeet-encoder-subsampling",
      layerIndex: null,
    },
  ];
  for (
    let layerIndex = 0;
    layerIndex < PARAKEET_ENCODER_LAYERS;
    layerIndex += 1
  ) {
    chunks.push({
      kind: "conformer-layer",
      label:
        `parakeet-encoder-layer-${layerIndex.toString().padStart(2, "0")}`,
      layerIndex,
    });
  }
  chunks.push({
    kind: "projector",
    label: "parakeet-encoder-projector",
    layerIndex: null,
  });
  if (chunks.length !== PARAKEET_ENCODER_SUBMISSION_CHUNKS) {
    throw new Error("Encoder submission chunk count changed");
  }
  return Object.freeze(chunks.map((chunk) => Object.freeze(chunk)));
}

const ENCODER_SUBMISSION_CHUNK_PLAN =
  createEncoderSubmissionChunkPlan();

/**
 * Plans the bounded diagnostic capture after subsampling, after every complete
 * Conformer layer, and after the projector. This is deliberately B1-only and
 * is not part of the production encoder schedule.
 */
export function planEncoderB1Checkpoints(
  precision: ParakeetExecutionProfile["precision"],
): EncoderB1CheckpointPlan {
  const elementBytes = precision === "fp16" ? 2 : 4;
  const checkpoints: EncoderB1Checkpoint[] = [];
  let elementOffset = 0;

  const append = (
    label: string,
    kind: EncoderB1CheckpointKind,
    layerIndex: number | null,
    channels:
      | typeof PARAKEET_ENCODER_CHANNELS
      | typeof PARAKEET_PROJECTED_CHANNELS,
  ): void => {
    const elementCount = PARAKEET_ENCODER_FRAMES * channels;
    const byteOffset = elementOffset * elementBytes;
    const byteLength = elementCount * elementBytes;
    checkpoints.push({
      label,
      kind,
      layerIndex,
      rows: PARAKEET_ENCODER_FRAMES,
      channels,
      elementOffset,
      elementCount,
      byteOffset,
      byteLength,
    });
    elementOffset += elementCount;
  };

  append(
    "subsampling",
    "subsampling",
    null,
    PARAKEET_ENCODER_CHANNELS,
  );
  for (
    let layerIndex = 0;
    layerIndex < PARAKEET_ENCODER_LAYERS;
    layerIndex += 1
  ) {
    append(
      `layer-${layerIndex.toString().padStart(2, "0")}`,
      "layer",
      layerIndex,
      PARAKEET_ENCODER_CHANNELS,
    );
  }
  append(
    "projector",
    "projector",
    null,
    PARAKEET_PROJECTED_CHANNELS,
  );

  const byteLength = elementOffset * elementBytes;
  return {
    precision,
    elementBytes,
    elementCount: elementOffset,
    byteLength,
    widenedFloat32ByteLength:
      elementOffset * Float32Array.BYTES_PER_ELEMENT,
    checkpoints,
  };
}

export function planSubsamplingDispatchStructure(): SubsamplingDispatchStructure {
  return {
    unconditionalPerMicrobatch: 5,
    conditionalFinalMasksPerMicrobatch: 1,
  };
}

export function planEncoderUniformParameterPools(
  minUniformBufferOffsetAlignment: number,
): EncoderUniformParameterPoolPlan {
  const microbatchCount = PARAKEET_ENCODER_BATCH_CAPACITY;
  const poolSpecs = [
    [
      SUBSAMPLING_FUSED_PARAMETER_BYTES,
      microbatchCount,
    ],
    [
      SUBSAMPLING_DEPTHWISE_PARAMETER_BYTES,
      microbatchCount,
    ],
    [
      SUBSAMPLING_MASK_PARAMETER_BYTES,
      microbatchCount,
    ],
    [LAYERNORM_PARAMETER_BYTES, 1],
    [RELATIVE_ATTENTION_PARAMETER_BYTES, 1],
    [CONFORMER_CONV_PARAMETER_BYTES, 1],
  ] as const;
  const uniqueSlotCount = poolSpecs.reduce(
    (total, [, capacity]) => total + capacity,
    0,
  );
  return {
    bufferCount: 6,
    uniqueSlotCount,
    byteLength: poolSpecs.reduce(
      (total, [parameterBytes, capacity]) =>
        total +
        planUniformParameterPool(
          parameterBytes,
          capacity,
          minUniformBufferOffsetAlignment,
        ).byteLength,
      0,
    ),
  };
}

/**
 * Plans two mutually exclusive phases in one allocation. Slot zero and both
 * length tables remain live while subsampling fills slot zero. The external
 * feature input and ping-pong subsampling scratch then become encoder slot one
 * and the 4096-channel shared temporary.
 */
export function planParakeetEncoderArena(
  precision: ParakeetExecutionProfile["precision"] = "fp16",
): ParakeetEncoderArenaPlan {
  if (precision === "fp32") {
    return planFp32ParakeetEncoderArena();
  }
  const ACTIVATION_ELEMENT_BYTES = FP16_ACTIVATION_ELEMENT_BYTES;
  const batchSize = PARAKEET_ENCODER_BATCH_CAPACITY;
  const microbatchSize = PARAKEET_SUBSAMPLING_MICROBATCH_SIZE;
  const rows = checkedProduct(
    [batchSize, PARAKEET_ENCODER_FRAMES],
    "encoder rows",
  );
  const encoderSlotBytes = checkedProduct(
    [rows, PARAKEET_ENCODER_CHANNELS, ACTIVATION_ELEMENT_BYTES],
    "encoder slot bytes",
  );
  const bigTemporaryBytes = checkedProduct(
    [rows, PARAKEET_INTERMEDIATE_CHANNELS, ACTIVATION_ELEMENT_BYTES],
    "encoder big temporary bytes",
  );
  const feedForwardTemporaryBytes = checkedProduct(
    [rows, PARAKEET_INTERMEDIATE_CHANNELS, ACTIVATION_ELEMENT_BYTES],
    "encoder feed-forward temporary bytes",
  );
  const projectedOutputBytes = checkedProduct(
    [rows, PARAKEET_PROJECTED_CHANNELS, ACTIVATION_ELEMENT_BYTES],
    "encoder projected output bytes",
  );

  const slot0 = slice("encoder-slot-0", 0, encoderSlotBytes);
  const featureLengths = slice(
    "feature-valid-lengths",
    alignedEnd(slot0),
    batchSize * LENGTH_ELEMENT_BYTES,
  );
  const encoderLengths = slice(
    "encoder-valid-lengths",
    alignedEnd(featureLengths),
    batchSize * LENGTH_ELEMENT_BYTES,
  );
  const scratchBase = alignedEnd(encoderLengths);

  const featureInputBytes = checkedProduct(
    [
      batchSize,
      PARAKEET_FEATURE_BINS,
      PARAKEET_FEATURE_FRAMES,
      FEATURE_ELEMENT_BYTES,
    ],
    "feature input bytes",
  );
  const featureInput = slice(
    "fbank-feature-major",
    scratchBase,
    featureInputBytes,
  );

  const stage376Bytes = activationBytes(
    microbatchSize,
    SUBSAMPLED_HEIGHT_2,
    SUBSAMPLED_WIDTH_2,
    PARAKEET_SUBSAMPLING_CHANNELS * ACTIVATION_ELEMENT_BYTES,
  );
  const stage188Bytes = activationBytes(
    microbatchSize,
    SUBSAMPLED_HEIGHT_3,
    SUBSAMPLED_WIDTH_3,
    PARAKEET_SUBSAMPLING_CHANNELS * ACTIVATION_ELEMENT_BYTES,
  );
  const pingBytes = Math.max(
    stage376Bytes,
    stage188Bytes,
  );
  const pingOffset = alignedEnd(featureInput);

  const projected376Bytes = stage376Bytes;
  const projected188Bytes = stage188Bytes;
  const pongBytes = Math.max(
    projected376Bytes,
    projected188Bytes,
  );
  const pongOffset = align(pingOffset + pingBytes, STORAGE_ALIGNMENT);
  const subsamplingEnd = pongOffset + pongBytes;

  const slot1 = slice("encoder-slot-1", scratchBase, encoderSlotBytes);
  const bigTemporary = slice(
    "encoder-big-temporary",
    alignedEnd(slot1),
    bigTemporaryBytes,
  );
  const feedForwardTemporary = slice(
    "encoder-feed-forward-temporary",
    bigTemporary.byteOffset,
    feedForwardTemporaryBytes,
  );
  const attentionQkv = slice(
    "encoder-attention-qkv",
    bigTemporary.byteOffset,
    encoderSlotBytes * 3,
  );
  const attentionContext = slice(
    "encoder-attention-context",
    bigTemporary.byteOffset + encoderSlotBytes * 3,
    encoderSlotBytes,
  );
  const convolutionPointwise = slice(
    "encoder-convolution-pointwise",
    bigTemporary.byteOffset,
    encoderSlotBytes * 2,
  );
  const convolutionOutput = slice(
    "encoder-convolution-output",
    bigTemporary.byteOffset + encoderSlotBytes * 2,
    encoderSlotBytes,
  );
  const projectedOutput = slice(
    "encoder-projector-output",
    bigTemporary.byteOffset,
    projectedOutputBytes,
  );
  const encoderEnd = bigTemporary.byteOffset + bigTemporary.byteLength;

  const subsampling: SubsamplingArenaSlices = {
    stage376: slice("subsampling-stage-376", pingOffset, stage376Bytes),
    stage188: slice("subsampling-stage-188", pingOffset, stage188Bytes),
    projected376: slice(
      "subsampling-projected-376",
      pongOffset,
      projected376Bytes,
    ),
    projected188: slice(
      "subsampling-projected-188",
      pongOffset,
      projected188Bytes,
    ),
  };

  const perBatchEncoderBytes = checkedProduct(
    [
      PARAKEET_ENCODER_FRAMES,
      PARAKEET_ENCODER_CHANNELS,
      ACTIVATION_ELEMENT_BYTES,
    ],
    "per-batch encoder bytes",
  );
  const microbatches: EncoderMicrobatchPlan[] = [];
  for (let activeBatchEnd = 1; activeBatchEnd <= batchSize; activeBatchEnd++) {
    const tail = activeBatchEnd % microbatchSize;
    const microbatchBatchSize = tail === 0 ? microbatchSize : tail;
    const batchOffset = activeBatchEnd - microbatchBatchSize;
    microbatches.push({
      batchOffset,
      batchSize: microbatchBatchSize,
      output: slice(
        `encoder-slot-0-batch-${batchOffset}-size-${microbatchBatchSize}`,
        slot0.byteOffset + batchOffset * perBatchEncoderBytes,
        microbatchBatchSize * perBatchEncoderBytes,
      ),
    });
  }

  const layerOutputSlots = Array.from(
    { length: PARAKEET_ENCODER_LAYERS },
    (_, layer) => (layer + 1) % 2,
  );
  const byteLength = align(
    Math.max(subsamplingEnd, encoderEnd),
    STORAGE_ALIGNMENT,
  );

  return {
    precision,
    batchSize,
    rows,
    intermediateChannels: PARAKEET_INTERMEDIATE_CHANNELS,
    subsamplingMicrobatchSize: microbatchSize,
    byteLength,
    buffers: [
      {
        label: "parakeet-encoder-arena",
        byteLength,
      },
    ],
    featureInput,
    featureLengths,
    encoderLengths,
    encoderSlots: [slot0, slot1],
    bigTemporary,
    feedForwardTemporary,
    attentionQkv,
    attentionContext,
    convolutionPointwise,
    convolutionOutput,
    projectedOutput,
    decoderScratch: slot0,
    subsampling,
    microbatches,
    layerOutputSlots,
  };
}

/**
 * FP32 keeps B40 while splitting the widened activation set into three
 * lifetime-aliased buffers:
 *
 * 0. encoder slot zero and feature-length storage;
 * 1. feature upload / encoder slot one / projected decoder input;
 * 2. subsampling ping-pong scratch / the 4096-channel encoder temporary.
 *
 * The largest allocation is the 4096-channel temporary and remains below the
 * stock 128 MiB storage-buffer boundary. After the even 24-layer encoder,
 * projector input is slot zero, so its output can safely alias slot one. The
 * decoder then reuses the dead tail of that same buffer, keeping its persistent
 * shader on one bounded arena binding.
 */
function planFp32ParakeetEncoderArena(): ParakeetEncoderArenaPlan {
  const precision = "fp32" as const;
  const activationElementBytes = FP32_ACTIVATION_ELEMENT_BYTES;
  const batchSize = PARAKEET_ENCODER_BATCH_CAPACITY;
  const microbatchSize = PARAKEET_SUBSAMPLING_MICROBATCH_SIZE;
  const rows = checkedProduct(
    [batchSize, PARAKEET_ENCODER_FRAMES],
    "encoder rows",
  );
  const encoderSlotBytes = checkedProduct(
    [rows, PARAKEET_ENCODER_CHANNELS, activationElementBytes],
    "encoder slot bytes",
  );
  const bigTemporaryBytes = checkedProduct(
    [rows, PARAKEET_INTERMEDIATE_CHANNELS, activationElementBytes],
    "encoder big temporary bytes",
  );
  const projectedOutputBytes = checkedProduct(
    [rows, PARAKEET_PROJECTED_CHANNELS, activationElementBytes],
    "encoder projected output bytes",
  );

  const slot0 = slice("encoder-slot-0", 0, encoderSlotBytes, 0);
  const featureLengths = slice(
    "feature-valid-lengths",
    alignedEnd(slot0),
    batchSize * LENGTH_ELEMENT_BYTES,
    0,
  );
  const slotZeroBufferBytes = align(
    featureLengths.byteOffset + featureLengths.byteLength,
    STORAGE_ALIGNMENT,
  );

  const slot1 = slice("encoder-slot-1", 0, encoderSlotBytes, 1);
  const encoderLengths = slice(
    "encoder-valid-lengths",
    alignedEnd(slot1),
    batchSize * LENGTH_ELEMENT_BYTES,
    1,
  );
  const slotOneBufferBytes = align(
    encoderLengths.byteOffset + encoderLengths.byteLength,
    STORAGE_ALIGNMENT,
  );
  const featureInputBytes = checkedProduct(
    [
      batchSize,
      PARAKEET_FEATURE_BINS,
      PARAKEET_FEATURE_FRAMES,
      FEATURE_ELEMENT_BYTES,
    ],
    "feature input bytes",
  );
  const featureInput = slice(
    "fbank-feature-major",
    0,
    featureInputBytes,
    1,
  );

  const stage376Bytes = activationBytes(
    microbatchSize,
    SUBSAMPLED_HEIGHT_2,
    SUBSAMPLED_WIDTH_2,
    PARAKEET_SUBSAMPLING_CHANNELS * activationElementBytes,
  );
  const stage188Bytes = activationBytes(
    microbatchSize,
    SUBSAMPLED_HEIGHT_3,
    SUBSAMPLED_WIDTH_3,
    PARAKEET_SUBSAMPLING_CHANNELS * activationElementBytes,
  );
  const pingBytes = Math.max(stage376Bytes, stage188Bytes);
  const pongBytes = pingBytes;
  const pingOffset = 0;
  const pongOffset = align(pingBytes, STORAGE_ALIGNMENT);
  const subsamplingEnd = pongOffset + pongBytes;

  const bigTemporary = slice(
    "encoder-big-temporary",
    0,
    bigTemporaryBytes,
    2,
  );
  const feedForwardTemporary = slice(
    "encoder-feed-forward-temporary",
    0,
    bigTemporaryBytes,
    2,
  );
  const attentionQkv = slice(
    "encoder-attention-qkv",
    0,
    encoderSlotBytes * 3,
    2,
  );
  const attentionContext = slice(
    "encoder-attention-context",
    encoderSlotBytes * 3,
    encoderSlotBytes,
    2,
  );
  const convolutionPointwise = slice(
    "encoder-convolution-pointwise",
    0,
    encoderSlotBytes * 2,
    2,
  );
  const convolutionOutput = slice(
    "encoder-convolution-output",
    encoderSlotBytes * 2,
    encoderSlotBytes,
    2,
  );
  const temporaryBufferBytes = align(
    Math.max(subsamplingEnd, bigTemporaryBytes),
    STORAGE_ALIGNMENT,
  );

  const projectedOutput = slice(
    "encoder-projector-output",
    0,
    projectedOutputBytes,
    1,
  );
  const decoderScratchOffset = alignedEnd(projectedOutput);
  const decoderScratch = slice(
    "tdt-decoder-scratch",
    decoderScratchOffset,
    encoderSlotBytes - decoderScratchOffset,
    1,
  );

  const subsampling: SubsamplingArenaSlices = {
    stage376: slice(
      "subsampling-stage-376",
      pingOffset,
      stage376Bytes,
      2,
    ),
    stage188: slice(
      "subsampling-stage-188",
      pingOffset,
      stage188Bytes,
      2,
    ),
    projected376: slice(
      "subsampling-projected-376",
      pongOffset,
      stage376Bytes,
      2,
    ),
    projected188: slice(
      "subsampling-projected-188",
      pongOffset,
      stage188Bytes,
      2,
    ),
  };

  const perBatchEncoderBytes = checkedProduct(
    [
      PARAKEET_ENCODER_FRAMES,
      PARAKEET_ENCODER_CHANNELS,
      activationElementBytes,
    ],
    "per-batch encoder bytes",
  );
  const microbatches: EncoderMicrobatchPlan[] = [];
  for (let activeBatchEnd = 1; activeBatchEnd <= batchSize; activeBatchEnd++) {
    const tail = activeBatchEnd % microbatchSize;
    const microbatchBatchSize = tail === 0 ? microbatchSize : tail;
    const batchOffset = activeBatchEnd - microbatchBatchSize;
    microbatches.push({
      batchOffset,
      batchSize: microbatchBatchSize,
      output: slice(
        `encoder-slot-0-batch-${batchOffset}-size-${microbatchBatchSize}`,
        batchOffset * perBatchEncoderBytes,
        microbatchBatchSize * perBatchEncoderBytes,
        0,
      ),
    });
  }

  const layerOutputSlots = Array.from(
    { length: PARAKEET_ENCODER_LAYERS },
    (_, layer) => (layer + 1) % 2,
  );
  if (layerOutputSlots.at(-1) !== 0) {
    throw new Error(
      "FP32 projector aliasing requires the fixed even encoder layer count",
    );
  }
  const buffers = [
    {
      label: "parakeet-encoder-fp32-slot-zero",
      byteLength: slotZeroBufferBytes,
    },
    {
      label: "parakeet-encoder-fp32-slot-one",
      byteLength: slotOneBufferBytes,
    },
    {
      label: "parakeet-encoder-fp32-temporary",
      byteLength: temporaryBufferBytes,
    },
  ] as const satisfies readonly ActivationArenaBufferPlan[];
  const byteLength = buffers.reduce(
    (total, buffer) => total + buffer.byteLength,
    0,
  );

  return {
    precision,
    batchSize,
    rows,
    intermediateChannels: PARAKEET_INTERMEDIATE_CHANNELS,
    subsamplingMicrobatchSize: microbatchSize,
    byteLength,
    buffers,
    featureInput,
    featureLengths,
    encoderLengths,
    encoderSlots: [slot0, slot1],
    bigTemporary,
    feedForwardTemporary,
    attentionQkv,
    attentionContext,
    convolutionPointwise,
    convolutionOutput,
    projectedOutput,
    decoderScratch,
    subsampling,
    microbatches,
    layerOutputSlots,
  };
}

export class ParakeetEncoderGraph {
  readonly arena: GpuActivationArena;
  readonly plan: ParakeetEncoderArenaPlan;
  readonly batchSize: number;
  readonly frames = PARAKEET_ENCODER_FRAMES;
  readonly transientPaletteParameterPoolBytes: number;
  readonly transientPaletteParameterPoolUsedSlots: number;
  readonly transientPaletteMatrixCount: number;
  readonly transientPaletteExpansionDispatches: number;
  readonly transientPaletteExpandedBytesPerGraph: number;
  readonly portableF16GemmLoadEncoding:
    F16PortableLoadEncoding | null;
  readonly uniformParameterPoolBytes: number;
  readonly uniformParameterPoolBufferCount: number;
  readonly uniformParameterPoolUsedSlots: number;

  private destroyed = false;
  private readonly featureInput: ArenaSlice;
  private readonly featureLengths: ArenaSlice;

  private constructor(
    private readonly device: GPUDevice,
    plan: ParakeetEncoderArenaPlan,
    arena: GpuActivationArena,
    private readonly subsamplingPreparedWeights:
      readonly EncoderPreparedPaletteWeight[],
    private readonly subsamplingCommands: readonly SubsamplingCommands[],
    private readonly layerDispatches: readonly ManagedDispatch[],
    private readonly projectorDispatch: ManagedDispatch,
    private readonly allDispatches: readonly ManagedDispatch[],
    private readonly ownedKernels: readonly ManagedEncoderKernel[],
    gemmKernel: EncoderGemmBackend,
    uniformPooledKernels: readonly UniformPooledEncoderKernel[],
  ) {
    this.plan = plan;
    this.arena = arena;
    this.featureInput = plan.featureInput;
    this.featureLengths = plan.featureLengths;
    this.batchSize = plan.batchSize;
    this.transientPaletteParameterPoolBytes =
      gemmKernel.transientPaletteParameterBytes;
    this.transientPaletteParameterPoolUsedSlots =
      gemmKernel.transientPaletteParameterUsedSlots;
    this.transientPaletteMatrixCount =
      PARAKEET_TRANSIENT_PALETTE_MATRIX_COUNT;
    this.transientPaletteExpansionDispatches =
      gemmKernel.transientPaletteDispatchCount;
    this.transientPaletteExpandedBytesPerGraph =
      gemmKernel.transientPaletteExpandedBytes;
    this.portableF16GemmLoadEncoding =
      gemmKernel.portableF16LoadEncoding;
    const expectedExpandedBytes =
      plan.precision === "fp16"
        ? PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH
        : PARAKEET_FP32_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH;
    if (
      this.transientPaletteParameterPoolBytes !==
        transientPaletteParameterPoolBytes(
          device.limits.minUniformBufferOffsetAlignment,
        ) ||
      this.transientPaletteParameterPoolUsedSlots !==
        TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT ||
      this.transientPaletteExpansionDispatches !==
        PARAKEET_TRANSIENT_PALETTE_DISPATCHES_PER_GRAPH ||
      this.transientPaletteExpandedBytesPerGraph !==
        expectedExpandedBytes
    ) {
      throw new Error(
        "Transient palette parameter pool inventory changed",
      );
    }
    const poolPlan = planEncoderUniformParameterPools(
      device.limits.minUniformBufferOffsetAlignment,
    );
    this.uniformParameterPoolBytes =
      uniformPooledKernels.reduce(
        (total, kernel) =>
          total + kernel.uniformParameterPoolBytes,
        0,
      );
    this.uniformParameterPoolBufferCount =
      uniformPooledKernels.length;
    this.uniformParameterPoolUsedSlots =
      uniformPooledKernels.reduce(
        (total, kernel) =>
          total + kernel.uniformParameterPoolUsedSlots,
        0,
      );
    if (
      this.uniformParameterPoolBytes !== poolPlan.byteLength ||
      this.uniformParameterPoolBufferCount !== poolPlan.bufferCount ||
      this.uniformParameterPoolUsedSlots !== poolPlan.uniqueSlotCount
    ) {
      throw new Error(
        "Encoder uniform parameter pool inventory changed",
      );
    }
  }

  static async create(
    device: GPUDevice,
    model: ParakeetGpuPackage,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<ParakeetEncoderGraph> {
    requireEncoderConfig(model, executionProfile);
    const plan = planParakeetEncoderArena(executionProfile.precision);
    if (plan.batchSize > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError("Encoder batch exceeds the WebGPU dispatch limit");
    }
    const arena =
      executionProfile.precision === "fp16"
        ? new GpuActivationArena(
            device,
            plan.byteLength,
            "parakeet-encoder-arena",
          )
        : new GpuActivationArena(device, plan.buffers);
    const allDispatches: ManagedDispatch[] = [];
    const ownedKernels: ManagedEncoderKernel[] = [];
    const uniformPooledKernels: UniformPooledEncoderKernel[] = [];
    try {
      const fusedSubsamplingKernel =
        executionProfile.precision === "fp16"
          ? await FusedSubsamplingConv0Depthwise2Kernel.create(
              device,
              arena,
              plan.microbatches.length,
            )
          : await FusedSubsamplingConv0Depthwise2Kernel.create(
              device,
              arena,
              plan.microbatches.length,
              executionProfile,
            );
      ownedKernels.push(fusedSubsamplingKernel);
      uniformPooledKernels.push(fusedSubsamplingKernel);
      const depthwiseKernel =
        executionProfile.precision === "fp16"
          ? await DepthwiseSubsamplingKernel.create(
              device,
              arena,
              plan.microbatches.length,
            )
          : await DepthwiseSubsamplingKernel.create(
              device,
              arena,
              plan.microbatches.length,
              executionProfile,
            );
      ownedKernels.push(depthwiseKernel);
      uniformPooledKernels.push(depthwiseKernel);
      const subsamplingMaskKernel =
        executionProfile.precision === "fp16"
          ? await SubsamplingTimeMaskKernel.create(
              device,
              arena,
              plan.microbatches.length,
            )
          : await SubsamplingTimeMaskKernel.create(
              device,
              arena,
              plan.microbatches.length,
              executionProfile,
            );
      ownedKernels.push(subsamplingMaskKernel);
      uniformPooledKernels.push(subsamplingMaskKernel);
      const ownedGemmKernel =
        executionProfile.precision === "fp16"
          ? executionProfile.kernelBackend === "subgroups"
            ? await F16GemmKernel.create(device, arena)
            : await F16GemmKernel.create(
                device,
                arena,
                executionProfile,
              )
          : executionProfile.kernelBackend === "subgroups"
            ? await F32GemmKernel.create(device, arena)
            : await F32GemmKernel.create(
                device,
                arena,
                executionProfile,
              );
      const gemmKernel =
        ownedGemmKernel instanceof F16GemmKernel
          ? createF16GemmBackend(ownedGemmKernel)
          : createF32GemmBackend(ownedGemmKernel);
      ownedKernels.push(ownedGemmKernel);
      const layerNormKernel =
        executionProfile.precision === "fp16" &&
        executionProfile.kernelBackend === "subgroups"
          ? await LayerNormKernel.create(
              device,
              arena,
              1,
            )
          : await LayerNormKernel.create(
              device,
              arena,
              1,
              executionProfile,
            );
      ownedKernels.push(layerNormKernel);
      uniformPooledKernels.push(layerNormKernel);
      const attentionKernel =
        executionProfile.precision === "fp16" &&
        executionProfile.kernelBackend === "subgroups"
          ? await RelativeAttentionKernel.create(
              device,
              arena,
              1,
            )
          : await RelativeAttentionKernel.create(
              device,
              arena,
              1,
              executionProfile,
            );
      ownedKernels.push(attentionKernel);
      uniformPooledKernels.push(attentionKernel);
      const conformerConvKernel =
        executionProfile.precision === "fp16"
          ? await FusedConformerConvKernel.create(
              device,
              arena,
              1,
            )
          : await FusedConformerConvKernel.create(
              device,
              arena,
              1,
              executionProfile,
            );
      ownedKernels.push(conformerConvKernel);
      uniformPooledKernels.push(conformerConvKernel);

      const subsampling = createSubsamplingCommands(
        model,
        plan,
        fusedSubsamplingKernel,
        depthwiseKernel,
        subsamplingMaskKernel,
        gemmKernel,
        executionProfile,
      );
      const subsamplingCommands = subsampling.commands;
      for (const prepared of subsampling.preparedWeights) {
        allDispatches.push(prepared.preparation);
      }
      for (const command of subsamplingCommands) {
        allDispatches.push(
          ...command.beforeFinalMaskDispatches,
          command.finalMaskDispatch,
          ...command.afterFinalMaskDispatches,
        );
      }

      const kernels: EncoderKernelSet = {
        gemm: gemmKernel,
        layerNorm: layerNormKernel,
        attention: attentionKernel,
        conformerConv: conformerConvKernel,
      };
      const layerDispatches: ManagedDispatch[] = [];
      let currentSlot = 0;
      for (
        let layerIndex = 0;
        layerIndex < PARAKEET_ENCODER_LAYERS;
        layerIndex++
      ) {
        const built = createEncoderLayerDispatches(
          model,
          plan,
          kernels,
          layerIndex,
          currentSlot,
          executionProfile,
        );
        if (built.outputSlot !== plan.layerOutputSlots[layerIndex]) {
          throw new Error(`Internal encoder slot rotation failed at layer ${layerIndex}`);
        }
        currentSlot = built.outputSlot;
        layerDispatches.push(...built.dispatches);
        allDispatches.push(...built.dispatches);
      }
      const plannedFinalSlot = plan.layerOutputSlots.at(-1) ?? 0;
      if (currentSlot !== plannedFinalSlot) {
        throw new Error("The encoder did not reach its planned final slot");
      }
      const projectorDispatch = gemmKernel.createDispatch({
        label: "encoder-projector",
        a: plan.encoderSlots[currentSlot]!,
        ...matrixTensor(
          model,
          "encoder_projector.weight",
          PARAKEET_ENCODER_CHANNELS,
          PARAKEET_PROJECTED_CHANNELS,
          executionProfile,
        ),
        bias: requiredTensor(model, "encoder_projector.bias", {
          dtype: "float32",
          logicalShape: [PARAKEET_PROJECTED_CHANNELS],
          storageShape: [PARAKEET_PROJECTED_CHANNELS],
          layout: "output-channel",
        }),
        output: plan.projectedOutput,
        rows: plan.rows,
        inner: PARAKEET_ENCODER_CHANNELS,
        columns: PARAKEET_PROJECTED_CHANNELS,
      });
      allDispatches.push(projectorDispatch);

      return new ParakeetEncoderGraph(
        device,
        plan,
        arena,
        subsampling.preparedWeights,
        subsamplingCommands,
        layerDispatches,
        projectorDispatch,
        allDispatches,
        ownedKernels,
        gemmKernel,
        uniformPooledKernels,
      );
    } catch (error) {
      for (const dispatch of allDispatches.toReversed()) {
        dispatch.destroy();
      }
      for (const kernel of ownedKernels.toReversed()) {
        kernel.destroy();
      }
      arena.destroy();
      throw error;
    }
  }

  uploadFeatures(features: Float32Array<ArrayBuffer>): void {
    this.assertAlive();
    const bytesPerBatchItem =
      PARAKEET_FEATURE_BINS *
      PARAKEET_FEATURE_FRAMES *
      FEATURE_ELEMENT_BYTES;
    if (
      features.byteLength <= 0 ||
      features.byteLength > this.featureInput.byteLength ||
      features.byteLength % bytesPerBatchItem !== 0
    ) {
      throw new RangeError(
        `Expected 1-${this.batchSize} complete feature items, got ` +
          `${features.byteLength} bytes`,
      );
    }
    this.arena.upload(this.device, this.featureInput, features);
  }

  uploadValidFeatureLengths(
    featureLengths: readonly number[],
  ): Uint32Array<ArrayBuffer> {
    this.assertAlive();
    const execution = planParakeetEncoderExecution(featureLengths.length);
    const raw = new Uint32Array(execution.batchSize);
    for (let index = 0; index < featureLengths.length; index++) {
      const value = featureLengths[index];
      if (
        value === undefined ||
        !Number.isSafeInteger(value) ||
        value <= 0 ||
        value > PARAKEET_FEATURE_FRAMES
      ) {
        throw new RangeError(
          `Feature length ${index} must be in [1, ${PARAKEET_FEATURE_FRAMES}]`,
        );
      }
      raw[index] = value;
    }
    const derived = deriveEncoderValidLengths(
      featureLengths,
      execution.batchSize,
    );
    this.arena.upload(this.device, this.featureLengths, raw);
    this.arena.upload(this.device, this.plan.encoderLengths, derived);
    return derived;
  }

  private encodeSubsampling(
    encoder: GPUCommandEncoder,
    validEncoderLengths: ArrayLike<number>,
    execution: ParakeetEncoderExecutionPlan,
  ): void {
    this.assertAlive();
    /*
     * These disjoint weight slices remain read-only through every active
     * microbatch. The conformer and projector reuse scratch offset zero only
     * after this loop, so command-buffer and queue order close the lifetime.
     */
    for (const prepared of this.subsamplingPreparedWeights) {
      prepared.preparation.encode(encoder);
    }
    for (
      let index = 0;
      index < execution.microbatchCount;
      index += 1
    ) {
      const activeBatchEnd = Math.min(
        (index + 1) * PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
        execution.batchSize,
      );
      const command = this.subsamplingCommands[activeBatchEnd - 1]!;
      for (const dispatch of command.beforeFinalMaskDispatches) {
        dispatch.encode(encoder);
      }
      if (
        subsamplingSliceNeedsFinalMask(
          validEncoderLengths,
          command.batchOffset,
          command.batchSize,
        )
      ) {
        command.finalMaskDispatch.encode(encoder);
      }
      for (const dispatch of command.afterFinalMaskDispatches) {
        dispatch.encode(encoder);
      }
    }
  }

  private encodeConformer(
    encoder: GPUCommandEncoder,
    execution: ParakeetEncoderExecutionPlan,
  ): void {
    this.assertAlive();
    for (
      let layerIndex = 0;
      layerIndex < PARAKEET_ENCODER_LAYERS;
      layerIndex += 1
    ) {
      this.encodeConformerLayer(encoder, execution, layerIndex);
    }
  }

  private encodeConformerLayer(
    encoder: GPUCommandEncoder,
    execution: ParakeetEncoderExecutionPlan,
    layerIndex: number,
  ): void {
    this.assertAlive();
    if (
      !Number.isSafeInteger(layerIndex) ||
      layerIndex < 0 ||
      layerIndex >= PARAKEET_ENCODER_LAYERS
    ) {
      throw new RangeError("Encoder layer index is out of range");
    }
    const dispatchesPerLayer = ENCODER_LAYER_LOGICAL_DISPATCHES;
    if (
      this.layerDispatches.length !==
      PARAKEET_ENCODER_LAYERS * dispatchesPerLayer
    ) {
      throw new Error("Encoder layer boundaries changed");
    }
    const dispatchStart = layerIndex * dispatchesPerLayer;
    const dispatchEnd = dispatchStart + dispatchesPerLayer;
    for (
      let dispatchIndex = dispatchStart;
      dispatchIndex < dispatchEnd;
      dispatchIndex += 1
    ) {
      this.layerDispatches[dispatchIndex]!.encode(
        encoder,
        undefined,
        execution,
      );
    }
  }

  private encodeProjector(
    encoder: GPUCommandEncoder,
    execution: ParakeetEncoderExecutionPlan,
  ): void {
    this.assertAlive();
    this.projectorDispatch.encode(encoder, undefined, execution);
  }

  encode(
    encoder: GPUCommandEncoder,
    validFeatureLengths: readonly number[],
  ): void {
    const execution =
      planParakeetEncoderExecution(validFeatureLengths.length);
    const validEncoderLengths =
      this.uploadValidFeatureLengths(validFeatureLengths);
    this.encodeSubsampling(
      encoder,
      validEncoderLengths,
      execution,
    );
    this.encodeConformer(encoder, execution);
    this.encodeProjector(encoder, execution);
  }

  /**
   * Records the production encoder as independently submitted scheduling
   * quanta without changing dispatch order or arithmetic.
   *
   * The caller must submit every returned command buffer in order while
   * retaining exclusive ownership of the shared graph and activation arena.
   * It may drain and yield between buffers, but no later batch may upload
   * features, controls, or parameters until this complete sequence and its
   * decoder/readback have finished.
   */
  encodeSubmissionChunks(
    validFeatureLengths: readonly number[],
  ): readonly GPUCommandBuffer[] {
    const execution =
      planParakeetEncoderExecution(validFeatureLengths.length);
    const validEncoderLengths =
      this.uploadValidFeatureLengths(validFeatureLengths);
    return planEncoderSubmissionChunks().map((chunk) => {
      const encoder = this.device.createCommandEncoder({
        label: chunk.label,
      });
      switch (chunk.kind) {
        case "subsampling":
          this.encodeSubsampling(
            encoder,
            validEncoderLengths,
            execution,
          );
          break;
        case "conformer-layer":
          this.encodeConformerLayer(
            encoder,
            execution,
            chunk.layerIndex,
          );
          break;
        case "projector":
          this.encodeProjector(encoder, execution);
          break;
      }
      return encoder.finish({ label: chunk.label });
    });
  }

  /**
   * Diagnostic-only B1 schedule with ordered copies at stable encoder
   * boundaries. The production `encode()` path and its command order remain
   * untouched; every compute dispatch here is one of the same graph-owned
   * dispatch objects used by `encode()`.
   */
  encodeDiagnosticB1Checkpoints(
    encoder: GPUCommandEncoder,
    validFeatureLength: number,
    destination: GPUBuffer,
    destinationByteOffset = 0,
  ): EncoderB1CheckpointPlan {
    this.assertAlive();
    const capture = planEncoderB1Checkpoints(this.plan.precision);
    if (
      !Number.isSafeInteger(destinationByteOffset) ||
      destinationByteOffset < 0 ||
      destinationByteOffset % 4 !== 0
    ) {
      throw new RangeError(
        "Encoder checkpoint destination offset must be a non-negative " +
          "4-byte multiple",
      );
    }
    if (
      (destination.usage & GPUBufferUsage.COPY_DST) === 0 ||
      destinationByteOffset + capture.byteLength > destination.size
    ) {
      throw new RangeError(
        "Encoder checkpoint capture does not fit a COPY_DST buffer",
      );
    }

    const execution = planParakeetEncoderExecution(1);
    const validEncoderLengths =
      this.uploadValidFeatureLengths([validFeatureLength]);
    this.encodeSubsampling(
      encoder,
      validEncoderLengths,
      execution,
    );
    this.copyDiagnosticCheckpoint(
      encoder,
      this.plan.encoderSlots[0]!,
      capture.checkpoints[0]!,
      destination,
      destinationByteOffset,
    );

    for (
      let layerIndex = 0;
      layerIndex < PARAKEET_ENCODER_LAYERS;
      layerIndex += 1
    ) {
      this.encodeConformerLayer(encoder, execution, layerIndex);
      const outputSlot = this.plan.layerOutputSlots[layerIndex]!;
      this.copyDiagnosticCheckpoint(
        encoder,
        this.plan.encoderSlots[outputSlot]!,
        capture.checkpoints[layerIndex + 1]!,
        destination,
        destinationByteOffset,
      );
    }

    this.encodeProjector(encoder, execution);
    this.copyDiagnosticCheckpoint(
      encoder,
      this.plan.projectedOutput,
      capture.checkpoints.at(-1)!,
      destination,
      destinationByteOffset,
    );
    return capture;
  }

  decoderInput(): ParakeetEncoderDecoderInput {
    this.assertAlive();
    return {
      encoderProjected: this.plan.projectedOutput,
      encodedLengths: this.plan.encoderLengths,
      batchSize: this.batchSize,
      frames: this.frames,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const dispatch of this.allDispatches.toReversed()) {
      dispatch.destroy();
    }
    for (const kernel of this.ownedKernels.toReversed()) {
      kernel.destroy();
    }
    this.arena.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Parakeet encoder graph was destroyed");
  }

  private copyDiagnosticCheckpoint(
    encoder: GPUCommandEncoder,
    source: ArenaSlice,
    checkpoint: EncoderB1Checkpoint,
    destination: GPUBuffer,
    destinationByteOffset: number,
  ): void {
    encoder.copyBufferToBuffer(
      this.arena.bufferFor(source),
      source.byteOffset,
      destination,
      destinationByteOffset + checkpoint.byteOffset,
      checkpoint.byteLength,
    );
  }
}

interface EncoderLayerBuild {
  readonly dispatches: readonly ManagedDispatch[];
  readonly outputSlot: number;
}

function createEncoderLayerDispatches(
  model: ParakeetGpuPackage,
  plan: ParakeetEncoderArenaPlan,
  kernels: EncoderKernelSet,
  layerIndex: number,
  inputSlot: number,
  executionProfile: ParakeetExecutionProfile,
): EncoderLayerBuild {
  const prefix = `encoder.layers.${layerIndex}`;
  const layerArena = planEncoderLayerArena(plan, inputSlot);
  const input = layerArena.input;
  const alternate = layerArena.alternate;
  const postFeedForward1 = layerArena.postFeedForward1;
  const moduleScratch = layerArena.moduleScratch;
  const outputSlot = layerArena.outputSlot;
  const rows = plan.rows;
  const feedForward = layerArena.feedForwardTemporary;
  const intermediateChannels = PARAKEET_INTERMEDIATE_CHANNELS;
  const dispatches: ManagedDispatch[] = [
    createNormDispatch(
      model,
      kernels.layerNorm,
      `${prefix}.norm_feed_forward1`,
      input,
      alternate,
      rows,
    ),
    kernels.gemm.createDispatch({
      label: `${prefix}-feed-forward1-linear1`,
      a: alternate,
      ...matrixTensor(
        model,
        `${prefix}.feed_forward1.linear1.weight`,
        PARAKEET_ENCODER_CHANNELS,
        intermediateChannels,
        executionProfile,
      ),
      output: feedForward,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: intermediateChannels,
      activation: "silu",
    }),
    kernels.gemm.createDispatch({
      label: `${prefix}-feed-forward1-linear2-residual`,
      a: feedForward,
      ...matrixTensor(
        model,
        `${prefix}.feed_forward1.linear2.weight`,
        intermediateChannels,
        PARAKEET_ENCODER_CHANNELS,
        executionProfile,
      ),
      output: alternate,
      residual: input,
      residualScale: 0.5,
      rows,
      inner: intermediateChannels,
      columns: PARAKEET_ENCODER_CHANNELS,
    }),
  ];

  dispatches.push(
    createNormDispatch(
      model,
      kernels.layerNorm,
      `${prefix}.norm_self_att`,
      postFeedForward1,
      moduleScratch,
      rows,
    ),
    kernels.gemm.createDispatch({
      label: `${prefix}-self-attention-qkv`,
      a: moduleScratch,
      ...matrixTensor(
        model,
        `${prefix}.self_attn.qkv.weight`,
        PARAKEET_ENCODER_CHANNELS,
        QKV_CHANNELS,
        executionProfile,
      ),
      output: layerArena.attentionQkv,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: QKV_CHANNELS,
    }),
    kernels.attention.createDispatch({
      label: `${prefix}-relative-attention`,
      qkv: layerArena.attentionQkv,
      projectedPositions: requiredTensor(
        model,
        `${prefix}.self_attn.projected_positions`,
        {
          dtype: executionDtype(executionProfile),
          logicalShape: [
            RELATIVE_POSITION_ROWS,
            PARAKEET_ENCODER_CHANNELS,
          ],
          storageShape: [
            8,
            32,
            RELATIVE_POSITION_ROWS,
            4,
          ],
          layout: "head-vector-position-vec4",
        },
      ),
      biasU: requiredTensor(model, `${prefix}.self_attn.bias_u`, {
        dtype: "float32",
        logicalShape: [8, 128],
        storageShape: [8, 128],
        layout: "head-by-channel",
      }),
      biasV: requiredTensor(model, `${prefix}.self_attn.bias_v`, {
        dtype: "float32",
        logicalShape: [8, 128],
        storageShape: [8, 128],
        layout: "head-by-channel",
      }),
      validLengths: plan.encoderLengths,
      output: layerArena.attentionContext,
      batchSize: plan.batchSize,
      frames: PARAKEET_ENCODER_FRAMES,
    }),
    kernels.gemm.createDispatch({
      label: `${prefix}-self-attention-output-residual`,
      a: layerArena.attentionContext,
      ...matrixTensor(
        model,
        `${prefix}.self_attn.o_proj.weight`,
        PARAKEET_ENCODER_CHANNELS,
        PARAKEET_ENCODER_CHANNELS,
        executionProfile,
      ),
      output: moduleScratch,
      residual: postFeedForward1,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: PARAKEET_ENCODER_CHANNELS,
    }),
  );

  dispatches.push(
    createNormDispatch(
      model,
      kernels.layerNorm,
      `${prefix}.norm_conv`,
      moduleScratch,
      postFeedForward1,
      rows,
    ),
    kernels.gemm.createDispatch({
      label: `${prefix}-convolution-pointwise1`,
      a: postFeedForward1,
      ...matrixTensor(
        model,
        `${prefix}.conv.pointwise_conv1.weight`,
        PARAKEET_ENCODER_CHANNELS,
        CONFORMER_POINTWISE_CHANNELS,
        executionProfile,
      ),
      output: layerArena.convolutionPointwise,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: CONFORMER_POINTWISE_CHANNELS,
    }),
    kernels.conformerConv.createDispatch({
      label: `${prefix}-convolution-glu-depthwise-bn-silu`,
      pointwiseInput: layerArena.convolutionPointwise,
      depthwiseWeights: requiredTensor(
        model,
        `${prefix}.conv.depthwise_conv.weight`,
        {
          dtype: executionDtype(executionProfile),
          logicalShape: [9, PARAKEET_ENCODER_CHANNELS],
          storageShape: [9, PARAKEET_ENCODER_CHANNELS / 4, 4],
          layout: "kernel-channel-vec4",
        },
      ),
      batchNormScale: channelTensor(
        model,
        `${prefix}.conv.norm.scale`,
      ),
      batchNormShift: channelTensor(
        model,
        `${prefix}.conv.norm.shift`,
      ),
      validLengths: plan.encoderLengths,
      output: layerArena.convolutionOutput,
      batchSize: plan.batchSize,
      frames: PARAKEET_ENCODER_FRAMES,
    }),
    kernels.gemm.createDispatch({
      label: `${prefix}-convolution-pointwise2-residual`,
      a: layerArena.convolutionOutput,
      ...matrixTensor(
        model,
        `${prefix}.conv.pointwise_conv2.weight`,
        PARAKEET_ENCODER_CHANNELS,
        PARAKEET_ENCODER_CHANNELS,
        executionProfile,
      ),
      output: postFeedForward1,
      residual: moduleScratch,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: PARAKEET_ENCODER_CHANNELS,
    }),
  );

  dispatches.push(
    createNormDispatch(
      model,
      kernels.layerNorm,
      `${prefix}.norm_feed_forward2`,
      postFeedForward1,
      moduleScratch,
      rows,
    ),
    kernels.gemm.createDispatch({
      label: `${prefix}-feed-forward2-linear1`,
      a: moduleScratch,
      ...matrixTensor(
        model,
        `${prefix}.feed_forward2.linear1.weight`,
        PARAKEET_ENCODER_CHANNELS,
        intermediateChannels,
        executionProfile,
      ),
      output: feedForward,
      rows,
      inner: PARAKEET_ENCODER_CHANNELS,
      columns: intermediateChannels,
      activation: "silu",
    }),
    kernels.gemm.createDispatch({
      label: `${prefix}-feed-forward2-linear2-residual`,
      a: feedForward,
      ...matrixTensor(
        model,
        `${prefix}.feed_forward2.linear2.weight`,
        intermediateChannels,
        PARAKEET_ENCODER_CHANNELS,
        executionProfile,
      ),
      output: moduleScratch,
      residual: postFeedForward1,
      residualScale: 0.5,
      rows,
      inner: intermediateChannels,
      columns: PARAKEET_ENCODER_CHANNELS,
    }),
    createNormDispatch(
      model,
      kernels.layerNorm,
      `${prefix}.norm_out`,
      moduleScratch,
      postFeedForward1,
      rows,
    ),
  );

  const structure = planEncoderLayerStructure();
  if (dispatches.length !== structure.logicalDispatches) {
    throw new Error(
      `Internal encoder layer ${layerIndex} dispatch structure mismatch`,
    );
  }
  return { dispatches, outputSlot };
}

function createSubsamplingCommands(
  model: ParakeetGpuPackage,
  plan: ParakeetEncoderArenaPlan,
  fusedSubsamplingKernel: FusedSubsamplingConv0Depthwise2Kernel,
  depthwiseKernel: DepthwiseSubsamplingKernel,
  maskKernel: SubsamplingTimeMaskKernel,
  gemmKernel: EncoderGemmBackend,
  executionProfile: ParakeetExecutionProfile,
): SubsamplingBuild {
  const activationDtype = executionDtype(executionProfile);
  const scratchOffsets =
    executionProfile.precision === "fp16"
      ? {
          flatten: SUBSAMPLING_FLATTEN_WEIGHT_OFFSET,
          pointwise3: SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET,
          pointwise6: SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET,
        }
      : {
          flatten: FP32_SUBSAMPLING_FLATTEN_WEIGHT_OFFSET,
          pointwise3: FP32_SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET,
          pointwise6: FP32_SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET,
        };
  const conv0Weights = requiredTensor(
    model,
    "encoder.subsampling.layers.0.weight",
    {
      dtype: activationDtype,
      logicalShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS],
      storageShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS / 4, 4],
      layout: "kh-kw-output-channel-vec4",
    },
  );
  const conv0Bias = outputBias(
    model,
    "encoder.subsampling.layers.0.bias",
    PARAKEET_SUBSAMPLING_CHANNELS,
  );
  const depthwise2Weights = requiredTensor(
    model,
    "encoder.subsampling.layers.2.weight",
    {
      dtype: activationDtype,
      logicalShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS],
      storageShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS / 4, 4],
      layout: "kh-kw-channel-vec4",
    },
  );
  const depthwise2Bias = outputBias(
    model,
    "encoder.subsampling.layers.2.bias",
    PARAKEET_SUBSAMPLING_CHANNELS,
  );
  const pointwise3Weights = matrixTensor(
    model,
    "encoder.subsampling.layers.3.weight",
    PARAKEET_SUBSAMPLING_CHANNELS,
    PARAKEET_SUBSAMPLING_CHANNELS,
    executionProfile,
  );
  const pointwise3Bias = outputBias(
    model,
    "encoder.subsampling.layers.3.bias",
    PARAKEET_SUBSAMPLING_CHANNELS,
  );
  const depthwise5Weights = requiredTensor(
    model,
    "encoder.subsampling.layers.5.weight",
    {
      dtype: activationDtype,
      logicalShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS],
      storageShape: [3, 3, PARAKEET_SUBSAMPLING_CHANNELS / 4, 4],
      layout: "kh-kw-channel-vec4",
    },
  );
  const depthwise5Bias = outputBias(
    model,
    "encoder.subsampling.layers.5.bias",
    PARAKEET_SUBSAMPLING_CHANNELS,
  );
  const pointwise6Weights = matrixTensor(
    model,
    "encoder.subsampling.layers.6.weight",
    PARAKEET_SUBSAMPLING_CHANNELS,
    PARAKEET_SUBSAMPLING_CHANNELS,
    executionProfile,
  );
  const pointwise6Bias = outputBias(
    model,
    "encoder.subsampling.layers.6.bias",
    PARAKEET_SUBSAMPLING_CHANNELS,
  );
  const flattenWeights = matrixTensor(
    model,
    "encoder.subsampling.linear.weight",
    SUBSAMPLED_FLAT_CHANNELS,
    PARAKEET_ENCODER_CHANNELS,
    executionProfile,
  );
  const flattenBias = outputBias(
    model,
    "encoder.subsampling.linear.bias",
    PARAKEET_ENCODER_CHANNELS,
  );

  const preparedWeights: EncoderPreparedPaletteWeight[] = [];
  const createdDispatches: ManagedDispatch[] = [];
  const ownDispatch = <Dispatch extends ManagedDispatch>(
    dispatch: Dispatch,
  ): Dispatch => {
    createdDispatches.push(dispatch);
    return dispatch;
  };
  try {
    const flattenPrepared =
      gemmKernel.createPreparedPaletteWeight(
        {
          label: "subsampling-flatten-project-shared",
          a: plan.subsampling.projected188,
          ...flattenWeights,
          bias: flattenBias,
          output: plan.microbatches[0]!.output,
          rows: PARAKEET_ENCODER_FRAMES,
          inner: SUBSAMPLED_FLAT_CHANNELS,
          columns: PARAKEET_ENCODER_CHANNELS,
        },
        scratchOffsets.flatten,
      );
    preparedWeights.push(flattenPrepared);
    const pointwise3Prepared =
      gemmKernel.createPreparedPaletteWeight(
        {
          label: "subsampling-pointwise3-shared",
          a: plan.subsampling.stage376,
          ...pointwise3Weights,
          bias: pointwise3Bias,
          output: plan.subsampling.projected376,
          rows: SUBSAMPLED_HEIGHT_2 * SUBSAMPLED_WIDTH_2,
          inner: PARAKEET_SUBSAMPLING_CHANNELS,
          columns: PARAKEET_SUBSAMPLING_CHANNELS,
          activation: "relu",
        },
        scratchOffsets.pointwise3,
      );
    preparedWeights.push(pointwise3Prepared);
    const pointwise6Prepared =
      gemmKernel.createPreparedPaletteWeight(
        {
          label: "subsampling-pointwise6-shared",
          a: plan.subsampling.stage188,
          ...pointwise6Weights,
          bias: pointwise6Bias,
          output: plan.subsampling.projected188,
          rows: SUBSAMPLED_HEIGHT_3 * SUBSAMPLED_WIDTH_3,
          inner: PARAKEET_SUBSAMPLING_CHANNELS,
          columns: PARAKEET_SUBSAMPLING_CHANNELS,
          activation: "relu",
        },
        scratchOffsets.pointwise6,
      );
    preparedWeights.push(pointwise6Prepared);
    const commands = plan.microbatches.map((microbatch) => {
      const batch = microbatch.batchSize;
      const commandLabel =
        `batch-${microbatch.batchOffset}-size-${microbatch.batchSize}`;
      const featureBytesPerBatch =
        PARAKEET_FEATURE_BINS *
        PARAKEET_FEATURE_FRAMES *
        FEATURE_ELEMENT_BYTES;
      const microbatchFeatures = slice(
        `fbank-feature-major-batch-${microbatch.batchOffset}`,
        plan.featureInput.byteOffset +
          microbatch.batchOffset * featureBytesPerBatch,
        batch * featureBytesPerBatch,
        plan.featureInput.bufferIndex ?? 0,
      );
      const beforeFinalMaskDispatches: ManagedDispatch[] = [
        ownDispatch(fusedSubsamplingKernel.createDispatch({
          label: `subsampling-fused-conv0-depthwise2-${commandLabel}`,
          input: microbatchFeatures,
          conv0Weights,
          conv0Bias,
          depthwiseWeights: depthwise2Weights,
          depthwiseBias: depthwise2Bias,
          output: plan.subsampling.stage376,
          validLengths: plan.featureLengths,
          validLengthBatchOffset: microbatch.batchOffset,
          batchSize: batch,
          inputHeight: PARAKEET_FEATURE_FRAMES,
          inputWidth: PARAKEET_FEATURE_BINS,
          channels: PARAKEET_SUBSAMPLING_CHANNELS,
        })),
        ownDispatch(gemmKernel.createDispatch({
          label: `subsampling-pointwise3-${commandLabel}`,
          a: plan.subsampling.stage376,
          ...pointwise3Weights,
          bias: pointwise3Bias,
          output: plan.subsampling.projected376,
          rows: batch * SUBSAMPLED_HEIGHT_2 * SUBSAMPLED_WIDTH_2,
          inner: PARAKEET_SUBSAMPLING_CHANNELS,
          columns: PARAKEET_SUBSAMPLING_CHANNELS,
          activation: "relu",
          preparedPaletteWeight: pointwise3Prepared,
        })),
        ownDispatch(depthwiseKernel.createDispatch({
          label: `subsampling-depthwise5-${commandLabel}`,
          input: plan.subsampling.projected376,
          weights: depthwise5Weights,
          bias: depthwise5Bias,
          output: plan.subsampling.stage188,
          validLengths: plan.featureLengths,
          validLengthBatchOffset: microbatch.batchOffset,
          validLengthSubsamplingSteps: 3,
          batchSize: batch,
          inputHeight: SUBSAMPLED_HEIGHT_2,
          inputWidth: SUBSAMPLED_WIDTH_2,
          channels: PARAKEET_SUBSAMPLING_CHANNELS,
        })),
        ownDispatch(gemmKernel.createDispatch({
          label: `subsampling-pointwise6-${commandLabel}`,
          a: plan.subsampling.stage188,
          ...pointwise6Weights,
          bias: pointwise6Bias,
          output: plan.subsampling.projected188,
          rows: batch * SUBSAMPLED_HEIGHT_3 * SUBSAMPLED_WIDTH_3,
          inner: PARAKEET_SUBSAMPLING_CHANNELS,
          columns: PARAKEET_SUBSAMPLING_CHANNELS,
          activation: "relu",
          preparedPaletteWeight: pointwise6Prepared,
        })),
      ];
      /*
       * Depthwise stage 5 never reads pointwise-3 rows beyond its valid input
       * height and zero-fills its own invalid output rows. The pointwise-3
       * mask is therefore redundant. Pointwise stage 6 can reintroduce
       * nonzero padding through its bias, so its mask remains between that
       * projection and flattening whenever this slice has fewer than 188
       * valid rows.
       */
      const finalMaskDispatch = ownDispatch(maskKernel.createDispatch({
        label: `subsampling-pointwise6-mask-${commandLabel}`,
        values: plan.subsampling.projected188,
        validLengths: plan.featureLengths,
        batchSize: batch,
        height: SUBSAMPLED_HEIGHT_3,
        width: SUBSAMPLED_WIDTH_3,
        channels: PARAKEET_SUBSAMPLING_CHANNELS,
        validLengthBatchOffset: microbatch.batchOffset,
        validLengthSubsamplingSteps: 3,
      }));
      const afterFinalMaskDispatches: ManagedDispatch[] = [
        ownDispatch(gemmKernel.createDispatch({
          label: `subsampling-flatten-project-${commandLabel}`,
          a: plan.subsampling.projected188,
          ...flattenWeights,
          bias: flattenBias,
          output: microbatch.output,
          rows: batch * PARAKEET_ENCODER_FRAMES,
          inner: SUBSAMPLED_FLAT_CHANNELS,
          columns: PARAKEET_ENCODER_CHANNELS,
          preparedPaletteWeight: flattenPrepared,
        })),
      ];
      const structure = planSubsamplingDispatchStructure();
      if (
        beforeFinalMaskDispatches.length +
          afterFinalMaskDispatches.length !==
        structure.unconditionalPerMicrobatch
      ) {
        throw new Error(
          "Internal subsampling dispatch structure mismatch",
        );
      }
      return {
        batchOffset: microbatch.batchOffset,
        batchSize: batch,
        beforeFinalMaskDispatches,
        finalMaskDispatch,
        afterFinalMaskDispatches,
      };
    });
    return { preparedWeights, commands };
  } catch (error) {
    for (const dispatch of createdDispatches.toReversed()) {
      dispatch.destroy();
    }
    for (const prepared of preparedWeights.toReversed()) {
      prepared.preparation.destroy();
    }
    throw error;
  }
}

function createNormDispatch(
  model: ParakeetGpuPackage,
  kernel: LayerNormKernel,
  tensorPrefix: string,
  input: ArenaSlice,
  output: ArenaSlice,
  rows: number,
): LayerNormDispatch {
  return kernel.createDispatch({
    label: tensorPrefix,
    input,
    output,
    gamma: channelTensor(model, `${tensorPrefix}.weight`),
    beta: channelTensor(model, `${tensorPrefix}.bias`),
    rows,
    channels: PARAKEET_ENCODER_CHANNELS,
    epsilon: LAYER_NORM_EPSILON,
  });
}

interface MatrixBindings {
  readonly b: GpuTensor;
  readonly bPalette: GpuTensor;
  readonly paletteGroupColumns: number;
}

function matrixTensor(
  model: ParakeetGpuPackage,
  name: string,
  inner: number,
  columns: number,
  executionProfile: ParakeetExecutionProfile,
): MatrixBindings {
  const bits = name.endsWith(".self_attn.o_proj.weight") ? 6 : 5;
  const codesPerGroup = bits === 5 ? 32 : 16;
  const wordsPerGroup = bits === 5 ? 5 : 3;
  const packedGroups = checkedProduct(
    [inner, columns],
    `${name} palette scalar count`,
  ) / codesPerGroup;
  if (!Number.isSafeInteger(packedGroups)) {
    throw new Error(`${name} is not palette packing aligned`);
  }
  const b = model.tensor(name);
  const packedSpec = {
    dtype: "uint32",
    logicalShape: [inner, columns],
    storageShape: [packedGroups, wordsPerGroup],
    layout: `k-by-output-row-major-palette${bits}-lsb-u32`,
  } as const satisfies TensorSpec;
  requireTensorRecord(
    name,
    b.sourceRecord,
    packedSpec,
    "source",
  );
  requireTensorRecord(
    name,
    b.runtimeRecord,
    packedSpec,
    "runtime",
  );
  const paletteGroups =
    name.endsWith(".self_attn.qkv.weight") ? 3 : 1;
  const paletteEntries = 1 << bits;
  const bPalette = requiredTensor(
    model,
    `${name}.palette`,
    {
      dtype: executionDtype(executionProfile),
      logicalShape: [paletteGroups, paletteEntries],
      storageShape: [paletteGroups, paletteEntries],
      layout: "output-group-by-palette-lut",
    },
  );
  return {
    b,
    bPalette,
    paletteGroupColumns: columns / paletteGroups,
  };
}

function channelTensor(
  model: ParakeetGpuPackage,
  name: string,
): GpuTensor {
  return requiredTensor(model, name, {
    dtype: "float32",
    logicalShape: [PARAKEET_ENCODER_CHANNELS],
    storageShape: [PARAKEET_ENCODER_CHANNELS],
    layout: "channel",
  });
}

function outputBias(
  model: ParakeetGpuPackage,
  name: string,
  channels: number,
): GpuTensor {
  return requiredTensor(model, name, {
    dtype: "float32",
    logicalShape: [channels],
    storageShape: [channels],
    layout: "output-channel",
  });
}

function requiredTensor(
  model: ParakeetGpuPackage,
  name: string,
  spec: TensorSpec,
): GpuTensor {
  const tensor = model.tensor(name);
  requireTensorRecord(name, tensor.sourceRecord, spec, "source");
  requireTensorRecord(name, tensor.runtimeRecord, spec, "runtime");
  return tensor;
}

function requireTensorRecord(
  name: string,
  record: {
    readonly dtype: TensorSpec["dtype"];
    readonly layout: string;
    readonly byteLength: number;
    readonly logicalShape: readonly number[];
    readonly storageShape: readonly number[];
  },
  spec: TensorSpec,
  kind: "source" | "runtime",
): void {
  const expectedByteLength =
    checkedProduct(spec.storageShape, `${name} storage elements`) *
    (spec.dtype === "float16"
      ? Uint16Array.BYTES_PER_ELEMENT
      : spec.dtype === "float32"
        ? Float32Array.BYTES_PER_ELEMENT
        : Uint32Array.BYTES_PER_ELEMENT);
  if (
    record.dtype !== spec.dtype ||
    record.layout !== spec.layout ||
    record.byteLength !== expectedByteLength ||
    !sameShape(record.logicalShape, spec.logicalShape) ||
    !sameShape(record.storageShape, spec.storageShape)
  ) {
    throw new Error(
      `${name} ${kind} must be ${spec.dtype} ${spec.layout}, logical ` +
        `[${spec.logicalShape.join(",")}], storage ` +
        `[${spec.storageShape.join(",")}]`,
    );
  }
}

function requireEncoderConfig(
  model: ParakeetGpuPackage,
  executionProfile: ParakeetExecutionProfile,
): void {
  if (model.precision !== executionProfile.precision) {
    throw new Error(
      `Parakeet ${executionProfile.precision} encoder cannot use ` +
        `${model.precision} model weights`,
    );
  }
  const config = model.manifest.config as unknown as Record<string, unknown>;
  const expected: Readonly<Record<string, number>> = {
    featureBins: PARAKEET_FEATURE_BINS,
    encoderLayers: PARAKEET_ENCODER_LAYERS,
    encoderHiddenSize: PARAKEET_ENCODER_CHANNELS,
    attentionHeads: 8,
    attentionHeadSize: 128,
    convolutionKernelSize: 9,
    decoderHiddenSize: PARAKEET_PROJECTED_CHANNELS,
    subsamplingFactor: 8,
  };
  for (const [name, value] of Object.entries(expected)) {
    if (config[name] !== value) {
      throw new Error(
        `Unsupported Parakeet encoder config ${name}=${String(config[name])}`,
      );
    }
  }
  const expectedWeightFormat =
    executionProfile.precision === "fp16"
      ? PARAKEET_FP16_ENCODER_WEIGHT_FORMAT
      : PARAKEET_FP32_ENCODER_WEIGHT_FORMAT;
  if (
    config.encoderWeightFormat === expectedWeightFormat &&
    config.intermediateSize === PARAKEET_INTERMEDIATE_CHANNELS
  ) {
    return;
  }
  throw new Error("Unsupported Parakeet fixed encoder package contract");
}

function executionDtype(
  executionProfile: ParakeetExecutionProfile,
): "float16" | "float32" {
  return executionProfile.precision === "fp16"
    ? "float16"
    : "float32";
}

function activationBytes(
  batchSize: number,
  height: number,
  width: number,
  channelBytes: number,
): number {
  return checkedProduct(
    [batchSize, height, width, channelBytes],
    "activation bytes",
  );
}

function slice(
  label: string,
  byteOffset: number,
  byteLength: number,
  bufferIndex = 0,
): ArenaSlice {
  if (
    !Number.isSafeInteger(bufferIndex) ||
    bufferIndex < 0 ||
    byteOffset % STORAGE_ALIGNMENT !== 0 ||
    byteLength <= 0 ||
    byteLength % 4 !== 0
  ) {
    throw new RangeError(`Invalid planned arena slice ${label}`);
  }
  return bufferIndex === 0
    ? { label, byteOffset, byteLength }
    : { label, bufferIndex, byteOffset, byteLength };
}

function alignedEnd(value: ArenaSlice): number {
  return align(value.byteOffset + value.byteLength, STORAGE_ALIGNMENT);
}

function checkedProduct(factors: readonly number[], label: string): number {
  const product = factors.reduce((value, factor) => value * factor, 1);
  if (!Number.isSafeInteger(product) || product <= 0) {
    throw new RangeError(`${label} exceeds the safe integer range`);
  }
  return product;
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function sameShape(
  actual: readonly number[],
  expected: readonly number[],
): boolean {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}
