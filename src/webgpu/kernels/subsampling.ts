/// <reference types="@webgpu/types" />

import type { GpuTensor } from "../../model/package";
import type { ArenaSlice } from "../arena";
import { GpuActivationArena } from "../arena";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../capabilities";
import { UniformParameterPool } from "../uniform-parameter-pool";

const KERNEL_SIZE = 3;
const STRIDE = 2;
const PADDING = 1;
const WORKGROUP_X = 8;
const WORKGROUP_Y = 2;
const WORKGROUP_Z = 16;
const WORKGROUP_LANES = WORKGROUP_X * WORKGROUP_Y * WORKGROUP_Z;
const FP32_WORKGROUP_Z = 8;
const FP32_WORKGROUP_LANES =
  WORKGROUP_X * WORKGROUP_Y * FP32_WORKGROUP_Z;
export const SUBSAMPLING_DEPTHWISE_PARAMETER_BYTES = 48;
export const SUBSAMPLING_FUSED_PARAMETER_BYTES = 16;
const FUSED_INPUT_HEIGHT = 1501;
const FUSED_INPUT_WIDTH = 128;
const FUSED_CONV0_HEIGHT = 751;
const FUSED_CONV0_WIDTH = 64;
const FUSED_OUTPUT_HEIGHT = 376;
const FUSED_OUTPUT_WIDTH = 32;
const FUSED_CHANNELS = 256;
const FUSED_CHANNEL_VECTORS = FUSED_CHANNELS / 4;
const FUSED_TILE_WIDTH = WORKGROUP_X * STRIDE + 1;
const FUSED_TILE_HEIGHT = WORKGROUP_Y * STRIDE + 1;
const FUSED_TILE_ELEMENTS =
  FUSED_TILE_WIDTH * FUSED_TILE_HEIGHT * WORKGROUP_Z;
const FP32_FUSED_TILE_ELEMENTS =
  FUSED_TILE_WIDTH * FUSED_TILE_HEIGHT * FP32_WORKGROUP_Z;
const FUSED_WORKGROUP_STORAGE_BYTES =
  FUSED_TILE_ELEMENTS * 4 * Uint16Array.BYTES_PER_ELEMENT +
  2 * Uint32Array.BYTES_PER_ELEMENT;
const FP32_FUSED_WORKGROUP_STORAGE_BYTES =
  FP32_FUSED_TILE_ELEMENTS * 4 * Float32Array.BYTES_PER_ELEMENT +
  2 * Uint32Array.BYTES_PER_ELEMENT;
const MASK_WORKGROUP_SIZE = 256;
export const SUBSAMPLING_MASK_PARAMETER_BYTES = 32;

/** Conv0 weights are transposed from [out, 1, kh, kw] to this packed order. */
export const SUBSAMPLING_CONV0_WEIGHT_LAYOUT =
  "kh-kw-output-channel-vec4";
/** Depthwise weights are transposed from [channel, 1, kh, kw]. */
export const SUBSAMPLING_DEPTHWISE_WEIGHT_LAYOUT =
  "kh-kw-channel-vec4";
export const SUBSAMPLING_BIAS_LAYOUT = "output-channel";

export interface SubsamplingShape {
  readonly batchSize: number;
  /** Time-like spatial dimension. */
  readonly inputHeight: number;
  /** Mel/frequency-like spatial dimension. */
  readonly inputWidth: number;
  readonly channels: number;
}

interface SubsamplingValidLengthDescriptor {
  /** Raw feature-frame lengths for the full encoder batch. */
  readonly validLengths: ArenaSlice;
  /** First sequence in the current subsampling microbatch. */
  readonly validLengthBatchOffset: number;
  /** Number of same-padded stride-2 convolutions through this output. */
  readonly validLengthSubsamplingSteps: number;
}

export interface SubsamplingPlan extends SubsamplingShape {
  readonly outputHeight: number;
  readonly outputWidth: number;
  readonly channelVectors: number;
  readonly inputByteLength: number;
  readonly outputByteLength: number;
  readonly weightByteLength: number;
  readonly biasByteLength: number;
  readonly workgroups: readonly [number, number, number];
}

export interface DepthwiseSubsamplingDescriptor
  extends SubsamplingShape,
    SubsamplingValidLengthDescriptor {
  readonly label: string;
  /** Profile-scalar [batch, inputHeight, inputWidth, channels]. */
  readonly input: ArenaSlice;
  /** Profile-scalar [3, 3, channels], contiguous channel-last. */
  readonly weights: GpuTensor;
  /** f32 [channels]. */
  readonly bias: GpuTensor;
  /** Profile-scalar [batch, outputHeight, outputWidth, channels]. */
  readonly output: ArenaSlice;
}

export interface FusedSubsamplingConv0Depthwise2Descriptor
  extends SubsamplingShape {
  readonly label: string;
  /** Feature-major f32 `[batch, 128, 1501]`. */
  readonly input: ArenaSlice;
  readonly conv0Weights: GpuTensor;
  readonly conv0Bias: GpuTensor;
  readonly depthwiseWeights: GpuTensor;
  readonly depthwiseBias: GpuTensor;
  /** Channel-last profile-scalar `[batch, 376, 32, 256]`. */
  readonly output: ArenaSlice;
  readonly validLengths: ArenaSlice;
  readonly validLengthBatchOffset: number;
}

export interface FusedSubsamplingPlan extends SubsamplingPlan {
  readonly conv0Height: typeof FUSED_CONV0_HEIGHT;
  readonly conv0Width: typeof FUSED_CONV0_WIDTH;
  readonly workgroupStorageBytes: typeof FUSED_WORKGROUP_STORAGE_BYTES;
}

export function subsamplingOutputDimension(input: number): number {
  requirePositiveInteger(input, "subsampling input dimension");
  return Math.floor((input + 2 * PADDING - KERNEL_SIZE) / STRIDE) + 1;
}

export function subsamplingValidLength(
  featureLength: number,
  steps: number,
): number {
  requirePositiveInteger(featureLength, "featureLength");
  requirePositiveInteger(steps, "subsampling steps");
  let length = featureLength;
  for (let step = 0; step < steps; step++) {
    length = subsamplingOutputDimension(length);
  }
  return length;
}

export function planDepthwiseSubsampling(
  shape: SubsamplingShape,
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): SubsamplingPlan {
  return planDepthwise(shape, executionProfile);
}

export function planFusedSubsamplingConv0Depthwise2(
  shape: SubsamplingShape,
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): FusedSubsamplingPlan {
  requirePositiveInteger(shape.batchSize, "batchSize");
  if (
    shape.inputHeight !== FUSED_INPUT_HEIGHT ||
    shape.inputWidth !== FUSED_INPUT_WIDTH ||
    shape.channels !== FUSED_CHANNELS
  ) {
    throw new RangeError(
      "Fused conv0+depthwise2 requires fixed [batch,1501,128,256]",
    );
  }
  const channelVectors = shape.channels / 4;
  const outputHeight = subsamplingOutputDimension(
    subsamplingOutputDimension(shape.inputHeight),
  );
  const outputWidth = subsamplingOutputDimension(
    subsamplingOutputDimension(shape.inputWidth),
  );
  const scalarBytes = executionScalarBytes(executionProfile);
  const workgroupZ = executionProfile.precision === "fp16"
    ? WORKGROUP_Z
    : FP32_WORKGROUP_Z;
  return {
    ...shape,
    conv0Height: FUSED_CONV0_HEIGHT,
    conv0Width: FUSED_CONV0_WIDTH,
    outputHeight,
    outputWidth,
    channelVectors,
    inputByteLength:
      shape.batchSize *
      shape.inputHeight *
      shape.inputWidth *
      Float32Array.BYTES_PER_ELEMENT,
    outputByteLength:
      shape.batchSize *
      outputHeight *
      outputWidth *
      shape.channels *
      scalarBytes,
    weightByteLength:
      KERNEL_SIZE *
      KERNEL_SIZE *
      shape.channels *
      scalarBytes,
    biasByteLength:
      shape.channels * Float32Array.BYTES_PER_ELEMENT,
    workgroups: [
      Math.ceil(outputWidth / WORKGROUP_X),
      Math.ceil(outputHeight / WORKGROUP_Y),
      Math.ceil(
        (shape.batchSize * channelVectors) / workgroupZ,
      ),
    ],
    workgroupStorageBytes:
      fusedSubsamplingWorkgroupStorageBytes(executionProfile),
  };
}

export class SubsamplingDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: ArenaSlice,
    readonly outputHeight: number,
    readonly outputWidth: number,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...this.workgroups);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
  }
}

/**
 * Fixed-shape conv0 + depthwise2 subsampling for one active microbatch.
 *
 * The FP16 profile rounds a 5x17x16 conv0 tile through f16 workgroup storage
 * before the depthwise accumulation. The FP32 profile keeps a 5x17x8 tile in
 * f32. Both avoid materializing the 751x64x256 activation in global memory.
 */
export class FusedSubsamplingConv0Depthwise2Kernel {
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
    private readonly parameterPool: UniformParameterPool,
    private readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  get uniformParameterPoolBytes(): number {
    return this.parameterPool.byteLength;
  }

  get uniformParameterPoolUsedSlots(): number {
    return this.parameterPool.usedSlots;
  }

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    uniformParameterCapacity: number,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<FusedSubsamplingConv0Depthwise2Kernel> {
    requireSubsamplingDevice(device, executionProfile);
    const workgroupStorageBytes =
      fusedSubsamplingWorkgroupStorageBytes(executionProfile);
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      workgroupStorageBytes
    ) {
      throw new Error(
        `Fused subsampling requires ${workgroupStorageBytes} ` +
          "bytes of workgroup storage",
      );
    }
    if (device.limits.maxStorageBuffersPerShaderStage < 7) {
      throw new Error("Fused subsampling requires seven storage bindings");
    }
    const layout = device.createBindGroupLayout({
      label: "parakeet-subsampling-fused-conv0-depthwise2-bindings",
      entries: [
        storageEntry(0, false),
        storageEntry(1, true),
        storageEntry(2, true),
        storageEntry(3, true),
        storageEntry(4, true),
        storageEntry(5, false),
        storageEntry(6, false),
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const label = executionProfile.precision === "fp16"
      ? "parakeet-subsampling-fused-conv0-depthwise2"
      : "parakeet-subsampling-fused-conv0-depthwise2-fp32";
    const pipeline = await createPipeline(
      device,
      layout,
      label,
      fusedSubsamplingWgslForProfile(executionProfile),
    );
    const parameterPool = new UniformParameterPool(
      device,
      SUBSAMPLING_FUSED_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-subsampling-fused-parameter-pool",
    );
    return new FusedSubsamplingConv0Depthwise2Kernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(
    descriptor: FusedSubsamplingConv0Depthwise2Descriptor,
  ): SubsamplingDispatch {
    const plan = planFusedSubsamplingConv0Depthwise2(
      descriptor,
      this.executionProfile,
    );
    requireArenaCapacity(
      descriptor.input,
      plan.inputByteLength,
      "input",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.output,
      plan.outputByteLength,
      "output",
      descriptor.label,
    );
    requireFusedValidLengthDescriptor(descriptor);
    requirePackedTensor(
      descriptor.conv0Weights,
      executionDtype(this.executionProfile),
      [KERNEL_SIZE, KERNEL_SIZE, FUSED_CHANNELS],
      SUBSAMPLING_CONV0_WEIGHT_LAYOUT,
      plan.weightByteLength,
      `${descriptor.label} conv0 weights`,
    );
    requirePackedTensor(
      descriptor.conv0Bias,
      "float32",
      [FUSED_CHANNELS],
      SUBSAMPLING_BIAS_LAYOUT,
      plan.biasByteLength,
      `${descriptor.label} conv0 bias`,
    );
    requirePackedTensor(
      descriptor.depthwiseWeights,
      executionDtype(this.executionProfile),
      [KERNEL_SIZE, KERNEL_SIZE, FUSED_CHANNELS],
      SUBSAMPLING_DEPTHWISE_WEIGHT_LAYOUT,
      plan.weightByteLength,
      `${descriptor.label} depthwise2 weights`,
    );
    requirePackedTensor(
      descriptor.depthwiseBias,
      "float32",
      [FUSED_CHANNELS],
      SUBSAMPLING_BIAS_LAYOUT,
      plan.biasByteLength,
      `${descriptor.label} depthwise2 bias`,
    );
    requireDispatchLimits(this.device, plan.workgroups, descriptor.label);

    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      new Uint32Array([
        descriptor.batchSize,
        descriptor.validLengthBatchOffset,
        0,
        0,
      ]),
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.input) },
        { binding: 1, resource: descriptor.conv0Weights.binding },
        { binding: 2, resource: descriptor.conv0Bias.binding },
        { binding: 3, resource: descriptor.depthwiseWeights.binding },
        { binding: 4, resource: descriptor.depthwiseBias.binding },
        { binding: 5, resource: this.arena.binding(descriptor.output) },
        {
          binding: 6,
          resource: this.arena.binding(descriptor.validLengths),
        },
        { binding: 7, resource: parameterBinding },
      ],
    });
    return new SubsamplingDispatch(
      descriptor.label,
      descriptor.output,
      plan.outputHeight,
      plan.outputWidth,
      this.pipeline,
      bindGroup,
      plan.workgroups,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameterPool.destroy();
  }
}

/**
 * Channel-last 3x3 stride-2 depthwise subsampling convolution.
 *
 * This stage deliberately performs no activation: the following packed
 * pointwise projection/activation remains a separate graph operation.
 */
export class DepthwiseSubsamplingKernel {
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
    private readonly parameterPool: UniformParameterPool,
    private readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  get uniformParameterPoolBytes(): number {
    return this.parameterPool.byteLength;
  }

  get uniformParameterPoolUsedSlots(): number {
    return this.parameterPool.usedSlots;
  }

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    uniformParameterCapacity: number,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<DepthwiseSubsamplingKernel> {
    requireSubsamplingDevice(device, executionProfile);
    const layout = createDepthwiseSubsamplingLayout(
      device,
      "parakeet-subsampling-depthwise",
    );
    const label = executionProfile.precision === "fp16"
      ? "parakeet-subsampling-depthwise"
      : "parakeet-subsampling-depthwise-fp32";
    const pipeline = await createPipeline(
      device,
      layout,
      label,
      depthwiseSubsamplingWgslForProfile(executionProfile),
    );
    const parameterPool = new UniformParameterPool(
      device,
      SUBSAMPLING_DEPTHWISE_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-subsampling-depthwise-parameter-pool",
    );
    return new DepthwiseSubsamplingKernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(
    descriptor: DepthwiseSubsamplingDescriptor,
  ): SubsamplingDispatch {
    const plan = planDepthwiseSubsampling(
      descriptor,
      this.executionProfile,
    );
    requireArenaCapacity(
      descriptor.input,
      plan.inputByteLength,
      "input",
      descriptor.label,
    );
    requireDepthwiseValidLengthDescriptor(descriptor);
    requireArenaCapacity(
      descriptor.output,
      plan.outputByteLength,
      "output",
      descriptor.label,
    );
    requirePackedTensor(
      descriptor.weights,
      executionDtype(this.executionProfile),
      [KERNEL_SIZE, KERNEL_SIZE, descriptor.channels],
      SUBSAMPLING_DEPTHWISE_WEIGHT_LAYOUT,
      plan.weightByteLength,
      `${descriptor.label} weights`,
    );
    requirePackedTensor(
      descriptor.bias,
      "float32",
      [descriptor.channels],
      SUBSAMPLING_BIAS_LAYOUT,
      plan.biasByteLength,
      `${descriptor.label} bias`,
    );
    requireDispatchLimits(this.device, plan.workgroups, descriptor.label);

    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      depthwiseParameterValues(
        descriptor,
        plan,
      ),
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.input) },
        { binding: 1, resource: descriptor.weights.binding },
        { binding: 2, resource: descriptor.bias.binding },
        { binding: 3, resource: this.arena.binding(descriptor.output) },
        {
          binding: 4,
          resource: this.arena.binding(descriptor.validLengths),
        },
        { binding: 5, resource: parameterBinding },
      ],
    });
    return new SubsamplingDispatch(
      descriptor.label,
      descriptor.output,
      plan.outputHeight,
      plan.outputWidth,
      this.pipeline,
      bindGroup,
      plan.workgroups,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameterPool.destroy();
  }
}

export interface SubsamplingTimeMaskDescriptor {
  readonly label: string;
  /** In-place profile-scalar `[batch, height, width, channels]`. */
  readonly values: ArenaSlice;
  /** Raw feature-frame lengths for the full encoder batch. */
  readonly validLengths: ArenaSlice;
  readonly batchSize: number;
  readonly height: number;
  readonly width: number;
  readonly channels: number;
  readonly validLengthBatchOffset: number;
  readonly validLengthSubsamplingSteps: number;
}

export class SubsamplingTimeMaskDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...this.workgroups);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
  }
}

/** Restores the model's time mask after a bias-bearing pointwise Conv2d. */
export class SubsamplingTimeMaskKernel {
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
    private readonly parameterPool: UniformParameterPool,
    private readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  get uniformParameterPoolBytes(): number {
    return this.parameterPool.byteLength;
  }

  get uniformParameterPoolUsedSlots(): number {
    return this.parameterPool.usedSlots;
  }

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    uniformParameterCapacity: number,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<SubsamplingTimeMaskKernel> {
    if (
      executionProfile.precision === "fp16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error("Parakeet subsampling mask requires WebGPU shader-f16");
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < MASK_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < MASK_WORKGROUP_SIZE
    ) {
      throw new Error(
        `Parakeet subsampling mask requires a ${MASK_WORKGROUP_SIZE}-lane workgroup`,
      );
    }
    const layout = device.createBindGroupLayout({
      label: "parakeet-subsampling-time-mask-bindings",
      entries: [
        storageEntry(0, false),
        storageEntry(1, false),
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const label = executionProfile.precision === "fp16"
      ? "parakeet-subsampling-time-mask"
      : "parakeet-subsampling-time-mask-fp32";
    const pipeline = await createPipeline(
      device,
      layout,
      label,
      subsamplingTimeMaskWgslForProfile(executionProfile),
    );
    const parameterPool = new UniformParameterPool(
      device,
      SUBSAMPLING_MASK_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-subsampling-mask-parameter-pool",
    );
    return new SubsamplingTimeMaskKernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(
    descriptor: SubsamplingTimeMaskDescriptor,
  ): SubsamplingTimeMaskDispatch {
    requirePositiveInteger(descriptor.batchSize, "batchSize");
    requirePositiveInteger(descriptor.height, "height");
    requirePositiveInteger(descriptor.width, "width");
    requirePositiveInteger(descriptor.channels, "channels");
    if (descriptor.channels % 4 !== 0) {
      throw new RangeError("Subsampling mask channels must be divisible by four");
    }
    if (
      !Number.isSafeInteger(descriptor.validLengthBatchOffset) ||
      descriptor.validLengthBatchOffset < 0
    ) {
      throw new RangeError(
        `${descriptor.label} valid-length batch offset must be non-negative`,
      );
    }
    requirePositiveInteger(
      descriptor.validLengthSubsamplingSteps,
      `${descriptor.label} valid-length subsampling steps`,
    );
    const channelVectors = descriptor.channels / 4;
    const totalVectors = checkedByteLength(
      [
        descriptor.batchSize,
        descriptor.height,
        descriptor.width,
        channelVectors,
      ],
      "subsampling mask vectors",
    );
    const valuesBytes =
      totalVectors * 4 * executionScalarBytes(this.executionProfile);
    requireArenaCapacity(
      descriptor.values,
      valuesBytes,
      "values",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.validLengths,
      (descriptor.validLengthBatchOffset + descriptor.batchSize) *
        Uint32Array.BYTES_PER_ELEMENT,
      "valid lengths",
      descriptor.label,
    );
    const workgroups = [
      descriptor.height,
      descriptor.batchSize,
      1,
    ] as const;
    requireDispatchLimits(
      this.device,
      workgroups,
      descriptor.label,
    );
    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      new Uint32Array([
        descriptor.batchSize,
        descriptor.height,
        descriptor.width,
        channelVectors,
        descriptor.validLengthBatchOffset,
        descriptor.validLengthSubsamplingSteps,
        0,
        0,
      ]),
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.values) },
        {
          binding: 1,
          resource: this.arena.binding(descriptor.validLengths),
        },
        { binding: 2, resource: parameterBinding },
      ],
    });
    return new SubsamplingTimeMaskDispatch(
      descriptor.label,
      this.pipeline,
      bindGroup,
      workgroups,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameterPool.destroy();
  }
}

function planDepthwise(
  shape: SubsamplingShape,
  executionProfile: ParakeetExecutionProfile,
): SubsamplingPlan {
  requirePositiveInteger(shape.batchSize, "batchSize");
  requirePositiveInteger(shape.inputHeight, "inputHeight");
  requirePositiveInteger(shape.inputWidth, "inputWidth");
  requirePositiveInteger(shape.channels, "channels");
  if (shape.channels % 4 !== 0) {
    throw new RangeError("Subsampling channels must be divisible by four");
  }

  const outputHeight = subsamplingOutputDimension(shape.inputHeight);
  const outputWidth = subsamplingOutputDimension(shape.inputWidth);
  const channelVectors = shape.channels / 4;
  const scalarBytes = executionScalarBytes(executionProfile);
  const inputByteLength = checkedByteLength(
    [
      shape.batchSize,
      shape.inputHeight,
      shape.inputWidth,
      shape.channels,
      scalarBytes,
    ],
    "subsampling input",
  );
  const outputByteLength = checkedByteLength(
    [
      shape.batchSize,
      outputHeight,
      outputWidth,
      shape.channels,
      scalarBytes,
    ],
    "subsampling output",
  );

  return {
    ...shape,
    outputHeight,
    outputWidth,
    channelVectors,
    inputByteLength,
    outputByteLength,
    weightByteLength: checkedByteLength(
      [KERNEL_SIZE, KERNEL_SIZE, shape.channels, scalarBytes],
      "subsampling weights",
    ),
    biasByteLength: checkedByteLength(
      [shape.channels, Float32Array.BYTES_PER_ELEMENT],
      "subsampling bias",
    ),
    workgroups: [
      Math.ceil(outputWidth / WORKGROUP_X),
      Math.ceil(outputHeight / WORKGROUP_Y),
      Math.ceil((shape.batchSize * channelVectors) / WORKGROUP_Z),
    ],
  };
}

function createDepthwiseSubsamplingLayout(
  device: GPUDevice,
  label: string,
): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      storageEntry(0, false),
      storageEntry(1, true),
      storageEntry(2, true),
      storageEntry(3, false),
      storageEntry(4, false),
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
}

async function createPipeline(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  label: string,
  code: string,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(
      `${label} WGSL failed: ${errors.map((message) => message.message).join("; ")}`,
    );
  }
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });
}

function depthwiseParameterValues(
  descriptor: DepthwiseSubsamplingDescriptor,
  plan: SubsamplingPlan,
): Uint32Array<ArrayBuffer> {
  return new Uint32Array([
    plan.batchSize,
    plan.inputHeight,
    plan.inputWidth,
    plan.outputHeight,
    plan.outputWidth,
    plan.channels,
    plan.channelVectors,
    descriptor.validLengthBatchOffset,
    descriptor.validLengthSubsamplingSteps,
    0,
    0,
    0,
  ]);
}

function requireSubsamplingDevice(
  device: GPUDevice,
  executionProfile: ParakeetExecutionProfile,
): void {
  if (
    executionProfile.precision === "fp16" &&
    !device.features.has("shader-f16")
  ) {
    throw new Error("Parakeet subsampling requires WebGPU shader-f16");
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_LANES ||
    device.limits.maxComputeWorkgroupSizeX < WORKGROUP_X ||
    device.limits.maxComputeWorkgroupSizeY < WORKGROUP_Y ||
    device.limits.maxComputeWorkgroupSizeZ < WORKGROUP_Z
  ) {
    throw new Error(
      `Parakeet subsampling requires a ` +
        `${WORKGROUP_X}x${WORKGROUP_Y}x${WORKGROUP_Z} compute workgroup`,
    );
  }
}

function requireDispatchLimits(
  device: GPUDevice,
  workgroups: readonly [number, number, number],
  label: string,
): void {
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  if (workgroups.some((count) => count > maximum)) {
    throw new RangeError(`${label} dispatch exceeds maxComputeWorkgroupsPerDimension`);
  }
}

function requireArenaCapacity(
  slice: ArenaSlice,
  expectedBytes: number,
  role: string,
  label: string,
): void {
  if (slice.byteLength < expectedBytes) {
    throw new RangeError(
      `${label} ${role} needs ${expectedBytes} bytes; arena slice has ${slice.byteLength}`,
    );
  }
}

function requireDepthwiseValidLengthDescriptor(
  descriptor: DepthwiseSubsamplingDescriptor,
): void {
  if (
    !Number.isSafeInteger(descriptor.validLengthBatchOffset) ||
    descriptor.validLengthBatchOffset < 0
  ) {
    throw new RangeError(
      `${descriptor.label} valid-length batch offset must be non-negative`,
    );
  }
  requirePositiveInteger(
    descriptor.validLengthSubsamplingSteps,
    `${descriptor.label} valid-length subsampling steps`,
  );
  const requiredEntries =
    descriptor.validLengthBatchOffset + descriptor.batchSize;
  requireArenaCapacity(
    descriptor.validLengths,
    requiredEntries * Uint32Array.BYTES_PER_ELEMENT,
    "valid lengths",
    descriptor.label,
  );
}

function requireFusedValidLengthDescriptor(
  descriptor: FusedSubsamplingConv0Depthwise2Descriptor,
): void {
  if (
    !Number.isSafeInteger(descriptor.validLengthBatchOffset) ||
    descriptor.validLengthBatchOffset < 0
  ) {
    throw new RangeError(
      `${descriptor.label} valid-length batch offset must be non-negative`,
    );
  }
  requireArenaCapacity(
    descriptor.validLengths,
    (descriptor.validLengthBatchOffset + descriptor.batchSize) *
      Uint32Array.BYTES_PER_ELEMENT,
    "valid lengths",
    descriptor.label,
  );
}

function requirePackedTensor(
  tensor: GpuTensor,
  dtype: "float16" | "float32",
  shape: readonly number[],
  layout: string,
  byteLength: number,
  label: string,
): void {
  const record = tensor.runtimeRecord;
  const storageShape =
    shape.length === 3
      ? [shape[0]!, shape[1]!, shape[2]! / 4, 4]
      : shape;
  if (
    record.dtype !== dtype ||
    record.layout !== layout ||
    record.byteLength !== byteLength ||
    !sameShape(record.logicalShape, shape) ||
    !sameShape(record.storageShape, storageShape)
  ) {
    throw new Error(
      `${label} must be ${dtype} ${layout} logical [${shape.join(",")}], ` +
        `storage [${storageShape.join(",")}], ${byteLength} bytes`,
    );
  }
}

function executionDtype(
  executionProfile: ParakeetExecutionProfile,
): "float16" | "float32" {
  return executionProfile.precision === "fp16" ? "float16" : "float32";
}

function executionScalarBytes(
  executionProfile: ParakeetExecutionProfile,
): 2 | 4 {
  return executionProfile.precision === "fp16" ? 2 : 4;
}

export function fusedSubsamplingWorkgroupStorageBytes(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): number {
  return executionProfile.precision === "fp16"
    ? FUSED_WORKGROUP_STORAGE_BYTES
    : FP32_FUSED_WORKGROUP_STORAGE_BYTES;
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

function checkedByteLength(factors: readonly number[], label: string): number {
  const result = factors.reduce((product, factor) => product * factor, 1);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(`${label} byte length exceeds the safe integer range`);
  }
  return result;
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function storageEntry(
  binding: number,
  readOnly: boolean,
): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: readOnly ? "read-only-storage" : "storage" },
  };
}

const FUSED_CONV0_DEPTHWISE2_WGSL = `
enable f16;

struct Params {
  batch_size: u32,
  valid_length_batch_offset: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> fbank: array<f32>;
@group(0) @binding(1) var<storage, read> conv0_weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> conv0_bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> depthwise_weights: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> depthwise_bias: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(6) var<storage, read_write> valid_lengths: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> conv0_tile: array<vec4<f16>, ${FUSED_TILE_ELEMENTS}>;
var<workgroup> valid_conv0_height: u32;
var<workgroup> valid_output_height: u32;

fn fused_subsampled_length(raw_length: u32, steps: u32) -> u32 {
  var length = raw_length;
  for (var step = 0u; step < steps; step += 1u) {
    length = (length + 1u) / 2u;
  }
  return length;
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, ${WORKGROUP_Z})
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let packed_channel_base = group.z * ${WORKGROUP_Z}u;
  let batch = packed_channel_base / ${FUSED_CHANNEL_VECTORS}u;
  let channel_vector_base =
    packed_channel_base % ${FUSED_CHANNEL_VECTORS}u;

  if (lane == 0u) {
    let raw_length =
      valid_lengths[params.valid_length_batch_offset + batch];
    valid_conv0_height = fused_subsampled_length(raw_length, 1u);
    valid_output_height = fused_subsampled_length(raw_length, 2u);
  }
  workgroupBarrier();
  let input_valid_height = workgroupUniformLoad(&valid_conv0_height);
  let output_valid_height = workgroupUniformLoad(&valid_output_height);

  for (
    var tile_index = lane;
    tile_index < ${FUSED_TILE_ELEMENTS}u;
    tile_index += ${WORKGROUP_LANES}u
  ) {
    let channel_in_group = tile_index % ${WORKGROUP_Z}u;
    let spatial_index = tile_index / ${WORKGROUP_Z}u;
    let tile_x = spatial_index % ${FUSED_TILE_WIDTH}u;
    let tile_y = spatial_index / ${FUSED_TILE_WIDTH}u;
    let conv0_x =
      i32(group.x * ${WORKGROUP_X * STRIDE}u + tile_x) - ${PADDING};
    let conv0_y =
      i32(group.y * ${WORKGROUP_Y * STRIDE}u + tile_y) - ${PADDING};
    let channel_vector = channel_vector_base + channel_in_group;
    var rounded = vec4<f16>(0.0h);
    if (
      batch < params.batch_size &&
      channel_vector < ${FUSED_CHANNEL_VECTORS}u &&
      conv0_x >= 0 &&
      conv0_x < ${FUSED_CONV0_WIDTH} &&
      conv0_y >= 0 &&
      conv0_y < ${FUSED_CONV0_HEIGHT} &&
      conv0_y < i32(input_valid_height)
    ) {
      var sum = conv0_bias[channel_vector];
      for (var kernel_y = 0u; kernel_y < ${KERNEL_SIZE}u; kernel_y += 1u) {
        let input_y =
          conv0_y * ${STRIDE} + i32(kernel_y) - ${PADDING};
        if (input_y < 0 || input_y >= ${FUSED_INPUT_HEIGHT}) {
          continue;
        }
        for (
          var kernel_x = 0u;
          kernel_x < ${KERNEL_SIZE}u;
          kernel_x += 1u
        ) {
          let input_x =
            conv0_x * ${STRIDE} + i32(kernel_x) - ${PADDING};
          if (input_x < 0 || input_x >= ${FUSED_INPUT_WIDTH}) {
            continue;
          }
          let input_index =
            (batch * ${FUSED_INPUT_WIDTH}u + u32(input_x)) *
              ${FUSED_INPUT_HEIGHT}u +
            u32(input_y);
          let weight_index =
            (kernel_y * ${KERNEL_SIZE}u + kernel_x) *
              ${FUSED_CHANNEL_VECTORS}u +
            channel_vector;
          let value = fbank[input_index];
          sum = fma(
            vec4<f32>(conv0_weights[weight_index]),
            vec4<f32>(value),
            sum
          );
        }
      }
      rounded = vec4<f16>(max(sum, vec4<f32>(0.0)));
    }
    conv0_tile[tile_index] = rounded;
  }
  workgroupBarrier();

  let output_x = group.x * ${WORKGROUP_X}u + local_id.x;
  let output_y = group.y * ${WORKGROUP_Y}u + local_id.y;
  let packed_batch_channel = packed_channel_base + local_id.z;
  let output_batch =
    packed_batch_channel / ${FUSED_CHANNEL_VECTORS}u;
  let channel_vector =
    packed_batch_channel % ${FUSED_CHANNEL_VECTORS}u;
  if (
    output_x >= ${FUSED_OUTPUT_WIDTH}u ||
    output_y >= ${FUSED_OUTPUT_HEIGHT}u ||
    output_batch >= params.batch_size
  ) {
    return;
  }
  let output_index =
    (
      (output_batch * ${FUSED_OUTPUT_HEIGHT}u + output_y) *
        ${FUSED_OUTPUT_WIDTH}u +
      output_x
    ) * ${FUSED_CHANNEL_VECTORS}u + channel_vector;
  if (output_y >= output_valid_height) {
    output[output_index] = vec4<f16>(0.0h);
    return;
  }

  var sum = depthwise_bias[channel_vector];
  for (var kernel_y = 0u; kernel_y < ${KERNEL_SIZE}u; kernel_y += 1u) {
    let input_y =
      i32(output_y * ${STRIDE}u + kernel_y) - ${PADDING};
    if (
      input_y < 0 ||
      input_y >= ${FUSED_CONV0_HEIGHT} ||
      input_y >= i32(input_valid_height)
    ) {
      continue;
    }
    for (var kernel_x = 0u; kernel_x < ${KERNEL_SIZE}u; kernel_x += 1u) {
      let input_x =
        i32(output_x * ${STRIDE}u + kernel_x) - ${PADDING};
      if (input_x < 0 || input_x >= ${FUSED_CONV0_WIDTH}) {
        continue;
      }
      let tile_x = local_id.x * ${STRIDE}u + kernel_x;
      let tile_y = local_id.y * ${STRIDE}u + kernel_y;
      let tile_index =
        (tile_y * ${FUSED_TILE_WIDTH}u + tile_x) *
          ${WORKGROUP_Z}u +
        local_id.z;
      let weight_index =
        (kernel_y * ${KERNEL_SIZE}u + kernel_x) *
          ${FUSED_CHANNEL_VECTORS}u +
        channel_vector;
      sum = fma(
        vec4<f32>(conv0_tile[tile_index]),
        vec4<f32>(depthwise_weights[weight_index]),
        sum
      );
    }
  }
  output[output_index] = vec4<f16>(sum);
}
`;

const DEPTHWISE_WGSL = `
enable f16;

struct Params {
  batch_size: u32,
  input_height: u32,
  input_width: u32,
  output_height: u32,
  output_width: u32,
  channels: u32,
  channel_vectors: u32,
  valid_length_batch_offset: u32,
  valid_length_subsampling_steps: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> valid_lengths: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn subsampled_length(raw_length: u32, steps: u32) -> u32 {
  var length = raw_length;
  for (var step = 0u; step < steps; step += 1u) {
    length = (length + 1u) / 2u;
  }
  return length;
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, ${WORKGROUP_Z})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let packed_batch_channel = global_id.z;
  let total_batch_channels = params.batch_size * params.channel_vectors;
  if (
    global_id.x >= params.output_width ||
    global_id.y >= params.output_height ||
    packed_batch_channel >= total_batch_channels
  ) {
    return;
  }

  let batch = packed_batch_channel / params.channel_vectors;
  let channel_vector = packed_batch_channel % params.channel_vectors;
  let raw_valid_length =
    valid_lengths[params.valid_length_batch_offset + batch];
  let valid_input_height = subsampled_length(
    raw_valid_length,
    params.valid_length_subsampling_steps - 1u
  );
  let valid_output_height = subsampled_length(
    raw_valid_length,
    params.valid_length_subsampling_steps
  );
  let output_index =
    (
      (batch * params.output_height + global_id.y) * params.output_width
      + global_id.x
    ) * params.channel_vectors + channel_vector;
  if (global_id.y >= valid_output_height) {
    output[output_index] = vec4<f16>(0.0h);
    return;
  }
  var sum = bias[channel_vector];

  for (var kernel_y = 0u; kernel_y < ${KERNEL_SIZE}u; kernel_y += 1u) {
    let input_y =
      i32(global_id.y * ${STRIDE}u + kernel_y) - ${PADDING};
    if (
      input_y < 0 ||
      input_y >= i32(params.input_height) ||
      input_y >= i32(valid_input_height)
    ) {
      continue;
    }
    for (var kernel_x = 0u; kernel_x < ${KERNEL_SIZE}u; kernel_x += 1u) {
      let input_x =
        i32(global_id.x * ${STRIDE}u + kernel_x) - ${PADDING};
      if (input_x < 0 || input_x >= i32(params.input_width)) {
        continue;
      }
      let input_index =
        (
          (batch * params.input_height + u32(input_y)) * params.input_width
          + u32(input_x)
        ) * params.channel_vectors + channel_vector;
      let weight_index =
        (kernel_y * ${KERNEL_SIZE}u + kernel_x) * params.channel_vectors
        + channel_vector;
      sum = fma(
        vec4<f32>(input[input_index]),
        vec4<f32>(weights[weight_index]),
        sum
      );
    }
  }

  output[output_index] = vec4<f16>(sum);
}
`;

const SUBSAMPLING_TIME_MASK_WGSL = `
enable f16;

struct Params {
  batch_size: u32,
  height: u32,
  width: u32,
  channel_vectors: u32,
  valid_length_batch_offset: u32,
  valid_length_subsampling_steps: u32,
  _pad0: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> values: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> valid_lengths: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn subsampled_length(raw_length: u32, steps: u32) -> u32 {
  var length = raw_length;
  for (var step = 0u; step < steps; step += 1u) {
    length = (length + 1u) / 2u;
  }
  return length;
}

@compute @workgroup_size(${MASK_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let frame = group.x;
  let batch = group.y;
  let vectors_per_frame = params.width * params.channel_vectors;
  let valid_height = subsampled_length(
    valid_lengths[params.valid_length_batch_offset + batch],
    params.valid_length_subsampling_steps
  );
  if (frame < valid_height) {
    return;
  }
  let frame_base =
    (batch * params.height + frame) * vectors_per_frame;
  for (
    var vector = lane;
    vector < vectors_per_frame;
    vector += ${MASK_WORKGROUP_SIZE}u
  ) {
    values[frame_base + vector] = vec4<f16>(0.0h);
  }
}
`;

const FUSED_CONV0_DEPTHWISE2_FP32_WGSL =
  fp32FusedSubsamplingVariant(FUSED_CONV0_DEPTHWISE2_WGSL);
const DEPTHWISE_FP32_WGSL = fp32ShaderVariant(DEPTHWISE_WGSL);
const SUBSAMPLING_TIME_MASK_FP32_WGSL = fp32ShaderVariant(
  SUBSAMPLING_TIME_MASK_WGSL,
);

export function fusedSubsamplingWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  return executionProfile.precision === "fp16"
    ? FUSED_CONV0_DEPTHWISE2_WGSL
    : FUSED_CONV0_DEPTHWISE2_FP32_WGSL;
}

export function depthwiseSubsamplingWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  return executionProfile.precision === "fp16"
    ? DEPTHWISE_WGSL
    : DEPTHWISE_FP32_WGSL;
}

export function subsamplingTimeMaskWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  return executionProfile.precision === "fp16"
    ? SUBSAMPLING_TIME_MASK_WGSL
    : SUBSAMPLING_TIME_MASK_FP32_WGSL;
}

function fp32FusedSubsamplingVariant(fp16Wgsl: string): string {
  let result = fp32ShaderVariant(fp16Wgsl);
  result = replaceRequired(
    result,
    `@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, ${WORKGROUP_Z})`,
    `@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, ${FP32_WORKGROUP_Z})`,
  );
  result = replaceRequired(
    result,
    `var<workgroup> conv0_tile: array<vec4<f32>, ${FUSED_TILE_ELEMENTS}>;`,
    `var<workgroup> conv0_tile: array<vec4<f32>, ${FP32_FUSED_TILE_ELEMENTS}>;`,
  );
  result = replaceRequired(
    result,
    `tile_index < ${FUSED_TILE_ELEMENTS}u`,
    `tile_index < ${FP32_FUSED_TILE_ELEMENTS}u`,
  );
  result = replaceRequired(
    result,
    `tile_index += ${WORKGROUP_LANES}u`,
    `tile_index += ${FP32_WORKGROUP_LANES}u`,
  );
  result = replaceRequired(
    result,
    `let packed_channel_base = group.z * ${WORKGROUP_Z}u;`,
    `let packed_channel_base = group.z * ${FP32_WORKGROUP_Z}u;`,
  );
  result = replaceRequired(
    result,
    `let channel_in_group = tile_index % ${WORKGROUP_Z}u;`,
    `let channel_in_group = tile_index % ${FP32_WORKGROUP_Z}u;`,
  );
  result = replaceRequired(
    result,
    `let spatial_index = tile_index / ${WORKGROUP_Z}u;`,
    `let spatial_index = tile_index / ${FP32_WORKGROUP_Z}u;`,
  );
  return replaceRequired(
    result,
    `) *
          ${WORKGROUP_Z}u +
        local_id.z;`,
    `) *
          ${FP32_WORKGROUP_Z}u +
        local_id.z;`,
  );
}

function fp32ShaderVariant(fp16Wgsl: string): string {
  return fp16Wgsl
    .replace("enable f16;\n", "")
    .replaceAll("f16", "f32")
    .replaceAll("0.0h", "0.0");
}

function replaceRequired(
  source: string,
  search: string,
  replacement: string,
): string {
  if (!source.includes(search)) {
    throw new Error(`Unable to construct FP32 subsampling shader: ${search}`);
  }
  return source.replace(search, replacement);
}
