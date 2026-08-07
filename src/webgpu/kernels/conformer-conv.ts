/// <reference types="@webgpu/types" />

import type { GpuTensor } from "../../model/package";
import type { ArenaSlice } from "../arena";
import { GpuActivationArena } from "../arena";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../capabilities";
import { UniformParameterPool } from "../uniform-parameter-pool";
import type { EncoderDispatchShape } from "./execution-shape";

export const CONFORMER_CONV_CHANNELS = 1024;
export const CONFORMER_CONV_POINTWISE_CHANNELS = 2048;
export const CONFORMER_CONV_KERNEL_SIZE = 9;
export const CONFORMER_DEPTHWISE_WEIGHT_LAYOUT =
  "kernel-channel-vec4";
export const CONFORMER_FOLDED_BATCH_NORM_LAYOUT = "channel";

const CHANNEL_VECTOR_TILE = 8;
const FRAME_TILE = 16;
const WORKGROUP_LANES = CHANNEL_VECTOR_TILE * FRAME_TILE;
const CHANNEL_VECTORS = CONFORMER_CONV_CHANNELS / 4;
const INPUT_VECTORS_PER_ROW = CONFORMER_CONV_POINTWISE_CHANNELS / 4;
const KERNEL_RADIUS = Math.floor(CONFORMER_CONV_KERNEL_SIZE / 2);
const SHARED_FRAMES = FRAME_TILE + CONFORMER_CONV_KERNEL_SIZE - 1;
const SHARED_VALUES = SHARED_FRAMES * CHANNEL_VECTOR_TILE;
const SHARED_BYTES = SHARED_VALUES * 4 * Float32Array.BYTES_PER_ELEMENT;
export const CONFORMER_CONV_PARAMETER_BYTES = 16;

export interface ConformerConvShape {
  readonly batchSize: number;
  readonly frames: number;
}

export interface ConformerConvPlan extends ConformerConvShape {
  readonly rows: number;
  readonly inputByteLength: number;
  readonly outputByteLength: number;
  readonly validLengthsByteLength: number;
  readonly depthwiseWeightByteLength: number;
  readonly affineByteLength: number;
  readonly workgroups: readonly [number, number, number];
  readonly workgroupStorageBytes: number;
}

export interface FusedConformerConvDescriptor extends ConformerConvShape {
  readonly label: string;
  /**
   * Profile-scalar [rows, 2048]. Channels [0,1024) are GLU values
   * and [1024,2048) are GLU gates.
   */
  readonly pointwiseInput: ArenaSlice;
  /** Profile-scalar [9, 1024], contiguous channel-last. */
  readonly depthwiseWeights: GpuTensor;
  /** f32 [1024], gamma / sqrt(runningVariance + epsilon). */
  readonly batchNormScale: GpuTensor;
  /** f32 [1024], beta - runningMean * scale, including any folded conv bias. */
  readonly batchNormShift: GpuTensor;
  /** u32 valid encoder-frame lengths, one per batch item. */
  readonly validLengths: ArenaSlice;
  /** Profile-scalar [rows, 1024]. */
  readonly output: ArenaSlice;
}

export interface ConformerConvDispatchResult {
  readonly output: ArenaSlice;
  readonly rows: number;
  readonly channels: typeof CONFORMER_CONV_CHANNELS;
}

export function planFusedConformerConv(
  shape: ConformerConvShape,
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): ConformerConvPlan {
  requirePositiveInteger(shape.batchSize, "batchSize");
  requirePositiveInteger(shape.frames, "frames");
  const rows = checkedProduct([shape.batchSize, shape.frames], "Conformer rows");
  const scalarBytes = executionScalarBytes(executionProfile);
  return {
    ...shape,
    rows,
    inputByteLength: checkedProduct(
      [rows, CONFORMER_CONV_POINTWISE_CHANNELS, scalarBytes],
      "Conformer pointwise input bytes",
    ),
    outputByteLength: checkedProduct(
      [rows, CONFORMER_CONV_CHANNELS, scalarBytes],
      "Conformer convolution output bytes",
    ),
    validLengthsByteLength: checkedProduct(
      [shape.batchSize, Uint32Array.BYTES_PER_ELEMENT],
      "Conformer valid-length bytes",
    ),
    depthwiseWeightByteLength: checkedProduct(
      [
        CONFORMER_CONV_KERNEL_SIZE,
        CONFORMER_CONV_CHANNELS,
        scalarBytes,
      ],
      "Conformer depthwise weight bytes",
    ),
    affineByteLength: checkedProduct(
      [CONFORMER_CONV_CHANNELS, Float32Array.BYTES_PER_ELEMENT],
      "Conformer folded batch norm bytes",
    ),
    workgroups: [
      Math.ceil(CHANNEL_VECTORS / CHANNEL_VECTOR_TILE),
      Math.ceil(shape.frames / FRAME_TILE),
      shape.batchSize,
    ],
    workgroupStorageBytes: SHARED_BYTES,
  };
}

export class FusedConformerConvDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly result: ConformerConvDispatchResult,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    shape?: EncoderDispatchShape,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    const activeBatchSize = shape?.batchSize ?? this.workgroups[2];
    if (
      !Number.isSafeInteger(activeBatchSize) ||
      activeBatchSize <= 0 ||
      activeBatchSize > this.workgroups[2]
    ) {
      throw new RangeError(
        `${this.label} active batch must be in [1, ${this.workgroups[2]}]`,
      );
    }
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      this.workgroups[0],
      this.workgroups[1],
      activeBatchSize,
    );
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
  }
}

/**
 * Fuses the Conformer convolution module's GLU, same-padded depthwise k9,
 * folded batch normalization, and SiLU.
 *
 * A 128-lane workgroup owns 16 frames by eight vec4 channel groups. It first
 * computes a 24-frame GLU tile (including the four-frame halo on each side)
 * into 3 KiB of f32 workgroup memory. This avoids recomputing sigmoid for every
 * one of the nine overlapping depthwise taps.
 */
export class FusedConformerConvKernel {
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
  ): Promise<FusedConformerConvKernel> {
    requireConformerDevice(device, executionProfile);
    const label = executionProfile.precision === "fp16"
      ? "parakeet-conformer-conv"
      : "parakeet-conformer-conv-fp32";
    const layout = device.createBindGroupLayout({
      label: "parakeet-conformer-conv-bindings",
      entries: [
        storageEntry(0, false),
        storageEntry(1, true),
        storageEntry(2, true),
        storageEntry(3, true),
        storageEntry(4, false),
        storageEntry(5, false),
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const module = device.createShaderModule({
      label,
      code: conformerConvWgslForProfile(executionProfile),
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Conformer convolution WGSL failed: ${errors
          .map((message) => message.message)
          .join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    const parameterPool = new UniformParameterPool(
      device,
      CONFORMER_CONV_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-conformer-conv-parameter-pool",
    );
    return new FusedConformerConvKernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(
    descriptor: FusedConformerConvDescriptor,
  ): FusedConformerConvDispatch {
    const plan = planFusedConformerConv(
      descriptor,
      this.executionProfile,
    );
    requireArenaCapacity(
      descriptor.pointwiseInput,
      plan.inputByteLength,
      "pointwise input",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.output,
      plan.outputByteLength,
      "output",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.validLengths,
      plan.validLengthsByteLength,
      "valid lengths",
      descriptor.label,
    );
    requirePackedTensor(
      descriptor.depthwiseWeights,
      this.executionProfile.precision === "fp16"
        ? "float16"
        : "float32",
      [CONFORMER_CONV_KERNEL_SIZE, CONFORMER_CONV_CHANNELS],
      CONFORMER_DEPTHWISE_WEIGHT_LAYOUT,
      plan.depthwiseWeightByteLength,
      `${descriptor.label} depthwise weights`,
    );
    requirePackedTensor(
      descriptor.batchNormScale,
      "float32",
      [CONFORMER_CONV_CHANNELS],
      CONFORMER_FOLDED_BATCH_NORM_LAYOUT,
      plan.affineByteLength,
      `${descriptor.label} folded batch norm scale`,
    );
    requirePackedTensor(
      descriptor.batchNormShift,
      "float32",
      [CONFORMER_CONV_CHANNELS],
      CONFORMER_FOLDED_BATCH_NORM_LAYOUT,
      plan.affineByteLength,
      `${descriptor.label} folded batch norm shift`,
    );
    requireDispatchLimits(this.device, plan.workgroups, descriptor.label);

    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      new Uint32Array([
        descriptor.batchSize,
        descriptor.frames,
        CHANNEL_VECTORS,
        INPUT_VECTORS_PER_ROW,
      ]),
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.pointwiseInput) },
        { binding: 1, resource: descriptor.depthwiseWeights.binding },
        { binding: 2, resource: descriptor.batchNormScale.binding },
        { binding: 3, resource: descriptor.batchNormShift.binding },
        { binding: 4, resource: this.arena.binding(descriptor.output) },
        {
          binding: 5,
          resource: this.arena.binding(descriptor.validLengths),
        },
        { binding: 6, resource: parameterBinding },
      ],
    });
    return new FusedConformerConvDispatch(
      descriptor.label,
      {
        output: descriptor.output,
        rows: plan.rows,
        channels: CONFORMER_CONV_CHANNELS,
      },
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

function requireConformerDevice(
  device: GPUDevice,
  executionProfile: ParakeetExecutionProfile,
): void {
  if (
    executionProfile.precision === "fp16" &&
    !device.features.has("shader-f16")
  ) {
    throw new Error("Parakeet Conformer convolution requires WebGPU shader-f16");
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_LANES ||
    device.limits.maxComputeWorkgroupSizeX < CHANNEL_VECTOR_TILE ||
    device.limits.maxComputeWorkgroupSizeY < FRAME_TILE
  ) {
    throw new Error(
      `Parakeet Conformer convolution requires a ${CHANNEL_VECTOR_TILE}x${FRAME_TILE} workgroup`,
    );
  }
  if (device.limits.maxComputeWorkgroupStorageSize < SHARED_BYTES) {
    throw new Error(
      `Parakeet Conformer convolution requires ${SHARED_BYTES} workgroup bytes`,
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
    shape.length === 2
      ? [shape[0]!, shape[1]! / 4, 4]
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

function executionScalarBytes(
  executionProfile: ParakeetExecutionProfile,
): 2 | 4 {
  return executionProfile.precision === "fp16" ? 2 : 4;
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

const CONFORMER_CONV_WGSL = `
enable f16;

struct Params {
  batch_size: u32,
  frames: u32,
  channel_vectors: u32,
  input_vectors_per_row: u32,
};

@group(0) @binding(0) var<storage, read_write> pointwise_input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> depthwise_weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> batch_norm_scale: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> batch_norm_shift: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> valid_lengths: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> glu_tile: array<vec4<f32>, ${SHARED_VALUES}>;

fn sigmoid(value: vec4<f32>) -> vec4<f32> {
  let bounded = clamp(value, vec4<f32>(-20.0), vec4<f32>(20.0));
  return vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-bounded));
}

@compute @workgroup_size(${CHANNEL_VECTOR_TILE}, ${FRAME_TILE}, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let channel_vector_base = group.x * ${CHANNEL_VECTOR_TILE}u;
  let frame_base = group.y * ${FRAME_TILE}u;

  for (
    var item = local_index;
    item < ${SHARED_VALUES}u;
    item += ${WORKGROUP_LANES}u
  ) {
    let tile_frame = item / ${CHANNEL_VECTOR_TILE}u;
    let tile_channel = item % ${CHANNEL_VECTOR_TILE}u;
    let channel_vector = channel_vector_base + tile_channel;
    let signed_frame =
      i32(frame_base + tile_frame) - ${KERNEL_RADIUS};
    var glu = vec4<f32>(0.0);
    if (
      group.z < params.batch_size &&
      signed_frame >= 0 &&
      signed_frame < i32(params.frames) &&
      signed_frame < i32(valid_lengths[group.z]) &&
      channel_vector < params.channel_vectors
    ) {
      let row = group.z * params.frames + u32(signed_frame);
      let row_base = row * params.input_vectors_per_row;
      let value = vec4<f32>(
        pointwise_input[row_base + channel_vector]
      );
      let gate = vec4<f32>(
        pointwise_input[
          row_base + params.channel_vectors + channel_vector
        ]
      );
      glu = value * sigmoid(gate);
    }
    glu_tile[item] = glu;
  }
  workgroupBarrier();

  let frame = frame_base + local_id.y;
  let channel_vector = channel_vector_base + local_id.x;
  if (
    group.z >= params.batch_size ||
    frame >= params.frames ||
    channel_vector >= params.channel_vectors
  ) {
    return;
  }

  var sum = vec4<f32>(0.0);
  for (
    var kernel_index = 0u;
    kernel_index < ${CONFORMER_CONV_KERNEL_SIZE}u;
    kernel_index += 1u
  ) {
    let shared_index =
      (local_id.y + kernel_index) * ${CHANNEL_VECTOR_TILE}u
      + local_id.x;
    let weight_index =
      kernel_index * params.channel_vectors + channel_vector;
    sum = fma(
      glu_tile[shared_index],
      vec4<f32>(depthwise_weights[weight_index]),
      sum
    );
  }

  let affine =
    sum * batch_norm_scale[channel_vector]
    + batch_norm_shift[channel_vector];
  let activated = affine * sigmoid(affine);
  let output_row = group.z * params.frames + frame;
  output[output_row * params.channel_vectors + channel_vector] =
    vec4<f16>(activated);
}
`;

const CONFORMER_CONV_FP32_WGSL = fp32ShaderVariant(CONFORMER_CONV_WGSL);

export function conformerConvWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  return executionProfile.precision === "fp16"
    ? CONFORMER_CONV_WGSL
    : CONFORMER_CONV_FP32_WGSL;
}

function fp32ShaderVariant(fp16Wgsl: string): string {
  return fp16Wgsl
    .replace("enable f16;\n", "")
    .replaceAll("f16", "f32")
    .replaceAll("0.0h", "0.0");
}
