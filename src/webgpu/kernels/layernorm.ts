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

const WORKGROUP_SIZE = 256;
const SUBGROUP_SIZE = 32;
const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
export const LAYERNORM_PARAMETER_BYTES = 16;

export interface LayerNormDescriptor {
  readonly label: string;
  readonly input: ArenaSlice;
  readonly output: ArenaSlice;
  readonly gamma: GpuTensor;
  readonly beta: GpuTensor;
  readonly rows: number;
  readonly channels: number;
  readonly epsilon?: number;
}

export class LayerNormDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: ArenaSlice,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly rows: number,
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    shape?: EncoderDispatchShape,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    const rows = shape?.rows ?? this.rows;
    if (
      !Number.isSafeInteger(rows) ||
      rows <= 0 ||
      rows > this.rows
    ) {
      throw new RangeError(
        `${this.label} active rows must be in [1, ${this.rows}]`,
      );
    }
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(rows);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
  }
}

export class LayerNormKernel {
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
  ): Promise<LayerNormKernel> {
    if (
      executionProfile.kernelBackend === "subgroups" &&
      !device.features.has("subgroups")
    ) {
      throw new Error("Parakeet LayerNorm requires WebGPU subgroups");
    }
    if (
      executionProfile.precision === "fp16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error("Parakeet FP16 LayerNorm requires WebGPU shader-f16");
    }
    const label = executionProfile.kernelBackend === "portable"
      ? executionProfile.precision === "fp16"
        ? "parakeet-layernorm-portable"
        : "parakeet-layernorm-fp32-portable"
      : executionProfile.precision === "fp16"
        ? "parakeet-layernorm"
        : "parakeet-layernorm-fp32";
    const layout = device.createBindGroupLayout({
      label: "parakeet-layernorm-bindings",
      entries: [
        storageEntry(0, false),
        storageEntry(1, true),
        storageEntry(2, true),
        storageEntry(3, false),
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const module = device.createShaderModule({
      label,
      code: layerNormWgslForProfile(executionProfile),
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `LayerNorm WGSL failed: ${errors.map((message) => message.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    const parameterPool = new UniformParameterPool(
      device,
      LAYERNORM_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-layernorm-parameter-pool",
    );
    return new LayerNormKernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(descriptor: LayerNormDescriptor): LayerNormDispatch {
    if (
      !Number.isSafeInteger(descriptor.rows) ||
      !Number.isSafeInteger(descriptor.channels) ||
      descriptor.rows <= 0 ||
      descriptor.channels <= 0 ||
      descriptor.channels % 4 !== 0
    ) {
      throw new RangeError(`Invalid LayerNorm shape for ${descriptor.label}`);
    }
    const scalarBytes = executionScalarBytes(this.executionProfile);
    const activationBytes = checkedByteLength(
      [descriptor.rows, descriptor.channels, scalarBytes],
      `${descriptor.label} LayerNorm activation`,
    );
    requireArenaCapacity(
      descriptor.input,
      activationBytes,
      "input",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.output,
      activationBytes,
      "output",
      descriptor.label,
    );
    const affineBytes = checkedByteLength(
      [descriptor.channels, Float32Array.BYTES_PER_ELEMENT],
      `${descriptor.label} LayerNorm affine`,
    );
    requirePackedTensor(
      descriptor.gamma,
      [descriptor.channels],
      affineBytes,
      `${descriptor.label} gamma`,
    );
    requirePackedTensor(
      descriptor.beta,
      [descriptor.channels],
      affineBytes,
      `${descriptor.label} beta`,
    );
    const params = new ArrayBuffer(LAYERNORM_PARAMETER_BYTES);
    new Uint32Array(params, 0, 2).set([descriptor.rows, descriptor.channels]);
    new Float32Array(params, 8, 1)[0] = descriptor.epsilon ?? 1e-5;
    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      params,
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.input) },
        { binding: 1, resource: descriptor.gamma.binding },
        { binding: 2, resource: descriptor.beta.binding },
        { binding: 3, resource: this.arena.binding(descriptor.output) },
        { binding: 4, resource: parameterBinding },
      ],
    });
    return new LayerNormDispatch(
      descriptor.label,
      descriptor.output,
      this.pipeline,
      bindGroup,
      descriptor.rows,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameterPool.destroy();
  }
}

function executionScalarBytes(
  executionProfile: ParakeetExecutionProfile,
): 2 | 4 {
  return executionProfile.precision === "fp16" ? 2 : 4;
}

function requireArenaCapacity(
  slice: ArenaSlice,
  expectedBytes: number,
  role: string,
  label: string,
): void {
  if (slice.byteLength < expectedBytes) {
    throw new RangeError(
      `${label} ${role} needs ${expectedBytes} bytes; arena slice has ` +
        `${slice.byteLength}`,
    );
  }
}

function requirePackedTensor(
  tensor: GpuTensor,
  shape: readonly number[],
  byteLength: number,
  label: string,
): void {
  const record = tensor.runtimeRecord;
  if (
    record.dtype !== "float32" ||
    record.layout !== "channel" ||
    record.byteLength !== byteLength ||
    !sameShape(record.logicalShape, shape) ||
    !sameShape(record.storageShape, shape)
  ) {
    throw new Error(
      `${label} must be float32 channel logical/storage ` +
        `[${shape.join(",")}], ${byteLength} bytes`,
    );
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

function checkedByteLength(factors: readonly number[], label: string): number {
  const result = factors.reduce((product, factor) => product * factor, 1);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(`${label} byte length exceeds the safe integer range`);
  }
  return result;
}

function storageEntry(binding: number, readOnly: boolean): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: readOnly ? "read-only-storage" : "storage" },
  };
}

const LAYERNORM_WGSL = `
enable f16;
enable subgroups;

struct Params {
  rows: u32,
  channels: u32,
  epsilon: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> gamma: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> beta: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> subgroup_sum: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> subgroup_square: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> mean_shared: f32;
var<workgroup> inverse_std_shared: f32;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup_slot: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.x;
  if (row >= params.rows || subgroup_size != ${SUBGROUP_SIZE}u) {
    return;
  }

  let vectors = params.channels / 4u;
  var sum = 0.0;
  var square = 0.0;
  for (var vector = lane; vector < vectors; vector += ${WORKGROUP_SIZE}u) {
    let value = vec4<f32>(input[row * vectors + vector]);
    sum += dot(value, vec4<f32>(1.0));
    square += dot(value, value);
  }
  let local_sum = subgroupAdd(sum);
  let local_square = subgroupAdd(square);
  if (subgroup_lane == 0u) {
    subgroup_sum[subgroup_slot] = local_sum;
    subgroup_square[subgroup_slot] = local_square;
  }
  workgroupBarrier();
  if (lane == 0u) {
    var total_sum = 0.0;
    var total_square = 0.0;
    for (
      var subgroup = 0u;
      subgroup < ${SUBGROUPS_PER_WORKGROUP}u;
      subgroup += 1u
    ) {
      total_sum += subgroup_sum[subgroup];
      total_square += subgroup_square[subgroup];
    }
    let inverse_channels = 1.0 / f32(params.channels);
    let mean = total_sum * inverse_channels;
    let variance = max(
      total_square * inverse_channels - mean * mean,
      0.0
    );
    mean_shared = mean;
    inverse_std_shared = inverseSqrt(variance + params.epsilon);
  }
  workgroupBarrier();
  let mean = workgroupUniformLoad(&mean_shared);
  let inverse_std = workgroupUniformLoad(&inverse_std_shared);
  for (var vector = lane; vector < vectors; vector += ${WORKGROUP_SIZE}u) {
    let value = vec4<f32>(input[row * vectors + vector]);
    let normalized = (value - vec4<f32>(mean)) * inverse_std;
    let affine = normalized * gamma[vector] + beta[vector];
    output[row * vectors + vector] = vec4<f16>(affine);
  }
}
`;

const LAYERNORM_FP32_WGSL = fp32ShaderVariant(LAYERNORM_WGSL);

let layerNormPortableWgsl: string | undefined;
let layerNormFp32PortableWgsl: string | undefined;

export function layerNormWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  if (executionProfile.kernelBackend === "portable") {
    layerNormPortableWgsl ??= portableLayerNormVariant(LAYERNORM_WGSL);
    if (executionProfile.precision === "fp16") {
      return layerNormPortableWgsl;
    }
    layerNormFp32PortableWgsl ??= fp32ShaderVariant(
      layerNormPortableWgsl,
    );
    return layerNormFp32PortableWgsl;
  }
  return executionProfile.precision === "fp16"
    ? LAYERNORM_WGSL
    : LAYERNORM_FP32_WGSL;
}

function fp32ShaderVariant(fp16Wgsl: string): string {
  return fp16Wgsl
    .replace("enable f16;\n", "")
    .replaceAll("f16", "f32")
    .replaceAll("0.0h", "0.0");
}

/**
 * Replace the two hardware reductions with eight virtual 32-lane trees. Each
 * tree combines adjacent pairs from offset 1 upward, matching the fixed-32
 * subgroup reduction order on the production Metal path. The final lane-zero
 * fold retains the subgroup path's increasing-group order;
 * input is deliberately reloaded after the barriers because retaining it has
 * previously changed accepted FP16 output bits on Metal.
 */
function portableLayerNormVariant(subgroupWgsl: string): string {
  let result = replaceRequired(subgroupWgsl, "enable subgroups;\n", "");
  result = replaceRequired(
    result,
    `var<workgroup> subgroup_sum: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> subgroup_square: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;`,
    `var<workgroup> moments: array<vec2<f32>, ${WORKGROUP_SIZE}>;`,
  );
  result = replaceRequired(
    result,
    `  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup_slot: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,`,
    `  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,`,
  );
  result = replaceRequired(
    result,
    `  if (row >= params.rows || subgroup_size != ${SUBGROUP_SIZE}u) {`,
    "  if (row >= params.rows) {",
  );

  const reductionStart = result.indexOf("  let local_sum = subgroupAdd(sum);");
  const affineStart = result.indexOf(
    "  let mean = workgroupUniformLoad(&mean_shared);",
    reductionStart,
  );
  if (reductionStart < 0 || affineStart < 0) {
    throw new Error("Unable to construct portable LayerNorm reduction");
  }
  const portableReduction = `  moments[lane] = vec2<f32>(sum, square);
  workgroupBarrier();
  let virtual_lane = lane & ${SUBGROUP_SIZE - 1}u;
  for (var offset = 1u; offset < ${SUBGROUP_SIZE}u; offset *= 2u) {
    if ((virtual_lane & (offset * 2u - 1u)) == 0u) {
      moments[lane] += moments[lane + offset];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    var totals = vec2<f32>(0.0);
    for (
      var virtual_group = 0u;
      virtual_group < ${SUBGROUPS_PER_WORKGROUP}u;
      virtual_group += 1u
    ) {
      totals += moments[virtual_group * ${SUBGROUP_SIZE}u];
    }
    let inverse_channels = 1.0 / f32(params.channels);
    let mean = totals.x * inverse_channels;
    let variance = max(
      totals.y * inverse_channels - mean * mean,
      0.0
    );
    mean_shared = mean;
    inverse_std_shared = inverseSqrt(variance + params.epsilon);
  }
  workgroupBarrier();
`;
  return (
    result.slice(0, reductionStart) +
    portableReduction +
    result.slice(affineStart)
  );
}

function replaceRequired(
  source: string,
  search: string,
  replacement: string,
): string {
  if (!source.includes(search)) {
    throw new Error(`Unable to construct portable LayerNorm shader: ${search}`);
  }
  return source.replace(search, replacement);
}
