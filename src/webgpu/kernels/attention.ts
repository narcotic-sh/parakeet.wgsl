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

const WORKGROUP_SIZE = 192;
const SUBGROUP_SIZE = 32;
const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
const HEAD_SIZE = 128;
const HEADS = 8;
const HIDDEN_SIZE = 1024;
const HEAD_VECTORS = HEAD_SIZE / 4;
const QKV_VECTORS_PER_ROW = (HIDDEN_SIZE * 3) / 4;
const HIDDEN_VECTORS = HIDDEN_SIZE / 4;
const QUERIES_PER_WORKGROUP = 6;
const VALUE_TILE_KEYS = 16;
const FP32_VALUE_TILE_KEYS = 8;
export const RELATIVE_ATTENTION_PARAMETER_BYTES = 32;
const WORKGROUP_STORAGE_BYTES =
  QUERIES_PER_WORKGROUP * HEAD_VECTORS * 16 * 2 +
  QUERIES_PER_WORKGROUP * WORKGROUP_SIZE * 4 +
  VALUE_TILE_KEYS * HEAD_VECTORS * 8 +
  SUBGROUPS_PER_WORKGROUP * 4 +
  16;
const FP32_WORKGROUP_STORAGE_BYTES =
  QUERIES_PER_WORKGROUP * HEAD_VECTORS * 16 * 2 +
  QUERIES_PER_WORKGROUP * WORKGROUP_SIZE * 4 +
  FP32_VALUE_TILE_KEYS * HEAD_VECTORS * 16 +
  SUBGROUPS_PER_WORKGROUP * 4 +
  16;

export interface RelativeAttentionDescriptor {
  readonly label: string;
  readonly qkv: ArenaSlice;
  readonly projectedPositions: GpuTensor;
  readonly biasU: GpuTensor;
  readonly biasV: GpuTensor;
  readonly validLengths: ArenaSlice;
  readonly output: ArenaSlice;
  readonly batchSize: number;
  readonly frames: number;
}

export class RelativeAttentionDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: ArenaSlice,
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
 * Fused full relative-position attention. It never materializes an O(T²)
 * score tensor: one workgroup owns six adjacent queries for one
 * (batch, head), reuses each K vector across those queries, computes FP32
 * softmax, then reuses 16-key V tiles across their context reductions.
 */
export class RelativeAttentionKernel {
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
  ): Promise<RelativeAttentionKernel> {
    if (
      executionProfile.kernelBackend === "subgroups" &&
      !device.features.has("subgroups")
    ) {
      throw new Error("Parakeet relative attention requires WebGPU subgroups");
    }
    if (
      executionProfile.precision === "fp16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error(
        "Parakeet FP16 relative attention requires WebGPU shader-f16",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
    ) {
      throw new Error(
        `Parakeet relative attention requires a ${WORKGROUP_SIZE}-lane workgroup`,
      );
    }
    const workgroupStorageBytes =
      relativeAttentionWorkgroupStorageBytes(executionProfile);
    if (
      device.limits.maxComputeWorkgroupStorageSize < workgroupStorageBytes
    ) {
      throw new Error(
        `Parakeet relative attention requires ${workgroupStorageBytes} ` +
          "bytes of workgroup storage",
      );
    }
    const label = executionProfile.kernelBackend === "portable"
      ? executionProfile.precision === "fp16"
        ? "parakeet-relative-attention-portable"
        : "parakeet-relative-attention-fp32-portable"
      : executionProfile.precision === "fp16"
        ? "parakeet-relative-attention"
        : "parakeet-relative-attention-fp32";
    const layout = device.createBindGroupLayout({
      label: "parakeet-relative-attention-bindings",
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
      code: relativeAttentionWgslForProfile(executionProfile),
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Relative attention WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    const parameterPool = new UniformParameterPool(
      device,
      RELATIVE_ATTENTION_PARAMETER_BYTES,
      uniformParameterCapacity,
      "parakeet-relative-attention-parameter-pool",
    );
    return new RelativeAttentionKernel(
      device,
      arena,
      pipeline,
      layout,
      parameterPool,
      executionProfile,
    );
  }

  createDispatch(
    descriptor: RelativeAttentionDescriptor,
  ): RelativeAttentionDispatch {
    if (
      descriptor.batchSize <= 0 ||
      descriptor.frames <= 0 ||
      descriptor.frames > WORKGROUP_SIZE
    ) {
      throw new RangeError(`Unsupported attention shape for ${descriptor.label}`);
    }
    const positionRows = descriptor.frames * 2 - 1;
    const scalarBytes = executionScalarBytes(this.executionProfile);
    const rows = checkedByteLength(
      [descriptor.batchSize, descriptor.frames],
      `${descriptor.label} attention rows`,
    );
    requireArenaCapacity(
      descriptor.qkv,
      checkedByteLength(
        [rows, HIDDEN_SIZE * 3, scalarBytes],
        `${descriptor.label} attention QKV`,
      ),
      "QKV",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.output,
      checkedByteLength(
        [rows, HIDDEN_SIZE, scalarBytes],
        `${descriptor.label} attention output`,
      ),
      "output",
      descriptor.label,
    );
    requireArenaCapacity(
      descriptor.validLengths,
      checkedByteLength(
        [descriptor.batchSize, Uint32Array.BYTES_PER_ELEMENT],
        `${descriptor.label} attention valid lengths`,
      ),
      "valid lengths",
      descriptor.label,
    );
    requireProjectedPositions(
      descriptor.projectedPositions,
      positionRows,
      this.executionProfile,
    );
    requireAttentionBias(
      descriptor.biasU,
      `${descriptor.label} bias U`,
    );
    requireAttentionBias(
      descriptor.biasV,
      `${descriptor.label} bias V`,
    );
    const params = new ArrayBuffer(
      RELATIVE_ATTENTION_PARAMETER_BYTES,
    );
    new Uint32Array(params, 0, 3).set([
      descriptor.batchSize,
      descriptor.frames,
      descriptor.frames * 2 - 1,
    ]);
    const parameterBinding = this.parameterPool.bindingFor(
      descriptor.label,
      params,
    );
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.arena.binding(descriptor.qkv) },
        { binding: 1, resource: descriptor.projectedPositions.binding },
        { binding: 2, resource: descriptor.biasU.binding },
        { binding: 3, resource: descriptor.biasV.binding },
        {
          binding: 4,
          resource: this.arena.binding(descriptor.validLengths),
        },
        { binding: 5, resource: this.arena.binding(descriptor.output) },
        { binding: 6, resource: parameterBinding },
      ],
    });
    return new RelativeAttentionDispatch(
      descriptor.label,
      descriptor.output,
      this.pipeline,
      bindGroup,
      [
        Math.ceil(descriptor.frames / QUERIES_PER_WORKGROUP),
        HEADS,
        descriptor.batchSize,
      ],
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameterPool.destroy();
  }
}

function requireProjectedPositions(
  tensor: GpuTensor,
  positionRows: number,
  executionProfile: ParakeetExecutionProfile,
): void {
  const expectedLogicalShape = [positionRows, HIDDEN_SIZE] as const;
  const expectedStorageShape = [
    HEADS,
    HEAD_VECTORS,
    positionRows,
    4,
  ] as const;
  const dtype = executionProfile.precision === "fp16"
    ? "float16"
    : "float32";
  const scalarBytes = executionScalarBytes(executionProfile);
  const record = tensor.runtimeRecord;
  if (
    record.dtype !== dtype ||
    record.layout !== "head-vector-position-vec4" ||
    record.byteLength !== positionRows * HIDDEN_SIZE * scalarBytes ||
    !sameShape(record.logicalShape, expectedLogicalShape) ||
    !sameShape(record.storageShape, expectedStorageShape)
  ) {
    throw new Error(
      `Projected positions must be ${dtype} head-vector-position-vec4, ` +
        `logical [${expectedLogicalShape.join(",")}], storage ` +
        `[${expectedStorageShape.join(",")}]`,
    );
  }
}

function requireAttentionBias(tensor: GpuTensor, label: string): void {
  const shape = [HEADS, HEAD_SIZE] as const;
  const record = tensor.runtimeRecord;
  if (
    record.dtype !== "float32" ||
    record.layout !== "head-by-channel" ||
    record.byteLength !== HIDDEN_SIZE * Float32Array.BYTES_PER_ELEMENT ||
    !sameShape(record.logicalShape, shape) ||
    !sameShape(record.storageShape, shape)
  ) {
    throw new Error(
      `${label} must be float32 head-by-channel ` +
        `logical/storage [${shape.join(",")}]`,
    );
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

function checkedByteLength(factors: readonly number[], label: string): number {
  const result = factors.reduce((product, factor) => product * factor, 1);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(`${label} byte length exceeds the safe integer range`);
  }
  return result;
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

function storageEntry(binding: number, readOnly: boolean): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: readOnly ? "read-only-storage" : "storage" },
  };
}

const RELATIVE_ATTENTION_WGSL = `
enable f16;
enable subgroups;

struct Params {
  batch_size: u32,
  frames: u32,
  position_rows: u32,
  _pad0: u32,
  _pad1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> qkv: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias_u: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias_v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> valid_lengths: array<u32>;
@group(0) @binding(5) var<storage, read_write> context: array<vec4<f16>>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> query_u: array<
  vec4<f32>,
  ${QUERIES_PER_WORKGROUP * HEAD_VECTORS}
>;
var<workgroup> query_v: array<
  vec4<f32>,
  ${QUERIES_PER_WORKGROUP * HEAD_VECTORS}
>;
var<workgroup> scores: array<
  f32,
  ${QUERIES_PER_WORKGROUP * WORKGROUP_SIZE}
>;
var<workgroup> value_tile: array<
  vec4<f16>,
  ${VALUE_TILE_KEYS * HEAD_VECTORS}
>;
var<workgroup> subgroup_partials: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> valid_shared: u32;
var<workgroup> reduction_shared: f32;

fn accumulate_score_vector(
  query_with_u: vec4<f32>,
  query_with_v: vec4<f32>,
  key: vec4<f32>,
  relative_key: vec4<f32>,
  initial: f32,
) -> f32 {
  var result = initial;
  result = fma(query_with_u.x, key.x, result);
  result = fma(query_with_v.x, relative_key.x, result);
  result = fma(query_with_u.y, key.y, result);
  result = fma(query_with_v.y, relative_key.y, result);
  result = fma(query_with_u.z, key.z, result);
  result = fma(query_with_v.z, relative_key.z, result);
  result = fma(query_with_u.w, key.w, result);
  result = fma(query_with_v.w, relative_key.w, result);
  return result;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup_slot: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let query_base = group.x * ${QUERIES_PER_WORKGROUP}u;
  let head = group.y;
  let batch = group.z;
  let head_vector_offset = head * ${HEAD_VECTORS}u;

  if (lane == 0u) {
    valid_shared = min(valid_lengths[batch], params.frames);
  }
  workgroupBarrier();
  if (subgroup_size != ${SUBGROUP_SIZE}u) {
    return;
  }
  let valid = workgroupUniformLoad(&valid_shared);

  if (query_base >= valid) {
    if (lane < ${QUERIES_PER_WORKGROUP * HEAD_VECTORS}u) {
      let query_slot = lane / ${HEAD_VECTORS}u;
      let vector = lane - query_slot * ${HEAD_VECTORS}u;
      let query = query_base + query_slot;
      if (query < params.frames) {
        let row = batch * params.frames + query;
        context[
          row * ${HIDDEN_VECTORS}u
          + head_vector_offset
          + vector
        ] = vec4<f16>(0.0h);
      }
    }
    return;
  }

  if (lane < ${QUERIES_PER_WORKGROUP * HEAD_VECTORS}u) {
    let query_slot = lane / ${HEAD_VECTORS}u;
    let vector = lane - query_slot * ${HEAD_VECTORS}u;
    let query = query_base + query_slot;
    let query_shared_index = query_slot * ${HEAD_VECTORS}u + vector;
    if (query < params.frames) {
      let row = batch * params.frames + query;
      let query_vector = vec4<f32>(
        qkv[
          row * ${QKV_VECTORS_PER_ROW}u
          + head_vector_offset
          + vector
        ]
      );
      query_u[query_shared_index] =
        query_vector + bias_u[head_vector_offset + vector];
      query_v[query_shared_index] =
        query_vector + bias_v[head_vector_offset + vector];
    } else {
      query_u[query_shared_index] = vec4<f32>(0.0);
      query_v[query_shared_index] = vec4<f32>(0.0);
    }
  }
  workgroupBarrier();

  var query_scores: array<f32, ${QUERIES_PER_WORKGROUP}>;
  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    query_scores[query_slot] = select(
      -3.402823466e+38,
      0.0,
      lane < valid
    );
  }

  if (lane < valid) {
    let key_row = batch * params.frames + lane;
    for (var vector = 0u; vector < ${HEAD_VECTORS}u; vector += 1u) {
      let key = vec4<f32>(
        qkv[
          key_row * ${QKV_VECTORS_PER_ROW}u
          + ${HIDDEN_VECTORS}u
          + head_vector_offset
          + vector
        ]
      );
      for (
        var query_slot = 0u;
        query_slot < ${QUERIES_PER_WORKGROUP}u;
        query_slot += 1u
      ) {
        let query = query_base + query_slot;
        if (query < valid) {
          let relative_position = params.frames - 1u - query + lane;
          let relative_key = vec4<f32>(
            positions[
              (head_vector_offset + vector) * params.position_rows
              + relative_position
            ]
          );
          let query_shared_index =
            query_slot * ${HEAD_VECTORS}u + vector;
          query_scores[query_slot] = accumulate_score_vector(
            query_u[query_shared_index],
            query_v[query_shared_index],
            key,
            relative_key,
            query_scores[query_slot]
          );
        }
      }
    }
    for (
      var query_slot = 0u;
      query_slot < ${QUERIES_PER_WORKGROUP}u;
      query_slot += 1u
    ) {
      if (query_base + query_slot < valid) {
        query_scores[query_slot] *= ${1 / Math.sqrt(128)};
      }
    }
  }

  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    let score_index = query_slot * ${WORKGROUP_SIZE}u + lane;
    scores[score_index] = query_scores[query_slot];

    let subgroup_maximum = subgroupMax(query_scores[query_slot]);
    if (subgroup_lane == 0u) {
      subgroup_partials[subgroup_slot] = subgroup_maximum;
    }
    workgroupBarrier();
    if (lane == 0u) {
      var maximum = -3.402823466e+38;
      for (
        var subgroup = 0u;
        subgroup < ${SUBGROUPS_PER_WORKGROUP}u;
        subgroup += 1u
      ) {
        maximum = max(maximum, subgroup_partials[subgroup]);
      }
      reduction_shared = maximum;
    }
    workgroupBarrier();
    let maximum = workgroupUniformLoad(&reduction_shared);
    var exponential = 0.0;
    if (lane < valid) {
      exponential = exp(query_scores[query_slot] - maximum);
    }

    let subgroup_sum = subgroupAdd(exponential);
    if (subgroup_lane == 0u) {
      subgroup_partials[subgroup_slot] = subgroup_sum;
    }
    workgroupBarrier();
    if (lane == 0u) {
      var sum = 0.0;
      for (
        var subgroup = 0u;
        subgroup < ${SUBGROUPS_PER_WORKGROUP}u;
        subgroup += 1u
      ) {
        sum += subgroup_partials[subgroup];
      }
      reduction_shared = 1.0 / sum;
    }
    workgroupBarrier();
    let inverse_sum = workgroupUniformLoad(&reduction_shared);
    scores[score_index] = exponential * inverse_sum;
  }
  workgroupBarrier();

  let output_lane =
    lane < ${QUERIES_PER_WORKGROUP * HEAD_VECTORS}u;
  let output_query_slot = lane / ${HEAD_VECTORS}u;
  let output_vector =
    lane - output_query_slot * ${HEAD_VECTORS}u;
  let output_query = query_base + output_query_slot;
  var output_value = vec4<f32>(0.0);

  for (
    var tile_start = 0u;
    tile_start < valid;
    tile_start += ${VALUE_TILE_KEYS}u
  ) {
    let tile_keys = min(${VALUE_TILE_KEYS}u, valid - tile_start);
    let tile_vectors = tile_keys * ${HEAD_VECTORS}u;
    for (
      var tile_vector = lane;
      tile_vector < tile_vectors;
      tile_vector += ${WORKGROUP_SIZE}u
    ) {
      let key_in_tile = tile_vector / ${HEAD_VECTORS}u;
      let vector =
        tile_vector - key_in_tile * ${HEAD_VECTORS}u;
      let key_row = batch * params.frames + tile_start + key_in_tile;
      value_tile[tile_vector] =
        qkv[
          key_row * ${QKV_VECTORS_PER_ROW}u
          + ${HIDDEN_VECTORS * 2}u
          + head_vector_offset
          + vector
        ];
    }
    workgroupBarrier();
    if (output_lane && output_query < valid) {
      for (
        var key_in_tile = 0u;
        key_in_tile < tile_keys;
        key_in_tile += 1u
      ) {
        let probability =
          scores[
            output_query_slot * ${WORKGROUP_SIZE}u
            + tile_start
            + key_in_tile
          ];
        let value = vec4<f32>(
          value_tile[
            key_in_tile * ${HEAD_VECTORS}u + output_vector
          ]
        );
        output_value = fma(
          vec4<f32>(probability),
          value,
          output_value
        );
      }
    }
    workgroupBarrier();
  }

  if (output_lane && output_query < params.frames) {
    let row = batch * params.frames + output_query;
    context[
      row * ${HIDDEN_VECTORS}u
      + head_vector_offset
      + output_vector
    ] = select(
      vec4<f16>(0.0h),
      vec4<f16>(output_value),
      output_query < valid
    );
  }
}
`;

const RELATIVE_ATTENTION_FP32_WGSL = fp32RelativeAttentionVariant(
  RELATIVE_ATTENTION_WGSL,
);

let relativeAttentionPortableWgsl: string | undefined;
let relativeAttentionFp32PortableWgsl: string | undefined;

export function relativeAttentionWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  if (executionProfile.kernelBackend === "portable") {
    relativeAttentionPortableWgsl ??= portableRelativeAttentionVariant(
      RELATIVE_ATTENTION_WGSL,
    );
    if (executionProfile.precision === "fp16") {
      return relativeAttentionPortableWgsl;
    }
    relativeAttentionFp32PortableWgsl ??= fp32RelativeAttentionVariant(
      relativeAttentionPortableWgsl,
    );
    return relativeAttentionFp32PortableWgsl;
  }
  return executionProfile.precision === "fp16"
    ? RELATIVE_ATTENTION_WGSL
    : RELATIVE_ATTENTION_FP32_WGSL;
}

export function relativeAttentionWorkgroupStorageBytes(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): number {
  return executionProfile.precision === "fp16"
    ? WORKGROUP_STORAGE_BYTES
    : FP32_WORKGROUP_STORAGE_BYTES;
}

function fp32RelativeAttentionVariant(fp16Wgsl: string): string {
  let result = fp32ShaderVariant(fp16Wgsl);
  result = replaceRequired(
    result,
    `var<workgroup> value_tile: array<
  vec4<f32>,
  ${VALUE_TILE_KEYS * HEAD_VECTORS}
>;`,
    `var<workgroup> value_tile: array<
  vec4<f32>,
  ${FP32_VALUE_TILE_KEYS * HEAD_VECTORS}
>;`,
  );
  result = replaceRequired(
    result,
    `tile_start += ${VALUE_TILE_KEYS}u`,
    `tile_start += ${FP32_VALUE_TILE_KEYS}u`,
  );
  return replaceRequired(
    result,
    `let tile_keys = min(${VALUE_TILE_KEYS}u, valid - tile_start);`,
    `let tile_keys = min(${FP32_VALUE_TILE_KEYS}u, valid - tile_start);`,
  );
}

function fp32ShaderVariant(fp16Wgsl: string): string {
  return fp16Wgsl
    .replace("enable f16;\n", "")
    .replaceAll("f16", "f32")
    .replaceAll("0.0h", "0.0");
}

/**
 * Build the standards-only attention path without changing the established
 * subgroup shader. Once Q/K score production finishes, query_u/query_v are
 * dead. The portable reduction reuses those two 192-entry vec4 arrays as a
 * lane-major six-query scratch, keeping the workgroup footprint and the Q6
 * geometry unchanged while reducing all queries together. The virtual trees
 * combine adjacent pairs from offset 1 upward so their Float32 sum order
 * matches the fixed-32 subgroup reduction used by the protected path.
 */
function portableRelativeAttentionVariant(subgroupWgsl: string): string {
  let result = replaceRequired(subgroupWgsl, "enable subgroups;\n", "");
  result = replaceRequired(
    result,
    `var<workgroup> subgroup_partials: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> valid_shared: u32;
var<workgroup> reduction_shared: f32;`,
    `var<workgroup> reduction_a: vec4<f32>;
var<workgroup> reduction_b: vec4<f32>;
var<workgroup> valid_shared: u32;`,
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
    `  workgroupBarrier();
  if (subgroup_size != ${SUBGROUP_SIZE}u) {
    return;
  }
  let valid = workgroupUniformLoad(&valid_shared);`,
    `  workgroupBarrier();
  let valid = workgroupUniformLoad(&valid_shared);`,
  );

  const softmaxStartMarker = `  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    let score_index = query_slot * ${WORKGROUP_SIZE}u + lane;`;
  const softmaxStart = result.indexOf(softmaxStartMarker);
  const outputStart = result.indexOf("\n  let output_lane =", softmaxStart);
  if (softmaxStart < 0 || outputStart < 0) {
    throw new Error("Unable to construct portable attention softmax");
  }
  const portableSoftmax = `  // Preserve every raw score while all six query reductions run together.
  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    scores[query_slot * ${WORKGROUP_SIZE}u + lane] =
      query_scores[query_slot];
  }
  // query_u/query_v are no longer read after score production. Wait for every
  // lane before reusing them as two vec4 reduction records per key lane.
  workgroupBarrier();
  query_u[lane] = vec4<f32>(
    query_scores[0u],
    query_scores[1u],
    query_scores[2u],
    query_scores[3u]
  );
  query_v[lane] = vec4<f32>(
    query_scores[4u],
    query_scores[5u],
    -3.402823466e+38,
    -3.402823466e+38
  );
  workgroupBarrier();

  let virtual_lane = lane & ${SUBGROUP_SIZE - 1}u;
  for (var offset = 1u; offset < ${SUBGROUP_SIZE}u; offset *= 2u) {
    if ((virtual_lane & (offset * 2u - 1u)) == 0u) {
      query_u[lane] = max(query_u[lane], query_u[lane + offset]);
      query_v[lane] = max(query_v[lane], query_v[lane + offset]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    var maximum_a = vec4<f32>(-3.402823466e+38);
    var maximum_b = vec4<f32>(-3.402823466e+38);
    for (
      var virtual_group = 0u;
      virtual_group < ${SUBGROUPS_PER_WORKGROUP}u;
      virtual_group += 1u
    ) {
      let base = virtual_group * ${SUBGROUP_SIZE}u;
      maximum_a = max(maximum_a, query_u[base]);
      maximum_b = max(maximum_b, query_v[base]);
    }
    reduction_a = maximum_a;
    reduction_b = maximum_b;
  }
  workgroupBarrier();
  let maximum_a = workgroupUniformLoad(&reduction_a);
  let maximum_b = workgroupUniformLoad(&reduction_b);

  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    var maximum = 0.0;
    if (query_slot < 4u) {
      maximum = maximum_a[query_slot];
    } else {
      maximum = maximum_b[query_slot - 4u];
    }
    var exponential = 0.0;
    if (lane < valid) {
      exponential = exp(
        scores[query_slot * ${WORKGROUP_SIZE}u + lane] - maximum
      );
    }
    query_scores[query_slot] = exponential;
    scores[query_slot * ${WORKGROUP_SIZE}u + lane] = exponential;
  }
  query_u[lane] = vec4<f32>(
    query_scores[0u],
    query_scores[1u],
    query_scores[2u],
    query_scores[3u]
  );
  query_v[lane] = vec4<f32>(
    query_scores[4u],
    query_scores[5u],
    0.0,
    0.0
  );
  workgroupBarrier();

  for (var offset = 1u; offset < ${SUBGROUP_SIZE}u; offset *= 2u) {
    if ((virtual_lane & (offset * 2u - 1u)) == 0u) {
      query_u[lane] += query_u[lane + offset];
      query_v[lane] += query_v[lane + offset];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    var sum_a = vec4<f32>(0.0);
    var sum_b = vec4<f32>(0.0);
    for (
      var virtual_group = 0u;
      virtual_group < ${SUBGROUPS_PER_WORKGROUP}u;
      virtual_group += 1u
    ) {
      let base = virtual_group * ${SUBGROUP_SIZE}u;
      sum_a += query_u[base];
      sum_b += query_v[base];
    }
    reduction_a = vec4<f32>(1.0) / sum_a;
    reduction_b = vec4<f32>(
      1.0 / sum_b.x,
      1.0 / sum_b.y,
      0.0,
      0.0
    );
  }
  workgroupBarrier();
  let inverse_a = workgroupUniformLoad(&reduction_a);
  let inverse_b = workgroupUniformLoad(&reduction_b);
  for (
    var query_slot = 0u;
    query_slot < ${QUERIES_PER_WORKGROUP}u;
    query_slot += 1u
  ) {
    var inverse_sum = 0.0;
    if (query_slot < 4u) {
      inverse_sum = inverse_a[query_slot];
    } else {
      inverse_sum = inverse_b[query_slot - 4u];
    }
    scores[query_slot * ${WORKGROUP_SIZE}u + lane] =
      query_scores[query_slot] * inverse_sum;
  }
  workgroupBarrier();
`;
  result =
    result.slice(0, softmaxStart) +
    portableSoftmax +
    result.slice(outputStart);
  return result;
}

function replaceRequired(
  source: string,
  search: string,
  replacement: string,
): string {
  if (!source.includes(search)) {
    throw new Error(`Unable to construct FP32 attention shader: ${search}`);
  }
  return source.replace(search, replacement);
}
