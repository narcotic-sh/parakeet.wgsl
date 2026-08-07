/// <reference types="@webgpu/types" />

import {
  PALETTE_EXPANSION_PARAMETER_BYTES,
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
} from "./runtime-plan";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../webgpu/capabilities";

const WORKGROUP_SIZE = 256;
export { PALETTE_EXPANSION_PARAMETER_BYTES };
const PIPELINE_CONFIGURATIONS = [
  [5, RUNTIME_ROW_MAJOR_F16_LAYOUT],
  [5, RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT],
  [6, RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT],
] as const;
const FP32_PIPELINE_CONFIGURATIONS = [
  [5, RUNTIME_ROW_MAJOR_F32_LAYOUT],
  [5, RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT],
  [6, RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT],
] as const;

type F16PaletteRuntimeLayout =
  | typeof RUNTIME_ROW_MAJOR_F16_LAYOUT
  | typeof RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT;
type F32PaletteRuntimeLayout =
  | typeof RUNTIME_ROW_MAJOR_F32_LAYOUT
  | typeof RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT;
type PaletteRuntimeLayout =
  | F16PaletteRuntimeLayout
  | F32PaletteRuntimeLayout;

export interface PaletteExpansionDispatch {
  readonly label: string;
  encode(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

export interface TransientPaletteExpansionDescriptor {
  readonly label: string;
  readonly source: GPUBufferBinding;
  readonly palette: GPUBufferBinding;
  readonly target: GPUBufferBinding;
  readonly scalarCount: number;
  readonly columns: number;
  readonly paletteBits: 5 | 6;
  readonly paletteGroupColumns: number;
  readonly paletteGroups: number;
  readonly runtimeLayout: PaletteRuntimeLayout;
}

export function paletteExpansionParameterPoolBytes(
  slotCount: number,
  minUniformBufferOffsetAlignment: number,
): number {
  if (
    !Number.isSafeInteger(slotCount) ||
    slotCount <= 0 ||
    !Number.isSafeInteger(minUniformBufferOffsetAlignment) ||
    minUniformBufferOffsetAlignment <= 0
  ) {
    throw new RangeError(
      "Palette parameter pool geometry must be positive integers",
    );
  }
  const slotStride =
    Math.ceil(
      PALETTE_EXPANSION_PARAMETER_BYTES /
        minUniformBufferOffsetAlignment,
    ) * minUniformBufferOffsetAlignment;
  const byteLength = slotCount * slotStride;
  if (!Number.isSafeInteger(byteLength)) {
    throw new RangeError("Palette parameter pool is too large");
  }
  return byteLength;
}

export class PaletteExpansionParameterPool {
  readonly byteLength: number;
  readonly slotStride: number;

  private readonly buffer: GPUBuffer;
  private readonly bindings = new Map<string, GPUBufferBinding>();
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly slotCapacity: number,
    label: string,
    private readonly executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ) {
    const alignment =
      device.limits.minUniformBufferOffsetAlignment;
    this.byteLength = paletteExpansionParameterPoolBytes(
      slotCapacity,
      alignment,
    );
    this.slotStride =
      this.byteLength / this.slotCapacity;
    this.buffer = device.createBuffer({
      label,
      size: this.byteLength,
      usage:
        GPUBufferUsage.UNIFORM |
        GPUBufferUsage.COPY_DST,
    });
  }

  get usedSlots(): number {
    return this.bindings.size;
  }

  bindingFor(
    descriptor: TransientPaletteExpansionDescriptor,
  ): GPUBufferBinding {
    if (this.destroyed) {
      throw new Error("Palette parameter pool was destroyed");
    }
    requireTransientDescriptor(
      descriptor,
      this.executionProfile,
    );
    const key = parameterKey(descriptor);
    const existing = this.bindings.get(key);
    if (existing !== undefined) return existing;
    if (this.bindings.size >= this.slotCapacity) {
      throw new Error(
        `Palette parameter pool exceeds ${this.slotCapacity} slots`,
      );
    }
    const offset = this.bindings.size * this.slotStride;
    this.device.queue.writeBuffer(
      this.buffer,
      offset,
      parameterValues(descriptor),
    );
    const binding = {
      buffer: this.buffer,
      offset,
      size: PALETTE_EXPANSION_PARAMETER_BYTES,
    };
    this.bindings.set(key, binding);
    return binding;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindings.clear();
    this.buffer.destroy();
  }
}

export class PaletteWeightExpander {
  private constructor(
    private readonly device: GPUDevice,
    private readonly layout: GPUBindGroupLayout,
    private readonly pipelines: ReadonlyMap<
      string,
      GPUComputePipeline
    >,
    readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  static async create(
    device: GPUDevice,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<PaletteWeightExpander> {
    if (
      executionProfile.precision === "fp16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error(
        "Palette expansion requires WebGPU shader-f16",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
      WORKGROUP_SIZE
    ) {
      throw new Error(
        `Palette expansion requires ${WORKGROUP_SIZE} workgroup lanes`,
      );
    }
    const layout = device.createBindGroupLayout({
      label: "parakeet-palette-expansion-bindings",
      entries: [
        storageEntry(0, true),
        storageEntry(1, true),
        storageEntry(2, false),
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [layout],
    });
    const pipelineConfigurations =
      executionProfile.precision === "fp16"
        ? PIPELINE_CONFIGURATIONS
        : FP32_PIPELINE_CONFIGURATIONS;
    const entries = await Promise.all(
      pipelineConfigurations.map(
        async ([bits, runtimeLayout]) => [
          pipelineKey(bits, runtimeLayout),
          await createPipeline(
            device,
            pipelineLayout,
            bits,
            runtimeLayout,
            executionProfile,
          ),
        ] as const,
      ),
    );
    return new PaletteWeightExpander(
      device,
      layout,
      new Map(entries),
      executionProfile,
    );
  }

  createPooledDispatch(
    descriptor: TransientPaletteExpansionDescriptor,
    parameters: GPUBufferBinding,
  ): PaletteExpansionDispatch {
    requireTransientDescriptor(
      descriptor,
      this.executionProfile,
    );
    const offset = parameters.offset ?? 0;
    if (
      parameters.size !== PALETTE_EXPANSION_PARAMETER_BYTES ||
      offset % this.device.limits.minUniformBufferOffsetAlignment !== 0
    ) {
      throw new Error(
        `${descriptor.label} has an invalid pooled parameter binding`,
      );
    }
    return this.createBoundDispatch(descriptor, parameters);
  }

  private createBoundDispatch(
    descriptor: TransientPaletteExpansionDescriptor,
    parameters: GPUBufferBinding,
  ): PaletteExpansionDispatch {
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: descriptor.source },
        { binding: 1, resource: descriptor.palette },
        { binding: 2, resource: descriptor.target },
        { binding: 3, resource: parameters },
      ],
    });
    const codesPerGroup =
      descriptor.paletteBits === 5 ? 32 : 16;
    const workgroups = Math.ceil(
      descriptor.scalarCount /
        codesPerGroup /
        WORKGROUP_SIZE,
    );
    const pipeline = this.pipelines.get(
      pipelineKey(
        descriptor.paletteBits,
        descriptor.runtimeLayout,
      ),
    );
    if (pipeline === undefined) {
      throw new Error(
        `${descriptor.label} has unsupported expanded layout ` +
          descriptor.runtimeLayout,
      );
    }
    let destroyed = false;
    return {
      label: descriptor.label,
      encode(encoder: GPUCommandEncoder): void {
        if (destroyed) {
          throw new Error(
            `${descriptor.label} palette dispatch was destroyed`,
          );
        }
        const pass = encoder.beginComputePass({
          label: descriptor.label,
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
      },
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
      },
    };
  }
}

function parameterValues(
  descriptor: TransientPaletteExpansionDescriptor,
): Uint32Array<ArrayBuffer> {
  return new Uint32Array([
    descriptor.scalarCount,
    descriptor.columns,
    descriptor.paletteGroupColumns,
    descriptor.paletteGroups,
  ]);
}

function parameterKey(
  descriptor: TransientPaletteExpansionDescriptor,
): string {
  return [
    descriptor.paletteBits,
    descriptor.runtimeLayout,
    descriptor.scalarCount,
    descriptor.columns,
    descriptor.paletteGroupColumns,
    descriptor.paletteGroups,
  ].join(":");
}

function requireTransientDescriptor(
  descriptor: TransientPaletteExpansionDescriptor,
  executionProfile: ParakeetExecutionProfile,
): void {
  const codesPerGroup =
    descriptor.paletteBits === 5 ? 32 : 16;
  const wordsPerGroup =
    descriptor.paletteBits === 5 ? 5 : 3;
  const paletteEntries = 1 << descriptor.paletteBits;
  const sourceBytes =
    descriptor.scalarCount /
    codesPerGroup *
    wordsPerGroup *
    Uint32Array.BYTES_PER_ELEMENT;
  const paletteBytes =
    descriptor.paletteGroups *
    paletteEntries *
    scalarBytes(executionProfile);
  const targetBytes =
    descriptor.scalarCount *
    scalarBytes(executionProfile);
  if (
    !Number.isSafeInteger(descriptor.scalarCount) ||
    descriptor.scalarCount <= 0 ||
    descriptor.scalarCount % codesPerGroup !== 0 ||
    !Number.isSafeInteger(descriptor.columns) ||
    descriptor.columns <= 0 ||
    descriptor.scalarCount % descriptor.columns !== 0 ||
    !Number.isSafeInteger(descriptor.paletteGroupColumns) ||
    descriptor.paletteGroupColumns <= 0 ||
    descriptor.columns % descriptor.paletteGroupColumns !== 0 ||
    !Number.isSafeInteger(descriptor.paletteGroups) ||
    descriptor.paletteGroups <= 0 ||
    descriptor.paletteGroupColumns * descriptor.paletteGroups !==
      descriptor.columns ||
    descriptor.source.size !== sourceBytes ||
    descriptor.palette.size !== paletteBytes ||
    descriptor.target.size !== targetBytes
  ) {
    throw new Error(
      `${descriptor.label} has invalid transient palette geometry`,
    );
  }
  if (
    executionProfile.precision === "fp16"
      ? !isF16RuntimeLayout(descriptor.runtimeLayout)
      : !isF32RuntimeLayout(descriptor.runtimeLayout)
  ) {
    throw new Error(
      `${descriptor.label} has a mismatched palette execution layout`,
    );
  }
  if (isTileMajorRuntimeLayout(descriptor.runtimeLayout)) {
    const inner = descriptor.scalarCount / descriptor.columns;
    const tileColumns = tileColumnsForRuntimeLayout(
      descriptor.runtimeLayout,
    );
    if (
      inner % 32 !== 0 ||
      descriptor.columns % tileColumns !== 0
    ) {
      throw new Error(
        `${descriptor.label} has invalid tile-major geometry`,
      );
    }
  }
}

async function createPipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  bits: 5 | 6,
  runtimeLayout: PaletteRuntimeLayout,
  executionProfile: ParakeetExecutionProfile,
): Promise<GPUComputePipeline> {
  const label = executionProfile.precision === "fp16"
    ? `parakeet-palette${bits}-${runtimeLayout}-expansion`
    : `parakeet-palette${bits}-${runtimeLayout}-fp32-expansion`;
  const module = device.createShaderModule({
    label,
    code: executionProfile.precision === "fp16"
      ? paletteDequantWgsl(
          bits,
          runtimeLayout as F16PaletteRuntimeLayout,
        )
      : paletteDequantFp32Wgsl(
          bits,
          runtimeLayout as F32PaletteRuntimeLayout,
        ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (errors.length > 0) {
    throw new Error(
      `${label} WGSL failed: ` +
        errors
          .map(
            (error) =>
              `${error.lineNum}:${error.linePos} ${error.message}`,
          )
          .join("; "),
    );
  }
  return device.createComputePipelineAsync({
    label,
    layout,
    compute: { module, entryPoint: "main" },
  });
}

function scalarBytes(
  executionProfile: ParakeetExecutionProfile,
): number {
  return executionProfile.precision === "fp16"
    ? Uint16Array.BYTES_PER_ELEMENT
    : Float32Array.BYTES_PER_ELEMENT;
}

export function paletteDequantWgsl(
  bits: 5 | 6,
  runtimeLayout: F16PaletteRuntimeLayout,
): string {
  return paletteDequantWgslForScalar(
    bits,
    runtimeLayout,
    "f16",
  );
}

export function paletteDequantFp32Wgsl(
  bits: 5 | 6,
  runtimeLayout: F32PaletteRuntimeLayout,
): string {
  return paletteDequantWgslForScalar(
    bits,
    runtimeLayout,
    "f32",
  );
}

function paletteDequantWgslForScalar(
  bits: 5 | 6,
  runtimeLayout: PaletteRuntimeLayout,
  scalarType: "f16" | "f32",
): string {
  if (
    bits === 6 &&
    isRowMajorRuntimeLayout(runtimeLayout)
  ) {
    throw new Error(
      "The fixed runtime has no row-major palette6 matrix",
    );
  }
  const codesPerGroup = bits === 5 ? 32 : 16;
  const wordsPerGroup = bits === 5 ? 5 : 3;
  const paletteEntries = 1 << bits;
  const mask = paletteEntries - 1;
  const wordLoads = Array.from(
    { length: wordsPerGroup },
    (_, index) =>
      `  let word${index} = palette_indices[word_base + ${index}u];`,
  ).join("\n");
  const codeExpression = (index: number): string => {
    const bit = index * bits;
    const word = Math.floor(bit / 32);
    const shift = bit % 32;
    if (shift + bits <= 32) {
      return `((word${word} >> ${shift}u) & ${mask}u)`;
    }
    return (
      `(((word${word} >> ${shift}u) | ` +
      `(word${word + 1} << ${32 - shift}u)) & ${mask}u)`
    );
  };
  const stores = Array.from(
    { length: codesPerGroup / 4 },
    (_, vector) => {
      const values = Array.from({ length: 4 }, (_, component) => {
        const index = vector * 4 + component;
        return `palette_value(palette_base + ${codeExpression(index)})`;
      });
      const scalarOffset = vector * 4;
      const outputVector =
        isRowMajorRuntimeLayout(runtimeLayout)
          ? `scalar_base / 4u + ${vector}u`
          : `tile_major_vector(scalar_base + ${scalarOffset}u)`;
      return (
        `  dequantized[${outputVector}] = ` +
        `vec4<${scalarType}>(${values.join(", ")});`
      );
    },
  ).join("\n");
  return /* wgsl */ `
${scalarType === "f16" ? "enable f16;\n\n" : ""}struct Params {
  scalar_count: u32,
  columns: u32,
  palette_group_columns: u32,
  palette_groups: u32,
};

@group(0) @binding(0) var<storage, read> palette_indices: array<u32>;
@group(0) @binding(1) var<storage, read> palette_values: array<vec4<${scalarType}>>;
@group(0) @binding(2) var<storage, read_write> dequantized: array<vec4<${scalarType}>>;
@group(0) @binding(3) var<uniform> params: Params;

fn palette_group(scalar_index: u32) -> u32 {
  let column = scalar_index % params.columns;
  return min(
    column / params.palette_group_columns,
    params.palette_groups - 1u
  );
}

fn palette_value(element: u32) -> ${scalarType} {
  return palette_values[element / 4u][element & 3u];
}
${isTileMajorRuntimeLayout(runtimeLayout)
    ? /* wgsl */ `
fn tile_major_vector(scalar_index: u32) -> u32 {
  let inner = params.scalar_count / params.columns;
  let k = scalar_index / params.columns;
  let n = scalar_index % params.columns;
  let n_tile = n / ${tileColumnsForRuntimeLayout(runtimeLayout)}u;
  let k_tile = k / 32u;
  let k_tiles = inner / 32u;
  let k_in_tile = k % 32u;
  let n_in_tile = n % ${tileColumnsForRuntimeLayout(runtimeLayout)}u;
  return (
    ((n_tile * k_tiles + k_tile) * 32u + k_in_tile) *
      ${tileColumnsForRuntimeLayout(runtimeLayout)}u + n_in_tile
  ) / 4u;
}
`
    : ""}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let group = global.x;
  let scalar_base = group * ${codesPerGroup}u;
  if (scalar_base >= params.scalar_count) {
    return;
  }
  let word_base = group * ${wordsPerGroup}u;
  let palette_base =
    palette_group(scalar_base) * ${paletteEntries}u;
${wordLoads}
${stores}
}
`;
}

function isF16RuntimeLayout(
  runtimeLayout: PaletteRuntimeLayout,
): runtimeLayout is F16PaletteRuntimeLayout {
  return (
    runtimeLayout === RUNTIME_ROW_MAJOR_F16_LAYOUT ||
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
  );
}

function isF32RuntimeLayout(
  runtimeLayout: PaletteRuntimeLayout,
): runtimeLayout is F32PaletteRuntimeLayout {
  return (
    runtimeLayout === RUNTIME_ROW_MAJOR_F32_LAYOUT ||
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
  );
}

function isRowMajorRuntimeLayout(
  runtimeLayout: PaletteRuntimeLayout,
): boolean {
  return (
    runtimeLayout === RUNTIME_ROW_MAJOR_F16_LAYOUT ||
    runtimeLayout === RUNTIME_ROW_MAJOR_F32_LAYOUT
  );
}

function isTileMajorRuntimeLayout(
  runtimeLayout: PaletteRuntimeLayout,
): boolean {
  return (
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT ||
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
  );
}

function tileColumnsForRuntimeLayout(
  runtimeLayout: PaletteRuntimeLayout,
): 128 | 256 {
  if (
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
  ) {
    return 128;
  }
  if (
    runtimeLayout === RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
  ) {
    return 256;
  }
  throw new Error(
    `Runtime layout ${runtimeLayout} is not tile-major`,
  );
}

function pipelineKey(
  bits: 5 | 6,
  runtimeLayout: string,
): string {
  return `${bits}:${runtimeLayout}`;
}

function storageEntry(
  binding: number,
  readOnly: boolean,
): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: readOnly ? "read-only-storage" : "storage",
    },
  };
}
