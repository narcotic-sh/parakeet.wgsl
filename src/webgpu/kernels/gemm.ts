/// <reference types="@webgpu/types" />

import type { ArenaSlice } from "../arena";
import { GpuActivationArena } from "../arena";
import type { GpuTensor } from "../../model/package";
import {
  PaletteExpansionParameterPool,
  PaletteWeightExpander,
  paletteExpansionParameterPoolBytes,
  type PaletteExpansionDispatch,
} from "../../model/palette-expander";
import {
  paletteBitsForLayout,
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
  runtimeGemmTileMajorF32StorageShape,
  runtimeGemmTileMajorStorageShape,
} from "../../model/runtime-plan";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../capabilities";
import type { EncoderDispatchShape } from "./execution-shape";

export {
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
} from "../../model/runtime-plan";

export const F16_SUBGROUP_TILE_COLUMNS = 256;
export const F16_SUBGROUP_TILE_INNER = 32;
export const F16_SUBGROUP_DIRECT_B_TILE_ROWS = 32;
export const F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE = 128;
export const F16_SUBGROUP_STAGED_B_TILE_ROWS = 48;
export const F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE = 192;
const SUBGROUP_SIZE = 32;
const PACKED_K = 4;
const COLUMN_VECTORS_PER_TILE = F16_SUBGROUP_TILE_COLUMNS / PACKED_K;
const COLUMN_VECTORS_PER_LANE =
  COLUMN_VECTORS_PER_TILE / SUBGROUP_SIZE;
const B_PACKS_PER_ROW = F16_SUBGROUP_TILE_COLUMNS / 8;
const B_PACKS_PER_TILE =
  F16_SUBGROUP_TILE_INNER * B_PACKS_PER_ROW;
export const TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES =
  8_650_752;
export const TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT = 9;
export const F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES =
  B_PACKS_PER_TILE * 4 * Uint32Array.BYTES_PER_ELEMENT;

interface PortableGemmGeometry {
  readonly tileRows: number;
  readonly tileColumns: number;
  readonly tileInner: number;
  readonly workgroupSize: number;
}

/** FP16 direct encoder weights favor a deeper contraction tile. */
export const PORTABLE_DIRECT_GEMM_GEOMETRY = Object.freeze({
  tileRows: 48,
  tileColumns: 128,
  tileInner: 128,
  workgroupSize: 128,
}) satisfies PortableGemmGeometry;

/** FP16 row-major weights keep their staged contraction tile bounded. */
export const PORTABLE_STAGED_GEMM_GEOMETRY = Object.freeze({
  tileRows: 48,
  tileColumns: 128,
  tileInner: 32,
  workgroupSize: 128,
}) satisfies PortableGemmGeometry;

export const PORTABLE_FP32_DIRECT_GEMM_GEOMETRY = Object.freeze({
  tileRows: 40,
  tileColumns: 128,
  tileInner: 128,
  workgroupSize: 128,
}) satisfies PortableGemmGeometry;

export const PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY = Object.freeze({
  tileRows: 36,
  tileColumns: 128,
  tileInner: 128,
  workgroupSize: 128,
}) satisfies PortableGemmGeometry;

export const PORTABLE_FP32_STAGED_GEMM_GEOMETRY = Object.freeze({
  tileRows: 32,
  tileColumns: 128,
  tileInner: 32,
  workgroupSize: 128,
}) satisfies PortableGemmGeometry;

export const F16_PORTABLE_GEMM_WORKGROUP_BYTES = 12_288;
export const F32_PORTABLE_GEMM_WORKGROUP_BYTES = 20_480;

export const F16_PORTABLE_BITCAST_LOAD_ENCODING =
  "bitcast-vec2-u32-to-vec4-f16" as const;
export const F16_PORTABLE_UNPACK_LOAD_ENCODING =
  "unpack2x16float-via-f32" as const;
export const F16_PORTABLE_NATIVE_LOAD_ENCODING =
  "native-vec4-f16-storage" as const;
export type F16PortableLoadEncoding =
  | typeof F16_PORTABLE_BITCAST_LOAD_ENCODING
  | typeof F16_PORTABLE_UNPACK_LOAD_ENCODING
  | typeof F16_PORTABLE_NATIVE_LOAD_ENCODING;

const F16_PORTABLE_BITCAST_PROBE_WGSL = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write>
  probe_output: array<vec4<f16>>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let packed = vec2<u32>(0u);
  probe_output[0] = bitcast<vec4<f16>>(packed);
}
`;

export function transientPaletteParameterPoolBytes(
  minUniformBufferOffsetAlignment: number,
): number {
  return paletteExpansionParameterPoolBytes(
    TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT,
    minUniformBufferOffsetAlignment,
  );
}

interface F16SubgroupGemmGeometry {
  readonly tileRows: number;
  readonly workgroupSize: number;
  readonly subgroupCount: number;
  readonly rowsPerSubgroup: number;
}

const DIRECT_B_GEOMETRY = createF16SubgroupGemmGeometry(
  F16_SUBGROUP_DIRECT_B_TILE_ROWS,
  F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
);
const STAGED_B_GEOMETRY = createF16SubgroupGemmGeometry(
  F16_SUBGROUP_STAGED_B_TILE_ROWS,
  F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
);

export type GemmActivation = "none" | "silu" | "relu";

export interface F16SubgroupGemmSpecialization {
  readonly label: string;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly activation: GemmActivation;
  readonly hasBias: boolean;
  readonly residualScale: number | null;
  readonly weightLayout:
    | typeof RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
    | typeof RUNTIME_ROW_MAJOR_F16_LAYOUT;
}

function createF16SubgroupGemmGeometry(
  tileRows: number,
  workgroupSize: number,
): F16SubgroupGemmGeometry {
  const subgroupCount = workgroupSize / SUBGROUP_SIZE;
  const rowsPerSubgroup = tileRows / subgroupCount;
  if (
    !Number.isSafeInteger(subgroupCount) ||
    subgroupCount <= 0 ||
    !Number.isSafeInteger(rowsPerSubgroup) ||
    rowsPerSubgroup !== 8
  ) {
    throw new Error(
      `Invalid scalar GEMM geometry M${tileRows}/WG${workgroupSize}`,
    );
  }
  return {
    tileRows,
    workgroupSize,
    subgroupCount,
    rowsPerSubgroup,
  };
}

function geometryForWeightLayout(
  weightLayout: F16SubgroupGemmSpecialization["weightLayout"],
): F16SubgroupGemmGeometry {
  if (weightLayout === RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT) {
    return DIRECT_B_GEOMETRY;
  }
  if (weightLayout === RUNTIME_ROW_MAJOR_F16_LAYOUT) {
    return STAGED_B_GEOMETRY;
  }
  throw new Error(`Unsupported scalar GEMM weight layout ${weightLayout}`);
}

/**
 * Every scalar GEMM instantiated by the fixed B40 encoder graph. The two
 * 1024x1024 residual call sites intentionally share one compiled pipeline.
 */
export const F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS =
  [
    {
      label: "encoder-layer-ff-expand",
      rows: 7_520,
      inner: 1_024,
      columns: 4_096,
      activation: "silu",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    },
    {
      label: "encoder-layer-ff-contract",
      rows: 7_520,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: false,
      residualScale: 0.5,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    },
    {
      label: "encoder-layer-attention-qkv",
      rows: 7_520,
      inner: 1_024,
      columns: 3_072,
      activation: "none",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    },
    {
      label: "encoder-layer-residual-1024",
      rows: 7_520,
      inner: 1_024,
      columns: 1_024,
      activation: "none",
      hasBias: false,
      residualScale: 1,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    },
    {
      label: "encoder-layer-convolution-expand",
      rows: 7_520,
      inner: 1_024,
      columns: 2_048,
      activation: "none",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    },
    {
      label: "encoder-projector",
      rows: 7_520,
      inner: 1_024,
      columns: 640,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise3",
      rows: 12_032,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise3-b2",
      rows: 24_064,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise3-b3",
      rows: 36_096,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise6",
      rows: 3_008,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise6-b2",
      rows: 6_016,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-pointwise6-b3",
      rows: 9_024,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-flatten-project",
      rows: 188,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-flatten-project-b2",
      rows: 376,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
    {
      label: "subsampling-flatten-project-b3",
      rows: 564,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F16_LAYOUT,
    },
  ] as const satisfies readonly F16SubgroupGemmSpecialization[];

export interface GemmDescriptor {
  readonly label: string;
  readonly a: ArenaSlice;
  readonly b: GpuTensor;
  readonly bPalette?: GpuTensor;
  readonly paletteGroupColumns?: number;
  readonly output: ArenaSlice;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly bias?: GpuTensor;
  readonly residual?: ArenaSlice;
  readonly residualScale?: number;
  readonly activation?: GemmActivation;
  readonly preparedPaletteWeight?: PreparedPaletteWeight;
}

export interface PreparedPaletteWeight {
  readonly source: GpuTensor;
  readonly palette: GpuTensor;
  readonly paletteGroupColumns: number;
  readonly inner: number;
  readonly columns: number;
  readonly runtimeLayout:
    F16SubgroupGemmSpecialization["weightLayout"];
  readonly binding: GPUBufferBinding;
  readonly preparation: PaletteExpansionDispatch;
}

interface SpecializedGemmPipeline {
  readonly specialization: F16SubgroupGemmSpecialization;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
}

interface F16DirectBindingSizes {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly optional: number | undefined;
}

export class GemmDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: ArenaSlice,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly workgroups: readonly [number, number, number],
    private readonly rows: number,
    private readonly inner: number,
    private readonly columns: number,
    private readonly weightLayout:
      F16SubgroupGemmSpecialization["weightLayout"],
    private readonly kernelBackend:
      ParakeetExecutionProfile["kernelBackend"],
    private readonly preparation?: PaletteExpansionDispatch,
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    shape?: EncoderDispatchShape,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    this.preparation?.encode(encoder);
    let workgroups = this.workgroups;
    if (shape !== undefined) {
      const activeRows = shape.rows;
      if (
        !Number.isSafeInteger(activeRows) ||
        activeRows <= 0 ||
        activeRows > this.rows
      ) {
        throw new RangeError(
          `${this.label} active rows must be in [1, ${this.rows}]`,
        );
      }
      workgroups = this.kernelBackend === "subgroups"
        ? planF16SubgroupGemm(
            activeRows,
            this.inner,
            this.columns,
            this.weightLayout,
          ).workgroups
        : planPortableGemm(
            activeRows,
            this.inner,
            this.columns,
            this.weightLayout,
          ).workgroups;
    }
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setBindGroup(0, this.bindGroup);
    if (hasWorkgroups(workgroups)) {
      pass.setPipeline(this.pipeline);
      pass.dispatchWorkgroups(...workgroups);
    }
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.preparation?.destroy();
  }
}

function hasWorkgroups(
  workgroups: readonly [number, number, number],
): boolean {
  return workgroups[0] > 0 && workgroups[1] > 0 && workgroups[2] > 0;
}

/**
 * Fixed-shape FP16 GEMM owner. The subgroup profile retains the established
 * direct and staged pipelines; the portable profile selects its separately
 * tuned standards-only pipelines without changing weight preparation.
 */
export class F16GemmKernel {
  private destroyed = false;
  private transientPaletteDispatchCountValue = 0;
  private transientPaletteExpandedBytesValue = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly specializedPipelines: ReadonlyMap<
      string,
      SpecializedGemmPipeline
    >,
    private readonly paletteExpander: PaletteWeightExpander,
    private readonly paletteParameterPool:
      PaletteExpansionParameterPool,
    private readonly dequantizedWeightScratch: GPUBuffer,
    private readonly executionProfile: ParakeetExecutionProfile,
    readonly portableLoadEncoding: F16PortableLoadEncoding | null,
  ) {}

  get transientPaletteParameterBytes(): number {
    return this.paletteParameterPool.byteLength;
  }

  get transientPaletteParameterUsedSlots(): number {
    return this.paletteParameterPool.usedSlots;
  }

  get transientPaletteDispatchCount(): number {
    return this.transientPaletteDispatchCountValue;
  }

  get transientPaletteExpandedBytes(): number {
    return this.transientPaletteExpandedBytesValue;
  }

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
    diagnosticPortableLoadEncoding?: F16PortableLoadEncoding,
  ): Promise<F16GemmKernel> {
    if (executionProfile.precision !== "fp16") {
      throw new Error("Parakeet FP16 GEMM requires an FP16 execution profile");
    }
    if (!device.features.has("shader-f16")) {
      throw new Error("Parakeet GEMM requires WebGPU shader-f16");
    }
    if (
      executionProfile.kernelBackend === "subgroups" &&
      !device.features.has("subgroups")
    ) {
      throw new Error("Parakeet GEMM requires WebGPU subgroups");
    }
    if (
      executionProfile.kernelBackend === "subgroups" &&
      diagnosticPortableLoadEncoding !== undefined
    ) {
      throw new Error(
        "A diagnostic portable FP16 load encoding requires the portable backend",
      );
    }
    const requiredWorkgroupSize =
      executionProfile.kernelBackend === "subgroups"
        ? Math.max(
            DIRECT_B_GEOMETRY.workgroupSize,
            STAGED_B_GEOMETRY.workgroupSize,
          )
        : Math.max(
            PORTABLE_DIRECT_GEMM_GEOMETRY.workgroupSize,
            PORTABLE_STAGED_GEMM_GEOMETRY.workgroupSize,
          );
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        requiredWorkgroupSize ||
      device.limits.maxComputeWorkgroupSizeX <
        requiredWorkgroupSize
    ) {
      throw new Error(
        `Parakeet GEMM requires ${requiredWorkgroupSize} workgroup lanes`,
      );
    }
    const requiredWorkgroupStorage =
      executionProfile.kernelBackend === "subgroups"
        ? F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES
        : F16_PORTABLE_GEMM_WORKGROUP_BYTES;
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      requiredWorkgroupStorage
    ) {
      throw new Error(
        `Parakeet GEMM requires ${requiredWorkgroupStorage} bytes of workgroup storage`,
      );
    }
    const portableLoadEncoding =
      executionProfile.kernelBackend === "portable"
        ? diagnosticPortableLoadEncoding ??
          (await selectF16PortableLoadEncoding(device))
        : null;
    const [specializedPipelines, paletteExpander] =
      await Promise.all([
        createSpecializedGemmPipelines(
          device,
          executionProfile.kernelBackend,
          portableLoadEncoding,
        ),
        PaletteWeightExpander.create(device, executionProfile),
      ]);
    const dequantizedWeightScratch = device.createBuffer({
      label: "parakeet-transient-palette-weight-scratch",
      size: TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
      usage: GPUBufferUsage.STORAGE,
    });
    try {
      const paletteParameterPool =
        new PaletteExpansionParameterPool(
          device,
          TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT,
          "parakeet-transient-palette-parameter-pool",
          executionProfile,
        );
      return new F16GemmKernel(
        device,
        arena,
        specializedPipelines,
        paletteExpander,
        paletteParameterPool,
        dequantizedWeightScratch,
        executionProfile,
        portableLoadEncoding,
      );
    } catch (error) {
      dequantizedWeightScratch.destroy();
      throw error;
    }
  }

  createDispatch(descriptor: GemmDescriptor): GemmDispatch {
    if (this.destroyed) {
      throw new Error("Parakeet GEMM kernel was destroyed");
    }
    requireGemmShape(descriptor);
    const specialized = requireProductionSpecialization(
      this.specializedPipelines,
      descriptor,
    );
    requireRuntimeWeight(descriptor, specialized.specialization);
    const plan = this.executionProfile.kernelBackend === "subgroups"
      ? planF16SubgroupGemm(
          descriptor.rows,
          descriptor.inner,
          descriptor.columns,
          specialized.specialization.weightLayout,
        )
      : planPortableGemm(
          descriptor.rows,
          descriptor.inner,
          descriptor.columns,
          specialized.specialization.weightLayout,
        );
    const optionalBinding =
      descriptor.bias?.binding ??
      (descriptor.residual === undefined
        ? undefined
        : this.arena.binding(descriptor.residual));
    const paletteBits = paletteBitsForLayout(
      descriptor.b.runtimeRecord.layout,
    );
    let preparation: PaletteExpansionDispatch | undefined;
    try {
      const prepared = descriptor.preparedPaletteWeight;
      const weightBinding =
        prepared === undefined
          ? this.createInlinePaletteWeight(
              descriptor,
              specialized.specialization,
              paletteBits,
            )
          : this.requirePreparedPaletteWeight(
              descriptor,
              specialized.specialization,
              prepared,
            );
      preparation = weightBinding.preparation;
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: specialized.layout,
        entries: [
          {
            binding: 0,
            resource: this.arena.binding(descriptor.a),
          },
          { binding: 1, resource: weightBinding.binding },
          {
            binding: 2,
            resource: this.arena.binding(descriptor.output),
          },
          ...(optionalBinding === undefined
            ? []
            : [{ binding: 3, resource: optionalBinding }]),
        ],
      });
      const dispatch = new GemmDispatch(
        descriptor.label,
        descriptor.output,
        specialized.pipeline,
        bindGroup,
        plan.workgroups,
        descriptor.rows,
        descriptor.inner,
        descriptor.columns,
        specialized.specialization.weightLayout,
        this.executionProfile.kernelBackend,
        preparation,
      );
      return dispatch;
    } catch (error) {
      preparation?.destroy();
      throw error;
    }
  }

  createPreparedPaletteWeight(
    descriptor: GemmDescriptor,
    scratchOffset: number,
  ): PreparedPaletteWeight {
    if (this.destroyed) {
      throw new Error("Parakeet GEMM kernel was destroyed");
    }
    if (descriptor.preparedPaletteWeight !== undefined) {
      throw new Error(
        `${descriptor.label} cannot prepare an already prepared weight`,
      );
    }
    requireGemmShape(descriptor);
    const specialized = requireProductionSpecialization(
      this.specializedPipelines,
      descriptor,
    );
    requireRuntimeWeight(descriptor, specialized.specialization);
    const paletteBits = paletteBitsForLayout(
      descriptor.b.runtimeRecord.layout,
    );
    if (paletteBits === undefined) {
      throw new Error(
        `${descriptor.label} is not a palette-compressed weight`,
      );
    }
    const created = this.createPaletteWeight(
      descriptor,
      specialized.specialization,
      paletteBits,
      scratchOffset,
    );
    return {
      source: descriptor.b,
      palette: descriptor.bPalette!,
      paletteGroupColumns: descriptor.paletteGroupColumns!,
      inner: descriptor.inner,
      columns: descriptor.columns,
      runtimeLayout: specialized.specialization.weightLayout,
      binding: created.binding,
      preparation: created.preparation!,
    };
  }

  private createInlinePaletteWeight(
    descriptor: GemmDescriptor,
    specialization: F16SubgroupGemmSpecialization,
    paletteBits: 5 | 6 | undefined,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation?: PaletteExpansionDispatch;
  } {
    if (paletteBits === undefined) {
      return { binding: descriptor.b.binding };
    }
    return this.createPaletteWeight(
      descriptor,
      specialization,
      paletteBits,
      0,
    );
  }

  private createPaletteWeight(
    descriptor: GemmDescriptor,
    specialization: F16SubgroupGemmSpecialization,
    paletteBits: 5 | 6,
    scratchOffset: number,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation: PaletteExpansionDispatch;
  } {
    const size =
      descriptor.inner *
      descriptor.columns *
      Uint16Array.BYTES_PER_ELEMENT;
    if (
      !Number.isSafeInteger(scratchOffset) ||
      scratchOffset < 0 ||
      scratchOffset %
        this.device.limits.minStorageBufferOffsetAlignment !==
        0 ||
      scratchOffset + size >
        TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
    ) {
      throw new RangeError(
        `${descriptor.label} has an invalid transient palette scratch slice`,
      );
    }
    const binding = {
      buffer: this.dequantizedWeightScratch,
      offset: scratchOffset,
      size,
    } satisfies GPUBufferBinding;
    const bPalette = descriptor.bPalette!;
    const expansionDescriptor = {
      label: `${descriptor.label}-palette-expand`,
      source: descriptor.b.binding,
      palette: bPalette.binding,
      target: binding,
      scalarCount: descriptor.inner * descriptor.columns,
      columns: descriptor.columns,
      paletteBits,
      paletteGroupColumns: descriptor.paletteGroupColumns!,
      paletteGroups: bPalette.runtimeRecord.storageShape[0]!,
      runtimeLayout: specialization.weightLayout,
    } as const;
    const preparation =
      this.paletteExpander.createPooledDispatch(
        expansionDescriptor,
        this.paletteParameterPool.bindingFor(
          expansionDescriptor,
        ),
      );
    this.transientPaletteDispatchCountValue += 1;
    this.transientPaletteExpandedBytesValue += size;
    return { binding, preparation };
  }

  private requirePreparedPaletteWeight(
    descriptor: GemmDescriptor,
    specialization: F16SubgroupGemmSpecialization,
    prepared: PreparedPaletteWeight,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation?: PaletteExpansionDispatch;
  } {
    if (
      prepared.source !== descriptor.b ||
      prepared.palette !== descriptor.bPalette ||
      prepared.paletteGroupColumns !==
        descriptor.paletteGroupColumns ||
      prepared.inner !== descriptor.inner ||
      prepared.columns !== descriptor.columns ||
      prepared.runtimeLayout !== specialization.weightLayout
    ) {
      throw new Error(
        `${descriptor.label} has a mismatched prepared palette weight`,
      );
    }
    return { binding: prepared.binding };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.paletteParameterPool.destroy();
    this.dequantizedWeightScratch.destroy();
  }
}

export interface F16SubgroupGemmPlan {
  readonly workgroups: readonly [number, number, 1];
}

export interface PortableGemmPlan {
  readonly workgroups: readonly [number, number, 1];
}

type PortableGemmWeightLayout =
  | F16SubgroupGemmSpecialization["weightLayout"]
  | F32SubgroupGemmSpecialization["weightLayout"];

function portableGeometryForGemm(
  weightLayout: PortableGemmWeightLayout,
  inner: number,
  columns: number,
): PortableGemmGeometry {
  if (weightLayout === RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT) {
    return PORTABLE_DIRECT_GEMM_GEOMETRY;
  }
  if (
    weightLayout === RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
  ) {
    return inner === 1_024 &&
        (columns === 2_048 || columns === 3_072)
      ? PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY
      : PORTABLE_FP32_DIRECT_GEMM_GEOMETRY;
  }
  if (weightLayout === RUNTIME_ROW_MAJOR_F32_LAYOUT) {
    return PORTABLE_FP32_STAGED_GEMM_GEOMETRY;
  }
  if (
    weightLayout === RUNTIME_ROW_MAJOR_F16_LAYOUT
  ) {
    return PORTABLE_STAGED_GEMM_GEOMETRY;
  }
  throw new Error(`Unsupported portable GEMM weight layout ${weightLayout}`);
}

/**
 * Workgroup-only GEMM used when the adapter has no compatible subgroup
 * surface. Both precisions retain their operation-native package layouts and
 * dispatch inventory.
 */
export function planPortableGemm(
  rows: number,
  inner: number,
  columns: number,
  weightLayout: PortableGemmWeightLayout,
): PortableGemmPlan {
  const geometry = portableGeometryForGemm(
    weightLayout,
    inner,
    columns,
  );
  if (
    !Number.isSafeInteger(rows) ||
    rows <= 0 ||
    !Number.isSafeInteger(inner) ||
    inner <= 0 ||
    inner % geometry.tileInner !== 0 ||
    !Number.isSafeInteger(columns) ||
    columns <= 0 ||
    columns % PACKED_K !== 0
  ) {
    throw new RangeError(
      `Portable GEMM requires K divisible by ${geometry.tileInner} and N divisible by 4`,
    );
  }
  return {
    workgroups: [
      Math.ceil(columns / geometry.tileColumns),
      Math.ceil(rows / geometry.tileRows),
      1,
    ],
  };
}

/**
 * Plan the fixed scalar-subgroup kernel for its operation-native weight
 * layout. Direct dispatches target a fixed B40-capacity shader, while staged
 * geometry bounds-checks its M tail. One pipeline per fixed signature covers
 * every active B1-B40 prefix.
 */
export function planF16SubgroupGemm(
  rows: number,
  inner: number,
  columns: number,
  weightLayout: F16SubgroupGemmSpecialization["weightLayout"],
): F16SubgroupGemmPlan {
  if (
    !Number.isSafeInteger(rows) ||
    rows <= 0 ||
    !Number.isSafeInteger(inner) ||
    inner <= 0 ||
    inner % F16_SUBGROUP_TILE_INNER !== 0 ||
    !Number.isSafeInteger(columns) ||
    columns <= 0 ||
    columns % 8 !== 0
  ) {
    throw new RangeError(
      "Subgroup GEMM requires K divisible by 32 and N divisible by 8",
    );
  }
  const geometry = geometryForWeightLayout(weightLayout);
  return {
    workgroups: [
      Math.ceil(columns / F16_SUBGROUP_TILE_COLUMNS),
      Math.ceil(rows / geometry.tileRows),
      1,
    ],
  };
}

function requireGemmShape(descriptor: GemmDescriptor): void {
  if (
    !Number.isSafeInteger(descriptor.rows) ||
    descriptor.rows <= 0 ||
    !Number.isSafeInteger(descriptor.inner) ||
    descriptor.inner <= 0 ||
    descriptor.inner % F16_SUBGROUP_TILE_INNER !== 0 ||
    !Number.isSafeInteger(descriptor.columns) ||
    descriptor.columns <= 0 ||
    descriptor.columns % 8 !== 0
  ) {
    throw new RangeError(`Unsupported GEMM shape for ${descriptor.label}`);
  }
  const expectedWeightElements = descriptor.inner * descriptor.columns;
  const source = descriptor.b.sourceRecord;
  const paletteBits = paletteBitsForLayout(source.layout);
  if (
    source.logicalShape.join(",") !==
    `${descriptor.inner},${descriptor.columns}`
  ) {
    throw new Error(
      `${descriptor.label} weight package does not match K=${descriptor.inner}, N=${descriptor.columns}`,
    );
  }
  if (paletteBits === undefined || source.dtype !== "uint32") {
    throw new Error(`${descriptor.label} has an unsupported weight format`);
  }
  const codesPerGroup = paletteBits === 5 ? 32 : 16;
  const wordsPerGroup = paletteBits === 5 ? 5 : 3;
  if (expectedWeightElements % codesPerGroup !== 0) {
    throw new Error(`${descriptor.label} palette codes are not group aligned`);
  }
  const expectedGroups = expectedWeightElements / codesPerGroup;
  if (
    source.byteLength !== expectedGroups * wordsPerGroup * 4 ||
    source.storageShape.join(",") !==
      `${expectedGroups},${wordsPerGroup}`
  ) {
    throw new Error(`${descriptor.label} has invalid packed palette indices`);
  }
}

function specializationKey(
  specialization: Omit<
    F16SubgroupGemmSpecialization,
    "label" | "weightLayout"
  >,
): string {
  return [
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.activation,
    specialization.hasBias ? "bias" : "no-bias",
    specialization.residualScale === null
      ? "no-residual"
      : `residual-${specialization.residualScale}`,
  ].join(":");
}

function requireProductionSpecialization(
  pipelines: ReadonlyMap<string, SpecializedGemmPipeline>,
  descriptor: GemmDescriptor,
): SpecializedGemmPipeline {
  if (
    descriptor.residual === undefined &&
    descriptor.residualScale !== undefined
  ) {
    throw new Error(
      `${descriptor.label} supplies a residual scale without a residual`,
    );
  }
  const signature = {
    rows: descriptor.rows,
    inner: descriptor.inner,
    columns: descriptor.columns,
    activation: descriptor.activation ?? "none",
    hasBias: descriptor.bias !== undefined,
    residualScale:
      descriptor.residual === undefined
        ? null
        : (descriptor.residualScale ?? 1),
  } satisfies Omit<
    F16SubgroupGemmSpecialization,
    "label" | "weightLayout"
  >;
  const specialized = pipelines.get(specializationKey(signature));
  if (specialized === undefined) {
    throw new Error(
      `${descriptor.label} is not a fixed production GEMM signature: ` +
        `M=${signature.rows}, K=${signature.inner}, N=${signature.columns}, ` +
        `activation=${signature.activation}, bias=${signature.hasBias}, ` +
        `residualScale=${signature.residualScale ?? "none"}`,
    );
  }
  return specialized;
}

function requireRuntimeWeight(
  descriptor: GemmDescriptor,
  specialization: F16SubgroupGemmSpecialization,
): void {
  const runtime = descriptor.b.runtimeRecord;
  const paletteBits = paletteBitsForLayout(runtime.layout);
  if (paletteBits !== undefined) {
    const source = descriptor.b.sourceRecord;
    const sourcePaletteBits = paletteBitsForLayout(source.layout);
    const palette = descriptor.bPalette;
    const paletteGroupColumns =
      descriptor.paletteGroupColumns;
    const paletteEntries =
      paletteBits === undefined ? 0 : 1 << paletteBits;
    const paletteGroups =
      palette?.runtimeRecord.storageShape[0];
    if (
      sourcePaletteBits !== paletteBits ||
      runtime.dtype !== "uint32" ||
      runtime.layout !== source.layout ||
      runtime.logicalShape.join(",") !==
        `${descriptor.inner},${descriptor.columns}` ||
      runtime.storageShape.join(",") !==
        source.storageShape.join(",") ||
      runtime.byteLength !== source.byteLength ||
      palette === undefined ||
      palette.sourceRecord.dtype !== "float16" ||
      palette.sourceRecord.layout !==
        "output-group-by-palette-lut" ||
      palette.runtimeRecord.dtype !== "float16" ||
      palette.runtimeRecord.layout !==
        "output-group-by-palette-lut" ||
      palette.runtimeRecord.logicalShape.join(",") !==
        palette.sourceRecord.logicalShape.join(",") ||
      palette.runtimeRecord.storageShape.join(",") !==
        `${paletteGroups},${paletteEntries}` ||
      palette.runtimeRecord.byteLength !==
        paletteGroups! *
          paletteEntries *
          Uint16Array.BYTES_PER_ELEMENT ||
      !Number.isSafeInteger(paletteGroupColumns) ||
      paletteGroupColumns! <= 0 ||
      paletteGroupColumns! * paletteGroups! !==
        descriptor.columns
    ) {
      throw new Error(
        `${descriptor.label} has invalid transient palette weights`,
      );
    }
    const expandedBytes =
      descriptor.inner *
      descriptor.columns *
      Uint16Array.BYTES_PER_ELEMENT;
    if (
      expandedBytes >
      TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
    ) {
      throw new Error(
        `${descriptor.label} exceeds transient palette scratch`,
      );
    }
    if (
      specialization.weightLayout ===
      RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
    ) {
      runtimeGemmTileMajorStorageShape(
        descriptor.inner,
        descriptor.columns,
      );
    }
    return;
  }
  const expectedStorageShape =
    [descriptor.inner, descriptor.columns];
  if (
    runtime.dtype !== "float16" ||
    runtime.layout !== specialization.weightLayout ||
    runtime.logicalShape.join(",") !==
      `${descriptor.inner},${descriptor.columns}` ||
    runtime.storageShape.join(",") !==
      expectedStorageShape.join(",") ||
    runtime.byteLength !==
      descriptor.inner *
        descriptor.columns *
        Uint16Array.BYTES_PER_ELEMENT ||
    descriptor.bPalette !== undefined ||
    descriptor.paletteGroupColumns !== undefined
  ) {
    throw new Error(
      `${descriptor.label} has invalid ${specialization.weightLayout} weights`,
    );
  }
}

async function createSpecializedGemmPipelines(
  device: GPUDevice,
  kernelBackend: ParakeetExecutionProfile["kernelBackend"],
  portableLoadEncoding: F16PortableLoadEncoding | null,
): Promise<ReadonlyMap<string, SpecializedGemmPipeline>> {
  if (
    (kernelBackend === "portable") !==
    (portableLoadEncoding !== null)
  ) {
    throw new Error(
      "Portable FP16 GEMM load encoding does not match its kernel backend",
    );
  }
  const entries = await Promise.all(
    F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.map(
      async (specialization) => {
        const directBindingSizes =
          f16DirectBindingSizes(specialization);
        const portableSuffix =
          kernelBackend === "portable" ? "-portable" : "";
        const layout = device.createBindGroupLayout({
          label:
            `parakeet-gemm-${specialization.label}${portableSuffix}-bindings`,
          entries: [
            storageEntry(0, false, directBindingSizes?.a),
            storageEntry(1, true, directBindingSizes?.b),
            storageEntry(2, false, directBindingSizes?.c),
            ...(specialization.hasBias
              ? [
                  storageEntry(
                    3,
                    true,
                    directBindingSizes?.optional,
                  ),
                ]
              : specialization.residualScale === null
                ? []
                : [
                    storageEntry(
                      3,
                      false,
                      directBindingSizes?.optional,
                    ),
                  ]),
          ],
        });
        const pipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [layout],
        });
        const pipeline = kernelBackend === "subgroups"
          ? await createF16SubgroupPipeline(
              device,
              pipelineLayout,
              specialization,
            )
          : await createF16PortablePipeline(
              device,
              pipelineLayout,
              specialization,
              portableLoadEncoding!,
            );
        const specialized: SpecializedGemmPipeline = {
          specialization,
          pipeline,
          layout,
        };
        return [specializationKey(specialization), specialized] as const;
      },
    ),
  );
  const pipelines = new Map(entries);
  if (pipelines.size !== F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.length) {
    throw new Error("Duplicate fixed production GEMM specialization");
  }
  return pipelines;
}

async function createF16SubgroupPipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  specialization: F16SubgroupGemmSpecialization,
): Promise<GPUComputePipeline> {
  const label = `parakeet-gemm-${specialization.label}`;
  const module = device.createShaderModule({
    label,
    code: f16SubgroupGemmWgsl(specialization),
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

/**
 * Select the fastest exact portable FP16 storage view for this device's WGSL
 * compiler. Compilers accepting the standard cross-width bitcast use it;
 * a reported source-compilation error selects native vec4<f16> storage.
 * API failures and device failures deliberately surface.
 */
export async function selectF16PortableLoadEncoding(
  device: GPUDevice,
): Promise<F16PortableLoadEncoding> {
  const module = device.createShaderModule({
    label: "parakeet-fp16-portable-bitcast-load-probe",
    code: F16_PORTABLE_BITCAST_PROBE_WGSL,
  });
  const compilation = await module.getCompilationInfo();
  return compilation.messages.some(
    (message) => message.type === "error",
  )
    ? F16_PORTABLE_NATIVE_LOAD_ENCODING
    : F16_PORTABLE_BITCAST_LOAD_ENCODING;
}

async function createF16PortablePipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  specialization: F16SubgroupGemmSpecialization,
  loadEncoding: F16PortableLoadEncoding,
): Promise<GPUComputePipeline> {
  const label = `parakeet-gemm-${specialization.label}-portable`;
  const module = device.createShaderModule({
    label,
    code: f16PortableGemmWgsl(specialization, loadEncoding),
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

function f16DirectBindingSizes(
  specialization: F16SubgroupGemmSpecialization,
): F16DirectBindingSizes | undefined {
  if (
    specialization.weightLayout !==
    RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
  ) {
    return undefined;
  }
  const activationElementBytes = Uint16Array.BYTES_PER_ELEMENT;
  const a =
    specialization.rows *
    specialization.inner *
    activationElementBytes;
  const b =
    specialization.inner *
    specialization.columns *
    activationElementBytes;
  const c =
    specialization.rows *
    specialization.columns *
    activationElementBytes;
  const optional = specialization.hasBias
    ? specialization.columns * Float32Array.BYTES_PER_ELEMENT
    : specialization.residualScale === null
      ? undefined
      : c;
  return { a, b, c, optional };
}

/**
 * Emit one fixed-shape N256/K32 f16 scalar-subgroup shader. Direct tile-major
 * weights use M32/WG128; staged row-major weights use M48/WG192. In either
 * geometry every lane owns two vec4 output columns for each of eight rows in
 * its subgroup, preserving the exact f16 contraction sequence.
 */
export function f16SubgroupGemmWgsl(
  specialization: F16SubgroupGemmSpecialization,
): string {
  planF16SubgroupGemm(
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.weightLayout,
  );
  if (
    specialization.hasBias &&
    specialization.residualScale !== null
  ) {
    throw new Error(
      `${specialization.label} cannot fuse bias and residual together`,
    );
  }
  const directResidentB =
    specialization.weightLayout ===
    RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT;
  if (
    directResidentB &&
    specialization.inner !== 1_024 &&
    specialization.inner !== 4_096
  ) {
    throw new Error(
      `${specialization.label} direct-B FP16 GEMM requires K1024 or K4096`,
    );
  }
  const wideDirectA =
    directResidentB && specialization.inner === 4_096;
  const aValuesPerStorageVector = wideDirectA
    ? PACKED_K * 2
    : PACKED_K;
  const aVectorsPerRow =
    specialization.inner / aValuesPerStorageVector;
  const weightPacksPerRow = specialization.columns / 8;
  const columnVectors = specialization.columns / PACKED_K;
  const hasPartialColumnTile =
    specialization.columns % F16_SUBGROUP_TILE_COLUMNS !== 0;
  const geometry = geometryForWeightLayout(
    specialization.weightLayout,
  );
  if (
    directResidentB &&
    (specialization.inner % F16_SUBGROUP_TILE_INNER !== 0 ||
      specialization.columns % F16_SUBGROUP_TILE_COLUMNS !== 0)
  ) {
    throw new Error(
      `${specialization.label} cannot use direct K32/N256 tile-major weights`,
    );
  }
  const kTiles =
    specialization.inner / F16_SUBGROUP_TILE_INNER;
  const directArrayLength = (
    elementCount: number,
  ): string => directResidentB ? `, ${elementCount}` : "";
  const matrixAArrayLength = directArrayLength(
    specialization.rows * aVectorsPerRow,
  );
  const matrixBArrayLength = directArrayLength(
    specialization.inner * weightPacksPerRow,
  );
  const matrixCArrayLength = directArrayLength(
    specialization.rows * columnVectors,
  );
  const directEntryProof = directResidentB
    ? /* wgsl */ `
  if (
    group.x >= ${specialization.columns / F16_SUBGROUP_TILE_COLUMNS}u ||
    group.y >= ${specialization.rows / geometry.tileRows}u ||
    group.z != 0u ||
    subgroup >= ${geometry.subgroupCount}u
  ) {
    return;
  }`
    : "";
  const aLoadGuard = directResidentB
    ? `if (subgroup_lane < ${geometry.rowsPerSubgroup}u)`
    : `if (
        subgroup_lane < ${geometry.rowsPerSubgroup}u &&
        a_row < ROWS
      )`;
  const declarations = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) =>
      Array.from(
        { length: COLUMN_VECTORS_PER_LANE },
        (_, column) =>
          `  var acc${row}_${column} = vec4<f16>(0.0h);`,
      ).join("\n"),
  ).join("\n");
  const fullBroadcasts = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) =>
      `      let a${row} = bitcast<vec4<f16>>(\n` +
      `        subgroupBroadcast(packed_a, ${row}u)\n` +
      `      );`,
  ).join("\n");
  const contractionBlock = (component: number): string => {
    const fmas = Array.from(
      { length: geometry.rowsPerSubgroup },
      (_, row) =>
        Array.from(
          { length: COLUMN_VECTORS_PER_LANE },
          (_, column) => {
            const aValue = directResidentB
              ? `a${row}_${component}`
              : `a${row}[${component}]`;
            const broadcast =
              directResidentB && column === 0
                ? `      let a${row}_${component} = subgroupBroadcast(` +
                  `local_a[${component}], ${row}u);\n`
                : "";
            return (
              broadcast +
              `      acc${row}_${column} = fma(` +
              `vec4<f16>(${aValue}), b${column}, ` +
              `acc${row}_${column});`
            );
          },
        ).join("\n"),
    ).join("\n");
    const packedB = directResidentB
      ? `matrix_b[
        direct_b_tile_base +
          (k_vector * ${PACKED_K}u + ${component}u)
            * ${B_PACKS_PER_ROW}u + subgroup_lane
      ]`
      : `tile_b[
        (k_vector * ${PACKED_K}u + ${component}u)
          * ${B_PACKS_PER_ROW}u + subgroup_lane
      ]`;
    return /* wgsl */ `
    {
      let packed_b = ${packedB};
      let b0 = bitcast<vec4<f16>>(packed_b.xy);
      let b1 = bitcast<vec4<f16>>(packed_b.zw);
${fmas}
    }`;
  };
  const broadcasts = directResidentB ? "" : fullBroadcasts;
  const contractionBlocks = Array.from(
    { length: PACKED_K },
    (_, component) => contractionBlock(component),
  ).join("\n");
  const directContractionHalves = wideDirectA
    ? /* wgsl */ `
    {
      let k_vector = a_vector * 2u;
      let local_a = bitcast<vec4<f16>>(packed_a.xy);
${contractionBlocks}
    }
    {
      let k_vector = a_vector * 2u + 1u;
      let local_a = bitcast<vec4<f16>>(packed_a.zw);
${contractionBlocks}
    }`
    : "";
  const stores = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) =>
      Array.from(
        { length: COLUMN_VECTORS_PER_LANE },
        (_, column) => {
          if (directResidentB) {
            return /* wgsl */ `
  {
    let row = row_base + ${row}u;
    let column_vector = column_vector_base + ${column}u;
    let output_base = column_vector * ${PACKED_K}u;
    var value = vec4<f32>(acc${row}_${column});
${specializedBias(specialization)}
${specializedActivation(specialization.activation)}
    let output_index =
      row * COLUMN_VECTORS + column_vector;
${specializedResidual(specialization)}
    matrix_c[output_index] = vec4<f16>(value);
  }`;
          }
          return /* wgsl */ `
  {
    let row = row_base + ${row}u;
    let column_vector = column_vector_base + ${column}u;
    if (${hasPartialColumnTile
      ? "row < ROWS && column_vector < COLUMN_VECTORS"
      : "row < ROWS"}) {
      let output_base = column_vector * ${PACKED_K}u;
      var value = vec4<f32>(acc${row}_${column});
${specializedBias(specialization)}
${specializedActivation(specialization.activation)}
      let output_index =
        row * COLUMN_VECTORS + column_vector;
${specializedResidual(specialization)}
      matrix_c[output_index] = vec4<f16>(value);
    }
  }`;
        },
      ).join("\n"),
  ).join("\n");
  const optionalDeclaration = specialization.hasBias
    ? /* wgsl */ `
@group(0) @binding(3) var<storage, read> bias: array<f32>;`
    : specialization.residualScale === null
      ? ""
      : /* wgsl */ `
@group(0) @binding(3) var<storage, read_write>
  residual: array<vec4<f16>${matrixCArrayLength}>;`;
  const weightLoad = hasPartialColumnTile
    ? /* wgsl */ `    if (b_pack < WEIGHT_PACKS_PER_ROW) {
      tile_b[item] = matrix_b[
        (k_base + b_k) * WEIGHT_PACKS_PER_ROW + b_pack
      ];
    } else {
      tile_b[item] = vec4<u32>(0u);
    }`
    : /* wgsl */ `    tile_b[item] = matrix_b[
      (k_base + b_k) * WEIGHT_PACKS_PER_ROW + b_pack
    ];`;
  const workgroupDeclaration = directResidentB
    ? ""
    : `var<workgroup> tile_b:
  array<vec4<u32>, ${B_PACKS_PER_TILE}>;
`;
  const loadFunction = directResidentB
    ? ""
    : rowMajorWeightLoad(weightLoad, geometry.workgroupSize);
  const localIndexParameter = directResidentB
    ? ""
    : "  @builtin(local_invocation_index) local_index: u32,\n";
  const beginInnerTile = directResidentB
    ? `    let direct_b_tile_base =
      (group.x * ${kTiles}u +
        k_base / ${F16_SUBGROUP_TILE_INNER}u) *
      ${B_PACKS_PER_TILE}u;`
    : `    load_b_tile(k_base, local_index, group.x);
    workgroupBarrier();`;
  const endInnerTile = directResidentB
    ? ""
    : "    workgroupBarrier();";
  const nestedContractionLoop = wideDirectA
    ? /* wgsl */ `
    for (
      var a_vector = 0u;
      a_vector < ${F16_SUBGROUP_TILE_INNER / (PACKED_K * 2)}u;
      a_vector += 1u
    ) {
      var packed_a = vec4<u32>(0u);
      let a_row = row_base + subgroup_lane;
      ${aLoadGuard} {
        packed_a = matrix_a[
          a_row * A_VECTORS_PER_ROW
            + k_base / ${PACKED_K * 2}u
            + a_vector
        ];
      }
${directContractionHalves}
    }`
    : /* wgsl */ `
    for (
      var k_vector = 0u;
      k_vector < ${F16_SUBGROUP_TILE_INNER / PACKED_K}u;
      k_vector += 1u
    ) {
      var packed_a = vec2<u32>(0u);
      let a_row = row_base + subgroup_lane;
      ${aLoadGuard} {
        packed_a = matrix_a[
          a_row * A_VECTORS_PER_ROW
            + k_base / ${PACKED_K}u
            + k_vector
        ];
      }
${broadcasts}
${contractionBlocks}
    }`;
  const contractionLoop =
    directResidentB && specialization.inner === 1_024
      ? /* wgsl */ `
  let direct_b_tile_base =
    group.x * ${kTiles * B_PACKS_PER_TILE}u;
  for (
    var k_vector = 0u;
    k_vector < A_VECTORS_PER_ROW;
    k_vector += 1u
  ) {
    var packed_a = vec2<u32>(0u);
    let a_row = row_base + subgroup_lane;
    ${aLoadGuard} {
      packed_a = matrix_a[
        a_row * A_VECTORS_PER_ROW + k_vector
      ];
    }
    let local_a = bitcast<vec4<f16>>(packed_a);
${contractionBlocks}
  }`
      : /* wgsl */ `
  for (
    var k_base = 0u;
    k_base < INNER;
    k_base += ${F16_SUBGROUP_TILE_INNER}u
  ) {
${beginInnerTile}
${nestedContractionLoop}
${endInnerTile}
  }`;
  const matrixAStorageType = wideDirectA
    ? "vec4<u32>"
    : "vec2<u32>";

  return /* wgsl */ `
enable f16;
enable subgroups;

const ROWS = ${specialization.rows}u;
const INNER = ${specialization.inner}u;
const A_VECTORS_PER_ROW = ${aVectorsPerRow}u;
const WEIGHT_PACKS_PER_ROW = ${weightPacksPerRow}u;
const COLUMN_VECTORS = ${columnVectors}u;

@group(0) @binding(0) var<storage, read_write>
  matrix_a: array<${matrixAStorageType}${matrixAArrayLength}>;
@group(0) @binding(1) var<storage, read>
  matrix_b: array<vec4<u32>${matrixBArrayLength}>;
@group(0) @binding(2) var<storage, read_write>
  matrix_c: array<vec4<f16>${matrixCArrayLength}>;
${optionalDeclaration}

${workgroupDeclaration}

${loadFunction}

@compute @workgroup_size(${geometry.workgroupSize}, 1, 1)
fn main(
${localIndexParameter}  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${SUBGROUP_SIZE}u) {
    return;
  }${directEntryProof}
  let row_base =
    group.y * ${geometry.tileRows}u
      + subgroup * ${geometry.rowsPerSubgroup}u;
  let column_vector_base =
    group.x * ${COLUMN_VECTORS_PER_TILE}u
      + subgroup_lane * ${COLUMN_VECTORS_PER_LANE}u;
${declarations}
${contractionLoop}

${stores}
}
`;
}

function rowMajorWeightLoad(
  weightLoad: string,
  workgroupSize: number,
): string {
  return /* wgsl */ `fn load_b_tile(
  k_base: u32,
  local_index: u32,
  group_x: u32,
) {
  for (
    var item = local_index;
    item < ${B_PACKS_PER_TILE}u;
    item += ${workgroupSize}u
  ) {
    let b_k = item / ${B_PACKS_PER_ROW}u;
    let b_pack =
      group_x * ${B_PACKS_PER_ROW}u
        + item % ${B_PACKS_PER_ROW}u;
${weightLoad}
  }
}`;
}

/**
 * Emit the standards-only FP16 fallback. WG128 uses an M48/N128/K128 tile
 * for direct encoder weights and M48/N128/K32 for staged row-major weights.
 * Operation-native weights and increasing-K accumulator order are preserved.
 */
export function f16PortableGemmWgsl(
  specialization: F16SubgroupGemmSpecialization,
  loadEncoding: F16PortableLoadEncoding =
    F16_PORTABLE_NATIVE_LOAD_ENCODING,
): string {
  planPortableGemm(
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.weightLayout,
  );
  if (
    specialization.hasBias &&
    specialization.residualScale !== null
  ) {
    throw new Error(
      `${specialization.label} cannot fuse bias and residual together`,
    );
  }
  const directResidentB =
    specialization.weightLayout ===
    RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT;
  const nativeF16Storage =
    loadEncoding === F16_PORTABLE_NATIVE_LOAD_ENCODING;
  const packedVectorType = nativeF16Storage
    ? "vec4<f16>"
    : "vec2<u32>";
  const zeroPackedVector = nativeF16Storage
    ? "vec4<f16>(0.0h)"
    : "vec2<u32>(0u)";
  const loadPackedVector = (expression: string): string =>
    nativeF16Storage
      ? expression
      : `unpack_f16x4(${expression})`;
  const geometry = portableGeometryForGemm(
    specialization.weightLayout,
    specialization.inner,
    specialization.columns,
  );
  const columnVectorsPerTile = geometry.tileColumns / PACKED_K;
  const rowGroups = geometry.workgroupSize / columnVectorsPerTile;
  const rowsPerLane = geometry.tileRows / rowGroups;
  const aTileVectorsPerRow = geometry.tileInner / PACKED_K;
  const aTileVectors = geometry.tileRows * aTileVectorsPerRow;
  const bTileVectors = geometry.tileInner * columnVectorsPerTile;
  const aVectorsPerRow = specialization.inner / PACKED_K;
  const columnVectors = specialization.columns / PACKED_K;
  const runtimeKTiles =
    specialization.inner / F16_SUBGROUP_TILE_INNER;
  const runtimeTileColumnVectors =
    F16_SUBGROUP_TILE_COLUMNS / PACKED_K;
  const portableSlicesPerRuntimeTile =
    F16_SUBGROUP_TILE_COLUMNS / geometry.tileColumns;
  const declarations = Array.from(
    { length: rowsPerLane },
    (_, row) => `  var acc${row} = vec4<f16>(0.0h);`,
  ).join("\n");
  const aLoads = Array.from(
    { length: rowsPerLane },
    (_, row) => {
      const tileValue =
        `tile_a[(local_row_base + ${row}u) * ` +
        `${aTileVectorsPerRow}u + k_vector]`;
      return `    let a${row} = ${loadPackedVector(tileValue)};`;
    },
  ).join("\n");
  const contractionBlocks = Array.from(
    { length: PACKED_K },
    (_, component) => {
      const bLoad = directResidentB
        ? `matrix_b[
        direct_b_tile_base +
          (k_vector * ${PACKED_K}u + ${component}u) *
            ${runtimeTileColumnVectors}u +
          direct_b_slice * ${columnVectorsPerTile}u +
          local_column_vector
      ]`
        : `tile_b[
        (k_vector * ${PACKED_K}u + ${component}u) *
          ${columnVectorsPerTile}u +
          local_column_vector
      ]`;
      const fmas = Array.from(
        { length: rowsPerLane },
        (_, row) =>
          `      acc${row} = fma(` +
          `vec4<f16>(f16(a${row}[${component}])), b, acc${row});`,
      ).join("\n");
      return /* wgsl */ `
    {
      let b = ${loadPackedVector(bLoad)};
${fmas}
    }`;
    },
  ).join("\n");
  const stores = Array.from(
    { length: rowsPerLane },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + local_row_base + ${row}u;
    if (row < ROWS && column_vector < COLUMN_VECTORS) {
      let output_base = column_vector * ${PACKED_K}u;
      var value = vec4<f32>(acc${row});
${specializedBias(specialization)}
${specializedActivation(specialization.activation)}
      let output_index = row * COLUMN_VECTORS + column_vector;
${specializedResidual(specialization)}
      matrix_c[output_index] = vec4<f16>(value);
    }
  }`,
  ).join("\n");
  const optionalDeclaration = specialization.hasBias
    ? /* wgsl */ `
@group(0) @binding(3) var<storage, read> bias: array<f32>;`
    : specialization.residualScale === null
      ? ""
      : /* wgsl */ `
@group(0) @binding(3) var<storage, read_write>
  residual: array<vec4<f16>>;`;
  const stagedBDeclaration = directResidentB
    ? ""
    : `var<workgroup> tile_b:
  array<${packedVectorType}, ${bTileVectors}>;`;
  const stagedBLoad = directResidentB
    ? ""
    : /* wgsl */ `
    for (
      var item = lane;
      item < ${bTileVectors}u;
      item += ${geometry.workgroupSize}u
    ) {
      let k_scalar = item / ${columnVectorsPerTile}u;
      let local_column = item % ${columnVectorsPerTile}u;
      let source_column =
        group.x * ${columnVectorsPerTile}u + local_column;
      if (source_column < COLUMN_VECTORS) {
        tile_b[item] = matrix_b[
          (k_base + k_scalar) * COLUMN_VECTORS + source_column
        ];
      } else {
        tile_b[item] = ${zeroPackedVector};
      }
    }`;
  const directBSetup = directResidentB
    ? /* wgsl */ `
    let direct_b_tile =
      group.x / ${portableSlicesPerRuntimeTile}u;
    let direct_b_slice =
      group.x % ${portableSlicesPerRuntimeTile}u;
    let direct_b_tile_base =
      (direct_b_tile * ${runtimeKTiles}u +
        k_base / ${F16_SUBGROUP_TILE_INNER}u) *
        ${F16_SUBGROUP_TILE_INNER * runtimeTileColumnVectors}u +
      (k_base % ${F16_SUBGROUP_TILE_INNER}u) *
        ${runtimeTileColumnVectors}u;`
    : "";
  const unpackFunction =
    nativeF16Storage
      ? ""
      : loadEncoding === F16_PORTABLE_BITCAST_LOAD_ENCODING
      ? /* wgsl */ `fn unpack_f16x4(
  packed: vec2<u32>,
) -> vec4<f16> {
  return bitcast<vec4<f16>>(packed);
}`
      : /* wgsl */ `fn unpack_f16x4(
  packed: vec2<u32>,
) -> vec4<f16> {
  let xy = unpack2x16float(packed.x);
  let zw = unpack2x16float(packed.y);
  return vec4<f16>(
    f16(xy.x),
    f16(xy.y),
    f16(zw.x),
    f16(zw.y)
  );
}`;

  return /* wgsl */ `
enable f16;

const ROWS = ${specialization.rows}u;
const INNER = ${specialization.inner}u;
const A_VECTORS_PER_ROW = ${aVectorsPerRow}u;
const COLUMN_VECTORS = ${columnVectors}u;

@group(0) @binding(0) var<storage, read_write>
  matrix_a: array<${packedVectorType}>;
@group(0) @binding(1) var<storage, read>
  matrix_b: array<${packedVectorType}>;
@group(0) @binding(2) var<storage, read_write>
  matrix_c: array<vec4<f16>>;
${optionalDeclaration}

var<workgroup> tile_a:
  array<${packedVectorType}, ${aTileVectors}>;
${stagedBDeclaration}

${unpackFunction}

@compute @workgroup_size(${geometry.workgroupSize}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let local_column_vector =
    lane % ${columnVectorsPerTile}u;
  let local_row_group =
    lane / ${columnVectorsPerTile}u;
  let local_row_base =
    local_row_group * ${rowsPerLane}u;
  let row_base = group.y * ${geometry.tileRows}u;
  let column_vector =
    group.x * ${columnVectorsPerTile}u +
      local_column_vector;
${declarations}

  for (
    var k_base = 0u;
    k_base < INNER;
    k_base += ${geometry.tileInner}u
  ) {
    for (
      var item = lane;
      item < ${aTileVectors}u;
      item += ${geometry.workgroupSize}u
    ) {
      let tile_row = item / ${aTileVectorsPerRow}u;
      let k_vector = item % ${aTileVectorsPerRow}u;
      let source_row = row_base + tile_row;
      if (source_row < ROWS) {
        tile_a[item] = matrix_a[
          source_row * A_VECTORS_PER_ROW +
            k_base / ${PACKED_K}u + k_vector
        ];
      } else {
        tile_a[item] = ${zeroPackedVector};
      }
    }
${stagedBLoad}
${directBSetup}
    workgroupBarrier();

    for (
      var k_vector = 0u;
      k_vector < ${aTileVectorsPerRow}u;
      k_vector += 1u
    ) {
${aLoads}
${contractionBlocks}
    }
    if (k_base + ${geometry.tileInner}u < INNER) {
      workgroupBarrier();
    }
  }
${stores}
}
`;
}

function specializedBias(
  specialization: F16SubgroupGemmSpecialization,
): string {
  if (!specialization.hasBias) return "";
  return /* wgsl */ `      value += vec4<f32>(
        bias[output_base + 0u],
        bias[output_base + 1u],
        bias[output_base + 2u],
        bias[output_base + 3u]
      );`;
}

function specializedActivation(activation: GemmActivation): string {
  switch (activation) {
    case "none":
      return "";
    case "silu":
      return /* wgsl */ `      let bounded = clamp(
        value,
        vec4<f32>(-20.0),
        vec4<f32>(20.0)
      );
      value = value / (vec4<f32>(1.0) + exp(-bounded));`;
    case "relu":
      return "      value = max(value, vec4<f32>(0.0));";
  }
}

function specializedResidual(
  specialization: F16SubgroupGemmSpecialization,
): string {
  const scale = specialization.residualScale;
  if (scale === null) return "";
  const scaleLiteral = scale === 1 ? "1.0" : String(scale);
  return (
    "      value = vec4<f32>(residual[output_index])\n" +
    `        + ${scaleLiteral} * value;`
  );
}

function storageEntry(
  binding: number,
  readOnly: boolean,
  minBindingSize?: number,
): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: readOnly ? "read-only-storage" : "storage",
      ...(minBindingSize === undefined ? {} : { minBindingSize }),
    },
  };
}

export const F32_SUBGROUP_TILE_COLUMNS = 128;
export const F32_SUBGROUP_TILE_INNER = 32;
export const F32_SUBGROUP_DIRECT_B_TILE_ROWS = 32;
export const F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE = 128;
export const F32_SUBGROUP_STAGED_B_TILE_ROWS = 48;
export const F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE = 192;
export const FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES =
  17_301_504;

const FP32_PACKED_K = 4;
const FP32_COLUMN_VECTORS_PER_TILE =
  F32_SUBGROUP_TILE_COLUMNS / FP32_PACKED_K;
const FP32_COLUMN_VECTORS_PER_LANE =
  FP32_COLUMN_VECTORS_PER_TILE / SUBGROUP_SIZE;
const FP32_B_VECTORS_PER_ROW =
  F32_SUBGROUP_TILE_COLUMNS / FP32_PACKED_K;
const FP32_B_VECTORS_PER_TILE =
  F32_SUBGROUP_TILE_INNER * FP32_B_VECTORS_PER_ROW;

export const F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES =
  FP32_B_VECTORS_PER_TILE *
  FP32_PACKED_K *
  Float32Array.BYTES_PER_ELEMENT;

interface F32SubgroupGemmGeometry {
  readonly tileRows: number;
  readonly workgroupSize: number;
  readonly subgroupCount: number;
  readonly rowsPerSubgroup: number;
}

const FP32_DIRECT_B_GEOMETRY =
  createF32SubgroupGemmGeometry(
    F32_SUBGROUP_DIRECT_B_TILE_ROWS,
    F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
  );
const FP32_STAGED_B_GEOMETRY =
  createF32SubgroupGemmGeometry(
    F32_SUBGROUP_STAGED_B_TILE_ROWS,
    F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
  );

export interface F32SubgroupGemmSpecialization {
  readonly label: string;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly activation: GemmActivation;
  readonly hasBias: boolean;
  readonly residualScale: number | null;
  readonly weightLayout:
    | typeof RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
    | typeof RUNTIME_ROW_MAJOR_F32_LAYOUT;
}

function createF32SubgroupGemmGeometry(
  tileRows: number,
  workgroupSize: number,
): F32SubgroupGemmGeometry {
  const subgroupCount = workgroupSize / SUBGROUP_SIZE;
  const rowsPerSubgroup = tileRows / subgroupCount;
  if (
    !Number.isSafeInteger(subgroupCount) ||
    subgroupCount <= 0 ||
    !Number.isSafeInteger(rowsPerSubgroup) ||
    rowsPerSubgroup !== 8
  ) {
    throw new Error(
      `Invalid FP32 scalar GEMM geometry M${tileRows}/WG${workgroupSize}`,
    );
  }
  return {
    tileRows,
    workgroupSize,
    subgroupCount,
    rowsPerSubgroup,
  };
}

function fp32GeometryForWeightLayout(
  weightLayout: F32SubgroupGemmSpecialization["weightLayout"],
): F32SubgroupGemmGeometry {
  if (weightLayout === RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT) {
    return FP32_DIRECT_B_GEOMETRY;
  }
  if (weightLayout === RUNTIME_ROW_MAJOR_F32_LAYOUT) {
    return FP32_STAGED_B_GEOMETRY;
  }
  throw new Error(
    `Unsupported FP32 scalar GEMM weight layout ${weightLayout}`,
  );
}

export const F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS =
  [
    {
      label: "encoder-layer-ff-expand",
      rows: 7_520,
      inner: 1_024,
      columns: 4_096,
      activation: "silu",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    },
    {
      label: "encoder-layer-ff-contract",
      rows: 7_520,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: false,
      residualScale: 0.5,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    },
    {
      label: "encoder-layer-attention-qkv",
      rows: 7_520,
      inner: 1_024,
      columns: 3_072,
      activation: "none",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    },
    {
      label: "encoder-layer-residual-1024",
      rows: 7_520,
      inner: 1_024,
      columns: 1_024,
      activation: "none",
      hasBias: false,
      residualScale: 1,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    },
    {
      label: "encoder-layer-convolution-expand",
      rows: 7_520,
      inner: 1_024,
      columns: 2_048,
      activation: "none",
      hasBias: false,
      residualScale: null,
      weightLayout: RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    },
    {
      label: "encoder-projector",
      rows: 7_520,
      inner: 1_024,
      columns: 640,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise3",
      rows: 12_032,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise3-b2",
      rows: 24_064,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise3-b3",
      rows: 36_096,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise6",
      rows: 3_008,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise6-b2",
      rows: 6_016,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-pointwise6-b3",
      rows: 9_024,
      inner: 256,
      columns: 256,
      activation: "relu",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-flatten-project",
      rows: 188,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-flatten-project-b2",
      rows: 376,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
    {
      label: "subsampling-flatten-project-b3",
      rows: 564,
      inner: 4_096,
      columns: 1_024,
      activation: "none",
      hasBias: true,
      residualScale: null,
      weightLayout: RUNTIME_ROW_MAJOR_F32_LAYOUT,
    },
  ] as const satisfies readonly F32SubgroupGemmSpecialization[];

export interface F32SubgroupGemmPlan {
  readonly workgroups: readonly [number, number, 1];
}

/**
 * Plan the N128/K32 FP32 scalar subgroup kernel. Both M32 direct-B and M48
 * staged-B kernels bounds-check their row tail, so every active B1-B40 prefix
 * uses the same fixed-shape pipeline.
 */
export function planF32SubgroupGemm(
  rows: number,
  inner: number,
  columns: number,
  weightLayout: F32SubgroupGemmSpecialization["weightLayout"],
): F32SubgroupGemmPlan {
  if (
    !Number.isSafeInteger(rows) ||
    rows <= 0 ||
    !Number.isSafeInteger(inner) ||
    inner <= 0 ||
    inner % F32_SUBGROUP_TILE_INNER !== 0 ||
    !Number.isSafeInteger(columns) ||
    columns <= 0 ||
    columns % FP32_PACKED_K !== 0
  ) {
    throw new RangeError(
      "FP32 subgroup GEMM requires K divisible by 32 and N divisible by 4",
    );
  }
  const geometry = fp32GeometryForWeightLayout(weightLayout);
  return {
    workgroups: [
      Math.ceil(columns / F32_SUBGROUP_TILE_COLUMNS),
      Math.ceil(rows / geometry.tileRows),
      1,
    ],
  };
}

/**
 * Emit one fixed-shape N128/K32 FP32 scalar-subgroup shader. Direct
 * tile-major weights use M32/WG128, while row-major common matrices use
 * M48/WG192 and one 16 KiB B tile. Each lane owns one vec4 output for each of
 * eight rows, keeping FP32 accumulator pressure bounded. Direct K1024
 * contractions traverse the operation-native weight tile as one contiguous
 * loop and broadcast one scalar component at a time; K4096 retains its
 * shorter nested K32 loops to avoid the target compiler's long-loop cliff.
 */
export function f32SubgroupGemmWgsl(
  specialization: F32SubgroupGemmSpecialization,
): string {
  planF32SubgroupGemm(
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.weightLayout,
  );
  if (
    specialization.hasBias &&
    specialization.residualScale !== null
  ) {
    throw new Error(
      `${specialization.label} cannot fuse bias and residual together`,
    );
  }
  const aVectorsPerRow =
    specialization.inner / FP32_PACKED_K;
  const weightVectorsPerRow =
    specialization.columns / FP32_PACKED_K;
  const columnVectors =
    specialization.columns / FP32_PACKED_K;
  const hasPartialColumnTile =
    specialization.columns % F32_SUBGROUP_TILE_COLUMNS !== 0;
  const directResidentB =
    specialization.weightLayout ===
    RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT;
  const geometry = fp32GeometryForWeightLayout(
    specialization.weightLayout,
  );
  if (
    directResidentB &&
    (specialization.inner % F32_SUBGROUP_TILE_INNER !== 0 ||
      specialization.columns % F32_SUBGROUP_TILE_COLUMNS !== 0)
  ) {
    throw new Error(
      `${specialization.label} cannot use direct K32/N128 tile-major weights`,
    );
  }
  const kTiles =
    specialization.inner / F32_SUBGROUP_TILE_INNER;
  const declarations = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) =>
      `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) =>
      `      let a${row} = subgroupBroadcast(packed_a, ${row}u);`,
  ).join("\n");
  const contractionBlocks = Array.from(
    { length: FP32_PACKED_K },
    (_, component) => {
      const fmas = Array.from(
        { length: geometry.rowsPerSubgroup },
        (_, row) =>
          directResidentB
            ? `      let a${row}_${component} = subgroupBroadcast(` +
              `packed_a[${component}], ${row}u);\n` +
              `      acc${row} = fma(` +
              `vec4<f32>(a${row}_${component}), b, acc${row});`
            : `      acc${row} = fma(` +
              `vec4<f32>(a${row}[${component}]), b, acc${row});`,
      ).join("\n");
      const vectorB = directResidentB
        ? `matrix_b[
        direct_b_tile_base +
          (k_vector * ${FP32_PACKED_K}u + ${component}u)
            * ${FP32_B_VECTORS_PER_ROW}u + subgroup_lane
      ]`
        : `tile_b[
        (k_vector * ${FP32_PACKED_K}u + ${component}u)
          * ${FP32_B_VECTORS_PER_ROW}u + subgroup_lane
      ]`;
      return /* wgsl */ `
    {
      let b = ${vectorB};
${fmas}
    }`;
    },
  ).join("\n");
  const stores = Array.from(
    { length: geometry.rowsPerSubgroup },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    let column_vector = column_vector_base;
    if (${hasPartialColumnTile
      ? "row < ROWS && column_vector < COLUMN_VECTORS"
      : "row < ROWS"}) {
      let output_base = column_vector * ${FP32_PACKED_K}u;
      var value = acc${row};
${fp32SpecializedBias(specialization)}
${specializedActivation(specialization.activation)}
      let output_index =
        row * COLUMN_VECTORS + column_vector;
${fp32SpecializedResidual(specialization)}
      matrix_c[output_index] = value;
    }
  }`,
  ).join("\n");
  const optionalDeclaration = specialization.hasBias
    ? /* wgsl */ `
@group(0) @binding(3) var<storage, read> bias: array<f32>;`
    : specialization.residualScale === null
      ? ""
      : /* wgsl */ `
@group(0) @binding(3) var<storage, read_write>
  residual: array<vec4<f32>>;`;
  const weightLoad = hasPartialColumnTile
    ? /* wgsl */ `    if (b_vector < WEIGHT_VECTORS_PER_ROW) {
      tile_b[item] = matrix_b[
        (k_base + b_k) * WEIGHT_VECTORS_PER_ROW + b_vector
      ];
    } else {
      tile_b[item] = vec4<f32>(0.0);
    }`
    : /* wgsl */ `    tile_b[item] = matrix_b[
      (k_base + b_k) * WEIGHT_VECTORS_PER_ROW + b_vector
    ];`;
  const workgroupDeclaration = directResidentB
    ? ""
    : `var<workgroup> tile_b:
  array<vec4<f32>, ${FP32_B_VECTORS_PER_TILE}>;
`;
  const loadFunction = directResidentB
    ? ""
    : fp32RowMajorWeightLoad(
        weightLoad,
        geometry.workgroupSize,
      );
  const localIndexParameter = directResidentB
    ? ""
    : "  @builtin(local_invocation_index) local_index: u32,\n";
  const beginInnerTile = directResidentB
    ? `    let direct_b_tile_base =
      (group.x * ${kTiles}u +
        k_base / ${F32_SUBGROUP_TILE_INNER}u) *
      ${FP32_B_VECTORS_PER_TILE}u;`
    : `    load_b_tile(k_base, local_index, group.x);
    workgroupBarrier();`;
  const endInnerTile = directResidentB
    ? ""
    : "    workgroupBarrier();";
  const contractionLoop =
    directResidentB && specialization.inner === 1_024
    ? /* wgsl */ `
  let direct_b_tile_base =
    group.x * ${kTiles * FP32_B_VECTORS_PER_TILE}u;
  for (
    var k_vector = 0u;
    k_vector < A_VECTORS_PER_ROW;
    k_vector += 1u
  ) {
    var packed_a = vec4<f32>(0.0);
    let a_row = row_base + subgroup_lane;
    if (
      subgroup_lane < ${geometry.rowsPerSubgroup}u &&
      a_row < ROWS
    ) {
      packed_a = matrix_a[
        a_row * A_VECTORS_PER_ROW + k_vector
      ];
    }
${contractionBlocks}
  }`
    : /* wgsl */ `
  for (
    var k_base = 0u;
    k_base < INNER;
    k_base += ${F32_SUBGROUP_TILE_INNER}u
  ) {
${beginInnerTile}

    for (
      var k_vector = 0u;
      k_vector < ${F32_SUBGROUP_TILE_INNER / FP32_PACKED_K}u;
      k_vector += 1u
    ) {
      var packed_a = vec4<f32>(0.0);
      let a_row = row_base + subgroup_lane;
      if (
        subgroup_lane < ${geometry.rowsPerSubgroup}u &&
        a_row < ROWS
      ) {
        packed_a = matrix_a[
          a_row * A_VECTORS_PER_ROW
            + k_base / ${FP32_PACKED_K}u
            + k_vector
        ];
      }
${directResidentB ? "" : broadcasts}
${contractionBlocks}
    }
${endInnerTile}
  }`;

  return /* wgsl */ `
enable subgroups;

const ROWS = ${specialization.rows}u;
const INNER = ${specialization.inner}u;
const A_VECTORS_PER_ROW = ${aVectorsPerRow}u;
const WEIGHT_VECTORS_PER_ROW = ${weightVectorsPerRow}u;
const COLUMN_VECTORS = ${columnVectors}u;

@group(0) @binding(0) var<storage, read_write>
  matrix_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>
  matrix_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write>
  matrix_c: array<vec4<f32>>;
${optionalDeclaration}

${workgroupDeclaration}

${loadFunction}

@compute @workgroup_size(${geometry.workgroupSize}, 1, 1)
fn main(
${localIndexParameter}  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${SUBGROUP_SIZE}u) {
    return;
  }
  let row_base =
    group.y * ${geometry.tileRows}u
      + subgroup * ${geometry.rowsPerSubgroup}u;
  let column_vector_base =
    group.x * ${FP32_COLUMN_VECTORS_PER_TILE}u
      + subgroup_lane * ${FP32_COLUMN_VECTORS_PER_LANE}u;
${declarations}

${contractionLoop}

${stores}
}
`;
}

function fp32RowMajorWeightLoad(
  weightLoad: string,
  workgroupSize: number,
): string {
  return /* wgsl */ `fn load_b_tile(
  k_base: u32,
  local_index: u32,
  group_x: u32,
) {
  for (
    var item = local_index;
    item < ${FP32_B_VECTORS_PER_TILE}u;
    item += ${workgroupSize}u
  ) {
    let b_k = item / ${FP32_B_VECTORS_PER_ROW}u;
    let b_vector =
      group_x * ${FP32_B_VECTORS_PER_ROW}u
        + item % ${FP32_B_VECTORS_PER_ROW}u;
${weightLoad}
  }
}`;
}

/**
 * FP32 counterpart: direct shapes use tuned M36/M40-N128-K128 tiles, while
 * staged row-major shapes use M32/N128/K32 to stay within 20 KiB.
 */
export function f32PortableGemmWgsl(
  specialization: F32SubgroupGemmSpecialization,
): string {
  planPortableGemm(
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.weightLayout,
  );
  if (
    specialization.hasBias &&
    specialization.residualScale !== null
  ) {
    throw new Error(
      `${specialization.label} cannot fuse bias and residual together`,
    );
  }
  const directResidentB =
    specialization.weightLayout ===
    RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT;
  const geometry = portableGeometryForGemm(
    specialization.weightLayout,
    specialization.inner,
    specialization.columns,
  );
  const columnVectorsPerTile =
    geometry.tileColumns / FP32_PACKED_K;
  const rowGroups = geometry.workgroupSize / columnVectorsPerTile;
  const rowsPerLane = geometry.tileRows / rowGroups;
  const aTileVectorsPerRow =
    geometry.tileInner / FP32_PACKED_K;
  const aTileVectors = geometry.tileRows * aTileVectorsPerRow;
  const bTileVectors = geometry.tileInner * columnVectorsPerTile;
  const aVectorsPerRow = specialization.inner / FP32_PACKED_K;
  const columnVectors = specialization.columns / FP32_PACKED_K;
  const runtimeKTiles =
    specialization.inner / F32_SUBGROUP_TILE_INNER;
  const runtimeTileColumnVectors =
    F32_SUBGROUP_TILE_COLUMNS / FP32_PACKED_K;
  const portableSlicesPerRuntimeTile =
    F32_SUBGROUP_TILE_COLUMNS / geometry.tileColumns;
  const declarations = Array.from(
    { length: rowsPerLane },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const aLoads = Array.from(
    { length: rowsPerLane },
    (_, row) =>
      `    let a${row} = tile_a[(local_row_base + ${row}u) * ` +
      `${aTileVectorsPerRow}u + k_vector];`,
  ).join("\n");
  const contractionBlocks = Array.from(
    { length: FP32_PACKED_K },
    (_, component) => {
      const bLoad = directResidentB
        ? `matrix_b[
        direct_b_tile_base +
          (k_vector * ${FP32_PACKED_K}u + ${component}u) *
            ${runtimeTileColumnVectors}u +
          direct_b_slice * ${columnVectorsPerTile}u +
          local_column_vector
      ]`
        : `tile_b[
        (k_vector * ${FP32_PACKED_K}u + ${component}u) *
          ${columnVectorsPerTile}u +
          local_column_vector
      ]`;
      const fmas = Array.from(
        { length: rowsPerLane },
        (_, row) =>
          `      acc${row} = fma(vec4<f32>(a${row}[${component}]), b, acc${row});`,
      ).join("\n");
      return /* wgsl */ `
    {
      let b = ${bLoad};
${fmas}
    }`;
    },
  ).join("\n");
  const stores = Array.from(
    { length: rowsPerLane },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + local_row_base + ${row}u;
    if (row < ROWS && column_vector < COLUMN_VECTORS) {
      let output_base = column_vector * ${FP32_PACKED_K}u;
      var value = acc${row};
${fp32SpecializedBias(specialization)}
${specializedActivation(specialization.activation)}
      let output_index = row * COLUMN_VECTORS + column_vector;
${fp32SpecializedResidual(specialization)}
      matrix_c[output_index] = value;
    }
  }`,
  ).join("\n");
  const optionalDeclaration = specialization.hasBias
    ? /* wgsl */ `
@group(0) @binding(3) var<storage, read> bias: array<f32>;`
    : specialization.residualScale === null
      ? ""
      : /* wgsl */ `
@group(0) @binding(3) var<storage, read_write>
  residual: array<vec4<f32>>;`;
  const stagedBDeclaration = directResidentB
    ? ""
    : `var<workgroup> tile_b:
  array<vec4<f32>, ${bTileVectors}>;`;
  const stagedBLoad = directResidentB
    ? ""
    : /* wgsl */ `
    for (
      var item = lane;
      item < ${bTileVectors}u;
      item += ${geometry.workgroupSize}u
    ) {
      let k_scalar = item / ${columnVectorsPerTile}u;
      let local_column = item % ${columnVectorsPerTile}u;
      let source_column =
        group.x * ${columnVectorsPerTile}u + local_column;
      if (source_column < COLUMN_VECTORS) {
        tile_b[item] = matrix_b[
          (k_base + k_scalar) * COLUMN_VECTORS + source_column
        ];
      } else {
        tile_b[item] = vec4<f32>(0.0);
      }
    }`;
  const directBSetup = directResidentB
    ? /* wgsl */ `
    let direct_b_tile =
      group.x / ${portableSlicesPerRuntimeTile}u;
    let direct_b_slice =
      group.x % ${portableSlicesPerRuntimeTile}u;
    let direct_b_tile_base =
      (direct_b_tile * ${runtimeKTiles}u +
        k_base / ${F32_SUBGROUP_TILE_INNER}u) *
        ${F32_SUBGROUP_TILE_INNER * runtimeTileColumnVectors}u +
      (k_base % ${F32_SUBGROUP_TILE_INNER}u) *
        ${runtimeTileColumnVectors}u;`
    : "";

  return /* wgsl */ `
const ROWS = ${specialization.rows}u;
const INNER = ${specialization.inner}u;
const A_VECTORS_PER_ROW = ${aVectorsPerRow}u;
const COLUMN_VECTORS = ${columnVectors}u;

@group(0) @binding(0) var<storage, read_write>
  matrix_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>
  matrix_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write>
  matrix_c: array<vec4<f32>>;
${optionalDeclaration}

var<workgroup> tile_a:
  array<vec4<f32>, ${aTileVectors}>;
${stagedBDeclaration}

@compute @workgroup_size(${geometry.workgroupSize}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let local_column_vector =
    lane % ${columnVectorsPerTile}u;
  let local_row_group =
    lane / ${columnVectorsPerTile}u;
  let local_row_base =
    local_row_group * ${rowsPerLane}u;
  let row_base = group.y * ${geometry.tileRows}u;
  let column_vector =
    group.x * ${columnVectorsPerTile}u +
      local_column_vector;
${declarations}

  for (
    var k_base = 0u;
    k_base < INNER;
    k_base += ${geometry.tileInner}u
  ) {
    for (
      var item = lane;
      item < ${aTileVectors}u;
      item += ${geometry.workgroupSize}u
    ) {
      let tile_row = item / ${aTileVectorsPerRow}u;
      let k_vector = item % ${aTileVectorsPerRow}u;
      let source_row = row_base + tile_row;
      if (source_row < ROWS) {
        tile_a[item] = matrix_a[
          source_row * A_VECTORS_PER_ROW +
            k_base / ${FP32_PACKED_K}u + k_vector
        ];
      } else {
        tile_a[item] = vec4<f32>(0.0);
      }
    }
${stagedBLoad}
${directBSetup}
    workgroupBarrier();

    for (
      var k_vector = 0u;
      k_vector < ${aTileVectorsPerRow}u;
      k_vector += 1u
    ) {
${aLoads}
${contractionBlocks}
    }
    if (k_base + ${geometry.tileInner}u < INNER) {
      workgroupBarrier();
    }
  }
${stores}
}
`;
}

function fp32SpecializedBias(
  specialization: F32SubgroupGemmSpecialization,
): string {
  if (!specialization.hasBias) return "";
  return /* wgsl */ `      value += vec4<f32>(
        bias[output_base + 0u],
        bias[output_base + 1u],
        bias[output_base + 2u],
        bias[output_base + 3u]
      );`;
}

function fp32SpecializedResidual(
  specialization: F32SubgroupGemmSpecialization,
): string {
  const scale = specialization.residualScale;
  if (scale === null) return "";
  const scaleLiteral = scale === 1 ? "1.0" : String(scale);
  return (
    "      value = residual[output_index]\n" +
    `        + ${scaleLiteral} * value;`
  );
}

export interface PreparedFp32PaletteWeight {
  readonly source: GpuTensor;
  readonly palette: GpuTensor;
  readonly paletteGroupColumns: number;
  readonly inner: number;
  readonly columns: number;
  readonly runtimeLayout:
    F32SubgroupGemmSpecialization["weightLayout"];
  readonly binding: GPUBufferBinding;
  readonly preparation: PaletteExpansionDispatch;
}

export interface Fp32GemmDescriptor
  extends Omit<GemmDescriptor, "preparedPaletteWeight"> {
  readonly preparedPaletteWeight?: PreparedFp32PaletteWeight;
}

interface Fp32SpecializedGemmPipeline {
  readonly specialization: F32SubgroupGemmSpecialization;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
}

export class F32GemmDispatch {
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: ArenaSlice,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly workgroups:
      readonly [number, number, number],
    private readonly rows: number,
    private readonly inner: number,
    private readonly columns: number,
    private readonly weightLayout:
      F32SubgroupGemmSpecialization["weightLayout"],
    private readonly kernelBackend:
      ParakeetExecutionProfile["kernelBackend"],
    private readonly preparation?: PaletteExpansionDispatch,
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    shape?: EncoderDispatchShape,
  ): void {
    if (this.destroyed) {
      throw new Error(`${this.label} was destroyed`);
    }
    this.preparation?.encode(encoder);
    let workgroups = this.workgroups;
    if (shape !== undefined) {
      const activeRows = shape.rows;
      if (
        !Number.isSafeInteger(activeRows) ||
        activeRows <= 0 ||
        activeRows > this.rows
      ) {
        throw new RangeError(
          `${this.label} active rows must be in [1, ${this.rows}]`,
        );
      }
      workgroups = this.kernelBackend === "subgroups"
        ? planF32SubgroupGemm(
            activeRows,
            this.inner,
            this.columns,
            this.weightLayout,
          ).workgroups
        : planPortableGemm(
            activeRows,
            this.inner,
            this.columns,
            this.weightLayout,
          ).workgroups;
    }
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setBindGroup(0, this.bindGroup);
    if (hasWorkgroups(workgroups)) {
      pass.setPipeline(this.pipeline);
      pass.dispatchWorkgroups(...workgroups);
    }
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.preparation?.destroy();
  }
}

/**
 * Fixed-shape FP32 GEMM owner. Both backends reuse the same palette expansion
 * and graph-local 17,301,504-byte scratch; only their compiled GEMM pipelines
 * and dispatch geometry differ.
 */
export class F32GemmKernel {
  private destroyed = false;
  private transientPaletteDispatchCountValue = 0;
  private transientPaletteExpandedBytesValue = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly specializedPipelines: ReadonlyMap<
      string,
      Fp32SpecializedGemmPipeline
    >,
    private readonly paletteExpander: PaletteWeightExpander,
    private readonly paletteParameterPool:
      PaletteExpansionParameterPool,
    private readonly dequantizedWeightScratch: GPUBuffer,
    private readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  get transientPaletteParameterBytes(): number {
    return this.paletteParameterPool.byteLength;
  }

  get transientPaletteParameterUsedSlots(): number {
    return this.paletteParameterPool.usedSlots;
  }

  get transientPaletteDispatchCount(): number {
    return this.transientPaletteDispatchCountValue;
  }

  get transientPaletteExpandedBytes(): number {
    return this.transientPaletteExpandedBytesValue;
  }

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP32_EXECUTION_PROFILE,
  ): Promise<F32GemmKernel> {
    if (executionProfile.precision !== "fp32") {
      throw new Error("Parakeet FP32 GEMM requires an FP32 execution profile");
    }
    if (
      executionProfile.kernelBackend === "subgroups" &&
      !device.features.has("subgroups")
    ) {
      throw new Error("Parakeet FP32 GEMM requires WebGPU subgroups");
    }
    const requiredWorkgroupSize =
      executionProfile.kernelBackend === "subgroups"
        ? Math.max(
            FP32_DIRECT_B_GEOMETRY.workgroupSize,
            FP32_STAGED_B_GEOMETRY.workgroupSize,
          )
        : Math.max(
            PORTABLE_FP32_DIRECT_GEMM_GEOMETRY.workgroupSize,
            PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY.workgroupSize,
            PORTABLE_FP32_STAGED_GEMM_GEOMETRY.workgroupSize,
          );
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        requiredWorkgroupSize ||
      device.limits.maxComputeWorkgroupSizeX <
        requiredWorkgroupSize
    ) {
      throw new Error(
        `Parakeet FP32 GEMM requires ${requiredWorkgroupSize} workgroup lanes`,
      );
    }
    const requiredWorkgroupStorage =
      executionProfile.kernelBackend === "subgroups"
        ? F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES
        : F32_PORTABLE_GEMM_WORKGROUP_BYTES;
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      requiredWorkgroupStorage
    ) {
      throw new Error(
        `Parakeet FP32 GEMM requires ${requiredWorkgroupStorage} bytes of workgroup storage`,
      );
    }
    const [specializedPipelines, paletteExpander] =
      await Promise.all([
        createFp32SpecializedGemmPipelines(
          device,
          executionProfile.kernelBackend,
        ),
        PaletteWeightExpander.create(
          device,
          executionProfile,
        ),
      ]);
    const dequantizedWeightScratch = device.createBuffer({
      label:
        "parakeet-fp32-transient-palette-weight-scratch",
      size: FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
      usage: GPUBufferUsage.STORAGE,
    });
    try {
      const paletteParameterPool =
        new PaletteExpansionParameterPool(
          device,
          TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT,
          "parakeet-fp32-transient-palette-parameter-pool",
          executionProfile,
        );
      return new F32GemmKernel(
        device,
        arena,
        specializedPipelines,
        paletteExpander,
        paletteParameterPool,
        dequantizedWeightScratch,
        executionProfile,
      );
    } catch (error) {
      dequantizedWeightScratch.destroy();
      throw error;
    }
  }

  createDispatch(
    descriptor: Fp32GemmDescriptor,
  ): F32GemmDispatch {
    if (this.destroyed) {
      throw new Error("Parakeet FP32 GEMM kernel was destroyed");
    }
    requireFp32GemmShape(descriptor);
    const specialized =
      requireFp32ProductionSpecialization(
        this.specializedPipelines,
        descriptor,
      );
    requireFp32RuntimeWeight(
      descriptor,
      specialized.specialization,
    );
    const plan = this.executionProfile.kernelBackend === "subgroups"
      ? planF32SubgroupGemm(
          descriptor.rows,
          descriptor.inner,
          descriptor.columns,
          specialized.specialization.weightLayout,
        )
      : planPortableGemm(
          descriptor.rows,
          descriptor.inner,
          descriptor.columns,
          specialized.specialization.weightLayout,
        );
    const optionalBinding =
      descriptor.bias?.binding ??
      (descriptor.residual === undefined
        ? undefined
        : this.arena.binding(descriptor.residual));
    const paletteBits = paletteBitsForLayout(
      descriptor.b.runtimeRecord.layout,
    );
    let preparation: PaletteExpansionDispatch | undefined;
    try {
      const prepared = descriptor.preparedPaletteWeight;
      const weightBinding =
        prepared === undefined
          ? this.createInlinePaletteWeight(
              descriptor,
              specialized.specialization,
              paletteBits,
            )
          : this.requirePreparedPaletteWeight(
              descriptor,
              specialized.specialization,
              prepared,
            );
      preparation = weightBinding.preparation;
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-fp32-bindings`,
        layout: specialized.layout,
        entries: [
          {
            binding: 0,
            resource: this.arena.binding(descriptor.a),
          },
          { binding: 1, resource: weightBinding.binding },
          {
            binding: 2,
            resource: this.arena.binding(descriptor.output),
          },
          ...(optionalBinding === undefined
            ? []
            : [{ binding: 3, resource: optionalBinding }]),
        ],
      });
      return new F32GemmDispatch(
        descriptor.label,
        descriptor.output,
        specialized.pipeline,
        bindGroup,
        plan.workgroups,
        descriptor.rows,
        descriptor.inner,
        descriptor.columns,
        specialized.specialization.weightLayout,
        this.executionProfile.kernelBackend,
        preparation,
      );
    } catch (error) {
      preparation?.destroy();
      throw error;
    }
  }

  createPreparedPaletteWeight(
    descriptor: Fp32GemmDescriptor,
    scratchOffset: number,
  ): PreparedFp32PaletteWeight {
    if (this.destroyed) {
      throw new Error("Parakeet FP32 GEMM kernel was destroyed");
    }
    if (descriptor.preparedPaletteWeight !== undefined) {
      throw new Error(
        `${descriptor.label} cannot prepare an already prepared weight`,
      );
    }
    requireFp32GemmShape(descriptor);
    const specialized =
      requireFp32ProductionSpecialization(
        this.specializedPipelines,
        descriptor,
      );
    requireFp32RuntimeWeight(
      descriptor,
      specialized.specialization,
    );
    const paletteBits = paletteBitsForLayout(
      descriptor.b.runtimeRecord.layout,
    );
    if (paletteBits === undefined) {
      throw new Error(
        `${descriptor.label} is not a palette-compressed weight`,
      );
    }
    const created = this.createPaletteWeight(
      descriptor,
      specialized.specialization,
      paletteBits,
      scratchOffset,
    );
    return {
      source: descriptor.b,
      palette: descriptor.bPalette!,
      paletteGroupColumns: descriptor.paletteGroupColumns!,
      inner: descriptor.inner,
      columns: descriptor.columns,
      runtimeLayout: specialized.specialization.weightLayout,
      binding: created.binding,
      preparation: created.preparation,
    };
  }

  private createInlinePaletteWeight(
    descriptor: Fp32GemmDescriptor,
    specialization: F32SubgroupGemmSpecialization,
    paletteBits: 5 | 6 | undefined,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation?: PaletteExpansionDispatch;
  } {
    if (paletteBits === undefined) {
      return { binding: descriptor.b.binding };
    }
    return this.createPaletteWeight(
      descriptor,
      specialization,
      paletteBits,
      0,
    );
  }

  private createPaletteWeight(
    descriptor: Fp32GemmDescriptor,
    specialization: F32SubgroupGemmSpecialization,
    paletteBits: 5 | 6,
    scratchOffset: number,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation: PaletteExpansionDispatch;
  } {
    const size =
      descriptor.inner *
      descriptor.columns *
      Float32Array.BYTES_PER_ELEMENT;
    if (
      !Number.isSafeInteger(scratchOffset) ||
      scratchOffset < 0 ||
      scratchOffset %
        this.device.limits.minStorageBufferOffsetAlignment !==
        0 ||
      scratchOffset + size >
        FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
    ) {
      throw new RangeError(
        `${descriptor.label} has an invalid FP32 transient palette scratch slice`,
      );
    }
    const binding = {
      buffer: this.dequantizedWeightScratch,
      offset: scratchOffset,
      size,
    } satisfies GPUBufferBinding;
    const bPalette = descriptor.bPalette!;
    const expansionDescriptor = {
      label: `${descriptor.label}-fp32-palette-expand`,
      source: descriptor.b.binding,
      palette: bPalette.binding,
      target: binding,
      scalarCount: descriptor.inner * descriptor.columns,
      columns: descriptor.columns,
      paletteBits,
      paletteGroupColumns: descriptor.paletteGroupColumns!,
      paletteGroups: bPalette.runtimeRecord.storageShape[0]!,
      runtimeLayout: specialization.weightLayout,
    } as const;
    const preparation =
      this.paletteExpander.createPooledDispatch(
        expansionDescriptor,
        this.paletteParameterPool.bindingFor(
          expansionDescriptor,
        ),
      );
    this.transientPaletteDispatchCountValue += 1;
    this.transientPaletteExpandedBytesValue += size;
    return { binding, preparation };
  }

  private requirePreparedPaletteWeight(
    descriptor: Fp32GemmDescriptor,
    specialization: F32SubgroupGemmSpecialization,
    prepared: PreparedFp32PaletteWeight,
  ): {
    readonly binding: GPUBufferBinding;
    readonly preparation?: PaletteExpansionDispatch;
  } {
    if (
      prepared.source !== descriptor.b ||
      prepared.palette !== descriptor.bPalette ||
      prepared.paletteGroupColumns !==
        descriptor.paletteGroupColumns ||
      prepared.inner !== descriptor.inner ||
      prepared.columns !== descriptor.columns ||
      prepared.runtimeLayout !== specialization.weightLayout
    ) {
      throw new Error(
        `${descriptor.label} has a mismatched prepared FP32 palette weight`,
      );
    }
    return { binding: prepared.binding };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.paletteParameterPool.destroy();
    this.dequantizedWeightScratch.destroy();
  }
}

function requireFp32GemmShape(
  descriptor: Fp32GemmDescriptor,
): void {
  if (
    !Number.isSafeInteger(descriptor.rows) ||
    descriptor.rows <= 0 ||
    !Number.isSafeInteger(descriptor.inner) ||
    descriptor.inner <= 0 ||
    descriptor.inner % F32_SUBGROUP_TILE_INNER !== 0 ||
    !Number.isSafeInteger(descriptor.columns) ||
    descriptor.columns <= 0 ||
    descriptor.columns % FP32_PACKED_K !== 0
  ) {
    throw new RangeError(
      `Unsupported FP32 GEMM shape for ${descriptor.label}`,
    );
  }
  const expectedWeightElements =
    descriptor.inner * descriptor.columns;
  const source = descriptor.b.sourceRecord;
  const paletteBits = paletteBitsForLayout(source.layout);
  if (
    source.logicalShape.join(",") !==
    `${descriptor.inner},${descriptor.columns}`
  ) {
    throw new Error(
      `${descriptor.label} FP32 weight package does not match K=${descriptor.inner}, N=${descriptor.columns}`,
    );
  }
  if (paletteBits === undefined || source.dtype !== "uint32") {
    throw new Error(
      `${descriptor.label} has an unsupported FP32 weight format`,
    );
  }
  const codesPerGroup = paletteBits === 5 ? 32 : 16;
  const wordsPerGroup = paletteBits === 5 ? 5 : 3;
  if (expectedWeightElements % codesPerGroup !== 0) {
    throw new Error(
      `${descriptor.label} FP32 palette codes are not group aligned`,
    );
  }
  const expectedGroups =
    expectedWeightElements / codesPerGroup;
  if (
    source.byteLength !==
      expectedGroups *
        wordsPerGroup *
        Uint32Array.BYTES_PER_ELEMENT ||
    source.storageShape.join(",") !==
      `${expectedGroups},${wordsPerGroup}`
  ) {
    throw new Error(
      `${descriptor.label} has invalid packed FP32 palette indices`,
    );
  }
}

function fp32SpecializationKey(
  specialization: Omit<
    F32SubgroupGemmSpecialization,
    "label" | "weightLayout"
  >,
): string {
  return [
    specialization.rows,
    specialization.inner,
    specialization.columns,
    specialization.activation,
    specialization.hasBias ? "bias" : "no-bias",
    specialization.residualScale === null
      ? "no-residual"
      : `residual-${specialization.residualScale}`,
  ].join(":");
}

function requireFp32ProductionSpecialization(
  pipelines: ReadonlyMap<
    string,
    Fp32SpecializedGemmPipeline
  >,
  descriptor: Fp32GemmDescriptor,
): Fp32SpecializedGemmPipeline {
  if (
    descriptor.residual === undefined &&
    descriptor.residualScale !== undefined
  ) {
    throw new Error(
      `${descriptor.label} supplies a residual scale without a residual`,
    );
  }
  const signature = {
    rows: descriptor.rows,
    inner: descriptor.inner,
    columns: descriptor.columns,
    activation: descriptor.activation ?? "none",
    hasBias: descriptor.bias !== undefined,
    residualScale:
      descriptor.residual === undefined
        ? null
        : (descriptor.residualScale ?? 1),
  } satisfies Omit<
    F32SubgroupGemmSpecialization,
    "label" | "weightLayout"
  >;
  const specialized = pipelines.get(
    fp32SpecializationKey(signature),
  );
  if (specialized === undefined) {
    throw new Error(
      `${descriptor.label} is not a fixed production FP32 GEMM signature: ` +
        `M=${signature.rows}, K=${signature.inner}, N=${signature.columns}, ` +
        `activation=${signature.activation}, bias=${signature.hasBias}, ` +
        `residualScale=${signature.residualScale ?? "none"}`,
    );
  }
  return specialized;
}

function requireFp32RuntimeWeight(
  descriptor: Fp32GemmDescriptor,
  specialization: F32SubgroupGemmSpecialization,
): void {
  const runtime = descriptor.b.runtimeRecord;
  const paletteBits = paletteBitsForLayout(runtime.layout);
  if (paletteBits !== undefined) {
    const source = descriptor.b.sourceRecord;
    const sourcePaletteBits = paletteBitsForLayout(source.layout);
    const palette = descriptor.bPalette;
    const paletteGroupColumns =
      descriptor.paletteGroupColumns;
    const paletteEntries = 1 << paletteBits;
    const paletteGroups =
      palette?.runtimeRecord.storageShape[0];
    if (
      sourcePaletteBits !== paletteBits ||
      runtime.dtype !== "uint32" ||
      runtime.layout !== source.layout ||
      runtime.logicalShape.join(",") !==
        `${descriptor.inner},${descriptor.columns}` ||
      runtime.storageShape.join(",") !==
        source.storageShape.join(",") ||
      runtime.byteLength !== source.byteLength ||
      palette === undefined ||
      palette.sourceRecord.dtype !== "float32" ||
      palette.sourceRecord.layout !==
        "output-group-by-palette-lut" ||
      palette.runtimeRecord.dtype !== "float32" ||
      palette.runtimeRecord.layout !==
        "output-group-by-palette-lut" ||
      palette.runtimeRecord.logicalShape.join(",") !==
        palette.sourceRecord.logicalShape.join(",") ||
      palette.runtimeRecord.storageShape.join(",") !==
        `${paletteGroups},${paletteEntries}` ||
      palette.runtimeRecord.byteLength !==
        paletteGroups! *
          paletteEntries *
          Float32Array.BYTES_PER_ELEMENT ||
      !Number.isSafeInteger(paletteGroupColumns) ||
      paletteGroupColumns! <= 0 ||
      paletteGroupColumns! * paletteGroups! !==
        descriptor.columns
    ) {
      throw new Error(
        `${descriptor.label} has invalid transient FP32 palette weights`,
      );
    }
    const expandedBytes =
      descriptor.inner *
      descriptor.columns *
      Float32Array.BYTES_PER_ELEMENT;
    if (
      expandedBytes >
      FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES
    ) {
      throw new Error(
        `${descriptor.label} exceeds transient FP32 palette scratch`,
      );
    }
    if (
      specialization.weightLayout ===
      RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
    ) {
      runtimeGemmTileMajorF32StorageShape(
        descriptor.inner,
        descriptor.columns,
      );
    }
    return;
  }
  const expectedStorageShape =
    specialization.weightLayout ===
    RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
      ? runtimeGemmTileMajorF32StorageShape(
          descriptor.inner,
          descriptor.columns,
        )
      : [descriptor.inner, descriptor.columns];
  if (
    runtime.dtype !== "float32" ||
    runtime.layout !== specialization.weightLayout ||
    runtime.logicalShape.join(",") !==
      `${descriptor.inner},${descriptor.columns}` ||
    runtime.storageShape.join(",") !==
      expectedStorageShape.join(",") ||
    runtime.byteLength !==
      descriptor.inner *
        descriptor.columns *
        Float32Array.BYTES_PER_ELEMENT ||
    descriptor.bPalette !== undefined ||
    descriptor.paletteGroupColumns !== undefined
  ) {
    throw new Error(
      `${descriptor.label} has invalid ${specialization.weightLayout} weights`,
    );
  }
}

async function createFp32SpecializedGemmPipelines(
  device: GPUDevice,
  kernelBackend: ParakeetExecutionProfile["kernelBackend"],
): Promise<ReadonlyMap<string, Fp32SpecializedGemmPipeline>> {
  const entries = await Promise.all(
    F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.map(
      async (specialization) => {
        const portableSuffix =
          kernelBackend === "portable" ? "-portable" : "";
        const layout = device.createBindGroupLayout({
          label:
            `parakeet-fp32-gemm-${specialization.label}${portableSuffix}-bindings`,
          entries: [
            storageEntry(0, false),
            storageEntry(1, true),
            storageEntry(2, false),
            ...(specialization.hasBias
              ? [storageEntry(3, true)]
              : specialization.residualScale === null
                ? []
                : [storageEntry(3, false)]),
          ],
        });
        const pipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [layout],
        });
        const pipeline = kernelBackend === "subgroups"
          ? await createF32SubgroupPipeline(
              device,
              pipelineLayout,
              specialization,
            )
          : await createF32PortablePipeline(
              device,
              pipelineLayout,
              specialization,
            );
        const specialized: Fp32SpecializedGemmPipeline = {
          specialization,
          pipeline,
          layout,
        };
        return [
          fp32SpecializationKey(specialization),
          specialized,
        ] as const;
      },
    ),
  );
  const pipelines = new Map(entries);
  if (
    pipelines.size !==
    F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.length
  ) {
    throw new Error(
      "Duplicate fixed production FP32 GEMM specialization",
    );
  }
  return pipelines;
}

async function createF32SubgroupPipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  specialization: F32SubgroupGemmSpecialization,
): Promise<GPUComputePipeline> {
  const label =
    `parakeet-fp32-gemm-${specialization.label}`;
  const module = device.createShaderModule({
    label,
    code: f32SubgroupGemmWgsl(specialization),
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

async function createF32PortablePipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  specialization: F32SubgroupGemmSpecialization,
): Promise<GPUComputePipeline> {
  const label =
    `parakeet-fp32-gemm-${specialization.label}-portable`;
  const module = device.createShaderModule({
    label,
    code: f32PortableGemmWgsl(specialization),
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
