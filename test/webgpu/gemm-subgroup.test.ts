import { readFileSync } from "node:fs";
import {
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
} from "../../src/model/runtime-plan";
import {
  F32GemmKernel,
  F16GemmKernel,
  F32_SUBGROUP_DIRECT_B_TILE_ROWS,
  F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
  F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS,
  F32_SUBGROUP_STAGED_B_TILE_ROWS,
  F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
  F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
  F32_SUBGROUP_TILE_COLUMNS,
  F32_SUBGROUP_TILE_INNER,
  F16_SUBGROUP_DIRECT_B_TILE_ROWS,
  F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
  F16_SUBGROUP_STAGED_B_TILE_ROWS,
  F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
  F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
  F16_SUBGROUP_TILE_COLUMNS,
  F16_SUBGROUP_TILE_INNER,
  F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS,
  FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
  TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT,
  TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
  f32SubgroupGemmWgsl,
  f16SubgroupGemmWgsl,
  planF32SubgroupGemm,
  planF16SubgroupGemm,
  transientPaletteParameterPoolBytes,
} from "../../src/webgpu/kernels/gemm";
import type { GpuActivationArena } from "../../src/webgpu/arena";

const dispatchSource = readFileSync(
  new URL("../../src/webgpu/kernels/gemm.ts", import.meta.url),
  "utf8",
);
const encoderSource = readFileSync(
  new URL("../../src/webgpu/encoder.ts", import.meta.url),
  "utf8",
);

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { COPY_DST: 1, STORAGE: 2, UNIFORM: 4 },
  });
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true,
    value: { COMPUTE: 1 },
  });
});

describe("production f16 scalar-subgroup GEMM", () => {
  it("covers the fixed encoder matrix shapes", () => {
    expect(
      planF16SubgroupGemm(
        7_520,
        1_024,
        4_096,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [16, 235, 1],
    });
    expect(
      planF16SubgroupGemm(
        7_520,
        4_096,
        1_024,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [4, 235, 1],
    });
    expect(
      planF16SubgroupGemm(
        7_520,
        1_024,
        3_072,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [12, 235, 1],
    });
    expect(
      planF16SubgroupGemm(
        7_520,
        1_024,
        2_048,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [8, 235, 1],
    });
    expect(
      planF16SubgroupGemm(
        7_520,
        1_024,
        640,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [3, 157, 1],
    });
    expect(
      planF16SubgroupGemm(
        24_064,
        256,
        256,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [1, 502, 1],
    });
  });

  it("covers arbitrary row tails in the main dispatch", () => {
    expect(
      planF16SubgroupGemm(
        39 * 188,
        1_024,
        4_096,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [16, 230, 1],
    });
    expect(
      planF16SubgroupGemm(
        188,
        1_024,
        4_096,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [16, 6, 1],
    });
    expect(
      planF16SubgroupGemm(
        376,
        4_096,
        1_024,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [4, 12, 1],
    });
    expect(
      planF16SubgroupGemm(
        1,
        32,
        8,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({
      workgroups: [1, 1, 1],
    });
  });

  it("rejects shapes outside the scalar kernel contract", () => {
    expect(() =>
      planF16SubgroupGemm(
        48,
        1_023,
        1_024,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toThrow(/K divisible by 32/);
    expect(() =>
      planF16SubgroupGemm(
        48,
        1_024,
        1_022,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toThrow(/N divisible by 8/);
    expect(() =>
      f16SubgroupGemmWgsl({
        ...F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS[0],
        label: "unsupported-direct-k",
        inner: 2_048,
      }),
    ).toThrow(/direct-B FP16 GEMM requires K1024 or K4096/);
  });

  it("locks every fixed production signature", () => {
    expect(F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS).toEqual([
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
    ]);
  });

  it("uses only standardized f16 and ordinary subgroup WGSL", () => {
    for (const specialization of F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f16SubgroupGemmWgsl(specialization);
      const directResidentB =
        specialization.weightLayout ===
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT;
      const wideDirectA =
        directResidentB && specialization.inner === 4_096;
      expect(source).toContain("enable f16;");
      expect(source).toContain("enable subgroups;");
      expect(source).toContain(
        "let b0 = bitcast<vec4<f16>>(packed_b.xy);",
      );
      expect(source).toContain(
        "let b1 = bitcast<vec4<f16>>(packed_b.zw);",
      );
      expect(source).toContain("bitcast<vec4<f16>>");
      expect(source).toContain("subgroupBroadcast(");
      expect(source).not.toMatch(/subgroup.?matrix/i);
      expect(source).not.toContain("chromium_experimental");
      if (directResidentB) {
        const aElementCount =
          (specialization.rows * specialization.inner) /
          (wideDirectA ? 8 : 4);
        const bElementCount =
          (specialization.inner * specialization.columns) / 8;
        const cElementCount =
          (specialization.rows * specialization.columns) / 4;
        if (wideDirectA) {
          expect(source).toContain(
            `matrix_a: array<vec4<u32>, ${aElementCount}>;`,
          );
          expect(source).toContain("var packed_a = vec4<u32>(0u);");
          expect(source).not.toContain("matrix_a: array<vec2<u32>");
          expect(source).not.toContain("var packed_a = vec2<u32>(0u);");
        } else {
          expect(source).toContain(
            `matrix_a: array<vec2<u32>, ${aElementCount}>;`,
          );
          expect(source).toContain("var packed_a = vec2<u32>(0u);");
          expect(source).not.toContain("matrix_a: array<vec4<u32>");
          expect(source).not.toContain("var packed_a = vec4<u32>(0u);");
        }
        expect(source).toContain(
          `matrix_b: array<vec4<u32>, ${bElementCount}>;`,
        );
        expect(source).toContain(
          `matrix_c: array<vec4<f16>, ${cElementCount}>;`,
        );
        expect(source).toContain(
          "@group(0) @binding(0) var<storage, read_write>",
        );
        expect(source).toContain(
          `group.x >= ${specialization.columns / F16_SUBGROUP_TILE_COLUMNS}u`,
        );
        expect(source).toContain("group.y >= 235u");
        expect(source).toContain("group.z != 0u");
        expect(source).toContain("subgroup >= 4u");
        expect(source).toContain("if (subgroup_lane < 8u)");
        expect(source).not.toContain("a_row < ROWS");
        expect(source).not.toContain("row < ROWS");
        expect(source).toContain(
          `@compute @workgroup_size(${F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE}, 1, 1)`,
        );
        expect(source).toContain(
          `group.y * ${F16_SUBGROUP_DIRECT_B_TILE_ROWS}u`,
        );
        expect(source).toContain("subgroup * 8u");
        expect(source).not.toContain("var<workgroup>");
        expect(source).not.toContain("workgroupBarrier");
        expect(source).not.toContain("load_b_tile");
        expect(source).not.toContain(
          "@builtin(local_invocation_index)",
        );
        expect(
          source.match(
            /let packed_b = matrix_b\[\s+direct_b_tile_base \+/g,
          ),
        ).toHaveLength(wideDirectA ? 8 : 4);
      } else {
        expect(source).toContain("matrix_b: array<vec4<u32>>;");
        expect(source).toContain("matrix_c: array<vec4<f16>>;");
        expect(source).toContain("matrix_a: array<vec2<u32>>;");
        expect(source).toContain(
          "@group(0) @binding(0) var<storage, read_write>",
        );
        expect(source).toContain("a_row < ROWS");
        expect(source).toContain("row < ROWS");
        expect(source).not.toContain("group.z != 0u");
        expect(source).toContain("var packed_a = vec2<u32>(0u);");
        expect(source).not.toContain("matrix_a: array<vec4<u32>>;");
        expect(source).not.toContain("var packed_a = vec4<u32>(0u);");
        expect(source).toContain(
          `@compute @workgroup_size(${F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE}, 1, 1)`,
        );
        expect(source).toContain(
          `group.y * ${F16_SUBGROUP_STAGED_B_TILE_ROWS}u`,
        );
        expect(source).toContain("subgroup * 8u");
        expect(source).toContain(
          `var<workgroup> tile_b:\n  array<vec4<u32>, 1024>;`,
        );
        expect(source).toContain(
          `item += ${F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE}u`,
        );
        expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
        expect(source).toContain("fn load_b_tile(");
        expect(source).toContain(
          "@builtin(local_invocation_index)",
        );
        expect(
          source.match(/let packed_b = tile_b\[/g),
        ).toHaveLength(4);
      }
    }
  });

  it("specializes shapes and epilogues without changing f16 FMA order", () => {
    for (const specialization of F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f16SubgroupGemmWgsl(specialization);
      expect(source.match(/var acc\d+_\d+ = vec4<f16>/g)).toHaveLength(16);
      expect(source.match(/ = fma\(/g)).toHaveLength(
        specialization.weightLayout ===
            RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT &&
          specialization.inner === 4_096
          ? 128
          : 64,
      );
      expect(source).toContain(`const ROWS = ${specialization.rows}u;`);
      expect(source).toContain(`const INNER = ${specialization.inner}u;`);
      if (
        specialization.weightLayout ===
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
      ) {
        expect(source).not.toContain("row < ROWS");
      } else {
        expect(source).toContain("row < ROWS");
      }
      expect(source).not.toContain("struct Params");
      expect(source).not.toContain("var<uniform>");
      expect(source).not.toContain("params.");
      expect(source).not.toContain("has_bias");
      expect(source).not.toContain("has_residual");
    }

    const ffExpand = sourceFor("encoder-layer-ff-expand");
    expect(ffExpand).toContain(
      "let direct_b_tile_base =",
    );
    expect(ffExpand).not.toContain("let b_pack =");
    expect(ffExpand).toContain("exp(-bounded)");
    expect(ffExpand).not.toContain("residual:");
    expect(ffExpand).not.toContain("bias:");

    const ffContract = sourceFor("encoder-layer-ff-contract");
    expect(ffContract).toContain(
      "residual: array<vec4<f16>, 1925120>;",
    );
    expect(ffContract).toContain(
      "@group(0) @binding(3) var<storage, read_write>",
    );
    expect(ffContract).toContain("+ 0.5 * value;");

    const residualOne = sourceFor("encoder-layer-residual-1024");
    expect(residualOne).toContain("+ 1.0 * value;");

    const pointwise = sourceFor("subsampling-pointwise3");
    expect(pointwise).toContain("bias: array<f32>;");
    expect(pointwise).toContain(
      "value = max(value, vec4<f32>(0.0));",
    );

    const projector = sourceFor("encoder-projector");
    expect(projector).toContain(
      "row < ROWS && column_vector < COLUMN_VECTORS",
    );
    expect(projector).toContain("if (b_pack < WEIGHT_PACKS_PER_ROW)");
    expect(projector).not.toContain("let tile_base =");
    expect(ffExpand).not.toContain("column_vector < COLUMN_VECTORS");
    for (const label of [
      "encoder-layer-ff-expand",
      "encoder-layer-ff-contract",
      "encoder-layer-attention-qkv",
      "encoder-layer-residual-1024",
      "encoder-layer-convolution-expand",
    ] as const) {
      const source = sourceFor(label);
      expect(source).toContain("let direct_b_tile_base =");
      expect(source).toContain(
        "let packed_b = matrix_b[\n        direct_b_tile_base +",
      );
      expect(source).not.toContain("var<workgroup> tile_b");
      expect(source).not.toContain("let packed_b = tile_b[");
      expect(source).not.toContain("workgroupBarrier");
    }
    for (const label of [
      "encoder-projector",
      "subsampling-pointwise3",
      "subsampling-pointwise6",
      "subsampling-flatten-project",
    ] as const) {
      const source = sourceFor(label);
      expect(source).toContain("let b_pack =");
      expect(source).toContain("var<workgroup> tile_b:");
      expect(loadFunction(source)).toContain(
        "tile_b[item] = matrix_b[",
      );
    }
  });

  it("pins direct A lifetime, width, and traversal specialization", () => {
    const bComponents = (source: string): string[] =>
      Array.from(
        source.matchAll(/\(k_vector \* 4u \+ (\d)u\)/g),
        (match) => match[1]!,
      );
    const aComponents = (source: string): string[] =>
      Array.from(
        source.matchAll(
          /acc0_0 = fma\(vec4<f16>\(a0_(\d)\), b0, acc0_0\);/g,
        ),
        (match) => match[1]!,
      );

    const k1024 = sourceFor("encoder-layer-ff-expand");
    expect(k1024).toContain(
      "matrix_a: array<vec2<u32>, 1925120>;",
    );
    expect(k1024).toContain("const A_VECTORS_PER_ROW = 256u;");
    expect(k1024).toContain(
      "let direct_b_tile_base =\n    group.x * 32768u;",
    );
    expect(k1024).toContain("k_vector < A_VECTORS_PER_ROW;");
    expect(k1024).toContain(
      "let local_a = bitcast<vec4<f16>>(packed_a);",
    );
    expect(k1024.match(/subgroupBroadcast\(local_a\[/g)).toHaveLength(32);
    expect(bComponents(k1024)).toEqual(["0", "1", "2", "3"]);
    expect(aComponents(k1024)).toEqual(["0", "1", "2", "3"]);
    expect(k1024).not.toContain("var k_base = 0u;");

    const k4096 = sourceFor("encoder-layer-ff-contract");
    expect(k4096).toContain(
      "matrix_a: array<vec4<u32>, 3850240>;",
    );
    expect(k4096).toContain("const A_VECTORS_PER_ROW = 512u;");
    expect(k4096).toContain(
      "(group.x * 128u +\n        k_base / 32u) *\n      1024u;",
    );
    expect(k4096).toContain("a_vector < 4u;");
    expect(k4096).toContain("k_base / 8u");
    expect(k4096).toContain(
      "let k_vector = a_vector * 2u;\n      let local_a = bitcast<vec4<f16>>(packed_a.xy);",
    );
    expect(k4096).toContain(
      "let k_vector = a_vector * 2u + 1u;\n      let local_a = bitcast<vec4<f16>>(packed_a.zw);",
    );
    expect(k4096.match(/subgroupBroadcast\(local_a\[/g)).toHaveLength(64);
    expect(bComponents(k4096)).toEqual([
      "0", "1", "2", "3", "0", "1", "2", "3",
    ]);
    expect(aComponents(k4096)).toEqual([
      "0", "1", "2", "3", "0", "1", "2", "3",
    ]);
    expect(k4096).not.toContain("a_vector < A_VECTORS_PER_ROW;");

    const staged = sourceFor("encoder-projector");
    expect(staged).toContain("matrix_a: array<vec2<u32>>;");
    expect(staged).toContain("var k_base = 0u;");
    expect(staged).toContain("k_vector < 8u;");
    expect(staged).toContain("k_base / 4u");
    expect(staged.match(/subgroupBroadcast\(packed_a,/g)).toHaveLength(8);
    expect(staged).not.toContain("let local_a =");
  });

  it("pins direct-B and staged-B geometries independently", () => {
    expect(F16_SUBGROUP_DIRECT_B_TILE_ROWS).toBe(32);
    expect(F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE).toBe(128);
    expect(F16_SUBGROUP_STAGED_B_TILE_ROWS).toBe(48);
    expect(F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE).toBe(192);
    expect(F16_SUBGROUP_TILE_COLUMNS).toBe(256);
    expect(F16_SUBGROUP_TILE_INNER).toBe(32);
    expect(F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES).toBe(16_384);
  });

  it("pins direct binding footprints without changing staged layouts", async () => {
    const layouts: GPUBindGroupLayoutDescriptor[] = [];
    const buffers: Array<{
      readonly destroy: ReturnType<typeof vi.fn>;
    }> = [];
    const device = {
      features: new Set(["shader-f16", "subgroups"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupStorageSize: 32_768,
        minStorageBufferOffsetAlignment: 256,
        minUniformBufferOffsetAlignment: 256,
      },
      createBindGroupLayout: vi.fn(
        (descriptor: GPUBindGroupLayoutDescriptor) => {
          layouts.push(descriptor);
          return {};
        },
      ),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule: vi.fn(
        (descriptor: GPUShaderModuleDescriptor) => ({
          descriptor,
          getCompilationInfo: async () => ({ messages: [] }),
        }),
      ),
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBuffer: vi.fn(() => {
        const buffer = { destroy: vi.fn() };
        buffers.push(buffer);
        return buffer;
      }),
    } as unknown as GPUDevice;

    const kernel = await F16GemmKernel.create(
      device,
      {} as GpuActivationArena,
    );
    const expected = new Map<
      string,
      readonly [number, number, number, number?]
    >([
      [
        "encoder-layer-ff-expand",
        [15_400_960, 8_388_608, 61_603_840],
      ],
      [
        "encoder-layer-ff-contract",
        [61_603_840, 8_388_608, 15_400_960, 15_400_960],
      ],
      [
        "encoder-layer-attention-qkv",
        [15_400_960, 6_291_456, 46_202_880],
      ],
      [
        "encoder-layer-residual-1024",
        [15_400_960, 2_097_152, 15_400_960, 15_400_960],
      ],
      [
        "encoder-layer-convolution-expand",
        [15_400_960, 4_194_304, 30_801_920],
      ],
    ]);
    for (const [label, sizes] of expected) {
      const layout = layouts.find(
        (candidate) =>
          candidate.label === `parakeet-gemm-${label}-bindings`,
      );
      expect(layout).toBeDefined();
      const entries = Array.from(layout!.entries);
      expect(entries).toEqual(
        sizes.map((minBindingSize, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type:
              binding === 1 ? "read-only-storage" : "storage",
            minBindingSize,
          },
        })),
      );
    }

    const staged = layouts.find(
      (candidate) =>
        candidate.label ===
        "parakeet-gemm-encoder-projector-bindings",
    );
    expect(Array.from(staged!.entries)).toEqual([
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ]);

    kernel.destroy();
    for (const { destroy } of buffers) {
      expect(destroy).toHaveBeenCalledOnce();
    }
  });

  it("expands all packed production weights into one graph-local scratch", () => {
    expect(TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES).toBe(
      8_650_752,
    );
    expect(TRANSIENT_PALETTE_PARAMETER_SLOT_COUNT).toBe(9);
    expect(transientPaletteParameterPoolBytes(256)).toBe(
      2_304,
    );
    expect(dispatchSource).toContain(
      "this.preparation?.encode(encoder);",
    );
    expect(dispatchSource).toContain(
      "this.paletteExpander.createPooledDispatch(",
    );
    expect(dispatchSource).toContain(
      "this.paletteParameterPool.bindingFor(",
    );
    expect(dispatchSource).toContain(
      "source: descriptor.b.binding",
    );
    expect(dispatchSource).toContain(
      "palette: bPalette.binding",
    );
    expect(dispatchSource).toContain(
      "buffer: this.dequantizedWeightScratch",
    );
    expect(dispatchSource).toContain(
      "runtimeLayout: specialization.weightLayout",
    );
    expect(dispatchSource).toContain(
      "createPreparedPaletteWeight(",
    );
    expect(dispatchSource).toContain(
      "prepared.source !== descriptor.b",
    );
    expect(dispatchSource).toContain(
      "minStorageBufferOffsetAlignment",
    );
    expect(dispatchSource).toContain(
      "scratchOffset + size >",
    );
    expect(dispatchSource).toContain(
      "this.dequantizedWeightScratch.destroy();",
    );
    expect(dispatchSource).toContain(
      "this.paletteParameterPool.destroy();",
    );
    expect(dispatchSource).not.toContain(
      "new Set<PaletteExpansionDispatch>()",
    );
    expect(dispatchSource).toContain(
      'runtime.dtype !== "uint32"',
    );
    expect(dispatchSource).toContain(
      "runtimeGemmTileMajorStorageShape(",
    );
    expect(encoderSource).toContain(
      "SUBSAMPLING_FLATTEN_WEIGHT_OFFSET = 0",
    );
    expect(encoderSource).toContain(
      "SUBSAMPLING_POINTWISE3_WEIGHT_OFFSET =\n  SUBSAMPLING_FLATTEN_WEIGHT_BYTES",
    );
    expect(encoderSource).toContain(
      "SUBSAMPLING_POINTWISE6_WEIGHT_OFFSET =",
    );
    expect(encoderSource).toContain(
      "prepared.preparation.encode(encoder);",
    );
    expect(
      encoderSource.match(/preparedPaletteWeight:/g),
    ).toHaveLength(3);
  });
});

describe("production fp32 scalar-subgroup GEMM", () => {
  it("covers the fixed N128 encoder matrix shapes", () => {
    expect(
      planF32SubgroupGemm(
        7_520,
        1_024,
        4_096,
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [32, 235, 1],
    });
    expect(
      planF32SubgroupGemm(
        7_520,
        4_096,
        1_024,
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [8, 235, 1],
    });
    expect(
      planF32SubgroupGemm(
        7_520,
        1_024,
        3_072,
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [24, 235, 1],
    });
    expect(
      planF32SubgroupGemm(
        7_520,
        1_024,
        2_048,
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [16, 235, 1],
    });
    expect(
      planF32SubgroupGemm(
        7_520,
        1_024,
        640,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [5, 157, 1],
    });
    expect(
      planF32SubgroupGemm(
        24_064,
        256,
        256,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [2, 502, 1],
    });
  });

  it("plans every B1-B40 active-prefix tail", () => {
    for (let batch = 1; batch <= 40; batch++) {
      const rows = batch * 188;
      expect(
        planF32SubgroupGemm(
          rows,
          1_024,
          4_096,
          RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
        ),
      ).toEqual({
        workgroups: [32, Math.ceil(rows / 32), 1],
      });
      expect(
        planF32SubgroupGemm(
          rows,
          1_024,
          640,
          RUNTIME_ROW_MAJOR_F32_LAYOUT,
        ),
      ).toEqual({
        workgroups: [5, Math.ceil(rows / 48), 1],
      });
    }
    expect(
      planF32SubgroupGemm(
        1,
        32,
        4,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({
      workgroups: [1, 1, 1],
    });
  });

  it("rejects shapes outside the fp32 scalar contract", () => {
    expect(() =>
      planF32SubgroupGemm(
        48,
        1_023,
        1_024,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toThrow(/K divisible by 32/);
    expect(() =>
      planF32SubgroupGemm(
        48,
        1_024,
        1_022,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toThrow(/N divisible by 4/);
  });

  it("keeps the fixed signatures while selecting fp32 layouts", () => {
    expect(F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS).toEqual(
      F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.map(
        (specialization) => ({
          ...specialization,
          weightLayout:
            specialization.weightLayout ===
            RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
              ? RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
              : RUNTIME_ROW_MAJOR_F32_LAYOUT,
        }),
      ),
    );
  });

  it("uses dedicated fp32 A, B, C, and accumulator storage", () => {
    for (const specialization of F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f32SubgroupGemmWgsl(specialization);
      expect(source).toContain("enable subgroups;");
      expect(source).not.toContain("enable f16;");
      expect(source).not.toMatch(/\bf16\b/);
      expect(source).toContain(
        "matrix_a: array<vec4<f32>>;",
      );
      expect(source).toContain(
        "matrix_b: array<vec4<f32>>;",
      );
      expect(source).toContain(
        "matrix_c: array<vec4<f32>>;",
      );
      expect(source).toContain(
        "var packed_a = vec4<f32>(0.0);",
      );
      expect(source).toContain(
        "subgroupBroadcast(packed_a",
      );
      expect(
        source.match(/var acc\d+ = vec4<f32>/g),
      ).toHaveLength(8);
      expect(source.match(/ = fma\(/g)).toHaveLength(32);
      expect(source).not.toMatch(/subgroup.?matrix/i);
      expect(source).not.toContain("chromium_experimental");

      if (
        specialization.weightLayout ===
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
      ) {
        expect(source).toContain(
          `@compute @workgroup_size(${F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE}, 1, 1)`,
        );
        expect(source).toContain(
          `group.y * ${F32_SUBGROUP_DIRECT_B_TILE_ROWS}u`,
        );
        expect(source).not.toContain("var<workgroup>");
        expect(source).not.toContain("workgroupBarrier");
        expect(source).not.toContain("load_b_tile");
        expect(
          source.match(
            /let b = matrix_b\[\s+direct_b_tile_base \+/g,
          ),
        ).toHaveLength(4);
        expect(
          source.match(
            /let a\d+_\d+ = subgroupBroadcast\(/g,
          ),
        ).toHaveLength(32);
        expect(source).not.toContain(
          "let a0 = subgroupBroadcast(packed_a",
        );
      } else {
        expect(source).toContain(
          `@compute @workgroup_size(${F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE}, 1, 1)`,
        );
        expect(source).toContain(
          `group.y * ${F32_SUBGROUP_STAGED_B_TILE_ROWS}u`,
        );
        expect(source).toContain(
          "var<workgroup> tile_b:\n  array<vec4<f32>, 1024>;",
        );
        expect(source).toContain(
          `item += ${F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE}u`,
        );
        expect(
          source.match(/workgroupBarrier\(\);/g),
        ).toHaveLength(2);
        expect(source).toContain("fn load_b_tile(");
        expect(
          source.match(/let b = tile_b\[/g),
        ).toHaveLength(4);
        expect(
          source.match(
            /let a\d+ = subgroupBroadcast\(packed_a/g,
          ),
        ).toHaveLength(8);
        expect(source).not.toContain(
          "let a0_0 = subgroupBroadcast(",
        );
      }
    }
  });

  it("retains epilogues and specializes direct K traversal", () => {
    const ffExpand = fp32SourceFor(
      "encoder-layer-ff-expand",
    );
    expect(ffExpand).toContain("exp(-bounded)");
    expect(ffExpand).toContain(
      "let direct_b_tile_base =\n    group.x * 32768u;",
    );
    expect(ffExpand).toContain(
      "k_vector < A_VECTORS_PER_ROW;",
    );
    expect(ffExpand).not.toContain("var k_base = 0u;");
    expect(ffExpand).toContain(
      "a_row * A_VECTORS_PER_ROW + k_vector",
    );
    expect(ffExpand).not.toContain("residual:");
    expect(ffExpand).not.toContain("bias:");

    const ffContract = fp32SourceFor(
      "encoder-layer-ff-contract",
    );
    expect(ffContract).toContain(
      "residual: array<vec4<f32>>;",
    );
    expect(ffContract).toContain("+ 0.5 * value;");
    expect(ffContract).toContain(
      "matrix_c[output_index] = value;",
    );
    expect(ffContract).toContain("var k_base = 0u;");
    expect(ffContract).toContain("k_base += 32u");
    expect(ffContract).toContain(
      `(group.x * 128u +\n        k_base / 32u) *\n      1024u;`,
    );

    const residualOne = fp32SourceFor(
      "encoder-layer-residual-1024",
    );
    expect(residualOne).toContain("+ 1.0 * value;");

    const pointwise = fp32SourceFor(
      "subsampling-pointwise3",
    );
    expect(pointwise).toContain("bias: array<f32>;");
    expect(pointwise).toContain(
      "value = max(value, vec4<f32>(0.0));",
    );
  });

  it("pins the fp32 geometry and transient scratch frontier", () => {
    expect(F32_SUBGROUP_TILE_COLUMNS).toBe(128);
    expect(F32_SUBGROUP_TILE_INNER).toBe(32);
    expect(F32_SUBGROUP_DIRECT_B_TILE_ROWS).toBe(32);
    expect(F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE).toBe(128);
    expect(F32_SUBGROUP_STAGED_B_TILE_ROWS).toBe(48);
    expect(F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE).toBe(192);
    expect(F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES).toBe(
      16_384,
    );
    expect(
      FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES,
    ).toBe(17_301_504);
  });

  it("creates every fp32 pipeline without requesting shader-f16", async () => {
    const buffers: Array<{
      readonly descriptor: GPUBufferDescriptor;
      readonly destroy: ReturnType<typeof vi.fn>;
    }> = [];
    const createShaderModule = vi.fn(
      (descriptor: GPUShaderModuleDescriptor) => ({
        descriptor,
        getCompilationInfo: async () => ({ messages: [] }),
      }),
    );
    const device = {
      features: new Set(["subgroups"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupStorageSize: 32_768,
        minStorageBufferOffsetAlignment: 256,
        minUniformBufferOffsetAlignment: 256,
      },
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule,
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBuffer: vi.fn(
        (descriptor: GPUBufferDescriptor) => {
          const buffer = {
            descriptor,
            destroy: vi.fn(),
          };
          buffers.push(buffer);
          return buffer;
        },
      ),
    } as unknown as GPUDevice;

    const kernel = await F32GemmKernel.create(
      device,
      {} as GpuActivationArena,
    );

    expect(createShaderModule).toHaveBeenCalledTimes(
      F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.length + 3,
    );
    for (const [descriptor] of createShaderModule.mock.calls) {
      expect(descriptor.code).not.toMatch(/\bf16\b/);
    }
    expect(
      createShaderModule.mock.calls
        .map(([descriptor]) => descriptor.label)
        .filter((label) =>
          label?.startsWith("parakeet-fp32-gemm-") === true,
        ),
    ).toHaveLength(
      F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.length,
    );
    expect(
      buffers.map(({ descriptor }) => descriptor),
    ).toContainEqual({
      label:
        "parakeet-fp32-transient-palette-weight-scratch",
      size: 17_301_504,
      usage: GPUBufferUsage.STORAGE,
    });
    kernel.destroy();
    for (const { destroy } of buffers) {
      expect(destroy).toHaveBeenCalledOnce();
    }
  });
});

function sourceFor(
  label: (typeof F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS)[number]["label"],
): string {
  const specialization = F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.find(
    (candidate) => candidate.label === label,
  );
  if (specialization === undefined) {
    throw new Error(`Missing GEMM specialization ${label}`);
  }
  return f16SubgroupGemmWgsl(specialization);
}

function fp32SourceFor(
  label: (typeof F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS)[number]["label"],
): string {
  const specialization =
    F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.find(
      (candidate) => candidate.label === label,
    );
  if (specialization === undefined) {
    throw new Error(`Missing FP32 GEMM specialization ${label}`);
  }
  return f32SubgroupGemmWgsl(specialization);
}

function loadFunction(source: string): string {
  const match = source.match(
    /fn load_b_tile\([\s\S]*?\n}\n(?=\n@compute)/,
  );
  if (match === null) throw new Error("Missing scalar GEMM B loader");
  return match[0];
}
