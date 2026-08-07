import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it, vi } from "vitest";

import {
  paletteExpansionParameterPoolBytes,
  paletteDequantFp32Wgsl,
  paletteDequantWgsl,
  PaletteExpansionParameterPool,
  PaletteWeightExpander,
} from "../../src/model/palette-expander";
import {
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
  runtimeGemmTileMajorF32ScalarIndex,
  runtimeGemmTileMajorScalarIndex,
} from "../../src/model/runtime-plan";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../../src/webgpu/capabilities";

const source = readFileSync(
  new URL("../../src/model/palette-expander.ts", import.meta.url),
  "utf8",
);

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { COPY_DST: 1, UNIFORM: 2 },
  });
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true,
    value: { COMPUTE: 1 },
  });
});

describe("palette weight expander", () => {
  it("preserves the exact palette5 and palette6 unpacking geometry", () => {
    const palette5 = paletteDequantWgsl(
      5,
      RUNTIME_ROW_MAJOR_F16_LAYOUT,
    );
    expect(palette5).toContain(
      "@compute @workgroup_size(256, 1, 1)",
    );
    expect(palette5.match(/let word\d =/g)).toHaveLength(5);
    expect(palette5.match(/dequantized\[scalar_base/g)).toHaveLength(8);
    expect(palette5).toContain("& 31u");

    const palette6 = paletteDequantWgsl(
      6,
      RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    );
    expect(palette6.match(/let word\d =/g)).toHaveLength(3);
    expect(
      palette6.match(/dequantized\[tile_major_vector/g),
    ).toHaveLength(4);
    expect(palette6).toContain("& 63u");
    for (const shader of [palette5, palette6]) {
      expect(shader).toContain("enable f16;");
      expect(shader).not.toMatch(/subgroup.?matrix/i);
      expect(shader).not.toContain("chromium_experimental");
    }
    expect(() =>
      paletteDequantWgsl(6, RUNTIME_ROW_MAJOR_F16_LAYOUT),
    ).toThrow(/no row-major palette6/);
  });

  it("writes palette5 and palette6 values directly into tile-major vectors", () => {
    for (const bits of [5, 6] as const) {
      const shader = paletteDequantWgsl(
        bits,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      );
      const vectorsPerGroup = bits === 5 ? 8 : 4;
      expect(
        shader.match(
          /dequantized\[tile_major_vector\(scalar_base \+ \d+u\)\]/g,
        ),
      ).toHaveLength(vectorsPerGroup);
      expect(shader).toContain(
        "let column = scalar_index % params.columns;",
      );
      expect(shader).toContain(
        "column / params.palette_group_columns",
      );
      expect(shader).toContain("let n_tile = n / 256u;");
      expect(shader).toContain("let k_tile = k / 32u;");
      expect(shader).not.toContain(
        "dequantized[scalar_base / 4u",
      );
      expect(shader).not.toContain("transient");
    }
  });

  it("widens only scalar storage in dedicated fp32 shaders", () => {
    for (const [bits, runtimeLayout] of [
      [5, RUNTIME_ROW_MAJOR_F16_LAYOUT],
      [5, RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT],
      [6, RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT],
    ] as const) {
      const fp16 = paletteDequantWgsl(bits, runtimeLayout);
      const fp32RuntimeLayout =
        runtimeLayout === RUNTIME_ROW_MAJOR_F16_LAYOUT
          ? RUNTIME_ROW_MAJOR_F32_LAYOUT
          : RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT;
      const fp32 = paletteDequantFp32Wgsl(
        bits,
        fp32RuntimeLayout,
      );
      expect(fp32).not.toContain("enable f16;");
      expect(fp32).not.toMatch(/\bf16\b/);
      expect(fp32).toContain(
        "palette_values: array<vec4<f32>>",
      );
      expect(fp32).toContain(
        "dequantized: array<vec4<f32>>",
      );
      expect(fp32).toContain(
        "fn palette_value(element: u32) -> f32",
      );
      expect(fp32).toContain(
        "@compute @workgroup_size(256, 1, 1)",
      );

      const normalizedFp16 = fp16
        .replace("enable f16;\n\n", "")
        .replaceAll("f16", "scalar");
      const normalizedFp32 = fp32.replaceAll(
        "f32",
        "scalar",
      ).replaceAll("128u", "256u");
      expect(normalizedFp32).toBe(normalizedFp16);
    }
    expect(() =>
      paletteDequantFp32Wgsl(
        6,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toThrow(/no row-major palette6/);
  });

  it("writes fp32 tile-major vectors into N128/K32 physical tiles", () => {
    const shader = paletteDequantFp32Wgsl(
      5,
      RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
    );
    expect(shader).toContain("let n_tile = n / 128u;");
    expect(shader).toContain("let n_in_tile = n % 128u;");
    expect(shader).toContain("128u + n_in_tile");
    expect(shader).not.toContain("256u");

    const inner = 64;
    const columns = 256;
    for (let scalar = 0; scalar < inner * columns; scalar += 4) {
      const physical = runtimeGemmTileMajorF32ScalarIndex(
        scalar,
        inner,
        columns,
      );
      expect(physical % 4).toBe(0);
      for (let component = 1; component < 4; component++) {
        expect(
          runtimeGemmTileMajorF32ScalarIndex(
            scalar + component,
            inner,
            columns,
          ),
        ).toBe(physical + component);
      }
    }
  });

  it("keeps palette5/6 groups and vec4 stores intact across the tile mapping", () => {
    for (const {
      bits,
      columns,
      paletteGroupColumns,
    } of [
      { bits: 5, columns: 3072, paletteGroupColumns: 1024 },
      { bits: 6, columns: 1024, paletteGroupColumns: 1024 },
    ] as const) {
      const inner = 64;
      const codesPerGroup = bits === 5 ? 32 : 16;
      let groupsChecked = 0;
      let vectorsChecked = 0;
      for (
        let scalarBase = 0;
        scalarBase < inner * columns;
        scalarBase += codesPerGroup
      ) {
        groupsChecked += 1;
        const paletteGroup = Math.floor(
          (scalarBase % columns) / paletteGroupColumns,
        );
        for (let code = 0; code < codesPerGroup; code++) {
          const actualGroup = Math.floor(
            ((scalarBase + code) % columns) /
              paletteGroupColumns,
          );
          if (actualGroup !== paletteGroup) {
            throw new Error(
              `Palette${bits} code group crossed a palette boundary`,
            );
          }
        }
        for (
          let vectorOffset = 0;
          vectorOffset < codesPerGroup;
          vectorOffset += 4
        ) {
          vectorsChecked += 1;
          const logical = scalarBase + vectorOffset;
          const physical = runtimeGemmTileMajorScalarIndex(
            logical,
            inner,
            columns,
          );
          if (physical % 4 !== 0) {
            throw new Error(
              `Palette${bits} tile store is not vec4 aligned`,
            );
          }
          for (let component = 1; component < 4; component++) {
            const mapped = runtimeGemmTileMajorScalarIndex(
              logical + component,
              inner,
              columns,
            );
            if (mapped !== physical + component) {
              throw new Error(
                `Palette${bits} vec4 crossed a physical tile boundary`,
              );
            }
          }
        }
      }
      expect(groupsChecked).toBe(
        inner * columns / codesPerGroup,
      );
      expect(vectorsChecked).toBe(inner * columns / 4);
    }
  });

  it("compiles each fixed bit-width/layout pair and binds descriptor ranges directly", () => {
    expect(source).toContain(
      "pipelineConfigurations.map",
    );
    expect(source).toContain("PIPELINE_CONFIGURATIONS");
    expect(source).toContain("FP32_PIPELINE_CONFIGURATIONS");
    expect(source).toContain("pipelineKey(bits, runtimeLayout)");
    expect(source).toContain(
      "{ binding: 0, resource: descriptor.source }",
    );
    expect(source).toContain(
      "{ binding: 1, resource: descriptor.palette }",
    );
    expect(source).toContain(
      "{ binding: 2, resource: descriptor.target }",
    );
    expect(source).toContain("createPooledDispatch(");
    expect(source).not.toContain("createTransientDispatch(");
  });

  it("rejects invalid pooled bindings before bind-group construction", async () => {
    const createBindGroup = vi.fn(() => ({}));
    const device = {
      features: new Set(["shader-f16"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        minUniformBufferOffsetAlignment: 256,
      },
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule: vi.fn(() => ({
        getCompilationInfo: async () => ({ messages: [] }),
      })),
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBindGroup,
    } as unknown as GPUDevice;
    const expander = await PaletteWeightExpander.create(device);
    expect(device.createShaderModule).toHaveBeenCalledTimes(3);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    expect(
      vi.mocked(device.createShaderModule).mock.calls.map(
        ([descriptor]) => descriptor,
      ),
    ).toEqual([
      {
        label:
          "parakeet-palette5-k-by-output-row-major-f16-expansion",
        code: paletteDequantWgsl(
          5,
          RUNTIME_ROW_MAJOR_F16_LAYOUT,
        ),
      },
      {
        label:
          "parakeet-palette5-n256-k32-tile-major-f16-expansion",
        code: paletteDequantWgsl(
          5,
          RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
        ),
      },
      {
        label:
          "parakeet-palette6-n256-k32-tile-major-f16-expansion",
        code: paletteDequantWgsl(
          6,
          RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
        ),
      },
    ]);

    expect(() =>
      expander.createPooledDispatch(
        testTransientDescriptor(5),
        {
          buffer: {} as GPUBuffer,
          offset: 0,
          size: 8,
        },
      ),
    ).toThrow(/invalid pooled parameter binding/);
    expect(() =>
      expander.createPooledDispatch(
        testTransientDescriptor(5),
        {
          buffer: {} as GPUBuffer,
          offset: 128,
          size: 16,
        },
      ),
    ).toThrow(/invalid pooled parameter binding/);
    expect(createBindGroup).not.toHaveBeenCalled();
  });

  it("creates fp32 pipelines without shader-f16 and validates fp32 scratch geometry", async () => {
    const destroy = vi.fn();
    const pooledBuffer = { destroy } as unknown as GPUBuffer;
    const createBindGroup = vi.fn(() => ({}));
    const device = {
      features: new Set(["subgroups"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        minUniformBufferOffsetAlignment: 256,
      },
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule: vi.fn(
        (descriptor: GPUShaderModuleDescriptor) => ({
          descriptor,
          getCompilationInfo: async () => ({ messages: [] }),
        }),
      ),
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBuffer: vi.fn(() => pooledBuffer),
      createBindGroup,
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;

    const expander = await PaletteWeightExpander.create(
      device,
      PARAKEET_FP32_EXECUTION_PROFILE,
    );

    expect(expander.executionProfile).toBe(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(device.createShaderModule).toHaveBeenCalledTimes(3);
    for (const [descriptor] of vi.mocked(
      device.createShaderModule,
    ).mock.calls) {
      expect(descriptor.label).toMatch(/-fp32-expansion$/);
      expect(descriptor.code).not.toMatch(/\bf16\b/);
      expect(descriptor.code).toContain("vec4<f32>");
    }

    const descriptor = testTransientDescriptor(
      5,
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(descriptor.palette.size).toBe(128);
    expect(descriptor.target.size).toBe(32_768);
    const pool = new PaletteExpansionParameterPool(
      device,
      1,
      "test-fp32-palette-parameter-pool",
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    const parameters = pool.bindingFor(descriptor);
    expect(() =>
      expander.createPooledDispatch(descriptor, parameters),
    ).not.toThrow();
    expect(createBindGroup).toHaveBeenCalledOnce();
    expect(() =>
      expander.createPooledDispatch(
        testTransientDescriptor(
          5,
          PARAKEET_FP16_EXECUTION_PROFILE,
        ),
        parameters,
      ),
    ).toThrow(/invalid transient palette geometry/);

    await expect(
      PaletteWeightExpander.create(
        device,
        PARAKEET_FP16_EXECUTION_PROFILE,
      ),
    ).rejects.toThrow(/shader-f16/);
    pool.destroy();
    expect(destroy).toHaveBeenCalledOnce();
  });

  it("binds resident palette tensors to transient scratch through pooled parameters", async () => {
    const destroy = vi.fn();
    const pooledBuffer = { destroy } as unknown as GPUBuffer;
    const setPipeline = vi.fn();
    const setBindGroup = vi.fn();
    const dispatchWorkgroups = vi.fn();
    const end = vi.fn();
    const createBindGroup = vi.fn(
      (_descriptor: GPUBindGroupDescriptor) => ({}),
    );
    const device = {
      features: new Set(["shader-f16"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        minUniformBufferOffsetAlignment: 256,
      },
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule: vi.fn(() => ({
        getCompilationInfo: async () => ({ messages: [] }),
      })),
      createComputePipelineAsync: vi.fn(async () => ({})),
      createBuffer: vi.fn(() => pooledBuffer),
      createBindGroup,
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const expander = await PaletteWeightExpander.create(device);
    const sourceBuffer = {} as GPUBuffer;
    const paletteBuffer = {} as GPUBuffer;
    const scratchBuffer = {} as GPUBuffer;
    const descriptor = {
      label: "encoder-layer-palette-expand",
      source: {
        buffer: sourceBuffer,
        offset: 256,
        size: 5_120,
      },
      palette: {
        buffer: paletteBuffer,
        offset: 512,
        size: 64,
      },
      target: {
        buffer: scratchBuffer,
        offset: 0,
        size: 16_384,
      },
      scalarCount: 8_192,
      columns: 256,
      paletteBits: 5,
      paletteGroupColumns: 256,
      paletteGroups: 1,
      runtimeLayout:
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    } as const;
    const pool = new PaletteExpansionParameterPool(
      device,
      1,
      "test-palette-parameter-pool",
    );
    const parameters = pool.bindingFor(descriptor);
    const dispatch = expander.createPooledDispatch(
      descriptor,
      parameters,
    );
    const pooledBindGroupCall =
      createBindGroup.mock.calls.at(-1)?.[0];
    expect(pooledBindGroupCall?.entries).toEqual([
      {
        binding: 0,
        resource: {
          buffer: sourceBuffer,
          offset: 256,
          size: 5_120,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: paletteBuffer,
          offset: 512,
          size: 64,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: scratchBuffer,
          offset: 0,
          size: 16_384,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: pooledBuffer,
          offset: 0,
          size: 16,
        },
      },
    ]);
    const encoder = {
      beginComputePass: vi.fn(() => ({
        setPipeline,
        setBindGroup,
        dispatchWorkgroups,
        end,
      })),
    } as unknown as GPUCommandEncoder;
    dispatch.encode(encoder);
    expect(dispatchWorkgroups).toHaveBeenCalledWith(1);
    expect(end).toHaveBeenCalledOnce();
    dispatch.destroy();
    dispatch.destroy();
    expect(() => dispatch.encode(encoder)).toThrow(
      /palette dispatch was destroyed/,
    );
    expect(destroy).not.toHaveBeenCalled();
    pool.destroy();
    pool.destroy();
    expect(destroy).toHaveBeenCalledOnce();
  });

  it("deduplicates aligned transient parameters in one owned pool", () => {
    expect(paletteExpansionParameterPoolBytes(9, 256)).toBe(
      2_304,
    );
    const destroy = vi.fn();
    const pooledBuffer = { destroy } as unknown as GPUBuffer;
    const writeBuffer = vi.fn();
    const device = {
      limits: {
        minUniformBufferOffsetAlignment: 256,
      },
      createBuffer: vi.fn(() => pooledBuffer),
      queue: { writeBuffer },
    } as unknown as GPUDevice;
    const pool = new PaletteExpansionParameterPool(
      device,
      9,
      "test-palette-parameter-pool",
    );
    expect(pool.byteLength).toBe(2_304);
    expect(pool.slotStride).toBe(256);

    const palette5 = testTransientDescriptor(5);
    const first = pool.bindingFor(palette5);
    const duplicate = pool.bindingFor({
      ...palette5,
      label: "same-geometry-different-layer",
    });
    expect(duplicate).toBe(first);
    expect(first).toEqual({
      buffer: pooledBuffer,
      offset: 0,
      size: 16,
    });
    expect(writeBuffer).toHaveBeenCalledTimes(1);
    expect(
      Array.from(writeBuffer.mock.calls[0]![2] as Uint32Array),
    ).toEqual([8_192, 256, 256, 1]);

    const palette6 = testTransientDescriptor(6);
    expect(pool.bindingFor(palette6)).toEqual({
      buffer: pooledBuffer,
      offset: 256,
      size: 16,
    });
    expect(pool.usedSlots).toBe(2);
    expect(writeBuffer).toHaveBeenCalledTimes(2);

    pool.destroy();
    pool.destroy();
    expect(destroy).toHaveBeenCalledOnce();
    expect(() => pool.bindingFor(palette5)).toThrow(
      /pool was destroyed/,
    );
  });
});

function testTransientDescriptor(
  paletteBits: 5 | 6,
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
) {
  const codesPerGroup = paletteBits === 5 ? 32 : 16;
  const wordsPerGroup = paletteBits === 5 ? 5 : 3;
  const scalarCount = 8_192;
  const scalarBytes =
    executionProfile.precision === "fp16"
      ? Uint16Array.BYTES_PER_ELEMENT
      : Float32Array.BYTES_PER_ELEMENT;
  return {
    label: `test-palette${paletteBits}`,
    source: {
      buffer: {} as GPUBuffer,
      offset: 256,
      size:
        scalarCount /
        codesPerGroup *
        wordsPerGroup *
        Uint32Array.BYTES_PER_ELEMENT,
    },
    palette: {
      buffer: {} as GPUBuffer,
      offset: 512,
      size:
        (1 << paletteBits) *
        scalarBytes,
    },
    target: {
      buffer: {} as GPUBuffer,
      offset: 0,
      size:
        scalarCount * scalarBytes,
    },
    scalarCount,
    columns: 256,
    paletteBits,
    paletteGroupColumns: 256,
    paletteGroups: 1,
    runtimeLayout:
      executionProfile.precision === "fp16"
        ? RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
        : RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  } as const;
}
