import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import {
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
} from "../../src/model/runtime-plan";
import {
  F16_PORTABLE_GEMM_WORKGROUP_BYTES,
  F16_PORTABLE_BITCAST_LOAD_ENCODING,
  F16_PORTABLE_NATIVE_LOAD_ENCODING,
  F16_PORTABLE_UNPACK_LOAD_ENCODING,
  F16_SUBGROUP_DIRECT_B_TILE_ROWS,
  F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
  F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS,
  F16_SUBGROUP_STAGED_B_TILE_ROWS,
  F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
  F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
  F16_SUBGROUP_TILE_COLUMNS,
  F16_SUBGROUP_TILE_INNER,
  F32_PORTABLE_GEMM_WORKGROUP_BYTES,
  F32_SUBGROUP_DIRECT_B_TILE_ROWS,
  F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
  F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS,
  F32_SUBGROUP_STAGED_B_TILE_ROWS,
  F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
  F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
  F32_SUBGROUP_TILE_COLUMNS,
  F32_SUBGROUP_TILE_INNER,
  PORTABLE_DIRECT_GEMM_GEOMETRY,
  PORTABLE_FP32_DIRECT_GEMM_GEOMETRY,
  PORTABLE_FP32_STAGED_GEMM_GEOMETRY,
  PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY,
  PORTABLE_STAGED_GEMM_GEOMETRY,
  f16PortableGemmWgsl,
  f16SubgroupGemmWgsl,
  f32PortableGemmWgsl,
  f32SubgroupGemmWgsl,
  planPortableGemm,
  selectF16PortableLoadEncoding,
} from "../../src/webgpu/kernels/gemm";

describe("portable GEMM planning", () => {
  it("pins the thermally selected FP16 direct and staged geometries", () => {
    expect(PORTABLE_DIRECT_GEMM_GEOMETRY).toEqual({
      tileRows: 48,
      tileColumns: 128,
      tileInner: 128,
      workgroupSize: 128,
    });
    expect(PORTABLE_STAGED_GEMM_GEOMETRY).toEqual({
      tileRows: 48,
      tileColumns: 128,
      tileInner: 32,
      workgroupSize: 128,
    });
    expect(Object.isFrozen(PORTABLE_DIRECT_GEMM_GEOMETRY)).toBe(true);
    expect(Object.isFrozen(PORTABLE_STAGED_GEMM_GEOMETRY)).toBe(true);
  });

  it("pins the tuned FP32 direct, wide-direct, and staged geometries", () => {
    expect(PORTABLE_FP32_DIRECT_GEMM_GEOMETRY).toEqual({
      tileRows: 40,
      tileColumns: 128,
      tileInner: 128,
      workgroupSize: 128,
    });
    expect(PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY).toEqual({
      tileRows: 36,
      tileColumns: 128,
      tileInner: 128,
      workgroupSize: 128,
    });
    expect(PORTABLE_FP32_STAGED_GEMM_GEOMETRY).toEqual({
      tileRows: 32,
      tileColumns: 128,
      tileInner: 32,
      workgroupSize: 128,
    });
    expect(Object.isFrozen(PORTABLE_FP32_DIRECT_GEMM_GEOMETRY)).toBe(true);
    expect(Object.isFrozen(PORTABLE_FP32_WIDE_DIRECT_GEMM_GEOMETRY)).toBe(
      true,
    );
    expect(Object.isFrozen(PORTABLE_FP32_STAGED_GEMM_GEOMETRY)).toBe(true);
  });

  it("plans every fixed FP16 specialization with its selected geometry", () => {
    for (const specialization of F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      expect(
        planPortableGemm(
          specialization.rows,
          specialization.inner,
          specialization.columns,
          specialization.weightLayout,
        ),
      ).toEqual({
        workgroups: [
          Math.ceil(specialization.columns / 128),
          Math.ceil(specialization.rows / 48),
          1,
        ],
      });
    }
  });

  it("plans every B1-B40 FP16 direct and staged active tail", () => {
    for (let batch = 1; batch <= 40; batch++) {
      const rows = batch * 188;
      expect(
        planPortableGemm(
          rows,
          1_024,
          4_096,
          RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
        ),
      ).toEqual({
        workgroups: [32, Math.ceil(rows / 48), 1],
      });
      expect(
        planPortableGemm(
          rows,
          1_024,
          640,
          RUNTIME_ROW_MAJOR_F16_LAYOUT,
        ),
      ).toEqual({
        workgroups: [5, Math.ceil(rows / 48), 1],
      });
    }
    expect(
      planPortableGemm(
        39 * 188,
        1_024,
        4_096,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({ workgroups: [32, 153, 1] });
    expect(
      planPortableGemm(
        40 * 188,
        1_024,
        640,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toEqual({ workgroups: [5, 157, 1] });
  });

  it("selects each FP32 specialization and every B1-B40 active tail", () => {
    for (const specialization of F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const tileRows =
        specialization.weightLayout ===
        RUNTIME_ROW_MAJOR_F32_LAYOUT
          ? 32
          : specialization.inner === 1_024 &&
              (specialization.columns === 2_048 ||
                specialization.columns === 3_072)
            ? 36
            : 40;
      expect(
        planPortableGemm(
          specialization.rows,
          specialization.inner,
          specialization.columns,
          specialization.weightLayout,
        ),
      ).toEqual({
        workgroups: [
          Math.ceil(specialization.columns / 128),
          Math.ceil(specialization.rows / tileRows),
          1,
        ],
      });
    }

    for (let batch = 1; batch <= 40; batch++) {
      const rows = batch * 188;
      expect(
        planPortableGemm(
          rows,
          1_024,
          4_096,
          RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
        ),
      ).toEqual({ workgroups: [32, Math.ceil(rows / 40), 1] });
      expect(
        planPortableGemm(
          rows,
          1_024,
          3_072,
          RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
        ),
      ).toEqual({ workgroups: [24, Math.ceil(rows / 36), 1] });
      expect(
        planPortableGemm(
          rows,
          1_024,
          640,
          RUNTIME_ROW_MAJOR_F32_LAYOUT,
        ),
      ).toEqual({ workgroups: [5, Math.ceil(rows / 32), 1] });
    }
    expect(
      planPortableGemm(
        39 * 188,
        1_024,
        3_072,
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({ workgroups: [24, 204, 1] });
    expect(
      planPortableGemm(
        40 * 188,
        1_024,
        640,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({ workgroups: [5, 235, 1] });
  });

  it("enforces each selected K tile without overconstraining N tails", () => {
    expect(() =>
      planPortableGemm(
        48,
        96,
        128,
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
      ),
    ).toThrow(/K divisible by 128/);
    expect(() =>
      planPortableGemm(
        48,
        1_024,
        126,
        RUNTIME_ROW_MAJOR_F16_LAYOUT,
      ),
    ).toThrow(/N divisible by 4/);
    expect(
      planPortableGemm(
        1,
        32,
        4,
        RUNTIME_ROW_MAJOR_F32_LAYOUT,
      ),
    ).toEqual({ workgroups: [1, 1, 1] });
  });
});

describe("portable GEMM workgroup storage", () => {
  it("pins the exact FP16 high-water footprint", () => {
    const direct = f16PortableSource("encoder-layer-ff-expand");
    const staged = f16PortableSource("encoder-projector");

    expect(F16_PORTABLE_GEMM_WORKGROUP_BYTES).toBe(12_288);
    expect(direct).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f16>, 1536>;",
    );
    expect(direct).not.toContain("var<workgroup> tile_b:");
    expect(1_536 * 4 * Uint16Array.BYTES_PER_ELEMENT).toBe(
      F16_PORTABLE_GEMM_WORKGROUP_BYTES,
    );

    expect(staged).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f16>, 384>;",
    );
    expect(staged).toContain(
      "var<workgroup> tile_b:\n  array<vec4<f16>, 1024>;",
    );
    expect((384 + 1_024) * 4 * Uint16Array.BYTES_PER_ELEMENT).toBe(
      11_264,
    );
  });

  it("pins the exact FP32 high-water footprint", () => {
    const direct = f32PortableSource("encoder-layer-ff-expand");
    const wide = f32PortableSource("encoder-layer-attention-qkv");
    const staged = f32PortableSource("encoder-projector");

    expect(F32_PORTABLE_GEMM_WORKGROUP_BYTES).toBe(20_480);
    expect(direct).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f32>, 1280>;",
    );
    expect(direct).not.toContain("var<workgroup> tile_b:");
    expect(1_280 * 4 * Float32Array.BYTES_PER_ELEMENT).toBe(
      F32_PORTABLE_GEMM_WORKGROUP_BYTES,
    );

    expect(wide).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f32>, 1152>;",
    );
    expect(staged).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f32>, 256>;",
    );
    expect(staged).toContain(
      "var<workgroup> tile_b:\n  array<vec4<f32>, 1024>;",
    );
    expect((256 + 1_024) * 4 * Float32Array.BYTES_PER_ELEMENT).toBe(
      F32_PORTABLE_GEMM_WORKGROUP_BYTES,
    );
  });
});

describe("portable GEMM WGSL", () => {
  it("uses no subgroup feature, builtin, or operation", () => {
    for (const specialization of F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f16PortableGemmWgsl(specialization);
      expect(source).not.toMatch(/\bsubgroup/i);
      expect(source).not.toContain("chromium_experimental");
      expect(source).not.toMatch(/subgroup.?matrix/i);
    }
    for (const specialization of F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f32PortableGemmWgsl(specialization);
      expect(source).not.toMatch(/\bsubgroup/i);
      expect(source).not.toContain("chromium_experimental");
      expect(source).not.toMatch(/subgroup.?matrix/i);
    }
  });

  it("retains increasing-K FP16 FMA order and FP16 accumulators", () => {
    for (const specialization of F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f16PortableGemmWgsl(specialization);
      const direct =
        specialization.weightLayout ===
        RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT;

      expect(source).toContain("enable f16;");
      expect(source).toContain("var acc0 = vec4<f16>(0.0h);");
      expect(source).toContain(
        `k_base += ${direct ? "128" : "32"}u`,
      );
      expect(source).toContain("k_vector += 1u");
      expect(componentOrder(source)).toEqual([0, 1, 2, 3]);
      expect(source).toContain(
        "acc0 = fma(vec4<f16>(f16(a0[0])), b, acc0);",
      );
      expect(source.indexOf("var value = vec4<f32>(acc0);")).toBeGreaterThan(
        source.lastIndexOf("acc0 = fma("),
      );
      expect(source).toContain("matrix_c[output_index] = vec4<f16>(value);");
      expect(source.match(/var acc\d+ = vec4<f16>/g)).toHaveLength(12);
      expect(source.match(/ = fma\(/g)).toHaveLength(48);
    }
  });

  it("emits exact compiler-selected FP16 packed-load encodings", () => {
    const specialization =
      F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS[0];
    const bitcast = f16PortableGemmWgsl(
      specialization,
      F16_PORTABLE_BITCAST_LOAD_ENCODING,
    );
    const unpack = f16PortableGemmWgsl(
      specialization,
      F16_PORTABLE_UNPACK_LOAD_ENCODING,
    );
    const native = f16PortableGemmWgsl(
      specialization,
      F16_PORTABLE_NATIVE_LOAD_ENCODING,
    );

    expect(f16PortableGemmWgsl(specialization)).toBe(native);
    expect(bitcast).toContain(
      "return bitcast<vec4<f16>>(packed);",
    );
    expect(bitcast).not.toContain("unpack2x16float");
    expect(unpack).toContain("let xy = unpack2x16float(packed.x);");
    expect(unpack).toContain("let zw = unpack2x16float(packed.y);");
    expect(unpack).not.toContain(
      "return bitcast<vec4<f16>>(packed);",
    );
    expect(native).toContain("matrix_a: array<vec4<f16>>;");
    expect(native).toContain("matrix_b: array<vec4<f16>>;");
    expect(native).toContain(
      "var<workgroup> tile_a:\n  array<vec4<f16>",
    );
    expect(native).not.toContain("fn unpack_f16x4(");
    expect(native).not.toContain("unpack2x16float");
    expect(native).not.toContain("bitcast<vec4<f16>>");
    expect(native).toContain("let a0 = tile_a[");
    expect(native).toContain("let b = matrix_b[");
    expect(native).toContain(
      "acc0 = fma(vec4<f16>(f16(a0[0])), b, acc0);",
    );
    expect(bitcast).not.toMatch(/\bsubgroup/i);
    expect(unpack).not.toMatch(/\bsubgroup/i);
    expect(native).not.toMatch(/\bsubgroup/i);
  });

  it("retains increasing-K FP32 FMA order and FP32-only arithmetic", () => {
    for (const specialization of F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS) {
      const source = f32PortableGemmWgsl(specialization);
      const direct =
        specialization.weightLayout ===
        RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT;
      const wide =
        direct &&
        specialization.inner === 1_024 &&
        (specialization.columns === 2_048 ||
          specialization.columns === 3_072);
      const accumulatorCount = direct ? (wide ? 9 : 10) : 8;

      expect(source).not.toContain("enable f16;");
      expect(source).not.toMatch(/\bf16\b/);
      expect(source).toContain("var acc0 = vec4<f32>(0.0);");
      expect(source).toContain(
        `k_base += ${direct ? "128" : "32"}u`,
      );
      expect(source).toContain("k_vector += 1u");
      expect(componentOrder(source)).toEqual([0, 1, 2, 3]);
      expect(source).toContain(
        "acc0 = fma(vec4<f32>(a0[0]), b, acc0);",
      );
      expect(source.match(/var acc\d+ = vec4<f32>/g)).toHaveLength(
        accumulatorCount,
      );
      expect(source.match(/ = fma\(/g)).toHaveLength(
        accumulatorCount * 4,
      );
      expect(source).toContain("matrix_c[output_index] = value;");
    }
  });
});

describe("portable FP16 packed-load compiler probe", () => {
  it("selects bitcast when accepted and native f16 storage on a reported compiler error", async () => {
    const accepted = probeDevice([]);
    const warningOnly = probeDevice([{ type: "warning" }]);
    const rejected = probeDevice([{ type: "error" }]);

    await expect(
      selectF16PortableLoadEncoding(accepted.device),
    ).resolves.toBe(F16_PORTABLE_BITCAST_LOAD_ENCODING);
    await expect(
      selectF16PortableLoadEncoding(warningOnly.device),
    ).resolves.toBe(F16_PORTABLE_BITCAST_LOAD_ENCODING);
    await expect(
      selectF16PortableLoadEncoding(rejected.device),
    ).resolves.toBe(F16_PORTABLE_NATIVE_LOAD_ENCODING);

    for (const captured of [accepted, warningOnly, rejected]) {
      expect(captured.createShaderModule).toHaveBeenCalledOnce();
      expect(
        captured.createShaderModule.mock.calls[0]![0],
      ).toMatchObject({
        label: "parakeet-fp16-portable-bitcast-load-probe",
      });
      expect(
        captured.createShaderModule.mock.calls[0]![0].code,
      ).toContain("bitcast<vec4<f16>>(packed)");
    }
  });

  it("surfaces probe API and device failures instead of treating them as compiler capability", async () => {
    const failure = new Error("device lost while reading compilation info");
    const createShaderModule = vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => {
        throw failure;
      }),
    }));
    const device = { createShaderModule } as unknown as GPUDevice;

    await expect(
      selectF16PortableLoadEncoding(device),
    ).rejects.toBe(failure);
    expect(createShaderModule).toHaveBeenCalledOnce();
  });
});

describe("retained subgroup GEMM boundary", () => {
  it("pins the existing FP16 and FP32 subgroup shader bytes", () => {
    expect(subgroupBundleHash("fp16")).toBe(
      "7184fa7840f4a8934a97092d4b52bdb97fbd7823ed72f98027dd611fc317f491",
    );
    expect(subgroupBundleHash("fp32")).toBe(
      "546393172678f0c1352012fe1fe3ef2cc322d6aefbbcb1a05432c9bf31b1a949",
    );
  });

  it("pins the unchanged subgroup geometry independently of portable tuning", () => {
    expect({
      columns: F16_SUBGROUP_TILE_COLUMNS,
      inner: F16_SUBGROUP_TILE_INNER,
      directRows: F16_SUBGROUP_DIRECT_B_TILE_ROWS,
      directWorkgroup: F16_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
      stagedRows: F16_SUBGROUP_STAGED_B_TILE_ROWS,
      stagedWorkgroup: F16_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
      stagedBytes: F16_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
    }).toEqual({
      columns: 256,
      inner: 32,
      directRows: 32,
      directWorkgroup: 128,
      stagedRows: 48,
      stagedWorkgroup: 192,
      stagedBytes: 16_384,
    });
    expect({
      columns: F32_SUBGROUP_TILE_COLUMNS,
      inner: F32_SUBGROUP_TILE_INNER,
      directRows: F32_SUBGROUP_DIRECT_B_TILE_ROWS,
      directWorkgroup: F32_SUBGROUP_DIRECT_B_WORKGROUP_SIZE,
      stagedRows: F32_SUBGROUP_STAGED_B_TILE_ROWS,
      stagedWorkgroup: F32_SUBGROUP_STAGED_B_WORKGROUP_SIZE,
      stagedBytes: F32_SUBGROUP_STAGED_B_WORKGROUP_BYTES,
    }).toEqual({
      columns: 128,
      inner: 32,
      directRows: 32,
      directWorkgroup: 128,
      stagedRows: 48,
      stagedWorkgroup: 192,
      stagedBytes: 16_384,
    });
  });
});

function f16PortableSource(
  label: (typeof F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS)[number]["label"],
): string {
  const specialization = F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.find(
    (candidate) => candidate.label === label,
  );
  if (specialization === undefined) {
    throw new Error(`Missing FP16 specialization ${label}`);
  }
  return f16PortableGemmWgsl(specialization);
}

function f32PortableSource(
  label: (typeof F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS)[number]["label"],
): string {
  const specialization = F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.find(
    (candidate) => candidate.label === label,
  );
  if (specialization === undefined) {
    throw new Error(`Missing FP32 specialization ${label}`);
  }
  return f32PortableGemmWgsl(specialization);
}

function componentOrder(source: string): number[] {
  return Array.from(
    source.matchAll(/\(k_vector \* 4u \+ ([0-3])u\)/g),
    (match) => Number(match[1]),
  );
}

function probeDevice(
  messages: readonly { readonly type: "error" | "warning" }[],
): {
  readonly device: GPUDevice;
  readonly createShaderModule: ReturnType<typeof vi.fn>;
} {
  const createShaderModule = vi.fn(
    (_descriptor: GPUShaderModuleDescriptor) => ({
      getCompilationInfo: vi.fn(async () => ({ messages })),
    }),
  );
  return {
    device: { createShaderModule } as unknown as GPUDevice,
    createShaderModule,
  };
}

function subgroupBundleHash(precision: "fp16" | "fp32"): string {
  const source =
    precision === "fp16"
      ? F16_SUBGROUP_PRODUCTION_SPECIALIZATIONS.map((specialization) =>
          f16SubgroupGemmWgsl(specialization),
        ).join("\u0000")
      : F32_SUBGROUP_PRODUCTION_SPECIALIZATIONS.map((specialization) =>
          f32SubgroupGemmWgsl(specialization),
        ).join("\u0000");
  return createHash("sha256").update(source).digest("hex");
}
