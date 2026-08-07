import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";
import {
  relativeAttentionWgslForProfile,
  relativeAttentionWorkgroupStorageBytes,
} from "../../src/webgpu/kernels/attention";

const source = readFileSync(
  new URL("../../src/webgpu/kernels/attention.ts", import.meta.url),
  "utf8",
);

describe("fixed stock relative-attention schedule", () => {
  it("uses six ordinary 32-lane subgroups with 16-key value tiles", () => {
    expect(source).toContain("const WORKGROUP_SIZE = 192;");
    expect(source).toContain("const SUBGROUP_SIZE = 32;");
    expect(source).toContain("const QUERIES_PER_WORKGROUP = 6;");
    expect(source).toContain("const VALUE_TILE_KEYS = 16;");
    expect(source).toContain("enable subgroups;");
    expect(source).not.toContain("subgroup_matrix");
    expect(source).not.toContain("chromium_experimental");
  });

  it("reads operation-native position rows contiguously across key lanes", () => {
    expect(source).toContain('record.layout !== "head-vector-position-vec4"');
    expect(source).toContain(
      "(head_vector_offset + vector) * params.position_rows",
    );
    expect(source).toContain("+ relative_position");
    expect(source).not.toContain(
      "relative_position * ${HIDDEN_VECTORS}u",
    );
  });

  it("provides an f16-free fp32 shader under stock storage limits", () => {
    const fp16 = relativeAttentionWgslForProfile(
      PARAKEET_FP16_EXECUTION_PROFILE,
    );
    const fp32 = relativeAttentionWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("value_tile: array<\n  vec4<f16>,\n  512");
    expect(fp32).not.toContain("f16");
    expect(fp32).not.toContain("0.0h");
    expect(fp32).toContain("qkv: array<vec4<f32>>");
    expect(fp32).toContain("positions: array<vec4<f32>>");
    expect(fp32).toContain("context: array<vec4<f32>>");
    expect(fp32).toContain("value_tile: array<\n  vec4<f32>,\n  256");
    expect(fp32).toContain("tile_start += 8u");
    expect(fp32).toContain("let tile_keys = min(8u, valid - tile_start);");
    expect(
      relativeAttentionWorkgroupStorageBytes(
        PARAKEET_FP32_EXECUTION_PROFILE,
      ),
    ).toBeLessThanOrEqual(16 * 1024);
  });

  it("provides standards-only portable shaders without changing Q6 geometry", () => {
    const fp16 = relativeAttentionWgslForProfile(
      PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
    );
    const fp32 = relativeAttentionWgslForProfile(
      PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
    );

    for (const portable of [fp16, fp32]) {
      expect(portable).not.toContain("subgroup");
      expect(portable).toContain("query_u[lane] = vec4<f32>(");
      expect(portable).toContain("let virtual_lane = lane & 31u;");
      expect(portable).toContain(
        "for (var offset = 1u; offset < 32u; offset *= 2u)",
      );
      expect(portable).toContain(
        "if ((virtual_lane & (offset * 2u - 1u)) == 0u)",
      );
      expect(portable).toContain("virtual_group < 6u;");
      expect(portable).toContain("query_u[lane] += query_u[lane + offset];");
      expect(portable).toContain("workgroupUniformLoad(&reduction_a)");
      expect(portable).toContain("@workgroup_size(192, 1, 1)");
    }
    expect(fp16).toContain("enable f16;");
    expect(fp32).not.toContain("f16");
  });
});
