import { describe, expect, it } from "vitest";

import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";
import { layerNormWgslForProfile } from "../../src/webgpu/kernels/layernorm";

describe("profile-specific LayerNorm shaders", () => {
  it("preserves the fp16 shader and emits an f16-free fp32 variant", () => {
    const fp16 = layerNormWgslForProfile(
      PARAKEET_FP16_EXECUTION_PROFILE,
    );
    const fp32 = layerNormWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );

    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("input: array<vec4<f16>>");
    expect(fp16).toContain("output: array<vec4<f16>>");
    expect(fp32).not.toContain("f16");
    expect(fp32).toContain("enable subgroups;");
    expect(fp32).toContain("input: array<vec4<f32>>");
    expect(fp32).toContain("output: array<vec4<f32>>");
    expect(fp32).toContain("var sum = 0.0;");
    expect(fp32).toContain("let local_sum = subgroupAdd(sum);");
    expect(fp32).toContain("gamma: array<vec4<f32>>");
    expect(fp32).toContain("beta: array<vec4<f32>>");
  });

  it("emits fixed-order workgroup reductions for portable profiles", () => {
    const fp16 = layerNormWgslForProfile(
      PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
    );
    const fp32 = layerNormWgslForProfile(
      PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
    );

    for (const portable of [fp16, fp32]) {
      expect(portable).not.toContain("subgroup");
      expect(portable).toContain(
        "var<workgroup> moments: array<vec2<f32>, 256>;",
      );
      expect(portable).toContain("let virtual_lane = lane & 31u;");
      expect(portable).toContain(
        "for (var offset = 1u; offset < 32u; offset *= 2u)",
      );
      expect(portable).toContain(
        "if ((virtual_lane & (offset * 2u - 1u)) == 0u)",
      );
      expect(portable).toContain(
        "totals += moments[virtual_group * 32u];",
      );
      expect(portable).toContain(
        "let value = vec4<f32>(input[row * vectors + vector]);",
      );
    }
    expect(fp16).toContain("enable f16;");
    expect(fp32).not.toContain("f16");
  });
});
