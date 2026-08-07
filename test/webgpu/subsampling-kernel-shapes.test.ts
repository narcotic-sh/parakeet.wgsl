import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  depthwiseSubsamplingWgslForProfile,
  fusedSubsamplingWorkgroupStorageBytes,
  fusedSubsamplingWgslForProfile,
  planDepthwiseSubsampling,
  planFusedSubsamplingConv0Depthwise2,
  SUBSAMPLING_CONV0_WEIGHT_LAYOUT,
  SUBSAMPLING_DEPTHWISE_WEIGHT_LAYOUT,
  subsamplingTimeMaskWgslForProfile,
  subsamplingOutputDimension,
  subsamplingValidLength,
} from "../../src/webgpu/kernels/subsampling";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";

const kernelSource = readFileSync(
  new URL("../../src/webgpu/kernels/subsampling.ts", import.meta.url),
  "utf8",
);

describe("Parakeet subsampling convolution plans", () => {
  it("reduces a 1501x128 fbank to 188x16 across three stride-2 stages", () => {
    const heights = [1501];
    const widths = [128];
    for (let index = 0; index < 3; index++) {
      heights.push(subsamplingOutputDimension(heights.at(-1)!));
      widths.push(subsamplingOutputDimension(widths.at(-1)!));
    }
    expect(heights).toEqual([1501, 751, 376, 188]);
    expect(widths).toEqual([128, 64, 32, 16]);
  });

  it("restores the model mask boundary after every stride-2 Conv2d", () => {
    expect([
      subsamplingValidLength(1500, 1),
      subsamplingValidLength(1500, 2),
      subsamplingValidLength(1500, 3),
    ]).toEqual([750, 375, 188]);
    expect([
      subsamplingValidLength(799, 1),
      subsamplingValidLength(799, 2),
      subsamplingValidLength(799, 3),
    ]).toEqual([400, 200, 100]);
    expect(subsamplingValidLength(1, 3)).toBe(1);
  });

  it("plans the retained depthwise5 stage with channel-last f16 input", () => {
    const plan = planDepthwiseSubsampling({
      batchSize: 1,
      inputHeight: 376,
      inputWidth: 32,
      channels: 256,
    });
    expect(plan.outputHeight).toBe(188);
    expect(plan.outputWidth).toBe(16);
    expect(plan.inputByteLength).toBe(376 * 32 * 256 * 2);
    expect(plan.outputByteLength).toBe(188 * 16 * 256 * 2);
    expect(plan.workgroups).toEqual([2, 94, 4]);
  });

  it("plans four-byte fp32 depthwise inputs, outputs, and weights", () => {
    const plan = planDepthwiseSubsampling(
      {
        batchSize: 1,
        inputHeight: 376,
        inputWidth: 32,
        channels: 256,
      },
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(plan.inputByteLength).toBe(376 * 32 * 256 * 4);
    expect(plan.outputByteLength).toBe(188 * 16 * 256 * 4);
    expect(plan.weightByteLength).toBe(3 * 3 * 256 * 4);
    expect(plan.biasByteLength).toBe(256 * 4);
    expect(plan.workgroups).toEqual([2, 94, 4]);
  });

  it("plans the fixed fused conv0-depthwise2 tile for B3", () => {
    const plan = planFusedSubsamplingConv0Depthwise2({
      batchSize: 3,
      inputHeight: 1501,
      inputWidth: 128,
      channels: 256,
    });
    expect(plan.conv0Height).toBe(751);
    expect(plan.conv0Width).toBe(64);
    expect(plan.outputHeight).toBe(376);
    expect(plan.outputWidth).toBe(32);
    expect(plan.channelVectors).toBe(64);
    expect(plan.inputByteLength).toBe(3 * 1501 * 128 * 4);
    expect(plan.outputByteLength).toBe(3 * 376 * 32 * 256 * 2);
    expect(plan.weightByteLength).toBe(3 * 3 * 256 * 2);
    expect(plan.biasByteLength).toBe(256 * 4);
    expect(plan.workgroups).toEqual([4, 188, 12]);
    expect(plan.workgroupStorageBytes).toBe(10_888);
    expect(() =>
      planFusedSubsamplingConv0Depthwise2({
        batchSize: 3,
        inputHeight: 1500,
        inputWidth: 128,
        channels: 256,
      }),
    ).toThrow(/fixed/);
  });

  it("keeps fp32 fused subsampling inside stock workgroup storage", () => {
    const plan = planFusedSubsamplingConv0Depthwise2(
      {
        batchSize: 3,
        inputHeight: 1501,
        inputWidth: 128,
        channels: 256,
      },
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(plan.inputByteLength).toBe(3 * 1501 * 128 * 4);
    expect(plan.outputByteLength).toBe(3 * 376 * 32 * 256 * 4);
    expect(plan.weightByteLength).toBe(3 * 3 * 256 * 4);
    expect(plan.workgroups).toEqual([4, 188, 24]);
    expect(plan.workgroupStorageBytes).toBe(10_888);
    expect(
      fusedSubsamplingWorkgroupStorageBytes(
        PARAKEET_FP32_EXECUTION_PROFILE,
      ),
    ).toBeLessThanOrEqual(16 * 1024);
  });

  it("rounds the fused conv0 result through the production f16 boundary", () => {
    expect(kernelSource).toContain(
      "export class FusedSubsamplingConv0Depthwise2Kernel",
    );
    expect(kernelSource).toContain(
      "var<workgroup> conv0_tile: array<vec4<f16>",
    );
    expect(kernelSource).toContain(
      "rounded = vec4<f16>(max(sum, vec4<f32>(0.0)));",
    );
    expect(kernelSource).toContain(
      "vec4<f32>(conv0_tile[tile_index])",
    );
    expect(kernelSource.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(kernelSource).not.toContain("export class SubsamplingConv0Kernel");
    expect(kernelSource).not.toContain("planSubsamplingConv0");
    expect(kernelSource).not.toContain("const CONV0_WGSL");
    expect(kernelSource).not.toContain("subgroup_matrix");
    expect(kernelSource).not.toContain("chromium_experimental");
  });

  it("provides f16-free fp32 shaders with four-byte storage", () => {
    const fp16 = fusedSubsamplingWgslForProfile(
      PARAKEET_FP16_EXECUTION_PROFILE,
    );
    const fusedFp32 = fusedSubsamplingWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    const depthwiseFp32 = depthwiseSubsamplingWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    const maskFp32 = subsamplingTimeMaskWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(fp16).toContain("enable f16;");
    for (const shader of [fusedFp32, depthwiseFp32, maskFp32]) {
      expect(shader).not.toContain("f16");
      expect(shader).not.toContain("0.0h");
    }
    expect(fusedFp32).toContain("@workgroup_size(8, 2, 8)");
    expect(fusedFp32).toContain(
      "var<workgroup> conv0_tile: array<vec4<f32>, 680>;",
    );
    expect(fusedFp32).toContain(
      "let channel_in_group = tile_index % 8u;",
    );
    expect(fusedFp32).toContain(
      "let spatial_index = tile_index / 8u;",
    );
    expect(fusedFp32).not.toContain(
      "let spatial_index = tile_index / 16u;",
    );
    expect(fusedFp32).toContain("conv0_weights: array<vec4<f32>>");
    expect(fusedFp32).toContain("depthwise_weights: array<vec4<f32>>");
    expect(fusedFp32).toContain("output: array<vec4<f32>>");
    expect(depthwiseFp32).toContain("input: array<vec4<f32>>");
    expect(depthwiseFp32).toContain("weights: array<vec4<f32>>");
    expect(depthwiseFp32).toContain("output: array<vec4<f32>>");
    expect(maskFp32).toContain("values: array<vec4<f32>>");
  });

  it("publishes the exact channel-vector packing contracts", () => {
    expect(SUBSAMPLING_CONV0_WEIGHT_LAYOUT).toBe(
      "kh-kw-output-channel-vec4",
    );
    expect(SUBSAMPLING_DEPTHWISE_WEIGHT_LAYOUT).toBe(
      "kh-kw-channel-vec4",
    );
  });

  it("rejects a channel count that cannot use vec4 loads", () => {
    expect(() =>
      planDepthwiseSubsampling({
        batchSize: 1,
        inputHeight: 16,
        inputWidth: 16,
        channels: 255,
      }),
    ).toThrow(/divisible by four/);
  });
});
