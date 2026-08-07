import { describe, expect, it } from "vitest";

import {
  CONFORMER_CONV_CHANNELS,
  CONFORMER_CONV_KERNEL_SIZE,
  CONFORMER_CONV_POINTWISE_CHANNELS,
  CONFORMER_DEPTHWISE_WEIGHT_LAYOUT,
  CONFORMER_FOLDED_BATCH_NORM_LAYOUT,
  conformerConvWgslForProfile,
  planFusedConformerConv,
} from "../../src/webgpu/kernels/conformer-conv";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";

describe("fused Parakeet Conformer convolution plan", () => {
  it("plans batched 188-frame encoder windows", () => {
    const plan = planFusedConformerConv({
      batchSize: 2,
      frames: 188,
    });
    expect(plan.rows).toBe(376);
    expect(plan.inputByteLength).toBe(
      376 * CONFORMER_CONV_POINTWISE_CHANNELS * 2,
    );
    expect(plan.outputByteLength).toBe(376 * CONFORMER_CONV_CHANNELS * 2);
    expect(plan.validLengthsByteLength).toBe(2 * 4);
    expect(plan.depthwiseWeightByteLength).toBe(
      CONFORMER_CONV_KERNEL_SIZE * CONFORMER_CONV_CHANNELS * 2,
    );
    expect(plan.affineByteLength).toBe(CONFORMER_CONV_CHANNELS * 4);
    expect(plan.workgroups).toEqual([32, 12, 2]);
  });

  it("uses a bounded 3 KiB shared GLU tile", () => {
    const plan = planFusedConformerConv({ batchSize: 1, frames: 1 });
    expect(plan.workgroupStorageBytes).toBe(24 * 8 * 4 * 4);
    expect(plan.workgroupStorageBytes).toBe(3072);
  });

  it("plans four-byte activations and depthwise weights for fp32", () => {
    const plan = planFusedConformerConv(
      { batchSize: 2, frames: 188 },
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(plan.inputByteLength).toBe(
      376 * CONFORMER_CONV_POINTWISE_CHANNELS * 4,
    );
    expect(plan.outputByteLength).toBe(
      376 * CONFORMER_CONV_CHANNELS * 4,
    );
    expect(plan.depthwiseWeightByteLength).toBe(
      CONFORMER_CONV_KERNEL_SIZE * CONFORMER_CONV_CHANNELS * 4,
    );
    expect(plan.affineByteLength).toBe(CONFORMER_CONV_CHANNELS * 4);
    expect(plan.workgroups).toEqual([32, 12, 2]);
    expect(plan.workgroupStorageBytes).toBe(3072);
  });

  it("keeps the fp16 shader and provides an f16-free fp32 shader", () => {
    const fp16 = conformerConvWgslForProfile(
      PARAKEET_FP16_EXECUTION_PROFILE,
    );
    const fp32 = conformerConvWgslForProfile(
      PARAKEET_FP32_EXECUTION_PROFILE,
    );
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("pointwise_input: array<vec4<f16>>");
    expect(fp32).not.toContain("f16");
    expect(fp32).toContain("pointwise_input: array<vec4<f32>>");
    expect(fp32).toContain("depthwise_weights: array<vec4<f32>>");
    expect(fp32).toContain("output: array<vec4<f32>>");
    expect(fp32).toContain("var<workgroup> glu_tile: array<vec4<f32>");
  });

  it("publishes exact depthwise and folded-BN packing contracts", () => {
    expect(CONFORMER_DEPTHWISE_WEIGHT_LAYOUT).toBe(
      "kernel-channel-vec4",
    );
    expect(CONFORMER_FOLDED_BATCH_NORM_LAYOUT).toBe("channel");
  });

  it("rejects invalid batch and frame dimensions", () => {
    expect(() => planFusedConformerConv({ batchSize: 0, frames: 188 })).toThrow(
      /batchSize/,
    );
    expect(() => planFusedConformerConv({ batchSize: 1, frames: 0 })).toThrow(
      /frames/,
    );
  });
});
