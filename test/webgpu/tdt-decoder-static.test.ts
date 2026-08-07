import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  GpuTensor,
  ParakeetGpuPackage,
} from "../../src/model/package";
import type { ArenaSlice } from "../../src/webgpu/arena";
import {
  PersistentTdtDecoderKernel,
  planTdtDecoderExecution,
  TDT_PAIRED_WORKGROUP_STORAGE_BYTES,
  TDT_PORTABLE_PAIRED_WORKGROUP_STORAGE_BYTES,
  tdtDecoderFp32Wgsl,
  tdtDecoderWgsl,
  tdtDecoderWgslForProfile,
} from "../../src/webgpu/kernels/tdt-decoder";
import {
  PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";

const dispatchSource = readFileSync(
  new URL("../../src/webgpu/kernels/tdt-decoder.ts", import.meta.url),
  "utf8",
);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("persistent TDT decoder arithmetic", () => {
  it("pins the FP16 decoder with per-window acoustic-context starts", () => {
    expect(
      createHash("sha256").update(tdtDecoderWgsl()).digest("hex"),
    ).toBe(
      "a0a7ed32757b02876ac639990274a71f49ff6d18b2e8667ff871d721a5c6a431",
    );
    expect(tdtDecoderWgsl()).toContain(
      "min(arena[params.decode_starts_offset + sequence_a], encoded_length_a)",
    );
    expect(tdtDecoderWgsl()).toContain(
      "arena[params.decode_ends_offset + sequence_a]",
    );
  });

  it("pins the v2 decoder dimensions in both precision shaders", () => {
    expect(dispatchSource).toContain("const TOKEN_OUTPUTS = 1025;");
    expect(dispatchSource).toContain("const BLANK_TOKEN = 1024;");
    expect(dispatchSource).toContain("const HEAD_STORAGE_OUTPUTS = 1032;");
    expect(dispatchSource).not.toContain("const TOKEN_OUTPUTS = 8193;");
    expect(dispatchSource).not.toContain("const BLANK_TOKEN = 8192;");
    expect(dispatchSource).not.toContain("const HEAD_STORAGE_OUTPUTS = 8200;");

    for (const source of [tdtDecoderWgsl(), tdtDecoderFp32Wgsl()]) {
      expect(source).toContain("k * 1032u +");
      expect(source).toContain("1024u");
      expect(source).not.toContain("k * 8200u +");
      expect(source).not.toContain("8192u");
    }
    expect(tdtDecoderWgsl()).toContain("let head_bias0 =");
    expect(tdtDecoderWgsl()).not.toContain("let head_bias1 =");
  });

  it("preserves f32 recurrent state in the reference GPU shader", () => {
    const source = tdtDecoderWgsl();

    expect(source).toContain("let stable_cell = cell;");
    expect(source).toContain("let stable_hidden = hidden;");
    expect(source).not.toContain(
      "let stable_cell = vec4<f32>(vec4<f16>(cell));",
    );
  });

  it("generates a shader-f16-free FP32 decoder with raw FP32 IO", () => {
    const source = tdtDecoderFp32Wgsl();

    expect(createHash("sha256").update(source).digest("hex")).toBe(
      "0739cd750d79b579075319ccec1bd636acf555e9f6d7f4a58aef4d71e0ff780d",
    );
    expect(source).toContain("enable subgroups;");
    expect(source).not.toContain("enable f16;");
    expect(source).not.toContain("f16");
    expect(source).not.toContain("unpack2x16float");
    expect(source).not.toContain("weight_pairs");
    expect(source).toContain(
      "weight_vectors: array<vec4<f32>>;",
    );
    expect(source).toContain(
      "return bitcast<f32>(weights[base_word + element]);",
    );
    expect(source).toContain(
      "return weight_vectors[(base_word + element) / 4u];",
    );
    expect(source).toContain(
      "return bitcast<f32>(arena[base_word + element]);",
    );
    expect(source).toContain(
      "var layer0_gate0 = read_weight_f32x4(",
    );
    expect(source).toContain(
      "layer0_gate0 = fma(vec4<f32>(input_even), read_weight_vec4(",
    );
    expect(source).toContain(
      "var layer1_gate0 = read_weight_f32x4(",
    );
    expect(source).toContain(
      "let head_bias0 = read_weight_f32x4(",
    );
    expect(source).toContain(
      "head_a0 = fma(\n            vec4<f32>(input_a),",
    );
    expect(source).toContain(
      "tail_a = fma(vec4<f32>(input_a), tail_weight, tail_a);",
    );
    expect(source).toContain(
      "var tail_a = vec4<f32>(-3.402823466e+38);",
    );
  });

  it("emits standards-only portable reductions for both precisions", () => {
    const fp16 = tdtDecoderWgslForProfile(
      PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
    );
    const fp32 = tdtDecoderWgslForProfile(
      PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
    );

    for (const portable of [fp16, fp32]) {
      expect(portable).not.toContain("subgroup");
      expect(portable).toContain(
        "var<workgroup> token_values: array<f32, 256>;",
      );
      expect(portable).toContain(
        "if (lane < 8u) {\n    let base = lane * 32u;",
      );
      expect(portable).toContain(
        "for (var member = 1u; member < 32u; member += 1u)",
      );
      expect(portable).toContain("virtual_group < 8u;");
      expect(portable).toContain(
        "right_value == left_value && right_id < left_id",
      );
    }
    expect(fp16).toContain("enable f16;");
    expect(fp32).not.toContain("f16");
  });

  it("creates and binds the FP32 decoder without shader-f16", async () => {
    vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
    vi.stubGlobal("GPUBufferUsage", {
      UNIFORM: 1,
      COPY_DST: 2,
    });
    const decoderBuffer = {
      size: 33_558_528,
    } as GPUBuffer;
    const arenaBuffer = {
      size: 1_048_576,
    } as GPUBuffer;
    const uniformBuffer = {
      size: 128,
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    let shaderCode = "";
    let bindGroupDescriptor: GPUBindGroupDescriptor | undefined;
    let uniformWords: Uint32Array | undefined;
    const device = {
      features: new Set(["subgroups"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupStorageSize: 32 * 1024,
      },
      queue: {
        writeBuffer(
          _buffer: GPUBuffer,
          _offset: number,
          data: Uint32Array,
        ) {
          uniformWords = new Uint32Array(data);
        },
      },
      createBindGroupLayout() {
        return {} as GPUBindGroupLayout;
      },
      createShaderModule(descriptor: GPUShaderModuleDescriptor) {
        shaderCode = descriptor.code;
        return {
          async getCompilationInfo() {
            return { messages: [] };
          },
        } as unknown as GPUShaderModule;
      },
      createPipelineLayout() {
        return {} as GPUPipelineLayout;
      },
      async createComputePipelineAsync() {
        return {} as GPUComputePipeline;
      },
      createBuffer() {
        return uniformBuffer;
      },
      createBindGroup(descriptor: GPUBindGroupDescriptor) {
        bindGroupDescriptor = descriptor;
        return {} as GPUBindGroup;
      },
    } as unknown as GPUDevice;
    const bufferFor = vi.fn((slice: ArenaSlice) => {
      expect(slice.bufferIndex).toBe(1);
      return arenaBuffer;
    });
    const arena = {
      bufferFor,
      bufferByteLengthFor(slice: ArenaSlice) {
        expect(slice.bufferIndex).toBe(1);
        return arenaBuffer.size;
      },
    };
    await expect(
      PersistentTdtDecoderKernel.create(
        device,
        arena as never,
        fp32DecoderModel(decoderBuffer),
      ),
    ).rejects.toThrow("requires WebGPU shader-f16");
    const kernel = await PersistentTdtDecoderKernel.create(
      device,
      arena as never,
      fp32DecoderModel(decoderBuffer),
      PARAKEET_FP32_EXECUTION_PROFILE,
    );

    expect(shaderCode).toBe(tdtDecoderFp32Wgsl());
    expect(shaderCode).not.toContain("f16");

    const descriptor = fp32DecoderDescriptor();
    kernel.createDispatch(descriptor);
    expect(bufferFor).toHaveBeenCalledTimes(11);
    const arenaEntry = Array.from(
      bindGroupDescriptor?.entries ?? [],
    ).find((entry) => entry.binding === 1);
    expect(arenaEntry?.resource).toEqual({
      buffer: arenaBuffer,
      offset: 0,
      size: arenaBuffer.size,
    });
    expect(uniformWords?.[3]).toBe(
      descriptor.encoderProjected.byteOffset / 4,
    );
    expect(uniformWords?.[4]).toBe(
      descriptor.encodedLengths.byteOffset / 4,
    );
    expect(uniformWords?.[5]).toBe(
      descriptor.state.byteOffset / 4,
    );
    expect(uniformWords?.[6]).toBe(
      descriptor.tokens.byteOffset / 4,
    );
    expect(uniformWords?.[7]).toBe(
      descriptor.tokenFrames.byteOffset / 4,
    );
    expect(uniformWords?.[8]).toBe(
      descriptor.tokenDurations.byteOffset / 4,
    );
    expect(uniformWords?.[9]).toBe(
      descriptor.metadata.byteOffset / 4,
    );
    expect(uniformWords?.[10]).toBe(
      descriptor.metadata.byteOffset / 4 + descriptor.batchSize,
    );
    expect(uniformWords?.[20]).toBe(
      descriptor.predictorScratch.byteOffset / 4,
    );
    expect(uniformWords?.[21]).toBe(
      descriptor.decodeStarts.byteOffset / 4,
    );
    expect(uniformWords?.[22]).toBe(
      descriptor.decodeEnds.byteOffset / 4,
    );
    expect(uniformWords?.[23]).toBe(
      descriptor.flushFinals.byteOffset / 4,
    );
  });

  it("pairs arbitrary batch tails within stock workgroup storage", () => {
    expect(
      [1, 2, 3, 39, 40].map(
        (batchSize) => planTdtDecoderExecution(batchSize).workgroupCount,
      ),
    ).toEqual([1, 1, 2, 20, 20]);
    expect(planTdtDecoderExecution(40).predictorScratchByteLength).toBe(
      512_000,
    );
    expect(TDT_PAIRED_WORKGROUP_STORAGE_BYTES).toBe(25_808);
    expect(TDT_PAIRED_WORKGROUP_STORAGE_BYTES).toBeLessThan(32 * 1024);
    expect(TDT_PORTABLE_PAIRED_WORKGROUP_STORAGE_BYTES).toBe(29_776);
    expect(TDT_PORTABLE_PAIRED_WORKGROUP_STORAGE_BYTES).toBeLessThan(
      32 * 1024,
    );
    expect(dispatchSource).toContain(
      "planTdtDecoderExecution(activeBatchSize).workgroupCount",
    );
    expect(dispatchSource).toContain(
      "this.device.queue.writeBuffer(",
    );
  });

  it("keeps final-tail continuation bounded and separates physical from effective frames", () => {
    const source = tdtDecoderWgsl();

    expect(source).toContain(
      "workgroupUniformLoad(&symbols_at_time_a) < 10u",
    );
    expect(source).toContain(
      "workgroupUniformLoad(&flush_blanks_a) < 5u",
    );
    expect(source).toContain(
      "arena[params.lengths_offset + sequence_a],\n                params.frames",
    );
    expect(source).toContain(
      "min(encoded_length_a - 1u, params.frames - 1u)",
    );
    expect(source).toContain(
      "max(encoded_length_a, 2u) - 2u",
    );
    expect(source).toContain(
      "min(flush_time_a, encoded_length_a - 1u)",
    );
    expect(source).toContain(
      "flush_time_a + max(1u, best_duration_a)",
    );
    expect(source).toContain(
      "arena[params.token_durations_offset + output] = best_duration_a;",
    );

    const probeFrames = (
      processingTime: number,
      physicalLength: number,
      effectiveLength: number,
    ) => [
      Math.min(processingTime, physicalLength - 1),
      Math.min(effectiveLength - 1, physicalLength - 1),
      Math.min(Math.max(effectiveLength, 2) - 2, physicalLength - 1),
    ];
    expect(probeFrames(186, 187, 186)).toEqual([186, 185, 184]);
    expect(probeFrames(187, 187, 187)).toEqual([186, 186, 185]);
  });

  it("binds the resident decoder source shard with runtime word offsets", () => {
    expect(dispatchSource).toContain(
      "tensor.runtimeRecord.byteOffset / 4",
    );
    expect(dispatchSource).toContain(
      'tensor.runtimeRecord.bufferId !== "source-shard-decoder.bin"',
    );
    expect(dispatchSource).toContain(
      "tensor.sourceRecord.shard !== \"decoder.bin\"",
    );
    expect(dispatchSource).toContain(
      "buffer: this.decoderWeights",
    );
    expect(dispatchSource).not.toContain("model.shard(");
    expect(dispatchSource).not.toContain(
      "tensor.sourceRecord.byteOffset / 4",
    );
  });

  it("selects profile-matched weights and a partition-local arena binding", () => {
    expect(dispatchSource).toContain(
      "executionProfile: ParakeetExecutionProfile =\n      PARAKEET_FP16_EXECUTION_PROFILE",
    );
    expect(dispatchSource).toContain(
      'executionProfile.precision === "fp16" &&',
    );
    expect(dispatchSource).toContain(
      "model.precision !== executionProfile.precision",
    );
    expect(dispatchSource).toContain(
      'executionProfile.precision === "fp16"\n      ? "float16"\n      : "float32"',
    );
    expect(dispatchSource).toContain(
      "code: tdtDecoderWgslForProfile(executionProfile)",
    );
    expect(dispatchSource).toContain(
      "const buffer = arena.bufferFor(slices[0]);",
    );
    expect(dispatchSource).toContain(
      "arena.bufferFor(slice) !== buffer",
    );
    expect(dispatchSource).toContain(
      "arena.bufferByteLengthFor(slices[0])",
    );
    for (const field of [
      "encoderProjected",
      "encodedLengths",
      "decodeStarts",
      "decodeEnds",
      "flushFinals",
      "state",
      "predictorScratch",
      "tokens",
      "tokenFrames",
      "tokenDurations",
      "metadata",
    ]) {
      expect(dispatchSource).toContain(`descriptor.${field},`);
    }
    expect(dispatchSource).toContain(
      '(executionProfile.precision === "fp16" ? 2 : 4)',
    );
  });

  it("persists paired predictor state with storage ordering", () => {
    const source = tdtDecoderWgsl();

    expect(source).toContain("let sequence_a = group.x * 2u;");
    expect(source).toContain(
      "let has_sequence_b = sequence_b < params.batch_size;",
    );
    expect(source).toContain(
      "sequence * 3200u +\n    slot * 640u +\n    channel;",
    );
    expect(source).toContain("predictor_scratch_index(sequence, 4u, output)");
    expect(source).toContain(
      "arena[state_index(sequence, 0u, channel)] =\n      arena[predictor_scratch_index(sequence, 0u, channel)];",
    );
    expect(source.match(/storageBarrier\(\);/g)).toHaveLength(3);
    expect(source).not.toContain("BENCH_SINGLE_SEQUENCE");
  });

  it("skips only exact positive-zero paired-head inputs", () => {
    const source = tdtDecoderWgsl();
    const headStart = source.indexOf("var zero_skip_safe_a = true;");
    const headEnd = source.indexOf(
      "var local_token_value_a =",
      headStart,
    );
    const head = source.slice(headStart, headEnd);

    expect(headStart).toBeGreaterThanOrEqual(0);
    expect(headEnd).toBeGreaterThan(headStart);
    expect(source.match(/var zero_skip_safe_a = true;/g)).toHaveLength(1);
    expect(source.match(/var zero_skip_safe_b = true;/g)).toHaveLength(1);
    expect(head).toContain(
      "(input_a_bits & 0x7fffffffu) > 0x7f800000u",
    );
    expect(head).toContain(
      "(input_b_bits & 0x7fffffffu) > 0x7f800000u",
    );
    expect(head).toContain(
      "input_a_bits == 0x80000000u || input_a_is_nan",
    );
    expect(head).toContain(
      "input_b_bits == 0x80000000u || input_b_is_nan",
    );
    expect(head).toContain(
      "zero_skip_safe_a = zero_skip_safe_a && !exceptional_a;",
    );
    expect(head).toContain(
      "zero_skip_safe_b = zero_skip_safe_b && !exceptional_b;",
    );
    expect(head).toContain(
      "let skip_a = zero_skip_safe_a && input_a_bits == 0u;",
    );
    expect(head).toContain(
      "let skip_b = zero_skip_safe_b && input_b_bits == 0u;",
    );
    expect(head.indexOf("if (!skip_a || !skip_b) {")).toBeLessThan(
      head.indexOf("let head_weight0 = read_weight_vec4_f16("),
    );
    expect(head).toContain("if (!skip_a) {");
    expect(head).toContain("if (!skip_b) {");
    expect(source).toContain(
      "head_a0 = fma(\n          vec4<f16>(f16(input_a)),\n          head_weight0,",
    );
    expect(source).toContain(
      "head_b0 = fma(\n          vec4<f16>(f16(input_b)),\n          head_weight0,",
    );
    expect(head).toContain(
      "tail_a = fma(vec4<f16>(f16(input_a)), tail_weight, tail_a);",
    );
    expect(head).toContain(
      "tail_b = fma(vec4<f16>(f16(input_b)), tail_weight, tail_b);",
    );
    expect(source.indexOf("let head_bias0 =")).toBeGreaterThanOrEqual(0);
    expect(source.indexOf("let head_bias0 =")).toBeLessThan(headStart);
    expect(source).not.toContain("input_a == 0.0");
    expect(source).not.toContain("input_b == 0.0");
    expect(source).not.toContain("input_a != input_a");
    expect(source).not.toContain("input_b != input_b");
  });

  it("preserves the exact positive-zero skip in the FP32 head", () => {
    const source = tdtDecoderFp32Wgsl();
    const blockStart = source.indexOf(
      "// V2's complete 1024-token block and joint tail share one increasing-K pass.",
    );
    const blockEnd = source.indexOf(
      "let blank_score = sanitize_score(f32(tail_a[0u]));",
      blockStart,
    );
    const block = source.slice(blockStart, blockEnd);

    expect(blockStart).toBeGreaterThanOrEqual(0);
    expect(blockEnd).toBeGreaterThan(blockStart);
    expect(source.match(/var zero_skip_safe_a = true;/g)).toHaveLength(1);
    expect(source.match(/var zero_skip_safe_b = true;/g)).toHaveLength(1);
    expect(block).toContain(
      "(input_a_bits & 0x7fffffffu) > 0x7f800000u",
    );
    expect(block).toContain(
      "(input_b_bits & 0x7fffffffu) > 0x7f800000u",
    );
    expect(block).toContain(
      "input_a_bits == 0x80000000u || input_a_is_nan",
    );
    expect(block).toContain(
      "input_b_bits == 0x80000000u || input_b_is_nan",
    );
    expect(block).toContain(
      "zero_skip_safe_a = zero_skip_safe_a && !exceptional_a;",
    );
    expect(block).toContain(
      "zero_skip_safe_b = zero_skip_safe_b && !exceptional_b;",
    );
    expect(block).toContain(
      "let skip_a = zero_skip_safe_a && input_a_bits == 0u;",
    );
    expect(block).toContain(
      "let skip_b = zero_skip_safe_b && input_b_bits == 0u;",
    );
    expect(block).toContain("if (!skip_a || !skip_b) {");
    expect(block).toContain("if (!skip_a) {");
    expect(block).toContain("if (!skip_b) {");
    expect(block).toContain(
      "let head_weight0 = read_weight_vec4(",
    );
    expect(block).not.toContain("head_weight1");
    expect(block).toContain(
      "tail_a = fma(vec4<f32>(input_a), tail_weight, tail_a);",
    );
    expect(block).toContain(
      "tail_b = fma(vec4<f32>(input_b), tail_weight, tail_b);",
    );
    expect(source).not.toContain("input_a == 0.0");
    expect(source).not.toContain("input_b == 0.0");
  });

  it("runs the complete FP32 v2 head in one stable increasing-K pass", () => {
    const source = tdtDecoderFp32Wgsl();
    const jointStart = source.indexOf("fn run_joint_pair(");
    const jointEnd = source.indexOf("\n}\n\n@compute", jointStart);
    const joint = source.slice(jointStart, jointEnd);

    expect(jointStart).toBeGreaterThanOrEqual(0);
    expect(jointEnd).toBeGreaterThan(jointStart);
    expect(
      joint.match(/for \(var k = 0u; k < 640u; k \+= 1u\)/g),
    ).toHaveLength(1);
    expect(
      joint.indexOf("var local_token_value_a = -3.402823466e+38;"),
    ).toBeLessThan(
      joint.indexOf("var tail_a = vec4<f32>"),
    );
    expect(joint.indexOf("var tail_a = vec4<f32>")).toBeLessThan(
      joint.indexOf(
        "// V2's complete 1024-token block and joint tail share one increasing-K pass.",
      ),
    );
    expect(
      joint.match(
        /tail_a = fma\(vec4<f32>\(input_a\), tail_weight, tail_a\);/g,
      ),
    ).toHaveLength(1);
    expect(
      joint.match(
        /tail_b = fma\(vec4<f32>\(input_b\), tail_weight, tail_b\);/g,
      ),
    ).toHaveLength(1);

    expect(joint.match(/let head_bias0 =/g)).toHaveLength(1);
    expect(joint.match(/head_a0 = fma\(/g)).toHaveLength(1);
    expect(joint.match(/head_b0 = fma\(/g)).toHaveLength(1);
    expect(joint).not.toContain("head_bias1");
    expect(joint).not.toMatch(/head_[ab]\d+\.\d+/u);
    expect(joint).not.toMatch(/\d+\.\d+u/u);
    expect(joint).toContain("k * 1032u +");
    expect(joint).toContain("(lane + 256u) * 4u");
    expect(joint).toContain("local_token_id_a = 1024u;");

    for (const prefix of ["a", "b"] as const) {
      let previous = -1;
      for (let component = 0; component < 4; component += 1) {
        const position = joint.indexOf(
          `sanitize_score(f32(head_${prefix}0[${component}u]))`,
        );
        expect(position).toBeGreaterThan(previous);
        previous = position;
      }
      expect(
        joint.indexOf(
          `let blank_score = sanitize_score(f32(tail_${prefix}[0u]));`,
        ),
      ).toBeGreaterThan(previous);
    }
    expect(source).toContain(
      "right_value == left_value && right_id < left_id",
    );
    expect(joint.indexOf("reduce_joint_result(\n    0u,")).toBeLessThan(
      joint.indexOf("reduce_joint_result(\n    1u,"),
    );
  });
});

function fp32DecoderModel(
  buffer: GPUBuffer,
): ParakeetGpuPackage {
  const specs: Readonly<
    Record<
      string,
      {
        readonly shape: readonly number[];
        readonly layout: string;
      }
    >
  > = {
    "decoder.embedding.weight": {
      shape: [1025, 640],
      layout: "token-by-channel-row-major",
    },
    "decoder.lstm.weight_l0": {
      shape: [1280, 2560],
      layout: "input-hidden-k-by-ifgo-row-major-vec4",
    },
    "decoder.lstm.bias_l0": {
      shape: [2560],
      layout: "ifgo-gates",
    },
    "decoder.lstm.weight_l1": {
      shape: [1280, 2560],
      layout: "input-hidden-k-by-ifgo-row-major-vec4",
    },
    "decoder.lstm.bias_l1": {
      shape: [2560],
      layout: "ifgo-gates",
    },
    "decoder.decoder_projector.weight": {
      shape: [640, 640],
      layout: "k-by-output-row-major-vec4",
    },
    "decoder.decoder_projector.bias": {
      shape: [640],
      layout: "output-channel",
    },
    "joint.head.weight": {
      shape: [640, 1032],
      layout: "k-by-output-row-major-vec4",
    },
    "joint.head.bias": {
      shape: [1032],
      layout: "output-channel-padded-vec4",
    },
  };
  const tensors = new Map<string, GpuTensor>(
    Object.entries(specs).map(([name, spec], index) => {
      const byteOffset = index * 256;
      const sourceRecord = {
        shard: "decoder.bin",
        byteOffset,
        byteLength: 256,
        dtype: "float32" as const,
        logicalShape: spec.shape,
        storageShape: spec.shape,
        layout: spec.layout,
      };
      return [
        name,
        {
          sourceRecord,
          runtimeRecord: {
            bufferId: "source-shard-decoder.bin" as const,
            byteOffset,
            byteLength: sourceRecord.byteLength,
            dtype: sourceRecord.dtype,
            logicalShape: spec.shape,
            storageShape: spec.shape,
            layout: spec.layout,
          },
          buffer,
          binding: {
            buffer,
            offset: byteOffset,
            size: sourceRecord.byteLength,
          },
        },
      ];
    }),
  );
  return {
    precision: "fp32",
    tensor(name: string) {
      const tensor = tensors.get(name);
      if (tensor === undefined) {
        throw new Error(`Unexpected test tensor ${name}`);
      }
      return tensor;
    },
  } as unknown as ParakeetGpuPackage;
}

function fp32DecoderDescriptor(): {
  readonly encoderProjected: ArenaSlice;
  readonly encodedLengths: ArenaSlice;
  readonly decodeStarts: ArenaSlice;
  readonly decodeEnds: ArenaSlice;
  readonly flushFinals: ArenaSlice;
  readonly state: ArenaSlice;
  readonly predictorScratch: ArenaSlice;
  readonly tokens: ArenaSlice;
  readonly tokenFrames: ArenaSlice;
  readonly tokenDurations: ArenaSlice;
  readonly metadata: ArenaSlice;
  readonly batchSize: 1;
  readonly frames: 1;
  readonly maxTokens: 4;
} {
  const slice = (
    label: string,
    byteOffset: number,
    byteLength: number,
  ): ArenaSlice => ({
    label,
    bufferIndex: 1,
    byteOffset,
    byteLength,
  });
  return {
    encoderProjected: slice("encoder-projected", 0, 2_560),
    encodedLengths: slice("encoded-lengths", 2_560, 4),
    decodeStarts: slice("decode-starts", 2_564, 4),
    decodeEnds: slice("decode-ends", 2_568, 4),
    flushFinals: slice("flush-finals", 2_572, 4),
    state: slice("decoder-state", 2_816, 10_240),
    predictorScratch: slice(
      "decoder-scratch",
      13_056,
      12_800,
    ),
    tokens: slice("tokens", 25_856, 16),
    tokenFrames: slice("token-frames", 26_112, 16),
    tokenDurations: slice(
      "token-durations",
      26_368,
      16,
    ),
    metadata: slice("metadata", 26_624, 8),
    batchSize: 1,
    frames: 1,
    maxTokens: 4,
  };
}
