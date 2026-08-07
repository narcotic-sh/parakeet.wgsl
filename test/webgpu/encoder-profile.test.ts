import { beforeAll, describe, expect, it, vi } from "vitest";

import type {
  GpuTensor,
  ParakeetGpuPackage,
} from "../../src/model/package";
import {
  PARAKEET_FP16_ENCODER_WEIGHT_FORMAT,
  PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
  type ParakeetModelPrecision,
  type ParakeetTensorRecord,
} from "../../src/model/manifest";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP32_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
} from "../../src/webgpu/capabilities";
import {
  F16_PORTABLE_BITCAST_LOAD_ENCODING,
  F16_PORTABLE_NATIVE_LOAD_ENCODING,
} from "../../src/webgpu/kernels/gemm";
import {
  PARAKEET_ENCODER_SUBMISSION_CHUNKS,
  PARAKEET_FP32_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
  PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
  ParakeetEncoderGraph,
  planEncoderSubmissionChunks,
} from "../../src/webgpu/encoder";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: {
      STORAGE: 1,
      COPY_SRC: 2,
      COPY_DST: 4,
      UNIFORM: 8,
    },
  });
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true,
    value: { COMPUTE: 1 },
  });
});

describe("Parakeet encoder execution profiles", () => {
  it("keeps omitted and explicit FP16 construction byte-identical", async () => {
    const defaultDevice = fakeDevice("fp16");
    const explicitDevice = fakeDevice("fp16");
    const model = fakeModel("fp16");

    const defaultGraph = await ParakeetEncoderGraph.create(
      defaultDevice.device,
      model,
    );
    const explicitGraph = await ParakeetEncoderGraph.create(
      explicitDevice.device,
      model,
      PARAKEET_FP16_EXECUTION_PROFILE,
    );

    expect(defaultGraph.plan.precision).toBe("fp16");
    expect(defaultGraph.arena.bufferCount).toBe(1);
    expect(defaultGraph.transientPaletteExpandedBytesPerGraph).toBe(
      PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
    );
    expect(defaultGraph.portableF16GemmLoadEncoding).toBeNull();
    expect(defaultDevice.shaderModules).toEqual(
      explicitDevice.shaderModules,
    );
    expect(defaultDevice.bufferDescriptors).toEqual(
      explicitDevice.bufferDescriptors,
    );
    expect(
      defaultDevice.shaderModules.some(({ code }) =>
        code.includes("enable f16;"),
      ),
    ).toBe(true);

    defaultGraph.destroy();
    explicitGraph.destroy();
  });

  it("constructs the complete FP32 graph without shader-f16", async () => {
    const captured = fakeDevice("fp32");
    const graph = await ParakeetEncoderGraph.create(
      captured.device,
      fakeModel("fp32"),
      PARAKEET_FP32_EXECUTION_PROFILE,
    );

    expect(graph.plan.precision).toBe("fp32");
    expect(graph.arena.bufferCount).toBe(3);
    expect(graph.arena.bufferByteLengths).toEqual(
      graph.plan.buffers.map(({ byteLength }) => byteLength),
    );
    expect(graph.transientPaletteExpansionDispatches).toBe(196);
    expect(graph.transientPaletteExpandedBytesPerGraph).toBe(
      PARAKEET_FP32_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
    );
    expect(graph.portableF16GemmLoadEncoding).toBeNull();
    expect(
      captured.shaderModules.some(({ label }) =>
        label.includes("fp32"),
      ),
    ).toBe(true);
    expect(
      captured.shaderModules.every(({ code }) =>
        !/\bf16\b/.test(code),
      ),
    ).toBe(true);
    const firstSubsampling = captured.bindGroups.find(
      ({ label }) =>
        label ===
        "subsampling-fused-conv0-depthwise2-batch-0-size-1-bindings",
    );
    const featureBinding = Array.from(
      firstSubsampling?.entries ?? [],
    ).find(({ binding }) => binding === 0)?.resource as
      | GPUBufferBinding
      | undefined;
    expect(featureBinding?.buffer).toBe(captured.buffers[1]);

    graph.destroy();
  });

  it("constructs complete portable graphs on devices without subgroups", async () => {
    for (const precision of ["fp16", "fp32"] as const) {
      const captured = fakeDevice(precision, false);
      const graph = await ParakeetEncoderGraph.create(
        captured.device,
        fakeModel(precision),
        precision === "fp16"
          ? PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE
          : PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
      );

      expect(graph.plan.precision).toBe(precision);
      expect(graph.portableF16GemmLoadEncoding).toBe(
        precision === "fp16"
          ? F16_PORTABLE_BITCAST_LOAD_ENCODING
          : null,
      );
      expect(captured.device.features.has("subgroups")).toBe(false);
      expect(captured.shaderModules).not.toHaveLength(0);
      expect(
        captured.shaderModules.every(({ code }) =>
          !code.includes("subgroup"),
        ),
      ).toBe(true);
      expect(
        captured.shaderModules.some(({ label }) =>
          label.includes("portable"),
        ),
      ).toBe(true);
      if (precision === "fp16") {
        const portableGemmModules =
          captured.shaderModules.filter(({ label }) =>
            label.startsWith("parakeet-gemm-") &&
            label.endsWith("-portable"),
          );
        expect(portableGemmModules.length).toBeGreaterThan(0);
        expect(
          portableGemmModules.every(({ code }) =>
            code.includes("return bitcast<vec4<f16>>(packed);"),
          ),
        ).toBe(true);
      }

      graph.destroy();
    }
  });

  it("selects native FP16 storage when the compiler reports the cross-width bitcast error", async () => {
    const captured = fakeDevice("fp16", false, false);
    const graph = await ParakeetEncoderGraph.create(
      captured.device,
      fakeModel("fp16"),
      PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
    );

    expect(graph.portableF16GemmLoadEncoding).toBe(
      F16_PORTABLE_NATIVE_LOAD_ENCODING,
    );
    const portableGemmModules = captured.shaderModules.filter(
      ({ label }) =>
        label.startsWith("parakeet-gemm-") &&
        label.endsWith("-portable"),
    );
    expect(portableGemmModules.length).toBeGreaterThan(0);
    expect(
      portableGemmModules.every(
        ({ code }) =>
          code.includes("matrix_a: array<vec4<f16>>;") &&
          code.includes("matrix_b: array<vec4<f16>>;") &&
          !code.includes("unpack2x16float") &&
          !code.includes("bitcast<vec4<f16>>"),
      ),
    ).toBe(true);

    graph.destroy();
  });

  it("preserves monolithic dispatch order across every encoder submission chunk", async () => {
    for (const precision of ["fp16", "fp32"] as const) {
      const captured = fakeDevice(precision);
      const graph = await ParakeetEncoderGraph.create(
        captured.device,
        fakeModel(precision),
        precision === "fp16"
          ? PARAKEET_FP16_EXECUTION_PROFILE
          : PARAKEET_FP32_EXECUTION_PROFILE,
      );

      for (const batchSize of [1, 38, 39, 40]) {
        const validFeatureLengths = Array.from(
          { length: batchSize },
          (_, index) => (index === batchSize - 1 ? 1_497 : 1_501),
        );
        captured.commandBuffers.splice(0);
        const monolithic = captured.device.createCommandEncoder({
          label: `monolithic-${precision}-b${batchSize}`,
        });
        graph.encode(monolithic, validFeatureLengths);
        monolithic.finish();
        const monolithicPasses =
          captured.commandBuffers.at(-1)!.passLabels;

        captured.commandBuffers.splice(0);
        const chunks = graph.encodeSubmissionChunks(validFeatureLengths);
        const chunkRecords = [...captured.commandBuffers];

        expect(chunks).toHaveLength(
          PARAKEET_ENCODER_SUBMISSION_CHUNKS,
        );
        expect(chunkRecords.map(({ label }) => label)).toEqual(
          planEncoderSubmissionChunks().map(({ label }) => label),
        );
        expect(chunkRecords.flatMap(({ passLabels }) => passLabels)).toEqual(
          monolithicPasses,
        );
        expect(chunks).toEqual(
          chunkRecords.map(({ commandBuffer }) => commandBuffer),
        );
      }

      graph.destroy();
    }
  });

  it("rejects model/profile precision mismatches before allocating", async () => {
    const captured = fakeDevice("fp32");
    await expect(
      ParakeetEncoderGraph.create(
        captured.device,
        fakeModel("fp16"),
        PARAKEET_FP32_EXECUTION_PROFILE,
      ),
    ).rejects.toThrow(/cannot use fp16 model weights/);
    expect(captured.bufferDescriptors).toHaveLength(0);
  });
});

interface CapturedShaderModule {
  readonly label: string;
  readonly code: string;
}

interface CapturedCommandBuffer {
  readonly commandBuffer: GPUCommandBuffer;
  readonly label: string;
  readonly passLabels: readonly string[];
}

interface CapturedDevice {
  readonly device: GPUDevice;
  readonly shaderModules: CapturedShaderModule[];
  readonly bufferDescriptors: GPUBufferDescriptor[];
  readonly buffers: GPUBuffer[];
  readonly bindGroups: GPUBindGroupDescriptor[];
  readonly commandBuffers: CapturedCommandBuffer[];
}

function fakeDevice(
  precision: ParakeetModelPrecision,
  exposesSubgroups = true,
  acceptsPortableF16Bitcast = true,
): CapturedDevice {
  const shaderModules: CapturedShaderModule[] = [];
  const bufferDescriptors: GPUBufferDescriptor[] = [];
  const buffers: GPUBuffer[] = [];
  const bindGroups: GPUBindGroupDescriptor[] = [];
  const commandBuffers: CapturedCommandBuffer[] = [];
  const features = new Set<string>([
    ...(exposesSubgroups ? ["subgroups"] : []),
    ...(precision === "fp16" ? ["shader-f16"] : []),
  ]);
  const device = {
    features,
    limits: {
      maxBufferSize: 512 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeInvocationsPerWorkgroup: 1_024,
      maxComputeWorkgroupSizeX: 1_024,
      maxComputeWorkgroupSizeY: 1_024,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupStorageSize: 64 * 1024,
      maxStorageBuffersPerShaderStage: 16,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: {
      writeBuffer: vi.fn(),
    },
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      bufferDescriptors.push({ ...descriptor });
      const buffer = {
        destroy: vi.fn(),
      } as unknown as GPUBuffer;
      buffers.push(buffer);
      return buffer;
    }),
    createBindGroupLayout: vi.fn(
      () => ({}) as GPUBindGroupLayout,
    ),
    createPipelineLayout: vi.fn(
      () => ({}) as GPUPipelineLayout,
    ),
    createShaderModule: vi.fn(
      (descriptor: GPUShaderModuleDescriptor) => {
        shaderModules.push({
          label: descriptor.label ?? "",
          code: descriptor.code,
        });
        return {
          getCompilationInfo: async () => ({
            messages:
              !acceptsPortableF16Bitcast &&
              descriptor.label ===
                "parakeet-fp16-portable-bitcast-load-probe"
                ? [{ type: "error" }]
                : [],
          }),
        } as unknown as GPUShaderModule;
      },
    ),
    createComputePipelineAsync: vi.fn(
      async () => ({}) as GPUComputePipeline,
    ),
    createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => {
      bindGroups.push(descriptor);
      return {} as GPUBindGroup;
    }),
    createCommandEncoder: vi.fn(
      (descriptor: GPUCommandEncoderDescriptor = {}) => {
        const passLabels: string[] = [];
        return {
          beginComputePass(
            passDescriptor: GPUComputePassDescriptor = {},
          ) {
            passLabels.push(passDescriptor.label ?? "");
            return {
              setBindGroup: vi.fn(),
              setPipeline: vi.fn(),
              dispatchWorkgroups: vi.fn(),
              end: vi.fn(),
            };
          },
          copyBufferToBuffer: vi.fn(),
          finish(
            finishDescriptor: GPUCommandBufferDescriptor = {},
          ) {
            const label =
              finishDescriptor.label ?? descriptor.label ?? "";
            const commandBuffer = {
              label,
            } as GPUCommandBuffer;
            commandBuffers.push({
              commandBuffer,
              label,
              passLabels: [...passLabels],
            });
            return commandBuffer;
          },
        } as unknown as GPUCommandEncoder;
      },
    ),
  } as unknown as GPUDevice;
  return {
    device,
    shaderModules,
    bufferDescriptors,
    buffers,
    bindGroups,
    commandBuffers,
  };
}

function fakeModel(
  precision: ParakeetModelPrecision,
): ParakeetGpuPackage {
  const modelBuffer = {
    destroy: vi.fn(),
  } as unknown as GPUBuffer;
  const config = {
    featureBins: 128,
    encoderLayers: 24,
    encoderHiddenSize: 1_024,
    intermediateSize: 4_096,
    attentionHeads: 8,
    attentionHeadSize: 128,
    convolutionKernelSize: 9,
    decoderHiddenSize: 640,
    subsamplingFactor: 8,
    encoderWeightFormat:
      precision === "fp16"
        ? PARAKEET_FP16_ENCODER_WEIGHT_FORMAT
        : PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
  };
  return {
    precision,
    manifest: { config },
    tensor(name: string): GpuTensor {
      const record = fakeTensorRecord(name, precision);
      return {
        sourceRecord: record,
        runtimeRecord: {
          ...record,
          bufferId: "source-shard-fake.bin",
        },
        buffer: modelBuffer,
        binding: {
          buffer: modelBuffer,
          offset: 0,
          size: record.byteLength,
        },
      };
    },
  } as unknown as ParakeetGpuPackage;
}

interface MatrixShape {
  readonly inner: number;
  readonly columns: number;
  readonly bits: 5 | 6;
  readonly paletteGroups: number;
}

function fakeTensorRecord(
  name: string,
  precision: ParakeetModelPrecision,
): ParakeetTensorRecord {
  const matrixName = name.endsWith(".palette")
    ? name.slice(0, -".palette".length)
    : name;
  const matrix = matrixShape(matrixName);
  if (matrix !== undefined) {
    if (name.endsWith(".palette")) {
      return tensorRecord({
        dtype: precision === "fp16" ? "float16" : "float32",
        logicalShape: [matrix.paletteGroups, 1 << matrix.bits],
        storageShape: [matrix.paletteGroups, 1 << matrix.bits],
        layout: "output-group-by-palette-lut",
      });
    }
    const codesPerGroup = matrix.bits === 5 ? 32 : 16;
    const wordsPerGroup = matrix.bits === 5 ? 5 : 3;
    return tensorRecord({
      dtype: "uint32",
      logicalShape: [matrix.inner, matrix.columns],
      storageShape: [
        (matrix.inner * matrix.columns) / codesPerGroup,
        wordsPerGroup,
      ],
      layout:
        `k-by-output-row-major-palette${matrix.bits}-lsb-u32`,
    });
  }

  const scalarDtype = precision === "fp16" ? "float16" : "float32";
  if (name === "encoder.subsampling.layers.0.weight") {
    return tensorRecord({
      dtype: scalarDtype,
      logicalShape: [3, 3, 256],
      storageShape: [3, 3, 64, 4],
      layout: "kh-kw-output-channel-vec4",
    });
  }
  if (
    name === "encoder.subsampling.layers.2.weight" ||
    name === "encoder.subsampling.layers.5.weight"
  ) {
    return tensorRecord({
      dtype: scalarDtype,
      logicalShape: [3, 3, 256],
      storageShape: [3, 3, 64, 4],
      layout: "kh-kw-channel-vec4",
    });
  }
  if (name.endsWith(".self_attn.projected_positions")) {
    return tensorRecord({
      dtype: scalarDtype,
      logicalShape: [375, 1_024],
      storageShape: [8, 32, 375, 4],
      layout: "head-vector-position-vec4",
    });
  }
  if (name.endsWith(".conv.depthwise_conv.weight")) {
    return tensorRecord({
      dtype: scalarDtype,
      logicalShape: [9, 1_024],
      storageShape: [9, 256, 4],
      layout: "kernel-channel-vec4",
    });
  }
  if (
    name.endsWith(".self_attn.bias_u") ||
    name.endsWith(".self_attn.bias_v")
  ) {
    return tensorRecord({
      dtype: "float32",
      logicalShape: [8, 128],
      storageShape: [8, 128],
      layout: "head-by-channel",
    });
  }
  if (
    name.endsWith(".norm.scale") ||
    name.endsWith(".norm.shift") ||
    (
      name.startsWith("encoder.layers.") &&
      (
        name.endsWith(".weight") ||
        name.endsWith(".bias")
      )
    )
  ) {
    return tensorRecord({
      dtype: "float32",
      logicalShape: [1_024],
      storageShape: [1_024],
      layout: "channel",
    });
  }
  const outputChannels =
    name === "encoder.subsampling.linear.bias"
      ? 1_024
      : name === "encoder_projector.bias"
        ? 640
        : 256;
  if (
    name.startsWith("encoder.subsampling.") ||
    name === "encoder_projector.bias"
  ) {
    return tensorRecord({
      dtype: "float32",
      logicalShape: [outputChannels],
      storageShape: [outputChannels],
      layout: "output-channel",
    });
  }
  throw new Error(`Unexpected encoder tensor ${name}`);
}

function matrixShape(name: string): MatrixShape | undefined {
  if (
    name === "encoder.subsampling.layers.3.weight" ||
    name === "encoder.subsampling.layers.6.weight"
  ) {
    return { inner: 256, columns: 256, bits: 5, paletteGroups: 1 };
  }
  if (name === "encoder.subsampling.linear.weight") {
    return { inner: 4_096, columns: 1_024, bits: 5, paletteGroups: 1 };
  }
  if (name === "encoder_projector.weight") {
    return { inner: 1_024, columns: 640, bits: 5, paletteGroups: 1 };
  }
  if (/\.feed_forward[12]\.linear1\.weight$/.test(name)) {
    return { inner: 1_024, columns: 4_096, bits: 5, paletteGroups: 1 };
  }
  if (/\.feed_forward[12]\.linear2\.weight$/.test(name)) {
    return { inner: 4_096, columns: 1_024, bits: 5, paletteGroups: 1 };
  }
  if (name.endsWith(".self_attn.qkv.weight")) {
    return { inner: 1_024, columns: 3_072, bits: 5, paletteGroups: 3 };
  }
  if (name.endsWith(".self_attn.o_proj.weight")) {
    return { inner: 1_024, columns: 1_024, bits: 6, paletteGroups: 1 };
  }
  if (name.endsWith(".conv.pointwise_conv1.weight")) {
    return { inner: 1_024, columns: 2_048, bits: 5, paletteGroups: 1 };
  }
  if (name.endsWith(".conv.pointwise_conv2.weight")) {
    return { inner: 1_024, columns: 1_024, bits: 5, paletteGroups: 1 };
  }
  return undefined;
}

function tensorRecord(spec: {
  readonly dtype: ParakeetTensorRecord["dtype"];
  readonly logicalShape: readonly number[];
  readonly storageShape: readonly number[];
  readonly layout: string;
}): ParakeetTensorRecord {
  const scalarBytes = spec.dtype === "float16" ? 2 : 4;
  return {
    shard: "fake.bin",
    byteOffset: 0,
    byteLength:
      spec.storageShape.reduce((product, value) => product * value, 1) *
      scalarBytes,
    dtype: spec.dtype,
    logicalShape: spec.logicalShape,
    storageShape: spec.storageShape,
    layout: spec.layout,
  };
}
