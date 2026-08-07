import { describe, expect, it } from "vitest";

import type {
  ParakeetFileRecord,
  ParakeetManifest,
  ParakeetTensorRecord,
} from "../../src/model/manifest";
import {
  PARAKEET_FP16_PACKAGE_FORMAT,
  PARAKEET_FP32_PACKAGE_FORMAT,
} from "../../src/model/manifest";
import {
  alignRuntimeOffset,
  FP16_EXPANDED_RUNTIME_BYTES,
  FP16_LARGEST_RUNTIME_SHARD_BYTES,
  FP16_PACKED_DOWNLOAD_BYTES,
  FP16_RETAINED_RUNTIME_BYTES,
  FP16_RUNTIME_GPU_BYTES,
  FP32_EXPANDED_RUNTIME_BYTES,
  FP32_LARGEST_RUNTIME_SHARD_BYTES,
  FP32_PACKED_DOWNLOAD_BYTES,
  FP32_RETAINED_RUNTIME_BYTES,
  FP32_RUNTIME_GPU_BYTES,
  planRuntimePackage,
  RUNTIME_BUFFER_LIMIT_BYTES,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
  runtimeGemmTileMajorF32ScalarIndex,
  runtimeGemmTileMajorF32StorageShape,
  runtimeGemmTileMajorScalarIndex,
  runtimeGemmTileMajorStorageShape,
} from "../../src/model/runtime-plan";

interface TestModelWeightFamily {
  readonly weightNames: readonly string[];
  readonly inner: number;
  readonly columns: number;
  readonly paletteBits: 5 | 6;
  readonly paletteGroupColumns: number;
}

function layerNames(...suffixes: readonly string[]): readonly string[] {
  return Array.from(
    { length: 24 },
    (_, layer) =>
      suffixes.map((suffix) => `encoder.layers.${layer}.${suffix}`),
  ).flat();
}

const MODEL_WEIGHT_FAMILIES = [
  {
    weightNames: layerNames(
      "feed_forward1.linear1.weight",
      "feed_forward2.linear1.weight",
    ),
    inner: 1024,
    columns: 4096,
    paletteBits: 5,
    paletteGroupColumns: 4096,
  },
  {
    weightNames: layerNames(
      "feed_forward1.linear2.weight",
      "feed_forward2.linear2.weight",
    ),
    inner: 4096,
    columns: 1024,
    paletteBits: 5,
    paletteGroupColumns: 1024,
  },
  {
    weightNames: layerNames("self_attn.qkv.weight"),
    inner: 1024,
    columns: 3072,
    paletteBits: 5,
    paletteGroupColumns: 1024,
  },
  {
    weightNames: layerNames("self_attn.o_proj.weight"),
    inner: 1024,
    columns: 1024,
    paletteBits: 6,
    paletteGroupColumns: 1024,
  },
  {
    weightNames: layerNames("conv.pointwise_conv1.weight"),
    inner: 1024,
    columns: 2048,
    paletteBits: 5,
    paletteGroupColumns: 2048,
  },
  {
    weightNames: layerNames("conv.pointwise_conv2.weight"),
    inner: 1024,
    columns: 1024,
    paletteBits: 5,
    paletteGroupColumns: 1024,
  },
  {
    weightNames: ["encoder_projector.weight"],
    inner: 1024,
    columns: 640,
    paletteBits: 5,
    paletteGroupColumns: 640,
  },
  {
    weightNames: ["encoder.subsampling.layers.3.weight"],
    inner: 256,
    columns: 256,
    paletteBits: 5,
    paletteGroupColumns: 256,
  },
  {
    weightNames: ["encoder.subsampling.layers.6.weight"],
    inner: 256,
    columns: 256,
    paletteBits: 5,
    paletteGroupColumns: 256,
  },
  {
    weightNames: ["encoder.subsampling.linear.weight"],
    inner: 4096,
    columns: 1024,
    paletteBits: 5,
    paletteGroupColumns: 1024,
  },
] as const satisfies readonly TestModelWeightFamily[];

describe("fixed runtime package plan", () => {
  it("retains every non-preprocessor shard and all 196 palette matrices packed", () => {
    const manifest = syntheticRuntimeManifest();
    const plan = planRuntimePackage(manifest);

    expect(plan.tensors.size).toBe(795);
    expect(plan.buffers).toHaveLength(26);
    expect(plan.memory).toEqual({
      packedDownloadBytes: FP16_PACKED_DOWNLOAD_BYTES,
      retainedRuntimeBytes: FP16_RETAINED_RUNTIME_BYTES,
      expandedRuntimeBytes: FP16_EXPANDED_RUNTIME_BYTES,
      runtimeGpuBytes: FP16_RUNTIME_GPU_BYTES,
      largestRuntimeShardBytes:
        FP16_LARGEST_RUNTIME_SHARD_BYTES,
      loaderPeakGpuBytes: FP16_RUNTIME_GPU_BYTES,
      runtimeBufferCount: 26,
    });
    expect(plan.buffers[0]).toMatchObject({
      id: "source-shard-decoder.bin",
      byteLength: FP16_LARGEST_RUNTIME_SHARD_BYTES,
    });
    for (let layer = 0; layer < 24; layer += 1) {
      expect(plan.buffers[layer + 1]).toMatchObject({
        id:
          `source-shard-encoder-layer-${String(layer).padStart(2, "0")}.bin`,
        byteLength: 16_050_176,
      });
    }
    expect(plan.buffers[25]).toMatchObject({
      id: "source-shard-encoder-subsampling.bin",
      byteLength: 2_727_168,
    });
    expect(
      plan.buffers.every(
        (buffer) =>
          buffer.byteLength > 0 &&
          buffer.byteLength <= RUNTIME_BUFFER_LIMIT_BYTES,
      ),
    ).toBe(true);

    const preprocessor = plan.shards.find(
      (shard) => shard.file.name === "preprocessor.bin",
    );
    expect(preprocessor).toEqual({
      file: expect.objectContaining({
        name: "preprocessor.bin",
      }),
      hashOnly: true,
    });
    expect(
      plan.shards
        .filter((shard) => !shard.hashOnly)
        .every(
          (shard) => shard.runtimeBufferId !== undefined,
        ),
    ).toBe(true);
    expect(plan.shards.map((shard) => shard.file.name)).toEqual([
      "decoder.bin",
      "encoder-subsampling.bin",
      ...Array.from(
        { length: 24 },
        (_, index) =>
          `encoder-layer-${String(index).padStart(2, "0")}.bin`,
      ),
      "preprocessor.bin",
    ]);
    expect(
      plan.shards
        .filter((shard) => !shard.hashOnly)
        .map((shard) => shard.runtimeBufferId),
    ).toHaveLength(26);

    const qkv = plan.tensors.get(
      "encoder.layers.7.self_attn.qkv.weight",
    );
    expect(qkv?.sourceRecord).toMatchObject({
      dtype: "uint32",
      logicalShape: [1024, 3072],
      layout: "k-by-output-row-major-palette5-lsb-u32",
    });
    expect(qkv?.runtimeRecord).toMatchObject({
      bufferId: "source-shard-encoder-layer-07.bin",
      byteLength: 1024 * 3072 / 32 * 5 * 4,
      dtype: "uint32",
      logicalShape: [1024, 3072],
      storageShape: [1024 * 3072 / 32, 5],
      layout: "k-by-output-row-major-palette5-lsb-u32",
    });
    const expanded = [...plan.tensors.values()].filter(
      (tensor) => tensor.runtimeRecord.dtype === "float16" &&
        tensor.runtimeRecord.layout ===
          RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
    );
    expect(expanded).toHaveLength(0);
    expect(
      plan.tensors.get("encoder_projector.weight")?.runtimeRecord,
    ).toMatchObject({
      bufferId: "source-shard-decoder.bin",
      logicalShape: [1024, 640],
      storageShape: [20_480, 5],
      layout: "k-by-output-row-major-palette5-lsb-u32",
    });
    expect(
      plan.tensors.has(
        "encoder.layers.7.self_attn.qkv.weight.palette",
      ),
    ).toBe(true);
    for (const name of [
      "encoder.subsampling.layers.3.weight",
      "encoder.subsampling.layers.6.weight",
      "encoder.subsampling.linear.weight",
      "encoder_projector.weight",
    ]) {
      expect(plan.tensors.has(`${name}.palette`)).toBe(true);
    }
    expect(plan.tensors.has("preprocessor.window")).toBe(false);
  });

  it("pins the FP32 package inventory with FP32 palette LUTs", () => {
    const manifest = syntheticRuntimeManifest("fp32");
    const plan = planRuntimePackage(manifest);

    expect(plan.tensors.size).toBe(795);
    expect(plan.buffers).toHaveLength(26);
    expect(plan.memory).toEqual({
      packedDownloadBytes: FP32_PACKED_DOWNLOAD_BYTES,
      retainedRuntimeBytes: FP32_RETAINED_RUNTIME_BYTES,
      expandedRuntimeBytes: FP32_EXPANDED_RUNTIME_BYTES,
      runtimeGpuBytes: FP32_RUNTIME_GPU_BYTES,
      largestRuntimeShardBytes:
        FP32_LARGEST_RUNTIME_SHARD_BYTES,
      loaderPeakGpuBytes: FP32_RUNTIME_GPU_BYTES,
      runtimeBufferCount: 26,
    });
    expect(plan.buffers[0]).toMatchObject({
      id: "source-shard-decoder.bin",
      byteLength: FP32_LARGEST_RUNTIME_SHARD_BYTES,
    });
    for (let layer = 0; layer < 24; layer += 1) {
      expect(plan.buffers[layer + 1]).toMatchObject({
        id:
          `source-shard-encoder-layer-${String(layer).padStart(2, "0")}.bin`,
        byteLength: 16_836_864,
      });
    }
    expect(plan.buffers[25]).toMatchObject({
      id: "source-shard-encoder-subsampling.bin",
      byteLength: 2_740_992,
    });
    expect(
      [...plan.tensors.values()].some(
        (tensor) => tensor.runtimeRecord.dtype === "float16",
      ),
    ).toBe(false);
    expect(
      plan.tensors.get(
        "encoder.layers.7.self_attn.qkv.weight.palette",
      )?.runtimeRecord,
    ).toMatchObject({
      dtype: "float32",
      byteLength: 3 * 32 * 4,
      logicalShape: [3, 32],
      storageShape: [3, 32],
      layout: "output-group-by-palette-lut",
    });
    expect(
      plan.tensors.get(
        "encoder.layers.7.self_attn.o_proj.weight.palette",
      )?.runtimeRecord,
    ).toMatchObject({
      dtype: "float32",
      byteLength: 4 * 64,
      logicalShape: [1, 64],
      storageShape: [1, 64],
      layout: "output-group-by-palette-lut",
    });
  });

  it("defines a bijective K32/N256 physical layout for every layer shape", () => {
    for (const [inner, columns] of [
      [1024, 4096],
      [4096, 1024],
      [1024, 3072],
      [1024, 1024],
      [1024, 2048],
    ] as const) {
      const storageShape =
        runtimeGemmTileMajorStorageShape(inner, columns);
      expect(
        storageShape.reduce((product, value) => product * value, 1),
      ).toBe(inner * columns);
      expect(
        runtimeGemmTileMajorScalarIndex(0, inner, columns),
      ).toBe(0);
      expect(
        runtimeGemmTileMajorScalarIndex(
          inner * columns - 1,
          inner,
          columns,
        ),
      ).toBe(inner * columns - 1);
    }

    const inner = 64;
    const columns = 512;
    const physicalIndices = new Set<number>();
    for (let scalar = 0; scalar < inner * columns; scalar++) {
      physicalIndices.add(
        runtimeGemmTileMajorScalarIndex(
          scalar,
          inner,
          columns,
        ),
      );
    }
    expect(physicalIndices.size).toBe(inner * columns);
    expect(Math.min(...physicalIndices)).toBe(0);
    expect(Math.max(...physicalIndices)).toBe(
      inner * columns - 1,
    );
    expect(() =>
      runtimeGemmTileMajorStorageShape(1024, 640),
    ).toThrow(/N%256/);
    expect(() =>
      runtimeGemmTileMajorScalarIndex(
        inner * columns,
        inner,
        columns,
      ),
    ).toThrow(/exceeds/);
  });

  it("defines FP32 row-major and bijective K32/N128 tile layouts", () => {
    expect(RUNTIME_ROW_MAJOR_F32_LAYOUT).toBe(
      "k-by-output-row-major-f32",
    );
    expect(RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT).toBe(
      "n128-k32-tile-major-f32",
    );
    for (const [inner, columns] of [
      [1024, 4096],
      [4096, 1024],
      [1024, 3072],
      [1024, 1024],
      [1024, 2048],
      [1024, 640],
    ] as const) {
      const storageShape =
        runtimeGemmTileMajorF32StorageShape(inner, columns);
      expect(
        storageShape.reduce(
          (product, value) => product * value,
          1,
        ),
      ).toBe(inner * columns);
      expect(
        runtimeGemmTileMajorF32ScalarIndex(
          0,
          inner,
          columns,
        ),
      ).toBe(0);
      expect(
        runtimeGemmTileMajorF32ScalarIndex(
          inner * columns - 1,
          inner,
          columns,
        ),
      ).toBe(inner * columns - 1);
    }
    expect(
      runtimeGemmTileMajorF32StorageShape(1024, 3072),
    ).toEqual([24, 32, 32, 128]);

    const inner = 64;
    const columns = 256;
    const physicalIndices = new Set<number>();
    for (let scalar = 0; scalar < inner * columns; scalar++) {
      physicalIndices.add(
        runtimeGemmTileMajorF32ScalarIndex(
          scalar,
          inner,
          columns,
        ),
      );
    }
    expect(physicalIndices.size).toBe(inner * columns);
    expect(Math.min(...physicalIndices)).toBe(0);
    expect(Math.max(...physicalIndices)).toBe(
      inner * columns - 1,
    );
    expect(() =>
      runtimeGemmTileMajorF32StorageShape(1024, 192),
    ).toThrow(/N%128/);
    expect(() =>
      runtimeGemmTileMajorF32ScalarIndex(
        inner * columns,
        inner,
        columns,
      ),
    ).toThrow(/exceeds/);
  });

  it("uses each validated whole shard directly as its runtime buffer", () => {
    const plan = planRuntimePackage(syntheticRuntimeManifest());
    for (const tensor of plan.tensors.values()) {
      expect(tensor.runtimeRecord.byteOffset % 256).toBe(0);
      expect(tensor.runtimeRecord).toMatchObject({
        byteOffset: tensor.sourceRecord.byteOffset,
        byteLength: tensor.sourceRecord.byteLength,
        dtype: tensor.sourceRecord.dtype,
        logicalShape: tensor.sourceRecord.logicalShape,
        storageShape: tensor.sourceRecord.storageShape,
        layout: tensor.sourceRecord.layout,
      });
      expect(tensor.runtimeRecord.bufferId).toBe(
        `source-shard-${tensor.sourceRecord.shard}`,
      );
    }
    expect(alignRuntimeOffset(0)).toBe(0);
    expect(alignRuntimeOffset(1)).toBe(256);
    expect(alignRuntimeOffset(257)).toBe(512);
    expect(() => alignRuntimeOffset(-1)).toThrow(/nonnegative/);
    expect(
      plan.tensors.get("encoder_projector.bias")?.runtimeRecord
        .byteOffset,
    ).toBe(15_671_296);
    expect(
      plan.tensors.get("encoder.subsampling.layers.3.bias")
        ?.runtimeRecord.byteOffset,
    ).toBe(58_112);
  });

  it("rejects inventory, palette, range, and retained-package drift", () => {
    const missingPalette = syntheticRuntimeManifest();
    delete mutableTensors(missingPalette)[
      "encoder.layers.0.self_attn.qkv.weight.palette"
    ];
    expect(() => planRuntimePackage(missingPalette)).toThrow(
      /missing encoder\.layers\.0\.self_attn\.qkv\.weight\.palette/,
    );

    const invalidRange = syntheticRuntimeManifest();
    const rangeRecord =
      mutableTensors(invalidRange)["retained.0000"]!;
    mutableTensors(invalidRange)["retained.0000"] = {
      ...rangeRecord,
      byteOffset: FP16_LARGEST_RUNTIME_SHARD_BYTES,
    };
    expect(() => planRuntimePackage(invalidRange)).toThrow(
      /exceeds its aligned source shard/,
    );

    const invalidPalette = syntheticRuntimeManifest();
    const paletteName =
      "encoder.layers.0.self_attn.o_proj.weight.palette";
    const palette = mutableTensors(invalidPalette)[paletteName]!;
    mutableTensors(invalidPalette)[paletteName] = {
      ...palette,
      logicalShape: [1, 32],
      storageShape: [1, 32],
      byteLength: 64,
    };
    expect(() => planRuntimePackage(invalidPalette)).toThrow(
      /invalid palette storage/,
    );

    const fp32WithFp16Palette =
      syntheticRuntimeManifest("fp32");
    const fp32Palette =
      mutableTensors(fp32WithFp16Palette)[paletteName]!;
    mutableTensors(fp32WithFp16Palette)[paletteName] = {
      ...fp32Palette,
      dtype: "float16",
      byteLength: fp32Palette.byteLength / 2,
    };
    expect(() =>
      planRuntimePackage(fp32WithFp16Palette),
    ).toThrow(/invalid palette storage/);

    const fp32WithFloat16Tensor =
      syntheticRuntimeManifest("fp32");
    const fp32Retained =
      mutableTensors(fp32WithFloat16Tensor)["retained.0000"]!;
    mutableTensors(fp32WithFloat16Tensor)["retained.0000"] = {
      ...fp32Retained,
      dtype: "float16",
    };
    expect(() =>
      planRuntimePackage(fp32WithFloat16Tensor),
    ).toThrow(/cannot retain float16 storage in the FP32 package/);

    const paletteTopologyDrift = syntheticRuntimeManifest();
    const qkvName =
      "encoder.layers.0.self_attn.qkv.weight";
    const qkv = mutableTensors(paletteTopologyDrift)[qkvName]!;
    const qkvGroups = 1024 * 3072 / 16;
    mutableTensors(paletteTopologyDrift)[qkvName] = {
      ...qkv,
      byteLength: qkvGroups * 3 * 4,
      storageShape: [qkvGroups, 3],
      layout: "k-by-output-row-major-palette6-lsb-u32",
    };
    mutableTensors(paletteTopologyDrift)[`${qkvName}.palette`] = {
      ...mutableTensors(paletteTopologyDrift)[
        `${qkvName}.palette`
      ]!,
      byteLength: 3 * 64 * 2,
      logicalShape: [3, 64],
      storageShape: [3, 64],
    };
    expect(() =>
      planRuntimePackage(paletteTopologyDrift),
    ).toThrow(/Packed palette topology changed/);

    const retainedDrift = syntheticRuntimeManifest();
    delete mutableTensors(retainedDrift)["retained.0000"];
    expect(() => planRuntimePackage(retainedDrift)).toThrow(
      /Retained runtime tensor count changed/,
    );

    const fp32ShardDrift = syntheticRuntimeManifest("fp32");
    const fp32Files =
      fp32ShardDrift.files as ParakeetFileRecord[];
    fp32Files[1] = {
      ...fp32Files[1]!,
      byteLength: fp32Files[1]!.byteLength + 256,
    };
    expect(() => planRuntimePackage(fp32ShardDrift)).toThrow(
      /fp32 common-source shard inventory changed/,
    );
  });
});

function syntheticRuntimeManifest(
  precision: "fp16" | "fp32" = "fp16",
): ParakeetManifest {
  const fp32 = precision === "fp32";
  const files: ParakeetFileRecord[] = [
    {
      name: "preprocessor.bin",
      byteLength: 133_376,
      sha256: "0".repeat(64),
    },
    {
      name: "decoder.bin",
      byteLength: fp32
        ? FP32_LARGEST_RUNTIME_SHARD_BYTES
        : FP16_LARGEST_RUNTIME_SHARD_BYTES,
      sha256: "0".repeat(64),
    },
    ...Array.from({ length: 24 }, (_, index) => ({
      name:
        `encoder-layer-${String(index).padStart(2, "0")}.bin`,
      byteLength: fp32 ? 16_836_864 : 16_050_176,
      sha256: "0".repeat(64),
    })),
    {
      name: "encoder-subsampling.bin",
      byteLength: fp32 ? 2_740_992 : 2_727_168,
      sha256: "0".repeat(64),
    },
  ];
  const tensors: Record<string, ParakeetTensorRecord> = {
    "preprocessor.window": record(
      "preprocessor.bin",
      4,
      "float32",
      [1],
      [1],
      "window",
    ),
    "preprocessor.mel_filters": record(
      "preprocessor.bin",
      4,
      "float32",
      [1],
      [1],
      "mel",
    ),
  };
  for (const family of MODEL_WEIGHT_FAMILIES) {
    const codesPerGroup =
      family.paletteBits === 5 ? 32 : 16;
    const wordsPerGroup =
      family.paletteBits === 5 ? 5 : 3;
    const groups =
      family.inner * family.columns / codesPerGroup;
    const paletteGroups =
      family.columns / family.paletteGroupColumns;
    const paletteEntries = 1 << family.paletteBits;
    for (const name of family.weightNames) {
      const layerMatch = name.match(
        /^encoder\.layers\.(\d+)\./,
      );
      const shard = layerMatch === null
        ? name.startsWith("encoder.subsampling.")
          ? "encoder-subsampling.bin"
          : "decoder.bin"
        : `encoder-layer-${String(
            Number(layerMatch[1]),
          ).padStart(2, "0")}.bin`;
      tensors[name] = record(
        shard,
        groups * wordsPerGroup * 4,
        "uint32",
        [family.inner, family.columns],
        [groups, wordsPerGroup],
        `k-by-output-row-major-palette${family.paletteBits}-lsb-u32`,
      );
      tensors[`${name}.palette`] = record(
        shard,
        paletteGroups * paletteEntries * (fp32 ? 4 : 2),
        fp32 ? "float32" : "float16",
        [paletteGroups, paletteEntries],
        [paletteGroups, paletteEntries],
        "output-group-by-palette-lut",
      );
    }
  }
  const commonOffsets = new Map<string, number>([
    ["encoder_projector.weight", fp32 ? 30_499_840 : 15_261_440],
    [
      "encoder_projector.weight.palette",
      fp32 ? 30_909_440 : 15_671_040,
    ],
    ["encoder.subsampling.layers.3.weight", 16_896],
    ["encoder.subsampling.layers.3.weight.palette", 57_856],
    ["encoder.subsampling.layers.6.weight", 59_136],
    ["encoder.subsampling.layers.6.weight.palette", 100_096],
    ["encoder.subsampling.linear.weight", 101_376],
    ["encoder.subsampling.linear.weight.palette", 2_722_816],
  ]);
  for (const [name, byteOffset] of commonOffsets) {
    tensors[name] = { ...tensors[name]!, byteOffset };
  }
  for (const [name, shard, byteOffset, byteLength] of [
    [
      "decoder.embedding.weight",
      "decoder.bin",
      0,
      fp32 ? 2_624_000 : 1_312_000,
    ],
    [
      "decoder.lstm.weight_l0",
      "decoder.bin",
      fp32 ? 2_624_000 : 1_312_000,
      fp32 ? 13_107_200 : 6_553_600,
    ],
    [
      "decoder.lstm.bias_l0",
      "decoder.bin",
      fp32 ? 15_731_200 : 7_865_600,
      10_240,
    ],
    [
      "decoder.lstm.weight_l1",
      "decoder.bin",
      fp32 ? 15_741_440 : 7_875_840,
      fp32 ? 13_107_200 : 6_553_600,
    ],
    [
      "decoder.lstm.bias_l1",
      "decoder.bin",
      fp32 ? 28_848_640 : 14_429_440,
      10_240,
    ],
    [
      "decoder.decoder_projector.weight",
      "decoder.bin",
      fp32 ? 28_858_880 : 14_439_680,
      fp32 ? 1_638_400 : 819_200,
    ],
    [
      "decoder.decoder_projector.bias",
      "decoder.bin",
      fp32 ? 30_497_280 : 15_258_880,
      2_560,
    ],
    [
      "encoder_projector.bias",
      "decoder.bin",
      fp32 ? 30_909_696 : 15_671_296,
      2_560,
    ],
    [
      "joint.head.weight",
      "decoder.bin",
      fp32 ? 30_912_256 : 15_673_856,
      fp32 ? 2_641_920 : 1_320_960,
    ],
    [
      "joint.head.bias",
      "decoder.bin",
      fp32 ? 33_554_176 : 16_994_816,
      4_128,
    ],
    ["encoder.subsampling.layers.0.weight", "encoder-subsampling.bin", 0, 4_608],
    ["encoder.subsampling.layers.0.bias", "encoder-subsampling.bin", 4_608, 1_024],
    ["encoder.subsampling.layers.2.weight", "encoder-subsampling.bin", 5_632, 4_608],
    ["encoder.subsampling.layers.2.bias", "encoder-subsampling.bin", 10_240, 1_024],
    ["encoder.subsampling.layers.5.weight", "encoder-subsampling.bin", 11_264, 4_608],
    ["encoder.subsampling.layers.5.bias", "encoder-subsampling.bin", 15_872, 1_024],
    ["encoder.subsampling.layers.3.bias", "encoder-subsampling.bin", 58_112, 1_024],
    ["encoder.subsampling.layers.6.bias", "encoder-subsampling.bin", 100_352, 1_024],
    ["encoder.subsampling.linear.bias", "encoder-subsampling.bin", 2_723_072, 4_096],
  ] as const) {
    tensors[name] = {
      ...record(
        shard,
        byteLength,
        "uint32",
        [byteLength / 4],
        [byteLength / 4],
        "test-retained-words",
      ),
      byteOffset,
    };
  }
  for (const index of Array.from({ length: 384 }, (_, value) => value)) {
    const name = `retained.${String(index).padStart(4, "0")}`;
    tensors[name] = record(
      files[2 + index % 24]!.name,
      256,
      "uint32",
      [64],
      [64],
      "test-retained-words",
    );
  }
  return {
    format: fp32
      ? PARAKEET_FP32_PACKAGE_FORMAT
      : PARAKEET_FP16_PACKAGE_FORMAT,
    files,
    tensors,
  } as unknown as ParakeetManifest;
}

function record(
  shard: string,
  byteLength: number,
  dtype: ParakeetTensorRecord["dtype"],
  logicalShape: readonly number[],
  storageShape: readonly number[],
  layout: string,
): ParakeetTensorRecord {
  return {
    shard,
    byteOffset: 0,
    byteLength,
    dtype,
    logicalShape,
    storageShape,
    layout,
  };
}

function mutableTensors(
  manifest: ParakeetManifest,
): Record<string, ParakeetTensorRecord> {
  return manifest.tensors as Record<
    string,
    ParakeetTensorRecord
  >;
}
