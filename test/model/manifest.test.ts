import { describe, expect, it } from "vitest";

import {
  parakeetManifestSha256,
  parakeetManifestPrecision,
  PARAKEET_FP16_ENCODER_WEIGHT_FORMAT,
  PARAKEET_FP16_MANIFEST_SHA256,
  PARAKEET_FP16_PACKAGE_FORMAT,
  PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
  PARAKEET_FP32_MANIFEST_SHA256,
  PARAKEET_FP32_PACKAGE_FORMAT,
  PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH,
  PARAKEET_SOURCE_ARCHIVE_NAME,
  PARAKEET_SOURCE_ARCHIVE_SHA256,
  PARAKEET_SOURCE_REVISION,
  PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH,
  PARAKEET_SOURCE_WEIGHTS_NAME,
  PARAKEET_SOURCE_WEIGHTS_SHA256,
  parseParakeetManifest,
} from "../../src/model/manifest";

function baseManifest(
  precision: "fp16" | "fp32" = "fp16",
): Record<string, unknown> {
  return {
    format:
      precision === "fp16"
        ? PARAKEET_FP16_PACKAGE_FORMAT
        : PARAKEET_FP32_PACKAGE_FORMAT,
    source: {
      repository: "nvidia/parakeet-tdt-0.6b-v2",
      revision: PARAKEET_SOURCE_REVISION,
      archive: {
        name: PARAKEET_SOURCE_ARCHIVE_NAME,
        byteLength: PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH,
        sha256: PARAKEET_SOURCE_ARCHIVE_SHA256,
      },
      weights: {
        name: PARAKEET_SOURCE_WEIGHTS_NAME,
        byteLength: PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH,
        sha256: PARAKEET_SOURCE_WEIGHTS_SHA256,
      },
      license: "CC-BY-4.0",
    },
    config: {
      sampleRate: 16000,
      windowSamples: 240000,
      windowSeconds: 15,
      windowOverlapSeconds: 2,
      featureBins: 128,
      fftSize: 512,
      windowSize: 400,
      hopSize: 160,
      preemphasis: 0.97,
      logZeroGuard: 2 ** -24,
      normalizationEpsilon: 1e-5,
      subsamplingFactor: 8,
      encoderLayers: 24,
      encoderHiddenSize: 1024,
      intermediateSize: 4096,
      attentionHeads: 8,
      attentionHeadSize: 128,
      convolutionKernelSize: 9,
      decoderHiddenSize: 640,
      decoderLayers: 2,
      vocabularySize: 1025,
      blankTokenId: 1024,
      jointOutputSize: 1030,
      jointStorageOutputSize: 1032,
      durations: [0, 1, 2, 3, 4],
      maxSymbolsPerStep: 10,
      encoderWeightFormat:
        precision === "fp16"
          ? PARAKEET_FP16_ENCODER_WEIGHT_FORMAT
          : PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
    },
    files: [],
    tensors: {},
  };
}

describe("Parakeet fixed package contract", () => {
  it("pins a distinct generated manifest digest for each precision", () => {
    expect(parakeetManifestSha256("fp16")).toBe(
      PARAKEET_FP16_MANIFEST_SHA256,
    );
    expect(parakeetManifestSha256("fp32")).toBe(
      PARAKEET_FP32_MANIFEST_SHA256,
    );
    expect(PARAKEET_FP16_MANIFEST_SHA256).not.toBe(
      PARAKEET_FP32_MANIFEST_SHA256,
    );
  });

  it("preserves the fixed FP16 package contract", () => {
    const manifest = parseParakeetManifest(baseManifest());
    expect(manifest.format).toBe(PARAKEET_FP16_PACKAGE_FORMAT);
    expect(parakeetManifestPrecision(manifest)).toBe("fp16");
    expect(manifest.config.intermediateSize).toBe(4096);
    expect(manifest.config.encoderWeightFormat).toBe(
      PARAKEET_FP16_ENCODER_WEIGHT_FORMAT,
    );
  });

  it("accepts and discriminates the fixed FP32 package contract", () => {
    const manifest = parseParakeetManifest(
      baseManifest("fp32"),
    );
    expect(manifest.format).toBe(PARAKEET_FP32_PACKAGE_FORMAT);
    expect(parakeetManifestPrecision(manifest)).toBe("fp32");
    if (manifest.format !== PARAKEET_FP32_PACKAGE_FORMAT) {
      throw new Error("Expected an FP32 manifest");
    }
    expect(manifest.config.intermediateSize).toBe(4096);
    expect(manifest.config.encoderWeightFormat).toBe(
      PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
    );
  });

  it("rejects cross-paired package and encoder weight formats", () => {
    const fp16WithFp32Weights = baseManifest();
    (
      fp16WithFp32Weights.config as Record<string, unknown>
    ).encoderWeightFormat = PARAKEET_FP32_ENCODER_WEIGHT_FORMAT;
    expect(() =>
      parseParakeetManifest(fp16WithFp32Weights),
    ).toThrow(/fixed encoder package contract/);

    const fp32WithFp16Weights = baseManifest("fp32");
    (
      fp32WithFp16Weights.config as Record<string, unknown>
    ).encoderWeightFormat = PARAKEET_FP16_ENCODER_WEIGHT_FORMAT;
    expect(() =>
      parseParakeetManifest(fp32WithFp16Weights),
    ).toThrow(/fixed encoder package contract/);
  });

  it("rejects float16 tensor storage in an FP32 manifest", () => {
    const manifest = baseManifest("fp32");
    manifest.files = [
      {
        name: "weights.bin",
        byteLength: 256,
        sha256: "0".repeat(64),
      },
    ];
    manifest.tensors = {
      weight: {
        shard: "weights.bin",
        byteOffset: 0,
        byteLength: 4,
        dtype: "float16",
        logicalShape: [2],
        storageShape: [2],
        layout: "test",
      },
    };
    expect(() => parseParakeetManifest(manifest)).toThrow(
      /Invalid packed tensor weight/,
    );

    (
      manifest.tensors as Record<
        string,
        Record<string, unknown>
      >
    ).weight!.dtype = "float32";
    expect(parseParakeetManifest(manifest).format).toBe(
      PARAKEET_FP32_PACKAGE_FORMAT,
    );
  });

  it("rejects any alternate encoder width or unknown format", () => {
    const wrongWidth = baseManifest();
    (wrongWidth.config as Record<string, unknown>).intermediateSize = 2560;
    expect(() => parseParakeetManifest(wrongWidth)).toThrow(
      /fixed encoder package contract/,
    );

    const unknownFormat = baseManifest();
    unknownFormat.format = "parakeet-webgpu-fp64-v1";
    expect(() => parseParakeetManifest(unknownFormat)).toThrow(
      /Expected parakeet-webgpu-v1 or parakeet-webgpu-fp32-v1/,
    );
  });

  it("rejects the former v3 decoder dimensions", () => {
    for (const [field, value] of [
      ["vocabularySize", 8193],
      ["blankTokenId", 8192],
      ["jointOutputSize", 8198],
      ["jointStorageOutputSize", 8200],
    ] as const) {
      const manifest = baseManifest();
      (manifest.config as Record<string, unknown>)[field] = value;
      expect(() => parseParakeetManifest(manifest)).toThrow(
        new RegExp(`Unsupported Parakeet config ${field}=`),
      );
    }
  });

  it("rejects any other fixed v2 source metadata", () => {
    for (const precision of ["fp16", "fp32"] as const) {
      const invalidSources = [
        { repository: "nvidia/parakeet-tdt-0.6b-v3" },
        { revision: "other" },
        {
          archive: {
            name: "other.nemo",
            byteLength: PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH,
            sha256: PARAKEET_SOURCE_ARCHIVE_SHA256,
          },
        },
        {
          archive: {
            name: PARAKEET_SOURCE_ARCHIVE_NAME,
            byteLength: PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH - 1,
            sha256: PARAKEET_SOURCE_ARCHIVE_SHA256,
          },
        },
        {
          archive: {
            name: PARAKEET_SOURCE_ARCHIVE_NAME,
            byteLength: PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH,
            sha256: "0".repeat(64),
          },
        },
        {
          weights: {
            name: "other.ckpt",
            byteLength: PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH,
            sha256: PARAKEET_SOURCE_WEIGHTS_SHA256,
          },
        },
        {
          weights: {
            name: PARAKEET_SOURCE_WEIGHTS_NAME,
            byteLength: PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH - 1,
            sha256: PARAKEET_SOURCE_WEIGHTS_SHA256,
          },
        },
        {
          weights: {
            name: PARAKEET_SOURCE_WEIGHTS_NAME,
            byteLength: PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH,
            sha256: "0".repeat(64),
          },
        },
      ] satisfies readonly Record<string, unknown>[];
      for (const invalidSource of invalidSources) {
        const manifest = baseManifest(precision);
        Object.assign(
          manifest.source as Record<string, unknown>,
          invalidSource,
        );
        expect(() => parseParakeetManifest(manifest)).toThrow(
          /source metadata/,
        );
      }
    }
  });
});
