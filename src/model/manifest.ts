export const PARAKEET_FP16_PACKAGE_FORMAT = "parakeet-webgpu-v1";
export const PARAKEET_FP32_PACKAGE_FORMAT =
  "parakeet-webgpu-fp32-v1";
export const PARAKEET_FP16_ENCODER_WEIGHT_FORMAT =
  "mixed-palette5-o-proj-palette6-fp16-lut";
export const PARAKEET_FP32_ENCODER_WEIGHT_FORMAT =
  "mixed-palette5-o-proj-palette6-fp32-lut";
export const PARAKEET_FP16_MANIFEST_SHA256 =
  "11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65";
export const PARAKEET_FP32_MANIFEST_SHA256 =
  "28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b";
export const PARAKEET_FFN_SOURCE_WIDTH = 4096;
export const PARAKEET_SOURCE_REVISION =
  "ae9ad07059c7c739ffaf932226a8fe64ae2620b0";
export const PARAKEET_SOURCE_ARCHIVE_NAME =
  "parakeet-tdt-0.6b-v2.nemo";
export const PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH = 2_472_222_720;
export const PARAKEET_SOURCE_ARCHIVE_SHA256 =
  "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c";
export const PARAKEET_SOURCE_WEIGHTS_NAME = "model_weights.ckpt";
export const PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH = 2_471_920_514;
export const PARAKEET_SOURCE_WEIGHTS_SHA256 =
  "fb1cbe2765fb80c02c0f874b6ea2c2bd24048c3618c092584fd086c6366e48a9";

export type ParakeetModelPrecision = "fp16" | "fp32";

export interface ParakeetTensorRecord {
  readonly shard: string;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly dtype: "float16" | "float32" | "uint32";
  readonly logicalShape: readonly number[];
  readonly storageShape: readonly number[];
  readonly layout: string;
}

export interface ParakeetFileRecord {
  readonly name: string;
  readonly byteLength: number;
  readonly sha256: string;
}

interface ParakeetCommonModelConfig {
  readonly sampleRate: 16000;
  readonly windowSamples: 240000;
  readonly windowSeconds: 15;
  readonly windowOverlapSeconds: 2;
  readonly featureBins: 128;
  readonly fftSize: 512;
  readonly windowSize: 400;
  readonly hopSize: 160;
  readonly preemphasis: number;
  readonly logZeroGuard: number;
  readonly normalizationEpsilon: number;
  readonly subsamplingFactor: 8;
  readonly encoderLayers: 24;
  readonly encoderHiddenSize: 1024;
  readonly attentionHeads: 8;
  readonly attentionHeadSize: 128;
  readonly convolutionKernelSize: 9;
  readonly decoderHiddenSize: 640;
  readonly decoderLayers: 2;
  readonly vocabularySize: 1025;
  readonly blankTokenId: 1024;
  readonly jointOutputSize: 1030;
  readonly jointStorageOutputSize: 1032;
  readonly durations: readonly [0, 1, 2, 3, 4];
  readonly maxSymbolsPerStep: 10;
}

export interface ParakeetFp16ModelConfig
  extends ParakeetCommonModelConfig {
  readonly intermediateSize: typeof PARAKEET_FFN_SOURCE_WIDTH;
  readonly encoderWeightFormat:
    typeof PARAKEET_FP16_ENCODER_WEIGHT_FORMAT;
}

export interface ParakeetFp32ModelConfig
  extends ParakeetCommonModelConfig {
  readonly intermediateSize: typeof PARAKEET_FFN_SOURCE_WIDTH;
  readonly encoderWeightFormat:
    typeof PARAKEET_FP32_ENCODER_WEIGHT_FORMAT;
}

interface ParakeetManifestBase {
  readonly source: {
    readonly repository: "nvidia/parakeet-tdt-0.6b-v2";
    readonly revision: typeof PARAKEET_SOURCE_REVISION;
    readonly archive: {
      readonly name: typeof PARAKEET_SOURCE_ARCHIVE_NAME;
      readonly byteLength: typeof PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH;
      readonly sha256: typeof PARAKEET_SOURCE_ARCHIVE_SHA256;
    };
    readonly weights: {
      readonly name: typeof PARAKEET_SOURCE_WEIGHTS_NAME;
      readonly byteLength: typeof PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH;
      readonly sha256: typeof PARAKEET_SOURCE_WEIGHTS_SHA256;
    };
    readonly license: "CC-BY-4.0";
  };
  readonly files: readonly ParakeetFileRecord[];
  readonly tensors: Readonly<Record<string, ParakeetTensorRecord>>;
}

export interface ParakeetFp16Manifest extends ParakeetManifestBase {
  readonly format: typeof PARAKEET_FP16_PACKAGE_FORMAT;
  readonly config: ParakeetFp16ModelConfig;
}

export interface ParakeetFp32Manifest extends ParakeetManifestBase {
  readonly format: typeof PARAKEET_FP32_PACKAGE_FORMAT;
  readonly config: ParakeetFp32ModelConfig;
}

export type ParakeetManifest =
  | ParakeetFp16Manifest
  | ParakeetFp32Manifest;

export function parakeetManifestPrecision(
  manifest: ParakeetManifest,
): ParakeetModelPrecision {
  return manifest.format === PARAKEET_FP16_PACKAGE_FORMAT
    ? "fp16"
    : "fp32";
}

export function parakeetManifestSha256(
  precision: ParakeetModelPrecision,
): string {
  return precision === "fp16"
    ? PARAKEET_FP16_MANIFEST_SHA256
    : PARAKEET_FP32_MANIFEST_SHA256;
}

export function parseParakeetManifest(value: unknown): ParakeetManifest {
  if (
    !isObject(value) ||
    (
      value.format !== PARAKEET_FP16_PACKAGE_FORMAT &&
      value.format !== PARAKEET_FP32_PACKAGE_FORMAT
    )
  ) {
    throw new Error(
      `Expected ${PARAKEET_FP16_PACKAGE_FORMAT} or ` +
        `${PARAKEET_FP32_PACKAGE_FORMAT} model manifest`,
    );
  }
  const expectedEncoderWeightFormat =
    value.format === PARAKEET_FP16_PACKAGE_FORMAT
      ? PARAKEET_FP16_ENCODER_WEIGHT_FORMAT
      : PARAKEET_FP32_ENCODER_WEIGHT_FORMAT;
  if (
    !isObject(value.source) ||
    value.source.repository !== "nvidia/parakeet-tdt-0.6b-v2" ||
    value.source.revision !== PARAKEET_SOURCE_REVISION ||
    !isObject(value.source.archive) ||
    value.source.archive.name !== PARAKEET_SOURCE_ARCHIVE_NAME ||
    value.source.archive.byteLength !==
      PARAKEET_SOURCE_ARCHIVE_BYTE_LENGTH ||
    value.source.archive.sha256 !== PARAKEET_SOURCE_ARCHIVE_SHA256 ||
    !isObject(value.source.weights) ||
    value.source.weights.name !== PARAKEET_SOURCE_WEIGHTS_NAME ||
    value.source.weights.byteLength !==
      PARAKEET_SOURCE_WEIGHTS_BYTE_LENGTH ||
    value.source.weights.sha256 !== PARAKEET_SOURCE_WEIGHTS_SHA256 ||
    value.source.license !== "CC-BY-4.0"
  ) {
    throw new Error("Invalid Parakeet source metadata");
  }
  if (!isObject(value.config)) throw new Error("Invalid Parakeet model config");
  requireFixedConfig(
    value.config,
    expectedEncoderWeightFormat,
  );
  if (!Array.isArray(value.files) || !isObject(value.tensors)) {
    throw new Error("Invalid Parakeet package tables");
  }

  for (const file of value.files) {
    if (
      !isObject(file) ||
      typeof file.name !== "string" ||
      !isPositiveInteger(file.byteLength) ||
      typeof file.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(file.sha256)
    ) {
      throw new Error("Invalid Parakeet file record");
    }
  }
  const knownFiles = new Set(value.files.map((file) => file.name));
  for (const [name, tensor] of Object.entries(value.tensors)) {
    if (
      !isObject(tensor) ||
      typeof tensor.shard !== "string" ||
      !knownFiles.has(tensor.shard) ||
      !isNonnegativeInteger(tensor.byteOffset) ||
      tensor.byteOffset % 256 !== 0 ||
      !isPositiveInteger(tensor.byteLength) ||
      (tensor.dtype !== "float16" &&
        tensor.dtype !== "float32" &&
        tensor.dtype !== "uint32") ||
      (
        value.format === PARAKEET_FP32_PACKAGE_FORMAT &&
        tensor.dtype === "float16"
      ) ||
      !isShape(tensor.logicalShape) ||
      !isShape(tensor.storageShape) ||
      typeof tensor.layout !== "string"
    ) {
      throw new Error(`Invalid packed tensor ${name}`);
    }
  }
  return value as unknown as ParakeetManifest;
}

function requireFixedConfig(
  config: Record<string, unknown>,
  expectedEncoderWeightFormat:
    | typeof PARAKEET_FP16_ENCODER_WEIGHT_FORMAT
    | typeof PARAKEET_FP32_ENCODER_WEIGHT_FORMAT,
): void {
  const fixed: Readonly<Record<string, number>> = {
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
    attentionHeads: 8,
    attentionHeadSize: 128,
    convolutionKernelSize: 9,
    decoderHiddenSize: 640,
    decoderLayers: 2,
    vocabularySize: 1025,
    blankTokenId: 1024,
    jointOutputSize: 1030,
    jointStorageOutputSize: 1032,
    maxSymbolsPerStep: 10,
  };
  for (const [name, expected] of Object.entries(fixed)) {
    if (config[name] !== expected) {
      throw new Error(`Unsupported Parakeet config ${name}=${String(config[name])}`);
    }
  }
  if (
    !Array.isArray(config.durations) ||
    config.durations.join(",") !== "0,1,2,3,4"
  ) {
    throw new Error("Unsupported TDT duration table");
  }
  if (
    config.encoderWeightFormat !== expectedEncoderWeightFormat ||
    config.intermediateSize !== PARAKEET_FFN_SOURCE_WIDTH
  ) {
    throw new Error(
      "Unsupported Parakeet fixed encoder package contract",
    );
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isShape(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(isPositiveInteger);
}
