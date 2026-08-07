import type {
  ParakeetFileRecord,
  ParakeetManifest,
  ParakeetTensorRecord,
} from "./manifest";
import {
  parakeetManifestPrecision,
} from "./manifest";

export const RUNTIME_STORAGE_ALIGNMENT = 256;
export const RUNTIME_BUFFER_LIMIT_BYTES = 128 * 1024 * 1024;
export const ENCODER_LAYER_COUNT = 24;
export const COMMON_TRANSIENT_PALETTE_MATRIX_COUNT = 4;
export const FP16_PACKED_DOWNLOAD_BYTES = 405_063_936;
export const FP16_RETAINED_RUNTIME_BYTES = 404_930_560;
export const FP16_EXPANDED_RUNTIME_BYTES = 0;
export const FP16_RUNTIME_GPU_BYTES =
  FP16_RETAINED_RUNTIME_BYTES + FP16_EXPANDED_RUNTIME_BYTES;
export const FP16_LARGEST_RUNTIME_SHARD_BYTES = 16_999_168;
export const FP32_PACKED_DOWNLOAD_BYTES = 440_517_632;
export const FP32_RETAINED_RUNTIME_BYTES = 440_384_256;
export const FP32_EXPANDED_RUNTIME_BYTES = 0;
export const FP32_RUNTIME_GPU_BYTES =
  FP32_RETAINED_RUNTIME_BYTES + FP32_EXPANDED_RUNTIME_BYTES;
export const FP32_LARGEST_RUNTIME_SHARD_BYTES = 33_558_528;
export const RUNTIME_ROW_MAJOR_F16_LAYOUT =
  "k-by-output-row-major-f16";
export const RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT =
  "n256-k32-tile-major-f16";
export const RUNTIME_ROW_MAJOR_F32_LAYOUT =
  "k-by-output-row-major-f32";
export const RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT =
  "n128-k32-tile-major-f32";
export const RUNTIME_GEMM_TILE_COLUMNS = 256;
export const RUNTIME_GEMM_F32_TILE_COLUMNS = 128;
export const RUNTIME_GEMM_TILE_INNER = 32;
export const PALETTE_EXPANSION_PARAMETER_BYTES = 16;

export type RuntimeBufferId = `source-shard-${string}`;

export interface RuntimeTensorRecord {
  readonly bufferId: RuntimeBufferId;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly dtype: ParakeetTensorRecord["dtype"];
  readonly logicalShape: readonly number[];
  readonly storageShape: readonly number[];
  readonly layout: string;
}

export interface RuntimeTensorPlan {
  readonly name: string;
  readonly sourceRecord: ParakeetTensorRecord;
  readonly runtimeRecord: RuntimeTensorRecord;
}

export interface RuntimeBufferPlan {
  readonly id: RuntimeBufferId;
  readonly label: string;
  readonly byteLength: number;
  readonly tensorCount: number;
}

export type RuntimeShardPlan =
  | {
      readonly file: ParakeetFileRecord;
      readonly hashOnly: true;
    }
  | {
      readonly file: ParakeetFileRecord;
      readonly hashOnly: false;
      readonly runtimeBufferId: RuntimeBufferId;
    };

export interface RuntimePackageMemoryPlan {
  readonly packedDownloadBytes: number;
  readonly retainedRuntimeBytes: number;
  readonly expandedRuntimeBytes: number;
  readonly runtimeGpuBytes: number;
  readonly largestRuntimeShardBytes: number;
  readonly loaderPeakGpuBytes: number;
  readonly runtimeBufferCount: number;
}

export interface RuntimePackagePlan {
  readonly buffers: readonly RuntimeBufferPlan[];
  readonly tensors: ReadonlyMap<string, RuntimeTensorPlan>;
  readonly shards: readonly RuntimeShardPlan[];
  readonly memory: RuntimePackageMemoryPlan;
}

interface MutableBufferPlan {
  readonly id: RuntimeBufferId;
  readonly label: string;
  cursor: number;
  tensorCount: number;
}

interface RuntimePackageContract {
  readonly precision: "fp16" | "fp32";
  readonly paletteDtype: "float16" | "float32";
  readonly paletteElementBytes: 2 | 4;
  readonly packedDownloadBytes: number;
  readonly retainedRuntimeBytes: number;
  readonly decoderShardBytes: number;
  readonly encoderLayerShardBytes: number;
  readonly subsamplingShardBytes: number;
}

const FP16_RUNTIME_PACKAGE_CONTRACT: RuntimePackageContract = {
  precision: "fp16",
  paletteDtype: "float16",
  paletteElementBytes: 2,
  packedDownloadBytes: FP16_PACKED_DOWNLOAD_BYTES,
  retainedRuntimeBytes: FP16_RETAINED_RUNTIME_BYTES,
  decoderShardBytes: FP16_LARGEST_RUNTIME_SHARD_BYTES,
  encoderLayerShardBytes: 16_050_176,
  subsamplingShardBytes: 2_727_168,
};

const FP32_RUNTIME_PACKAGE_CONTRACT: RuntimePackageContract = {
  precision: "fp32",
  paletteDtype: "float32",
  paletteElementBytes: 4,
  packedDownloadBytes: FP32_PACKED_DOWNLOAD_BYTES,
  retainedRuntimeBytes: FP32_RETAINED_RUNTIME_BYTES,
  decoderShardBytes: FP32_LARGEST_RUNTIME_SHARD_BYTES,
  encoderLayerShardBytes: 16_836_864,
  subsamplingShardBytes: 2_740_992,
};

const PREPROCESSOR_TENSORS = new Set([
  "preprocessor.window",
  "preprocessor.mel_filters",
]);
const COMMON_TRANSIENT_PALETTE_TENSORS = new Set([
  "encoder.subsampling.layers.3.weight",
  "encoder.subsampling.layers.6.weight",
  "encoder.subsampling.linear.weight",
  "encoder_projector.weight",
]);
const PALETTE_LAYOUT = "output-group-by-palette-lut";

export function planRuntimePackage(
  manifest: ParakeetManifest,
): RuntimePackagePlan {
  const contract = runtimePackageContract(manifest);
  const binaryFiles = manifest.files.filter((file) =>
    file.name.endsWith(".bin"),
  );
  const binaryFileMap = new Map(
    binaryFiles.map((file) => [file.name, file]),
  );
  const sourceBuffers = new Map<string, MutableBufferPlan>(
    binaryFiles
      .filter((file) => file.name !== "preprocessor.bin")
      .map((file) => [
        file.name,
        {
          id: sourceBufferId(file.name),
          label: `parakeet-source-${file.name}`,
          cursor: file.byteLength,
          tensorCount: 0,
        },
      ]),
  );
  const runtimeTensors = new Map<string, RuntimeTensorPlan>();
  const packedPaletteWeights: RuntimeTensorPlan[] = [];
  const consumedPalettes = new Set<string>();
  let skippedPreprocessorTensorCount = 0;

  for (const [name, sourceRecord] of Object.entries(
    manifest.tensors,
  ).sort(([left], [right]) => left.localeCompare(right))) {
    requireSourceRange(
      name,
      sourceRecord,
      binaryFileMap,
    );
    requirePackageTensorPrecision(
      name,
      sourceRecord,
      contract,
    );
    if (PREPROCESSOR_TENSORS.has(name)) {
      if (sourceRecord.shard !== "preprocessor.bin") {
        throw new Error(
          `${name} must remain sourced from preprocessor.bin`,
        );
      }
      skippedPreprocessorTensorCount += 1;
      continue;
    }
    const sourceBuffer = sourceBuffers.get(sourceRecord.shard);
    if (sourceBuffer === undefined) {
      throw new Error(
        `${name} cannot use GPU-unresident ${sourceRecord.shard}`,
      );
    }
    const paletteBits = paletteBitsForLayout(
      sourceRecord.layout,
    );
    if (paletteBits !== undefined) {
      const paletteName = `${name}.palette`;
      const paletteRecord = manifest.tensors[paletteName];
      if (paletteRecord === undefined) {
        throw new Error(`${name} is missing ${paletteName}`);
      }
      requireSourceRange(
        paletteName,
        paletteRecord,
        binaryFileMap,
      );
      validatePalettePair(
        name,
        sourceRecord,
        paletteName,
        paletteRecord,
        paletteBits,
        contract,
      );
      const packed = planResidentSourceTensor(
        name,
        sourceRecord,
        sourceBuffer,
      );
      addRuntimeTensor(runtimeTensors, packed);
      packedPaletteWeights.push(packed);
      consumedPalettes.add(paletteName);
      continue;
    }
    const resident = planResidentSourceTensor(
      name,
      sourceRecord,
      sourceBuffer,
    );
    addRuntimeTensor(runtimeTensors, resident);
  }

  const packagedPalettes = Object.entries(manifest.tensors)
    .filter(([, record]) => record.layout === PALETTE_LAYOUT)
    .map(([name]) => name);
  const unconsumedPalettes = packagedPalettes.filter(
    (name) => !consumedPalettes.has(name),
  );
  if (unconsumedPalettes.length > 0) {
    throw new Error(
      "Unowned palette LUT tensors: " +
        unconsumedPalettes.join(","),
    );
  }
  if (
    skippedPreprocessorTensorCount !==
    PREPROCESSOR_TENSORS.size
  ) {
    throw new Error(
      "The package must contain exactly two Wasm-owned preprocessor tensors",
    );
  }

  requireFixedInventory(
    sourceBuffers,
    packedPaletteWeights,
    contract,
  );
  const retainedRuntimeBytes = sum(
    [...sourceBuffers.values()].map(
      (buffer) => alignRuntimeOffset(buffer.cursor),
    ),
  );
  if (
    retainedRuntimeBytes !==
    contract.retainedRuntimeBytes
  ) {
    throw new Error(
      `Retained ${contract.precision} runtime package changed: ` +
        `${retainedRuntimeBytes} bytes`,
    );
  }
  const retainedTensorCount = runtimeTensors.size;
  if (retainedTensorCount !== 795) {
    throw new Error(
      `Retained runtime tensor count changed: ${retainedTensorCount}`,
    );
  }
  const mutableBuffers = [...sourceBuffers.values()];
  const buffers = mutableBuffers.map(finalizeBufferPlan);
  for (const buffer of buffers) requireRuntimeBufferSupport(buffer);

  const shardPlans = [...binaryFiles]
    .sort(compareRuntimeShardLoadOrder)
    .map((file) => {
      if (file.name === "preprocessor.bin") {
        return {
          file,
          hashOnly: true,
        } as const;
      }
      const sourceBuffer = sourceBuffers.get(file.name);
      if (sourceBuffer === undefined) {
        throw new Error(
          `${file.name} has no resident source buffer`,
        );
      }
      return {
        file,
        runtimeBufferId: sourceBuffer.id,
        hashOnly: false,
      };
    });
  requireWholeRuntimeShardInventory(shardPlans, buffers);
  const packedDownloadBytes = sum(
    binaryFiles.map((file) => file.byteLength),
  );
  const expandedRuntimeBytes = 0;
  const runtimeGpuBytes =
    retainedRuntimeBytes + expandedRuntimeBytes;
  const largestRuntimeShardBytes = Math.max(
    ...shardPlans
      .filter((shard) => !shard.hashOnly)
      .map((shard) => shard.file.byteLength),
  );
  const loaderPeakGpuBytes = runtimeGpuBytes;
  const memory: RuntimePackageMemoryPlan = {
    packedDownloadBytes,
    retainedRuntimeBytes,
    expandedRuntimeBytes,
    runtimeGpuBytes,
    largestRuntimeShardBytes,
    loaderPeakGpuBytes,
    runtimeBufferCount: buffers.length,
  };
  requireFixedMemory(memory, contract);
  return {
    buffers,
    tensors: runtimeTensors,
    shards: shardPlans,
    memory,
  };
}

function runtimePackageContract(
  manifest: ParakeetManifest,
): RuntimePackageContract {
  const precision = parakeetManifestPrecision(manifest);
  if (precision === "fp16") {
    return FP16_RUNTIME_PACKAGE_CONTRACT;
  }
  return FP32_RUNTIME_PACKAGE_CONTRACT;
}

export function paletteBitsForLayout(
  layout: string,
): 5 | 6 | undefined {
  if (
    layout ===
    "k-by-output-row-major-palette5-lsb-u32"
  ) {
    return 5;
  }
  if (
    layout ===
    "k-by-output-row-major-palette6-lsb-u32"
  ) {
    return 6;
  }
  return undefined;
}

export function alignRuntimeOffset(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("Runtime tensor offset must be nonnegative");
  }
  return (
    Math.ceil(value / RUNTIME_STORAGE_ALIGNMENT) *
    RUNTIME_STORAGE_ALIGNMENT
  );
}

export function runtimeGemmTileMajorStorageShape(
  inner: number,
  columns: number,
): readonly [number, number, number, number] {
  return runtimeGemmTileMajorStorageShapeForColumns(
    inner,
    columns,
    RUNTIME_GEMM_TILE_COLUMNS,
  );
}

export function runtimeGemmTileMajorF32StorageShape(
  inner: number,
  columns: number,
): readonly [number, number, number, number] {
  return runtimeGemmTileMajorStorageShapeForColumns(
    inner,
    columns,
    RUNTIME_GEMM_F32_TILE_COLUMNS,
  );
}

function runtimeGemmTileMajorStorageShapeForColumns(
  inner: number,
  columns: number,
  tileColumns: 128 | 256,
): readonly [number, number, number, number] {
  if (
    !Number.isSafeInteger(inner) ||
    inner <= 0 ||
    inner % RUNTIME_GEMM_TILE_INNER !== 0 ||
    !Number.isSafeInteger(columns) ||
    columns <= 0 ||
    columns % tileColumns !== 0
  ) {
    throw new RangeError(
      "Tile-major runtime weights require K%32=0 and " +
        `N%${tileColumns}=0`,
    );
  }
  return [
    columns / tileColumns,
    inner / RUNTIME_GEMM_TILE_INNER,
    RUNTIME_GEMM_TILE_INNER,
    tileColumns,
  ];
}

export function runtimeGemmTileMajorScalarIndex(
  scalarIndex: number,
  inner: number,
  columns: number,
): number {
  return runtimeGemmTileMajorScalarIndexForColumns(
    scalarIndex,
    inner,
    columns,
    RUNTIME_GEMM_TILE_COLUMNS,
  );
}

export function runtimeGemmTileMajorF32ScalarIndex(
  scalarIndex: number,
  inner: number,
  columns: number,
): number {
  return runtimeGemmTileMajorScalarIndexForColumns(
    scalarIndex,
    inner,
    columns,
    RUNTIME_GEMM_F32_TILE_COLUMNS,
  );
}

function runtimeGemmTileMajorScalarIndexForColumns(
  scalarIndex: number,
  inner: number,
  columns: number,
  tileColumns: 128 | 256,
): number {
  runtimeGemmTileMajorStorageShapeForColumns(
    inner,
    columns,
    tileColumns,
  );
  const scalarCount = inner * columns;
  if (
    !Number.isSafeInteger(scalarCount) ||
    !Number.isSafeInteger(scalarIndex) ||
    scalarIndex < 0 ||
    scalarIndex >= scalarCount
  ) {
    throw new RangeError(
      "Tile-major scalar index exceeds the logical KxN matrix",
    );
  }
  const k = Math.floor(scalarIndex / columns);
  const n = scalarIndex % columns;
  const nTile = Math.floor(n / tileColumns);
  const kTile = Math.floor(k / RUNTIME_GEMM_TILE_INNER);
  const kTiles = inner / RUNTIME_GEMM_TILE_INNER;
  const kInTile = k % RUNTIME_GEMM_TILE_INNER;
  const nInTile = n % tileColumns;
  return (
    (
      (nTile * kTiles + kTile) * RUNTIME_GEMM_TILE_INNER +
      kInTile
    ) * tileColumns +
    nInTile
  );
}

function planResidentSourceTensor(
  name: string,
  sourceRecord: ParakeetTensorRecord,
  target: MutableBufferPlan,
): RuntimeTensorPlan {
  requireCopyAlignment(name, sourceRecord);
  target.tensorCount += 1;
  return {
    name,
    sourceRecord,
    runtimeRecord: {
      bufferId: target.id,
      byteOffset: sourceRecord.byteOffset,
      byteLength: sourceRecord.byteLength,
      dtype: sourceRecord.dtype,
      logicalShape: sourceRecord.logicalShape,
      storageShape: sourceRecord.storageShape,
      layout: sourceRecord.layout,
    },
  };
}

function requirePackageTensorPrecision(
  name: string,
  sourceRecord: ParakeetTensorRecord,
  contract: RuntimePackageContract,
): void {
  if (
    contract.precision === "fp32" &&
    sourceRecord.dtype === "float16"
  ) {
    throw new Error(
      `${name} cannot retain float16 storage in the FP32 package`,
    );
  }
}

function validatePalettePair(
  name: string,
  sourceRecord: ParakeetTensorRecord,
  paletteName: string,
  paletteRecord: ParakeetTensorRecord,
  paletteBits: 5 | 6,
  contract: RuntimePackageContract,
): number {
  if (
    sourceRecord.dtype !== "uint32" ||
    sourceRecord.logicalShape.length !== 2
  ) {
    throw new Error(`${name} has invalid palette indices`);
  }
  const inner = sourceRecord.logicalShape[0]!;
  const columns = sourceRecord.logicalShape[1]!;
  const codesPerGroup = paletteBits === 5 ? 32 : 16;
  const wordsPerGroup = paletteBits === 5 ? 5 : 3;
  const scalarCount = inner * columns;
  if (
    !Number.isSafeInteger(scalarCount) ||
    columns % codesPerGroup !== 0 ||
    scalarCount % codesPerGroup !== 0
  ) {
    throw new Error(`${name} has unaligned palette codes`);
  }
  const packedGroups = scalarCount / codesPerGroup;
  if (
    sourceRecord.storageShape.join(",") !==
      `${packedGroups},${wordsPerGroup}` ||
    sourceRecord.byteLength !== packedGroups * wordsPerGroup * 4
  ) {
    throw new Error(`${name} has invalid packed palette storage`);
  }
  const paletteEntries = 1 << paletteBits;
  if (
    paletteRecord.shard !== sourceRecord.shard ||
    paletteRecord.dtype !== contract.paletteDtype ||
    paletteRecord.layout !== PALETTE_LAYOUT ||
    paletteRecord.logicalShape.length !== 2 ||
    paletteRecord.logicalShape[1] !== paletteEntries ||
    paletteRecord.storageShape.join(",") !==
      paletteRecord.logicalShape.join(",")
  ) {
    throw new Error(`${paletteName} has invalid palette storage`);
  }
  const paletteGroups = paletteRecord.logicalShape[0]!;
  const paletteGroupColumns = columns / paletteGroups;
  if (
    columns % paletteGroups !== 0 ||
    paletteGroupColumns % codesPerGroup !== 0 ||
    paletteRecord.byteLength !==
      paletteGroups *
        paletteEntries *
        contract.paletteElementBytes
  ) {
    throw new Error(`${paletteName} does not partition output columns`);
  }
  return paletteGroupColumns;
}

function sourceBufferId(
  fileName: string,
): `source-shard-${string}` {
  return `source-shard-${fileName}`;
}

function addRuntimeTensor(
  tensors: Map<string, RuntimeTensorPlan>,
  tensor: RuntimeTensorPlan,
): void {
  if (tensors.has(tensor.name)) {
    throw new Error(`Duplicate runtime tensor ${tensor.name}`);
  }
  tensors.set(tensor.name, tensor);
}

function requireSourceRange(
  name: string,
  record: ParakeetTensorRecord,
  files: ReadonlyMap<string, ParakeetFileRecord>,
): void {
  const file = files.get(record.shard);
  if (file === undefined) {
    throw new Error(`${name} is not stored in a binary shard`);
  }
  if (
    record.byteOffset % RUNTIME_STORAGE_ALIGNMENT !== 0 ||
    record.byteLength % 4 !== 0 ||
    record.byteOffset + record.byteLength > file.byteLength
  ) {
    throw new Error(`${name} exceeds its aligned source shard`);
  }
}

function requireCopyAlignment(
  name: string,
  record: ParakeetTensorRecord,
): void {
  if (
    record.byteOffset % 4 !== 0 ||
    record.byteLength % 4 !== 0
  ) {
    throw new Error(`${name} cannot be copied as GPU words`);
  }
}

function requireFixedInventory(
  sourceBuffers: ReadonlyMap<string, MutableBufferPlan>,
  packedPaletteWeights: readonly RuntimeTensorPlan[],
  contract: RuntimePackageContract,
): void {
  if (sourceBuffers.size !== 26) {
    throw new Error(
      `Resident source shard count changed: ${sourceBuffers.size}`,
    );
  }
  const decoder = sourceBuffers.get("decoder.bin");
  const subsampling = sourceBuffers.get(
    "encoder-subsampling.bin",
  );
  if (
    decoder === undefined ||
    alignRuntimeOffset(decoder.cursor) !==
      contract.decoderShardBytes ||
    subsampling === undefined ||
    alignRuntimeOffset(subsampling.cursor) !==
      contract.subsamplingShardBytes
  ) {
    throw new Error(
      `Resident ${contract.precision} common-source shard inventory changed`,
    );
  }
  for (let layer = 0; layer < ENCODER_LAYER_COUNT; layer += 1) {
    const name =
      `encoder-layer-${String(layer).padStart(2, "0")}.bin`;
    const buffer = sourceBuffers.get(name);
    if (
      buffer === undefined ||
      alignRuntimeOffset(buffer.cursor) !==
        contract.encoderLayerShardBytes
    ) {
      throw new Error(
        `Resident ${contract.precision} encoder-layer shard ` +
          `inventory changed: ${name}`,
      );
    }
  }
  if (
    packedPaletteWeights.length !==
      ENCODER_LAYER_COUNT * 8 +
        COMMON_TRANSIENT_PALETTE_MATRIX_COUNT ||
    packedPaletteWeights.some(
      (item) =>
        paletteBitsForLayout(item.runtimeRecord.layout) ===
          undefined ||
        item.runtimeRecord.dtype !== "uint32" ||
        (
          item.name.startsWith("encoder.layers.")
            ? !item.runtimeRecord.bufferId.startsWith(
                "source-shard-encoder-layer-",
              )
            : !COMMON_TRANSIENT_PALETTE_TENSORS.has(item.name)
        ),
    )
  ) {
    throw new Error(
      "Packed encoder-layer matrix inventory changed: " +
        packedPaletteWeights.length,
    );
  }
  const palette5Count = packedPaletteWeights.filter(
    (item) =>
      paletteBitsForLayout(item.runtimeRecord.layout) === 5,
  ).length;
  const palette6Count =
    packedPaletteWeights.length - palette5Count;
  if (palette5Count !== 172 || palette6Count !== 24) {
    throw new Error(
      "Packed palette topology changed: " +
        `${palette5Count} palette5, ${palette6Count} palette6`,
    );
  }
  const layerCounts = new Array<number>(
    ENCODER_LAYER_COUNT,
  ).fill(0);
  for (const item of packedPaletteWeights) {
    if (COMMON_TRANSIENT_PALETTE_TENSORS.has(item.name)) {
      continue;
    }
    const match = item.name.match(
      /^encoder\.layers\.(\d+)\./,
    );
    if (match === null) {
      throw new Error(
        `Packed matrix ${item.name} has no encoder layer`,
      );
    }
    const layer = Number(match[1]);
    if (layerCounts[layer] === undefined) {
      throw new Error(
        `Packed matrix ${item.name} has invalid encoder layer`,
      );
    }
    layerCounts[layer] += 1;
  }
  if (layerCounts.some((count) => count !== 8)) {
    throw new Error(
      `Packed encoder layer inventories changed: ${layerCounts.join(",")}`,
    );
  }
  const commonPacked = packedPaletteWeights.filter((item) =>
    COMMON_TRANSIENT_PALETTE_TENSORS.has(item.name)
  );
  if (commonPacked.length !== COMMON_TRANSIENT_PALETTE_MATRIX_COUNT) {
    throw new Error(
      `Common packed matrix inventory changed: ${commonPacked.length}`,
    );
  }
}

function finalizeBufferPlan(
  buffer: MutableBufferPlan,
): RuntimeBufferPlan {
  return {
    id: buffer.id,
    label: buffer.label,
    byteLength: alignRuntimeOffset(buffer.cursor),
    tensorCount: buffer.tensorCount,
  };
}

function requireRuntimeBufferSupport(
  buffer: RuntimeBufferPlan,
): void {
  if (
    buffer.byteLength <= 0 ||
    buffer.byteLength > RUNTIME_BUFFER_LIMIT_BYTES
  ) {
    throw new Error(
      `${buffer.id} exceeds the fixed 128 MiB runtime-buffer boundary`,
    );
  }
}

function requireFixedMemory(
  memory: RuntimePackageMemoryPlan,
  contract: RuntimePackageContract,
): void {
  const expected: Readonly<
    Partial<Record<keyof RuntimePackageMemoryPlan, number>>
  > = {
    packedDownloadBytes: contract.packedDownloadBytes,
    retainedRuntimeBytes: contract.retainedRuntimeBytes,
    expandedRuntimeBytes: 0,
    runtimeGpuBytes: contract.retainedRuntimeBytes,
    largestRuntimeShardBytes: contract.decoderShardBytes,
    loaderPeakGpuBytes: contract.retainedRuntimeBytes,
    runtimeBufferCount: 26,
  };
  for (const [name, value] of Object.entries(expected)) {
    if (
      memory[name as keyof RuntimePackageMemoryPlan] !== value
    ) {
      throw new Error(
        `Runtime ${contract.precision} package memory ${name} changed: ` +
          String(memory[name as keyof RuntimePackageMemoryPlan]),
      );
    }
  }
}

function compareRuntimeShardLoadOrder(
  left: ParakeetFileRecord,
  right: ParakeetFileRecord,
): number {
  return (
    runtimeShardLoadRank(left.name) -
      runtimeShardLoadRank(right.name) ||
    left.name.localeCompare(right.name)
  );
}

function runtimeShardLoadRank(name: string): number {
  if (name === "decoder.bin") return 0;
  if (name === "encoder-subsampling.bin") return 1;
  if (/^encoder-layer-\d{2}\.bin$/.test(name)) return 2;
  return 3;
}

function requireWholeRuntimeShardInventory(
  shards: readonly RuntimeShardPlan[],
  buffers: readonly RuntimeBufferPlan[],
): void {
  const bufferPlans = new Map(
    buffers.map((buffer) => [buffer.id, buffer]),
  );
  const runtimeBufferIds = new Set<RuntimeBufferId>();
  let hashOnlyCount = 0;
  for (const shard of shards) {
    if (shard.hashOnly) {
      hashOnlyCount += 1;
      if (shard.file.name !== "preprocessor.bin") {
        throw new Error(
          `${shard.file.name} cannot be a hash-only runtime shard`,
        );
      }
      continue;
    }
    const target = bufferPlans.get(shard.runtimeBufferId);
    if (
      target === undefined ||
      target.byteLength !== shard.file.byteLength ||
      runtimeBufferIds.has(shard.runtimeBufferId)
    ) {
      throw new Error(
        `${shard.file.name} is not a unique whole runtime shard`,
      );
    }
    runtimeBufferIds.add(shard.runtimeBufferId);
  }
  if (
    hashOnlyCount !== 1 ||
    runtimeBufferIds.size !== ENCODER_LAYER_COUNT + 2 ||
    runtimeBufferIds.size !== buffers.length
  ) {
    throw new Error(
      "The fixed package must contain one hash-only preprocessor shard and 26 whole runtime shards",
    );
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
