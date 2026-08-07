/// <reference types="@webgpu/types" />

import { IncrementalSha256 } from "./sha256";
import {
  parakeetManifestSha256,
  parakeetManifestPrecision,
  parseParakeetManifest,
  type ParakeetFileRecord,
  type ParakeetManifest,
  type ParakeetModelPrecision,
  type ParakeetTensorRecord,
} from "./manifest";
import {
  planRuntimePackage,
  type RuntimeBufferId,
  type RuntimePackageMemoryPlan,
  type RuntimePackagePlan,
  type RuntimeShardPlan,
  type RuntimeTensorRecord,
  type RuntimeTensorPlan,
} from "./runtime-plan";

interface PackageLoadProgress {
  readonly phase: "manifest" | "weights" | "tokenizer" | "ready";
  readonly file?: string;
  readonly loadedBytes: number;
  readonly totalBytes: number;
}

interface ParakeetPackageLoadOptions {
  readonly fetch?: typeof fetch;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: PackageLoadProgress) => void;
  /** Register the selected package inventory before any child asset fetch. */
  readonly onManifest?: (
    manifest: ParakeetManifest,
  ) => void | Promise<void>;
  /**
   * Capability-selected precision. This pins both the manifest precision and
   * the exact generated manifest digest before any model shard is requested.
   */
  readonly expectedPrecision: ParakeetModelPrecision;
}

export interface GpuTensor {
  readonly sourceRecord: ParakeetTensorRecord;
  readonly runtimeRecord: RuntimeTensorRecord;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

export class ParakeetGpuPackage {
  readonly memory: RuntimePackageMemoryPlan;
  readonly precision: ParakeetModelPrecision;
  readonly manifestUrl: string;
  readonly manifestSha256: string;
  readonly manifestListedBytes: number;
  private manifestValue: ParakeetManifest | undefined;
  private tokenizerValue: unknown;
  private tensorPlans: ReadonlyMap<string, RuntimeTensorPlan> | undefined;
  private destroyed = false;

  private constructor(
    manifest: ParakeetManifest,
    tensorPlans: ReadonlyMap<string, RuntimeTensorPlan>,
    private readonly buffers: Map<RuntimeBufferId, GPUBuffer>,
    tokenizer: unknown,
    memory: RuntimePackageMemoryPlan,
    provenance: {
      readonly manifestUrl: string;
      readonly manifestSha256: string;
    },
  ) {
    this.manifestValue = manifest;
    this.tokenizerValue = tokenizer;
    this.tensorPlans = tensorPlans;
    this.memory = memory;
    this.precision = parakeetManifestPrecision(manifest);
    this.manifestUrl = provenance.manifestUrl;
    this.manifestSha256 = provenance.manifestSha256;
    this.manifestListedBytes = manifest.files.reduce(
      (total, file) => total + file.byteLength,
      0,
    );
  }

  get manifest(): ParakeetManifest {
    if (this.manifestValue === undefined) {
      throw new Error("Parakeet model initialization metadata was released");
    }
    return this.manifestValue;
  }

  get tokenizer(): unknown {
    if (this.tokenizerValue === undefined) {
      throw new Error("Parakeet tokenizer metadata was released");
    }
    return this.tokenizerValue;
  }

  static async load(
    device: GPUDevice,
    manifestUrl: string,
    options: ParakeetPackageLoadOptions,
  ): Promise<ParakeetGpuPackage> {
    const fetchAsset = options.fetch ?? fetch;
    const fetchInit: RequestInit | undefined =
      options.signal === undefined
        ? undefined
        : { signal: options.signal };
    throwIfAborted(options.signal);
    options.onProgress?.({
      phase: "manifest",
      loadedBytes: 0,
      totalBytes: 0,
    });
    const manifestResponse = await fetchAsset(
      manifestUrl,
      fetchInit,
    );
    if (!manifestResponse.ok) {
      throw new Error(
        `Model manifest request failed (${manifestResponse.status} ` +
          `${manifestResponse.statusText})`,
      );
    }
    const manifestBytes = new Uint8Array(
      await manifestResponse.arrayBuffer(),
    );
    const manifestSha256 = new IncrementalSha256()
      .update(manifestBytes)
      .digestHex();
    const expectedManifestSha256 = parakeetManifestSha256(
      options.expectedPrecision,
    );
    if (manifestSha256 !== expectedManifestSha256) {
      throw new Error(
        `Selected ${options.expectedPrecision} model manifest SHA-256 ` +
          `changed: ${manifestSha256}`,
      );
    }
    const manifest = parseParakeetManifest(
      JSON.parse(new TextDecoder().decode(manifestBytes)) as unknown,
    );
    const precision = parakeetManifestPrecision(manifest);
    if (precision !== options.expectedPrecision) {
      throw new Error(
        `Selected ${options.expectedPrecision} execution but ` +
          `received a ${precision} model package`,
      );
    }
    await options.onManifest?.(manifest);
    throwIfAborted(options.signal);
    const plan = planRuntimePackage(manifest);
    requirePlanSupport(device, plan);
    const loaded = await loadRuntimeAssets(
      device,
      manifestUrl,
      manifest,
      plan,
      fetchAsset,
      fetchInit,
      options,
    );
    return new ParakeetGpuPackage(
      manifest,
      plan.tensors,
      loaded.buffers,
      loaded.tokenizer,
      plan.memory,
      {
        manifestUrl,
        manifestSha256,
      },
    );
  }

  tensor(name: string): GpuTensor {
    if (this.destroyed) {
      throw new Error("Parakeet model package was destroyed");
    }
    const plan = this.tensorPlans?.get(name);
    if (plan === undefined) {
      if (this.tensorPlans === undefined) {
        throw new Error("Parakeet tensor metadata was released");
      }
      throw new Error(`Unknown runtime tensor ${name}`);
    }
    const runtime = plan.runtimeRecord;
    const buffer = this.buffers.get(runtime.bufferId);
    if (buffer === undefined) {
      throw new Error(
        `Runtime buffer ${runtime.bufferId} is not resident`,
      );
    }
    return {
      sourceRecord: plan.sourceRecord,
      runtimeRecord: runtime,
      buffer,
      binding: {
        buffer,
        offset: runtime.byteOffset,
        size: runtime.byteLength,
      },
    };
  }

  releaseInitializationMetadata(): void {
    if (this.destroyed) {
      throw new Error("Parakeet model package was destroyed");
    }
    this.manifestValue = undefined;
    this.tokenizerValue = undefined;
    this.tensorPlans = undefined;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.manifestValue = undefined;
    this.tokenizerValue = undefined;
    this.tensorPlans = undefined;
    destroyBuffers(this.buffers.values());
    this.buffers.clear();
  }
}

interface LoadedRuntimeAssets {
  readonly buffers: Map<RuntimeBufferId, GPUBuffer>;
  readonly tokenizer: unknown;
}

async function loadRuntimeAssets(
  device: GPUDevice,
  manifestUrl: string,
  manifest: ParakeetManifest,
  plan: RuntimePackagePlan,
  fetchAsset: typeof fetch,
  fetchInit: RequestInit | undefined,
  options: Pick<
    ParakeetPackageLoadOptions,
    "signal" | "onProgress"
  >,
): Promise<LoadedRuntimeAssets> {
  const buffers = new Map<RuntimeBufferId, GPUBuffer>();
  const onProgress = options.onProgress;
  let completedBytes = 0;
  try {
    for (const shard of plan.shards) {
      throwIfAborted(options.signal);
      const response = await fetchAsset(
        resolveAssetUrl(shard.file.name, manifestUrl),
        fetchInit,
      );
      if (!response.ok) {
        throw new Error(
          `Model shard ${shard.file.name} failed ` +
            `(${response.status} ${response.statusText})`,
        );
      }
      const reportProgress =
        onProgress === undefined
          ? undefined
          : (loadedBytes: number): void => {
              onProgress({
                phase: "weights",
                file: shard.file.name,
                loadedBytes: completedBytes + loadedBytes,
                totalBytes: plan.memory.packedDownloadBytes,
              });
            };
      if (shard.hashOnly) {
        await streamAndValidateShard(
          device,
          response,
          shard.file,
          undefined,
          options.signal,
          reportProgress,
        );
      } else {
        const runtimeBuffer = await loadRuntimeShard(
          device,
          response,
          shard,
          options.signal,
          reportProgress,
        );
        if (buffers.has(shard.runtimeBufferId)) {
          runtimeBuffer.destroy();
          throw new Error(
            `Runtime buffer ${shard.runtimeBufferId} was already allocated`,
          );
        }
        buffers.set(shard.runtimeBufferId, runtimeBuffer);
      }
      completedBytes += shard.file.byteLength;
    }
    requireCompleteRuntimeBuffers(plan, buffers);

    const tokenizerRecord = requiredFile(
      manifest,
      "tokenizer.json",
    );
    onProgress?.({
      phase: "tokenizer",
      loadedBytes: 0,
      totalBytes: tokenizerRecord.byteLength,
    });
    const tokenizerResponse = await fetchAsset(
      resolveAssetUrl(tokenizerRecord.name, manifestUrl),
      fetchInit,
    );
    if (!tokenizerResponse.ok) {
      throw new Error(
        `Tokenizer request failed (${tokenizerResponse.status})`,
      );
    }
    const tokenizerBytes = new Uint8Array(
      await tokenizerResponse.arrayBuffer(),
    );
    throwIfAborted(options.signal);
    if (
      tokenizerBytes.byteLength !== tokenizerRecord.byteLength ||
      new IncrementalSha256()
        .update(tokenizerBytes)
        .digestHex() !== tokenizerRecord.sha256
    ) {
      throw new Error("Tokenizer integrity validation failed");
    }
    const tokenizer = JSON.parse(
      new TextDecoder().decode(tokenizerBytes),
    ) as unknown;
    onProgress?.({
      phase: "ready",
      loadedBytes: plan.memory.packedDownloadBytes,
      totalBytes: plan.memory.packedDownloadBytes,
    });
    return { buffers, tokenizer };
  } catch (error) {
    destroyBuffers(buffers.values());
    throw error;
  }
}

async function loadRuntimeShard(
  device: GPUDevice,
  response: Response,
  shard: Extract<RuntimeShardPlan, { readonly hashOnly: false }>,
  signal: AbortSignal | undefined,
  onProgress?: (loadedBytes: number) => void,
): Promise<GPUBuffer> {
  const runtimeBuffer = await createCheckedBuffer(
    device,
    {
      label: `parakeet-source-${shard.file.name}`,
      size: shard.file.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    `${shard.file.name} runtime allocation`,
  );
  let transferred = false;
  try {
    await streamAndValidateShard(
      device,
      response,
      shard.file,
      runtimeBuffer,
      signal,
      onProgress,
    );
    throwIfAborted(signal);
    await device.queue.onSubmittedWorkDone();
    throwIfAborted(signal);
    transferred = true;
    return runtimeBuffer;
  } finally {
    if (!transferred) runtimeBuffer.destroy();
  }
}

async function streamAndValidateShard(
  device: GPUDevice,
  response: Response,
  file: ParakeetFileRecord,
  target: GPUBuffer | undefined,
  signal: AbortSignal | undefined,
  onProgress?: (loadedBytes: number) => void,
): Promise<void> {
  const contentLength = response.headers.get("content-length");
  if (
    contentLength !== null &&
    Number(contentLength) !== file.byteLength
  ) {
    throw new Error(
      `${file.name} Content-Length does not match manifest`,
    );
  }
  if (response.body === null) {
    throw new Error(`${file.name} is not streamable`);
  }
  const reader = response.body.getReader();
  const hash = new IncrementalSha256();
  let readBytes = 0;
  let uploadedBytes = 0;
  let carry = new Uint8Array(0);
  let lastReported = 0;
  const abortReader = (): void => {
    void reader.cancel(signal?.reason).catch(() => undefined);
  };
  signal?.addEventListener("abort", abortReader, { once: true });
  try {
    while (true) {
      throwIfAborted(signal);
      const item = await reader.read();
      throwIfAborted(signal);
      if (item.done) break;
      const chunk = item.value;
      if (readBytes + chunk.byteLength > file.byteLength) {
        throw new Error(`${file.name} exceeds declared size`);
      }
      hash.update(chunk);
      if (target !== undefined) {
        let offset = 0;
        if (carry.byteLength > 0) {
          const needed = 4 - carry.byteLength;
          if (chunk.byteLength < needed) {
            const combined = new Uint8Array(
              carry.byteLength + chunk.byteLength,
            );
            combined.set(carry);
            combined.set(chunk, carry.byteLength);
            carry = combined;
            readBytes += chunk.byteLength;
            continue;
          }
          const word = new Uint8Array(4);
          word.set(carry);
          word.set(
            chunk.subarray(0, needed),
            carry.byteLength,
          );
          device.queue.writeBuffer(target, uploadedBytes, word);
          uploadedBytes += 4;
          offset = needed;
          carry = new Uint8Array(0);
        }
        const alignedLength =
          (chunk.byteLength - offset) & ~3;
        if (alignedLength > 0) {
          device.queue.writeBuffer(
            target,
            uploadedBytes,
            chunk.subarray(
              offset,
              offset + alignedLength,
            ),
          );
          uploadedBytes += alignedLength;
          offset += alignedLength;
        }
        if (offset < chunk.byteLength) {
          carry = chunk.slice(offset);
        }
      }
      readBytes += chunk.byteLength;
      if (
        onProgress !== undefined &&
        readBytes - lastReported >= 1024 * 1024
      ) {
        lastReported = readBytes;
        onProgress(readBytes);
      }
    }
    if (
      readBytes !== file.byteLength ||
      (target !== undefined &&
        (uploadedBytes !== file.byteLength ||
          carry.byteLength !== 0))
    ) {
      throw new Error(
        `${file.name} ended before its aligned declared length`,
      );
    }
    if (hash.digestHex() !== file.sha256) {
      throw new Error(`${file.name} SHA-256 validation failed`);
    }
    onProgress?.(readBytes);
  } catch (error) {
    void reader.cancel(error).catch(() => undefined);
    throw error;
  } finally {
    signal?.removeEventListener("abort", abortReader);
    reader.releaseLock();
  }
}

async function createCheckedBuffer(
  device: GPUDevice,
  descriptor: GPUBufferDescriptor,
  role: string,
): Promise<GPUBuffer> {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let errorScopesOpen = true;
  let buffer: GPUBuffer | undefined;
  try {
    buffer = device.createBuffer(descriptor);
    const validation = device.popErrorScope();
    const outOfMemory = device.popErrorScope();
    errorScopesOpen = false;
    const [validationResult, memoryResult] =
      await Promise.allSettled([validation, outOfMemory]);
    if (validationResult.status === "rejected") {
      throw validationResult.reason;
    }
    if (memoryResult.status === "rejected") {
      throw memoryResult.reason;
    }
    if (validationResult.value !== null) {
      throw new Error(
        `${role} failed validation: ` +
          validationResult.value.message,
      );
    }
    if (memoryResult.value !== null) {
      throw new Error(
        `${role} exhausted GPU memory: ` +
          memoryResult.value.message,
      );
    }
    return buffer;
  } catch (error) {
    if (errorScopesOpen) {
      await device.popErrorScope().catch(() => null);
      await device.popErrorScope().catch(() => null);
      errorScopesOpen = false;
    }
    buffer?.destroy();
    throw error;
  }
}

function requirePlanSupport(
  device: GPUDevice,
  plan: RuntimePackagePlan,
): void {
  for (const buffer of plan.buffers) {
    if (
      buffer.byteLength > device.limits.maxBufferSize ||
      buffer.byteLength >
        device.limits.maxStorageBufferBindingSize
    ) {
      throw new Error(
        `${buffer.id} (${buffer.byteLength} bytes) exceeds ` +
          "this WebGPU device's limits",
      );
    }
  }
  for (const shard of plan.shards) {
    if (
      !shard.hashOnly &&
      (shard.file.byteLength > device.limits.maxBufferSize ||
        shard.file.byteLength >
          device.limits.maxStorageBufferBindingSize)
    ) {
      throw new Error(
        `${shard.file.name} runtime shard exceeds this WebGPU device's limits`,
      );
    }
  }
}

function requiredRuntimeBuffer(
  buffers: ReadonlyMap<RuntimeBufferId, GPUBuffer>,
  id: RuntimeBufferId,
): GPUBuffer {
  const buffer = buffers.get(id);
  if (buffer === undefined) {
    throw new Error(`Runtime buffer ${id} was not allocated`);
  }
  return buffer;
}

function requireCompleteRuntimeBuffers(
  plan: RuntimePackagePlan,
  buffers: ReadonlyMap<RuntimeBufferId, GPUBuffer>,
): void {
  if (buffers.size !== plan.buffers.length) {
    throw new Error(
      `Runtime loader materialized ${buffers.size} of ${plan.buffers.length} buffers`,
    );
  }
  for (const buffer of plan.buffers) {
    requiredRuntimeBuffer(buffers, buffer.id);
  }
}

function destroyBuffers(buffers: Iterable<GPUBuffer>): void {
  for (const buffer of new Set(buffers)) buffer.destroy();
}

function throwIfAborted(
  signal: AbortSignal | undefined,
): void {
  signal?.throwIfAborted();
}

function requiredFile(
  manifest: ParakeetManifest,
  name: string,
): ParakeetFileRecord {
  const record = manifest.files.find(
    (candidate) => candidate.name === name,
  );
  if (record === undefined) {
    throw new Error(`Model package is missing ${name}`);
  }
  return record;
}

function resolveAssetUrl(
  file: string,
  manifestUrl: string,
): string {
  const base = new URL(
    manifestUrl,
    globalThis.location?.href ?? "http://localhost/",
  );
  return new URL(file, base).href;
}

/**
 * Narrow test seam for the production loader's streaming and cleanup
 * invariants. The public package path always constructs the fixed plan above.
 */
export const __packageLoaderInternals = Object.freeze({
  loadRuntimeAssets,
  loadRuntimeShard,
  streamAndValidateShard,
});
