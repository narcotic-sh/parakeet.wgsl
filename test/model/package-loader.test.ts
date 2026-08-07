import { createHash } from "node:crypto";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type {
  ParakeetFileRecord,
  ParakeetManifest,
} from "../../src/model/manifest";
import {
  __packageLoaderInternals,
  ParakeetGpuPackage,
} from "../../src/model/package";
import type {
  RuntimePackagePlan,
  RuntimeShardPlan,
} from "../../src/model/runtime-plan";

type ResidentShardPlan = Extract<
  RuntimeShardPlan,
  { readonly hashOnly: false }
>;

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: {
      COPY_DST: 2,
      STORAGE: 4,
    },
  });
});

describe("streamed runtime package loader", () => {
  it("rejects a changed manifest digest before requesting any shard", async () => {
    const fetchAsset = vi.fn(async () =>
      new Response("{}", {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(
      ParakeetGpuPackage.load(
        {} as GPUDevice,
        "https://example.test/model/files/manifest.json",
        {
          expectedPrecision: "fp16",
          fetch: fetchAsset as unknown as typeof fetch,
        },
      ),
    ).rejects.toThrow(
      /Selected fp16 model manifest SHA-256 changed/,
    );

    expect(fetchAsset).toHaveBeenCalledTimes(1);
    expect(fetchAsset).toHaveBeenCalledWith(
      "https://example.test/model/files/manifest.json",
      undefined,
    );
  });

  it("uploads unaligned stream chunks without materializing the shard", async () => {
    const bytes = Uint8Array.from([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]);
    const device = new FakeDevice();
    const resident = device.createBuffer({
      size: bytes.byteLength,
      usage: GPUBufferUsage.COPY_DST,
    });
    const progress: number[] = [];

    await __packageLoaderInternals.streamAndValidateShard(
      device.asGpuDevice(),
      chunkedResponse(bytes, [3, 2, 7]),
      fileRecord("runtime.bin", bytes),
      resident as unknown as GPUBuffer,
      undefined,
      (loaded) => progress.push(loaded),
    );

    expect([...resident.bytes]).toEqual([...bytes]);
    expect(device.queue.writeCalls.map((call) => call.size)).toEqual([
      4, 4, 4,
    ]);
    expect(progress.at(-1)).toBe(bytes.byteLength);
  });

  it("cancels a pending stream promptly when aborted", async () => {
    const controller = new AbortController();
    let cancelReason: unknown;
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(Uint8Array.from([0, 1, 2, 3]));
      },
      pull() {
        return new Promise<void>(() => undefined);
      },
      cancel(reason) {
        cancelReason = reason;
      },
    });
    const bytes = new Uint8Array(8);
    const promise =
      __packageLoaderInternals.streamAndValidateShard(
        new FakeDevice().asGpuDevice(),
        new Response(stream, {
          headers: {
            "content-length": String(bytes.byteLength),
          },
        }),
        fileRecord("runtime.bin", bytes),
        undefined,
        controller.signal,
        () => undefined,
      );
    await Promise.resolve();
    const reason = new DOMException("stop", "AbortError");
    controller.abort(reason);

    await expect(promise).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(cancelReason).toBe(reason);
  });

  it("destroys the resident buffer after a hash failure", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice();

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes),
        shardPlan(bytes, {
          sha256: "0".repeat(64),
        }),
        undefined,
      ),
    ).rejects.toThrow(/runtime\.bin SHA-256/);

    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.completedCalls).toBe(0);
    expect(device.scopePushes).toBe(2);
    expect(device.scopePops).toBe(2);
  });

  it("destroys the resident buffer after a declared-length failure", async () => {
    const declaredBytes = Uint8Array.from([
      1, 2, 3, 4, 5, 6, 7, 8,
    ]);
    const responseBytes = declaredBytes.subarray(0, 4);
    const device = new FakeDevice();

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(responseBytes),
        shardPlan(declaredBytes),
        undefined,
        () => undefined,
      ),
    ).rejects.toThrow(
      /runtime\.bin Content-Length does not match manifest/,
    );

    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.writeCalls).toHaveLength(0);
    expect(device.queue.completedCalls).toBe(0);
  });

  it("destroys the resident buffer and cancels its stream when aborted", async () => {
    const controller = new AbortController();
    let cancelReason: unknown;
    const bytes = new Uint8Array(8);
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(bytes.subarray(0, 4));
      },
      pull() {
        return new Promise<void>(() => undefined);
      },
      cancel(reason) {
        cancelReason = reason;
      },
    });
    const device = new FakeDevice();
    const promise = __packageLoaderInternals.loadRuntimeShard(
      device.asGpuDevice(),
      new Response(stream, {
        headers: {
          "content-length": String(bytes.byteLength),
        },
      }),
      shardPlan(bytes),
      controller.signal,
      () => undefined,
    );
    await Promise.resolve();
    const reason = new DOMException("stop", "AbortError");
    controller.abort(reason);

    await expect(promise).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(cancelReason).toBe(reason);
    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.completedCalls).toBe(0);
  });

  it("destroys the resident buffer when queue completion fails", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice({
      queueFailure: new Error("queue completion failed"),
    });

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes),
        shardPlan(bytes),
        undefined,
        () => undefined,
      ),
    ).rejects.toThrow(/queue completion failed/);

    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.completedCalls).toBe(1);
    expect(device.scopePushes).toBe(2);
    expect(device.scopePops).toBe(2);
  });

  it("surfaces allocation validation errors before uploading", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice({
      validationErrors: [
        {
          message: "invalid resident buffer",
        } as GPUError,
      ],
    });

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes),
        shardPlan(bytes),
        undefined,
        () => undefined,
      ),
    ).rejects.toThrow(
      /runtime\.bin runtime allocation failed validation: invalid resident buffer/,
    );

    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.writeCalls).toHaveLength(0);
    expect(device.queue.completedCalls).toBe(0);
  });

  it("streams a validated shard directly into its final resident buffer", async () => {
    const bytes = Uint8Array.from([
      11, 12, 13, 14, 21, 22, 23, 24,
    ]);
    const device = new FakeDevice();

    const resident =
      await __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes, [3, 5]),
        shardPlan(bytes),
        undefined,
      );

    expect(resident).toBe(
      device.createdBuffers[0] as unknown as GPUBuffer,
    );
    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(false);
    expect(device.createdBuffers[0]!.size).toBe(bytes.byteLength);
    expect([...device.createdBuffers[0]!.bytes]).toEqual([...bytes]);
    expect(device.createdBuffers[0]!.usage).toBe(
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    expect(device.queue.completedCalls).toBe(1);
  });

  it("destroys the failed runtime buffer when package queue initialization fails", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice({
      queueFailure: new Error("device lost"),
    });
    const plan = tinyPackagePlan(bytes);
    const manifest = tinyManifest(bytes);
    const fetchAsset = vi.fn(async (url: string | URL | Request) => {
      const href = String(url);
      if (href.endsWith("/runtime.bin")) {
        return chunkedResponse(bytes);
      }
      throw new Error(`Unexpected request ${href}`);
    }) as unknown as typeof fetch;

    await expect(
      __packageLoaderInternals.loadRuntimeAssets(
        device.asGpuDevice(),
        "https://example.test/model/manifest.json",
        manifest,
        plan,
        fetchAsset,
        undefined,
        {},
      ),
    ).rejects.toThrow(/device lost/);

    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(fetchAsset).toHaveBeenCalledTimes(1);
  });

  it("settles allocation scopes when resident-buffer creation throws", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice({ failBufferCreationAt: 1 });

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes),
        shardPlan(bytes),
        undefined,
        () => undefined,
      ),
    ).rejects.toThrow(/buffer allocation failed/);

    expect(device.createdBuffers).toHaveLength(0);
    expect(device.scopePushes).toBe(2);
    expect(device.scopePops).toBe(2);
  });

  it("destroys resident buffers rejected by the out-of-memory scope", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const device = new FakeDevice({
      outOfMemoryErrors: [
        {
          message: "allocation limit",
        } as GPUError,
      ],
    });

    await expect(
      __packageLoaderInternals.loadRuntimeShard(
        device.asGpuDevice(),
        chunkedResponse(bytes),
        shardPlan(bytes),
        undefined,
        () => undefined,
      ),
    ).rejects.toThrow(
      /runtime\.bin runtime allocation exhausted GPU memory: allocation limit/,
    );

    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
    expect(device.queue.writeCalls).toHaveLength(0);
    expect(device.queue.completedCalls).toBe(0);
  });

  it("returns parsed tokenizer state and retains only runtime shards", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const hashOnly = Uint8Array.from([5, 6, 7, 8]);
    const tokenizer = new TextEncoder().encode('{"tokens":["a"]}');
    const device = new FakeDevice();
    const plan = tinyPackagePlan(bytes, hashOnly);
    const manifest = tinyManifest(bytes, tokenizer, hashOnly);
    const progress: string[] = [];
    const fetchAsset = vi.fn(async (url: string | URL | Request) => {
      const href = String(url);
      if (href.endsWith("/preprocessor.bin")) {
        return chunkedResponse(hashOnly);
      }
      if (href.endsWith("/runtime.bin")) {
        return chunkedResponse(bytes);
      }
      if (href.endsWith("/tokenizer.json")) {
        return chunkedResponse(tokenizer);
      }
      throw new Error(`Unexpected request ${href}`);
    }) as unknown as typeof fetch;

    const loaded =
      await __packageLoaderInternals.loadRuntimeAssets(
        device.asGpuDevice(),
        "https://example.test/model/manifest.json",
        manifest,
        plan,
        fetchAsset,
        undefined,
        {
          onProgress(item) {
            progress.push(item.phase);
          },
        },
      );

    expect(loaded.tokenizer).toEqual({ tokens: ["a"] });
    expect([...loaded.buffers.keys()]).toEqual([
      "source-shard-runtime.bin",
    ]);
    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(false);
    expect([...device.createdBuffers[0]!.bytes]).toEqual([...bytes]);
    expect(progress).toEqual([
      "weights",
      "weights",
      "tokenizer",
      "ready",
    ]);
  });

  it("destroys the runtime shard when tokenizer integrity validation fails", async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const tokenizer = new TextEncoder().encode('{"tokens":["a"]}');
    const validManifest = tinyManifest(bytes, tokenizer);
    const manifest = {
      ...validManifest,
      files: validManifest.files.map((file) =>
        file.name === "tokenizer.json"
          ? { ...file, sha256: "0".repeat(64) }
          : file,
      ),
    } as ParakeetManifest;
    const device = new FakeDevice();
    const fetchAsset = vi.fn(async (url: string | URL | Request) => {
      const href = String(url);
      if (href.endsWith("/runtime.bin")) {
        return chunkedResponse(bytes);
      }
      if (href.endsWith("/tokenizer.json")) {
        return chunkedResponse(tokenizer);
      }
      throw new Error(`Unexpected request ${href}`);
    }) as unknown as typeof fetch;

    await expect(
      __packageLoaderInternals.loadRuntimeAssets(
        device.asGpuDevice(),
        "https://example.test/model/manifest.json",
        manifest,
        tinyPackagePlan(bytes),
        fetchAsset,
        undefined,
        {},
      ),
    ).rejects.toThrow(/Tokenizer integrity validation failed/);

    expect(device.createdBuffers).toHaveLength(1);
    expect(device.createdBuffers[0]!.destroyed).toBe(true);
  });

  it("destroys both buffers when two shards claim one runtime id", async () => {
    const first = Uint8Array.from([1, 2, 3, 4]);
    const second = Uint8Array.from([5, 6, 7, 8]);
    const plan = duplicateOwnershipPackagePlan(first, second);
    const manifest = tinyManifestForFiles([
      fileRecord("first.bin", first),
      fileRecord("second.bin", second),
    ]);
    const device = new FakeDevice();
    const fetchAsset = vi.fn(async (url: string | URL | Request) => {
      const href = String(url);
      if (href.endsWith("/first.bin")) {
        return chunkedResponse(first);
      }
      if (href.endsWith("/second.bin")) {
        return chunkedResponse(second);
      }
      throw new Error(`Unexpected request ${href}`);
    }) as unknown as typeof fetch;

    await expect(
      __packageLoaderInternals.loadRuntimeAssets(
        device.asGpuDevice(),
        "https://example.test/model/manifest.json",
        manifest,
        plan,
        fetchAsset,
        undefined,
        {},
      ),
    ).rejects.toThrow(
      /Runtime buffer source-shard-runtime\.bin was already allocated/,
    );

    expect(device.createdBuffers).toHaveLength(2);
    expect(
      device.createdBuffers.every((buffer) => buffer.destroyed),
    ).toBe(true);
  });

  it("destroys an earlier resident shard when a later shard fails integrity", async () => {
    const first = Uint8Array.from([1, 2, 3, 4]);
    const second = Uint8Array.from([5, 6, 7, 8]);
    const plan = residentPairPackagePlan(first, second);
    const manifest = tinyManifestForFiles([
      fileRecord("first.bin", first),
      {
        ...fileRecord("second.bin", second),
        sha256: "0".repeat(64),
      },
    ]);
    const device = new FakeDevice();
    const fetchAsset = vi.fn(async (url: string | URL | Request) => {
      const href = String(url);
      if (href.endsWith("/first.bin")) {
        return chunkedResponse(first);
      }
      if (href.endsWith("/second.bin")) {
        return chunkedResponse(second);
      }
      throw new Error(`Unexpected request ${href}`);
    }) as unknown as typeof fetch;

    await expect(
      __packageLoaderInternals.loadRuntimeAssets(
        device.asGpuDevice(),
        "https://example.test/model/manifest.json",
        manifest,
        plan,
        fetchAsset,
        undefined,
        {},
      ),
    ).rejects.toThrow(/second\.bin SHA-256/);

    expect(device.createdBuffers).toHaveLength(2);
    expect(
      device.createdBuffers.every((buffer) => buffer.destroyed),
    ).toBe(true);
  });
});

interface FakeDeviceOptions {
  readonly queueFailure?: Error;
  readonly validationErrors?: readonly (GPUError | null)[];
  readonly outOfMemoryErrors?: readonly (GPUError | null)[];
  readonly failBufferCreationAt?: number;
}

class FakeBuffer {
  readonly bytes: Uint8Array;
  destroyed = false;

  constructor(
    readonly label: string,
    readonly size: number,
    readonly usage = 0,
  ) {
    this.bytes = new Uint8Array(size);
  }

  destroy(): void {
    this.destroyed = true;
  }
}

class FakeQueue {
  readonly writeCalls: Array<{ readonly size: number }> = [];
  completedCalls = 0;

  constructor(private readonly failure?: Error) {}

  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: AllowSharedBufferSource,
  ): void {
    const target = buffer as unknown as FakeBuffer;
    const bytes =
      ArrayBuffer.isView(data)
        ? new Uint8Array(
            data.buffer,
            data.byteOffset,
            data.byteLength,
          )
        : new Uint8Array(data);
    target.bytes.set(bytes, bufferOffset);
    this.writeCalls.push({ size: bytes.byteLength });
  }

  onSubmittedWorkDone(): Promise<void> {
    this.completedCalls += 1;
    return this.failure === undefined
      ? Promise.resolve()
      : Promise.reject(this.failure);
  }
}

class FakeDevice {
  readonly createdBuffers: FakeBuffer[] = [];
  readonly queue: FakeQueue;
  scopePushes = 0;
  scopePops = 0;
  private bufferCreations = 0;
  private readonly scopeStack: GPUErrorFilter[] = [];
  private readonly validationErrors: Array<GPUError | null>;
  private readonly outOfMemoryErrors: Array<GPUError | null>;

  constructor(private readonly options: FakeDeviceOptions = {}) {
    this.queue = new FakeQueue(options.queueFailure);
    this.validationErrors = [
      ...(options.validationErrors ?? []),
    ];
    this.outOfMemoryErrors = [
      ...(options.outOfMemoryErrors ?? []),
    ];
  }

  createBuffer(descriptor: GPUBufferDescriptor): FakeBuffer {
    this.bufferCreations += 1;
    if (
      this.options.failBufferCreationAt ===
      this.bufferCreations
    ) {
      throw new Error("buffer allocation failed");
    }
    const buffer = new FakeBuffer(
      descriptor.label ?? "buffer",
      Number(descriptor.size),
      descriptor.usage,
    );
    this.createdBuffers.push(buffer);
    return buffer;
  }

  pushErrorScope(filter: GPUErrorFilter): void {
    this.scopePushes += 1;
    this.scopeStack.push(filter);
  }

  popErrorScope(): Promise<GPUError | null> {
    this.scopePops += 1;
    const filter = this.scopeStack.pop();
    if (filter === undefined) {
      return Promise.reject(new Error("No fake error scope"));
    }
    if (filter === "validation") {
      return Promise.resolve(
        this.validationErrors.shift() ?? null,
      );
    }
    if (filter === "out-of-memory") {
      return Promise.resolve(
        this.outOfMemoryErrors.shift() ?? null,
      );
    }
    return Promise.resolve(null);
  }

  asGpuDevice(): GPUDevice {
    return this as unknown as GPUDevice;
  }
}

function chunkedResponse(
  bytes: Uint8Array,
  chunkSizes: readonly number[] = [bytes.byteLength],
): Response {
  let offset = 0;
  const remainingChunkSizes = [...chunkSizes];
  return new Response(
    new ReadableStream<Uint8Array>({
      pull(controller) {
        const size = remainingChunkSizes.shift();
        const nextSize =
          size ??
          (offset < bytes.byteLength
            ? bytes.byteLength - offset
            : 0);
        if (nextSize <= 0 || offset >= bytes.byteLength) {
          controller.close();
          return;
        }
        controller.enqueue(
          bytes.slice(offset, offset + nextSize),
        );
        offset += nextSize;
        if (offset >= bytes.byteLength) controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        "content-length": String(bytes.byteLength),
      },
    },
  );
}

function fileRecord(
  name: string,
  bytes: Uint8Array,
): ParakeetFileRecord {
  return {
    name,
    byteLength: bytes.byteLength,
    sha256: digest(bytes),
  };
}

function shardPlan(
  bytes: Uint8Array,
  overrides: {
    readonly sha256?: string;
  } = {},
): ResidentShardPlan {
  return {
    file: {
      ...fileRecord("runtime.bin", bytes),
      ...(overrides.sha256 === undefined
        ? {}
        : { sha256: overrides.sha256 }),
    },
    runtimeBufferId: "source-shard-runtime.bin",
    hashOnly: false,
  };
}

function tinyPackagePlan(
  bytes: Uint8Array,
  hashOnly?: Uint8Array,
): RuntimePackagePlan {
  const skippedShard:
    | Extract<RuntimeShardPlan, { readonly hashOnly: true }>
    | undefined =
    hashOnly === undefined
      ? undefined
      : {
          file: fileRecord("preprocessor.bin", hashOnly),
          hashOnly: true,
        };
  const packedDownloadBytes =
    bytes.byteLength + (hashOnly?.byteLength ?? 0);
  return {
    buffers: [
      {
        id: "source-shard-runtime.bin",
        label: "source-shard-runtime.bin",
        byteLength: bytes.byteLength,
        tensorCount: 1,
      },
    ],
    tensors: new Map(),
    shards: [
      ...(skippedShard === undefined ? [] : [skippedShard]),
      shardPlan(bytes),
    ],
    memory: {
      packedDownloadBytes,
      retainedRuntimeBytes: bytes.byteLength,
      expandedRuntimeBytes: 0,
      runtimeGpuBytes: bytes.byteLength,
      largestRuntimeShardBytes: bytes.byteLength,
      loaderPeakGpuBytes: bytes.byteLength,
      runtimeBufferCount: 1,
    },
  };
}

function residentPairPackagePlan(
  first: Uint8Array,
  second: Uint8Array,
): RuntimePackagePlan {
  const plans = [
    ["first.bin", first],
    ["second.bin", second],
  ] as const;
  return {
    buffers: plans.map(([name, bytes]) => ({
      id: `source-shard-${name}` as const,
      label: `source-shard-${name}`,
      byteLength: bytes.byteLength,
      tensorCount: 1,
    })),
    tensors: new Map(),
    shards: plans.map(([name, bytes], index) => ({
      file: {
        ...fileRecord(name, bytes),
        ...(index === 1 ? { sha256: "0".repeat(64) } : {}),
      },
      runtimeBufferId: `source-shard-${name}` as const,
      hashOnly: false as const,
    })),
    memory: {
      packedDownloadBytes: first.byteLength + second.byteLength,
      retainedRuntimeBytes: first.byteLength + second.byteLength,
      expandedRuntimeBytes: 0,
      runtimeGpuBytes: first.byteLength + second.byteLength,
      largestRuntimeShardBytes: Math.max(
        first.byteLength,
        second.byteLength,
      ),
      loaderPeakGpuBytes: first.byteLength + second.byteLength,
      runtimeBufferCount: 2,
    },
  };
}

function duplicateOwnershipPackagePlan(
  first: Uint8Array,
  second: Uint8Array,
): RuntimePackagePlan {
  return {
    buffers: [
      {
        id: "source-shard-runtime.bin",
        label: "source-shard-runtime.bin",
        byteLength: first.byteLength,
        tensorCount: 1,
      },
    ],
    tensors: new Map(),
    shards: [
      {
        file: fileRecord("first.bin", first),
        runtimeBufferId: "source-shard-runtime.bin",
        hashOnly: false,
      },
      {
        file: fileRecord("second.bin", second),
        runtimeBufferId: "source-shard-runtime.bin",
        hashOnly: false,
      },
    ],
    memory: {
      packedDownloadBytes: first.byteLength + second.byteLength,
      retainedRuntimeBytes: first.byteLength,
      expandedRuntimeBytes: 0,
      runtimeGpuBytes: first.byteLength,
      largestRuntimeShardBytes: Math.max(
        first.byteLength,
        second.byteLength,
      ),
      loaderPeakGpuBytes: first.byteLength,
      runtimeBufferCount: 1,
    },
  };
}

function tinyManifest(
  bytes: Uint8Array,
  tokenizer = new TextEncoder().encode("{}"),
  hashOnly?: Uint8Array,
): ParakeetManifest {
  return {
    files: [
      ...(hashOnly === undefined
        ? []
        : [fileRecord("preprocessor.bin", hashOnly)]),
      fileRecord("runtime.bin", bytes),
      fileRecord("tokenizer.json", tokenizer),
    ],
    tensors: {},
  } as unknown as ParakeetManifest;
}

function tinyManifestForFiles(
  files: readonly ParakeetFileRecord[],
): ParakeetManifest {
  return {
    files,
    tensors: {},
  } as unknown as ParakeetManifest;
}

function digest(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
