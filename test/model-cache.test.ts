import { describe, expect, it, vi } from "vitest";

import {
  ModelCacheSession,
  ModelNotCachedError,
  clearStoredModels,
  crossOriginStorageBundleLayout,
  inspectCrossOriginStorageBundles,
  inspectStoredModels,
  type CrossOriginStorageBundle,
  type CrossOriginStorageManager,
  type ModelCacheAsset,
} from "../src/model-cache";

const MODEL_CACHE_PREFIX = "parakeet.wgsl:models:v1:";

describe("ModelCacheSession", () => {
  it("streams a cold network response into Cache Storage with no-store", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(19, 7);
    const asset = await modelAsset("https://models.example/v1/model.bin", bytes);
    const tail = deferred<void>();
    const networkCalls: Array<{
      readonly url: string;
      readonly cache: RequestCache | undefined;
    }> = [];
    const networkFetch: typeof fetch = async (input, init) => {
      networkCalls.push({
        url: requestUrl(input),
        cache: init?.cache,
      });
      return gatedResponse(bytes.subarray(0, 6), bytes.subarray(6), tail.promise);
    };
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: networkFetch,
    });

    const response = await session.fetch(asset.url);
    const bodyPromise = response.arrayBuffer();
    let bodySettled = false;
    void bodyPromise.finally(() => {
      bodySettled = true;
    });
    await Promise.resolve();

    expect(bodySettled).toBe(false);
    expect(networkCalls).toEqual([{ url: asset.url, cache: "no-store" }]);

    tail.resolve();
    expect(new Uint8Array(await bodyPromise)).toEqual(bytes);
    await session.settle();

    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 1,
      sizeBytes: bytes.byteLength,
    });
    expect(await storage.onlyModelCache().bytes(asset.url)).toEqual(bytes);
  });

  it("serves a warm verified hit without touching the network", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(31, 11);
    const asset = await modelAsset("https://models.example/v1/warm.bin", bytes);
    let coldNetworkCalls = 0;
    const cold = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => {
        coldNetworkCalls += 1;
        return responseFromChunks(bytes.subarray(0, 10), bytes.subarray(10));
      },
    });

    expect(new Uint8Array(await (await cold.fetch(asset.url)).arrayBuffer())).toEqual(
      bytes,
    );
    await cold.settle();

    let warmNetworkCalls = 0;
    const warm = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => {
        warmNetworkCalls += 1;
        throw new Error("the warm path must not fetch");
      },
    });

    const warmBytes = new Uint8Array(
      await (await warm.fetch(asset.url)).arrayBuffer(),
    );

    expect(warmBytes).toEqual(bytes);
    expect(coldNetworkCalls).toBe(1);
    expect(warmNetworkCalls).toBe(0);
    expect(warm.consumeCacheCorruption()).toBe(false);
  });

  it("does not create an empty namespace or reach the network on a cache-only miss", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(17, 83);
    const asset = await modelAsset(
      "https://models.example/v1/cache-only-miss.bin",
      bytes,
    );
    let networkCalls = 0;
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(session.fetch(asset.url)).rejects.toMatchObject({
      name: "ModelNotCachedError",
      code: "MODEL_NOT_CACHED",
      url: asset.url,
    });
    expect(networkCalls).toBe(0);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("does not reach the network when cache-only storage is unavailable", async () => {
    const bytes = byteSequence(11, 87);
    const asset = await modelAsset(
      "https://models.example/v1/no-cache-storage.bin",
      bytes,
    );
    let networkCalls = 0;
    const session = new ModelCacheSession(asset, {
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(session.fetch(asset.url)).rejects.toBeInstanceOf(
      ModelNotCachedError,
    );
    expect(networkCalls).toBe(0);
  });

  it("requires the exact cache namespace without creating another one", async () => {
    const storage = new MemoryCacheStorage();
    const firstBytes = byteSequence(19, 89);
    const secondBytes = byteSequence(23, 97);
    const first = await modelAsset(
      "https://models.example/v1/first.bin",
      firstBytes,
    );
    const second = await modelAsset(
      "https://models.example/v1/second.bin",
      secondBytes,
    );
    const cold = new ModelCacheSession(first, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(firstBytes),
    });
    await (await cold.fetch(first.url)).arrayBuffer();
    await cold.settle();
    const existingNames = await storage.asCacheStorage().keys();
    let networkCalls = 0;
    const selected = new ModelCacheSession(second, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(secondBytes);
      },
    });

    await expect(selected.fetch(second.url)).rejects.toBeInstanceOf(
      ModelNotCachedError,
    );
    expect(networkCalls).toBe(0);
    expect(await storage.asCacheStorage().keys()).toEqual(existingNames);
  });

  it("treats cache match failures as cache-only misses", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(29, 101);
    const asset = await modelAsset(
      "https://models.example/v1/match-failure.bin",
      bytes,
    );
    await seedManagedAssets(storage, asset, [[asset, bytes]]);
    storage.onlyModelCache().failMatches(
      new DOMException("cache unavailable", "InvalidStateError"),
    );
    let networkCalls = 0;
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(session.fetch(asset.url)).rejects.toMatchObject({
      code: "MODEL_NOT_CACHED",
    });
    expect(networkCalls).toBe(0);
  });

  it("rejects invalid cache-only metadata while normal mode replaces it", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(31, 103);
    const asset = await modelAsset(
      "https://models.example/v1/invalid-metadata.bin",
      bytes,
    );
    await seedManagedAssets(storage, asset, [[asset, bytes]]);
    const cache = storage.onlyModelCache();
    await cache.seed(asset.url, new Response(bytes));
    let cacheOnlyNetworkCalls = 0;
    const cacheOnly = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        cacheOnlyNetworkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(cacheOnly.fetch(asset.url)).rejects.toBeInstanceOf(
      ModelNotCachedError,
    );
    expect(cacheOnlyNetworkCalls).toBe(0);
    expect(cache.has(asset.url)).toBe(true);

    let normalNetworkCalls = 0;
    const normal = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => {
        normalNetworkCalls += 1;
        return new Response(bytes);
      },
    });
    expect(
      new Uint8Array(await (await normal.fetch(asset.url)).arrayBuffer()),
    ).toEqual(bytes);
    await normal.settle();
    expect(normalNetworkCalls).toBe(1);
    expect(await cache.bytes(asset.url)).toEqual(bytes);
  });

  it("rejects a null cached body without reaching the network", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(13, 107);
    const asset = await modelAsset(
      "https://models.example/v1/null-body.bin",
      bytes,
    );
    await seedManagedAssets(storage, asset, [[asset, bytes]]);
    storage.onlyModelCache().replaceBodyWithNull(asset.url);
    let networkCalls = 0;
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(session.fetch(asset.url)).rejects.toMatchObject({
      code: "MODEL_NOT_CACHED",
    });
    expect(networkCalls).toBe(0);
  });

  it("requires complete cached metadata for a registered runtime subset", async () => {
    const storage = new MemoryCacheStorage();
    const manifestBytes = byteSequence(17, 109);
    const firstBytes = byteSequence(23, 113);
    const secondBytes = byteSequence(37, 127);
    const manifest = await modelAsset(
      "https://models.example/v1/manifest.json",
      manifestBytes,
    );
    const first = await modelAsset(
      "https://models.example/v1/first-runtime.bin",
      firstBytes,
    );
    const second = await modelAsset(
      "https://models.example/v1/second-runtime.bin",
      secondBytes,
    );
    await seedManagedAssets(storage, manifest, [
      [manifest, manifestBytes],
      [first, firstBytes],
    ]);
    let networkCalls = 0;
    const partial = new ModelCacheSession(manifest, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        return new Response(secondBytes);
      },
    });

    await expect(
      partial.requireCachedAssets([first, second]),
    ).rejects.toMatchObject({
      code: "MODEL_NOT_CACHED",
      url: second.url,
    });
    expect(networkCalls).toBe(0);

    await seedManagedAssets(storage, manifest, [[second, secondBytes]]);
    const complete = new ModelCacheSession(manifest, {
      cacheStorage: storage.asCacheStorage(),
      cacheOnly: true,
      fetch: async () => {
        networkCalls += 1;
        throw new Error("complete metadata validation must not fetch");
      },
    });
    await expect(
      complete.requireCachedAssets([first, second]),
    ).resolves.toBeUndefined();
    expect(networkCalls).toBe(0);
  });

  it("signals and evicts same-length cached corruption before a network retry", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(23, 5);
    const asset = await modelAsset(
      "https://models.example/v1/corruptible.bin",
      bytes,
    );
    const cold = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(bytes),
    });
    await (await cold.fetch(asset.url)).arrayBuffer();
    await cold.settle();

    const cache = storage.onlyModelCache();
    const corruptBytes = bytes.slice();
    corruptBytes[9] = corruptBytes[9]! ^ 0xff;
    cache.replaceBody(asset.url, corruptBytes);

    const networkCaches: Array<RequestCache | undefined> = [];
    const retry = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async (_input, init) => {
        networkCaches.push(init?.cache);
        return responseFromChunks(bytes.subarray(0, 8), bytes.subarray(8));
      },
    });

    const cachedResponse = await retry.fetch(asset.url);
    await expect(cachedResponse.arrayBuffer()).rejects.toThrow(
      "Cached model asset SHA-256 mismatch",
    );
    expect(retry.consumeCacheCorruption()).toBe(true);
    expect(retry.consumeCacheCorruption()).toBe(false);
    expect(cache.has(asset.url)).toBe(false);
    expect(networkCaches).toEqual([]);

    expect(
      new Uint8Array(await (await retry.fetch(asset.url)).arrayBuffer()),
    ).toEqual(bytes);
    await retry.settle();

    expect(networkCaches).toEqual(["no-store"]);
    expect(await cache.bytes(asset.url)).toEqual(bytes);
  });

  it("evicts a cached response that fails while being read", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(27, 53);
    const asset = await modelAsset(
      "https://models.example/v1/unreadable.bin",
      bytes,
    );
    const cold = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(bytes),
    });
    await (await cold.fetch(asset.url)).arrayBuffer();
    await cold.settle();

    const cache = storage.onlyModelCache();
    cache.replaceBodyWithError(
      asset.url,
      new DOMException("cached body read failed", "NotReadableError"),
    );
    let networkCalls = 0;
    const retry = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => {
        networkCalls += 1;
        return new Response(bytes);
      },
    });

    await expect(
      (await retry.fetch(asset.url)).arrayBuffer(),
    ).rejects.toThrow("could not be read");
    expect(retry.consumeCacheCorruption()).toBe(true);
    expect(cache.has(asset.url)).toBe(false);

    expect(
      new Uint8Array(await (await retry.fetch(asset.url)).arrayBuffer()),
    ).toEqual(bytes);
    await retry.settle();
    expect(networkCalls).toBe(1);
  });

  it("does not mark a valid cached entry corrupt when its consumer cancels", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(33, 59);
    const asset = await modelAsset(
      "https://models.example/v1/cancelled.bin",
      bytes,
    );
    const cold = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(bytes),
    });
    await (await cold.fetch(asset.url)).arrayBuffer();
    await cold.settle();

    const cache = storage.onlyModelCache();
    cache.replaceBodyWithPendingRead(asset.url);
    const warm = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => {
        throw new Error("the cancelled cache hit must not fetch");
      },
    });
    const response = await warm.fetch(asset.url);
    const reader = response.body!.getReader();
    const pendingRead = reader.read();
    await Promise.resolve();
    await reader.cancel("consumer stopped");
    await pendingRead;

    expect(warm.consumeCacheCorruption()).toBe(false);
    expect(cache.has(asset.url)).toBe(true);
  });

  it("never caches a network body that fails its pinned integrity", async () => {
    const storage = new MemoryCacheStorage();
    const expectedBytes = byteSequence(29, 17);
    const badBytes = expectedBytes.slice();
    const finalIndex = badBytes.length - 1;
    badBytes[finalIndex] = badBytes[finalIndex]! ^ 0x80;
    const asset = await modelAsset(
      "https://models.example/v1/bad-network.bin",
      expectedBytes,
    );
    const networkCaches: Array<RequestCache | undefined> = [];
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async (_input, init) => {
        networkCaches.push(init?.cache);
        return responseFromChunks(
          badBytes.subarray(0, 13),
          badBytes.subarray(13),
        );
      },
    });

    await expect(
      (await session.fetch(asset.url)).arrayBuffer(),
    ).rejects.toThrow("Model asset SHA-256 mismatch");
    await session.settle();

    expect(networkCaches).toEqual(["no-store"]);
    expect(session.consumeCacheCorruption()).toBe(false);
    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 0,
      sizeBytes: 0,
    });
    expect(storage.onlyModelCache().has(asset.url)).toBe(false);
  });

  it("continues streaming when cache.put fails with a quota error", async () => {
    const storage = new MemoryCacheStorage();
    storage.putError = new DOMException("quota exhausted", "QuotaExceededError");
    const bytes = byteSequence(37, 3);
    const asset = await modelAsset("https://models.example/v1/quota.bin", bytes);
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () =>
        responseFromChunks(bytes.subarray(0, 4), bytes.subarray(4, 21), bytes.subarray(21)),
    });

    const received = new Uint8Array(
      await (await session.fetch(asset.url)).arrayBuffer(),
    );
    await session.settle();

    expect(received).toEqual(bytes);
    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 0,
      sizeBytes: 0,
    });
  });

  it("continues without managed caching when Cache Storage cannot open", async () => {
    const storage = new MemoryCacheStorage();
    storage.openError = new DOMException(
      "storage unavailable",
      "InvalidStateError",
    );
    const bytes = byteSequence(17, 47);
    const asset = await modelAsset(
      "https://models.example/v1/no-cache.bin",
      bytes,
    );
    const networkCaches: Array<RequestCache | undefined> = [];
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async (_input, init) => {
        networkCaches.push(init?.cache);
        return new Response(bytes);
      },
    });

    const received = new Uint8Array(
      await (await session.fetch(asset.url)).arrayBuffer(),
    );
    await session.settle();

    expect(received).toEqual(bytes);
    expect(networkCaches).toEqual([undefined]);
    expect(storage.modelCaches()).toEqual([]);
  });

  it("cancels an abandoned response when model initialization fails early", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(43, 13);
    const asset = await modelAsset(
      "https://models.example/v1/abandoned.bin",
      bytes,
    );
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () =>
        new Response(
          new ReadableStream<Uint8Array<ArrayBuffer>>({
            start(controller) {
              controller.enqueue(bytes.subarray(0, 8));
            },
          }),
        ),
    });

    const abandoned = await session.fetch(asset.url);
    await Promise.resolve();
    await session.cancelInFlight(new Error("model loader rejected"));
    await session.settle();

    await expect(abandoned.arrayBuffer()).rejects.toThrow(
      "model loader rejected",
    );
    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 0,
      sizeBytes: 0,
    });
  });

  it("reuses a verified sibling asset after another model request fails", async () => {
    const storage = new MemoryCacheStorage();
    const manifestBytes = byteSequence(19, 67);
    const weightsBytes = byteSequence(41, 71);
    const failingBytes = byteSequence(23, 73);
    const manifest = await modelAsset(
      "https://models.example/v1/manifest.json",
      manifestBytes,
    );
    const weights = await modelAsset(
      "https://models.example/v1/weights.bin",
      weightsBytes,
    );
    const failing = await modelAsset(
      "https://models.example/v1/failing.bin",
      failingBytes,
    );
    const networkCalls = new Map<string, number>();
    const session = new ModelCacheSession(manifest, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async (input) => {
        const url = requestUrl(input);
        networkCalls.set(url, (networkCalls.get(url) ?? 0) + 1);
        if (url === failing.url) throw new TypeError("temporary network failure");
        return new Response(url === manifest.url ? manifestBytes : weightsBytes);
      },
    });
    session.registerAssets([weights, failing]);

    await (await session.fetch(weights.url)).arrayBuffer();
    await session.settle();
    await expect(session.fetch(failing.url)).rejects.toThrow(
      "temporary network failure",
    );
    await session.cancelInFlight(new Error("restart complete model load"));

    expect(
      new Uint8Array(await (await session.fetch(weights.url)).arrayBuffer()),
    ).toEqual(weightsBytes);
    expect(networkCalls.get(weights.url)).toBe(1);
    expect(networkCalls.get(failing.url)).toBe(1);
  });
});

describe("ModelCacheSession with the Cross-Origin Storage bundle", () => {
  it("moves a cold download into the bundle and keeps no other copy", async () => {
    const fixture = await bundleFixture("cold");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });

    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests).toEqual([
      { value: fixture.identity.sha256, create: false, origins: undefined },
      { value: fixture.identity.sha256, create: true, origins: "*" },
    ]);
    expect(shared.files.get(fixture.identity.sha256)).toEqual(fixture.bundle);
    expect(fixture.networkCacheModes).toEqual([
      "no-store",
      "no-store",
      "no-store",
    ]);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("assembles the bundle from files that arrive out of order", async () => {
    const fixture = await bundleFixture("out-of-order");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });

    for (const asset of [
      fixture.parts[1]!,
      fixture.manifest,
      fixture.parts[0]!,
    ]) {
      await (await session.fetch(asset.url)).arrayBuffer();
    }
    await finishLoad(session);

    expect(shared.files.get(fixture.identity.sha256)).toEqual(fixture.bundle);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("serves another origin from one bundle lookup without the network", async () => {
    const fixture = await bundleFixture("reuse");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    shared.files.set(fixture.identity.sha256, fixture.bundle.slice());
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });

    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(fixture.networkCalls).toBe(0);
    expect(shared.requests).toHaveLength(1);
    expect(session.consumeCacheCorruption()).toBe(false);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("moves a Cache Storage copy into the bundle and deletes it", async () => {
    const fixture = await bundleFixture("legacy");
    const storage = new MemoryCacheStorage();
    const legacy = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: null,
      fetch: fixture.network(),
    });
    await readAll(legacy, fixture);
    await legacy.settle();
    expect(storage.onlyModelCache().has(fixture.manifest.url)).toBe(true);

    const shared = new MemoryCrossOriginStorage();
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: async () => {
        throw new Error("a cached load must not fetch");
      },
    });
    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests.map(({ create }) => create)).toEqual([false, true]);
    expect(shared.files.get(fixture.identity.sha256)).toEqual(fixture.bundle);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("deletes a Cache Storage copy that duplicates an existing bundle", async () => {
    const fixture = await bundleFixture("duplicate");
    const storage = new MemoryCacheStorage();
    const legacy = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: null,
      fetch: fixture.network(),
    });
    await readAll(legacy, fixture);
    await legacy.settle();

    const shared = new MemoryCrossOriginStorage();
    shared.files.set(fixture.identity.sha256, fixture.bundle.slice());
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });
    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests.map(({ create }) => create)).toEqual([false]);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("keeps a Cache Storage copy when the bundle cannot be written", async () => {
    const fixture = await bundleFixture("legacy-failure");
    const storage = new MemoryCacheStorage();
    const legacy = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: null,
      fetch: fixture.network(),
    });
    await readAll(legacy, fixture);
    await legacy.settle();

    const shared = new MemoryCrossOriginStorage();
    shared.writeError = new DOMException("quota exhausted", "QuotaExceededError");
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });
    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.files.size).toBe(0);
    expect(shared.aborts).toBe(1);
    expect(await inspectStoredModels(storage.asCacheStorage())).toMatchObject({
      assetCount: 3,
    });
  });

  it("satisfies cache-only loads from the bundle", async () => {
    const fixture = await bundleFixture("cache-only");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    const cacheOnlySession = () =>
      bundleSession(fixture, {
        cacheStorage: storage.asCacheStorage(),
        crossOriginStorage: shared,
        cacheOnly: true,
        fetch: fixture.network(),
      });

    await expect(
      cacheOnlySession().requireCachedAssets(fixture.parts),
    ).rejects.toMatchObject({ name: "ModelNotCachedError" });

    shared.files.set(fixture.identity.sha256, fixture.bundle.slice());
    shared.requests.length = 0;
    const session = cacheOnlySession();
    await expect(
      session.requireCachedAssets(fixture.parts),
    ).resolves.toBeUndefined();
    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests).toHaveLength(1);
    expect(fixture.networkCalls).toBe(0);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("skips a damaged bundle slice and rewrites the bundle", async () => {
    const fixture = await bundleFixture("damaged");
    const shared = new MemoryCrossOriginStorage();
    const damaged = fixture.bundle.slice();
    const offset = fixture.manifest.byteLength + 3;
    damaged[offset] = damaged[offset]! ^ 0xff;
    shared.files.set(fixture.identity.sha256, damaged);
    const storage = new MemoryCacheStorage();
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });
    const [first, second] = fixture.parts as [ModelCacheAsset, ModelCacheAsset];

    await (await session.fetch(fixture.manifest.url)).arrayBuffer();
    await expect(
      (await session.fetch(first.url)).arrayBuffer(),
    ).rejects.toThrow("Cross-Origin Storage model asset SHA-256 mismatch");
    expect(session.consumeCacheCorruption()).toBe(true);
    expect(
      new Uint8Array(await (await session.fetch(first.url)).arrayBuffer()),
    ).toEqual(fixture.bodies[1]);
    expect(
      new Uint8Array(await (await session.fetch(second.url)).arrayBuffer()),
    ).toEqual(fixture.bodies[2]);
    await finishLoad(session);

    expect(fixture.networkCalls).toBe(1);
    expect(shared.files.get(fixture.identity.sha256)).toEqual(fixture.bundle);
  });

  it("resumes an interrupted download from staged files", async () => {
    const fixture = await bundleFixture("resume");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    const second = fixture.parts[1]!;
    const interrupted = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: async (input, init) => {
        if (requestUrl(input) === second.url) throw new TypeError("offline");
        return fixture.network()(input, init);
      },
    });
    await (await interrupted.fetch(fixture.manifest.url)).arrayBuffer();
    await (await interrupted.fetch(fixture.parts[0]!.url)).arrayBuffer();
    await expect(interrupted.fetch(second.url)).rejects.toThrow("offline");
    await finishLoad(interrupted);
    expect(fixture.networkCalls).toBe(2);
    expect(shared.files.size).toBe(0);

    const resumed = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });
    await expect(readAll(resumed, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(resumed);

    expect(fixture.networkCalls).toBe(3);
    expect(shared.files.get(fixture.identity.sha256)).toEqual(fixture.bundle);
    expect(await storage.asCacheStorage().keys()).toEqual([]);
  });

  it("keeps staged files and writes no bundle after a load that fails", async () => {
    const fixture = await bundleFixture("failed-load");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    const second = fixture.parts[1]!;
    const bad = fixture.bodies[2]!.slice();
    bad[0] = bad[0]! ^ 0x01;
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: async (input, init) =>
        requestUrl(input) === second.url
          ? new Response(bad)
          : fixture.network()(input, init),
    });

    await (await session.fetch(fixture.manifest.url)).arrayBuffer();
    await (await session.fetch(fixture.parts[0]!.url)).arrayBuffer();
    await expect(
      (await session.fetch(second.url)).arrayBuffer(),
    ).rejects.toThrow("Model asset SHA-256 mismatch");
    await finishLoad(session);

    expect(shared.requests.filter(({ create }) => create)).toEqual([]);
    expect(shared.files.size).toBe(0);
    expect(await inspectStoredModels(storage.asCacheStorage())).toMatchObject({
      assetCount: 2,
    });
  });

  it("disables the bundle when its layout misses the pinned length", async () => {
    const fixture = await bundleFixture("mismatch");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    shared.files.set(fixture.identity.sha256, fixture.bundle.slice());
    const session = new ModelCacheSession(fixture.manifest, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      crossOriginStorageBundle: {
        ...fixture.identity,
        byteLength: fixture.identity.byteLength + 1,
      },
      fetch: fixture.network(),
    });
    session.setCrossOriginStorageBundleLayout(fixture.parts);

    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests).toEqual([]);
    expect(fixture.networkCalls).toBe(3);
    expect(session.maximumCorruptionRetries).toBe(3);
    expect(await inspectStoredModels(storage.asCacheStorage())).toMatchObject({
      assetCount: 3,
    });
  });

  it("falls back to Cache Storage when Cross-Origin Storage refuses", async () => {
    const fixture = await bundleFixture("refused");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    shared.requestError = new DOMException("blocked", "NotAllowedError");
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });

    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.requests).toHaveLength(1);
    expect(await inspectStoredModels(storage.asCacheStorage())).toMatchObject({
      assetCount: 3,
    });
  });

  it("keeps the staged copy when the bundle write fails", async () => {
    const fixture = await bundleFixture("write-failure");
    const storage = new MemoryCacheStorage();
    const shared = new MemoryCrossOriginStorage();
    shared.writeError = new DOMException("quota exhausted", "QuotaExceededError");
    const session = bundleSession(fixture, {
      cacheStorage: storage.asCacheStorage(),
      crossOriginStorage: shared,
      fetch: fixture.network(),
    });

    await expect(readAll(session, fixture)).resolves.toEqual(fixture.bodies);
    await finishLoad(session);

    expect(shared.files.size).toBe(0);
    expect(shared.aborts).toBe(1);
    expect(await inspectStoredModels(storage.asCacheStorage())).toMatchObject({
      assetCount: 3,
    });
  });

  it("detects navigator.crossOriginStorage and honors opt-outs", async () => {
    const fixture = await bundleFixture("default");
    const shared = new MemoryCrossOriginStorage();
    shared.files.set(fixture.identity.sha256, fixture.bundle.slice());
    vi.stubGlobal("navigator", { crossOriginStorage: shared });
    try {
      const detected = bundleSession(fixture, { fetch: fixture.network() });
      await expect(readAll(detected, fixture)).resolves.toEqual(fixture.bodies);
      expect(fixture.networkCalls).toBe(0);

      const disabled = bundleSession(fixture, {
        crossOriginStorage: null,
        fetch: fixture.network(),
      });
      await readAll(disabled, fixture);
      expect(fixture.networkCalls).toBe(3);

      const unpinned = new ModelCacheSession(fixture.manifest, {
        fetch: fixture.network(),
      });
      unpinned.setCrossOriginStorageBundleLayout(fixture.parts);
      await readAll(unpinned, fixture);
      expect(fixture.networkCalls).toBe(6);
      expect(shared.requests).toHaveLength(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("crossOriginStorageBundleLayout()", () => {
  it("lists tensor shards and tokenizer.json in manifest order", () => {
    const file = (name: string) => ({
      name,
      byteLength: name.length,
      sha256: "c".repeat(64),
    });
    expect(
      crossOriginStorageBundleLayout(
        "https://models.example/v1/fp16/manifest.json",
        {
          files: [
            file("decoder.bin"),
            file("encoder-layer-00.bin"),
            file("model_config.yaml"),
            file("preprocessor.bin"),
            file("tokenizer.json"),
            file("tokenizer.model"),
          ],
          tensors: {
            "preprocessor.window": { shard: "preprocessor.bin" },
            "decoder.weight": { shard: "decoder.bin" },
            "encoder.layers.0.weight": { shard: "encoder-layer-00.bin" },
          },
        },
      ).map(({ url }) => url),
    ).toEqual([
      "https://models.example/v1/fp16/decoder.bin",
      "https://models.example/v1/fp16/encoder-layer-00.bin",
      "https://models.example/v1/fp16/preprocessor.bin",
      "https://models.example/v1/fp16/tokenizer.json",
    ]);
  });
});

describe("inspectCrossOriginStorageBundles()", () => {
  const bundles = {
    fp16: { byteLength: 405, sha256: "d".repeat(64) },
    fp32: { byteLength: 441, sha256: "e".repeat(64) },
  };

  it("reports unsupported without the API", async () => {
    expect(await inspectCrossOriginStorageBundles(bundles, undefined)).toEqual({
      supported: false,
      bundles: [],
      sizeBytes: 0,
    });
  });

  it("reports stored bundles with one lookup each and no reads", async () => {
    const shared = new MemoryCrossOriginStorage();
    shared.files.set(bundles.fp32.sha256, new Uint8Array(1));

    expect(await inspectCrossOriginStorageBundles(bundles, shared)).toEqual({
      supported: true,
      bundles: ["fp32"],
      sizeBytes: 441,
    });
    expect(shared.requests).toEqual([
      { value: bundles.fp16.sha256, create: false, origins: undefined },
      { value: bundles.fp32.sha256, create: false, origins: undefined },
    ]);
  });

  it("reports a refused lookup as absent", async () => {
    const shared = new MemoryCrossOriginStorage();
    shared.files.set(bundles.fp16.sha256, new Uint8Array(1));
    shared.requestError = new DOMException("blocked", "NotAllowedError");

    expect(await inspectCrossOriginStorageBundles(bundles, shared)).toEqual({
      supported: true,
      bundles: [],
      sizeBytes: 0,
    });
  });
});

describe("stored model cache management", () => {
  it("reports unsupported when Cache Storage is unavailable", async () => {
    expect(await inspectStoredModels()).toEqual({
      supported: false,
      assetCount: 0,
      sizeBytes: 0,
    });
    await expect(clearStoredModels()).resolves.toBeUndefined();
  });

  it("totals only managed, well-formed entries across model cache versions", async () => {
    const storage = new MemoryCacheStorage();
    const firstBytes = byteSequence(12, 19);
    const secondBytes = byteSequence(21, 23);
    const thirdBytes = byteSequence(8, 29);
    const first = await modelAsset(
      "https://models.example/v1/manifest.json",
      firstBytes,
    );
    const second = await modelAsset(
      "https://models.example/v1/weights.bin",
      secondBytes,
    );
    const third = await modelAsset(
      "https://models.example/v2/manifest.json",
      thirdBytes,
    );
    const firstSession = new ModelCacheSession(first, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async (input) =>
        new Response(requestUrl(input) === first.url ? firstBytes : secondBytes),
    });
    firstSession.registerAssets([second]);
    await Promise.all([
      (await firstSession.fetch(first.url)).arrayBuffer(),
      (await firstSession.fetch(second.url)).arrayBuffer(),
    ]);
    await firstSession.settle();

    const secondSession = new ModelCacheSession(third, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(thirdBytes),
    });
    await (await secondSession.fetch(third.url)).arrayBuffer();
    await secondSession.settle();

    const firstCache = storage.modelCaches()[0]!;
    await firstCache.seed(
      "https://models.example/unmanaged.bin",
      new Response(byteSequence(40, 31), {
        headers: { "content-length": "40" },
      }),
    );
    const unrelated = storage.cache("site-assets");
    await unrelated.seed(
      "https://app.example/app.js",
      new Response(byteSequence(99, 37)),
    );

    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 3,
      sizeBytes:
        firstBytes.byteLength + secondBytes.byteLength + thirdBytes.byteLength,
    });
  });

  it("clears every model cache idempotently without touching other caches", async () => {
    const storage = new MemoryCacheStorage();
    const bytes = byteSequence(16, 41);
    const asset = await modelAsset(
      "https://models.example/v1/clear.bin",
      bytes,
    );
    const session = new ModelCacheSession(asset, {
      cacheStorage: storage.asCacheStorage(),
      fetch: async () => new Response(bytes),
    });
    await (await session.fetch(asset.url)).arrayBuffer();
    await session.settle();
    await storage.cache("site-assets").seed(
      "https://app.example/app.js",
      new Response(byteSequence(7, 43)),
    );

    await clearStoredModels(storage.asCacheStorage());
    await clearStoredModels(storage.asCacheStorage());

    expect(await storage.asCacheStorage().keys()).toEqual(["site-assets"]);
    expect(await inspectStoredModels(storage.asCacheStorage())).toEqual({
      supported: true,
      assetCount: 0,
      sizeBytes: 0,
    });
    expect(storage.cache("site-assets").has("https://app.example/app.js")).toBe(
      true,
    );
  });
});

interface StoredResponse {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly bodyNull: boolean;
  readonly status: number;
  readonly statusText: string;
  readonly headers: [string, string][];
}

class MemoryCache {
  private readonly entries = new Map<string, StoredResponse>();
  private readonly pendingReads = new Set<string>();
  private readonly readErrors = new Map<string, Error>();
  private matchError: Error | undefined;

  public constructor(private readonly storage: MemoryCacheStorage) {}

  public asCache(): Cache {
    return this as unknown as Cache;
  }

  public async match(
    request: RequestInfo | URL,
    _options?: CacheQueryOptions,
  ): Promise<Response | undefined> {
    if (this.matchError !== undefined) throw this.matchError;
    const url = requestUrl(request);
    const entry = this.entries.get(url);
    if (entry === undefined) return undefined;
    if (this.pendingReads.has(url)) return restorePendingResponse(entry);
    const error = this.readErrors.get(url);
    return error === undefined
      ? restoreResponse(entry)
      : restoreErroredResponse(entry, error);
  }

  public async put(
    request: RequestInfo | URL,
    response: Response,
  ): Promise<void> {
    if (this.storage.putError !== undefined) {
      throw this.storage.putError;
    }
    const stored = await snapshotResponse(response);
    const url = requestUrl(request);
    this.pendingReads.delete(url);
    this.readErrors.delete(url);
    this.entries.set(url, stored);
  }

  public async delete(
    request: RequestInfo | URL,
    _options?: CacheQueryOptions,
  ): Promise<boolean> {
    const url = requestUrl(request);
    this.pendingReads.delete(url);
    this.readErrors.delete(url);
    return this.entries.delete(url);
  }

  public async keys(
    request?: RequestInfo | URL,
    _options?: CacheQueryOptions,
  ): Promise<readonly Request[]> {
    if (request !== undefined) {
      const url = requestUrl(request);
      return this.entries.has(url) ? [new Request(url)] : [];
    }
    return [...this.entries.keys()].map((url) => new Request(url));
  }

  public async seed(url: string, response: Response): Promise<void> {
    this.entries.set(requestUrl(url), await snapshotResponse(response));
  }

  public replaceBody(url: string, bytes: Uint8Array<ArrayBuffer>): void {
    const key = requestUrl(url);
    const entry = this.entries.get(key);
    if (entry === undefined) {
      throw new Error(`No cached response for ${key}`);
    }
    this.entries.set(key, {
      ...entry,
      bytes: bytes.slice(),
      bodyNull: false,
    });
  }

  public replaceBodyWithNull(url: string): void {
    const key = requestUrl(url);
    const entry = this.entries.get(key);
    if (entry === undefined) {
      throw new Error(`No cached response for ${key}`);
    }
    this.entries.set(key, {
      ...entry,
      bytes: new Uint8Array(),
      bodyNull: true,
    });
  }

  public replaceBodyWithError(url: string, error: Error): void {
    const key = requestUrl(url);
    if (!this.entries.has(key)) {
      throw new Error(`No cached response for ${key}`);
    }
    this.readErrors.set(key, error);
  }

  public replaceBodyWithPendingRead(url: string): void {
    const key = requestUrl(url);
    if (!this.entries.has(key)) {
      throw new Error(`No cached response for ${key}`);
    }
    this.pendingReads.add(key);
  }

  public failMatches(error: Error): void {
    this.matchError = error;
  }

  public has(url: string): boolean {
    return this.entries.has(requestUrl(url));
  }

  public async bytes(url: string): Promise<Uint8Array<ArrayBuffer> | undefined> {
    const response = await this.match(url);
    return response === undefined
      ? undefined
      : new Uint8Array(await response.arrayBuffer());
  }
}

class MemoryCacheStorage {
  private readonly caches = new Map<string, MemoryCache>();
  public openError: Error | undefined;
  public putError: Error | undefined;

  public asCacheStorage(): CacheStorage {
    return this as unknown as CacheStorage;
  }

  public async open(name: string): Promise<Cache> {
    if (this.openError !== undefined) throw this.openError;
    return this.cache(name).asCache();
  }

  public async keys(): Promise<string[]> {
    return [...this.caches.keys()];
  }

  public async delete(name: string): Promise<boolean> {
    return this.caches.delete(name);
  }

  public cache(name: string): MemoryCache {
    let cache = this.caches.get(name);
    if (cache === undefined) {
      cache = new MemoryCache(this);
      this.caches.set(name, cache);
    }
    return cache;
  }

  public modelCaches(): MemoryCache[] {
    return [...this.caches.entries()]
      .filter(([name]) => name.startsWith(MODEL_CACHE_PREFIX))
      .map(([, cache]) => cache);
  }

  public onlyModelCache(): MemoryCache {
    const caches = this.modelCaches();
    if (caches.length !== 1) {
      throw new Error(`Expected one model cache, found ${caches.length}`);
    }
    return caches[0]!;
  }
}

type CrossOriginStorageRequest = Parameters<
  CrossOriginStorageManager["getFileHandle"]
>;

interface BundleFixture {
  readonly manifest: ModelCacheAsset;
  readonly parts: readonly ModelCacheAsset[];
  /** The manifest body followed by each part body, in bundle order. */
  readonly bodies: readonly Uint8Array<ArrayBuffer>[];
  readonly bundle: Uint8Array<ArrayBuffer>;
  readonly identity: CrossOriginStorageBundle;
  /** Network requests made through every `network()` fetch so far. */
  readonly networkCalls: number;
  /** The `cache` mode of every network request so far. */
  readonly networkCacheModes: readonly (RequestCache | undefined)[];
  network(): typeof fetch;
}

async function bundleFixture(name: string): Promise<BundleFixture> {
  const base = `https://models.example/v1/${name}/`;
  const bodies = [
    byteSequence(13, 3),
    byteSequence(29, 7),
    byteSequence(21, 11),
  ];
  const [manifest, ...parts] = await Promise.all(
    ["manifest.json", "a.bin", "b.bin"].map((file, index) =>
      modelAsset(`${base}${file}`, bodies[index]!),
    ),
  );
  const bundle = new Uint8Array(
    bodies.reduce((total, body) => total + body.byteLength, 0),
  );
  let offset = 0;
  for (const body of bodies) {
    bundle.set(body, offset);
    offset += body.byteLength;
  }
  const byUrl = new Map(
    [manifest!, ...parts].map((asset, index) => [asset.url, bodies[index]!]),
  );
  const networkCacheModes: (RequestCache | undefined)[] = [];
  return {
    manifest: manifest!,
    parts,
    bodies,
    bundle,
    identity: { byteLength: bundle.byteLength, sha256: await sha256Hex(bundle) },
    get networkCalls() {
      return networkCacheModes.length;
    },
    networkCacheModes,
    network: () => async (input, init) => {
      networkCacheModes.push(init?.cache);
      const body = byUrl.get(requestUrl(input));
      if (body === undefined) {
        throw new Error(`Unexpected request ${requestUrl(input)}`);
      }
      return responseFromChunks(body.slice(0, 5), body.slice(5));
    },
  };
}

function bundleSession(
  fixture: BundleFixture,
  options: NonNullable<ConstructorParameters<typeof ModelCacheSession>[1]>,
): ModelCacheSession {
  const session = new ModelCacheSession(fixture.manifest, {
    crossOriginStorageBundle: fixture.identity,
    ...options,
  });
  session.setCrossOriginStorageBundleLayout(fixture.parts);
  return session;
}

async function readAll(
  session: ModelCacheSession,
  fixture: BundleFixture,
): Promise<Uint8Array<ArrayBuffer>[]> {
  const bodies: Uint8Array<ArrayBuffer>[] = [];
  for (const asset of [fixture.manifest, ...fixture.parts]) {
    const response = await session.fetch(asset.url);
    bodies.push(new Uint8Array(await response.arrayBuffer()));
  }
  return bodies;
}

/** Finish a load the way the engine does, and wait for the bundle move. */
async function finishLoad(session: ModelCacheSession): Promise<void> {
  await session.settle();
  await session.settleCrossOriginStorage();
}

/**
 * Content-addressed storage with the observable behavior of native
 * Cross-Origin Storage: lookups miss with `NotFoundError`, writes may be
 * positioned and out of order, nothing is stored until `close()` verifies the
 * complete digest, and `abort()` discards the write.
 */
class MemoryCrossOriginStorage implements CrossOriginStorageManager {
  public readonly files = new Map<string, Uint8Array<ArrayBuffer>>();
  public readonly requests: Array<{
    readonly value: string;
    readonly create: boolean;
    readonly origins: "*" | readonly string[] | undefined;
  }> = [];
  public requestError: Error | undefined;
  public writeError: Error | undefined;
  public aborts = 0;

  public async getFileHandle(
    hash: CrossOriginStorageRequest[0],
    options: CrossOriginStorageRequest[1] = {},
  ): ReturnType<CrossOriginStorageManager["getFileHandle"]> {
    const create = options.create === true;
    this.requests.push({ value: hash.value, create, origins: options.origins });
    if (this.requestError !== undefined) throw this.requestError;
    if (!create && !this.files.has(hash.value)) {
      throw new DOMException("not in cross-origin storage", "NotFoundError");
    }
    return {
      getFile: async () => new Blob([this.files.get(hash.value)!]),
      createWritable: async () => {
        let buffer = new Uint8Array(0);
        return {
          write: async ({ position, data }) => {
            if (this.writeError !== undefined) throw this.writeError;
            const bytes =
              data instanceof Blob
                ? new Uint8Array(await data.arrayBuffer())
                : new Uint8Array(data as Uint8Array<ArrayBuffer>);
            if (position + bytes.byteLength > buffer.byteLength) {
              const grown = new Uint8Array(position + bytes.byteLength);
              grown.set(buffer);
              buffer = grown;
            }
            buffer.set(bytes, position);
          },
          close: async () => {
            if ((await sha256Hex(buffer)) !== hash.value) {
              throw new DOMException("digest mismatch", "DataError");
            }
            this.files.set(hash.value, buffer);
          },
          abort: async () => {
            this.aborts += 1;
          },
        };
      },
    };
  }
}

async function modelAsset(
  url: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<ModelCacheAsset> {
  return {
    url,
    byteLength: bytes.byteLength,
    sha256: await sha256Hex(bytes),
  };
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(
    await globalThis.crypto.subtle.digest("SHA-256", bytes),
  );
  return [...digest]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function seedManagedAssets(
  storage: MemoryCacheStorage,
  manifest: ModelCacheAsset,
  entries: readonly (readonly [
    asset: ModelCacheAsset,
    bytes: Uint8Array<ArrayBuffer>,
  ])[],
): Promise<void> {
  const bodies = new Map(
    entries.map(([asset, bytes]) => [requestUrl(asset.url), bytes]),
  );
  const session = new ModelCacheSession(manifest, {
    cacheStorage: storage.asCacheStorage(),
    fetch: async (input) => {
      const url = requestUrl(input);
      const bytes = bodies.get(url);
      if (bytes === undefined) throw new Error(`Unexpected request ${url}`);
      return new Response(bytes);
    },
  });
  session.registerAssets(entries.map(([asset]) => asset));
  for (const [asset] of entries) {
    await (await session.fetch(asset.url)).arrayBuffer();
  }
  await session.settle();
}

function requestUrl(input: RequestInfo | URL): string {
  const raw = input instanceof Request ? input.url : input.toString();
  return new URL(raw, "http://localhost/").href;
}

function responseFromChunks(
  ...chunks: readonly Uint8Array<ArrayBuffer>[]
): Response {
  let index = 0;
  return new Response(
    new ReadableStream<Uint8Array<ArrayBuffer>>({
      pull(controller) {
        const chunk = chunks[index];
        if (chunk === undefined) {
          controller.close();
          return;
        }
        index += 1;
        controller.enqueue(chunk);
      },
    }),
    {
      status: 200,
      headers: { "content-type": "application/octet-stream" },
    },
  );
}

function gatedResponse(
  head: Uint8Array<ArrayBuffer>,
  tail: Uint8Array<ArrayBuffer>,
  releaseTail: Promise<void>,
): Response {
  let sentHead = false;
  return new Response(
    new ReadableStream<Uint8Array<ArrayBuffer>>({
      async pull(controller) {
        if (!sentHead) {
          sentHead = true;
          controller.enqueue(head);
          return;
        }
        await releaseTail;
        controller.enqueue(tail);
        controller.close();
      },
    }),
    {
      status: 200,
      headers: { "content-type": "application/octet-stream" },
    },
  );
}

function byteSequence(length: number, seed: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) {
    bytes[index] = (seed + index * 47) & 0xff;
  }
  return bytes;
}

async function snapshotResponse(response: Response): Promise<StoredResponse> {
  const bodyNull = response.body === null;
  return {
    bytes: bodyNull
      ? new Uint8Array()
      : new Uint8Array(await response.arrayBuffer()),
    bodyNull,
    status: response.status,
    statusText: response.statusText,
    headers: [...response.headers.entries()],
  };
}

function restoreResponse(stored: StoredResponse): Response {
  return new Response(stored.bodyNull ? null : stored.bytes.slice(), {
    status: stored.status,
    statusText: stored.statusText,
    headers: stored.headers,
  });
}

function restoreErroredResponse(
  stored: StoredResponse,
  error: Error,
): Response {
  return new Response(
    new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        controller.error(error);
      },
    }),
    {
      status: stored.status,
      statusText: stored.statusText,
      headers: stored.headers,
    },
  );
}

function restorePendingResponse(stored: StoredResponse): Response {
  return new Response(
    new ReadableStream<Uint8Array<ArrayBuffer>>({
      pull() {
        // Intentionally leave the read pending until its consumer cancels.
      },
    }),
    {
      status: stored.status,
      statusText: stored.statusText,
      headers: stored.headers,
    },
  );
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T | PromiseLike<T>) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}
