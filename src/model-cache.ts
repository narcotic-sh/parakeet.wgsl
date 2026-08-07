import { IncrementalSha256 } from "./model/sha256";
import {
  modelBodyFailure,
  truncatedModelBody,
} from "./model-transport";

const MODEL_CACHE_FAMILY_PREFIX = "parakeet.wgsl:models:";
const MODEL_CACHE_PREFIX = `${MODEL_CACHE_FAMILY_PREFIX}v1:`;
const CACHE_FORMAT_HEADER = "x-parakeet-wgsl-cache-format";
const CACHE_SHA256_HEADER = "x-parakeet-wgsl-sha256";
const CACHE_BYTE_LENGTH_HEADER = "x-parakeet-wgsl-byte-length";
const CACHE_FORMAT = "1";

export interface ModelCacheAsset {
  readonly url: string;
  readonly byteLength: number;
  readonly sha256: string;
}

export interface StoredModelCacheInfo {
  readonly supported: boolean;
  readonly assetCount: number;
  readonly sizeBytes: number;
}

interface ModelCacheSessionOptions {
  readonly cacheStorage?: CacheStorage;
  readonly fetch?: typeof fetch;
  /** Refuse every network fallback and use only the exact managed cache. */
  readonly cacheOnly?: boolean;
}

/** @internal */
export class ModelNotCachedError extends Error {
  readonly code = "MODEL_NOT_CACHED";
  readonly url: string | undefined;

  constructor(url?: string, detail?: string) {
    super(
      detail ??
        (url === undefined
          ? "The selected Parakeet model is not fully cached"
          : `Parakeet model asset is not cached: ${url}`),
    );
    this.name = "ModelNotCachedError";
    this.url = url;
  }
}

interface ActiveNetworkLoad {
  cancel(reason: unknown): Promise<void>;
}

/**
 * One model-initialization cache context. Every response is pinned before it
 * enters Cache Storage, and the large Parakeet shards remain streaming.
 *
 * @internal
 */
export class ModelCacheSession {
  readonly fetch: typeof fetch;

  private readonly assets = new Map<string, ModelCacheAsset>();
  private readonly activeNetworkLoads = new Set<ActiveNetworkLoad>();
  private readonly ignoredCacheUrls = new Set<string>();
  private readonly pendingCacheWrites = new Set<Promise<void>>();
  private readonly cachePromise: Promise<Cache | undefined>;
  private readonly networkFetch: typeof fetch;
  private readonly cacheOnly: boolean;
  private cacheCorruptionDetected = false;

  constructor(
    manifest: ModelCacheAsset,
    options: ModelCacheSessionOptions = {},
  ) {
    assertAsset(manifest);
    this.registerAssets([manifest]);
    this.cacheOnly = options.cacheOnly === true;
    this.networkFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    const storage =
      options.cacheStorage ??
      (typeof globalThis.caches === "undefined"
        ? undefined
        : globalThis.caches);
    const cacheName =
      `${MODEL_CACHE_PREFIX}${manifest.sha256.toLowerCase()}`;
    this.cachePromise =
      storage === undefined
        ? Promise.resolve(undefined)
        : this.cacheOnly
          ? openExistingCache(storage, cacheName)
          : storage.open(cacheName).catch(() => undefined);
    this.fetch = this.fetchAsset.bind(this) as typeof fetch;
  }

  registerAssets(assets: readonly ModelCacheAsset[]): void {
    for (const asset of assets) {
      assertAsset(asset);
      const url = absoluteUrl(asset.url);
      const normalized = {
        url,
        byteLength: asset.byteLength,
        sha256: asset.sha256.toLowerCase(),
      } satisfies ModelCacheAsset;
      const existing = this.assets.get(url);
      if (
        existing !== undefined &&
        (existing.byteLength !== normalized.byteLength ||
          existing.sha256 !== normalized.sha256)
      ) {
        throw new Error(`Conflicting integrity records for model asset ${url}`);
      }
      this.assets.set(url, normalized);
    }
  }

  /**
   * Require exact managed entries for a known runtime subset without reading
   * their bodies. The later package load remains responsible for streaming
   * and verifying every byte before use.
   */
  async requireCachedAssets(
    assets: readonly ModelCacheAsset[],
  ): Promise<void> {
    this.registerAssets(assets);
    if (assets.length === 0) return;

    const cache = await this.cachePromise;
    if (cache === undefined) {
      throw new ModelNotCachedError();
    }

    for (const requested of assets) {
      const url = absoluteUrl(requested.url);
      const asset = this.assets.get(url)!;
      if (this.ignoredCacheUrls.has(url)) {
        throw new ModelNotCachedError(url);
      }

      let response: Response | undefined;
      try {
        response = await cache.match(url, { ignoreVary: true });
      } catch {
        throw new ModelNotCachedError(
          url,
          `Could not inspect cached Parakeet model asset ${url}`,
        );
      }
      if (
        response === undefined ||
        response.body === null ||
        !isMatchingCacheEntry(response, asset)
      ) {
        throw new ModelNotCachedError(url);
      }
    }
  }

  consumeCacheCorruption(): boolean {
    const detected = this.cacheCorruptionDetected;
    this.cacheCorruptionDetected = false;
    return detected;
  }

  get maximumCorruptionRetries(): number {
    // A corrupt object is ignored for the remainder of this session. Bounding
    // recovery by the selected package inventory permits each distinct cached
    // object to be replaced once without a hard-coded model-file count.
    return this.assets.size;
  }

  async settle(): Promise<void> {
    while (this.pendingCacheWrites.size > 0) {
      await Promise.all([...this.pendingCacheWrites]);
    }
  }

  async cancelInFlight(reason: unknown): Promise<void> {
    await Promise.allSettled(
      [...this.activeNetworkLoads].map((load) => load.cancel(reason)),
    );
    await this.settle();
  }

  private async fetchAsset(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const url = requestUrl(input);
    const asset = this.assets.get(url);
    if (asset === undefined) {
      throw new Error(`Refusing to fetch unpinned model asset ${url}`);
    }
    if (requestMethod(input, init) !== "GET") {
      throw new Error(`Model assets must be fetched with GET: ${url}`);
    }

    const cache = await this.cachePromise;
    if (cache !== undefined && !this.ignoredCacheUrls.has(url)) {
      let cached: Response | undefined;
      try {
        cached = await cache.match(url, { ignoreVary: true });
      } catch {
        if (this.cacheOnly) {
          throw new ModelNotCachedError(
            url,
            `Could not inspect cached Parakeet model asset ${url}`,
          );
        }
        cached = undefined;
      }
      if (cached !== undefined) {
        if (isMatchingCacheEntry(cached, asset) && cached.body !== null) {
          return this.verifiedCacheResponse(cache, url, cached, asset);
        }
        if (this.cacheOnly) throw new ModelNotCachedError(url);
        try {
          await cache.delete(url, { ignoreVary: true });
        } catch {
          // A failed cleanup must not prevent a verified network load.
        }
      }
    }

    if (this.cacheOnly) throw new ModelNotCachedError(url);

    const networkInit =
      cache === undefined
        ? init
        : {
            ...init,
            cache: "no-store" as const,
          };
    const response = await this.networkFetch(input, networkInit);
    if (response.status !== 200 || response.body === null) {
      return response;
    }
    return this.streamingNetworkResponse(cache, url, response, asset);
  }

  private verifiedCacheResponse(
    cache: Cache,
    url: string,
    response: Response,
    asset: ModelCacheAsset,
  ): Response {
    const source = response.body!.getReader();
    const hash = new IncrementalSha256();
    let byteLength = 0;
    let cancelled = false;
    let released = false;
    const releaseSource = (): void => {
      if (released) return;
      released = true;
      source.releaseLock();
    };
    const markCorrupt = async (message: string): Promise<Error> => {
      this.cacheCorruptionDetected = true;
      this.ignoredCacheUrls.add(url);
      try {
        await cache.delete(url, { ignoreVary: true });
      } catch {
        // The ignored URL still guarantees that the retry uses the network.
      }
      return this.cacheOnly
        ? new ModelNotCachedError(url, message)
        : new Error(message);
    };
    const failCorrupt = async (
      controller: ReadableStreamDefaultController<Uint8Array>,
      message: string,
    ): Promise<void> => {
      const error = await markCorrupt(message);
      try {
        await source.cancel(error);
      } catch {
        // A failed cached body is already being discarded.
      } finally {
        releaseSource();
      }
      controller.error(error);
    };
    const verified = new ReadableStream<Uint8Array>({
      pull: async (controller) => {
        let item: ReadableStreamReadResult<Uint8Array>;
        try {
          item = await source.read();
        } catch {
          if (cancelled) {
            releaseSource();
            return;
          }
          await failCorrupt(
            controller,
            `Cached model asset ${url} could not be read`,
          );
          return;
        }
        if (cancelled) {
          releaseSource();
          return;
        }

        if (item.done) {
          if (byteLength !== asset.byteLength) {
            await failCorrupt(
              controller,
              `Cached model asset ${url} has ${byteLength} bytes; expected ${asset.byteLength}`,
            );
            return;
          }
          const actual = hash.digestHex();
          if (actual !== asset.sha256) {
            await failCorrupt(
              controller,
              `Cached model asset SHA-256 mismatch for ${url}: ${actual}`,
            );
            return;
          }
          releaseSource();
          controller.close();
          return;
        }

        const chunk = item.value;
        byteLength += chunk.byteLength;
        if (byteLength > asset.byteLength) {
          await failCorrupt(
            controller,
            `Cached model asset ${url} is longer than its pinned byte length`,
          );
          return;
        }
        hash.update(chunk);
        controller.enqueue(chunk);
      },
      cancel: async (reason) => {
        cancelled = true;
        try {
          await source.cancel(reason);
        } finally {
          releaseSource();
        }
      },
    });
    return responseWithBody(response, verified, asset, false);
  }

  private streamingNetworkResponse(
    cache: Cache | undefined,
    url: string,
    response: Response,
    asset: ModelCacheAsset,
  ): Response {
    const source = response.body!.getReader();
    const consumerPipe = new TransformStream<Uint8Array, Uint8Array>();
    const cachePipe =
      cache === undefined
        ? undefined
        : new TransformStream<Uint8Array, Uint8Array>();
    const consumerWriter = consumerPipe.writable.getWriter();
    const cacheWriter = cachePipe?.writable.getWriter();
    const hash = new IncrementalSha256();
    let byteLength = 0;
    let cacheBranchOpen = cacheWriter !== undefined;
    let cancellation: Promise<void> | undefined;
    const activeLoad: ActiveNetworkLoad = {
      cancel: (reason) => {
        cancellation ??= Promise.allSettled([
          source.cancel(reason),
          consumerWriter.abort(reason),
          ...(cacheWriter === undefined ? [] : [cacheWriter.abort(reason)]),
        ]).then(() => undefined);
        cacheBranchOpen = false;
        return cancellation;
      },
    };
    this.activeNetworkLoads.add(activeLoad);

    const cachePut =
      cache === undefined || cachePipe === undefined || cacheWriter === undefined
        ? undefined
        : cache
            .put(
              url,
              responseWithBody(response, cachePipe.readable, asset, true),
            )
            .catch(async (error) => {
              cacheBranchOpen = false;
              try {
                await cacheWriter.abort(error);
              } catch {
                // The stream may already have been closed or aborted.
              }
            });

    const pump = (async () => {
      try {
        while (true) {
          let item: ReadableStreamReadResult<Uint8Array>;
          try {
            item = await source.read();
          } catch (error) {
            throw modelBodyFailure(url, error);
          }
          if (item.done) break;
          const chunk = item.value;
          byteLength += chunk.byteLength;
          if (byteLength > asset.byteLength) {
            throw new Error(
              `Model asset ${url} is longer than its pinned byte length`,
            );
          }
          hash.update(chunk);

          await consumerWriter.write(chunk);
          if (cacheBranchOpen && cacheWriter !== undefined) {
            try {
              await cacheWriter.write(chunk);
            } catch {
              cacheBranchOpen = false;
            }
          }
        }

        if (byteLength !== asset.byteLength) {
          throw truncatedModelBody(url, byteLength, asset.byteLength);
        }
        const actual = hash.digestHex();
        if (actual !== asset.sha256) {
          throw new Error(`Model asset SHA-256 mismatch for ${url}: ${actual}`);
        }

        await consumerWriter.close();
        if (cacheBranchOpen && cacheWriter !== undefined) {
          try {
            await cacheWriter.close();
          } catch {
            cacheBranchOpen = false;
          }
        }
      } catch (error) {
        await activeLoad.cancel(error);
        throw error;
      } finally {
        this.activeNetworkLoads.delete(activeLoad);
        source.releaseLock();
        consumerWriter.releaseLock();
        cacheWriter?.releaseLock();
      }
    })();

    // The response consumer observes pump failures through its stream. Keep a
    // handled task here as well so failures never become unhandled rejections.
    const completed = Promise.allSettled([
      pump,
      ...(cachePut === undefined ? [] : [cachePut]),
    ]).then(() => undefined);
    this.pendingCacheWrites.add(completed);
    void completed.then(() => this.pendingCacheWrites.delete(completed));

    return responseWithBody(response, consumerPipe.readable, asset, false);
  }
}

export async function inspectStoredModels(
  storage: CacheStorage | undefined = cacheStorage(),
): Promise<StoredModelCacheInfo> {
  if (storage === undefined) {
    return { supported: false, assetCount: 0, sizeBytes: 0 };
  }

  const names = (await storage.keys()).filter((name) =>
    name.startsWith(MODEL_CACHE_FAMILY_PREFIX),
  );
  let assetCount = 0;
  let sizeBytes = 0;
  for (const name of names) {
    const cache = await storage.open(name);
    const keys = await cache.keys();
    for (const key of keys) {
      const response = await cache.match(key, { ignoreVary: true });
      if (response === undefined) continue;
      const length = cacheEntryByteLength(response);
      if (length === undefined) continue;
      assetCount += 1;
      sizeBytes += length;
    }
  }
  return { supported: true, assetCount, sizeBytes };
}

export async function clearStoredModels(
  storage: CacheStorage | undefined = cacheStorage(),
): Promise<void> {
  if (storage === undefined) return;
  const names = await storage.keys();
  await Promise.all(
    names
      .filter((name) => name.startsWith(MODEL_CACHE_FAMILY_PREFIX))
      .map((name) => storage.delete(name)),
  );
}

function cacheStorage(): CacheStorage | undefined {
  return typeof globalThis.caches === "undefined"
    ? undefined
    : globalThis.caches;
}

async function openExistingCache(
  storage: CacheStorage,
  name: string,
): Promise<Cache | undefined> {
  try {
    const names = await storage.keys();
    if (!names.includes(name)) return undefined;
    return await storage.open(name);
  } catch {
    return undefined;
  }
}

function isMatchingCacheEntry(
  response: Response,
  asset: ModelCacheAsset,
): boolean {
  return (
    response.ok &&
    response.headers.get(CACHE_FORMAT_HEADER) === CACHE_FORMAT &&
    response.headers.get(CACHE_SHA256_HEADER)?.toLowerCase() ===
      asset.sha256 &&
    response.headers.get(CACHE_BYTE_LENGTH_HEADER) ===
      String(asset.byteLength)
  );
}

function cacheEntryByteLength(response: Response): number | undefined {
  if (response.headers.get(CACHE_FORMAT_HEADER) !== CACHE_FORMAT) {
    return undefined;
  }
  const value = Number(response.headers.get(CACHE_BYTE_LENGTH_HEADER));
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function responseWithBody(
  source: Response,
  body: ReadableStream<Uint8Array>,
  asset: ModelCacheAsset,
  managedCacheEntry: boolean,
): Response {
  const headers = new Headers(source.headers);
  headers.delete("content-encoding");
  headers.set("content-length", String(asset.byteLength));
  if (managedCacheEntry) {
    headers.set(CACHE_FORMAT_HEADER, CACHE_FORMAT);
    headers.set(CACHE_SHA256_HEADER, asset.sha256);
    headers.set(CACHE_BYTE_LENGTH_HEADER, String(asset.byteLength));
  }
  return new Response(body, {
    status: source.status,
    statusText: source.statusText,
    headers,
  });
}

function requestUrl(input: RequestInfo | URL): string {
  const raw = input instanceof Request ? input.url : input.toString();
  return absoluteUrl(raw);
}

function requestMethod(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): string {
  return (init?.method ?? (input instanceof Request ? input.method : "GET"))
    .toUpperCase();
}

function absoluteUrl(url: string): string {
  return new URL(
    url,
    globalThis.location?.href ?? "http://localhost/",
  ).href;
}

function assertAsset(asset: ModelCacheAsset): void {
  if (!Number.isSafeInteger(asset.byteLength) || asset.byteLength <= 0) {
    throw new TypeError("Model asset byteLength must be a positive integer");
  }
  if (!/^[0-9a-f]{64}$/i.test(asset.sha256)) {
    throw new TypeError(
      "Model asset sha256 must be a 64-character hexadecimal digest",
    );
  }
  absoluteUrl(asset.url);
}
