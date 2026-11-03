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

/**
 * The byte length and SHA-256 of a package's Cross-Origin Storage bundle: the
 * manifest followed by every runtime file, in the order that
 * `crossOriginStorageBundleLayout()` defines. Browsers assemble the bundle
 * from verified files, so it is never downloaded, and one lookup retrieves a
 * whole package.
 *
 * @internal
 */
export interface CrossOriginStorageBundle {
  readonly byteLength: number;
  readonly sha256: string;
}

/**
 * The subset of the experimental Cross-Origin Storage API that the model
 * cache uses. Files are addressed by their SHA-256 digest, so a model
 * downloaded on one origin can be reused on another without a new transfer.
 *
 * @see https://github.com/WICG/cross-origin-storage
 * @internal
 */
export interface CrossOriginStorageManager {
  getFileHandle(
    hash: CrossOriginStorageHash,
    options?: CrossOriginStorageRequestOptions,
  ): Promise<CrossOriginStorageFileHandle>;
}

interface CrossOriginStorageHash {
  readonly algorithm: "SHA-256";
  readonly value: string;
}

interface CrossOriginStorageRequestOptions {
  readonly create?: boolean;
  readonly origins?: "*" | readonly string[];
}

interface CrossOriginStorageFileHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<CrossOriginStorageWritable>;
}

interface CrossOriginStorageWritable {
  write(params: {
    readonly type: "write";
    readonly position: number;
    readonly data: Blob | BufferSource;
  }): Promise<unknown>;
  close(): Promise<unknown>;
  abort(reason?: unknown): Promise<unknown>;
}

interface ModelCacheSessionOptions {
  readonly cacheStorage?: CacheStorage;
  /** The pinned bundle identity. Without it, Cross-Origin Storage is unused. */
  readonly crossOriginStorageBundle?: CrossOriginStorageBundle;
  /**
   * Shared, content-addressed storage consulted after Cache Storage. It
   * defaults to `navigator.crossOriginStorage` when the browser exposes it;
   * `null` disables it.
   */
  readonly crossOriginStorage?: CrossOriginStorageManager | null;
  readonly fetch?: typeof fetch;
  /**
   * Refuse every network fallback and use only the exact managed cache or the
   * verified Cross-Origin Storage bundle.
   */
  readonly cacheOnly?: boolean;
}

/**
 * The structural subset of a package manifest that defines the bundle.
 *
 * @internal
 */
export interface CrossOriginStorageBundleManifest {
  readonly files: readonly {
    readonly name: string;
    readonly byteLength: number;
    readonly sha256: string;
  }[];
  readonly tensors: Readonly<Record<string, { readonly shard: string }>>;
}

/**
 * Return the files that follow the manifest in the Cross-Origin Storage
 * bundle: every file that holds tensors and `tokenizer.json`, which is the
 * complete runtime package, in manifest order. This must match
 * `cross_origin_storage_bundle_identity()` in `model/convert.py`.
 *
 * @internal
 */
export function crossOriginStorageBundleLayout(
  manifestUrl: string,
  manifest: CrossOriginStorageBundleManifest,
): ModelCacheAsset[] {
  const shards = new Set(
    Object.values(manifest.tensors).map(({ shard }) => shard),
  );
  const base = absoluteUrl(manifestUrl);
  return manifest.files
    .filter(({ name }) => shards.has(name) || name === "tokenizer.json")
    .map(({ name, byteLength, sha256 }) => ({
      url: new URL(name, base).href,
      byteLength,
      sha256,
    }));
}

interface BundlePart {
  readonly offset: number;
  readonly asset: ModelCacheAsset;
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
  private cachePromise: Promise<Cache | undefined>;
  private readonly cacheStorage: CacheStorage | undefined;
  private readonly cacheName: string;
  private readonly networkFetch: typeof fetch;
  private readonly cacheOnly: boolean;
  private cacheCorruptionDetected = false;

  // Cross-Origin Storage bundle state. `bundle` is undefined when the API, the
  // pinned identity, or a consistent layout is missing, or when the browser
  // refuses Cross-Origin Storage. While it is defined, Cache Storage only
  // stages downloads, so an interrupted load can resume, and the complete
  // package moves into the bundle.
  private bundle:
    | {
        readonly identity: CrossOriginStorageBundle;
        readonly storage: CrossOriginStorageManager;
      }
    | undefined;
  private readonly bundleParts = new Map<string, BundlePart>();
  private bundleLayoutComplete = false;
  private bundleLookup: Promise<void> | undefined;
  private bundleFile: Blob | undefined;
  private readonly ignoredBundleUrls = new Set<string>();
  private readonly verifiedUrls = new Set<string>();
  private bundleWriter: Promise<CrossOriginStorageWritable | undefined> | undefined;
  private bundleWrites: Promise<void> = Promise.resolve();
  private bundleWriteFailed = false;
  private bundleFinish: Promise<void> | undefined;

  constructor(
    manifest: ModelCacheAsset,
    options: ModelCacheSessionOptions = {},
  ) {
    assertAsset(manifest);
    this.registerAssets([manifest]);
    this.cacheOnly = options.cacheOnly === true;
    const sharedStorage =
      options.crossOriginStorage === undefined
        ? defaultCrossOriginStorage()
        : (options.crossOriginStorage ?? undefined);
    const identity = options.crossOriginStorageBundle;
    if (sharedStorage !== undefined && identity !== undefined) {
      assertAsset({ url: manifest.url, ...identity });
      this.bundle = {
        identity: {
          byteLength: identity.byteLength,
          sha256: identity.sha256.toLowerCase(),
        },
        storage: sharedStorage,
      };
      // The manifest always leads the bundle, so it can be read before the
      // rest of the layout is known.
      const url = absoluteUrl(manifest.url);
      this.bundleParts.set(url, { offset: 0, asset: this.assets.get(url)! });
    }
    this.networkFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    const storage =
      options.cacheStorage ??
      (typeof globalThis.caches === "undefined"
        ? undefined
        : globalThis.caches);
    this.cacheStorage = storage;
    this.cacheName =
      `${MODEL_CACHE_PREFIX}${manifest.sha256.toLowerCase()}`;
    this.cachePromise =
      storage === undefined
        ? Promise.resolve(undefined)
        : this.cacheOnly
          ? openExistingCache(storage, this.cacheName)
          : storage.open(this.cacheName).catch(() => undefined);
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
   * Declare the files that follow the manifest in the Cross-Origin Storage
   * bundle, as returned by `crossOriginStorageBundleLayout()`. A layout that
   * does not add up to the pinned bundle length disables the bundle.
   */
  setCrossOriginStorageBundleLayout(
    assets: readonly ModelCacheAsset[],
  ): void {
    this.registerAssets(assets);
    if (this.bundle === undefined || this.bundleLayoutComplete) return;
    let offset = [...this.bundleParts.values()][0]!.asset.byteLength;
    for (const requested of assets) {
      const url = absoluteUrl(requested.url);
      if (this.bundleParts.has(url)) {
        this.fallBackToCacheStorage();
        return;
      }
      const asset = this.assets.get(url)!;
      this.bundleParts.set(url, { offset, asset });
      offset += asset.byteLength;
    }
    if (offset !== this.bundle.identity.byteLength) {
      this.fallBackToCacheStorage();
      return;
    }
    this.bundleLayoutComplete = true;
  }

  /**
   * Require exact managed entries, or the Cross-Origin Storage bundle, for a
   * known runtime subset without reading their bodies. The later package load
   * remains responsible for streaming and verifying every byte before use.
   */
  async requireCachedAssets(
    assets: readonly ModelCacheAsset[],
  ): Promise<void> {
    this.registerAssets(assets);
    if (assets.length === 0) return;

    await this.lookUpBundle();
    const cache = await this.cachePromise;
    if (cache === undefined && this.bundle === undefined) {
      throw new ModelNotCachedError();
    }

    for (const requested of assets) {
      const url = absoluteUrl(requested.url);
      const asset = this.assets.get(url)!;
      let failure = new ModelNotCachedError(url);

      if (cache !== undefined && !this.ignoredCacheUrls.has(url)) {
        let response: Response | undefined;
        try {
          response = await cache.match(url, { ignoreVary: true });
        } catch {
          failure = new ModelNotCachedError(
            url,
            `Could not inspect cached Parakeet model asset ${url}`,
          );
        }
        if (
          response !== undefined &&
          response.body !== null &&
          isMatchingCacheEntry(response, asset)
        ) {
          continue;
        }
      }

      if (
        this.readableBundlePart(url) === undefined ||
        this.bundleFile === undefined
      ) {
        throw failure;
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
    // object to be replaced once without a hard-coded model-file count. With
    // the Cross-Origin Storage bundle, an object can be damaged in both stores.
    const stores = this.bundle === undefined ? 1 : 2;
    return this.assets.size * stores;
  }

  /**
   * Wait for pending Cache Storage writes. Once every bundle file has been
   * verified, this also moves the package into the Cross-Origin Storage
   * bundle, which continues in the background; see
   * `settleCrossOriginStorage()`.
   */
  async settle(): Promise<void> {
    while (this.pendingCacheWrites.size > 0) {
      await Promise.all([...this.pendingCacheWrites]);
    }
    if (
      this.bundleFinish === undefined &&
      this.bundle !== undefined &&
      this.bundleLayoutComplete &&
      [...this.bundleParts.keys()].every((url) => this.verifiedUrls.has(url))
    ) {
      this.bundleFinish = this.finishBundle();
    }
  }

  /** Wait for a background Cross-Origin Storage bundle write, if any. */
  async settleCrossOriginStorage(): Promise<void> {
    await this.bundleFinish;
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

    await this.lookUpBundle();
    const cache = await this.cachePromise;
    let cacheOnlyFailure = new ModelNotCachedError(url);
    if (cache !== undefined && !this.ignoredCacheUrls.has(url)) {
      let cached: Response | undefined;
      try {
        cached = await cache.match(url, { ignoreVary: true });
      } catch {
        cacheOnlyFailure = new ModelNotCachedError(
          url,
          `Could not inspect cached Parakeet model asset ${url}`,
        );
        cached = undefined;
      }
      if (cached !== undefined) {
        if (isMatchingCacheEntry(cached, asset) && cached.body !== null) {
          return this.verifiedStoredResponse(url, cached, asset, {
            label: "Cached model asset",
            evict: async () => {
              this.ignoredCacheUrls.add(url);
              try {
                await cache.delete(url, { ignoreVary: true });
              } catch {
                // The ignored URL still guarantees that the retry skips it.
              }
            },
          });
        }
        if (!this.cacheOnly) {
          try {
            await cache.delete(url, { ignoreVary: true });
          } catch {
            // A failed cleanup must not prevent a verified network load.
          }
        }
      }
    }

    const shared = await this.bundleResponse(url);
    if (shared !== undefined) return shared;

    if (this.cacheOnly) throw cacheOnlyFailure;

    // Keep the HTTP cache from holding another copy of a managed asset.
    const networkInit =
      cache === undefined && this.bundle === undefined
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

  private disableBundle(): void {
    this.bundle = undefined;
    this.bundleParts.clear();
    this.bundleLayoutComplete = false;
  }

  private readableBundlePart(url: string): BundlePart | undefined {
    return this.ignoredBundleUrls.has(url)
      ? undefined
      : this.bundleParts.get(url);
  }

  /**
   * Look up the bundle once per session, before any asset is read, because
   * browsers may budget lookups. `NotFoundError` covers both an absent bundle
   * and one the browser withholds for privacy reasons, and either way this
   * session writes the bundle. Any other failure means that Cross-Origin
   * Storage is unusable here, so the session falls back to Cache Storage.
   */
  private lookUpBundle(): Promise<void> {
    this.bundleLookup ??= (async () => {
      const bundle = this.bundle;
      if (bundle === undefined) return;
      try {
        const handle = await bundle.storage.getFileHandle(
          crossOriginStorageHash(bundle.identity),
        );
        const file = await handle.getFile();
        if (file.size === bundle.identity.byteLength) this.bundleFile = file;
      } catch (error) {
        if ((error as { name?: unknown } | null)?.name !== "NotFoundError") {
          this.fallBackToCacheStorage();
        }
      }
    })();
    return this.bundleLookup;
  }

  private fallBackToCacheStorage(): void {
    this.disableBundle();
    const storage = this.cacheStorage;
    if (storage !== undefined && !this.cacheOnly) {
      this.cachePromise = storage
        .open(this.cacheName)
        .catch(() => undefined);
    }
  }

  private bundleWritable(): Promise<CrossOriginStorageWritable | undefined> {
    const bundle = this.bundle;
    if (bundle === undefined) return Promise.resolve(undefined);
    this.bundleWriter ??= (async () => {
      try {
        // The model is public, so the bundle is shared with every origin.
        const handle = await bundle.storage.getFileHandle(
          crossOriginStorageHash(bundle.identity),
          { create: true, origins: "*" },
        );
        return await handle.createWritable();
      } catch {
        return undefined;
      }
    })();
    return this.bundleWriter;
  }

  /**
   * Queue a positioned write of one file into the bundle at its offset.
   * Cross-Origin Storage verifies the complete digest when it is closed.
   */
  private writeToBundle(position: number, data: Blob): void {
    this.bundleWrites = this.bundleWrites.then(async () => {
      if (this.bundleWriteFailed) return;
      const writable = await this.bundleWritable();
      try {
        if (writable === undefined) throw new Error("No bundle writable");
        await writable.write({
          type: "write",
          position,
          data,
        });
      } catch {
        this.bundleWriteFailed = true;
      }
    });
  }

  private async abortBundleWriter(): Promise<void> {
    this.bundleWriteFailed = true;
    await this.bundleWrites;
    const writable = await this.bundleWriter;
    try {
      await writable?.abort();
    } catch {
      // Nothing is stored until a successful close.
    }
  }

  /**
   * Once every file is verified, move the package into the bundle and delete
   * this origin's Cache Storage copy, so that the model exists only in
   * Cross-Origin Storage. Each file comes from Cache Storage, which staged
   * this load's downloads or holds a copy from before the browser supported
   * Cross-Origin Storage, or from an intact slice of a damaged bundle. Both
   * sources are disk-backed. A failure discards the write and keeps the
   * Cache Storage copy.
   */
  private async finishBundle(): Promise<void> {
    try {
      const existing = this.bundleFile;
      if (existing === undefined || this.ignoredBundleUrls.size > 0) {
        const cache = await this.cachePromise;
        for (const [url, { offset, asset }] of this.bundleParts) {
          const response = await cache?.match(url, { ignoreVary: true });
          let part: Blob | undefined;
          if (response !== undefined && isMatchingCacheEntry(response, asset)) {
            part = await response.blob();
          } else if (
            existing !== undefined &&
            this.readableBundlePart(url) !== undefined
          ) {
            // `verifiedUrls` holds every part, so this slice passed its digest.
            part = existing.slice(offset, offset + asset.byteLength);
          }
          if (part?.size !== asset.byteLength) {
            throw new Error(`No verified source for bundle part ${url}`);
          }
          this.writeToBundle(offset, part);
        }
        await this.bundleWrites;
        const writable = await this.bundleWritable();
        if (writable === undefined || this.bundleWriteFailed) {
          throw new Error("The bundle could not be written");
        }
        await writable.close();
      }
    } catch {
      await this.abortBundleWriter();
      return;
    }
    try {
      await this.cacheStorage?.delete(this.cacheName);
    } catch {
      // A later load retries the removal.
    }
  }

  /**
   * Serve one file as a slice of the bundle. Cross-Origin Storage verified the
   * bundle digest when it was written, and the slice is still verified against
   * the file's own pinned length and digest before any byte is used. It is
   * deliberately not copied into this origin's Cache Storage.
   */
  private async bundleResponse(url: string): Promise<Response | undefined> {
    const part = this.readableBundlePart(url);
    if (part === undefined) return undefined;
    const file = this.bundleFile;
    if (file === undefined) return undefined;
    const { offset, asset } = part;
    const slice = file.slice(offset, offset + asset.byteLength);
    return this.verifiedStoredResponse(url, new Response(slice), asset, {
      label: "Cross-Origin Storage model asset",
      evict: async () => {
        // A damaged bundle is skipped for this file and rewritten after the
        // load from verified sources.
        this.ignoredBundleUrls.add(url);
      },
    });
  }

  private verifiedStoredResponse(
    url: string,
    response: Response,
    asset: ModelCacheAsset,
    store: {
      readonly label: string;
      readonly evict: () => Promise<void>;
    },
  ): Response {
    const { label } = store;
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
      await store.evict();
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
            `${label} ${url} could not be read`,
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
              `${label} ${url} has ${byteLength} bytes; expected ${asset.byteLength}`,
            );
            return;
          }
          const actual = hash.digestHex();
          if (actual !== asset.sha256) {
            await failCorrupt(
              controller,
              `${label} SHA-256 mismatch for ${url}: ${actual}`,
            );
            return;
          }
          releaseSource();
          this.verifiedUrls.add(url);
          controller.close();
          return;
        }

        const chunk = item.value;
        byteLength += chunk.byteLength;
        if (byteLength > asset.byteLength) {
          await failCorrupt(
            controller,
            `${label} ${url} is longer than its pinned byte length`,
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

        this.verifiedUrls.add(url);
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

function defaultCrossOriginStorage(): CrossOriginStorageManager | undefined {
  const navigator = globalThis.navigator as
    | { readonly crossOriginStorage?: unknown }
    | undefined;
  const storage = navigator?.crossOriginStorage as
    | Partial<CrossOriginStorageManager>
    | null
    | undefined;
  return typeof storage?.getFileHandle === "function"
    ? { getFileHandle: storage.getFileHandle.bind(storage) }
    : undefined;
}

function crossOriginStorageHash(
  asset: Pick<ModelCacheAsset, "sha256">,
): CrossOriginStorageHash {
  return { algorithm: "SHA-256", value: asset.sha256 };
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
