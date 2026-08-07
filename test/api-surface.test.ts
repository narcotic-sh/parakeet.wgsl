import { afterEach, describe, expect, it, vi } from "vitest";

import * as publicApi from "../src/index";
import {
  checkSupport,
  createTranscriber,
  DEFAULT_MODEL_URLS,
  deleteCachedModels,
} from "../src/api";
import {
  PARAKEET_FP16_MANIFEST_SHA256,
  PARAKEET_FP32_MANIFEST_SHA256,
} from "../src/model/manifest";
import type {
  ParakeetClientMessage,
  ParakeetWorkerMessage,
} from "../src/runtime/protocol";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parakeet.wgsl public API", () => {
  it("exports only the intended runtime values", () => {
    expect(Object.keys(publicApi).sort()).toEqual([
      "DEFAULT_MODEL_URLS",
      "PacedTranscriptDomRenderer",
      "ParakeetError",
      "checkSupport",
      "createTranscriber",
      "deleteCachedModels",
      "getModelCacheInfo",
      "pacedTranscriptWordAtOffset",
      "requiresAudioDecoding",
    ]);
  });

  it("pins distinct immutable FP16 and FP32 manifests on the model host", () => {
    const fp16 = new URL(DEFAULT_MODEL_URLS.fp16);
    const fp32 = new URL(DEFAULT_MODEL_URLS.fp32);

    expect(Object.isFrozen(DEFAULT_MODEL_URLS)).toBe(true);
    expect(fp16.protocol).toBe("https:");
    expect(fp32.protocol).toBe("https:");
    expect(fp16.hostname).toBe("parakeet-wgsl-models.narcotic.sh");
    expect(fp32.hostname).toBe(fp16.hostname);
    expect(fp16.pathname).toBe(
      `/v1/fp16/${PARAKEET_FP16_MANIFEST_SHA256}/manifest.json`,
    );
    expect(fp32.pathname).toBe(
      `/v1/fp32/${PARAKEET_FP32_MANIFEST_SHA256}/manifest.json`,
    );
    expect(fp16.href).not.toBe(fp32.href);
  });

  it("constructs a transcriber without probing WebGPU or starting model I/O", () => {
    const worker = new SilentWorker();
    const networkFetch = vi.fn(() => {
      throw new Error("construction must not fetch");
    });
    const navigatorValue: Record<string, unknown> = {};
    Object.defineProperty(navigatorValue, "gpu", {
      configurable: true,
      get() {
        throw new Error("construction must not inspect WebGPU");
      },
    });
    vi.stubGlobal("fetch", networkFetch);
    vi.stubGlobal("navigator", navigatorValue);

    const transcriber = createTranscriber({
      workerFactory: () => worker as unknown as Worker,
    });

    expect(worker.messages).toEqual([]);
    expect(networkFetch).not.toHaveBeenCalled();
    expect(transcriber.initialized).toBe(false);
    expect(transcriber.diagnostics).toBeNull();

    transcriber.dispose();
    expect(worker.terminated).toBe(true);
  });

  it("rejects non-Blob audio normalization preflight values", async () => {
    await expect(
      publicApi.requiresAudioDecoding("recording.mp3" as never),
    ).rejects.toThrow(/File or Blob/);
  });

  it("reports an inaccessible OPFS root as unavailable", async () => {
    vi.stubGlobal("isSecureContext", true);
    vi.stubGlobal("navigator", {
      storage: {
        getDirectory: async () => {
          throw new DOMException("storage denied", "SecurityError");
        },
      },
    });

    const support = await checkSupport();
    expect(support.supported).toBe(false);
    expect(support.audioDecoding).toEqual({
      available: false,
      reason: "Origin Private File System access is unavailable. storage denied",
    });
  });

  it("checks adapter support without fetching or creating a device", async () => {
    const networkFetch = vi.fn(() => {
      throw new Error("support checks must not fetch");
    });
    const requestDevice = vi.fn(() => {
      throw new Error("the adapter-only check must not create a device");
    });
    const requestAdapter = vi.fn(async () =>
      ({
        features: new Set(["shader-f16"]),
        limits: {
          maxBufferSize: 256 * 1024 * 1024,
          maxStorageBufferBindingSize: 256 * 1024 * 1024,
          maxComputeWorkgroupStorageSize: 64 * 1024,
        },
        info: {
          vendor: "test-vendor",
          architecture: "test-architecture",
          device: "test-device",
          description: "test-adapter",
          isFallbackAdapter: false,
        },
        requestDevice,
      }) as unknown as GPUAdapter,
    );
    vi.stubGlobal("fetch", networkFetch);
    vi.stubGlobal("isSecureContext", true);
    vi.stubGlobal("navigator", {
      gpu: { requestAdapter } as unknown as GPU,
      storage: {
        getDirectory: vi.fn(),
      },
    });

    await expect(checkSupport()).resolves.toEqual({
      supported: true,
      errors: [],
      warnings: [],
      executionProfile: {
        precision: "fp16",
        kernelBackend: "portable",
        requiredFeatures: ["shader-f16"],
      },
      adapter: {
        vendor: "test-vendor",
        architecture: "test-architecture",
        device: "test-device",
        description: "test-adapter",
        isFallbackAdapter: false,
      },
      audioDecoding: {
        available: true,
      },
    });
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(requestDevice).not.toHaveBeenCalled();
    expect(networkFetch).not.toHaveBeenCalled();
  });

  it("deletes only caches in the package-owned family", async () => {
    const deleted: string[] = [];
    vi.stubGlobal("caches", {
      keys: async () => [
        "parakeet.wgsl:models:v1:fp16",
        "site-assets",
        "parakeet.wgsl:models:v2:fp32",
        "other-package:models:v1",
      ],
      delete: async (name: string) => {
        deleted.push(name);
        return true;
      },
    } satisfies Pick<CacheStorage, "keys" | "delete">);

    await deleteCachedModels();

    expect(deleted.sort()).toEqual([
      "parakeet.wgsl:models:v1:fp16",
      "parakeet.wgsl:models:v2:fp32",
    ]);
  });
});

class SilentWorker {
  readonly messages: ParakeetClientMessage[] = [];
  terminated = false;

  postMessage(message: ParakeetClientMessage): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  addEventListener(
    _type: "message" | "error",
    _listener:
      | ((event: MessageEvent<ParakeetWorkerMessage>) => void)
      | ((event: ErrorEvent) => void),
  ): void {}

  removeEventListener(
    _type: "message" | "error",
    _listener:
      | ((event: MessageEvent<ParakeetWorkerMessage>) => void)
      | ((event: ErrorEvent) => void),
  ): void {}
}
