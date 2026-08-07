/// <reference types="@webgpu/types" />

import {
  clearStoredModels,
  inspectStoredModels,
} from "./model-cache.js";
import {
  requiresAudioDecoding as inspectAudioDecoding,
} from "./audio/wav-stream.js";
import {
  ParakeetTranscriber,
  type ModelLoadProgress,
  type ParakeetWorkerLike,
} from "./runtime/transcriber.js";
import { resolveAudioDecodingRuntimeAssets } from "./runtime-assets.js";
import {
  findWebGpuLimitDeficits,
  selectParakeetExecutionProfile,
} from "./webgpu/capabilities.js";

const REQUIRED_STORAGE_BYTES = 128 * 1024 * 1024;
const REQUIRED_WORKGROUP_STORAGE_BYTES = 32 * 1024;

export interface ParakeetModelUrls {
  /** Canonical FP16 manifest, selected only when shader-f16 is available. */
  readonly fp16: string | URL;
  /** Canonical FP32 manifest, selected only when shader-f16 is unavailable. */
  readonly fp32: string | URL;
}

export type ParakeetModelPrecision = "fp16" | "fp32";
export type ParakeetKernelBackend = "subgroups" | "portable";

/** Capability-derived execution profile fixed for a transcriber's lifetime. */
export interface ParakeetExecutionProfile {
  readonly precision: ParakeetModelPrecision;
  readonly kernelBackend: ParakeetKernelBackend;
  readonly requiredFeatures: readonly ("shader-f16" | "subgroups")[];
}

export const DEFAULT_MODEL_URLS: Readonly<ParakeetModelUrls> = Object.freeze({
  fp16:
    "https://parakeet-wgsl-models.narcotic.sh/v1/fp16/" +
    "11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65/manifest.json",
  fp32:
    "https://parakeet-wgsl-models.narcotic.sh/v1/fp32/" +
    "28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b/manifest.json",
});

export interface CreateTranscriberOptions {
  /** Override one or both canonical package locations for self-hosting. */
  readonly modelUrls?: Partial<ParakeetModelUrls>;
  /** Model-load events from the first initialization that needs the model. */
  readonly onLoadProgress?: (progress: ModelLoadProgress) => void;
  /** Override the streaming filter-bank Wasm URL for unusual bundlers/CSP. */
  readonly wasmUrl?: string | URL;
  /** Override construction of the WebGPU inference worker. */
  readonly workerFactory?: () => Worker;
}

export interface ParakeetAdapterSummary {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly isFallbackAdapter?: boolean;
}

export interface ParakeetSupport {
  readonly supported: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly executionProfile: ParakeetExecutionProfile | null;
  readonly adapter: ParakeetAdapterSummary | null;
  /** Optional compressed/container input support; canonical WAV is independent. */
  readonly audioDecoding: {
    readonly available: boolean;
    readonly reason?: string;
  };
}

export interface ModelCacheInfo {
  /** Whether this browser context exposes Cache Storage. */
  readonly supported: boolean;
  /** Number of parakeet.wgsl-managed model objects stored for this origin. */
  readonly assetCount: number;
  /** Sum of the stored objects' pinned byte lengths. */
  readonly sizeBytes: number;
}

/**
 * Construct a reusable transcriber without requesting WebGPU or downloading
 * model bytes. Optional loadCachedModel() warmup is strictly model-network-free.
 * Otherwise the first transcribe() call prepares the audio, performs
 * capability selection, opens the selected persistent cache, downloads only
 * that precision when needed, compiles the fixed graph, and transcribes it.
 */
export function createTranscriber(
  options: CreateTranscriberOptions = {},
): ParakeetTranscriber {
  const fp16ModelUrl = normalizeUrl(
    options.modelUrls?.fp16 ?? DEFAULT_MODEL_URLS.fp16,
    "FP16 model manifest",
  );
  const fp32ModelUrl = normalizeUrl(
    options.modelUrls?.fp32 ?? DEFAULT_MODEL_URLS.fp32,
    "FP32 model manifest",
  );
  const wasmUrl = normalizeUrl(
    options.wasmUrl ??
      new URL("./cpp/wasm/parakeet-fbank.wasm", import.meta.url),
    "Parakeet FBank Wasm",
  );
  const loadOptions = {
    fp16ModelUrl,
    fp32ModelUrl,
    wasmUrl,
    ...resolveAudioDecodingRuntimeAssets(),
    ...(options.onLoadProgress === undefined
      ? {}
      : { onLoadProgress: options.onLoadProgress }),
  };
  if (options.workerFactory === undefined) {
    return ParakeetTranscriber.createLazy(loadOptions);
  }
  return ParakeetTranscriber.createLazy(
    loadOptions,
    options.workerFactory() as unknown as ParakeetWorkerLike,
  );
}

/**
 * Perform a network-free adapter and OPFS-root support check without creating
 * a GPU device. The worker repeats the authoritative execution-profile
 * selection when its first transcription initializes.
 */
export async function checkSupport(): Promise<ParakeetSupport> {
  const audioDecoding = await audioDecodingSupport();
  if (globalThis.isSecureContext === false) {
    return unsupported(
      "WebGPU requires HTTPS or a localhost secure context",
      audioDecoding,
    );
  }
  const gpu = (globalThis.navigator as
    | (Navigator & { readonly gpu?: GPU })
    | undefined)?.gpu;
  if (gpu === undefined) {
    return unsupported(
      "This browser does not expose WebGPU",
      audioDecoding,
    );
  }

  try {
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });
    if (adapter === null) {
      return unsupported(
        "No WebGPU adapter is available",
        audioDecoding,
      );
    }
    const info = adapter.info as GPUAdapterInfo & {
      readonly subgroupMinSize?: number;
      readonly subgroupMaxSize?: number;
      readonly isFallbackAdapter?: boolean;
    };
    const executionProfile = selectParakeetExecutionProfile(
      adapter.features,
      info.subgroupMinSize,
      info.subgroupMaxSize,
    );
    const deficits = findWebGpuLimitDeficits(adapter.limits, {
      maxBufferSize: REQUIRED_STORAGE_BYTES,
      maxStorageBufferBindingSize: REQUIRED_STORAGE_BYTES,
      maxComputeWorkgroupStorageSize: REQUIRED_WORKGROUP_STORAGE_BYTES,
    });
    const errors = deficits.map(
      ({ name, requested, available, relation }) =>
        `${name} must be ${relation} ${requested}; the adapter reports ${available}`,
    );
    const adapterSummary: ParakeetAdapterSummary = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      ...(info.isFallbackAdapter === undefined
        ? {}
        : { isFallbackAdapter: info.isFallbackAdapter }),
    };
    return {
      supported: errors.length === 0,
      errors,
      warnings:
        info.isFallbackAdapter === true
          ? ["The browser selected a fallback WebGPU adapter."]
          : [],
      executionProfile,
      adapter: adapterSummary,
      audioDecoding,
    };
  } catch (error) {
    return unsupported(
      error instanceof Error ? error.message : String(error),
      audioDecoding,
    );
  }
}

export async function getModelCacheInfo(): Promise<ModelCacheInfo> {
  return await inspectStoredModels();
}

/**
 * Remove every parakeet.wgsl-managed model object from this origin. A live
 * transcriber keeps its already uploaded GPU model until dispose() is called.
 */
export async function deleteCachedModels(): Promise<void> {
  await clearStoredModels();
}

/**
 * Return whether an audio Blob will use the packaged FFmpeg decoding
 * path. This performs only bounded header reads and no network or OPFS I/O.
 */
export async function requiresAudioDecoding(
  audio: Blob,
): Promise<boolean> {
  if (!(audio instanceof Blob)) {
    throw new TypeError(
      "requiresAudioDecoding() expects an audio File or Blob",
    );
  }
  return await inspectAudioDecoding(audio);
}

function normalizeUrl(value: string | URL, label: string): string {
  const url = value.toString();
  if (url.length === 0) throw new TypeError(`${label} URL is empty`);
  return url;
}

function unsupported(
  message: string,
  audioDecoding: ParakeetSupport["audioDecoding"],
): ParakeetSupport {
  return {
    supported: false,
    errors: [message],
    warnings: [],
    executionProfile: null,
    adapter: null,
    audioDecoding,
  };
}

async function audioDecodingSupport(): Promise<
  ParakeetSupport["audioDecoding"]
> {
  const storage = globalThis.navigator?.storage;
  if (storage === undefined || typeof storage.getDirectory !== "function") {
    return {
      available: false,
      reason:
        "Compressed and non-canonical audio requires the Origin Private File System.",
    };
  }
  try {
    await storage.getDirectory();
    return { available: true };
  } catch (error) {
    const detail = error instanceof Error ? ` ${error.message}` : "";
    return {
      available: false,
      reason: `Origin Private File System access is unavailable.${detail}`,
    };
  }
}
