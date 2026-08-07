/**
 * WebGPU capability negotiation for the stock-browser Parakeet runtime.
 *
 * Parakeet selects its execution profile from the stock features exposed by
 * the adapter. Precision depends only on shader-f16. Kernel selection is an
 * independent capability choice: a fixed 32-lane subgroup keeps the
 * established fast path, while every other adapter uses standards-only
 * portable kernels. No experimental browser feature is requested.
 */

export const PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES = [
  "shader-f16",
  "subgroups",
] as const;

export const PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES = ["subgroups"] as const;

export const PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES = [
  "shader-f16",
] as const;

export const PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES = [] as const;

export const PARAKEET_REQUIRED_SUBGROUP_SIZE = 32;

export const PARAKEET_OPTIONAL_WEBGPU_FEATURES = ["timestamp-query"] as const;

export const PARAKEET_STOCK_WEBGPU_FEATURES = [
  ...PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES,
  ...PARAKEET_OPTIONAL_WEBGPU_FEATURES,
] as const;

export type ParakeetRequiredWebGpuFeature =
  | (typeof PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES)[number]
  | (typeof PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES)[number]
  | (typeof PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES)[number]
  | (typeof PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES)[number];

export type ParakeetOptionalWebGpuFeature =
  (typeof PARAKEET_OPTIONAL_WEBGPU_FEATURES)[number];

export type ParakeetStockWebGpuFeature =
  (typeof PARAKEET_STOCK_WEBGPU_FEATURES)[number];

export interface ParakeetFp16ExecutionProfile {
  readonly precision: "fp16";
  readonly kernelBackend: "subgroups" | "portable";
  readonly requiredFeatures:
    | typeof PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES
    | typeof PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES;
}

export interface ParakeetFp32ExecutionProfile {
  readonly precision: "fp32";
  readonly kernelBackend: "subgroups" | "portable";
  readonly requiredFeatures:
    | typeof PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES
    | typeof PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES;
}

export type ParakeetExecutionProfile =
  | ParakeetFp16ExecutionProfile
  | ParakeetFp32ExecutionProfile;

export const PARAKEET_FP16_EXECUTION_PROFILE: ParakeetFp16ExecutionProfile =
  Object.freeze({
    precision: "fp16",
    kernelBackend: "subgroups",
    requiredFeatures: PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES,
  });

export const PARAKEET_FP32_EXECUTION_PROFILE: ParakeetFp32ExecutionProfile =
  Object.freeze({
    precision: "fp32",
    kernelBackend: "subgroups",
    requiredFeatures: PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES,
  });

export const PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE:
  ParakeetFp16ExecutionProfile = Object.freeze({
    precision: "fp16",
    kernelBackend: "portable",
    requiredFeatures: PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES,
  });

export const PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE:
  ParakeetFp32ExecutionProfile = Object.freeze({
    precision: "fp32",
    kernelBackend: "portable",
    requiredFeatures: PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES,
  });

/**
 * Limits that directly constrain Parakeet's weight shards, activation arena,
 * bind-group layout, and compute kernels.
 */
export const PARAKEET_WEBGPU_LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxDynamicStorageBuffersPerPipelineLayout",
  "maxDynamicUniformBuffersPerPipelineLayout",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
  "minStorageBufferOffsetAlignment",
  "minUniformBufferOffsetAlignment",
] as const;

export type ParakeetWebGpuLimitName =
  (typeof PARAKEET_WEBGPU_LIMIT_NAMES)[number];

export type ParakeetWebGpuLimits = Readonly<
  Record<ParakeetWebGpuLimitName, number>
>;

const MINIMUM_LIMIT_NAMES = new Set<ParakeetWebGpuLimitName>([
  "minStorageBufferOffsetAlignment",
  "minUniformBufferOffsetAlignment",
]);

export interface WebGpuAdapterInfoSnapshot {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly isFallbackAdapter?: boolean;
}

export interface WebGpuStockFeatureStatus {
  readonly adapterSupported: boolean;
  readonly deviceEnabled: boolean;
  readonly required: boolean;
  readonly requested: boolean;
}

export type WebGpuStockFeatureReport = Readonly<
  Record<ParakeetStockWebGpuFeature, WebGpuStockFeatureStatus>
>;

export interface WebGpuCapabilityReport {
  readonly executionProfile: ParakeetExecutionProfile;
  readonly adapterInfo: WebGpuAdapterInfoSnapshot;
  readonly adapterFeatures: readonly string[];
  readonly deviceFeatures: readonly string[];
  readonly stockFeatures: WebGpuStockFeatureReport;
  readonly requiredFeatures: readonly ParakeetRequiredWebGpuFeature[];
  readonly requestedOptionalFeatures: readonly ParakeetOptionalWebGpuFeature[];
  readonly adapterLimits: ParakeetWebGpuLimits;
  readonly deviceLimits: ParakeetWebGpuLimits;
  readonly requestedLimits: Readonly<Partial<Record<ParakeetWebGpuLimitName, number>>>;
}

export interface WebGpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly executionProfile: ParakeetExecutionProfile;
  readonly report: WebGpuCapabilityReport;
}

export interface RequestWebGpuContextOptions {
  /**
   * Optional features to negotiate. Unsupported entries are omitted rather
   * than making device creation fail.
   */
  readonly optionalFeatures?: readonly ParakeetOptionalWebGpuFeature[];
  /**
   * Limits the runtime actually needs. Requesting only known requirements is
   * preferable to requesting every adapter maximum.
   */
  readonly requiredLimits?: Readonly<
    Partial<Record<ParakeetWebGpuLimitName, number>>
  >;
  /** Primarily useful for deterministic tests or an embedded browser host. */
  readonly gpu?: GPU;
}

export interface WebGpuLimitDeficit {
  readonly name: ParakeetWebGpuLimitName;
  readonly requested: number;
  readonly available: number;
  readonly relation: "at-least" | "at-most";
}

export type WebGpuUnavailableCode =
  | "INSECURE_CONTEXT"
  | "WEBGPU_UNAVAILABLE"
  | "ADAPTER_UNAVAILABLE"
  | "FEATURE_UNAVAILABLE"
  | "LIMIT_UNAVAILABLE";

export class WebGpuUnavailableError extends Error {
  readonly code: WebGpuUnavailableCode;
  readonly deficits: readonly WebGpuLimitDeficit[];

  constructor(
    code: WebGpuUnavailableCode,
    message: string,
    deficits: readonly WebGpuLimitDeficit[] = [],
  ) {
    super(message);
    this.name = "WebGpuUnavailableError";
    this.code = code;
    this.deficits = deficits;
  }
}

export function negotiateOptionalWebGpuFeatures(
  availableFeatures: Iterable<string>,
  requestedFeatures: readonly ParakeetOptionalWebGpuFeature[] =
    PARAKEET_OPTIONAL_WEBGPU_FEATURES,
): ParakeetOptionalWebGpuFeature[] {
  const available = new Set(availableFeatures);
  const negotiated: ParakeetOptionalWebGpuFeature[] = [];

  for (const feature of requestedFeatures) {
    if (!available.has(feature) || negotiated.includes(feature)) continue;
    negotiated.push(feature);
  }

  return negotiated;
}

/**
 * Prefer FP16 whenever shader-f16 is available. Independently retain the
 * established subgroup implementation only for its fixed 32-lane contract.
 * A subgroup feature without both reported bounds is not enough to prove the
 * fixed-32 contract, so omitted or partial size metadata selects portable
 * kernels.
 */
export function selectParakeetExecutionProfile(
  availableFeatures: Iterable<string>,
  subgroupMinSize?: number,
  subgroupMaxSize?: number,
): ParakeetExecutionProfile {
  const features = new Set(availableFeatures);
  const fixedSubgroups =
    features.has("subgroups") &&
    subgroupMinSize === PARAKEET_REQUIRED_SUBGROUP_SIZE &&
    subgroupMaxSize === PARAKEET_REQUIRED_SUBGROUP_SIZE;
  if (features.has("shader-f16")) {
    return fixedSubgroups
      ? PARAKEET_FP16_EXECUTION_PROFILE
      : PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE;
  }
  return fixedSubgroups
    ? PARAKEET_FP32_EXECUTION_PROFILE
    : PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE;
}

export function snapshotParakeetWebGpuLimits(
  limits: GPUSupportedLimits | Readonly<Record<string, number>>,
): ParakeetWebGpuLimits {
  const values = limits as unknown as Readonly<Record<string, number>>;
  const snapshot = {} as Record<ParakeetWebGpuLimitName, number>;

  for (const name of PARAKEET_WEBGPU_LIMIT_NAMES) {
    const value = values[name];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(`WebGPU limit ${name} is unavailable or non-numeric`);
    }
    snapshot[name] = value;
  }

  return Object.freeze(snapshot);
}

/**
 * Compare requested device limits with adapter capabilities. WebGPU's `min*`
 * limits are lower-is-better, unlike the other limits.
 */
export function findWebGpuLimitDeficits(
  availableLimits: GPUSupportedLimits | Readonly<Record<string, number>>,
  requestedLimits: Readonly<Partial<Record<ParakeetWebGpuLimitName, number>>>,
): WebGpuLimitDeficit[] {
  const available = availableLimits as unknown as Readonly<Record<string, number>>;
  const deficits: WebGpuLimitDeficit[] = [];

  for (const [rawName, requested] of Object.entries(requestedLimits)) {
    if (requested === undefined) continue;
    const name = rawName as ParakeetWebGpuLimitName;
    if (!(PARAKEET_WEBGPU_LIMIT_NAMES as readonly string[]).includes(name)) {
      throw new TypeError(`Unknown Parakeet WebGPU limit ${rawName}`);
    }
    if (!Number.isFinite(requested) || requested < 0 || !Number.isInteger(requested)) {
      throw new RangeError(`Requested WebGPU limit ${name} must be a non-negative integer`);
    }

    const supported = available[name];
    if (typeof supported !== "number" || !Number.isFinite(supported)) {
      deficits.push({
        name,
        requested,
        available: Number.NaN,
        relation: MINIMUM_LIMIT_NAMES.has(name) ? "at-most" : "at-least",
      });
      continue;
    }

    if (MINIMUM_LIMIT_NAMES.has(name)) {
      if (supported > requested) {
        deficits.push({
          name,
          requested,
          available: supported,
          relation: "at-most",
        });
      }
    } else if (supported < requested) {
      deficits.push({
        name,
        requested,
        available: supported,
        relation: "at-least",
      });
    }
  }

  return deficits;
}

export async function requestWebGpuContext(
  options: RequestWebGpuContextOptions = {},
): Promise<WebGpuContext> {
  if (globalThis.isSecureContext === false) {
    throw new WebGpuUnavailableError(
      "INSECURE_CONTEXT",
      "WebGPU requires HTTPS or a localhost secure context",
    );
  }

  const gpu =
    options.gpu ??
    (globalThis.navigator as (Navigator & { readonly gpu?: GPU }) | undefined)?.gpu;
  if (!gpu) {
    throw new WebGpuUnavailableError(
      "WEBGPU_UNAVAILABLE",
      "This runtime does not expose navigator.gpu",
    );
  }

  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (!adapter) {
    throw new WebGpuUnavailableError(
      "ADAPTER_UNAVAILABLE",
      "No WebGPU adapter is available",
    );
  }

  const requiredLimits = options.requiredLimits ?? {};
  const deficits = findWebGpuLimitDeficits(adapter.limits, requiredLimits);
  if (deficits.length > 0) {
    const summary = deficits
      .map(
        ({ name, requested, available, relation }) =>
          `${name} must be ${relation} ${requested}; adapter reports ${available}`,
      )
      .join("; ");
    throw new WebGpuUnavailableError(
      "LIMIT_UNAVAILABLE",
      `The WebGPU adapter cannot satisfy model limits: ${summary}`,
      deficits,
    );
  }

  const adapterFeatures = sortedFeatureNames(adapter.features);
  const adapterInfo = snapshotAdapterInfo(adapter);
  const executionProfile = selectParakeetExecutionProfile(
    adapterFeatures,
    adapterInfo.subgroupMinSize,
    adapterInfo.subgroupMaxSize,
  );
  const missingRequiredFeatures = executionProfile.requiredFeatures.filter(
    (feature) => !adapterFeatures.includes(feature),
  );
  if (missingRequiredFeatures.length > 0) {
    throw new WebGpuUnavailableError(
      "FEATURE_UNAVAILABLE",
      "This browser does not expose the required WebGPU feature" +
        `${missingRequiredFeatures.length === 1 ? "" : "s"}: ` +
        missingRequiredFeatures.join(", "),
    );
  }
  const negotiatedFeatures = negotiateOptionalWebGpuFeatures(
    adapterFeatures,
    options.optionalFeatures ?? PARAKEET_OPTIONAL_WEBGPU_FEATURES,
  );
  const requestedFeatures = [
    ...executionProfile.requiredFeatures,
    ...negotiatedFeatures,
  ];

  const device = await adapter.requestDevice({
    requiredFeatures: requestedFeatures as GPUFeatureName[],
    requiredLimits: { ...requiredLimits } as Record<string, GPUSize64>,
    defaultQueue: { label: "parakeet-wgsl-default-queue" },
    label: "parakeet-wgsl-device",
  });

  const deviceFeatures = sortedFeatureNames(device.features);
  const stockFeatures = {} as Record<
    ParakeetStockWebGpuFeature,
    WebGpuStockFeatureStatus
  >;
  for (const feature of PARAKEET_STOCK_WEBGPU_FEATURES) {
    stockFeatures[feature] = Object.freeze({
      adapterSupported: adapterFeatures.includes(feature),
      deviceEnabled: deviceFeatures.includes(feature),
      required: (executionProfile.requiredFeatures as readonly string[]).includes(
        feature,
      ),
      requested: requestedFeatures.includes(feature),
    });
  }

  return {
    adapter,
    device,
    executionProfile,
    report: Object.freeze({
      executionProfile,
      adapterInfo,
      adapterFeatures,
      deviceFeatures,
      stockFeatures: Object.freeze(stockFeatures),
      requiredFeatures: Object.freeze([...executionProfile.requiredFeatures]),
      requestedOptionalFeatures: Object.freeze([...negotiatedFeatures]),
      adapterLimits: snapshotParakeetWebGpuLimits(adapter.limits),
      deviceLimits: snapshotParakeetWebGpuLimits(device.limits),
      requestedLimits: Object.freeze({ ...requiredLimits }),
    }),
  };
}

function sortedFeatureNames(features: GPUSupportedFeatures | Iterable<string>): string[] {
  return Array.from(features, String).sort();
}

function snapshotAdapterInfo(adapter: GPUAdapter): WebGpuAdapterInfoSnapshot {
  const info = adapter.info as unknown as Readonly<Record<string, unknown>>;
  const subgroupMinSize = finiteNumber(info.subgroupMinSize);
  const subgroupMaxSize = finiteNumber(info.subgroupMaxSize);
  const isFallbackAdapter =
    typeof info.isFallbackAdapter === "boolean"
      ? info.isFallbackAdapter
      : undefined;

  return Object.freeze({
    vendor: stringValue(info.vendor),
    architecture: stringValue(info.architecture),
    device: stringValue(info.device),
    description: stringValue(info.description),
    ...(subgroupMinSize === undefined ? {} : { subgroupMinSize }),
    ...(subgroupMaxSize === undefined ? {} : { subgroupMaxSize }),
    ...(isFallbackAdapter === undefined ? {} : { isFallbackAdapter }),
  });
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
