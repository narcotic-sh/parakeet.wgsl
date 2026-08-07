import { expect, it } from "vitest";

import {
  findWebGpuLimitDeficits,
  negotiateOptionalWebGpuFeatures,
  PARAKEET_FP16_EXECUTION_PROFILE,
  PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES,
  PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES,
  PARAKEET_FP32_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
  PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES,
  PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES,
  PARAKEET_REQUIRED_SUBGROUP_SIZE,
  PARAKEET_WEBGPU_LIMIT_NAMES,
  requestWebGpuContext,
  selectParakeetExecutionProfile,
  snapshotParakeetWebGpuLimits,
} from "../../src/webgpu/capabilities";

it("negotiates only exposed optional WebGPU features", () => {
  expect(
    negotiateOptionalWebGpuFeatures(
      new Set(["shader-f16", "timestamp-query", "texture-compression-bc"]),
    ),
  ).toEqual(["timestamp-query"]);
});

it("keeps the established subgroup profiles and defines portable requirements", () => {
  expect(PARAKEET_FP16_REQUIRED_WEBGPU_FEATURES).toEqual([
    "shader-f16",
    "subgroups",
  ]);
  expect(PARAKEET_FP32_REQUIRED_WEBGPU_FEATURES).toEqual(["subgroups"]);
  expect(PARAKEET_FP16_PORTABLE_REQUIRED_WEBGPU_FEATURES).toEqual([
    "shader-f16",
  ]);
  expect(PARAKEET_FP32_PORTABLE_REQUIRED_WEBGPU_FEATURES).toEqual([]);

  expect(PARAKEET_FP16_EXECUTION_PROFILE).toEqual({
    precision: "fp16",
    kernelBackend: "subgroups",
    requiredFeatures: ["shader-f16", "subgroups"],
  });
  expect(PARAKEET_FP32_EXECUTION_PROFILE).toEqual({
    precision: "fp32",
    kernelBackend: "subgroups",
    requiredFeatures: ["subgroups"],
  });
});

it("selects precision independently from the subgroup backend", () => {
  expect(
    selectParakeetExecutionProfile(
      new Set(["subgroups", "shader-f16"]),
      32,
      32,
    ),
  ).toBe(PARAKEET_FP16_EXECUTION_PROFILE);
  expect(
    selectParakeetExecutionProfile(new Set(["subgroups"]), 32, 32),
  ).toBe(PARAKEET_FP32_EXECUTION_PROFILE);

  expect(
    selectParakeetExecutionProfile(new Set(["shader-f16"])),
  ).toBe(PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE);
  expect(
    selectParakeetExecutionProfile(
      new Set(["shader-f16", "subgroups"]),
      16,
      32,
    ),
  ).toBe(PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE);
  expect(
    selectParakeetExecutionProfile(
      new Set(["shader-f16", "subgroups"]),
    ),
  ).toBe(PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE);
  expect(
    selectParakeetExecutionProfile(
      new Set(["shader-f16", "subgroups"]),
      32,
    ),
  ).toBe(PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE);
  expect(selectParakeetExecutionProfile(new Set())).toBe(
    PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
  );
  expect(
    selectParakeetExecutionProfile(new Set(["subgroups"]), 16, 32),
  ).toBe(
    PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
  );
  expect(
    selectParakeetExecutionProfile(new Set(["subgroups"])),
  ).toBe(PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE);
  expect(
    selectParakeetExecutionProfile(
      new Set(["subgroups"]),
      undefined,
      32,
    ),
  ).toBe(PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE);
});

it("snapshots every model-relevant compute and buffer limit", () => {
  const source = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name, index) => [name, index + 1]),
  );
  const snapshot = snapshotParakeetWebGpuLimits(source);

  expect(Object.keys(snapshot)).toEqual([...PARAKEET_WEBGPU_LIMIT_NAMES]);
  for (const [index, name] of PARAKEET_WEBGPU_LIMIT_NAMES.entries()) {
    expect(snapshot[name]).toBe(index + 1);
  }
  expect(Object.isFrozen(snapshot)).toBe(true);
});

it("reports higher-is-better and lower-is-better limit deficits", () => {
  const available = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 256]),
  );
  const deficits = findWebGpuLimitDeficits(available, {
    maxBufferSize: 512,
    maxStorageBufferBindingSize: 128,
    minStorageBufferOffsetAlignment: 128,
    minUniformBufferOffsetAlignment: 512,
  });

  expect(deficits).toEqual([
    {
      name: "maxBufferSize",
      requested: 512,
      available: 256,
      relation: "at-least",
    },
    {
      name: "minStorageBufferOffsetAlignment",
      requested: 128,
      available: 256,
      relation: "at-most",
    },
  ]);
});

it("requests only adapter-supported optional features and reports device limits", async () => {
  const adapterLimits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 1_024]),
  );
  const deviceLimits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 512]),
  );
  let adapterOptions: GPURequestAdapterOptions | undefined;
  let descriptor: GPUDeviceDescriptor | undefined;
  const device = {
    features: new Set(["shader-f16", "subgroups", "timestamp-query"]),
    limits: deviceLimits,
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set([
      "shader-f16",
      "subgroups",
      "timestamp-query",
      "texture-compression-bc",
    ]),
    limits: adapterLimits,
    info: {
      vendor: "test-vendor",
      architecture: "test-architecture",
      device: "test-device",
      description: "test adapter",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    },
    async requestDevice(requested: GPUDeviceDescriptor) {
      descriptor = requested;
      return device;
    },
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter(options: GPURequestAdapterOptions) {
      adapterOptions = options;
      return adapter;
    },
  } as unknown as GPU;

  const context = await requestWebGpuContext({
    gpu,
    requiredLimits: { maxStorageBufferBindingSize: 256 },
  });

  if (!descriptor) throw new Error("requestDevice was not invoked");
  expect(adapterOptions).toEqual({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  expect(descriptor.requiredFeatures).toEqual([
    "shader-f16",
    "subgroups",
    "timestamp-query",
  ]);
  expect(descriptor.requiredLimits).toEqual({
    maxStorageBufferBindingSize: 256,
  });
  expect(context.report.adapterInfo.vendor).toBe("test-vendor");
  expect(context.report.deviceLimits.maxBufferSize).toBe(512);
  expect(context.executionProfile).toBe(PARAKEET_FP16_EXECUTION_PROFILE);
  expect(context.report.executionProfile).toBe(context.executionProfile);
  expect(context.report.requiredFeatures).toEqual([
    "shader-f16",
    "subgroups",
  ]);
  expect(context.report.stockFeatures["shader-f16"]).toEqual({
    adapterSupported: true,
    deviceEnabled: true,
    required: true,
    requested: true,
  });
  expect(context.report.stockFeatures["timestamp-query"].required).toBe(false);
  expect(PARAKEET_REQUIRED_SUBGROUP_SIZE).toBe(32);
});

it("selects fp32 and does not request shader-f16 when it is unavailable", async () => {
  const limits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 1_024]),
  );
  let descriptor: GPUDeviceDescriptor | undefined;
  const device = {
    features: new Set(["subgroups", "timestamp-query"]),
    limits,
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set(["subgroups", "timestamp-query"]),
    limits,
    info: {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    },
    async requestDevice(requested: GPUDeviceDescriptor) {
      descriptor = requested;
      return device;
    },
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter() {
      return adapter;
    },
  } as unknown as GPU;

  const context = await requestWebGpuContext({ gpu });

  if (!descriptor) throw new Error("requestDevice was not invoked");
  expect(descriptor.requiredFeatures).toEqual([
    "subgroups",
    "timestamp-query",
  ]);
  expect(context.executionProfile).toBe(PARAKEET_FP32_EXECUTION_PROFILE);
  expect(context.report.executionProfile).toBe(context.executionProfile);
  expect(context.report.requiredFeatures).toEqual(["subgroups"]);
  expect(context.report.stockFeatures["shader-f16"]).toEqual({
    adapterSupported: false,
    deviceEnabled: false,
    required: false,
    requested: false,
  });
  expect(context.report.stockFeatures["subgroups"]).toEqual({
    adapterSupported: true,
    deviceEnabled: true,
    required: true,
    requested: true,
  });
});

it("keeps fp16 and requests supported timestamp queries without subgroups", async () => {
  const adapterLimits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 1_024]),
  );
  let descriptor: GPUDeviceDescriptor | undefined;
  const device = {
    features: new Set(["shader-f16", "timestamp-query"]),
    limits: adapterLimits,
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set(["shader-f16", "timestamp-query"]),
    limits: adapterLimits,
    info: {},
    async requestDevice(requested: GPUDeviceDescriptor) {
      descriptor = requested;
      return device;
    },
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter() {
      return adapter;
    },
  } as unknown as GPU;

  const context = await requestWebGpuContext({ gpu });

  if (!descriptor) throw new Error("requestDevice was not invoked");
  expect(descriptor.requiredFeatures).toEqual([
    "shader-f16",
    "timestamp-query",
  ]);
  expect(context.executionProfile).toBe(
    PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  );
  expect(context.report.requiredFeatures).toEqual(["shader-f16"]);
  expect(context.report.requestedOptionalFeatures).toEqual([
    "timestamp-query",
  ]);
  expect(context.report.stockFeatures["subgroups"]).toEqual({
    adapterSupported: false,
    deviceEnabled: false,
    required: false,
    requested: false,
  });
});

it("uses fp16 portable kernels for a non-32 subgroup and does not request it", async () => {
  const adapterLimits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 1_024]),
  );
  let descriptor: GPUDeviceDescriptor | undefined;
  const device = {
    features: new Set(["shader-f16"]),
    limits: adapterLimits,
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set(["shader-f16", "subgroups", "timestamp-query"]),
    limits: adapterLimits,
    info: {
      subgroupMinSize: 16,
      subgroupMaxSize: 32,
    },
    async requestDevice(requested: GPUDeviceDescriptor) {
      descriptor = requested;
      return device;
    },
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter() {
      return adapter;
    },
  } as unknown as GPU;

  const context = await requestWebGpuContext({
    gpu,
    optionalFeatures: [],
  });

  if (!descriptor) throw new Error("requestDevice was not invoked");
  expect(descriptor.requiredFeatures).toEqual(["shader-f16"]);
  expect(context.executionProfile).toBe(
    PARAKEET_FP16_PORTABLE_EXECUTION_PROFILE,
  );
  expect(context.report.requestedOptionalFeatures).toEqual([]);
  expect(context.report.stockFeatures["subgroups"]).toEqual({
    adapterSupported: true,
    deviceEnabled: false,
    required: false,
    requested: false,
  });
});

it("selects fp32 portable kernels when shader-f16 and subgroups are unavailable", async () => {
  const adapterLimits = Object.fromEntries(
    PARAKEET_WEBGPU_LIMIT_NAMES.map((name) => [name, 1_024]),
  );
  let descriptor: GPUDeviceDescriptor | undefined;
  const device = {
    features: new Set(["timestamp-query"]),
    limits: adapterLimits,
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set(["timestamp-query"]),
    limits: adapterLimits,
    info: {},
    async requestDevice(requested: GPUDeviceDescriptor) {
      descriptor = requested;
      return device;
    },
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter() {
      return adapter;
    },
  } as unknown as GPU;

  const context = await requestWebGpuContext({ gpu });

  if (!descriptor) throw new Error("requestDevice was not invoked");
  expect(descriptor.requiredFeatures).toEqual(["timestamp-query"]);
  expect(context.executionProfile).toBe(
    PARAKEET_FP32_PORTABLE_EXECUTION_PROFILE,
  );
  expect(context.report.requiredFeatures).toEqual([]);
  expect(context.report.requestedOptionalFeatures).toEqual([
    "timestamp-query",
  ]);
  expect(context.report.stockFeatures["shader-f16"]).toEqual({
    adapterSupported: false,
    deviceEnabled: false,
    required: false,
    requested: false,
  });
  expect(context.report.stockFeatures["subgroups"]).toEqual({
    adapterSupported: false,
    deviceEnabled: false,
    required: false,
    requested: false,
  });
});
