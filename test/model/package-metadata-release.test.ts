import { describe, expect, it, vi } from "vitest";

import {
  PARAKEET_FP16_PACKAGE_FORMAT,
  type ParakeetManifest,
} from "../../src/model/manifest";
import { ParakeetGpuPackage } from "../../src/model/package";
import type {
  RuntimePackageMemoryPlan,
  RuntimeTensorPlan,
} from "../../src/model/runtime-plan";

describe("ParakeetGpuPackage initialization metadata", () => {
  it("releases CPU-only construction records without affecting GPU teardown", () => {
    const destroy = vi.fn();
    const buffer = { destroy } as unknown as GPUBuffer;
    const manifest = {
      format: PARAKEET_FP16_PACKAGE_FORMAT,
      source: { repository: "test", revision: "test" },
      config: {},
      files: [],
    } as unknown as ParakeetManifest;
    const tokenizer = { model: { vocab: {} } };
    const memory = {
      runtimeGpuBytes: 256,
    } as RuntimePackageMemoryPlan;
    const tensorPlans = new Map<string, RuntimeTensorPlan>();
    const model = Reflect.construct(ParakeetGpuPackage, [
      manifest,
      tensorPlans,
      new Map([["retained", buffer]]),
      tokenizer,
      memory,
      {
        manifestUrl: "/model/files/manifest.json",
        manifestSha256: "0".repeat(64),
      },
    ]) as ParakeetGpuPackage;

    expect(model.precision).toBe("fp16");
    expect(model.manifestUrl).toBe("/model/files/manifest.json");
    expect(model.manifestSha256).toBe("0".repeat(64));
    expect(model.manifestListedBytes).toBe(0);
    expect(model.manifest).toBe(manifest);
    expect(model.tokenizer).toBe(tokenizer);

    model.releaseInitializationMetadata();

    expect(() => model.manifest).toThrow("metadata was released");
    expect(() => model.tokenizer).toThrow("metadata was released");
    expect(() => model.tensor("missing")).toThrow(
      "tensor metadata was released",
    );

    model.destroy();
    expect(destroy).toHaveBeenCalledOnce();
  });
});
