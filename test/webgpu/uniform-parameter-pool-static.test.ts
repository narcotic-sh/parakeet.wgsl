import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const encoderSource = source("../../src/webgpu/encoder.ts");
const kernelSources = [
  source("../../src/webgpu/kernels/attention.ts"),
  source("../../src/webgpu/kernels/conformer-conv.ts"),
  source("../../src/webgpu/kernels/layernorm.ts"),
  source("../../src/webgpu/kernels/subsampling.ts"),
];

describe("encoder uniform-parameter ownership", () => {
  it("keeps non-GEMM parameter buffers on their kernel owners", () => {
    for (const kernelSource of kernelSources) {
      expect(kernelSource).toContain("new UniformParameterPool(");
      expect(kernelSource).toContain("this.parameterPool.destroy();");
      expect(kernelSource).not.toContain(
        "usage: GPUBufferUsage.UNIFORM",
      );
    }
  });

  it("cleans every constructed kernel after its dispatches", () => {
    expect(
      encoderSource.match(/ownedKernels\.push\(/g),
    ).toHaveLength(7);
    expect(
      encoderSource.match(
        /for \(const kernel of (?:this\.)?ownedKernels\.toReversed\(\)\)/g,
      ),
    ).toHaveLength(2);

    const failedConstructionCleanup = encoderSource.slice(
      encoderSource.indexOf("    } catch (error) {"),
      encoderSource.indexOf(
        "  uploadFeatures(",
        encoderSource.indexOf("    } catch (error) {"),
      ),
    );
    expect(failedConstructionCleanup.indexOf("dispatch.destroy();")).toBeLessThan(
      failedConstructionCleanup.indexOf("kernel.destroy();"),
    );
    expect(failedConstructionCleanup.indexOf("kernel.destroy();")).toBeLessThan(
      failedConstructionCleanup.indexOf("arena.destroy();"),
    );
  });
});

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}
