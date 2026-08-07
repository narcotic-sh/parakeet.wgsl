import { describe, expect, it } from "vitest";

import {
  GpuTimestampRecorder,
  gpuTimestampSpanMilliseconds,
  type GpuTimestampSample,
} from "../../src/webgpu/timestamp";

function sample(
  label: string,
  startNanoseconds: bigint,
  endNanoseconds: bigint,
): GpuTimestampSample {
  const durationNanoseconds = Number(endNanoseconds - startNanoseconds);
  return {
    label,
    startNanoseconds,
    endNanoseconds,
    durationNanoseconds,
    durationMilliseconds: durationNanoseconds / 1_000_000,
  };
}

describe("gpuTimestampSpanMilliseconds", () => {
  it("measures from the first pass beginning through the last pass end", () => {
    const samples = [
      sample("encoder-first-pass", 1_000_000n, 2_000_000n),
      sample("encoder-last-pass", 8_000_000n, 11_000_000n),
    ];

    expect(
      gpuTimestampSpanMilliseconds(
        samples,
        "encoder-first-pass",
        "encoder-last-pass",
      ),
    ).toBe(10);
  });

  it("measures one pass when both labels are the same", () => {
    const samples = [
      sample("tdt-decoder", 2_500_000n, 6_000_000n),
    ];

    expect(
      gpuTimestampSpanMilliseconds(
        samples,
        "tdt-decoder",
        "tdt-decoder",
      ),
    ).toBe(3.5);
  });

  it.each([
    {
      name: "a missing endpoint",
      samples: [sample("encoder-first-pass", 1n, 2n)],
    },
    {
      name: "zero unavailable samples",
      samples: [
        sample("encoder-first-pass", 0n, 0n),
        sample("encoder-last-pass", 0n, 0n),
      ],
    },
    {
      name: "a non-monotonic span",
      samples: [
        sample("encoder-first-pass", 20n, 30n),
        sample("encoder-last-pass", 5n, 10n),
      ],
    },
  ])("rejects $name", ({ samples }) => {
    expect(
      gpuTimestampSpanMilliseconds(
        samples,
        "encoder-first-pass",
        "encoder-last-pass",
      ),
    ).toBeUndefined();
  });

  it("rejects a span whose bigint conversion is not finite", () => {
    const enormous = 10n ** 400n;
    expect(
      gpuTimestampSpanMilliseconds(
        [
          sample("encoder-first-pass", 0n, 1n),
          sample("encoder-last-pass", enormous - 1n, enormous),
        ],
        "encoder-first-pass",
        "encoder-last-pass",
      ),
    ).toBeUndefined();
  });
});

describe("GpuTimestampRecorder disabled mode", () => {
  it("does not inspect features or allocate/encode query resources", async () => {
    const device = {
      features: {
        has(): boolean {
          throw new Error("disabled recorder must not inspect device features");
        },
      },
      createQuerySet(): never {
        throw new Error("disabled recorder must not allocate a query set");
      },
      createBuffer(): never {
        throw new Error("disabled recorder must not allocate buffers");
      },
    } as unknown as GPUDevice;
    const recorder = new GpuTimestampRecorder(device, 3, false);
    const encoder = {
      resolveQuerySet(): never {
        throw new Error("disabled recorder must not encode query resolution");
      },
      copyBufferToBuffer(): never {
        throw new Error("disabled recorder must not encode query readback");
      },
    } as unknown as GPUCommandEncoder;

    expect(recorder.supported).toBe(false);
    recorder.reset();
    expect(recorder.allocate("encoder-first-pass")).toBeUndefined();
    recorder.resolve(encoder);
    await expect(recorder.read()).resolves.toEqual([]);
    recorder.destroy();
  });
});
