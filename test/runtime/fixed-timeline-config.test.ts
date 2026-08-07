import { describe, expect, it } from "vitest";

import {
  PARAKEET_FBANK_MAX_FRAMES,
  PARAKEET_FBANK_MAX_INPUT_SAMPLES,
  reusableFrameShiftForWindows,
} from "../../src/audio/fbank-wasm";
import {
  PARAKEET_ENCODER_FRAME_SAMPLES,
  PARAKEET_FBANK_HOP_SAMPLES,
  PARAKEET_OVERLAP_SAMPLES,
  PARAKEET_PRIMARY_CONTEXT_SAMPLES,
  PARAKEET_PRIMARY_VISIBLE_SAMPLES,
  PARAKEET_WINDOW_SAMPLES,
  PARAKEET_WINDOW_STRIDE_SAMPLES,
  planStatelessWindows,
} from "../../src/runtime/chunking";

describe("fixed production timeline", () => {
  it("pins one integer timeline", () => {
    expect(PARAKEET_WINDOW_SAMPLES).toBe(240_000);
    expect(PARAKEET_PRIMARY_VISIBLE_SAMPLES).toBe(238_080);
    expect(PARAKEET_PRIMARY_CONTEXT_SAMPLES).toBe(1_280);
    expect(PARAKEET_OVERLAP_SAMPLES).toBe(32_000);
    expect(PARAKEET_WINDOW_STRIDE_SAMPLES).toBe(206_080);
    expect(PARAKEET_FBANK_HOP_SAMPLES).toBe(160);
    expect(PARAKEET_ENCODER_FRAME_SAMPLES).toBe(1_280);
    expect(
      PARAKEET_WINDOW_SAMPLES / PARAKEET_FBANK_HOP_SAMPLES,
    ).toBe(1_500);
    expect(
      PARAKEET_WINDOW_STRIDE_SAMPLES / PARAKEET_FBANK_HOP_SAMPLES,
    ).toBe(1_288);
    expect(
      PARAKEET_OVERLAP_SAMPLES / PARAKEET_FBANK_HOP_SAMPLES,
    ).toBe(200);
    expect(
      PARAKEET_ENCODER_FRAME_SAMPLES / PARAKEET_FBANK_HOP_SAMPLES,
    ).toBe(8);
    expect(PARAKEET_FBANK_MAX_INPUT_SAMPLES).toBe(
      PARAKEET_WINDOW_SAMPLES,
    );
    expect(PARAKEET_FBANK_MAX_FRAMES).toBe(1_501);
  });

  it("pins the raw Cowen file-length grid independently of silence trim", () => {
    const plans = planStatelessWindows(58_134_779);
    expect(plans).toHaveLength(282);
    expect(Math.floor(plans.length / 40)).toBe(7);
    expect(plans.length % 40).toBe(2);

    const previous = plans.at(-2)!;
    const last = plans.at(-1)!;
    expect(reuseShift(previous, last)).toBe(1_224);
  });

  it("plans silence-aware Cowen as seven B40 batches and one active B3 tail", () => {
    const plans = planStatelessWindows(58_134_779, 58_124_539);
    expect(plans).toHaveLength(283);
    expect(Math.floor(plans.length / 40)).toBe(7);
    expect(plans.length % 40).toBe(3);
  });

  it("reuses the stable interior of regular windows", () => {
    const plans = planStatelessWindows(
      PARAKEET_WINDOW_SAMPLES +
        PARAKEET_WINDOW_STRIDE_SAMPLES,
    );
    const reusableFrameShift = reuseShift(plans[0]!, plans[1]!);
    expect(reusableFrameShift).toBe(1_280);
    expect(
      PARAKEET_FBANK_MAX_FRAMES - 1 - reusableFrameShift - 2 * 3,
    ).toBe(214);
  });
});

function reuseShift(
  previous: ReturnType<typeof planStatelessWindows>[number],
  current: ReturnType<typeof planStatelessWindows>[number],
): number {
  return reusableFrameShiftForWindows(
    {
      startSample: previous.sourceStartSample,
      sampleCount: previous.windowSampleCount,
      validSampleCount: previous.validSampleCount,
    },
    {
      startSample: current.sourceStartSample,
      sampleCount: current.windowSampleCount,
      validSampleCount: current.validSampleCount,
    },
  );
}
