import { describe, expect, it } from "vitest";

import { PARAKEET_SAMPLE_RATE } from "../../src/audio/wav-stream";
import { STREAMING_DECODER_INITIAL_SAMPLES } from "../../src/audio/streaming-decoder-protocol";
import {
  PARAKEET_OVERLAP_SAMPLES,
  PARAKEET_PRIMARY_CONTEXT_SAMPLES,
  PARAKEET_PRIMARY_VISIBLE_SAMPLES,
  PARAKEET_REPAIR_VISIBLE_SAMPLES,
  PARAKEET_WINDOW_SAMPLES,
  PARAKEET_WINDOW_STRIDE_SAMPLES,
  OnlineStatelessWindowPlanner,
  planStatelessWindows,
} from "../../src/runtime/chunking";

describe("fixed stateless long-form chunking", () => {
  it("maps the frame-aligned visible grid onto fixed model storage", () => {
    const plans = planStatelessWindows(34 * PARAKEET_SAMPLE_RATE);

    expect(plans.length).toBeGreaterThanOrEqual(3);
    expect(plans.map((plan) => plan.logicalStartSample)).toEqual(
      plans.map(
        (_plan, index) => index * PARAKEET_WINDOW_STRIDE_SAMPLES,
      ),
    );
    expect(
      plans[0]!.logicalEndSample - plans[1]!.logicalStartSample,
    ).toBe(PARAKEET_OVERLAP_SAMPLES);
    expect(
      plans.every(
        (plan) => plan.windowSampleCount === PARAKEET_WINDOW_SAMPLES,
      ),
    ).toBe(true);
    expect(PARAKEET_PRIMARY_VISIBLE_SAMPLES).toBe(238_080);
    expect(PARAKEET_REPAIR_VISIBLE_SAMPLES).toBe(239_360);
    expect(plans[0]).toMatchObject({
      sourceStartSample: 0,
      sourceEndSample: 238_080,
      readSampleCount: 238_080,
      decodeStartFrame: 0,
      decodeEndFrame: 186,
    });
    expect(plans[1]).toMatchObject({
      sourceStartSample:
        PARAKEET_WINDOW_STRIDE_SAMPLES -
        PARAKEET_PRIMARY_CONTEXT_SAMPLES,
      timestampOffsetSample: PARAKEET_WINDOW_STRIDE_SAMPLES,
      readSampleCount:
        PARAKEET_PRIMARY_VISIBLE_SAMPLES +
        PARAKEET_PRIMARY_CONTEXT_SAMPLES,
      decodeStartFrame: 1,
      decodeEndFrame: 186,
    });
  });

  it("plans Cowen on the default 12.88-second grid", () => {
    expect(planStatelessWindows(58_134_779)).toHaveLength(282);
    expect(planStatelessWindows(58_134_779).at(-1)).toMatchObject({
      logicalStartSample: 57_908_480,
      logicalEndSample: 58_134_779,
      sourceStartSample: 57_896_960,
      sourceEndSample: 58_134_779,
      timestampOffsetSample: 57_896_960,
      emissionStartSample: 57_908_480,
      decodeStartFrame: 0,
      decodeEndFrame: 186,
    });
  });

  it("covers arbitrary non-empty lengths through the final sample", () => {
    for (const sampleCount of [
      1,
      80_000,
      PARAKEET_WINDOW_STRIDE_SAMPLES,
      PARAKEET_WINDOW_SAMPLES,
      PARAKEET_WINDOW_SAMPLES + 1,
      1_234_567,
    ]) {
      const plans = planStatelessWindows(sampleCount);
      expect(plans.length).toBeGreaterThan(0);
      expect(plans[0]?.logicalStartSample).toBe(0);
      expect(plans.at(-1)?.logicalEndSample).toBe(sampleCount);
      expect(
        plans.every(
          (plan, index) =>
            plan.index === index &&
            plan.logicalStartSample ===
              index * PARAKEET_WINDOW_STRIDE_SAMPLES,
        ),
      ).toBe(true);
    }
  });

  it("plans one complete source window", () => {
    expect(planStatelessWindows(PARAKEET_WINDOW_SAMPLES)).toEqual([
      expect.objectContaining({
        index: 0,
        sourceStartSample: 0,
        sourceEndSample: PARAKEET_PRIMARY_VISIBLE_SAMPLES,
        readSampleCount: PARAKEET_PRIMARY_VISIBLE_SAMPLES,
        validSampleCount: PARAKEET_PRIMARY_VISIBLE_SAMPLES,
      }),
      expect.objectContaining({
        index: 1,
        sourceEndSample: PARAKEET_WINDOW_SAMPLES,
      }),
    ]);
  });

  it("end-aligns the final window with frame-aligned real-audio backfill", () => {
    expect(planStatelessWindows(34 * PARAKEET_SAMPLE_RATE).at(-1)).toMatchObject({
      sourceStartSample: 305_920,
      sourceEndSample: 544_000,
      timestampOffsetSample: 305_920,
      logicalStartSample: 412_160,
      emissionStartSample: 412_160,
      validSampleCount: 238_080,
      decodeStartFrame: 0,
      decodeEndFrame: 186,
    });
  });

  it("ends a backfilled physical window at the last speech-bearing sample", () => {
    const totalSamples = 34 * PARAKEET_SAMPLE_RATE;
    const speechEndSample = totalSamples - 40 * 1_280;
    const final = planStatelessWindows(totalSamples, speechEndSample).at(-1);

    expect(final).toMatchObject({
      logicalStartSample: 412_160,
      logicalEndSample: speechEndSample,
      sourceEndSample: speechEndSample,
      emissionStartSample: 412_160,
      decodeStartFrame: 0,
      decodeEndFrame: 186,
      flushFinal: false,
    });
    expect(final!.readSampleCount).toBeLessThanOrEqual(
      PARAKEET_WINDOW_SAMPLES,
    );
  });

  it("retains FluidAudio's sub-frame tail after end backfill", () => {
    const totalSamples = 34 * PARAKEET_SAMPLE_RATE;
    const speechEndSample =
      totalSamples - 40 * 1_280 - 640;
    expect(
      planStatelessWindows(totalSamples, speechEndSample).at(-1),
    ).toMatchObject({
      sourceStartSample: 254_720,
      sourceEndSample: speechEndSample + 640,
      readSampleCount: PARAKEET_PRIMARY_VISIBLE_SAMPLES,
      emissionEndSample: speechEndSample + 640,
      decodeStartFrame: 0,
      decodeEndFrame: 186,
    });
  });

  it("retains FluidAudio's extra grid iteration after a non-final backfill", () => {
    const secondStart = PARAKEET_WINDOW_STRIDE_SAMPLES;
    const totalSamples = secondStart + 230_000;
    const speechEndSample = secondStart + 220_000;
    const plans = planStatelessWindows(
      totalSamples,
      speechEndSample,
    );

    expect(plans).toHaveLength(3);
    expect(plans[1]).toMatchObject({
      logicalStartSample: secondStart,
      logicalEndSample: secondStart + 220_160,
      sourceEndSample: secondStart + 220_160,
      flushFinal: false,
    });
    expect(plans[2]).toMatchObject({
      logicalStartSample: 2 * PARAKEET_WINDOW_STRIDE_SAMPLES,
      logicalEndSample: speechEndSample + 160,
      sourceEndSample: speechEndSample + 160,
      flushFinal: false,
    });
  });

  it("right-pads a short file", () => {
    expect(planStatelessWindows(5 * PARAKEET_SAMPLE_RATE)).toEqual([
      expect.objectContaining({
        sourceStartSample: 0,
        readSampleCount: 80_000,
        validSampleCount: 80_000,
        windowSampleCount: PARAKEET_WINDOW_SAMPLES,
        decodeStartFrame: 0,
        decodeEndFrame: 63,
      }),
    ]);
  });

  it("rejects invalid lengths", () => {
    expect(planStatelessWindows(0)).toEqual([]);
    expect(() => planStatelessWindows(-1)).toThrow(/non-negative/);
    expect(() => planStatelessWindows(1, 2)).toThrow(/must not exceed/);
  });

  it("waits at an exact-boundary EOF instead of misclassifying it as non-final", () => {
    const planner = new OnlineStatelessWindowPlanner();

    expect(planner.takeStablePrefix(PARAKEET_PRIMARY_VISIBLE_SAMPLES)).toEqual(
      [],
    );
    const resolved = planner.finish(
      PARAKEET_PRIMARY_VISIBLE_SAMPLES,
      PARAKEET_PRIMARY_VISIBLE_SAMPLES,
    );
    expect(resolved.allPlans).toHaveLength(1);
    expect(resolved.allPlans[0]).toMatchObject({ flushFinal: true });
  });

  it("emits a stable window after one sample of strict lookahead", () => {
    const planner = new OnlineStatelessWindowPlanner();
    const totalSamples = PARAKEET_PRIMARY_VISIBLE_SAMPLES + 1;

    expect(planner.takeStablePrefix(totalSamples)).toEqual([
      expect.objectContaining({ index: 0, flushFinal: false }),
    ]);
    const resolved = planner.finish(totalSamples, totalSamples);
    expect(resolved.allPlans[0]).toEqual(
      expect.objectContaining({ index: 0, flushFinal: false }),
    );
    expect(resolved.remainingPlans).toEqual([
      expect.objectContaining({ index: 1, flushFinal: true }),
    ]);
  });

  it("preserves B40/B40-plus-tail online partitions at EOF", () => {
    const firstGraphEnd =
      39 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES;
    const secondGraphEnd =
      79 * PARAKEET_WINDOW_STRIDE_SAMPLES +
      PARAKEET_PRIMARY_VISIBLE_SAMPLES;
    const planner = new OnlineStatelessWindowPlanner();

    expect(STREAMING_DECODER_INITIAL_SAMPLES).toBe(firstGraphEnd + 1);
    expect(planner.takeStablePrefix(firstGraphEnd + 1)).toHaveLength(40);
    expect(planner.takeStablePrefix(secondGraphEnd + 1)).toHaveLength(40);
    const final = planner.finish(secondGraphEnd + 73_000, secondGraphEnd + 73_000);
    expect(final.allPlans.slice(0, 40).map(({ index }) => index)).toEqual(
      Array.from({ length: 40 }, (_, index) => index),
    );
    expect(final.allPlans.slice(40, 80).map(({ index }) => index)).toEqual(
      Array.from({ length: 40 }, (_, index) => index + 40),
    );
    expect(final.remainingPlans.length).toBeGreaterThan(0);
    expect(final.remainingPlans[0]!.index).toBe(80);
  });
});
