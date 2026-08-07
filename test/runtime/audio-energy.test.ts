import { describe, expect, it } from "vitest";

import {
  AdaptiveSpeechRmsAccumulator,
  SEAM_GAP_MIN_SPEECH_SECONDS,
  SPEECH_RMS_CEILING,
  SPEECH_RMS_FLOOR,
  adaptiveSpeechRmsThreshold,
  completeAdaptiveSpeechThresholdOnly,
  findSpeechEndSample,
  scanAdaptiveSpeechThreshold,
  scanAdaptiveSpeechThresholdOnly,
  speechLikeSeconds,
  type PcmSampleReader,
} from "../../src/runtime/audio-energy";

const FRAME_SAMPLES = 1_280;

describe("long-form speech energy", () => {
  it("retains the independent half-second seam-repair speech gate", () => {
    expect(SEAM_GAP_MIN_SPEECH_SECONDS).toBe(0.5);
  });

  it("applies the adaptive threshold clamps", () => {
    expect(adaptiveSpeechRmsThreshold(0.1)).toBe(
      SPEECH_RMS_CEILING,
    );
    expect(adaptiveSpeechRmsThreshold(0.0028)).toBeCloseTo(
      0.00084,
    );
    expect(adaptiveSpeechRmsThreshold(0)).toBe(SPEECH_RMS_FLOOR);
  });

  it("finds the last speech-bearing frame and preserves all-silence EOF", async () => {
    const samples = new Float32Array(FRAME_SAMPLES * 5);
    samples.fill(0.01, FRAME_SAMPLES, FRAME_SAMPLES * 3);
    expect(
      await findSpeechEndSample(sampleReader(samples)),
    ).toBe(FRAME_SAMPLES * 3);
    expect(
      await findSpeechEndSample(
        sampleReader(new Float32Array(FRAME_SAMPLES * 2)),
      ),
    ).toBe(FRAME_SAMPLES * 2);
  });

  it("preserves EOF for nonzero audio below the speech floor", async () => {
    const samples = new Float32Array(FRAME_SAMPLES * 2);
    samples.fill(Math.fround(SPEECH_RMS_FLOOR / 4));
    expect(
      await findSpeechEndSample(sampleReader(samples)),
    ).toBe(samples.length);
  });

  it("anchors the final partial speech frame at EOF", async () => {
    const samples = new Float32Array(FRAME_SAMPLES * 2 + 100);
    samples.fill(0.01, samples.length - 100);
    expect(
      await findSpeechEndSample(sampleReader(samples)),
    ).toBe(samples.length);
  });

  it("excludes digital silence and adapts to quiet speech", async () => {
    const samples = new Float32Array(FRAME_SAMPLES * 8);
    samples.fill(0.0028, FRAME_SAMPLES * 2);
    const result = await scanAdaptiveSpeechThreshold(
      sampleReader(samples),
    );

    expect(result.nonzeroFrameCount).toBe(6);
    expect(result.referenceRms).toBeCloseTo(0.0028, 4);
    expect(result.threshold).toBeCloseTo(0.00084, 4);
  });

  it("falls back to the ceiling for an all-digital-silence file", async () => {
    const result = await scanAdaptiveSpeechThreshold(
      sampleReader(new Float32Array(FRAME_SAMPLES * 2)),
    );
    expect(result).toEqual({
      threshold: SPEECH_RMS_CEILING,
      referenceRms: 0,
      nonzeroFrameCount: 0,
    });
  });

  it("skips the exact low-bit pass when the high bucket proves the ceiling clamp", async () => {
    const samples = constantFrames(8, 0.1);
    let reads = 0;

    const threshold = await scanAdaptiveSpeechThresholdOnly(
      sampleReader(samples, () => {
        reads += 1;
      }),
    );

    expect(threshold).toBe(SPEECH_RMS_CEILING);
    expect(reads).toBe(1);
  });

  it("skips the exact low-bit pass when the high bucket proves the floor clamp", async () => {
    const samples = constantFrames(8, 0.0001);
    let reads = 0;

    const threshold = await scanAdaptiveSpeechThresholdOnly(
      sampleReader(samples, () => {
        reads += 1;
      }),
    );

    expect(threshold).toBe(SPEECH_RMS_FLOOR);
    expect(reads).toBe(1);
  });

  it("resolves exact low bits when the selected bucket is unclamped", async () => {
    const samples = constantFrames(8, 0.0028);
    let reads = 0;

    const threshold = await scanAdaptiveSpeechThresholdOnly(
      sampleReader(samples, () => {
        reads += 1;
      }),
    );

    expect(threshold).toBe(
      adaptiveSpeechRmsThreshold(
        fluidFrameRms(Math.fround(0.0028)),
      ),
    );
    expect(reads).toBe(2);
  });

  it("finishes an incremental high-bit pass from only the unseen suffix", async () => {
    const samples = constantFrames(8, 0.1);
    const accumulator = new AdaptiveSpeechRmsAccumulator();
    accumulator.addContiguousFrameBits(
      0,
      Uint32Array.from(
        { length: 6 },
        () => floatBits(fluidFrameRms(Math.fround(0.1))),
      ),
    );
    const reads: Array<{
      readonly startSample: number;
      readonly sampleCount: number;
    }> = [];

    const threshold = await completeAdaptiveSpeechThresholdOnly(
      sampleReader(
        samples,
        (startSample, sampleCount) => {
          reads.push({ startSample, sampleCount });
        },
      ),
      accumulator,
    );

    expect(threshold).toBe(SPEECH_RMS_CEILING);
    expect(reads).toEqual([
      {
        startSample: 6 * FRAME_SAMPLES,
        sampleCount: 2 * FRAME_SAMPLES,
      },
    ]);
    expect(accumulator.nextFrameStartSample).toBe(samples.length);
  });

  it("retains the exact full low-bit fallback after incremental accumulation", async () => {
    const amplitude = Math.fround(0.0028);
    const samples = constantFrames(8, amplitude);
    const accumulator = new AdaptiveSpeechRmsAccumulator();
    accumulator.addContiguousFrameBits(
      0,
      Uint32Array.of(
        floatBits(fluidFrameRms(amplitude)),
        floatBits(fluidFrameRms(amplitude)),
        floatBits(fluidFrameRms(amplitude)),
      ),
    );
    const readStarts: number[] = [];

    const threshold = await completeAdaptiveSpeechThresholdOnly(
      sampleReader(samples, (startSample) => {
        readStarts.push(startSample);
      }),
      accumulator,
    );

    expect(threshold).toBe(
      adaptiveSpeechRmsThreshold(fluidFrameRms(amplitude)),
    );
    expect(readStarts).toEqual([3 * FRAME_SAMPLES, 0]);
  });

  it("rejects duplicate or noncontiguous incremental RMS frames", () => {
    const accumulator = new AdaptiveSpeechRmsAccumulator();
    accumulator.addContiguousFrameBits(
      0,
      Uint32Array.of(floatBits(0.01)),
    );
    expect(() =>
      accumulator.addContiguousFrameBits(
        0,
        Uint32Array.of(floatBits(0.01)),
      ),
    ).toThrow(/continue at sample/);
    expect(() =>
      accumulator.addContiguousFrameBits(
        FRAME_SAMPLES * 2,
        Uint32Array.of(floatBits(0.01)),
      ),
    ).toThrow(/continue at sample/);
  });

  it("rejects negative-zero, infinity, and NaN RMS bit patterns", () => {
    for (const bits of [
      0x8000_0000,
      0x7f80_0000,
      0x7fc0_0000,
    ]) {
      const accumulator = new AdaptiveSpeechRmsAccumulator();
      expect(() =>
        accumulator.addContiguousFrameBits(
          0,
          Uint32Array.of(bits),
        ),
      ).toThrow(/non-negative finite Float32/);
      expect(accumulator.nextFrameStartSample).toBe(0);
    }
  });

  it("keeps the exact-reference API exact even when its threshold clamps", async () => {
    const amplitude = Math.fround(0.1);
    const samples = constantFrames(8, amplitude);
    let reads = 0;

    const result = await scanAdaptiveSpeechThreshold(
      sampleReader(samples, () => {
        reads += 1;
      }),
    );

    expect(result).toEqual({
      threshold: SPEECH_RMS_CEILING,
      referenceRms: fluidFrameRms(amplitude),
      nonzeroFrameCount: 8,
    });
    expect(reads).toBe(2);
  });

  it("selects FluidAudio's exact Float32 p75 value", async () => {
    const amplitudes = [
      0.100007,
      0.100001,
      0.100006,
      0.100003,
      0.100008,
      0.100002,
      0.100005,
      0.100004,
    ].map(Math.fround);
    const samples = new Float32Array(
      FRAME_SAMPLES * amplitudes.length,
    );
    for (const [index, amplitude] of amplitudes.entries()) {
      samples.fill(
        amplitude,
        index * FRAME_SAMPLES,
        (index + 1) * FRAME_SAMPLES,
      );
    }
    const expected = amplitudes
      .map(fluidFrameRms)
      .sort((left, right) => left - right)[
      Math.floor(amplitudes.length * 0.75)
    ]!;
    const result = await scanAdaptiveSpeechThreshold(
      sampleReader(samples),
    );

    expect(result.referenceRms).toBe(expected);
  });

  it("excludes an incomplete high-energy tail from the p75 reference", async () => {
    const amplitude = Math.fround(0.002);
    const samples = new Float32Array(FRAME_SAMPLES + 100);
    samples.fill(amplitude, 0, FRAME_SAMPLES);
    samples.fill(0.5, FRAME_SAMPLES);
    const result = await scanAdaptiveSpeechThreshold(
      sampleReader(samples),
    );

    expect(result.nonzeroFrameCount).toBe(1);
    expect(result.referenceRms).toBe(
      fluidFrameRms(amplitude),
    );
  });

  it("multiplies the adaptive reference with Float32 constants", () => {
    const storage = new ArrayBuffer(4);
    const words = new Uint32Array(storage);
    const floats = new Float32Array(storage);
    words[0] = 0x3ada7428;
    const reference = floats[0]!;
    const expected = Math.min(
      SPEECH_RMS_CEILING,
      Math.max(
        SPEECH_RMS_FLOOR,
        Math.fround(reference * Math.fround(0.3)),
      ),
    );

    expect(adaptiveSpeechRmsThreshold(reference)).toBe(expected);
  });

  it("counts only complete frames strictly above the gate", async () => {
    const samples = new Float32Array(FRAME_SAMPLES * 4 + 200);
    samples.fill(0.01, FRAME_SAMPLES, FRAME_SAMPLES * 3);
    expect(
      await speechLikeSeconds(
        sampleReader(samples),
        0,
        samples.length,
        0.008,
      ),
    ).toBeCloseTo(0.16);
  });

  it("does not count a frame exactly equal to the speech gate", async () => {
    const amplitude = Math.fround(0.01);
    const samples = new Float32Array(FRAME_SAMPLES);
    samples.fill(amplitude);
    expect(
      await speechLikeSeconds(
        sampleReader(samples),
        0,
        samples.length,
        fluidFrameRms(amplitude),
      ),
    ).toBe(0);
  });
});

function constantFrames(
  frameCount: number,
  amplitude: number,
): Float32Array {
  const samples = new Float32Array(FRAME_SAMPLES * frameCount);
  samples.fill(Math.fround(amplitude));
  return samples;
}

function sampleReader(
  samples: Float32Array,
  onRead?: (startSample: number, sampleCount: number) => void,
): PcmSampleReader {
  return {
    sampleCount: samples.length,
    async readSamplesInto(
      startSample,
      target,
      targetOffset = 0,
      sampleCount = target.length - targetOffset,
    ) {
      const count = Math.min(
        sampleCount,
        Math.max(0, samples.length - startSample),
      );
      onRead?.(startSample, count);
      target.set(
        samples.subarray(startSample, startSample + count),
        targetOffset,
      );
      return count;
    },
  };
}

function fluidFrameRms(amplitude: number): number {
  let sum = Math.fround(0);
  for (let index = 0; index < FRAME_SAMPLES; index += 1) {
    sum = Math.fround(
      sum + Math.fround(amplitude * amplitude),
    );
  }
  return Math.fround(
    Math.sqrt(Math.fround(sum / FRAME_SAMPLES)),
  );
}

function floatBits(value: number): number {
  const storage = new ArrayBuffer(4);
  new Float32Array(storage)[0] = value;
  return new Uint32Array(storage)[0]!;
}
