import { PARAKEET_SAMPLE_RATE } from "../audio/wav-stream";
import { PARAKEET_ENCODER_FRAME_SAMPLES } from "./chunking";

export const SPEECH_RMS_CEILING = Math.fround(0.008);
export const SPEECH_RMS_FLOOR = Math.fround(0.0005);
export const SEAM_GAP_MIN_SPEECH_SECONDS = 0.5;

const SPEECH_RMS_REFERENCE_SCALE = Math.fround(0.3);
const SPEECH_RMS_REFERENCE_PERCENTILE = 0.75;
const RMS_RADIX_BUCKETS = 65_536;
// Exactly fills the fixed 320,000-byte PCM16 BYOB reader arena.
const SCAN_FRAMES_PER_READ = 125;

export interface PcmSampleReader {
  readonly sampleCount: number;
  readSamplesInto(
    startSample: number,
    target: Float32Array,
    targetOffset?: number,
    sampleCount?: number,
  ): Promise<number>;
}

export interface AdaptiveSpeechThreshold {
  readonly threshold: number;
  readonly referenceRms: number;
  readonly nonzeroFrameCount: number;
}

interface ReferenceHighBucket {
  readonly histogram: Uint32Array;
  readonly nonzeroFrameCount: number;
  readonly highBits: number;
  readonly indexWithinHighBucket: number;
}

/**
 * Exact bounded-memory high-radix state for the adaptive speech threshold.
 *
 * Primary PCM windows append only the next globally aligned 80 ms frames.
 * Their overlap is therefore never counted twice. Completion scans any
 * contiguous suffix that was not already resident in the Wasm input arena.
 */
export class AdaptiveSpeechRmsAccumulator {
  private readonly histogram = new Uint32Array(RMS_RADIX_BUCKETS);
  private nonzeroFrameCount = 0;
  private nextSample = 0;
  private finished = false;

  get nextFrameStartSample(): number {
    return this.nextSample;
  }

  addContiguousFrameBits(
    startSample: number,
    bits: Uint32Array,
  ): void {
    this.requireAppendStart(startSample);
    for (const value of bits) {
      this.addRmsBits(value);
    }
    this.advance(bits.length);
  }

  addContiguousFrameRms(
    startSample: number,
    rms: number,
  ): void {
    this.requireAppendStart(startSample);
    this.addRmsBits(float32Bits(rms));
    this.advance(1);
  }

  finishReferenceHighBucket(): ReferenceHighBucket {
    if (this.finished) {
      throw new Error("Adaptive speech RMS accumulator is already finished");
    }
    this.finished = true;
    if (this.nonzeroFrameCount === 0) {
      return {
        histogram: this.histogram,
        nonzeroFrameCount: 0,
        highBits: 0,
        indexWithinHighBucket: 0,
      };
    }

    const referenceIndex = Math.min(
      this.nonzeroFrameCount - 1,
      Math.floor(
        this.nonzeroFrameCount *
          SPEECH_RMS_REFERENCE_PERCENTILE,
      ),
    );
    let cumulative = 0;
    let highBits = 0;
    for (; highBits < this.histogram.length; highBits += 1) {
      cumulative += this.histogram[highBits]!;
      if (cumulative > referenceIndex) break;
    }
    const indexWithinHighBucket =
      referenceIndex -
      (cumulative - this.histogram[highBits]!);
    return {
      histogram: this.histogram,
      nonzeroFrameCount: this.nonzeroFrameCount,
      highBits,
      indexWithinHighBucket,
    };
  }

  private addRmsBits(bits: number): void {
    if (
      !Number.isSafeInteger(bits) ||
      bits < 0 ||
      bits > 0xffff_ffff ||
      (bits & 0x8000_0000) !== 0 ||
      (bits & 0x7f80_0000) === 0x7f80_0000
    ) {
      throw new RangeError(
        "RMS bits must encode a non-negative finite Float32 value",
      );
    }
    // FluidAudio excludes only exact positive digital silence. The native RMS
    // result cannot be negative zero, NaN, or infinity.
    if (bits === 0) return;
    const bucket = bits >>> 16;
    this.histogram[bucket] = this.histogram[bucket]! + 1;
    this.nonzeroFrameCount += 1;
  }

  private requireAppendStart(startSample: number): void {
    if (this.finished) {
      throw new Error("Adaptive speech RMS accumulator is already finished");
    }
    if (
      !Number.isSafeInteger(startSample) ||
      startSample !== this.nextSample
    ) {
      throw new RangeError(
        `Adaptive RMS frames must continue at sample ${this.nextSample}`,
      );
    }
  }

  private advance(frameCount: number): void {
    const next =
      this.nextSample +
      frameCount * PARAKEET_ENCODER_FRAME_SAMPLES;
    if (!Number.isSafeInteger(next)) {
      throw new RangeError("Adaptive RMS sample position is unsafe");
    }
    this.nextSample = next;
  }
}

export function adaptiveSpeechRmsThreshold(referenceRms: number): number {
  if (!Number.isFinite(referenceRms) || referenceRms < 0) {
    throw new RangeError("referenceRms must be non-negative and finite");
  }
  return Math.min(
    SPEECH_RMS_CEILING,
    Math.max(
      SPEECH_RMS_FLOOR,
      Math.fround(referenceRms * SPEECH_RMS_REFERENCE_SCALE),
    ),
  );
}

/**
 * Find the end of the final speech-bearing 80 ms frame. An all-sub-floor file
 * deliberately returns EOF unchanged, matching FluidAudio's final-tail rule.
 */
export async function findSpeechEndSample(
  reader: PcmSampleReader,
  signal?: AbortSignal,
): Promise<number> {
  const scanSamples =
    SCAN_FRAMES_PER_READ * PARAKEET_ENCODER_FRAME_SAMPLES;
  const samples = new Float32Array(scanSamples);
  let endSample = reader.sampleCount;
  while (endSample > 0) {
    throwIfAborted(signal);
    const startSample = Math.max(
      0,
      endSample - scanSamples,
    );
    const sampleCount = endSample - startSample;
    await readExactly(
      reader,
      startSample,
      samples,
      sampleCount,
    );
    for (let localEnd = sampleCount; localEnd > 0; ) {
      const localStart = Math.max(
        0,
        localEnd - PARAKEET_ENCODER_FRAME_SAMPLES,
      );
      if (
        rootMeanSquare(
          samples,
          localEnd - localStart,
          localStart,
        ) >= SPEECH_RMS_FLOOR
      ) {
        return startSample + localEnd;
      }
      localEnd = localStart;
    }
    endSample = startSample;
  }
  return reader.sampleCount;
}

/**
 * Streaming, bounded-memory equivalent of FluidAudio's exact sorted-Float p75
 * selection. Positive IEEE-754 Float32 values have the same ordering as their
 * unsigned bit patterns, so two 16-bit radix passes locate the exact selected
 * value without retaining one RMS value per frame.
 */
export async function scanAdaptiveSpeechThreshold(
  reader: PcmSampleReader,
  signal?: AbortSignal,
): Promise<AdaptiveSpeechThreshold> {
  const bucket = await scanReferenceHighBucket(reader, signal);
  if (bucket.nonzeroFrameCount === 0) {
    return {
      threshold: SPEECH_RMS_CEILING,
      referenceRms: 0,
      nonzeroFrameCount: 0,
    };
  }

  const referenceRms = await selectReferenceRms(
    reader,
    bucket,
    signal,
  );
  return {
    threshold: adaptiveSpeechRmsThreshold(referenceRms),
    referenceRms,
    nonzeroFrameCount: bucket.nonzeroFrameCount,
  };
}

/**
 * Production threshold scan. When the selected high-16 Float32 bucket proves
 * that every possible p75 value clamps to the same floor or ceiling, avoid
 * the second full-file radix pass. Unclamped buckets still resolve the exact
 * low bits, so the returned threshold is identical to
 * `scanAdaptiveSpeechThreshold(...).threshold`.
 */
export async function scanAdaptiveSpeechThresholdOnly(
  reader: PcmSampleReader,
  signal?: AbortSignal,
): Promise<number> {
  return completeAdaptiveSpeechThresholdOnly(
    reader,
    new AdaptiveSpeechRmsAccumulator(),
    signal,
  );
}

/**
 * Finish an incrementally accumulated high-radix pass and preserve the exact
 * low-16 second pass for the uncommon unclamped selected bucket.
 */
export async function completeAdaptiveSpeechThresholdOnly(
  reader: PcmSampleReader,
  accumulator: AdaptiveSpeechRmsAccumulator,
  signal?: AbortSignal,
): Promise<number> {
  const completeSampleCount =
    Math.floor(
      reader.sampleCount / PARAKEET_ENCODER_FRAME_SAMPLES,
    ) * PARAKEET_ENCODER_FRAME_SAMPLES;
  if (accumulator.nextFrameStartSample > completeSampleCount) {
    throw new RangeError(
      "Adaptive RMS accumulator extends past the complete audio frames",
    );
  }
  await forEachCompleteFrameRms(
    reader,
    signal,
    (rms, startSample) => {
      accumulator.addContiguousFrameRms(startSample, rms);
    },
    accumulator.nextFrameStartSample,
  );
  const bucket = accumulator.finishReferenceHighBucket();
  if (bucket.nonzeroFrameCount === 0) {
    return SPEECH_RMS_CEILING;
  }

  const clampedThreshold = clampedThresholdForHighBucket(
    bucket.highBits,
  );
  if (clampedThreshold !== undefined) {
    return clampedThreshold;
  }

  return adaptiveSpeechRmsThreshold(
    await selectReferenceRms(reader, bucket, signal),
  );
}

async function scanReferenceHighBucket(
  reader: PcmSampleReader,
  signal: AbortSignal | undefined,
): Promise<ReferenceHighBucket> {
  const accumulator = new AdaptiveSpeechRmsAccumulator();
  await forEachCompleteFrameRms(
    reader,
    signal,
    (rms, startSample) => {
      accumulator.addContiguousFrameRms(startSample, rms);
    },
  );
  return accumulator.finishReferenceHighBucket();
}

async function selectReferenceRms(
  reader: PcmSampleReader,
  bucket: ReferenceHighBucket,
  signal: AbortSignal | undefined,
): Promise<number> {
  const {
    histogram,
    highBits,
    indexWithinHighBucket,
  } = bucket;
  histogram.fill(0);
  await forEachCompleteFrameRms(reader, signal, (rms) => {
    if (rms === 0) return;
    const bits = float32Bits(rms);
    if (bits >>> 16 === highBits) {
      const bucket = bits & 0xffff;
      histogram[bucket] = histogram[bucket]! + 1;
    }
  });

  let cumulative = 0;
  let lowBits = 0;
  for (; lowBits < histogram.length; lowBits += 1) {
    cumulative += histogram[lowBits]!;
    if (cumulative > indexWithinHighBucket) break;
  }
  const referenceRms = float32FromBits(
    (highBits << 16) | lowBits,
  );
  return referenceRms;
}

function clampedThresholdForHighBucket(
  highBits: number,
): number | undefined {
  const minimumReferenceRms = float32FromBits(
    highBits << 16,
  );
  const minimumScaledRms = Math.fround(
    minimumReferenceRms * SPEECH_RMS_REFERENCE_SCALE,
  );
  if (minimumScaledRms >= SPEECH_RMS_CEILING) {
    return SPEECH_RMS_CEILING;
  }

  const maximumReferenceRms = float32FromBits(
    (highBits << 16) | 0xffff,
  );
  const maximumScaledRms = Math.fround(
    maximumReferenceRms * SPEECH_RMS_REFERENCE_SCALE,
  );
  if (maximumScaledRms <= SPEECH_RMS_FLOOR) {
    return SPEECH_RMS_FLOOR;
  }

  return undefined;
}

/** Cumulative duration of complete 80 ms frames above the adaptive gate. */
export async function speechLikeSeconds(
  reader: PcmSampleReader,
  startSample: number,
  endSample: number,
  threshold: number,
  signal?: AbortSignal,
): Promise<number> {
  assertSampleRange(startSample, endSample, reader.sampleCount);
  if (!Number.isFinite(threshold) || threshold < 0) {
    throw new RangeError("threshold must be non-negative and finite");
  }

  const scanSamples =
    SCAN_FRAMES_PER_READ * PARAKEET_ENCODER_FRAME_SAMPLES;
  const samples = new Float32Array(scanSamples);
  let speechFrames = 0;
  let readStart = startSample;
  while (
    readStart + PARAKEET_ENCODER_FRAME_SAMPLES <=
    endSample
  ) {
    throwIfAborted(signal);
    const remainingCompleteSamples =
      Math.floor(
        (endSample - readStart) /
          PARAKEET_ENCODER_FRAME_SAMPLES,
      ) * PARAKEET_ENCODER_FRAME_SAMPLES;
    const readSampleCount = Math.min(
      scanSamples,
      remainingCompleteSamples,
    );
    await readExactly(
      reader,
      readStart,
      samples,
      readSampleCount,
    );
    for (
      let frameStart = 0;
      frameStart < readSampleCount;
      frameStart += PARAKEET_ENCODER_FRAME_SAMPLES
    ) {
      if (
        rootMeanSquare(
          samples,
          PARAKEET_ENCODER_FRAME_SAMPLES,
          frameStart,
        ) > threshold
      ) {
        speechFrames += 1;
      }
    }
    readStart += readSampleCount;
  }
  return (
    speechFrames *
    (PARAKEET_ENCODER_FRAME_SAMPLES / PARAKEET_SAMPLE_RATE)
  );
}

function rootMeanSquare(
  samples: Float32Array,
  sampleCount: number,
  sampleOffset = 0,
): number {
  let sum = Math.fround(0);
  for (
    let index = sampleOffset;
    index < sampleOffset + sampleCount;
    index += 1
  ) {
    const value = samples[index]!;
    sum = Math.fround(
      sum + Math.fround(value * value),
    );
  }
  return Math.fround(
    Math.sqrt(Math.fround(sum / sampleCount)),
  );
}

async function forEachCompleteFrameRms(
  reader: PcmSampleReader,
  signal: AbortSignal | undefined,
  visit: (rms: number, startSample: number) => void,
  startSample = 0,
): Promise<void> {
  const scanSamples =
    SCAN_FRAMES_PER_READ * PARAKEET_ENCODER_FRAME_SAMPLES;
  const samples = new Float32Array(scanSamples);
  const completeSampleCount =
    Math.floor(reader.sampleCount / PARAKEET_ENCODER_FRAME_SAMPLES) *
    PARAKEET_ENCODER_FRAME_SAMPLES;
  if (
    !Number.isSafeInteger(startSample) ||
    startSample < 0 ||
    startSample > completeSampleCount ||
    startSample % PARAKEET_ENCODER_FRAME_SAMPLES !== 0
  ) {
    throw new RangeError(
      "RMS scan start must be a complete-frame sample boundary",
    );
  }

  for (
    let readStart = startSample;
    readStart < completeSampleCount;
    readStart += scanSamples
  ) {
    throwIfAborted(signal);
    const readSampleCount = Math.min(
      scanSamples,
      completeSampleCount - readStart,
    );
    await readExactly(
      reader,
      readStart,
      samples,
      readSampleCount,
    );
    for (
      let frameStart = 0;
      frameStart < readSampleCount;
      frameStart += PARAKEET_ENCODER_FRAME_SAMPLES
    ) {
      visit(
        rootMeanSquare(
          samples,
          PARAKEET_ENCODER_FRAME_SAMPLES,
          frameStart,
        ),
        readStart + frameStart,
      );
    }
  }
}

const FLOAT32_BITS = new ArrayBuffer(4);
const FLOAT32_VIEW = new Float32Array(FLOAT32_BITS);
const UINT32_VIEW = new Uint32Array(FLOAT32_BITS);

function float32Bits(value: number): number {
  FLOAT32_VIEW[0] = value;
  return UINT32_VIEW[0]!;
}

function float32FromBits(bits: number): number {
  UINT32_VIEW[0] = bits >>> 0;
  return FLOAT32_VIEW[0]!;
}

async function readExactly(
  reader: Pick<PcmSampleReader, "readSamplesInto">,
  startSample: number,
  target: Float32Array,
  sampleCount: number,
): Promise<void> {
  const written = await reader.readSamplesInto(
    startSample,
    target,
    0,
    sampleCount,
  );
  if (written !== sampleCount) {
    throw new RangeError(
      `PCM reader wrote ${written} samples for a ${sampleCount}-sample request`,
    );
  }
}

function assertSampleRange(
  startSample: number,
  endSample: number,
  totalSamples: number,
): void {
  if (
    !Number.isSafeInteger(startSample) ||
    !Number.isSafeInteger(endSample) ||
    startSample < 0 ||
    endSample < startSample ||
    endSample > totalSamples
  ) {
    throw new RangeError("invalid PCM sample range");
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  signal?.throwIfAborted();
}
