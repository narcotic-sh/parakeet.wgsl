import { PARAKEET_SAMPLE_RATE } from "../audio/wav-stream";
import { PARAKEET_FBANK_HOP_SAMPLES } from "../audio/fbank-wasm";

export { PARAKEET_FBANK_HOP_SAMPLES };

export const PARAKEET_WINDOW_SECONDS = 15;
export const PARAKEET_OVERLAP_SECONDS = 2;
export const PARAKEET_WINDOW_SAMPLES =
  PARAKEET_WINDOW_SECONDS * PARAKEET_SAMPLE_RATE;
export const PARAKEET_ENCODER_FRAME_SAMPLES =
  8 * PARAKEET_FBANK_HOP_SAMPLES;
export const PARAKEET_PRIMARY_CONTEXT_SAMPLES =
  PARAKEET_ENCODER_FRAME_SAMPLES;
export const PARAKEET_PRIMARY_VISIBLE_SAMPLES =
  Math.floor(
    (
      PARAKEET_WINDOW_SAMPLES -
      PARAKEET_PRIMARY_CONTEXT_SAMPLES -
      PARAKEET_FBANK_HOP_SAMPLES
    ) /
      PARAKEET_ENCODER_FRAME_SAMPLES,
  ) * PARAKEET_ENCODER_FRAME_SAMPLES;
export const PARAKEET_OVERLAP_SAMPLES =
  PARAKEET_OVERLAP_SECONDS * PARAKEET_SAMPLE_RATE;
export const PARAKEET_WINDOW_STRIDE_SAMPLES =
  PARAKEET_PRIMARY_VISIBLE_SAMPLES - PARAKEET_OVERLAP_SAMPLES;
export const PARAKEET_REPAIR_VISIBLE_SAMPLES =
  Math.floor(
    (PARAKEET_WINDOW_SAMPLES - PARAKEET_FBANK_HOP_SAMPLES) /
      PARAKEET_ENCODER_FRAME_SAMPLES,
  ) * PARAKEET_ENCODER_FRAME_SAMPLES;

if (
  PARAKEET_WINDOW_SAMPLES / PARAKEET_FBANK_HOP_SAMPLES !==
    1_500 ||
  PARAKEET_PRIMARY_CONTEXT_SAMPLES !== 1_280 ||
  PARAKEET_PRIMARY_VISIBLE_SAMPLES !== 238_080 ||
  PARAKEET_WINDOW_STRIDE_SAMPLES !== 206_080 ||
  PARAKEET_REPAIR_VISIBLE_SAMPLES !== 239_360
) {
  throw new Error("Invalid fixed Parakeet timeline constants");
}

export interface StatelessWindowPlan {
  readonly index: number;
  /** Fixed PCM capacity decoded and passed to the Wasm frontend. */
  readonly windowSampleCount: number;
  /** Start of this window on the regular, stateless chunk grid. */
  readonly logicalStartSample: number;
  /** End of audio covered by this logical window, exclusive. */
  readonly logicalEndSample: number;

  /** Blob-backed PCM range actually read, before right padding. */
  readonly sourceStartSample: number;
  readonly sourceEndSample: number;
  readonly readSampleCount: number;

  /**
   * Global timestamp corresponding to local model sample zero.
   * Encoder-frame timestamps should be offset by this value.
   */
  readonly timestampOffsetSample: number;

  /**
   * Earliest global timestamp whose decoded tokens may be emitted.
   * It differs from `sourceStartSample` for an end-aligned final window.
   */
  readonly emissionStartSample: number;
  readonly emissionEndSample: number;

  /**
   * First local encoder frame consumed by the cold TDT decoder. Ordinary
   * non-first windows skip the one hidden acoustic-context frame; an
   * end-aligned final window skips its larger real-audio backfill.
   */
  readonly decodeStartFrame: number;
  /** Exclusive local encoder-frame boundary for TDT decoding. */
  readonly decodeEndFrame: number;
  /** Run the shared bounded FluidAudio end-of-stream decoder flush. */
  readonly flushFinal: boolean;

  readonly validSampleCount: number;
}

export interface StatelessAudioWindow extends StatelessWindowPlan {
  /**
   * Fixed-size source PCM input. Samples after `validSampleCount` are padding.
   */
  readonly samples: Float32Array;
}

/**
 * Plans primary windows whose geometry can no longer change when more decoded
 * PCM arrives. The strict lookahead sample distinguishes a non-final window
 * from an exact-boundary EOF window, which must use final flush semantics.
 */
export class OnlineStatelessWindowPlanner {
  private readonly stablePlans: StatelessWindowPlan[] = [];

  get plannedWindowCount(): number {
    return this.stablePlans.length;
  }

  takeStablePrefix(availableSamples: number): readonly StatelessWindowPlan[] {
    assertNonNegativeInteger(availableSamples, "availableSamples");
    const added: StatelessWindowPlan[] = [];
    while (true) {
      const index = this.stablePlans.length;
      const logicalStartSample = index * PARAKEET_WINDOW_STRIDE_SAMPLES;
      const logicalEndSample =
        logicalStartSample + PARAKEET_PRIMARY_VISIBLE_SAMPLES;
      if (availableSamples <= logicalEndSample) break;

      const decoderContextSamples =
        index === 0 ? 0 : PARAKEET_PRIMARY_CONTEXT_SAMPLES;
      const sourceStartSample = logicalStartSample - decoderContextSamples;
      const readSampleCount =
        logicalEndSample - sourceStartSample;
      const plan: StatelessWindowPlan = {
        index,
        windowSampleCount: PARAKEET_WINDOW_SAMPLES,
        logicalStartSample,
        logicalEndSample,
        sourceStartSample,
        sourceEndSample: logicalEndSample,
        readSampleCount,
        timestampOffsetSample: logicalStartSample,
        emissionStartSample: logicalStartSample,
        emissionEndSample: logicalEndSample,
        decodeStartFrame:
          decoderContextSamples / PARAKEET_ENCODER_FRAME_SAMPLES,
        decodeEndFrame: Math.ceil(
          (readSampleCount - decoderContextSamples) /
            PARAKEET_ENCODER_FRAME_SAMPLES,
        ),
        flushFinal: false,
        validSampleCount: readSampleCount,
      };
      this.stablePlans.push(plan);
      added.push(plan);
    }
    return added;
  }

  /**
   * Resolve EOF with the canonical planner and prove every already-executed
   * window is an exact field-for-field prefix of the final plan.
   */
  finish(
    totalSamples: number,
    speechEndSample: number,
  ): {
    readonly allPlans: readonly StatelessWindowPlan[];
    readonly remainingPlans: readonly StatelessWindowPlan[];
  } {
    const allPlans = planStatelessWindows(totalSamples, speechEndSample);
    if (this.stablePlans.length > allPlans.length) {
      throw new Error("Online primary plan extends past the final plan");
    }
    for (let index = 0; index < this.stablePlans.length; index += 1) {
      if (!sameStatelessWindowPlan(this.stablePlans[index]!, allPlans[index]!)) {
        throw new Error(
          `Online primary window ${index} changed when EOF was resolved`,
        );
      }
    }
    return {
      allPlans,
      remainingPlans: allPlans.slice(this.stablePlans.length),
    };
  }
}

export function planStatelessWindows(
  totalSamples: number,
  speechEndSample = totalSamples,
): readonly StatelessWindowPlan[] {
  assertNonNegativeInteger(totalSamples, "totalSamples");
  assertNonNegativeInteger(speechEndSample, "speechEndSample");
  if (speechEndSample > totalSamples) {
    throw new RangeError("speechEndSample must not exceed totalSamples");
  }
  if (totalSamples === 0) return [];

  const plans: StatelessWindowPlan[] = [];

  for (
    let logicalStartSample = 0, index = 0;
    logicalStartSample < totalSamples;
    logicalStartSample += PARAKEET_WINDOW_STRIDE_SAMPLES, index += 1
  ) {
    const remainingSpeechSamples =
      Math.min(speechEndSample, totalSamples) - logicalStartSample;
    const reachesEofWithDefaultWindow =
      logicalStartSample + PARAKEET_PRIMARY_VISIBLE_SAMPLES >=
      totalSamples;
    let backfillSamples = 0;
    if (
      reachesEofWithDefaultWindow &&
      remainingSpeechSamples > 0 &&
      logicalStartSample > 0
    ) {
      const requestedBackfillSamples =
        Math.floor(
          (
            PARAKEET_PRIMARY_VISIBLE_SAMPLES -
            remainingSpeechSamples
          ) /
            PARAKEET_ENCODER_FRAME_SAMPLES,
        ) * PARAKEET_ENCODER_FRAME_SAMPLES;
      if (requestedBackfillSamples > 0) {
        const availableBackfillSamples =
          Math.floor(
            logicalStartSample / PARAKEET_ENCODER_FRAME_SAMPLES,
          ) * PARAKEET_ENCODER_FRAME_SAMPLES;
        backfillSamples = Math.min(
          availableBackfillSamples,
          requestedBackfillSamples,
        );
      }
    }

    // FluidAudio recomputes its last-chunk decision after choosing the
    // suppressed warmup prefix. With a long trailing-silence run this can
    // intentionally leave one final regular-grid iteration to plan.
    const visibleSamples = Math.max(
      PARAKEET_ENCODER_FRAME_SAMPLES,
      PARAKEET_PRIMARY_VISIBLE_SAMPLES - backfillSamples,
    );
    const candidateEndSample =
      logicalStartSample + visibleSamples;
    const isLast = candidateEndSample >= totalSamples;
    const logicalEndSample = isLast
      ? totalSamples
      : candidateEndSample;
    if (logicalEndSample <= logicalStartSample) break;

    const sourceEndSample = isLast
      ? Math.min(logicalEndSample, speechEndSample)
      : logicalEndSample;
    if (sourceEndSample <= logicalStartSample) break;

    const decoderContextSamples =
      backfillSamples > 0 || index === 0
        ? 0
        : PARAKEET_PRIMARY_CONTEXT_SAMPLES;
    const sourceStartSample =
      logicalStartSample -
      Math.max(backfillSamples, decoderContextSamples);
    const flushFinal = isLast;

    const readSampleCount = sourceEndSample - sourceStartSample;
    if (
      readSampleCount <= 0 ||
      readSampleCount > PARAKEET_WINDOW_SAMPLES
    ) {
      throw new Error("Invalid stateless source-window geometry");
    }
    const validSampleCount = readSampleCount;
    const decodeStartSample = decoderContextSamples;
    if (
      decodeStartSample % PARAKEET_ENCODER_FRAME_SAMPLES !== 0
    ) {
      throw new Error("Stateless decoder start must be frame-aligned");
    }

    const plan: StatelessWindowPlan = {
      index,
      windowSampleCount: PARAKEET_WINDOW_SAMPLES,
      logicalStartSample,
      logicalEndSample,
      sourceStartSample,
      sourceEndSample,
      readSampleCount,
      // FluidAudio timestamps ordinary context windows from the logical
      // chunk start even though decoder frame one is the first evaluated
      // frame. A decoded final backfill instead timestamps from its physical
      // start so its suppressed prefix and visible tail share one timeline.
      timestampOffsetSample:
        decoderContextSamples > 0
          ? logicalStartSample
          : sourceStartSample,
      emissionStartSample: logicalStartSample,
      emissionEndSample: sourceEndSample,
      decodeStartFrame:
        decodeStartSample / PARAKEET_ENCODER_FRAME_SAMPLES,
      decodeEndFrame: Math.ceil(
        (readSampleCount - decoderContextSamples) /
          PARAKEET_ENCODER_FRAME_SAMPLES,
      ),
      flushFinal,
      validSampleCount,
    };
    plans.push(plan);
    if (isLast) break;
  }

  return plans;
}

export function sameStatelessWindowPlan(
  left: StatelessWindowPlan,
  right: StatelessWindowPlan,
): boolean {
  return (
    left.index === right.index &&
    left.windowSampleCount === right.windowSampleCount &&
    left.logicalStartSample === right.logicalStartSample &&
    left.logicalEndSample === right.logicalEndSample &&
    left.sourceStartSample === right.sourceStartSample &&
    left.sourceEndSample === right.sourceEndSample &&
    left.readSampleCount === right.readSampleCount &&
    left.timestampOffsetSample === right.timestampOffsetSample &&
    left.emissionStartSample === right.emissionStartSample &&
    left.emissionEndSample === right.emissionEndSample &&
    left.decodeStartFrame === right.decodeStartFrame &&
    left.decodeEndFrame === right.decodeEndFrame &&
    left.flushFinal === right.flushFinal &&
    left.validSampleCount === right.validSampleCount
  );
}
function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}
