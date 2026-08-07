import { PARAKEET_SAMPLE_RATE } from "../audio/wav-stream";
import {
  PARAKEET_ENCODER_FRAME_SAMPLES,
  PARAKEET_REPAIR_VISIBLE_SAMPLES,
  PARAKEET_WINDOW_SAMPLES,
  type StatelessWindowPlan,
} from "./chunking";
import {
  isPunctuationOnlyToken,
  isSpliceSafeToken,
  tokenIdsMatch,
  type TimedToken,
  type TokenPieceResolver,
} from "./stitch";

// Fifteen encoder frames (1.20 s) admit speech-filled gaps that sit below
// FluidAudio's 1.5 s default without probing shorter pauses. The independent
// 0.5 s speech-energy gate and fixed probe/round budgets bound repair work.
const SEAM_GAP_MIN_SECONDS = 1.2;
export const SEAM_GAP_MIN_FRAMES = Math.max(
  2,
  Math.floor(
    SEAM_GAP_MIN_SECONDS /
      (PARAKEET_ENCODER_FRAME_SAMPLES / PARAKEET_SAMPLE_RATE),
  ),
);
export const SEAM_GAP_MAX_PROBES = 32;
export const SEAM_GAP_MAX_ROUNDS = 3;
export const SEAM_GAP_EDGE_TOLERANCE_FRAMES = 6;

if (SEAM_GAP_MIN_FRAMES !== 15) {
  throw new Error("Invalid fixed seam-gap repair constants");
}

export type SeamRepairPlacement = "gap-start" | "centered";

export interface SeamGap<Token extends TimedToken = TimedToken> {
  readonly adjacentIndex: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startSample: number;
  readonly endSample: number;
  readonly leadNeighbor: Token;
  readonly tailNeighbor: Token;
}

/**
 * Identify a cold repair decode independently of its diagnostic window index.
 * The cache is deliberately stricter than the current engine requirements:
 * every other plan field participates so future scheduler changes cannot
 * accidentally reuse a decode across different timeline semantics.
 */
export function seamRepairPlanCacheKey(
  plan: StatelessWindowPlan,
): string {
  return JSON.stringify([
    plan.windowSampleCount,
    plan.logicalStartSample,
    plan.logicalEndSample,
    plan.sourceStartSample,
    plan.sourceEndSample,
    plan.readSampleCount,
    plan.timestampOffsetSample,
    plan.emissionStartSample,
    plan.emissionEndSample,
    plan.decodeStartFrame,
    plan.decodeEndFrame,
    plan.flushFinal,
    plan.validSampleCount,
  ]);
}

/**
 * Resolve one adjacent inter-token gap on the same global 80 ms frame
 * timeline used by the TDT decoder. Energy gating and probe memoization are
 * intentionally left to the caller so an arbitrarily long file stays
 * streamed and bounded-memory.
 */
export function seamGapAt<Token extends TimedToken>(
  tokens: readonly Token[],
  adjacentIndex: number,
  totalSamples: number,
  tokenPiece: TokenPieceResolver,
): SeamGap<Token> | undefined {
  assertNonNegativeInteger(totalSamples, "totalSamples");
  if (
    !Number.isSafeInteger(adjacentIndex) ||
    adjacentIndex < 0 ||
    adjacentIndex + 1 >= tokens.length
  ) {
    throw new RangeError("adjacentIndex must identify a token pair");
  }
  const current = tokens[adjacentIndex]!;
  const next = tokens[adjacentIndex + 1]!;
  const currentFrame = sampleToFrame(current.startSample);
  const nextFrame = sampleToFrame(next.startSample);
  const startFrame =
    currentFrame + Math.max(1, current.durationFrames);
  const endFrame = nextFrame;
  if (endFrame - startFrame < SEAM_GAP_MIN_FRAMES) return undefined;

  const startSample =
    startFrame * PARAKEET_ENCODER_FRAME_SAMPLES;
  const endSample = Math.min(
    endFrame * PARAKEET_ENCODER_FRAME_SAMPLES,
    totalSamples,
  );
  if (endSample <= startSample) return undefined;

  return {
    adjacentIndex,
    startFrame,
    endFrame,
    startSample,
    endSample,
    leadNeighbor: wordNeighbor(
      tokens,
      adjacentIndex,
      -1,
      tokenPiece,
    ),
    tailNeighbor: wordNeighbor(
      tokens,
      adjacentIndex + 1,
      1,
      tokenPiece,
    ),
  };
}

/**
 * Plan FluidAudio's standalone, context-free repair probe while retaining the
 * existing fixed 240,000-sample Wasm/GPU input. Only 239,360 real samples are
 * exposed to the decoder; the remainder is right padding.
 */
export function planSeamRepairWindow(
  totalSamples: number,
  gap: Pick<SeamGap, "startSample" | "endSample">,
  placement: SeamRepairPlacement,
  index: number,
): StatelessWindowPlan {
  assertNonNegativeInteger(totalSamples, "totalSamples");
  assertNonNegativeInteger(index, "index");
  if (
    !Number.isSafeInteger(gap.startSample) ||
    !Number.isSafeInteger(gap.endSample) ||
    gap.startSample < 0 ||
    gap.endSample <= gap.startSample ||
    gap.endSample > totalSamples
  ) {
    throw new RangeError("invalid seam-gap sample range");
  }

  const requestedStart =
    placement === "gap-start"
      ? gap.startSample
      : Math.floor((gap.startSample + gap.endSample) / 2) -
        PARAKEET_REPAIR_VISIBLE_SAMPLES / 2;
  let sourceStartSample = Math.max(
    0,
    Math.min(
      requestedStart,
      totalSamples - PARAKEET_REPAIR_VISIBLE_SAMPLES,
    ),
  );
  sourceStartSample =
    Math.floor(
      sourceStartSample / PARAKEET_ENCODER_FRAME_SAMPLES,
    ) * PARAKEET_ENCODER_FRAME_SAMPLES;
  const sourceEndSample = Math.min(
    sourceStartSample + PARAKEET_REPAIR_VISIBLE_SAMPLES,
    totalSamples,
  );
  const readSampleCount = sourceEndSample - sourceStartSample;
  if (readSampleCount <= 0) {
    throw new RangeError("seam-repair window has no source audio");
  }

  return {
    index,
    windowSampleCount: PARAKEET_WINDOW_SAMPLES,
    logicalStartSample: sourceStartSample,
    logicalEndSample: sourceEndSample,
    sourceStartSample,
    sourceEndSample,
    readSampleCount,
    timestampOffsetSample: sourceStartSample,
    emissionStartSample: sourceStartSample,
    emissionEndSample: sourceEndSample,
    decodeStartFrame: 0,
    decodeEndFrame: Math.ceil(
      readSampleCount / PARAKEET_ENCODER_FRAME_SAMPLES,
    ),
    flushFinal: sourceEndSample >= totalSamples,
    validSampleCount: readSampleCount,
  };
}

/**
 * Filter a standalone probe down to an insertion-only, word-safe run strictly
 * inside its original gap. Existing merged tokens are never rewritten.
 */
export function spliceRepairCandidate<Token extends TimedToken>(
  windowTokens: readonly Token[],
  gap: Pick<
    SeamGap<Token>,
    | "startFrame"
    | "endFrame"
    | "leadNeighbor"
    | "tailNeighbor"
  >,
  tokenPiece: TokenPieceResolver,
): Token[] {
  let start = 0;
  let end = windowTokens.length;
  while (start < end) {
    const frame = sampleToFrame(windowTokens[start]!.startSample);
    if (frame > gap.startFrame && frame < gap.endFrame - 1) break;
    start += 1;
  }

  const candidate: Token[] = [];
  for (let index = start; index < end; index += 1) {
    const token = windowTokens[index]!;
    const frame = sampleToFrame(token.startSample);
    if (frame > gap.startFrame && frame < gap.endFrame - 1) {
      candidate.push(token);
    }
  }

  while (
    candidate.length > 0 &&
    !isSpliceSafeToken(candidate[0]!.tokenId, tokenPiece)
  ) {
    candidate.shift();
  }
  while (
    candidate.length > 0 &&
    tokenIdsMatch(
      candidate[0]!.tokenId,
      gap.leadNeighbor.tokenId,
      tokenPiece,
    ) &&
    Math.abs(
      sampleToFrame(candidate[0]!.startSample) -
        sampleToFrame(gap.leadNeighbor.startSample),
    ) <= SEAM_GAP_EDGE_TOLERANCE_FRAMES
  ) {
    candidate.shift();
  }
  while (
    candidate.length > 0 &&
    tokenIdsMatch(
      candidate.at(-1)!.tokenId,
      gap.tailNeighbor.tokenId,
      tokenPiece,
    ) &&
    Math.abs(
      sampleToFrame(gap.tailNeighbor.startSample) -
        sampleToFrame(candidate.at(-1)!.startSample),
    ) <= SEAM_GAP_EDGE_TOLERANCE_FRAMES
  ) {
    candidate.pop();
  }
  while (
    candidate.length > 0 &&
    (
      !isSpliceSafeToken(candidate[0]!.tokenId, tokenPiece) ||
      isPunctuationOnlyToken(candidate[0]!.tokenId, tokenPiece)
    )
  ) {
    candidate.shift();
  }
  return candidate;
}

function wordNeighbor<Token extends TimedToken>(
  tokens: readonly Token[],
  initialIndex: number,
  step: -1 | 1,
  tokenPiece: TokenPieceResolver,
): Token {
  let index = initialIndex;
  while (
    index + step >= 0 &&
    index + step < tokens.length &&
    isPunctuationOnlyToken(tokens[index]!.tokenId, tokenPiece)
  ) {
    index += step;
  }
  return tokens[index]!;
}

function sampleToFrame(sample: number): number {
  assertNonNegativeInteger(sample, "token startSample");
  if (sample % PARAKEET_ENCODER_FRAME_SAMPLES !== 0) {
    throw new RangeError("token timestamps must be encoder-frame aligned");
  }
  return sample / PARAKEET_ENCODER_FRAME_SAMPLES;
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}
