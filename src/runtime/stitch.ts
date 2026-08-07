import {
  PARAKEET_ENCODER_FRAME_SAMPLES,
  PARAKEET_OVERLAP_SAMPLES,
  type StatelessWindowPlan,
} from "./chunking";
import type { TdtEmittedToken } from "./tdt-types";

export interface TimedToken {
  readonly tokenId: number;
  readonly startSample: number;
  readonly endSample: number;
  /** Raw TDT duration, retained for conservative seam-gap boundaries. */
  readonly durationFrames: number;
}

export interface DecodedTokenWindow<Token extends TimedToken = TimedToken> {
  readonly index: number;
  /** Logical emission range, not an end-aligned window's physical read range. */
  readonly startSample: number;
  readonly endSample: number;
  readonly tokens: readonly Token[];
}

export type TokenPieceResolver = (
  tokenId: number,
) => string | undefined;

export type StitchMethod = "disjoint" | "contiguous" | "lcs" | "midpoint";

export interface StitchTrace {
  readonly leftWindowIndex: number;
  readonly rightWindowIndex: number;
  readonly method: StitchMethod;
  readonly overlapStartSample: number;
  readonly overlapEndSample: number;
}

export interface StitchedTokens<Token extends TimedToken = TimedToken> {
  readonly tokens: readonly Token[];
  readonly trace: readonly StitchTrace[];
}

export interface IncrementalStitchAccumulatorOptions {
  readonly tokenPiece?: TokenPieceResolver;
  /**
   * Retain one diagnostic record per appended seam. Production callers that
   * only need tokens can leave this disabled to avoid trace allocations.
   */
  readonly collectTrace?: boolean;
}

const EMPTY_STITCH_TRACE: readonly StitchTrace[] = Object.freeze([]);

interface IndexedToken<Token extends TimedToken> {
  readonly index: number;
  readonly token: Token;
}

interface MatchPair {
  readonly leftOffset: number;
  readonly rightOffset: number;
}

/**
 * Convert local compressed-timeline TDT frames into source-sample offsets.
 * Each encoder frame maps to 1,280 source samples. Hidden acoustic context and
 * a final window's real-audio backfill are suppressed by the window's logical
 * emission range.
 */
export function globalizeTdtTokens(
  tokens: readonly TdtEmittedToken[],
  window: StatelessWindowPlan,
): readonly TimedToken[] {
  const result: TimedToken[] = [];
  for (const token of tokens) {
    const startSample =
      window.timestampOffsetSample +
      token.frameIndex * PARAKEET_ENCODER_FRAME_SAMPLES;
    const endSample =
      startSample +
      token.durationFrames * PARAKEET_ENCODER_FRAME_SAMPLES;

    if (
      startSample < window.emissionStartSample ||
      startSample >= window.emissionEndSample
    ) {
      continue;
    }

    result.push({
      tokenId: token.tokenId,
      startSample,
      endSample: Math.min(endSample, window.emissionEndSample),
      durationFrames: token.durationFrames,
    });
  }
  return result;
}

/**
 * Merge independently decoded windows using the pinned FluidAudio behavior
 * as the operational reference. This remains a parakeet.wgsl
 * implementation: sample offsets are native to our engine and no native
 * worker or CoreML scheduling is imported.
 *
 * Reference: FluidAudio ChunkProcessor.swift at
 * 5390df9752c8fc583596018360c5fd70d6fa6c75 (Apache-2.0), including the
 * merge-order correction from c802b43ca9ececd0f4870dedaf6fb95a4d27d3eb.
 */
export function stitchTokenWindows<Token extends TimedToken>(
  windows: readonly DecodedTokenWindow<Token>[],
  tokenPiece?: TokenPieceResolver,
): StitchedTokens<Token> {
  const ordered = [...windows].sort(
    (left, right) =>
      left.startSample - right.startSample || left.index - right.index,
  );
  const accumulator = new IncrementalStitchAccumulator<Token>({
    ...(tokenPiece === undefined ? {} : { tokenPiece }),
    collectTrace: true,
  });
  for (const window of ordered) {
    accumulator.append(window);
  }
  return accumulator.snapshot();
}

/**
 * Exact streaming form of `stitchTokenWindows` for windows that are already
 * ordered by `(startSample, index)`.
 *
 * Every call to `append` performs only the newly introduced seam merge.
 * `snapshot` finalizes a copy, so timestamp clamping and duplicate collapse
 * never mutate the merge state needed by a later window.
 */
export class IncrementalStitchAccumulator<
  Token extends TimedToken = TimedToken,
> {
  readonly #resolveTokenPiece: TokenPieceResolver;
  readonly #hasTokenPieces: boolean;
  readonly #trace: StitchTrace[] | undefined;
  #merged: Token[] = [];
  #previousWindow: DecodedTokenWindow<Token> | undefined;

  constructor(options: IncrementalStitchAccumulatorOptions = {}) {
    this.#resolveTokenPiece = options.tokenPiece ?? (() => undefined);
    this.#hasTokenPieces = options.tokenPiece !== undefined;
    this.#trace = options.collectTrace === true ? [] : undefined;
  }

  append(window: DecodedTokenWindow<Token>): void {
    validateWindow(window);
    const previous = this.#previousWindow;
    if (previous === undefined) {
      this.#merged = [...window.tokens];
      this.#previousWindow = window;
      return;
    }
    if (compareWindowOrder(previous, window) > 0) {
      throw new RangeError(
        "incremental stitch windows must be appended in chronological order",
      );
    }

    const merge = mergeTokenStreams(
      this.#merged,
      window.tokens,
      this.#resolveTokenPiece,
      this.#hasTokenPieces,
    );
    this.#merged = merge.tokens;
    this.#trace?.push({
      leftWindowIndex: previous.index,
      rightWindowIndex: window.index,
      method: merge.method,
      overlapStartSample: window.startSample,
      overlapEndSample: Math.min(previous.endSample, window.endSample),
    });
    this.#previousWindow = window;
  }

  snapshot(): StitchedTokens<Token> {
    let tokens = enforceMonotonicTokenTimes(this.#merged);
    if (tokens.length > 1) {
      if (this.#hasTokenPieces) {
        tokens = collapseSeamWordDuplicates(
          tokens,
          this.#resolveTokenPiece,
        );
      }
    }
    return {
      tokens,
      trace:
        this.#trace === undefined
          ? EMPTY_STITCH_TRACE
          : [...this.#trace],
    };
  }
}

/**
 * Keep the merge-produced text order authoritative while making its timing
 * metadata non-decreasing. TDT can emit multiple pieces on one 80 ms frame,
 * and independently decoded overlap windows can place a later text piece on
 * a slightly earlier global frame. Sorting on that coarse timestamp corrupts
 * the splice order; clamp the later interval forward instead.
 *
 * Adjusted tokens are copied and their complete interval is shifted by the
 * same delta, preserving duration and any additional token fields. Neither
 * the input array nor its token objects are mutated.
 */
export function enforceMonotonicTokenTimes<Token extends TimedToken>(
  tokens: readonly Token[],
): Token[] {
  if (tokens.length === 0) return [];

  const result: Token[] = [tokens[0]!];
  let lastStartSample = tokens[0]!.startSample;
  for (let index = 1; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token.startSample >= lastStartSample) {
      result.push(token);
      lastStartSample = token.startSample;
      continue;
    }

    const shiftSamples = lastStartSample - token.startSample;
    result.push({
      ...token,
      startSample: lastStartSample,
      endSample: token.endSample + shiftSamples,
    });
  }
  return result;
}

function compareWindowOrder<Token extends TimedToken>(
  left: DecodedTokenWindow<Token>,
  right: DecodedTokenWindow<Token>,
): number {
  return left.startSample - right.startSample || left.index - right.index;
}

function mergeTokenStreams<Token extends TimedToken>(
  left: readonly Token[],
  right: readonly Token[],
  tokenPiece: TokenPieceResolver,
  hasTokenPieces: boolean,
): { readonly tokens: Token[]; readonly method: StitchMethod } {
  if (left.length === 0) {
    return { tokens: [...right], method: "disjoint" };
  }
  if (right.length === 0) {
    return { tokens: [...left], method: "disjoint" };
  }

  const leftEndSample =
    left.at(-1)!.startSample + PARAKEET_ENCODER_FRAME_SAMPLES;
  const rightStartSample = right[0]!.startSample;
  if (leftEndSample <= rightStartSample) {
    return { tokens: [...left, ...right], method: "disjoint" };
  }

  const overlapLeft = indexedTokens(
    left,
    (token) =>
      token.startSample + PARAKEET_ENCODER_FRAME_SAMPLES >
      rightStartSample - PARAKEET_OVERLAP_SAMPLES,
  );
  const overlapRight = indexedTokens(
    right,
    (token) =>
      token.startSample <
      leftEndSample + PARAKEET_OVERLAP_SAMPLES,
  );
  if (overlapLeft.length < 2 || overlapRight.length < 2) {
    return {
      tokens: mergeByMidpoint(
        left,
        right,
        leftEndSample,
        rightStartSample,
        tokenPiece,
        hasTokenPieces,
      ),
      method: "midpoint",
    };
  }

  const matches = (
    leftToken: IndexedToken<Token>,
    rightToken: IndexedToken<Token>,
  ): boolean =>
    tokenIdsMatch(
      leftToken.token.tokenId,
      rightToken.token.tokenId,
      tokenPiece,
    ) &&
    Math.abs(
      leftToken.token.startSample -
        rightToken.token.startSample,
    ) <
      PARAKEET_OVERLAP_SAMPLES / 2;

  const contiguous = longestContiguousMatches(
    overlapLeft,
    overlapRight,
    matches,
  );
  const minimumPairs = Math.max(
    Math.floor(overlapLeft.length / 2),
    1,
  );
  if (contiguous.length >= minimumPairs) {
    return {
      tokens: mergeUsingMatches(
        contiguous,
        overlapLeft,
        overlapRight,
        left,
        right,
        tokenPiece,
        hasTokenPieces,
      ),
      method: "contiguous",
    };
  }

  const lcs = longestCommonSubsequence(
    overlapLeft,
    overlapRight,
    matches,
  );
  if (lcs.length === 0) {
    return {
      tokens: mergeByMidpoint(
        left,
        right,
        leftEndSample,
        rightStartSample,
        tokenPiece,
        hasTokenPieces,
      ),
      method: "midpoint",
    };
  }
  return {
    tokens: mergeUsingMatches(
      lcs,
      overlapLeft,
      overlapRight,
      left,
      right,
      tokenPiece,
      hasTokenPieces,
    ),
    method: "lcs",
  };
}

function indexedTokens<Token extends TimedToken>(
  tokens: readonly Token[],
  include: (token: Token) => boolean,
): readonly IndexedToken<Token>[] {
  const result: IndexedToken<Token>[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (include(token)) result.push({ index, token });
  }
  return result;
}

function longestContiguousMatches<Token extends TimedToken>(
  left: readonly IndexedToken<Token>[],
  right: readonly IndexedToken<Token>[],
  matches: (
    left: IndexedToken<Token>,
    right: IndexedToken<Token>,
  ) => boolean,
): readonly MatchPair[] {
  let best: MatchPair[] = [];
  for (let leftOffset = 0; leftOffset < left.length; leftOffset += 1) {
    for (
      let rightOffset = 0;
      rightOffset < right.length;
      rightOffset += 1
    ) {
      if (!matches(left[leftOffset]!, right[rightOffset]!)) continue;
      const current: MatchPair[] = [];
      let leftCursor = leftOffset;
      let rightCursor = rightOffset;
      while (
        leftCursor < left.length &&
        rightCursor < right.length &&
        matches(left[leftCursor]!, right[rightCursor]!)
      ) {
        current.push({
          leftOffset: leftCursor,
          rightOffset: rightCursor,
        });
        leftCursor += 1;
        rightCursor += 1;
      }
      // FluidAudio deliberately keeps the first scan-order run on ties.
      if (current.length > best.length) best = current;
    }
  }
  return best;
}

function longestCommonSubsequence<Token extends TimedToken>(
  left: readonly IndexedToken<Token>[],
  right: readonly IndexedToken<Token>[],
  matches: (
    left: IndexedToken<Token>,
    right: IndexedToken<Token>,
  ) => boolean,
): readonly MatchPair[] {
  const width = right.length + 1;
  const table = new Uint32Array((left.length + 1) * width);
  for (let leftCount = 1; leftCount <= left.length; leftCount += 1) {
    for (
      let rightCount = 1;
      rightCount <= right.length;
      rightCount += 1
    ) {
      const cell = leftCount * width + rightCount;
      table[cell] = matches(
        left[leftCount - 1]!,
        right[rightCount - 1]!,
      )
        ? table[(leftCount - 1) * width + rightCount - 1]! + 1
        : Math.max(
            table[(leftCount - 1) * width + rightCount]!,
            table[leftCount * width + rightCount - 1]!,
          );
    }
  }

  const reversed: MatchPair[] = [];
  let leftCount = left.length;
  let rightCount = right.length;
  while (leftCount > 0 && rightCount > 0) {
    if (
      matches(
        left[leftCount - 1]!,
        right[rightCount - 1]!,
      )
    ) {
      reversed.push({
        leftOffset: leftCount - 1,
        rightOffset: rightCount - 1,
      });
      leftCount -= 1;
      rightCount -= 1;
    } else if (
      table[(leftCount - 1) * width + rightCount]! >
      table[leftCount * width + rightCount - 1]!
    ) {
      leftCount -= 1;
    } else {
      // Match FluidAudio's deterministic backtracking tie rule.
      rightCount -= 1;
    }
  }
  reversed.reverse();
  return reversed;
}

function mergeUsingMatches<Token extends TimedToken>(
  matches: readonly MatchPair[],
  overlapLeft: readonly IndexedToken<Token>[],
  overlapRight: readonly IndexedToken<Token>[],
  left: readonly Token[],
  right: readonly Token[],
  tokenPiece: TokenPieceResolver,
  hasTokenPieces: boolean,
): Token[] {
  const leftIndexes = matches.map(
    ({ leftOffset }) => overlapLeft[leftOffset]!.index,
  );
  const rightIndexes = matches.map(
    ({ rightOffset }) => overlapRight[rightOffset]!.index,
  );
  const result: Token[] = [];
  const firstLeft = leftIndexes[0]!;
  if (firstLeft > 0) result.push(...left.slice(0, firstLeft));

  for (let matchIndex = 0; matchIndex < matches.length; matchIndex += 1) {
    const leftIndex = leftIndexes[matchIndex]!;
    const rightIndex = rightIndexes[matchIndex]!;
    result.push(left[leftIndex]!);
    if (matchIndex === matches.length - 1) continue;

    const nextLeftIndex = leftIndexes[matchIndex + 1]!;
    const nextRightIndex = rightIndexes[matchIndex + 1]!;
    const leftGap = left.slice(leftIndex + 1, nextLeftIndex);
    const rightGap = right.slice(rightIndex + 1, nextRightIndex);
    result.push(...(rightGap.length > leftGap.length ? rightGap : leftGap));
  }

  const lastRight = rightIndexes.at(-1)!;
  if (lastRight + 1 >= right.length) return result;
  const tail = right.slice(lastRight + 1);
  if (!hasTokenPieces) {
    result.push(...tail);
    return result;
  }
  if (isSpliceSafeToken(tail[0]!.tokenId, tokenPiece)) {
    result.push(...tail);
    return result;
  }

  const rightWordStart = wordInitialIndex(
    right,
    lastRight,
    tokenPiece,
  );
  if (
    rightWordStart !== undefined &&
    popSeamWord(result, tokenPiece)
  ) {
    result.push(...right.slice(rightWordStart));
    return result;
  }

  const lastLeft = leftIndexes.at(-1)!;
  let leftCursor = lastLeft + 1;
  while (
    leftCursor < left.length &&
    !isSpliceSafeToken(left[leftCursor]!.tokenId, tokenPiece)
  ) {
    result.push(left[leftCursor]!);
    leftCursor += 1;
  }
  const resume = tail.findIndex((token) =>
    isSpliceSafeToken(token.tokenId, tokenPiece),
  );
  if (resume >= 0) {
    result.push(...tail.slice(resume));
  } else {
    // A possible glued word is preferable to deleting the right tail.
    result.push(...tail);
  }
  return result;
}

function wordInitialIndex<Token extends TimedToken>(
  stream: readonly Token[],
  endingAt: number,
  tokenPiece: TokenPieceResolver,
): number | undefined {
  for (let index = endingAt; index >= 0; index -= 1) {
    if (isSpliceSafeToken(stream[index]!.tokenId, tokenPiece)) {
      return index;
    }
  }
  return undefined;
}

function popSeamWord<Token extends TimedToken>(
  result: Token[],
  tokenPiece: TokenPieceResolver,
): boolean {
  for (let index = result.length - 1; index >= 0; index -= 1) {
    if (isSpliceSafeToken(result[index]!.tokenId, tokenPiece)) {
      result.splice(index);
      return true;
    }
  }
  return false;
}

function mergeByMidpoint<Token extends TimedToken>(
  left: readonly Token[],
  right: readonly Token[],
  leftEndSample: number,
  rightStartSample: number,
  tokenPiece: TokenPieceResolver,
  hasTokenPieces: boolean,
): Token[] {
  const cutoff = (leftEndSample + rightStartSample) / 2;
  let leftEnd = left.findIndex(
    (token) => token.startSample >= cutoff,
  );
  if (leftEnd < 0) leftEnd = left.length;
  let rightStart = right.findIndex(
    (token) => token.startSample >= cutoff,
  );
  if (rightStart < 0) rightStart = right.length;

  if (hasTokenPieces && leftEnd > 0) {
    while (
      leftEnd < left.length &&
      !isSpliceSafeToken(left[leftEnd]!.tokenId, tokenPiece)
    ) {
      leftEnd += 1;
    }
  }
  if (hasTokenPieces) {
    let safeRightStart = rightStart;
    while (
      safeRightStart < right.length &&
      !isSpliceSafeToken(
        right[safeRightStart]!.tokenId,
        tokenPiece,
      )
    ) {
      safeRightStart += 1;
    }
    // Retain the original cutoff if no safe right token exists.
    if (safeRightStart < right.length) rightStart = safeRightStart;
  }
  return [...left.slice(0, leftEnd), ...right.slice(rightStart)];
}

export function isSpliceSafeToken(
  tokenId: number,
  tokenPiece: TokenPieceResolver,
): boolean {
  const piece = tokenPiece(tokenId);
  if (piece === undefined || piece.length === 0) return false;
  return (
    startsWord(piece) ||
    /^[\p{P}\p{S}]+$/u.test(piece)
  );
}

export function isPunctuationOnlyToken(
  tokenId: number,
  tokenPiece: TokenPieceResolver,
): boolean {
  const piece = tokenPiece(tokenId);
  return (
    piece !== undefined &&
    piece.length > 0 &&
    /^[\p{P}\p{S}]+$/u.test(piece)
  );
}

export function tokenIdsMatch(
  leftTokenId: number,
  rightTokenId: number,
  tokenPiece: TokenPieceResolver,
): boolean {
  if (leftTokenId === rightTokenId) return true;
  const leftPiece = tokenPiece(leftTokenId);
  const rightPiece = tokenPiece(rightTokenId);
  return (
    leftPiece !== undefined &&
    rightPiece !== undefined &&
    leftPiece.toLowerCase() === rightPiece.toLowerCase()
  );
}

export function collapseSeamWordDuplicates<Token extends TimedToken>(
  tokens: readonly Token[],
  tokenPiece: TokenPieceResolver,
): Token[] {
  if (tokens.length < 2) return [...tokens];
  interface Word {
    readonly tokens: Token[];
    core: string;
    readonly startSample: number;
    endsSentence: boolean;
  }

  const words: Word[] = [];
  for (const token of tokens) {
    const piece = tokenPiece(token.tokenId) ?? "";
    if (words.length === 0 || startsWord(piece)) {
      words.push({
        tokens: [token],
        core: "",
        startSample: token.startSample,
        endsSentence: false,
      });
    } else {
      words.at(-1)!.tokens.push(token);
    }
  }

  for (const word of words) {
    const text = word.tokens
      .map((token) =>
        stripWordBoundaryPrefix(tokenPiece(token.tokenId) ?? ""),
      )
      .join("");
    word.core = text.replace(/^[\p{P}\s]+|[\p{P}\s]+$/gu, "");
    word.endsSentence = /[.?!:]$/u.test(text);
  }

  const keep = new Array<boolean>(words.length).fill(true);
  let lastKept = -1;
  for (let index = 0; index < words.length; index += 1) {
    if (lastKept < 0) {
      lastKept = index;
      continue;
    }
    const previous = words[lastKept]!;
    const current = words[index]!;
    const duplicate =
      previous.core.length > 0 &&
      current.core.length > 0 &&
      previous.core !== current.core &&
      previous.core.toLowerCase() === current.core.toLowerCase() &&
      /^\p{L}/u.test(current.core) &&
      !previous.endsSentence &&
      current.startSample - previous.startSample <=
        PARAKEET_OVERLAP_SAMPLES;
    if (!duplicate) {
      lastKept = index;
      continue;
    }
    if (
      current.core === current.core.toLowerCase() &&
      previous.core !== previous.core.toLowerCase()
    ) {
      keep[lastKept] = false;
      lastKept = index;
    } else {
      keep[index] = false;
    }
  }

  const result: Token[] = [];
  for (let index = 0; index < words.length; index += 1) {
    if (keep[index]) result.push(...words[index]!.tokens);
  }
  return result;
}

function startsWord(piece: string): boolean {
  return piece.startsWith("▁") || piece.startsWith(" ");
}

function stripWordBoundaryPrefix(piece: string): string {
  return startsWord(piece) ? piece.slice(1) : piece;
}

function validateWindow<Token extends TimedToken>(
  window: DecodedTokenWindow<Token>,
): void {
  assertNonNegativeInteger(window.index, "window index");
  assertNonNegativeInteger(window.startSample, "window startSample");
  assertNonNegativeInteger(window.endSample, "window endSample");
  if (window.endSample < window.startSample) {
    throw new RangeError("window endSample must not precede startSample");
  }

  let previousStart = Number.NEGATIVE_INFINITY;
  for (const token of window.tokens) {
    assertNonNegativeInteger(token.tokenId, "tokenId");
    assertNonNegativeInteger(token.startSample, "token startSample");
    assertNonNegativeInteger(token.endSample, "token endSample");
    assertNonNegativeInteger(
      token.durationFrames,
      "token durationFrames",
    );
    if (token.endSample < token.startSample) {
      throw new RangeError("token endSample must not precede startSample");
    }
    if (token.startSample < previousStart) {
      throw new RangeError("tokens must be sorted by startSample");
    }
    previousStart = token.startSample;
  }
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}
