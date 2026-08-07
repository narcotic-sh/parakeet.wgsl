import { describe, expect, it } from "vitest";

import {
  SEAM_GAP_EDGE_TOLERANCE_FRAMES,
  SEAM_GAP_MAX_PROBES,
  SEAM_GAP_MAX_ROUNDS,
  SEAM_GAP_MIN_FRAMES,
  planSeamRepairWindow,
  seamGapAt,
  seamRepairPlanCacheKey,
  spliceRepairCandidate,
} from "../../src/runtime/seam-gap-repair";
import type { StatelessWindowPlan } from "../../src/runtime/chunking";
import type {
  TimedToken,
  TokenPieceResolver,
} from "../../src/runtime/stitch";

const FRAME_SAMPLES = 1_280;

function token(
  tokenId: number,
  frame: number,
  durationFrames = 1,
): TimedToken {
  return {
    tokenId,
    startSample: frame * FRAME_SAMPLES,
    endSample: (frame + durationFrames) * FRAME_SAMPLES,
    durationFrames,
  };
}

function resolver(
  vocabulary: Readonly<Record<number, string>>,
): TokenPieceResolver {
  return (tokenId) => vocabulary[tokenId];
}

describe("seam-gap detection and planning", () => {
  const pieces = resolver({ 1: "▁left", 2: "▁right" });

  it("retains the fixed probe and round budgets", () => {
    expect(SEAM_GAP_MAX_PROBES).toBe(32);
    expect(SEAM_GAP_MAX_ROUNDS).toBe(3);
  });

  it("uses decoded duration and the exact 15-frame default threshold", () => {
    expect(SEAM_GAP_MIN_FRAMES).toBe(15);
    expect(
      seamGapAt(
        [token(1, 10, 2), token(2, 27)],
        0,
        100 * FRAME_SAMPLES,
        pieces,
      ),
    ).toMatchObject({
      startFrame: 12,
      endFrame: 27,
      startSample: 12 * FRAME_SAMPLES,
      endSample: 27 * FRAME_SAMPLES,
    });
    expect(
      seamGapAt(
        [token(1, 10, 2), token(2, 26)],
        0,
        100 * FRAME_SAMPLES,
        pieces,
      ),
    ).toBeUndefined();
  });

  it("walks past punctuation to choose the bordering words", () => {
    const punctuationPieces = resolver({
      1: "▁left",
      2: ".",
      3: ",",
      4: "▁right",
    });
    const gap = seamGapAt(
      [token(1, 8), token(2, 10), token(3, 30), token(4, 31)],
      1,
      100 * FRAME_SAMPLES,
      punctuationPieces,
    );
    expect(gap?.leadNeighbor.tokenId).toBe(1);
    expect(gap?.tailNeighbor.tokenId).toBe(4);
  });

  it("plans frame-aligned, context-free gap-start and centered probes", () => {
    const gap = {
      startSample: 300_000,
      endSample: 500_000,
    };
    expect(
      planSeamRepairWindow(1_000_000, gap, "gap-start", 7),
    ).toMatchObject({
      index: 7,
      windowSampleCount: 240_000,
      sourceStartSample: 299_520,
      sourceEndSample: 538_880,
      readSampleCount: 239_360,
      decodeStartFrame: 0,
      decodeEndFrame: 187,
    });
    expect(
      planSeamRepairWindow(1_000_000, gap, "centered", 8),
    ).toMatchObject({
      index: 8,
      sourceStartSample: 280_320,
      sourceEndSample: 519_680,
      decodeStartFrame: 0,
      decodeEndFrame: 187,
    });
  });

  it("right-pads a short repair probe without exceeding the fixed arena", () => {
    expect(
      planSeamRepairWindow(
        100_000,
        { startSample: 20_000, endSample: 80_000 },
        "gap-start",
        0,
      ),
    ).toMatchObject({
      sourceStartSample: 0,
      sourceEndSample: 100_000,
      readSampleCount: 100_000,
      validSampleCount: 100_000,
      decodeEndFrame: 79,
    });
  });

  it("retains Fluid's ceil-divided actual-frame end at sub-hop tails", () => {
    for (const totalSamples of [1_281, 1_439]) {
      expect(
        planSeamRepairWindow(
          totalSamples,
          { startSample: 0, endSample: 1_280 },
          "gap-start",
          0,
        ).decodeEndFrame,
      ).toBe(2);
    }
  });

  it("keys every repair-plan field except its diagnostic index", () => {
    const plan = planSeamRepairWindow(
      1_000_000,
      { startSample: 300_000, endSample: 500_000 },
      "gap-start",
      7,
    );
    expect(
      seamRepairPlanCacheKey({ ...plan, index: 99 }),
    ).toBe(seamRepairPlanCacheKey(plan));

    const keyedFields = [
      "windowSampleCount",
      "logicalStartSample",
      "logicalEndSample",
      "sourceStartSample",
      "sourceEndSample",
      "readSampleCount",
      "timestampOffsetSample",
      "emissionStartSample",
      "emissionEndSample",
      "decodeStartFrame",
      "decodeEndFrame",
      "validSampleCount",
    ] as const satisfies readonly (keyof StatelessWindowPlan)[];
    for (const field of keyedFields) {
      expect(
        seamRepairPlanCacheKey({
          ...plan,
          [field]: plan[field] + 1,
        }),
      ).not.toBe(seamRepairPlanCacheKey(plan));
    }
    expect(
      seamRepairPlanCacheKey({
        ...plan,
        flushFinal: !plan.flushFinal,
      }),
    ).not.toBe(seamRepairPlanCacheKey(plan));
  });
});

describe("seam-gap splice filtering", () => {
  const pieces = resolver({
    1: "▁left",
    2: ".",
    3: "ontinuation",
    4: "▁recovered",
    5: "▁right",
    6: "▁Left",
  });

  it("keeps only the word-safe run strictly inside the gap", () => {
    const gap = {
      startFrame: 11,
      endFrame: 30,
      leadNeighbor: token(1, 8),
      tailNeighbor: token(5, 31),
    };
    const candidate = spliceRepairCandidate(
      [
        token(3, 11),
        token(6, 13),
        token(2, 14),
        token(3, 15),
        token(4, 16),
        token(5, 26),
        token(4, 29),
      ],
      gap,
      pieces,
    );
    expect(candidate.map(({ tokenId }) => tokenId)).toEqual([4]);
  });

  it("deduplicates case-only border pieces through six frames", () => {
    const gap = {
      startFrame: 10,
      endFrame: 40,
      leadNeighbor: token(1, 5),
      tailNeighbor: token(5, 45),
    };
    expect(SEAM_GAP_EDGE_TOLERANCE_FRAMES).toBe(6);
    expect(
      spliceRepairCandidate(
        [token(6, 11), token(4, 20)],
        gap,
        pieces,
      ).map(({ tokenId }) => tokenId),
    ).toEqual([4]);
    expect(
      spliceRepairCandidate(
        [token(6, 12), token(4, 20)],
        gap,
        pieces,
      ).map(({ tokenId }) => tokenId),
    ).toEqual([6, 4]);
  });

  it("returns original token objects and never rewrites either edge", () => {
    const recovered = token(4, 20);
    const lead = token(1, 5);
    const tail = token(5, 45);
    const result = spliceRepairCandidate(
      [recovered],
      {
        startFrame: 10,
        endFrame: 40,
        leadNeighbor: lead,
        tailNeighbor: tail,
      },
      pieces,
    );
    expect(result).toEqual([recovered]);
    expect(result[0]).toBe(recovered);
    expect(lead).toEqual(token(1, 5));
    expect(tail).toEqual(token(5, 45));
  });
});
