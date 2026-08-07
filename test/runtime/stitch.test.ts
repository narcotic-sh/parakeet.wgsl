import { describe, expect, it } from "vitest";

import {
  collapseSeamWordDuplicates,
  enforceMonotonicTokenTimes,
  globalizeTdtTokens,
  IncrementalStitchAccumulator,
  isSpliceSafeToken,
  stitchTokenWindows,
  type DecodedTokenWindow,
  type TokenPieceResolver,
  type TimedToken,
} from "../../src/runtime/stitch";
import { planStatelessWindows } from "../../src/runtime/chunking";
import type { TdtEmittedToken } from "../../src/runtime/tdt-types";

function token(
  tokenId: number,
  startSample: number,
  endSample = startSample,
): TimedToken {
  return {
    tokenId,
    startSample,
    endSample,
    durationFrames: Math.max(
      1,
      Math.ceil((endSample - startSample) / 1_280),
    ),
  };
}

function frameToken(tokenId: number, frame: number): TimedToken {
  const startSample = frame * 1_280;
  return token(tokenId, startSample, startSample + 1_280);
}

function resolver(
  vocabulary: Readonly<Record<number, string>>,
): TokenPieceResolver {
  return (tokenId) => vocabulary[tokenId];
}

function tokenWindows(
  left: readonly TimedToken[],
  right: readonly TimedToken[],
) {
  return [
    {
      index: 0,
      startSample: 0,
      endSample: 240_000,
      tokens: left,
    },
    {
      index: 1,
      startSample: 206_080,
      endSample: 446_080,
      tokens: right,
    },
  ] as const;
}

describe("stitchTokenWindows", () => {
  it("splices on a timestamp-compatible contiguous token run", () => {
    const result = stitchTokenWindows(
      [
        {
          index: 0,
          startSample: 0,
          endSample: 240_000,
          tokens: [
            token(1, 190_000),
            token(2, 212_000),
            token(3, 224_000),
          ],
        },
        {
          index: 1,
          startSample: 208_000,
          endSample: 448_000,
          tokens: [
            token(2, 213_000),
            token(3, 225_000),
            token(4, 250_000),
          ],
        },
      ],
    );

    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      1, 2, 3, 4,
    ]);
    expect(result.trace.map(({ method }) => method)).toEqual(["contiguous"]);
  });

  it("uses a deterministic midpoint LCS pivot when overlap differs", () => {
    const result = stitchTokenWindows(
      [
        {
          index: 0,
          startSample: 0,
          endSample: 240_000,
          tokens: [
            token(1, 190_000),
            token(2, 210_000),
            token(9, 218_000),
            token(3, 225_000),
            token(4, 235_000),
          ],
        },
        {
          index: 1,
          startSample: 208_000,
          endSample: 448_000,
          tokens: [
            token(2, 211_000),
            token(3, 226_000),
            token(8, 232_000),
            token(4, 236_000),
            token(5, 250_000),
          ],
        },
      ],
    );

    expect(result.trace[0]!.method).toBe("lcs");
    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      1, 2, 9, 3, 8, 4, 5,
    ]);
  });

  it("falls back to a deterministic timestamp midpoint without token matches", () => {
    const result = stitchTokenWindows([
      {
        index: 0,
        startSample: 0,
        endSample: 240_000,
        tokens: [token(1, 210_000), token(2, 230_000)],
      },
      {
        index: 1,
        startSample: 208_000,
        endSample: 448_000,
        tokens: [token(3, 215_000), token(4, 240_000)],
      },
    ]);

    expect(result.trace[0]!.method).toBe("midpoint");
    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([1, 4]);
  });

  it("keeps a right window that has no safe token after the midpoint", () => {
    const vocabulary = resolver({
      1: "▁hello",
      511: "▁world",
      549: "pre",
      550: "ne",
      551: "ta",
      552: "tion",
    });
    const result = stitchTokenWindows(
      tokenWindows(
        [frameToken(1, 120), frameToken(511, 140)],
        [
          frameToken(549, 140),
          frameToken(550, 141),
          frameToken(551, 142),
          frameToken(552, 143),
        ],
      ),
      vocabulary,
    );

    expect(result.trace[0]!.method).toBe("midpoint");
    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      1, 511, 550, 551, 552,
    ]);
  });

  it("keeps a continuation tail when the right window ends mid-word", () => {
    const vocabulary = resolver({
      1: "▁hello",
      424: "▁fan",
      425: "tas",
      426: "ti",
      427: "cal",
    });
    const result = stitchTokenWindows(
      tokenWindows(
        [
          frameToken(1, 120),
          frameToken(424, 130),
          frameToken(425, 131),
        ],
        [
          frameToken(425, 131),
          frameToken(426, 132),
          frameToken(427, 133),
        ],
      ),
      vocabulary,
    );

    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      1, 424, 425, 426, 427,
    ]);
  });

  it("uses case-only token variants as seam anchors", () => {
    const vocabulary = resolver({
      10: "▁the",
      20: "▁meeting",
      21: "▁Meeting",
      30: "▁was",
      40: "▁good",
    });
    const result = stitchTokenWindows(
      tokenWindows(
        [
          frameToken(10, 128),
          frameToken(30, 130),
          frameToken(20, 132),
        ],
        [
          frameToken(10, 128),
          frameToken(30, 130),
          frameToken(21, 132),
          frameToken(40, 134),
        ],
      ),
      vocabulary,
    );

    expect(result.trace[0]!.method).toBe("contiguous");
    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      10, 30, 20, 40,
    ]);
  });

  it("finishes the left word and skips an orphaned right continuation", () => {
    const vocabulary = resolver({
      10: "▁hello",
      20: "▁wor",
      21: "ld",
      30: "▁there",
      50: "ne",
      60: "▁o",
    });
    const result = stitchTokenWindows(
      tokenWindows(
        [
          frameToken(10, 120),
          frameToken(20, 133),
          frameToken(21, 135),
        ],
        [
          frameToken(60, 134),
          frameToken(50, 136),
          frameToken(30, 138),
        ],
      ),
      vocabulary,
    );

    expect(result.trace[0]!.method).toBe("midpoint");
    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([
      10, 20, 21, 30,
    ]);
  });
});

describe("enforceMonotonicTokenTimes", () => {
  it("keeps same-frame subwords in emission order", () => {
    const tokens = [
      frameToken(100, 40),
      frameToken(101, 40),
      frameToken(102, 40),
    ] as const;

    const result = enforceMonotonicTokenTimes(tokens);

    expect(result.map(({ tokenId }) => tokenId)).toEqual([100, 101, 102]);
    expect(result.map(({ startSample }) => startSample)).toEqual([
      51_200, 51_200, 51_200,
    ]);
    expect(result).not.toBe(tokens);
  });

  it("leaves an already-monotonic stream unchanged", () => {
    const tokens = [
      frameToken(1, 10),
      frameToken(2, 12),
      frameToken(3, 12),
      frameToken(4, 20),
    ] as const;

    expect(enforceMonotonicTokenTimes(tokens)).toEqual(tokens);
    expect(enforceMonotonicTokenTimes([])).toEqual([]);
  });
});

describe("IncrementalStitchAccumulator", () => {
  const windows: readonly DecodedTokenWindow[] = [
    {
      index: 0,
      startSample: 0,
      endSample: 240_000,
      tokens: [
        token(100, 100_000),
        token(1, 190_000),
        token(2, 212_000),
        token(3, 224_000),
      ],
    },
    {
      index: 1,
      startSample: 208_000,
      endSample: 448_000,
      tokens: [
        token(2, 213_000),
        token(3, 225_000),
        token(4, 300_000),
        token(10, 390_000),
        token(20, 410_000),
        token(99, 418_000),
        token(30, 425_000),
        token(40, 435_000),
      ],
    },
    {
      index: 2,
      startSample: 416_000,
      endSample: 656_000,
      tokens: [
        token(20, 411_000),
        token(30, 426_000),
        token(88, 432_000),
        token(40, 436_000),
        token(50, 500_000),
      ],
    },
    {
      index: 3,
      startSample: 624_000,
      endSample: 864_000,
      tokens: [
        token(60, 630_000),
        token(61, 810_000),
        token(62, 850_000),
      ],
    },
    {
      index: 4,
      startSample: 832_000,
      endSample: 1_072_000,
      tokens: [token(70, 835_000), token(71, 870_000)],
    },
  ];
  const expectedTokenPrefixes = [
    [100, 1, 2, 3],
    [100, 1, 2, 3, 4, 10, 20, 99, 30, 40],
    [100, 1, 2, 3, 4, 10, 20, 99, 30, 88, 40, 50],
    [100, 1, 2, 3, 4, 10, 20, 99, 30, 88, 40, 50, 60, 61, 62],
    [100, 1, 2, 3, 4, 10, 20, 99, 30, 88, 40, 50, 60, 61, 71],
  ] as const;
  const expectedMethods = [
    "contiguous",
    "lcs",
    "disjoint",
    "midpoint",
  ] as const;
  const expectedTrace = [
    {
      leftWindowIndex: 0,
      rightWindowIndex: 1,
      method: "contiguous",
      overlapStartSample: 208_000,
      overlapEndSample: 240_000,
    },
    {
      leftWindowIndex: 1,
      rightWindowIndex: 2,
      method: "lcs",
      overlapStartSample: 416_000,
      overlapEndSample: 448_000,
    },
    {
      leftWindowIndex: 2,
      rightWindowIndex: 3,
      method: "disjoint",
      overlapStartSample: 624_000,
      overlapEndSample: 656_000,
    },
    {
      leftWindowIndex: 3,
      rightWindowIndex: 4,
      method: "midpoint",
      overlapStartSample: 832_000,
      overlapEndSample: 864_000,
    },
  ] as const;
  const pieces = resolver(
    Object.fromEntries(
      expectedTokenPrefixes
        .flat()
        .map((tokenId) => [tokenId, `▁token-${tokenId}`]),
    ),
  );

  it("matches every one-shot prefix across all merge strategies", () => {
    for (const tokenPiece of [undefined, pieces] as const) {
      const accumulator = new IncrementalStitchAccumulator({
        ...(tokenPiece === undefined ? {} : { tokenPiece }),
        collectTrace: true,
      });

      for (let index = 0; index < windows.length; index += 1) {
        accumulator.append(windows[index]!);
        const snapshot = accumulator.snapshot();
        const oneShot = stitchTokenWindows(
          windows.slice(0, index + 1),
          tokenPiece,
        );

        expect(snapshot).toEqual(oneShot);
        expect(snapshot.tokens.map((item) => item.tokenId)).toEqual(
          expectedTokenPrefixes[index],
        );
        expect(snapshot.trace.map((item) => item.method)).toEqual(
          expectedMethods.slice(0, index),
        );
        expect(snapshot.trace).toEqual(expectedTrace.slice(0, index));
      }
    }
  });

  it("returns finalized snapshots without trace allocations when disabled", () => {
    const accumulator = new IncrementalStitchAccumulator({
      tokenPiece: pieces,
    });
    for (const window of windows) accumulator.append(window);

    const first = accumulator.snapshot();
    const second = accumulator.snapshot();
    const oneShot = stitchTokenWindows(windows, pieces);

    expect(first.tokens).toEqual(oneShot.tokens);
    expect(first.trace).toEqual([]);
    expect(second).toEqual(first);
    expect(second.tokens).not.toBe(first.tokens);
    expect(second.trace).toBe(first.trace);
  });

  it("keeps snapshot duplicate cleanup out of subsequent merge state", () => {
    const tokenPieces = resolver({
      1: "▁ha",
      2: "ve",
      3: "▁Ha",
      7: "▁after",
    });
    const accumulator = new IncrementalStitchAccumulator({
      tokenPiece: tokenPieces,
      collectTrace: true,
    });
    const firstWindow = {
      index: 0,
      startSample: 0,
      endSample: 240_000,
      tokens: [
        frameToken(1, 100),
        frameToken(2, 101),
        frameToken(3, 102),
        frameToken(2, 103),
      ],
    };
    const secondWindow = {
      index: 1,
      startSample: 400_000,
      endSample: 640_000,
      tokens: [frameToken(7, 320)],
    };

    accumulator.append(firstWindow);
    expect(
      accumulator.snapshot().tokens.map((item) => item.tokenId),
    ).toEqual([1, 2]);
    accumulator.append(secondWindow);

    expect(accumulator.snapshot()).toEqual(
      stitchTokenWindows([firstWindow, secondWindow], tokenPieces),
    );
  });

  it("preserves cross-window text order while clamping a backward timestamp", () => {
    const leftPrefix = Object.freeze(frameToken(10, 158));
    const leftAnchor = Object.freeze(frameToken(20, 160));
    const rightAnchor = Object.freeze(frameToken(20, 159));
    const rightTail = Object.freeze(frameToken(30, 159));
    const firstWindow = {
      index: 0,
      startSample: 0,
      endSample: 240_000,
      tokens: Object.freeze([leftPrefix, leftAnchor]),
    } as const;
    const secondWindow = {
      index: 1,
      startSample: 206_080,
      endSample: 446_080,
      tokens: Object.freeze([rightAnchor, rightTail]),
    } as const;
    const accumulator = new IncrementalStitchAccumulator({
      collectTrace: true,
    });

    accumulator.append(firstWindow);
    accumulator.append(secondWindow);
    const firstSnapshot = accumulator.snapshot();

    expect(firstSnapshot.tokens.map(({ tokenId }) => tokenId)).toEqual([
      10, 20, 30,
    ]);
    expect(firstSnapshot.tokens.map(({ startSample }) => startSample)).toEqual([
      202_240, 204_800, 204_800,
    ]);
    expect(firstSnapshot.tokens[2]).toMatchObject({
      endSample: 206_080,
      durationFrames: 1,
    });
    expect(rightTail).toEqual({
      tokenId: 30,
      startSample: 203_520,
      endSample: 204_800,
      durationFrames: 1,
    });

    accumulator.append({
      index: 2,
      startSample: 500_000,
      endSample: 740_000,
      tokens: [frameToken(40, 400)],
    });
    const secondSnapshot = accumulator.snapshot();

    expect(secondSnapshot.tokens.map(({ tokenId }) => tokenId)).toEqual([
      10, 20, 30, 40,
    ]);
    expect(firstSnapshot.tokens.map(({ tokenId }) => tokenId)).toEqual([
      10, 20, 30,
    ]);
    expect(firstSnapshot.tokens[2]!.startSample).toBe(204_800);
  });

  it("rejects windows appended out of chronological order", () => {
    const accumulator = new IncrementalStitchAccumulator();
    accumulator.append(windows[1]!);
    expect(() => accumulator.append(windows[0]!)).toThrow(
      "incremental stitch windows must be appended in chronological order",
    );

    const sameStart = new IncrementalStitchAccumulator();
    sameStart.append({
      index: 2,
      startSample: 0,
      endSample: 240_000,
      tokens: [],
    });
    expect(() =>
      sameStart.append({
        index: 1,
        startSample: 0,
        endSample: 240_000,
        tokens: [],
      }),
    ).toThrow(
      "incremental stitch windows must be appended in chronological order",
    );

    expect(stitchTokenWindows([windows[1]!, windows[0]!])).toEqual(
      stitchTokenWindows([windows[0]!, windows[1]!]),
    );
  });
});

describe("token-aware seam cleanup", () => {
  const pieces = resolver({
    1: "▁ha",
    2: "ve",
    3: "▁Ha",
    4: "▁ei",
    5: "ther",
    6: "▁Ei",
    7: "▁a",
    9: ".",
    20: "continuation",
    21: ",",
  });

  it("classifies word starts and punctuation as splice-safe", () => {
    expect(isSpliceSafeToken(1, pieces)).toBe(true);
    expect(isSpliceSafeToken(21, pieces)).toBe(true);
    expect(isSpliceSafeToken(20, pieces)).toBe(false);
  });

  it("collapses a multi-token case-only seam duplicate", () => {
    const collapsed = collapseSeamWordDuplicates(
      [
        frameToken(1, 100),
        frameToken(2, 101),
        frameToken(3, 102),
        frameToken(2, 103),
        frameToken(7, 104),
      ],
      pieces,
    );
    expect(collapsed.map(({ tokenId }) => tokenId)).toEqual([1, 2, 7]);
  });

  it("clamps before duplicate collapse without scrambling seam pieces", () => {
    const result = stitchTokenWindows(
      tokenWindows(
        [frameToken(1, 158), frameToken(2, 160)],
        [
          frameToken(2, 159),
          frameToken(3, 159),
          frameToken(2, 160),
          frameToken(7, 162),
        ],
      ),
      pieces,
    );

    expect(result.tokens.map(({ tokenId }) => tokenId)).toEqual([1, 2, 7]);
    expect(result.tokens.map(({ startSample }) => startSample)).toEqual([
      202_240, 204_800, 207_360,
    ]);
  });

  it("keeps lowercase when the capitalized duplicate comes first", () => {
    const collapsed = collapseSeamWordDuplicates(
      [
        frameToken(6, 100),
        frameToken(5, 101),
        frameToken(4, 102),
        frameToken(5, 103),
        frameToken(7, 104),
      ],
      pieces,
    );
    expect(collapsed.map(({ tokenId }) => tokenId)).toEqual([4, 5, 7]);
  });

  it("preserves genuine repeats and sentence-boundary capitalization", () => {
    expect(
      collapseSeamWordDuplicates(
        [
          frameToken(1, 100),
          frameToken(2, 101),
          frameToken(1, 102),
          frameToken(2, 103),
        ],
        pieces,
      ).map(({ tokenId }) => tokenId),
    ).toEqual([1, 2, 1, 2]);
    expect(
      collapseSeamWordDuplicates(
        [
          frameToken(1, 100),
          frameToken(2, 101),
          frameToken(9, 102),
          frameToken(3, 103),
          frameToken(2, 104),
          frameToken(7, 105),
        ],
        pieces,
      ).map(({ tokenId }) => tokenId),
    ).toEqual([1, 2, 9, 3, 2, 7]);
  });
});

describe("globalizeTdtTokens", () => {
  it("offsets local frame timestamps and suppresses an end-aligned warmup", () => {
    const finalWindow = planStatelessWindows(34 * 16_000)[2]!;
    const localTokens: TdtEmittedToken[] = [
      {
        tokenId: 1,
        frameIndex: 80,
        durationFrames: 1,
      },
      {
        tokenId: 2,
        frameIndex: 89,
        durationFrames: 2,
      },
    ];

    const global = globalizeTdtTokens(localTokens, finalWindow);
    expect(global).toHaveLength(1);
    expect(global[0]).toEqual(
      expect.objectContaining({
        tokenId: 2,
        startSample: 419_840,
        endSample: 422_400,
      }),
    );
  });

  it("uses the fixed source samples per encoder frame", () => {
    const window = planStatelessWindows(384_000)[0]!;
    const global = globalizeTdtTokens(
      [
        {
          tokenId: 7,
          frameIndex: 10,
          durationFrames: 2,
        },
      ],
      window,
    );

    expect(global[0]).toMatchObject({
      startSample: 12_800,
      endSample: 15_360,
    });
  });

  it("uses the default half-overlap timestamp tolerance while stitching", () => {
    const windows = [
      {
        index: 0,
        startSample: 0,
        endSample: 240_000,
        tokens: [token(1, 210_000), token(2, 220_000)],
      },
      {
        index: 1,
        startSample: 208_000,
        endSample: 448_000,
        tokens: [token(1, 211_400), token(2, 221_400)],
      },
    ];
    expect(stitchTokenWindows(windows).trace[0]!.method).toBe("contiguous");
  });
});
