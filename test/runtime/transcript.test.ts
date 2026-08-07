import { describe, expect, it } from "vitest";

import { serializeTimedTranscript } from "../../src/runtime/transcript";
import type { TimedToken } from "../../src/runtime/stitch";

describe("timestamped transcript serialization", () => {
  it("groups continuation and punctuation pieces into clean timestamped words", () => {
    const pieces: Record<number, string> = {
      1: "▁hel",
      2: "lo",
      3: ",",
      4: "▁world",
      5: "!",
    };
    const tokens = [
      timedToken(1, 0, 80),
      timedToken(2, 80, 160),
      timedToken(3, 160, 200),
      timedToken(4, 240, 320),
      timedToken(5, 320, 360),
    ];

    const result = serializeTimedTranscript(tokens, 80, {
      tokenPiece: (tokenId) => pieces[tokenId],
      decodeTokenIds: (tokenIds) =>
        tokenIds
          .map((tokenId) => pieces[tokenId]!)
          .join("")
          .replaceAll("▁", " ")
          .trimStart(),
    });

    expect(result.text).toBe("hello, world!");
    expect(result.tokens).toEqual([
      { tokenId: 1, startSeconds: 0, endSeconds: 1 },
      { tokenId: 2, startSeconds: 1, endSeconds: 2 },
      { tokenId: 3, startSeconds: 2, endSeconds: 2.5 },
      { tokenId: 4, startSeconds: 3, endSeconds: 4 },
      { tokenId: 5, startSeconds: 4, endSeconds: 4.5 },
    ]);
    expect(result.words).toEqual([
      { text: "hello,", startSeconds: 0, endSeconds: 2.5 },
      { text: "world!", startSeconds: 3, endSeconds: 4.5 },
    ]);
  });

  it("gives unknown injected-engine pieces their own timestamped words", () => {
    const result = serializeTimedTranscript(
      [timedToken(7, 160, 320), timedToken(8, 320, 480)],
      160,
      {
        tokenPiece: () => undefined,
        decodeTokenIds: (tokenIds) => tokenIds.join(" "),
      },
    );

    expect(result.text).toBe("7 8");
    expect(result.words).toEqual([
      { text: "7", startSeconds: 1, endSeconds: 2 },
      { text: "8", startSeconds: 2, endSeconds: 3 },
    ]);
  });

  it("keeps skipped special tokens out of word timing", () => {
    const pieces: Record<number, string> = {
      1: "▁hel",
      2: "<unk>",
      3: "lo",
      4: "▁world",
    };
    const result = serializeTimedTranscript(
      [
        timedToken(1, 0, 80),
        timedToken(2, 80, 800),
        timedToken(3, 160, 240),
        timedToken(4, 320, 400),
      ],
      80,
      {
        tokenPiece: (tokenId) => pieces[tokenId],
        decodeTokenIds: (tokenIds) =>
          tokenIds
            .filter((tokenId) => tokenId !== 2)
            .map((tokenId) => pieces[tokenId]!)
            .join("")
            .replaceAll("▁", " ")
            .trimStart(),
      },
    );

    expect(result.text).toBe("hello world");
    expect(result.words).toEqual([
      { text: "hello", startSeconds: 0, endSeconds: 3 },
      { text: "world", startSeconds: 4, endSeconds: 5 },
    ]);
  });

  it("returns empty timestamp collections for an empty transcript", () => {
    expect(
      serializeTimedTranscript([], 16_000, {
        tokenPiece: () => undefined,
        decodeTokenIds: () => "",
      }),
    ).toEqual({ text: "", tokens: [], words: [] });
  });
});

function timedToken(
  tokenId: number,
  startSample: number,
  endSample: number,
): TimedToken {
  return {
    tokenId,
    startSample,
    endSample,
    durationFrames: 1,
  };
}
