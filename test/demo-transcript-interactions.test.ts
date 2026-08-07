import { describe, expect, it } from "vitest";

import {
  isTranscriptTailReached,
  transcriptWordAtTime,
} from "../demo/src/transcript-interactions";

describe("demo transcript playback lookup", () => {
  const words = [
    { text: "one", startSeconds: 0.25, endSeconds: 0.8 },
    { text: "two", startSeconds: 1, endSeconds: 1.4 },
    { text: "three", startSeconds: 1.4, endSeconds: 2 },
  ];

  it("uses half-open word timestamps and leaves real gaps unhighlighted", () => {
    expect(transcriptWordAtTime(words, 0.249)).toBeUndefined();
    expect(transcriptWordAtTime(words, 0.25)?.text).toBe("one");
    expect(transcriptWordAtTime(words, 0.799)?.text).toBe("one");
    expect(transcriptWordAtTime(words, 0.8)).toBeUndefined();
    expect(transcriptWordAtTime(words, 0.95)).toBeUndefined();
    expect(transcriptWordAtTime(words, 1)?.text).toBe("two");
    expect(transcriptWordAtTime(words, 1.4)?.text).toBe("three");
    expect(transcriptWordAtTime(words, 2)).toBeUndefined();
    expect(transcriptWordAtTime(words, Number.NaN)).toBeUndefined();
  });
});

describe("demo transcript tail following", () => {
  it("arms only when an overflowing viewport reaches the bottom", () => {
    expect(
      isTranscriptTailReached({
        scrollTop: 0,
        scrollHeight: 400,
        clientHeight: 400,
      }),
    ).toBe(false);
    expect(
      isTranscriptTailReached({
        scrollTop: 250,
        scrollHeight: 800,
        clientHeight: 400,
      }),
    ).toBe(false);
    expect(
      isTranscriptTailReached({
        scrollTop: 397,
        scrollHeight: 800,
        clientHeight: 400,
      }),
    ).toBe(true);
    expect(
      isTranscriptTailReached({
        scrollTop: 400,
        scrollHeight: 800,
        clientHeight: 400,
      }),
    ).toBe(true);
  });

  it("rejects overscroll and invalid viewport measurements", () => {
    expect(
      isTranscriptTailReached({
        scrollTop: 420,
        scrollHeight: 800,
        clientHeight: 400,
      }),
    ).toBe(false);
    expect(
      isTranscriptTailReached({
        scrollTop: -1,
        scrollHeight: 800,
        clientHeight: 400,
      }),
    ).toBe(false);
    expect(
      isTranscriptTailReached({
        scrollTop: 0,
        scrollHeight: Number.NaN,
        clientHeight: 400,
      }),
    ).toBe(false);
  });
});
