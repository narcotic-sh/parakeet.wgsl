import { describe, expect, it } from "vitest";

import { formatProgressElapsedMilliseconds } from "../demo/src/progress-timing";

describe("demo transcription elapsed timer", () => {
  it("formats a compact stopwatch below one hour", () => {
    expect(formatProgressElapsedMilliseconds(0)).toBe("0:00 elapsed");
    expect(formatProgressElapsedMilliseconds(12_999)).toBe("0:12 elapsed");
    expect(formatProgressElapsedMilliseconds(59_999)).toBe("0:59 elapsed");
    expect(formatProgressElapsedMilliseconds(60_000)).toBe("1:00 elapsed");
    expect(formatProgressElapsedMilliseconds(3_599_999)).toBe(
      "59:59 elapsed",
    );
  });

  it("adds hours and clamps invalid input", () => {
    expect(formatProgressElapsedMilliseconds(3_600_000)).toBe(
      "1:00:00 elapsed",
    );
    expect(formatProgressElapsedMilliseconds(3_723_000)).toBe(
      "1:02:03 elapsed",
    );
    expect(formatProgressElapsedMilliseconds(-1)).toBe("0:00 elapsed");
    expect(formatProgressElapsedMilliseconds(Number.NaN)).toBe(
      "0:00 elapsed",
    );
  });
});
