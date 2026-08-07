import { describe, expect, it } from "vitest";

import {
  parseRetryAfterMilliseconds,
  retryableSampleStatus,
  sampleRetryDelayMilliseconds,
} from "../demo/src/sample-download-policy";

describe("demo sample audio retry policy", () => {
  it("retries only transient HTTP statuses", () => {
    for (const status of [408, 425, 429, 500, 503, 599]) {
      expect(retryableSampleStatus(status)).toBe(true);
    }
    for (const status of [200, 400, 404, 499, 600]) {
      expect(retryableSampleStatus(status)).toBe(false);
    }
  });

  it("uses bounded exponential jitter and capped Retry-After", () => {
    expect(sampleRetryDelayMilliseconds(1, 0, () => 0)).toBe(250);
    expect(sampleRetryDelayMilliseconds(2, 0, () => 1)).toBe(1_000);
    expect(sampleRetryDelayMilliseconds(4, 45_000, () => 0)).toBe(30_000);
    expect(
      sampleRetryDelayMilliseconds(1, Number.NaN, () => Number.NaN),
    ).toBe(250);
  });

  it("parses seconds and HTTP dates", () => {
    const now = Date.parse("2026-08-03T00:00:00Z");
    expect(parseRetryAfterMilliseconds("1.5", now)).toBe(1_500);
    expect(
      parseRetryAfterMilliseconds("Mon, 03 Aug 2026 00:00:04 GMT", now),
    ).toBe(4_000);
    expect(parseRetryAfterMilliseconds("invalid", now)).toBe(0);
    expect(parseRetryAfterMilliseconds(null, now)).toBe(0);
  });
});
