import { describe, expect, it } from "vitest";

import { isStreamingDecoderControlResponse } from "../../src/audio/streaming-decoder-protocol";

describe("streaming decoder control protocol", () => {
  it("accepts safe advisory sample estimates and an unknown estimate", () => {
    expect(
      isStreamingDecoderControlResponse(
        {
          type: "streaming-decoder-ready",
          requestId: "request",
          estimatedSampleCount: 160_000,
        },
        "request",
      ),
    ).toBe(true);
    expect(
      isStreamingDecoderControlResponse(
        {
          type: "streaming-decoder-ready",
          requestId: "request",
          estimatedSampleCount: null,
        },
        "request",
      ),
    ).toBe(true);
  });

  it("rejects absent, negative, fractional, and wrong-request estimates", () => {
    for (const estimatedSampleCount of [undefined, -1, 1.5]) {
      expect(
        isStreamingDecoderControlResponse(
          {
            type: "streaming-decoder-ready",
            requestId: "request",
            ...(estimatedSampleCount === undefined
              ? {}
              : { estimatedSampleCount }),
          },
          "request",
        ),
      ).toBe(false);
    }
    expect(
      isStreamingDecoderControlResponse(
        {
          type: "streaming-decoder-ready",
          requestId: "other",
          estimatedSampleCount: 160_000,
        },
        "request",
      ),
    ).toBe(false);
  });
});
