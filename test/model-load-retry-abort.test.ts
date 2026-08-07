import { describe, expect, it } from "vitest";

import { runModelLoadWithRetry } from "../src/model-load-retry";

describe("model retry cancellation", () => {
  it("aborts an in-progress backoff without starting another attempt", async () => {
    const controller = new AbortController();
    const reason = new DOMException("user cancelled", "AbortError");
    let attempts = 0;

    const result = runModelLoadWithRetry(
      async (attempt) => {
        attempts += 1;
        return await attempt.fetch("https://models.example/manifest.json");
      },
      {
        signal: controller.signal,
        fetch: async () => {
          throw new TypeError("temporary network failure");
        },
        baseDelayMs: 30_000,
        maxDelayMs: 30_000,
        onRetry: () => controller.abort(reason),
      },
    );

    await expect(result).rejects.toBe(reason);
    expect(attempts).toBe(1);
  });
});
