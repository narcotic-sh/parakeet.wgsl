import { afterEach, describe, expect, it, vi } from "vitest";

import { ParakeetFbankWasm } from "../../src/audio/fbank-wasm";

describe("Parakeet FBank Wasm loading", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("passes lazy-initialization cancellation into the Wasm request", async () => {
    const controller = new AbortController();
    const reason = new DOMException("cancelled", "AbortError");
    controller.abort(reason);
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) => {
        expect(init?.signal).toBe(controller.signal);
        throw init?.signal?.reason;
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      ParakeetFbankWasm.create(
        "https://example.test/parakeet-fbank-abort.wasm",
        controller.signal,
      ),
    ).rejects.toBe(reason);
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
