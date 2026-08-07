import { describe, expect, it } from "vitest";

import {
  ModelLoadRetryExhaustedError,
  runModelLoadWithRetry,
} from "../src/model-load-retry";
import { ModelTransportError, truncatedModelBody } from "../src/model-transport";

const MODEL_URL = "https://models.example/v1/manifest.json";

describe("runModelLoadWithRetry", () => {
  it("restarts the complete attempt after a native request failure", async () => {
    let requests = 0;
    const delays: number[] = [];
    const value = await runModelLoadWithRetry(
      async (attempt) => {
        const response = await attempt.fetch(MODEL_URL);
        return await response.text();
      },
      {
        fetch: async () => {
          requests += 1;
          if (requests === 1) throw new TypeError("Failed to fetch");
          return new Response("ready");
        },
        baseDelayMs: 100,
        maxDelayMs: 1_000,
        random: () => 0,
        sleep: async (delay) => {
          delays.push(delay);
        },
      },
    );

    expect(value).toBe("ready");
    expect(requests).toBe(2);
    expect(delays).toEqual([50]);
  });

  it("retries a request AbortError when no caller signal was aborted", async () => {
    let requests = 0;
    await expect(
      runModelLoadWithRetry(
        async (attempt) =>
          await (await attempt.fetch(MODEL_URL)).text(),
        {
          fetch: async () => {
            requests += 1;
            if (requests === 1) {
              throw new DOMException("transport interrupted", "AbortError");
            }
            return new Response("ready");
          },
          sleep: async () => undefined,
        },
      ),
    ).resolves.toBe("ready");

    expect(requests).toBe(2);
  });

  it("classifies a response stream failure as a retryable body error", async () => {
    let requests = 0;
    const failures: ModelTransportError[] = [];
    await expect(
      runModelLoadWithRetry(
        async (attempt) => {
          try {
            return await (await attempt.fetch(MODEL_URL)).arrayBuffer();
          } catch (error) {
            if (error instanceof ModelTransportError) failures.push(error);
            throw error;
          }
        },
        {
          fetch: async () => {
            requests += 1;
            if (requests > 1) return new Response(new Uint8Array([1, 2, 3]));
            return new Response(
              new ReadableStream<Uint8Array>({
                start(controller) {
                  controller.enqueue(new Uint8Array([1]));
                  controller.error(
                    new DOMException("transport interrupted", "AbortError"),
                  );
                },
              }),
            );
          },
          sleep: async () => undefined,
        },
      ),
    ).resolves.toHaveProperty("byteLength", 3);

    expect(requests).toBe(2);
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatchObject({
      url: MODEL_URL,
      phase: "body",
      attempt: 1,
      retryable: true,
    });
  });

  it("honors a capped Retry-After for retryable HTTP responses", async () => {
    let requests = 0;
    const delays: number[] = [];
    await expect(
      runModelLoadWithRetry(
        async (attempt) =>
          await (await attempt.fetch(MODEL_URL)).text(),
        {
          fetch: async () => {
            requests += 1;
            return requests === 1
              ? new Response("busy", {
                  status: 503,
                  headers: { "retry-after": "120" },
                })
              : new Response("ready");
          },
          baseDelayMs: 100,
          maxDelayMs: 1_000,
          maxRetryAfterMs: 1_500,
          random: () => 0,
          sleep: async (delay) => {
            delays.push(delay);
          },
        },
      ),
    ).resolves.toBe("ready");

    expect(requests).toBe(2);
    expect(delays).toEqual([1_500]);
  });

  it("does not retry deterministic HTTP failures", async () => {
    let requests = 0;
    const failure = await runModelLoadWithRetry(
      async (attempt) => await attempt.fetch(MODEL_URL),
      {
        fetch: async () => {
          requests += 1;
          return new Response("missing", { status: 404 });
        },
        sleep: async () => {
          throw new Error("terminal HTTP failures must not sleep");
        },
      },
    ).catch((error: unknown) => error);

    expect(requests).toBe(1);
    expect(failure).toBeInstanceOf(ModelTransportError);
    expect(failure).toMatchObject({
      url: MODEL_URL,
      phase: "response",
      status: 404,
      retryable: false,
    });
  });

  it("caps exponential backoff and reports exhaustion after four attempts", async () => {
    const delays: number[] = [];
    let attempts = 0;
    const failure = await runModelLoadWithRetry(
      async (attempt) => {
        attempts += 1;
        return await attempt.fetch(MODEL_URL);
      },
      {
        fetch: async () => {
          throw new TypeError("network unavailable");
        },
        baseDelayMs: 100,
        maxDelayMs: 250,
        random: () => 1,
        sleep: async (delay) => {
          delays.push(delay);
        },
      },
    ).catch((error: unknown) => error);

    expect(attempts).toBe(4);
    expect(delays).toEqual([100, 200, 250]);
    expect(failure).toBeInstanceOf(ModelLoadRetryExhaustedError);
    expect(failure).toMatchObject({
      attempts: 4,
      lastError: {
        url: MODEL_URL,
        phase: "request",
        attempt: 4,
      },
    });
    expect((failure as Error).message).toContain("failed after 4 attempts");
  });

  it("cancels abandoned response bodies before retry cleanup", async () => {
    let sourceCancelled = false;
    let cleanupSawCancellation = false;
    const pendingReads: Promise<unknown>[] = [];
    await expect(
      runModelLoadWithRetry(
        async (attempt) => {
          if (attempt.number > 1) return "ready";
          const abandoned = await attempt.fetch(
            "https://models.example/v1/weights.bin",
          );
          const reader = abandoned.body!.getReader();
          pendingReads.push(reader.read().catch((error: unknown) => error));
          await Promise.resolve();
          await attempt.fetch(MODEL_URL);
          return "unreachable";
        },
        {
          fetch: async (input) => {
            if (input.toString() === MODEL_URL) {
              throw new TypeError("manifest request failed");
            }
            return new Response(
              new ReadableStream<Uint8Array>({
                pull() {
                  // Stay pending until the failed attempt cancels this sibling.
                },
                cancel() {
                  sourceCancelled = true;
                },
              }),
            );
          },
          cleanup: async () => {
            cleanupSawCancellation = sourceCancelled;
          },
          sleep: async () => undefined,
        },
      ),
    ).resolves.toBe("ready");
    await Promise.all(pendingReads);

    expect(sourceCancelled).toBe(true);
    expect(cleanupSawCancellation).toBe(true);
  });

  it("discards a response that resolves after its attempt was cancelled", async () => {
    const reason = new DOMException("attempt superseded", "AbortError");
    let deliverResponse:
      | ((response: Response) => void)
      | undefined;
    let responseBodyCancelled = false;

    const failure = await runModelLoadWithRetry(
      async (attempt) => {
        const pending = attempt.fetch(MODEL_URL);
        await Promise.resolve();
        await attempt.cancel(reason);
        deliverResponse?.(
          new Response(
            new ReadableStream<Uint8Array>({
              cancel() {
                responseBodyCancelled = true;
              },
            }),
          ),
        );
        return await pending;
      },
      {
        fetch: async () =>
          await new Promise<Response>((resolve) => {
            deliverResponse = resolve;
          }),
      },
    ).catch((error: unknown) => error);

    expect(failure).toBe(reason);
    expect(responseBodyCancelled).toBe(true);
  });

  it("rejects retry policies that could exceed four total attempts", async () => {
    await expect(
      runModelLoadWithRetry(async () => "unused", { maxAttempts: 5 }),
    ).rejects.toThrow("maxAttempts must be an integer from 1 through 4");
  });

  it("retries a gracefully truncated pinned response", async () => {
    let attempts = 0;
    const bytes = await runModelLoadWithRetry(
      async (attempt) => {
        attempts += 1;
        const data = await (await attempt.fetch(MODEL_URL)).arrayBuffer();
        if (data.byteLength < 4) {
          throw truncatedModelBody(MODEL_URL, data.byteLength, 4);
        }
        return new Uint8Array(data);
      },
      {
        fetch: async () =>
          new Response(
            attempts === 1
              ? new Uint8Array([1, 2])
              : new Uint8Array([1, 2, 3, 4]),
          ),
        sleep: async () => undefined,
      },
    );

    expect(attempts).toBe(2);
    expect(bytes).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  it("does not retry deterministic validation failures", async () => {
    let attempts = 0;
    await expect(
      runModelLoadWithRetry(
        async () => {
          attempts += 1;
          throw new Error("model integrity validation failed");
        },
        {
          sleep: async () => {
            throw new Error("validation failures must not sleep");
          },
        },
      ),
    ).rejects.toBeInstanceOf(Error);

    expect(attempts).toBe(1);
  });

  it("does not retry an aborted model load", async () => {
    const controller = new AbortController();
    controller.abort(new DOMException("user cancelled", "AbortError"));
    let attempts = 0;

    await expect(
      runModelLoadWithRetry(
        async () => {
          attempts += 1;
          return "unused";
        },
        { signal: controller.signal },
      ),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(attempts).toBe(0);
  });

  it("honors an aborted Request signal without retrying", async () => {
    const controller = new AbortController();
    const reason = new DOMException("request cancelled", "AbortError");
    controller.abort(reason);
    const request = new Request(MODEL_URL, { signal: controller.signal });
    let requests = 0;

    const failure = await runModelLoadWithRetry(
      async (attempt) => await attempt.fetch(request),
      {
        fetch: async (_input, init) => {
          requests += 1;
          expect(init?.signal?.aborted).toBe(true);
          throw init?.signal?.reason;
        },
        sleep: async () => {
          throw new Error("an intentional request abort must not sleep");
        },
      },
    ).catch((error: unknown) => error);

    expect(failure).toBe(reason);
    expect(requests).toBe(1);
  });
});
