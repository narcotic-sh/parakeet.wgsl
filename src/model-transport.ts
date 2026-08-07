export type ModelTransportPhase = "request" | "response" | "body";

export interface ModelTransportErrorOptions {
  readonly url: string;
  readonly phase: ModelTransportPhase;
  readonly attempt?: number;
  readonly status?: number;
  readonly retryAfterMs?: number;
  readonly retryable: boolean;
  readonly detail?: string;
  readonly cause?: unknown;
}

/**
 * A failure while transferring one of the files that make up a browser model.
 *
 * HTTP failures are represented here as well so the model-set loader has one
 * explicit retry classification instead of relying on browser-specific error
 * strings.
 */
export class ModelTransportError extends Error {
  readonly url: string;
  readonly phase: ModelTransportPhase;
  readonly attempt: number | undefined;
  readonly status: number | undefined;
  readonly retryAfterMs: number | undefined;
  readonly retryable: boolean;
  readonly detail: string | undefined;

  constructor(options: ModelTransportErrorOptions) {
    const status =
      options.status === undefined ? "" : ` (HTTP ${options.status})`;
    const attempt =
      options.attempt === undefined ? "" : ` on attempt ${options.attempt}`;
    const detail = options.detail === undefined ? "" : `: ${options.detail}`;
    super(
      `Model ${options.phase} failed for ${options.url}${status}${attempt}${detail}`,
      options.cause === undefined ? undefined : { cause: options.cause },
    );
    this.name = "ModelTransportError";
    this.url = options.url;
    this.phase = options.phase;
    this.attempt = options.attempt;
    this.status = options.status;
    this.retryAfterMs = options.retryAfterMs;
    this.retryable = options.retryable;
    this.detail = options.detail;
  }
}

interface ActiveBody {
  cancel(reason: unknown): Promise<void>;
}

interface ModelFetchAttemptOptions {
  readonly attempt: number;
  readonly signal?: AbortSignal;
  readonly now?: () => number;
}

/**
 * One cancellable set of model requests. A fresh instance is created whenever
 * the selected Parakeet package restarts after a transient failure.
 */
export class ModelFetchAttempt {
  readonly fetch: typeof fetch;

  private readonly controller = new AbortController();
  private readonly activeBodies = new Set<ActiveBody>();
  private readonly now: () => number;
  private cancellation: Promise<void> | undefined;

  constructor(
    private readonly networkFetch: typeof fetch,
    private readonly options: ModelFetchAttemptOptions,
  ) {
    this.now = options.now ?? Date.now;
    this.fetch = this.fetchModelAsset.bind(this) as typeof fetch;
  }

  async cancel(reason: unknown): Promise<void> {
    if (this.cancellation !== undefined) return await this.cancellation;
    if (!this.controller.signal.aborted) this.controller.abort(reason);
    this.cancellation = Promise.allSettled(
      [...this.activeBodies].map((body) => body.cancel(reason)),
    ).then(() => undefined);
    await this.cancellation;
  }

  private async fetchModelAsset(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const url = requestUrl(input);
    const combined = combineAbortSignals([
      this.controller.signal,
      this.options.signal,
      input instanceof Request ? input.signal : undefined,
      init?.signal,
    ]);
    let response: Response;
    try {
      response = await this.networkFetch(input, {
        ...init,
        signal: combined.signal,
      });
    } catch (error) {
      combined.dispose();
      if (combined.signal.aborted) {
        throw abortReason(combined.signal.reason);
      }
      throw new ModelTransportError({
        url,
        phase: "request",
        attempt: this.options.attempt,
        retryable: true,
        detail: errorMessage(error),
        cause: error,
      });
    }

    if (combined.signal.aborted) {
      try {
        await response.body?.cancel(combined.signal.reason);
      } finally {
        combined.dispose();
      }
      throw abortReason(combined.signal.reason);
    }

    if (!response.ok) {
      combined.dispose();
      try {
        await response.body?.cancel();
      } catch {
        // There is no useful response body after an HTTP failure.
      }
      const retryAfterMs = parseRetryAfter(
        response.headers.get("retry-after"),
        this.now(),
      );
      throw new ModelTransportError({
        url,
        phase: "response",
        attempt: this.options.attempt,
        status: response.status,
        ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
        retryable: isRetryableHttpStatus(response.status),
        detail: response.statusText || "request was not successful",
      });
    }

    if (response.body === null) {
      combined.dispose();
      return response;
    }
    return this.wrapBody(response, url, combined);
  }

  private wrapBody(
    response: Response,
    url: string,
    combined: CombinedAbortSignal,
  ): Response {
    const source = response.body!.getReader();
    let finished = false;
    let cancelled = false;
    let cancellationReason: unknown;
    const finish = (): void => {
      if (finished) return;
      finished = true;
      this.activeBodies.delete(activeBody);
      combined.dispose();
      source.releaseLock();
    };
    const activeBody: ActiveBody = {
      cancel: async (reason) => {
        if (finished || cancelled) return;
        cancelled = true;
        cancellationReason = reason;
        try {
          await source.cancel(reason);
        } finally {
          finish();
        }
      },
    };
    this.activeBodies.add(activeBody);

    const body = new ReadableStream<Uint8Array>({
      pull: async (controller) => {
        try {
          const item = await source.read();
          if (cancelled) {
            controller.error(abortReason(cancellationReason));
            return;
          }
          if (item.done) {
            finish();
            controller.close();
            return;
          }
          controller.enqueue(item.value);
        } catch (error) {
          if (!finished) finish();
          if (combined.signal.aborted) {
            controller.error(abortReason(combined.signal.reason));
            return;
          }
          controller.error(
            modelBodyFailure(url, error, this.options.attempt),
          );
        }
      },
      cancel: async (reason) => {
        if (finished) return;
        cancelled = true;
        cancellationReason = reason;
        try {
          await source.cancel(reason);
        } finally {
          finish();
        }
      },
    });
    return copyResponse(response, body);
  }
}

export function modelBodyFailure(
  url: string,
  cause: unknown,
  attempt?: number,
): ModelTransportError {
  if (cause instanceof ModelTransportError) return cause;
  return new ModelTransportError({
    url,
    phase: "body",
    ...(attempt === undefined ? {} : { attempt }),
    retryable: true,
    detail: errorMessage(cause),
    cause,
  });
}

export function truncatedModelBody(
  url: string,
  actualBytes: number,
  expectedBytes: number,
): ModelTransportError {
  return new ModelTransportError({
    url,
    phase: "body",
    retryable: true,
    detail: `response ended after ${actualBytes} bytes; expected ${expectedBytes}`,
  });
}

export function isRetryableHttpStatus(status: number): boolean {
  return (
    status === 408 ||
    status === 425 ||
    status === 429 ||
    (status >= 500 && status <= 599)
  );
}

function parseRetryAfter(
  value: string | null,
  now: number,
): number | undefined {
  if (value === null) return undefined;
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) {
    const seconds = Number(trimmed);
    return Number.isSafeInteger(seconds) ? seconds * 1_000 : undefined;
  }
  const date = Date.parse(trimmed);
  return Number.isFinite(date) ? Math.max(0, date - now) : undefined;
}

interface CombinedAbortSignal {
  readonly signal: AbortSignal;
  dispose(): void;
}

function combineAbortSignals(
  values: readonly (AbortSignal | null | undefined)[],
): CombinedAbortSignal {
  const signals = values.filter(
    (value): value is AbortSignal => value !== undefined && value !== null,
  );
  if (signals.length === 1) {
    return { signal: signals[0]!, dispose: () => undefined };
  }

  const controller = new AbortController();
  const listeners: Array<{
    readonly signal: AbortSignal;
    readonly listener: () => void;
  }> = [];
  for (const signal of signals) {
    const listener = (): void => {
      if (!controller.signal.aborted) controller.abort(signal.reason);
    };
    if (signal.aborted) {
      listener();
      break;
    }
    signal.addEventListener("abort", listener, { once: true });
    listeners.push({ signal, listener });
  }
  return {
    signal: controller.signal,
    dispose: () => {
      for (const { signal, listener } of listeners) {
        signal.removeEventListener("abort", listener);
      }
    },
  };
}

function copyResponse(
  response: Response,
  body: ReadableStream<Uint8Array>,
): Response {
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

function requestUrl(input: RequestInfo | URL): string {
  const raw = input instanceof Request ? input.url : input.toString();
  return new URL(
    raw,
    globalThis.location?.href ?? "http://localhost/",
  ).href;
}

function abortReason(reason: unknown): unknown {
  return reason ?? new DOMException("The operation was aborted", "AbortError");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
