import {
  ModelFetchAttempt,
  ModelTransportError,
} from "./model-transport";

const DEFAULT_MAX_ATTEMPTS = 4;
const MAX_MAX_ATTEMPTS = 4;
const DEFAULT_BASE_DELAY_MS = 500;
const DEFAULT_MAX_DELAY_MS = 8_000;
const DEFAULT_MAX_RETRY_AFTER_MS = 30_000;

export interface ModelLoadAttempt {
  readonly number: number;
  readonly fetch: typeof fetch;
  cancel(reason: unknown): Promise<void>;
}

export interface ModelLoadRetryEvent {
  readonly attempt: number;
  readonly nextAttempt: number;
  readonly maxAttempts: number;
  readonly delayMs: number;
  readonly error: ModelTransportError;
}

export interface ModelLoadRetryOptions {
  readonly fetch?: typeof fetch;
  readonly signal?: AbortSignal;
  readonly maxAttempts?: number;
  readonly baseDelayMs?: number;
  readonly maxDelayMs?: number;
  readonly maxRetryAfterMs?: number;
  readonly random?: () => number;
  readonly now?: () => number;
  readonly sleep?: (delayMs: number, signal?: AbortSignal) => Promise<void>;
  readonly cleanup?: (error: unknown) => Promise<void>;
  readonly recover?: (error: unknown) => boolean;
  readonly onRetry?: (event: ModelLoadRetryEvent) => void;
}

export class ModelLoadRetryExhaustedError extends Error {
  readonly attempts: number;
  readonly lastError: ModelTransportError;

  constructor(attempts: number, error: ModelTransportError) {
    const status =
      error.status === undefined ? "" : ` (HTTP ${error.status})`;
    const detail =
      error.detail === undefined ? "" : `: ${error.detail}`;
    super(
      `Model loading failed after ${attempts} attempts; last failure was during ` +
        `${error.phase} for ${error.url}${status}${detail}`,
      { cause: error },
    );
    this.name = "ModelLoadRetryExhaustedError";
    this.attempts = attempts;
    this.lastError = error;
  }
}

/**
 * Runs an entire BrowserModelSet load attempt again only for explicitly
 * classified transient transport failures.
 */
export async function runModelLoadWithRetry<T>(
  operation: (attempt: ModelLoadAttempt) => Promise<T>,
  options: ModelLoadRetryOptions = {},
): Promise<T> {
  const policy = normalizedPolicy(options);
  const networkFetch =
    options.fetch ?? globalThis.fetch.bind(globalThis);
  let attemptNumber = 1;

  while (true) {
    throwIfAborted(options.signal);
    const fetchAttempt = new ModelFetchAttempt(networkFetch, {
      attempt: attemptNumber,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      now: policy.now,
    });
    const attempt: ModelLoadAttempt = {
      number: attemptNumber,
      fetch: fetchAttempt.fetch,
      cancel: (reason) => fetchAttempt.cancel(reason),
    };
    try {
      return await operation(attempt);
    } catch (error) {
      await fetchAttempt.cancel(error);
      await options.cleanup?.(error);
      throwIfAborted(options.signal);

      if (options.recover?.(error) === true) continue;
      if (!(error instanceof ModelTransportError) || !error.retryable) {
        throw error;
      }
      if (attemptNumber >= policy.maxAttempts) {
        throw new ModelLoadRetryExhaustedError(attemptNumber, error);
      }

      const delayMs = retryDelayMs(error, attemptNumber, policy);
      options.onRetry?.({
        attempt: attemptNumber,
        nextAttempt: attemptNumber + 1,
        maxAttempts: policy.maxAttempts,
        delayMs,
        error,
      });
      await policy.sleep(delayMs, options.signal);
      attemptNumber += 1;
    }
  }
}

interface NormalizedPolicy {
  readonly maxAttempts: number;
  readonly baseDelayMs: number;
  readonly maxDelayMs: number;
  readonly maxRetryAfterMs: number;
  readonly random: () => number;
  readonly now: () => number;
  readonly sleep: (delayMs: number, signal?: AbortSignal) => Promise<void>;
}

function normalizedPolicy(
  options: ModelLoadRetryOptions,
): NormalizedPolicy {
  const maxAttempts = integerInRange(
    options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
    "maxAttempts",
    1,
    MAX_MAX_ATTEMPTS,
  );
  const baseDelayMs = nonNegativeFinite(
    options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS,
    "baseDelayMs",
  );
  const maxDelayMs = nonNegativeFinite(
    options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS,
    "maxDelayMs",
  );
  const maxRetryAfterMs = nonNegativeFinite(
    options.maxRetryAfterMs ?? DEFAULT_MAX_RETRY_AFTER_MS,
    "maxRetryAfterMs",
  );
  if (maxDelayMs < baseDelayMs) {
    throw new RangeError("maxDelayMs must be greater than or equal to baseDelayMs");
  }
  return {
    maxAttempts,
    baseDelayMs,
    maxDelayMs,
    maxRetryAfterMs,
    random: options.random ?? Math.random,
    now: options.now ?? Date.now,
    sleep: options.sleep ?? abortableSleep,
  };
}

function retryDelayMs(
  error: ModelTransportError,
  attempt: number,
  policy: NormalizedPolicy,
): number {
  const exponential = Math.min(
    policy.maxDelayMs,
    policy.baseDelayMs * 2 ** (attempt - 1),
  );
  const random = Math.min(1, Math.max(0, policy.random()));
  const jittered = exponential * (0.5 + random * 0.5);
  const retryAfter =
    error.retryAfterMs === undefined
      ? 0
      : Math.min(error.retryAfterMs, policy.maxRetryAfterMs);
  return Math.round(Math.max(jittered, retryAfter));
}

async function abortableSleep(
  delayMs: number,
  signal?: AbortSignal,
): Promise<void> {
  throwIfAborted(signal);
  if (delayMs === 0) return;
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(finish, delayMs);
    const abort = (): void => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
      reject(signal?.reason ?? new DOMException("The operation was aborted", "AbortError"));
    };
    function finish(): void {
      signal?.removeEventListener("abort", abort);
      resolve();
    }
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}

function integerInRange(
  value: number,
  name: string,
  minimum: number,
  maximum: number,
): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function nonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative finite number`);
  }
  return value;
}
