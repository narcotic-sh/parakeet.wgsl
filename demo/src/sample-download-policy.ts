export const SAMPLE_AUDIO_MAX_DOWNLOAD_ATTEMPTS = 4;

const BASE_DELAY_MILLISECONDS = 500;
const MAX_DELAY_MILLISECONDS = 8_000;
const MAX_RETRY_AFTER_MILLISECONDS = 30_000;

export function retryableSampleStatus(status: number): boolean {
  return (
    status === 408 ||
    status === 425 ||
    status === 429 ||
    (status >= 500 && status <= 599)
  );
}

export function parseRetryAfterMilliseconds(
  value: string | null,
  now = Date.now(),
): number {
  if (value === null) return 0;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1_000;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? Math.max(0, timestamp - now) : 0;
}

export function sampleRetryDelayMilliseconds(
  failedAttempt: number,
  retryAfterMilliseconds: number,
  random = Math.random,
): number {
  const exponential = Math.min(
    MAX_DELAY_MILLISECONDS,
    BASE_DELAY_MILLISECONDS * 2 ** Math.max(0, failedAttempt - 1),
  );
  const sample = random();
  const boundedRandom = Number.isFinite(sample)
    ? Math.min(1, Math.max(0, sample))
    : 0;
  const jittered = exponential * (0.5 + boundedRandom * 0.5);
  const retryAfter = Number.isFinite(retryAfterMilliseconds)
    ? Math.min(
        Math.max(0, retryAfterMilliseconds),
        MAX_RETRY_AFTER_MILLISECONDS,
      )
    : 0;
  return Math.round(Math.max(jittered, retryAfter));
}
