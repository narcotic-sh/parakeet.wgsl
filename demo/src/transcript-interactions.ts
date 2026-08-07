export interface TimedTranscriptWord {
  readonly startSeconds: number;
  readonly endSeconds: number;
}

export interface TranscriptScrollPosition {
  readonly scrollTop: number;
  readonly scrollHeight: number;
  readonly clientHeight: number;
}

const DEFAULT_BOTTOM_TOLERANCE_PIXELS = 4;

/** Return the latest-starting word active at one playback timestamp. */
export function transcriptWordAtTime<T extends TimedTranscriptWord>(
  words: readonly T[],
  seconds: number,
): T | undefined {
  if (!Number.isFinite(seconds) || seconds < 0) return undefined;

  let low = 0;
  let high = words.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (words[middle]!.startSeconds <= seconds) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }

  const candidate = words[low - 1];
  return candidate !== undefined && seconds < candidate.endSeconds
    ? candidate
    : undefined;
}

/**
 * Whether a user-controlled transcript viewport has reached its real bottom.
 * A non-overflowing viewport deliberately does not arm live tail following.
 */
export function isTranscriptTailReached(
  position: TranscriptScrollPosition,
  tolerancePixels = DEFAULT_BOTTOM_TOLERANCE_PIXELS,
): boolean {
  const { scrollTop, scrollHeight, clientHeight } = position;
  if (
    !Number.isFinite(scrollTop) ||
    !Number.isFinite(scrollHeight) ||
    !Number.isFinite(clientHeight) ||
    !Number.isFinite(tolerancePixels) ||
    scrollTop < 0 ||
    scrollHeight < 0 ||
    clientHeight < 0 ||
    tolerancePixels < 0
  ) {
    return false;
  }

  const bottomScrollTop = scrollHeight - clientHeight;
  return (
    bottomScrollTop > 0 &&
    scrollTop <= bottomScrollTop + tolerancePixels &&
    bottomScrollTop - scrollTop <= tolerancePixels
  );
}
