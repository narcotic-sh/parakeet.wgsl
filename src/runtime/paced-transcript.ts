import type {
  TranscriptionResult,
  TranscriptionSnapshot,
  TranscriptionWord,
} from "./protocol.js";

// Thirty presentation updates per second remain visually continuous while
// bounding DOM/layout competition with the GPU inference workload.
export const PACED_TRANSCRIPT_FRAME_MILLISECONDS = 1000 / 30;
export const PACED_TRANSCRIPT_EWMA_ALPHA = 0.35;
export const PACED_TRANSCRIPT_MIN_INTERVAL_MILLISECONDS = 100;
export const PACED_TRANSCRIPT_MAX_INTERVAL_MILLISECONDS = 30_000;

export interface PacedTranscriptTextSplice {
  /** UTF-16 code-unit offset in the text before this update. */
  readonly startOffset: number;
  readonly deleteCount: number;
  readonly insertText: string;
}

export interface PacedTranscriptWord extends TranscriptionWord {
  /** Half-open UTF-16 code-unit range after this update. */
  readonly startOffset: number;
  readonly endOffset: number;
}

export interface PacedTranscriptWordSplice {
  readonly startIndex: number;
  readonly deleteCount: number;
  readonly insertWords: readonly PacedTranscriptWord[];
}

/**
 * One presentation-paced mutation of the derived transcript.
 *
 * `revision` belongs to this paced delivery track. Several paced revisions
 * can share one authoritative raw `sourceRevision`. The terminal
 * reconciliation uses the next synthetic source revision.
 */
export interface PacedTranscriptUpdate {
  readonly revision: number;
  readonly sourceRevision: number;
  readonly textSplice: PacedTranscriptTextSplice;
  readonly wordSplice: PacedTranscriptWordSplice;
  readonly audioDurationSeconds: number;
  readonly processedAudioSeconds: number;
  readonly isFinal: boolean;
}

export interface PacedTranscriptScheduler {
  now(): number;
  /**
   * Schedule callback asynchronously after at least delayMilliseconds.
   * The returned handle must not be undefined.
   */
  schedule(
    callback: () => void,
    delayMilliseconds: number,
  ): object | number;
  cancel(handle: object | number): void;
}

interface CompiledWord {
  readonly segmentText: string;
  readonly word: PacedTranscriptWord;
}

interface CompiledTarget {
  readonly sourceRevision: number;
  readonly text: string;
  readonly words: readonly CompiledWord[];
  readonly audioDurationSeconds: number;
  readonly processedAudioSeconds: number;
}

interface FinalCompletion {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
  readonly reject: (reason: unknown) => void;
}

export interface PacedTranscriptDeliveryOptions {
  readonly onUpdate: (update: PacedTranscriptUpdate) => void;
  readonly onError?: (error: unknown) => void;
  readonly scheduler?: PacedTranscriptScheduler;
}

/**
 * Derives a smooth, timestamped word-delivery track from authoritative raw
 * snapshots. It never changes inference cadence or requests additional
 * transcript snapshots; a subscriber enables one primary-complete boundary
 * signal.
 */
export class PacedTranscriptDelivery {
  private readonly onUpdate: (update: PacedTranscriptUpdate) => void;
  private readonly onError: ((error: unknown) => void) | undefined;
  private readonly scheduler: PacedTranscriptScheduler;
  private readonly startedAt: number;

  private target: CompiledTarget | undefined;
  private displayedText = "";
  private displayedWords: CompiledWord[] = [];
  private lastSourceRevision = 0;
  private presentationRevision = 0;
  private lastSourceArrivalAt: number | undefined;
  private estimatedIntervalMs: number | undefined;
  private cycleStartedAt = 0;
  private cycleStartWordCount = 0;
  private cycleTargetWordCount = 0;
  private cycleDurationMs = 0;
  private scheduledFrame: object | number | undefined;
  private immediateDelivery = false;
  private finalRequested = false;
  private finalCompletion: FinalCompletion | undefined;
  private closed = false;
  private failed = false;
  private failure: unknown | undefined;
  private lastClockAt: number;

  constructor(options: PacedTranscriptDeliveryOptions) {
    this.onUpdate = options.onUpdate;
    this.onError = options.onError;
    this.scheduler = options.scheduler ?? defaultPacedTranscriptScheduler;
    this.startedAt = finiteNow(this.scheduler.now());
    this.lastClockAt = this.startedAt;
  }

  get pendingWordCount(): number {
    return Math.max(
      0,
      (this.target?.words.length ?? 0) - this.displayedWords.length,
    );
  }

  get sourceIntervalEstimateMilliseconds(): number | undefined {
    return this.estimatedIntervalMs;
  }

  get hasScheduledFrame(): boolean {
    return this.scheduledFrame !== undefined;
  }

  pushSnapshot(
    snapshot: TranscriptionSnapshot,
    arrivalAtMilliseconds?: number,
  ): void {
    if (
      this.closed ||
      this.finalRequested ||
      snapshot.revision <= this.lastSourceRevision
    ) {
      return;
    }
    try {
      const now = this.immediateDelivery
        ? this.lastClockAt
        : this.readClock(arrivalAtMilliseconds);
      if (!this.immediateDelivery) {
        this.observeSourceArrival(now);
      }
      this.lastSourceRevision = snapshot.revision;
      this.setTarget(
        compileTarget({
          sourceRevision: snapshot.revision,
          text: snapshot.text,
          words: snapshot.words,
          audioDurationSeconds: snapshot.audioDurationSeconds,
          processedAudioSeconds: snapshot.processedAudioSeconds,
        }),
        now,
      );
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * Reveal the complete latest primary target and make every later repair or
   * final target immediate. This is idempotent and also works before the first
   * source snapshot arrives.
   */
  flushPendingAndDisablePacing(): void {
    if (this.closed || this.finalRequested || this.immediateDelivery) return;
    this.immediateDelivery = true;
    this.cancelScheduledFrame();
    const target = this.target;
    if (target === undefined) return;
    try {
      this.revealTargetFully(target, false);
    } catch (error) {
      this.fail(error);
    }
  }

  finish(result: TranscriptionResult): Promise<void> {
    if (this.finalCompletion !== undefined) {
      return this.finalCompletion.promise;
    }
    if (this.failed) {
      return Promise.reject(this.failure);
    }
    if (this.closed) return Promise.resolve();

    let resolveCompletion!: () => void;
    let rejectCompletion!: (reason: unknown) => void;
    const promise = new Promise<void>((resolve, reject) => {
      resolveCompletion = resolve;
      rejectCompletion = reject;
    });
    this.finalCompletion = {
      promise,
      resolve: resolveCompletion,
      reject: rejectCompletion,
    };
    this.finalRequested = true;
    this.immediateDelivery = true;
    this.cancelScheduledFrame();

    try {
      const audioDurationSeconds =
        result.metrics.audioDurationSeconds;
      const sourceRevision = Math.max(1, this.lastSourceRevision + 1);
      this.lastSourceRevision = sourceRevision;
      const target = compileTarget({
        sourceRevision,
        text: result.text,
        words: result.words,
        audioDurationSeconds,
        processedAudioSeconds: audioDurationSeconds,
      });
      this.target = target;
      this.revealTargetFully(target, true);
      if (!this.closed && this.target === target) {
        this.closed = true;
        this.target = undefined;
        this.finalCompletion.resolve();
      }
    } catch (error) {
      this.fail(error);
    }
    return promise;
  }

  cancel(reason: unknown = abortError()): void {
    if (this.closed) return;
    this.failed = true;
    this.failure = reason;
    this.closed = true;
    this.cancelScheduledFrame();
    this.target = undefined;
    this.displayedText = "";
    this.displayedWords = [];
    this.finalCompletion?.reject(reason);
  }

  private observeSourceArrival(now: number): void {
    const observed =
      this.lastSourceArrivalAt === undefined
        ? now - this.startedAt
        : now - this.lastSourceArrivalAt;
    const bounded = clampSourceInterval(observed);
    this.estimatedIntervalMs =
      this.estimatedIntervalMs === undefined
        ? bounded
        : this.estimatedIntervalMs +
          PACED_TRANSCRIPT_EWMA_ALPHA *
            (bounded - this.estimatedIntervalMs);
    this.lastSourceArrivalAt = now;
  }

  private setTarget(target: CompiledTarget, now: number): void {
    this.target = target;
    if (this.immediateDelivery) {
      this.cancelScheduledFrame();
      this.revealTargetFully(target, false);
      return;
    }
    const visibleTargetWordCount = targetWordCountAtVisibleFrontier(
      this.displayedWords,
      target.words,
    );
    const commonWordCount = commonCompiledWordPrefix(
      this.displayedWords,
      target.words,
      visibleTargetWordCount,
    );
    const visibleTextEndOffset =
      visibleTargetWordCount === 0
        ? 0
        : target.words[visibleTargetWordCount - 1]!.word.endOffset;
    const visibleText =
      target.words.length === 0
        ? target.text
        : target.text.slice(0, visibleTextEndOffset);
    if (
      commonWordCount < this.displayedWords.length ||
      commonWordCount < visibleTargetWordCount ||
      this.displayedText !== visibleText
    ) {
      const startOffset =
        commonWordCount === 0
          ? 0
          : target.words[commonWordCount - 1]!.word.endOffset;
      if (
        this.displayedText.slice(0, startOffset) !==
        target.text.slice(0, startOffset)
      ) {
        throw new Error(
          "Paced transcript stable prefix does not match its source",
        );
      }
      this.emitSplice({
        target,
        textSplice: {
          startOffset,
          deleteCount: this.displayedText.length - startOffset,
          insertText: visibleText.slice(startOffset),
        },
        wordSplice: {
          startIndex: commonWordCount,
          deleteCount:
            this.displayedWords.length - commonWordCount,
          insertWords: target.words
            .slice(commonWordCount, visibleTargetWordCount)
            .map(({ word }) => word),
        },
        replacementWords: target.words.slice(
          commonWordCount,
          visibleTargetWordCount,
        ),
        isFinal: false,
      });
      if (this.closed || this.target !== target) return;
    }

    if (
      this.displayedText !==
      target.text.slice(0, this.displayedText.length)
    ) {
      throw new Error(
        "Paced transcript target does not preserve its visible prefix",
      );
    }
    this.startDeliveryCycle(now);
  }

  private startDeliveryCycle(now: number): void {
    this.cancelScheduledFrame();
    const target = this.target;
    if (
      target === undefined ||
      this.displayedWords.length >= target.words.length
    ) {
      this.cancelScheduledFrame();
      return;
    }
    this.cycleStartedAt = now;
    this.cycleStartWordCount = this.displayedWords.length;
    this.cycleTargetWordCount = target.words.length;
    this.cycleDurationMs =
      this.estimatedIntervalMs ??
      PACED_TRANSCRIPT_MIN_INTERVAL_MILLISECONDS;
    this.scheduleNextFrame(now);
  }

  private scheduleNextFrame(now: number): void {
    if (this.closed || this.scheduledFrame !== undefined) return;
    const cycleWordCount =
      this.cycleTargetWordCount - this.cycleStartWordCount;
    const deliveredCycleWords =
      this.displayedWords.length - this.cycleStartWordCount;
    if (
      cycleWordCount <= 0 ||
      deliveredCycleWords >= cycleWordCount
    ) {
      return;
    }
    const nextDeadline =
      this.cycleStartedAt +
      (this.cycleDurationMs * (deliveredCycleWords + 1)) /
        cycleWordCount;
    const delayMilliseconds = Math.max(
      PACED_TRANSCRIPT_FRAME_MILLISECONDS,
      nextDeadline - now,
    );
    let invokedSynchronously = false;
    let scheduling = true;
    let scheduledHandle: object | number | undefined;
    const handle = this.scheduler.schedule(() => {
      if (scheduling) {
        invokedSynchronously = true;
        return;
      }
      if (
        scheduledHandle === undefined ||
        this.scheduledFrame !== scheduledHandle
      ) {
        return;
      }
      this.scheduledFrame = undefined;
      this.deliverFrame();
    }, delayMilliseconds);
    scheduling = false;
    if (invokedSynchronously) {
      throw new Error(
        "Paced transcript scheduler invoked its callback synchronously",
      );
    }
    if (handle === undefined) {
      throw new Error(
        "Paced transcript scheduler returned an undefined handle",
      );
    }
    scheduledHandle = handle;
    this.scheduledFrame = handle;
  }

  private deliverFrame(): void {
    if (this.closed) return;
    const target = this.target;
    if (target === undefined) return;

    try {
      const now = this.readClock();
      const elapsedMs = Math.max(0, now - this.cycleStartedAt);
      const fraction = Math.min(
        1,
        elapsedMs / Math.max(1, this.cycleDurationMs),
      );
      const cycleWords =
        this.cycleTargetWordCount - this.cycleStartWordCount;
      const dueWordCount =
        this.cycleStartWordCount +
        Math.floor(cycleWords * fraction);
      const endWordCount = Math.min(
        target.words.length,
        Math.max(this.displayedWords.length, dueWordCount),
      );
      if (endWordCount > this.displayedWords.length) {
        this.appendTargetWords(target, endWordCount);
      }
      if (this.closed) return;
      if (this.displayedWords.length < target.words.length) {
        this.scheduleNextFrame(now);
      }
    } catch (error) {
      this.fail(error);
    }
  }

  private appendTargetWords(
    target: CompiledTarget,
    endWordCount: number,
  ): void {
    const startWordCount = this.displayedWords.length;
    const insertEntries = target.words.slice(
      startWordCount,
      endWordCount,
    );
    if (insertEntries.length === 0) return;
    const endOffset = insertEntries.at(-1)!.word.endOffset;
    this.emitSplice({
      target,
      textSplice: {
        startOffset: this.displayedText.length,
        deleteCount: 0,
        insertText: target.text.slice(
          this.displayedText.length,
          endOffset,
        ),
      },
      wordSplice: {
        startIndex: startWordCount,
        deleteCount: 0,
        insertWords: insertEntries.map(({ word }) => word),
      },
      replacementWords: insertEntries,
      isFinal: false,
    });
  }

  private revealTargetFully(
    target: CompiledTarget,
    isFinal: boolean,
  ): void {
    const commonWordCount = commonCompiledWordPrefix(
      this.displayedWords,
      target.words,
    );
    const startOffset =
      commonWordCount === 0
        ? 0
        : target.words[commonWordCount - 1]!.word.endOffset;
    if (
      this.displayedText.slice(0, startOffset) !==
      target.text.slice(0, startOffset)
    ) {
      throw new Error(
        "Paced transcript stable prefix does not match its source",
      );
    }
    const insertText = target.text.slice(startOffset);
    const replacementWords = target.words.slice(commonWordCount);
    if (
      !isFinal &&
      this.displayedText.length - startOffset === insertText.length &&
      this.displayedText.slice(startOffset) === insertText &&
      commonWordCount === this.displayedWords.length &&
      commonWordCount === target.words.length
    ) {
      return;
    }
    this.emitSplice({
      target,
      textSplice: {
        startOffset,
        deleteCount: this.displayedText.length - startOffset,
        insertText,
      },
      wordSplice: {
        startIndex: commonWordCount,
        deleteCount: this.displayedWords.length - commonWordCount,
        insertWords: replacementWords.map(({ word }) => word),
      },
      replacementWords,
      isFinal,
    });
  }

  private emitSplice(input: {
    readonly target: CompiledTarget;
    readonly textSplice: PacedTranscriptTextSplice;
    readonly wordSplice: PacedTranscriptWordSplice;
    readonly replacementWords: readonly CompiledWord[];
    readonly isFinal: boolean;
  }): void {
    const {
      target,
      textSplice,
      wordSplice,
      replacementWords,
      isFinal,
    } = input;
    validateTextSplice(textSplice, this.displayedText.length);
    validateWordSplice(wordSplice, this.displayedWords.length);
    if (
      textSplice.startOffset === this.displayedText.length &&
      textSplice.deleteCount === 0
    ) {
      this.displayedText += textSplice.insertText;
    } else {
      this.displayedText =
        this.displayedText.slice(0, textSplice.startOffset) +
        textSplice.insertText +
        this.displayedText.slice(
          textSplice.startOffset + textSplice.deleteCount,
        );
    }
    applyCompiledWordSplice(
      this.displayedWords,
      wordSplice,
      replacementWords,
    );
    this.presentationRevision += 1;
    this.onUpdate({
      revision: this.presentationRevision,
      sourceRevision: target.sourceRevision,
      textSplice,
      wordSplice,
      audioDurationSeconds: target.audioDurationSeconds,
      processedAudioSeconds: target.processedAudioSeconds,
      isFinal,
    });
  }

  private cancelScheduledFrame(): void {
    const scheduled = this.scheduledFrame;
    if (scheduled === undefined) return;
    this.scheduledFrame = undefined;
    this.scheduler.cancel(scheduled);
  }

  private fail(error: unknown): void {
    if (this.closed) return;
    const failure =
      error === undefined
        ? new Error("Paced transcript delivery failed")
        : error;
    this.failed = true;
    this.failure = failure;
    this.closed = true;
    this.cancelScheduledFrame();
    this.target = undefined;
    this.displayedText = "";
    this.displayedWords = [];
    this.finalCompletion?.reject(failure);
    try {
      this.onError?.(failure);
    } catch {
      // Preserve the original delivery failure. onError is diagnostic only.
    }
  }

  private readClock(value = this.scheduler.now()): number {
    const now = finiteNow(value);
    if (now < this.lastClockAt) {
      throw new RangeError(
        "Paced transcript scheduler clock moved backwards",
      );
    }
    this.lastClockAt = now;
    return now;
  }
}

const defaultPacedTranscriptScheduler: PacedTranscriptScheduler = {
  now: () => performance.now(),
  schedule: (callback, delayMilliseconds) =>
    setTimeout(callback, delayMilliseconds),
  cancel: (handle) =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};

function compileTarget(input: {
  readonly sourceRevision: number;
  readonly text: string;
  readonly words: readonly TranscriptionWord[];
  readonly audioDurationSeconds: number;
  readonly processedAudioSeconds: number;
}): CompiledTarget {
  if (
    !Number.isSafeInteger(input.sourceRevision) ||
    input.sourceRevision < 1
  ) {
    throw new RangeError(
      "Paced transcript source revision must be a positive integer",
    );
  }
  validateProgressSeconds(
    input.audioDurationSeconds,
    input.processedAudioSeconds,
  );
  const words: CompiledWord[] = [];
  let cursor = 0;
  let previousWordEnd = 0;
  let previousStartSeconds = 0;
  for (const sourceWord of input.words) {
    if (sourceWord.text.length === 0) {
      throw new RangeError("Paced transcript words must not be empty");
    }
    if (
      !Number.isFinite(sourceWord.startSeconds) ||
      !Number.isFinite(sourceWord.endSeconds) ||
      sourceWord.startSeconds < 0 ||
      sourceWord.endSeconds < sourceWord.startSeconds ||
      sourceWord.startSeconds < previousStartSeconds
    ) {
      throw new RangeError(
        "Paced transcript word timestamps must be finite and ordered",
      );
    }
    const startOffset = input.text.indexOf(sourceWord.text, cursor);
    if (startOffset < cursor) {
      throw new Error(
        `Could not locate paced transcript word ${JSON.stringify(sourceWord.text)}`,
      );
    }
    const endOffset = startOffset + sourceWord.text.length;
    const word: PacedTranscriptWord = {
      text: sourceWord.text,
      startSeconds: sourceWord.startSeconds,
      endSeconds: sourceWord.endSeconds,
      startOffset,
      endOffset,
    };
    words.push({
      segmentText: input.text.slice(previousWordEnd, endOffset),
      word,
    });
    cursor = endOffset;
    previousWordEnd = endOffset;
    previousStartSeconds = sourceWord.startSeconds;
  }
  return {
    sourceRevision: input.sourceRevision,
    text: input.text,
    words,
    audioDurationSeconds: input.audioDurationSeconds,
    processedAudioSeconds: input.processedAudioSeconds,
  };
}

function targetWordCountAtVisibleFrontier(
  displayedWords: readonly CompiledWord[],
  targetWords: readonly CompiledWord[],
): number {
  const displayedLast = displayedWords.at(-1);
  if (displayedLast === undefined) return 0;

  const frontierSeconds = displayedLast.word.endSeconds;
  let visibleCount = 0;
  while (
    visibleCount < targetWords.length &&
    targetWords[visibleCount]!.word.startSeconds < frontierSeconds
  ) {
    visibleCount += 1;
  }

  // Timestamp boundaries are the primary frontier. Preserve an exactly
  // surviving last visible word as a guard for zero-duration or equal-boundary
  // tokens, while still allowing insertions/deletions before it.
  for (let index = visibleCount; index < targetWords.length; index += 1) {
    if (sameWordIdentity(displayedLast, targetWords[index]!)) {
      return index + 1;
    }
  }
  return visibleCount;
}

function commonCompiledWordPrefix(
  left: readonly CompiledWord[],
  right: readonly CompiledWord[],
  rightLimit = right.length,
): number {
  const limit = Math.min(left.length, right.length, rightLimit);
  let index = 0;
  while (index < limit && sameCompiledWord(left[index]!, right[index]!)) {
    index += 1;
  }
  return index;
}

function applyCompiledWordSplice(
  words: CompiledWord[],
  splice: PacedTranscriptWordSplice,
  replacementWords: readonly CompiledWord[],
): void {
  const oldLength = words.length;
  const insertCount = replacementWords.length;
  const removedEnd = splice.startIndex + splice.deleteCount;
  const tailCount = oldLength - removedEnd;
  const lengthDelta = insertCount - splice.deleteCount;

  if (lengthDelta > 0) {
    words.length = oldLength + lengthDelta;
    for (let index = tailCount - 1; index >= 0; index -= 1) {
      words[removedEnd + lengthDelta + index] =
        words[removedEnd + index]!;
    }
  } else if (lengthDelta < 0) {
    for (let index = 0; index < tailCount; index += 1) {
      words[splice.startIndex + insertCount + index] =
        words[removedEnd + index]!;
    }
    words.length = oldLength + lengthDelta;
  }
  for (let index = 0; index < insertCount; index += 1) {
    words[splice.startIndex + index] = replacementWords[index]!;
  }
}

function sameCompiledWord(
  left: CompiledWord,
  right: CompiledWord,
): boolean {
  return (
    left.segmentText === right.segmentText &&
    sameWordIdentity(left, right)
  );
}

function sameWordIdentity(
  left: CompiledWord,
  right: CompiledWord,
): boolean {
  return (
    left.word.text === right.word.text &&
    left.word.startSeconds === right.word.startSeconds &&
    left.word.endSeconds === right.word.endSeconds
  );
}

function validateTextSplice(
  splice: PacedTranscriptTextSplice,
  textLength: number,
): void {
  if (
    !Number.isSafeInteger(splice.startOffset) ||
    !Number.isSafeInteger(splice.deleteCount) ||
    splice.startOffset < 0 ||
    splice.deleteCount < 0 ||
    splice.startOffset + splice.deleteCount > textLength
  ) {
    throw new RangeError("Invalid paced transcript text splice");
  }
}

function validateWordSplice(
  splice: PacedTranscriptWordSplice,
  wordCount: number,
): void {
  if (
    !Number.isSafeInteger(splice.startIndex) ||
    !Number.isSafeInteger(splice.deleteCount) ||
    splice.startIndex < 0 ||
    splice.deleteCount < 0 ||
    splice.startIndex + splice.deleteCount > wordCount
  ) {
    throw new RangeError("Invalid paced transcript word splice");
  }
}

function clampSourceInterval(value: number): number {
  const finite =
    Number.isFinite(value) && value > 0
      ? value
      : PACED_TRANSCRIPT_MIN_INTERVAL_MILLISECONDS;
  return Math.min(
    PACED_TRANSCRIPT_MAX_INTERVAL_MILLISECONDS,
    Math.max(PACED_TRANSCRIPT_MIN_INTERVAL_MILLISECONDS, finite),
  );
}

function finiteNow(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("Paced transcript scheduler returned invalid time");
  }
  return value;
}

function validateProgressSeconds(
  audioDurationSeconds: number,
  processedAudioSeconds: number,
): void {
  if (
    !Number.isFinite(audioDurationSeconds) ||
    !Number.isFinite(processedAudioSeconds) ||
    audioDurationSeconds < 0 ||
    processedAudioSeconds < 0
  ) {
    throw new RangeError(
      "Paced transcript progress seconds must be finite and nonnegative",
    );
  }
}

function abortError(): Error {
  return new DOMException("The operation was aborted", "AbortError");
}
