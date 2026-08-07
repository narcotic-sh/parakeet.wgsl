import { describe, expect, it } from "vitest";

import {
  PACED_TRANSCRIPT_EWMA_ALPHA,
  PacedTranscriptDelivery,
  type PacedTranscriptScheduler,
  type PacedTranscriptUpdate,
  type PacedTranscriptWord,
} from "../../src/runtime/paced-transcript";
import type {
  TranscriptionMetrics,
  TranscriptionResult,
  TranscriptionSnapshot,
  TranscriptionWord,
} from "../../src/runtime/protocol";

describe("PacedTranscriptDelivery", () => {
  it("seeds its interval from the first arrival and updates the EWMA", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(1_000);
    delivery.pushSnapshot(
      snapshot(1, "one two", ["one", "two"], 20),
    );
    expect(delivery.sourceIntervalEstimateMilliseconds).toBe(1_000);

    scheduler.advance(600);
    delivery.pushSnapshot(
      snapshot(2, "one two three", ["one", "two", "three"], 30),
    );
    expect(delivery.sourceIntervalEstimateMilliseconds).toBe(
      1_000 +
        PACED_TRANSCRIPT_EWMA_ALPHA * (600 - 1_000),
    );
    expect(delivery.pendingWordCount).toBe(3);
    expect(replay.updates).toHaveLength(0);
  });

  it("paces one raw target across multiple independently replayable updates", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(400);
    delivery.pushSnapshot(
      snapshot(
        1,
        "one two three four",
        ["one", "two", "three", "four"],
        40,
      ),
    );

    scheduler.advanceAndRun(100);
    expect(replay.text).toBe("one");
    expect(replay.wordTexts).toEqual(["one"]);
    expect(delivery.pendingWordCount).toBe(3);

    scheduler.advanceAndRun(100);
    expect(replay.text).toBe("one two");
    expect(replay.wordTexts).toEqual(["one", "two"]);

    scheduler.advanceAndRun(200);
    expect(replay.text).toBe("one two three four");
    expect(replay.wordTexts).toEqual([
      "one",
      "two",
      "three",
      "four",
    ]);
    expect(delivery.pendingWordCount).toBe(0);
    expect(delivery.hasScheduledFrame).toBe(false);
    expect(replay.updates.map(({ revision }) => revision)).toEqual([
      1, 2, 3,
    ]);
    expect(
      replay.updates.map(({ sourceRevision }) => sourceRevision),
    ).toEqual([1, 1, 1]);
  });

  it("replaces an undrained target and catches its real-word backlog up", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(400);
    delivery.pushSnapshot(
      snapshot(
        1,
        "a stale one two",
        ["a", "stale", "one", "two"],
        40,
      ),
    );
    scheduler.advanceAndRun(100);
    expect(replay.text).toBe("a");

    delivery.pushSnapshot(
      snapshot(
        2,
        "a fresh three four five six",
        ["a", "fresh", "three", "four", "five", "six"],
        60,
      ),
    );
    expect(delivery.sourceIntervalEstimateMilliseconds).toBe(295);

    scheduler.advanceAndRun(59);
    expect(replay.text).toBe("a fresh");
    expect(replay.text).not.toContain("stale");

    scheduler.advanceAndRun(236);
    expect(replay.text).toBe("a fresh three four five six");
    expect(replay.wordTexts).toEqual([
      "a",
      "fresh",
      "three",
      "four",
      "five",
      "six",
    ]);
    expect(delivery.pendingWordCount).toBe(0);
    expect(replay.updates.at(-1)?.sourceRevision).toBe(2);
  });

  it("immediately replaces a visible corrected suffix before pacing onward", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(400);
    delivery.pushSnapshot(
      snapshot(
        1,
        "alpha beta gamma delta",
        ["alpha", "beta", "gamma", "delta"],
        40,
      ),
    );
    scheduler.advanceAndRun(300);
    expect(replay.text).toBe("alpha beta gamma");

    const updateCount = replay.updates.length;
    delivery.pushSnapshot({
      ...snapshot(
        2,
        "alpha BETA gamma delta epsilon",
        ["alpha", "BETA", "gamma", "delta", "epsilon"],
        50,
      ),
      words: [
        timedWord("alpha", 0),
        timedWord("BETA", 1, 1.25),
        timedWord("gamma", 2),
        timedWord("delta", 3),
        timedWord("epsilon", 4),
      ],
    });

    expect(replay.updates).toHaveLength(updateCount + 1);
    expect(replay.text).toBe("alpha BETA gamma");
    expect(replay.wordTexts).toEqual(["alpha", "BETA", "gamma"]);
    expect(replay.words[1]).toMatchObject({
      text: "BETA",
      startSeconds: 1.25,
    });
    expect(replay.updates.at(-1)).toMatchObject({
      sourceRevision: 2,
      isFinal: false,
      wordSplice: {
        startIndex: 1,
        deleteCount: 2,
      },
    });

    scheduler.advanceAndRun(
      delivery.sourceIntervalEstimateMilliseconds!,
    );
    expect(replay.text).toBe(
      "alpha BETA gamma delta epsilon",
    );
  });

  it("preserves the visible time frontier across insertions and deletions", () => {
    const insertionScheduler = new ManualScheduler();
    const insertionReplay = new SpliceReplay();
    const insertionDelivery = createDelivery(
      insertionScheduler,
      insertionReplay,
    );
    insertionScheduler.advance(300);
    insertionDelivery.pushSnapshot({
      ...snapshot(1, "one three", [], 20),
      words: [
        timedWord("one", 0),
        timedWord("three", 2),
      ],
    });
    insertionScheduler.advanceAndRun(300);
    expect(insertionReplay.text).toBe("one three");

    insertionDelivery.pushSnapshot({
      ...snapshot(2, "one two three four", [], 40),
      words: [
        timedWord("one", 0),
        timedWord("two", 1),
        timedWord("three", 2),
        timedWord("four", 3),
      ],
    });
    expect(insertionReplay.text).toBe("one two three");
    expect(insertionReplay.text).not.toContain("four");

    const deletionScheduler = new ManualScheduler();
    const deletionReplay = new SpliceReplay();
    const deletionDelivery = createDelivery(
      deletionScheduler,
      deletionReplay,
    );
    deletionScheduler.advance(300);
    deletionDelivery.pushSnapshot(
      snapshot(
        1,
        "one two three",
        ["one", "two", "three"],
        30,
      ),
    );
    deletionScheduler.advanceAndRun(300);
    expect(deletionReplay.text).toBe("one two three");

    deletionDelivery.pushSnapshot({
      ...snapshot(2, "one three four", [], 40),
      words: [
        timedWord("one", 0),
        timedWord("three", 2),
        timedWord("four", 3),
      ],
    });
    expect(deletionReplay.text).toBe("one three");
    expect(deletionReplay.text).not.toContain("four");
  });

  it("uses the earliest repeated zero-duration frontier match", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(200);
    delivery.pushSnapshot({
      ...snapshot(1, "echo", [], 10),
      words: [
        {
          text: "echo",
          startSeconds: 1,
          endSeconds: 1,
        },
      ],
    });
    scheduler.advanceAndRun(200);
    expect(replay.text).toBe("echo");

    delivery.pushSnapshot({
      ...snapshot(2, "echo echo tail", [], 20),
      words: [
        {
          text: "echo",
          startSeconds: 1,
          endSeconds: 1,
        },
        {
          text: "echo",
          startSeconds: 1,
          endSeconds: 1,
        },
        timedWord("tail", 2),
      ],
    });
    expect(replay.text).toBe("echo");
    expect(delivery.pendingWordCount).toBe(2);
  });

  it("clears a provisional zero-word text before pacing wordful content", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(200);
    delivery.pushSnapshot(snapshot(1, "…", [], 0));
    expect(replay.text).toBe("…");

    scheduler.advance(200);
    delivery.pushSnapshot(
      snapshot(2, "hello world", ["hello", "world"], 20),
    );
    expect(replay.text).toBe("");
    scheduler.advanceAndRun(
      delivery.sourceIntervalEstimateMilliseconds!,
    );
    expect(replay.text).toBe("hello world");
  });

  it("flushes the exact final target immediately and marks final once", async () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(400);
    delivery.pushSnapshot(
      snapshot(1, "one old three", ["one", "old", "three"], 30),
    );
    scheduler.advanceAndRun(267);
    expect(replay.text).toBe("one old");
    const cancelledHandle = scheduler.onlyPendingHandle();

    const result = transcriptionResult(
      "one corrected three four \n",
      ["one", "corrected", "three", "four"],
    );
    const completion = delivery.finish(result);
    expect(delivery.finish(result)).toBe(completion);
    expect(replay.text).toBe(result.text);
    expect(replay.wordTexts).toEqual([
      "one",
      "corrected",
      "three",
      "four",
    ]);
    expect(replay.updates.filter(({ isFinal }) => isFinal)).toHaveLength(1);
    expect(replay.updates.at(-1)?.isFinal).toBe(true);
    expect(delivery.hasScheduledFrame).toBe(false);
    await expect(completion).resolves.toBeUndefined();

    const updateCount = replay.updates.length;
    scheduler.runHandleIncludingCancelled(cancelledHandle);
    expect(replay.updates).toHaveLength(updateCount);
    expect(replay.updates.filter(({ isFinal }) => isFinal)).toHaveLength(1);
  });

  it("flushes pending primary words and makes repair updates immediate", async () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(400);
    delivery.pushSnapshot(
      snapshot(
        1,
        "one two three \n",
        ["one", "two", "three"],
        30,
      ),
    );
    scheduler.advanceAndRun(134);
    expect(replay.text).toBe("one");
    const cancelledHandle = scheduler.onlyPendingHandle();

    delivery.flushPendingAndDisablePacing();
    expect(replay.text).toBe("one two three \n");
    expect(replay.wordTexts).toEqual(["one", "two", "three"]);
    expect(delivery.pendingWordCount).toBe(0);
    expect(delivery.hasScheduledFrame).toBe(false);
    expect(replay.updates.at(-1)?.isFinal).toBe(false);

    const updateCount = replay.updates.length;
    scheduler.runHandleIncludingCancelled(cancelledHandle);
    expect(replay.updates).toHaveLength(updateCount);

    const repaired = snapshot(
      2,
      "one repaired two three \n",
      ["one", "repaired", "two", "three"],
      30,
    );
    delivery.pushSnapshot(repaired);
    expect(replay.text).toBe(repaired.text);
    expect(replay.wordTexts).toEqual([
      "one",
      "repaired",
      "two",
      "three",
    ]);
    expect(delivery.hasScheduledFrame).toBe(false);

    const result = transcriptionResult(repaired.text, [
      "one",
      "repaired",
      "two",
      "three",
    ]);
    const completion = delivery.finish(result);
    expect(replay.text).toBe(result.text);
    expect(replay.updates.at(-1)?.isFinal).toBe(true);
    await expect(completion).resolves.toBeUndefined();
  });

  it("can disable pacing before the first snapshot arrives", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    delivery.flushPendingAndDisablePacing();
    delivery.flushPendingAndDisablePacing();
    delivery.pushSnapshot(
      snapshot(1, "ready now \n", ["ready", "now"], 20),
    );

    expect(replay.text).toBe("ready now \n");
    expect(replay.wordTexts).toEqual(["ready", "now"]);
    expect(delivery.hasScheduledFrame).toBe(false);
  });

  it("ignores stale revisions without changing timing or target state", () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const errors: unknown[] = [];
    const delivery = createDelivery(scheduler, replay, errors);

    scheduler.advance(500);
    delivery.pushSnapshot(snapshot(2, "new words", ["new", "words"], 20));
    const estimate = delivery.sourceIntervalEstimateMilliseconds;
    const pending = delivery.pendingWordCount;

    scheduler.advance(2_000);
    delivery.pushSnapshot({
      ...snapshot(2, "malformed", ["missing"], 99),
      text: "word cannot be located",
    });
    delivery.pushSnapshot(snapshot(1, "older", ["older"], 10));

    expect(delivery.sourceIntervalEstimateMilliseconds).toBe(estimate);
    expect(delivery.pendingWordCount).toBe(pending);
    expect(errors).toEqual([]);

    scheduler.advanceAndRun(estimate!);
    expect(replay.text).toBe("new words");
  });

  it("cancels pending delivery and ignores a late scheduled callback", async () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(300);
    delivery.pushSnapshot(
      snapshot(1, "one two three", ["one", "two", "three"], 30),
    );
    const scheduledHandle = scheduler.onlyPendingHandle();
    const reason = new DOMException("stop", "AbortError");

    delivery.cancel(reason);
    expect(delivery.hasScheduledFrame).toBe(false);
    expect(scheduler.pendingCount).toBe(0);
    await expect(
      delivery.finish(
        transcriptionResult(
          "one two three four",
          ["one", "two", "three", "four"],
        ),
      ),
    ).rejects.toBe(reason);

    scheduler.runHandleIncludingCancelled(scheduledHandle);
    expect(replay.updates).toHaveLength(0);
    delivery.pushSnapshot(snapshot(3, "late", ["late"], 40));
    expect(replay.updates).toHaveLength(0);
  });

  it("reports callback failures, closes delivery, and rejects finish", async () => {
    const scheduler = new ManualScheduler();
    const callbackError = new Error("consumer failed");
    const errors: unknown[] = [];
    let calls = 0;
    const delivery = new PacedTranscriptDelivery({
      scheduler,
      onUpdate: () => {
        calls += 1;
        throw callbackError;
      },
      onError: (error) => errors.push(error),
    });

    scheduler.advance(200);
    delivery.pushSnapshot(snapshot(1, "one two", ["one", "two"], 20));
    scheduler.advanceAndRun(100);

    expect(calls).toBe(1);
    expect(errors).toEqual([callbackError]);
    expect(delivery.hasScheduledFrame).toBe(false);
    await expect(
      delivery.finish(transcriptionResult("one two", ["one", "two"])),
    ).rejects.toBe(callbackError);

    scheduler.runAllIncludingCancelled();
    expect(calls).toBe(1);
  });

  it("replays Unicode, repeated words, and exact trailing text", async () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);
    const text = "🙂 echo echo 世界 \n";
    const words = ["🙂", "echo", "echo", "世界"];

    scheduler.advance(400);
    delivery.pushSnapshot(snapshot(1, text, words, 40));
    scheduler.advanceAndRun(100);
    scheduler.advanceAndRun(100);
    scheduler.advanceAndRun(200);
    expect(replay.text).toBe("🙂 echo echo 世界");
    expect(replay.wordTexts).toEqual(words);
    expect(replay.words[0]).toMatchObject({
      startOffset: 0,
      endOffset: 2,
    });
    expect(replay.words[1]).toMatchObject({
      startOffset: 3,
      endOffset: 7,
    });
    expect(replay.words[2]).toMatchObject({
      startOffset: 8,
      endOffset: 12,
    });

    const completion = delivery.finish(transcriptionResult(text, words));
    await expect(completion).resolves.toBeUndefined();
    expect(replay.text).toBe(text);
    expect(replay.updates.at(-1)).toMatchObject({
      isFinal: true,
      textSplice: {
        startOffset: "🙂 echo echo 世界".length,
        deleteCount: 0,
        insertText: " \n",
      },
    });
  });

  it("emits a text-only final result in one terminal update", async () => {
    const scheduler = new ManualScheduler();
    const replay = new SpliceReplay();
    const delivery = createDelivery(scheduler, replay);

    scheduler.advance(250);
    const result = transcriptionResult("…", []);
    await expect(delivery.finish(result)).resolves.toBeUndefined();
    expect(replay.text).toBe("…");
    expect(replay.updates).toHaveLength(1);
    expect(replay.updates[0]).toMatchObject({
      isFinal: true,
      textSplice: {
        startOffset: 0,
        deleteCount: 0,
        insertText: "…",
      },
    });
  });
});

function createDelivery(
  scheduler: ManualScheduler,
  replay: SpliceReplay,
  errors: unknown[] = [],
): PacedTranscriptDelivery {
  return new PacedTranscriptDelivery({
    scheduler,
    onUpdate: (update) => replay.apply(update),
    onError: (error) => errors.push(error),
  });
}

class ManualScheduler implements PacedTranscriptScheduler {
  time = 0;
  private nextHandle = 1;
  private readonly callbacks = new Map<
    number,
    { readonly callback: () => void; readonly dueAt: number }
  >();
  private readonly allCallbacks = new Map<number, () => void>();

  get pendingCount(): number {
    return this.callbacks.size;
  }

  now(): number {
    return this.time;
  }

  schedule(callback: () => void, delayMilliseconds: number): number {
    const handle = this.nextHandle;
    this.nextHandle += 1;
    this.callbacks.set(handle, {
      callback,
      dueAt: this.time + delayMilliseconds,
    });
    this.allCallbacks.set(handle, callback);
    return handle;
  }

  cancel(handle: unknown): void {
    this.callbacks.delete(this.numericHandle(handle));
  }

  advance(milliseconds: number): void {
    this.time += milliseconds;
  }

  advanceAndRun(milliseconds: number): void {
    this.advance(milliseconds);
    this.runNext();
  }

  runNext(): void {
    const handle = this.callbacks.keys().next().value as
      | number
      | undefined;
    if (handle === undefined) {
      throw new Error("No scheduled paced-transcript frame");
    }
    const scheduled = this.callbacks.get(handle)!;
    if (scheduled.dueAt > this.time) {
      throw new Error(
        `Next paced callback is due at ${scheduled.dueAt}, current time is ${this.time}`,
      );
    }
    this.callbacks.delete(handle);
    scheduled.callback();
  }

  onlyPendingHandle(): number {
    expect(this.callbacks.size).toBe(1);
    return this.callbacks.keys().next().value as number;
  }

  runHandleIncludingCancelled(handle: number): void {
    const callback = this.allCallbacks.get(handle);
    if (callback === undefined) {
      throw new Error(`Unknown scheduler handle ${handle}`);
    }
    this.callbacks.delete(handle);
    callback();
  }

  runAllIncludingCancelled(): void {
    for (const [handle, callback] of this.allCallbacks) {
      this.callbacks.delete(handle);
      callback();
    }
  }

  private numericHandle(handle: unknown): number {
    if (typeof handle !== "number") {
      throw new TypeError("Manual scheduler handle must be a number");
    }
    return handle;
  }
}

class SpliceReplay {
  text = "";
  words: PacedTranscriptWord[] = [];
  readonly updates: PacedTranscriptUpdate[] = [];

  get wordTexts(): readonly string[] {
    return this.words.map(({ text }) => text);
  }

  apply(update: PacedTranscriptUpdate): void {
    expect(update.revision).toBe(this.updates.length + 1);
    expect(update.textSplice.startOffset).toBeGreaterThanOrEqual(0);
    expect(
      update.textSplice.startOffset + update.textSplice.deleteCount,
    ).toBeLessThanOrEqual(this.text.length);
    expect(update.wordSplice.startIndex).toBeGreaterThanOrEqual(0);
    expect(
      update.wordSplice.startIndex + update.wordSplice.deleteCount,
    ).toBeLessThanOrEqual(this.words.length);

    this.text =
      this.text.slice(0, update.textSplice.startOffset) +
      update.textSplice.insertText +
      this.text.slice(
        update.textSplice.startOffset + update.textSplice.deleteCount,
      );
    this.words.splice(
      update.wordSplice.startIndex,
      update.wordSplice.deleteCount,
      ...update.wordSplice.insertWords,
    );
    for (const word of this.words) {
      expect(word.startOffset).toBeGreaterThanOrEqual(0);
      expect(word.endOffset).toBeGreaterThanOrEqual(word.startOffset);
      expect(this.text.slice(word.startOffset, word.endOffset)).toBe(
        word.text,
      );
    }
    this.updates.push(update);
  }
}

function snapshot(
  revision: number,
  text: string,
  wordTexts: readonly string[],
  processedAudioSeconds: number,
): TranscriptionSnapshot {
  return {
    revision,
    text,
    tokens: [],
    words: wordTexts.map((word, index) => timedWord(word, index)),
    audioDurationSeconds: 100,
    processedAudioSeconds,
  };
}

function transcriptionResult(
  text: string,
  wordTexts: readonly string[],
): TranscriptionResult {
  return {
    text,
    tokens: [],
    words: wordTexts.map((word, index) => timedWord(word, index)),
    metrics: transcriptionMetrics(),
  };
}

function timedWord(
  text: string,
  index: number,
  startSeconds = index,
): TranscriptionWord {
  return {
    text,
    startSeconds,
    endSeconds: startSeconds + 0.5,
  };
}

function transcriptionMetrics(): TranscriptionMetrics {
  return {
    audioDecoding: {
      applied: false,
      wallMs: 0,
      activeMs: 0,
      inputByteLength: 44,
      outputByteLength: 44,
      overlapMs: 0,
      pcmWaitMs: 0,
      pcmStarvationMs: 0,
      pcmStarvationCount: 0,
    },
    audioDurationSeconds: 100,
    processedAudioSeconds: 100,
    elapsedMs: 1,
    audioReadMs: 0,
    inferenceMs: 0,
    frontendMs: 0,
    encoderMs: 0,
    decoderMs: 0,
    speedFactor: 100_000,
    completedWindows: 1,
    totalWindows: 1,
    repairGapProbes: 0,
    repairWindowDecodes: 0,
    repairRounds: 0,
    recoveredTokens: 0,
    stitchingMs: 0,
    repairMs: 0,
    repairPrefetchWindowDecodes: 0,
    repairPrefetchCacheHits: 0,
    repairPrefetchUnusedDecodes: 0,
    batchSubmissionTimeline: [],
    batchSubmissionTimelineOmitted: 0,
    phaseTimings: {
      trailingSpeechScanMs: 0,
      adaptiveThresholdScanMs: 0,
      gapEnergyScanMs: 0,
      incrementalStitchMs: 0,
      prefixStitchMs: 0,
      finalStitchMs: 0,
      repairPlanningMs: 0,
      repairSplicingMs: 0,
      partialResultMs: 0,
      tokenDecodeMs: 0,
    },
    totalMs: 1,
  };
}
