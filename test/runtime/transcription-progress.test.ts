import { describe, expect, it } from "vitest";

import {
  MonotonicTranscriptionProgress,
  TRANSCRIPTION_PRE_DONE_FRACTION,
  graphWeight,
  packedGraphWeight,
} from "../../src/runtime/transcription-progress";

describe("monotonic overall transcription progress", () => {
  it("integrates exact primary and repair graph milestones", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 59,
      batchCapacity: 40,
    });
    const values = [progress.fraction];

    const first = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    values.push(first.update(1, 2));
    values.push(first.complete());

    const tail = progress.beginGraph({ primaryRows: 19, repairRows: 0 });
    values.push(tail.update(1, 2));
    values.push(tail.complete());

    progress.setEstimatedRepairWaves([1, 1]);
    const gapStart = progress.beginGraph({ primaryRows: 0, repairRows: 1 });
    values.push(gapStart.update(1, 2));
    values.push(gapStart.complete());
    progress.setEstimatedRepairWaves([1]);
    const centered = progress.beginGraph({ primaryRows: 0, repairRows: 1 });
    values.push(centered.update(1, 2));
    values.push(centered.complete());
    values.push(progress.setEstimatedRepairWaves([]));

    expect(
      values.every(
        (value, index) => index === 0 || value >= values[index - 1]!,
      ),
    ).toBe(true);
    expect(values.every((value) => value < 1)).toBe(true);
    expect(progress.fraction).toBe(TRANSCRIPTION_PRE_DONE_FRACTION);
    expect(progress.finish()).toBe(1);
    expect(progress.isFinished).toBe(true);
  });

  it("plateaus instead of regressing when new repair work is discovered", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 40,
      batchCapacity: 40,
    });
    const primary = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    primary.update(27, 27);
    primary.complete();
    const beforeDiscovery = progress.fraction;

    const afterDiscovery = progress.setEstimatedRepairWaves([32, 32]);
    expect(afterDiscovery).toBe(beforeDiscovery);

    const repair = progress.beginGraph({ primaryRows: 0, repairRows: 32 });
    expect(repair.update(1, 27)).toBe(beforeDiscovery);
    repair.update(27, 27);
    repair.complete();
    expect(progress.fraction).toBeGreaterThanOrEqual(beforeDiscovery);
  });

  it("publishes streamed graph work before EOF and reconciles the exact plan", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 80,
      batchCapacity: 40,
      primaryWindowsSealed: false,
    });
    const values = [progress.fraction];

    const first = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    values.push(first.update(1, 27));
    expect(values.at(-1)).toBeGreaterThan(0);
    values.push(first.complete());
    const beforeGrowth = progress.fraction;

    values.push(progress.extendPrimaryWindowEstimate(120));
    expect(values.at(-1)).toBe(beforeGrowth);
    const second = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    values.push(second.update(27, 27));
    values.push(second.complete());

    // EOF proves the metadata/horizon overestimated this stream.
    values.push(progress.sealPrimaryWindows(81));
    expect(progress.primaryWindowsSealed).toBe(true);
    expect(progress.totalPrimaryWindowEstimate).toBe(81);
    const tail = progress.beginGraph({ primaryRows: 1, repairRows: 0 });
    values.push(tail.complete());
    values.push(progress.setEstimatedRepairWaves([]));

    expect(
      values.every(
        (value, index) => index === 0 || value >= values[index - 1]!,
      ),
    ).toBe(true);
    expect(values.every((value) => value < 1)).toBe(true);
    expect(progress.finish()).toBe(1);
  });

  it("rejects invalid streamed primary progress transitions", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 40,
      batchCapacity: 40,
      primaryWindowsSealed: false,
    });
    progress.beginGraph({ primaryRows: 40, repairRows: 0 }).complete();

    expect(() => progress.extendPrimaryWindowEstimate(39)).toThrow(
      "can only grow",
    );
    expect(() => progress.sealPrimaryWindows(39)).toThrow(
      "below registered",
    );
    expect(() => progress.setEstimatedRepairWaves([])).toThrow(
      "before primary work is sealed",
    );
    expect(() =>
      progress.beginGraph({ primaryRows: 0, repairRows: 1 }),
    ).toThrow("before primary progress is sealed");
    expect(() => progress.finish()).toThrow(
      "before primary work is sealed",
    );

    progress.sealPrimaryWindows(40);
    expect(() => progress.sealPrimaryWindows(40)).toThrow(
      "already sealed",
    );
    progress.setEstimatedRepairWaves([]);
    expect(progress.finish()).toBe(1);
  });

  it("counts a mixed primary-prefetch graph once", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 43,
      batchCapacity: 40,
    });
    const first = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    first.complete();
    expect(progress.setEstimatedRepairWaves([2, 2])).toBe(
      42 / (42 + 5 + 2 * graphWeight(43 / 40)),
    );

    const mixed = progress.beginGraph({ primaryRows: 3, repairRows: 2 });
    const before = progress.fraction;
    const halfway = mixed.update(1, 2);
    const complete = mixed.complete();

    expect(before).toBe(42 / 53);
    expect(halfway).toBe(45.5 / 53);
    expect(complete).toBe(49 / 53);
    expect(graphWeight(5)).toBe(7);
    expect(packedGraphWeight(43, 40)).toBe(47);
  });

  it("does not double-count cached work and retains unused prefetch cost", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 3,
      batchCapacity: 40,
    });
    progress.setEstimatedRepairWaves([1, 1]);

    const mixed = progress.beginGraph({ primaryRows: 3, repairRows: 1 });
    mixed.complete();
    expect(progress.fraction).toBe(2 / 3);

    // The official gap-start request is a cache hit: changing only the
    // remaining centered forecast must not register another physical graph.
    expect(progress.setEstimatedRepairWaves([1])).toBe(2 / 3);
    expect(progress.setEstimatedRepairWaves([])).toBe(
      TRANSCRIPTION_PRE_DONE_FRACTION,
    );
    expect(progress.finish()).toBe(1);
  });

  it("accounts for two in-flight physical graphs without changing the denominator", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 80,
      batchCapacity: 40,
    });
    const first = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    expect(first.update(1, 2)).toBe(21 / 92);

    const second = progress.beginGraph({ primaryRows: 40, repairRows: 0 });
    expect(progress.fraction).toBe(21 / 92);
    expect(second.update(1, 2)).toBe(42 / 92);
    first.complete();
    expect(progress.fraction).toBe(63 / 92);
    second.complete();
    expect(progress.fraction).toBe(84 / 92);
  });

  it("rejects completion while planned work remains unresolved", () => {
    const primaryPending = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 2,
      batchCapacity: 40,
    });
    expect(() => primaryPending.finish()).toThrow(
      "before all primary graphs",
    );

    const repairPending = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 2,
      batchCapacity: 40,
    });
    repairPending.beginGraph({ primaryRows: 2, repairRows: 0 }).complete();
    expect(() => repairPending.finish()).toThrow(
      "before repair is resolved",
    );
  });

  it("reserves no adaptive work for a single primary window", () => {
    const progress = new MonotonicTranscriptionProgress({
      totalPrimaryWindows: 1,
      batchCapacity: 40,
    });
    const graph = progress.beginGraph({ primaryRows: 1, repairRows: 0 });
    graph.complete();
    expect(progress.fraction).toBe(TRANSCRIPTION_PRE_DONE_FRACTION);
    expect(progress.finish()).toBe(1);
  });
});
