import { SEAM_GAP_MAX_PROBES } from "./seam-gap-repair";

/**
 * Two row-equivalent units model the fixed palette-expansion, submission,
 * drain, and readback floor of every physical graph. Retained B1/B8/B19/B40
 * timings support the same deterministic approximation in both precisions.
 */
export const TRANSCRIPTION_GRAPH_FIXED_WORK_UNITS = 2;
export const TRANSCRIPTION_PRE_DONE_FRACTION = 0.999_999;

export interface TranscriptionGraphProgress {
  readonly update: (
    completedWorkUnits: number,
    totalWorkUnits: number,
  ) => number;
  readonly complete: () => number;
}

interface ActiveGraph {
  readonly weight: number;
  fraction: number;
  completedWorkUnits: number;
  totalWorkUnits: number | undefined;
}

/**
 * One monotonic work estimate spanning primary and adaptive repair inference.
 *
 * Physical graph milestones are exact. A streamed primary plan and
 * not-yet-discoverable repair branches are estimated; callers replace those
 * estimates as plans resolve. An increased estimate creates a plateau because
 * the published fraction is clamped to its previous value. Exactly 1 is
 * reserved for finish().
 */
export class MonotonicTranscriptionProgress {
  readonly #batchCapacity: number;
  readonly #activeGraphs = new Map<number, ActiveGraph>();

  #nextGraphId = 0;
  #totalPrimaryWindows: number;
  #registeredPrimaryRows = 0;
  #primaryWindowsSealed: boolean;
  #futurePrimaryWeight: number;
  #estimatedRepairWeight: number;
  #usingDefaultRepairEstimate = true;
  #completedWeight = 0;
  #publishedFraction = 0;
  #finished = false;

  constructor(options: {
    readonly totalPrimaryWindows: number;
    readonly batchCapacity: number;
    /** False while streamed PCM has not reached an exact EOF-derived plan. */
    readonly primaryWindowsSealed?: boolean;
  }) {
    const {
      totalPrimaryWindows,
      batchCapacity,
      primaryWindowsSealed = true,
    } = options;
    assertNonNegativeInteger(
      totalPrimaryWindows,
      "totalPrimaryWindows",
    );
    assertPositiveInteger(batchCapacity, "batchCapacity");

    this.#batchCapacity = batchCapacity;
    this.#totalPrimaryWindows = totalPrimaryWindows;
    this.#primaryWindowsSealed = primaryWindowsSealed;
    this.#futurePrimaryWeight = packedGraphWeight(
      totalPrimaryWindows,
      batchCapacity,
    );
    // Reserve one possible probe per primary-graph equivalent for each of
    // the gap-start and centered placements. Exact repair plans replace this
    // length-scaled prior as soon as decoder-dependent work becomes known.
    this.#estimatedRepairWeight = this.#defaultRepairWeight(
      totalPrimaryWindows,
    );
  }

  get fraction(): number {
    return this.#publishCurrentFraction();
  }

  get isFinished(): boolean {
    return this.#finished;
  }

  get totalPrimaryWindowEstimate(): number {
    return this.#totalPrimaryWindows;
  }

  get primaryWindowsSealed(): boolean {
    return this.#primaryWindowsSealed;
  }

  /**
   * Grow the provisional streamed-input primary plan. Every already-started
   * physical graph remains counted exactly; a larger denominator is published
   * as a plateau rather than a backwards step.
   */
  extendPrimaryWindowEstimate(totalPrimaryWindows: number): number {
    this.#assertActive();
    if (this.#primaryWindowsSealed) {
      throw new Error("Primary transcription progress is already sealed");
    }
    assertNonNegativeInteger(
      totalPrimaryWindows,
      "totalPrimaryWindows",
    );
    if (totalPrimaryWindows < this.#totalPrimaryWindows) {
      throw new RangeError(
        "Provisional primary-window progress can only grow",
      );
    }
    this.#setPrimaryWindowTotal(totalPrimaryWindows);
    return this.#publishCurrentFraction();
  }

  /** Replace the provisional streamed-input plan with the exact EOF plan. */
  sealPrimaryWindows(totalPrimaryWindows: number): number {
    this.#assertActive();
    if (this.#primaryWindowsSealed) {
      throw new Error("Primary transcription progress is already sealed");
    }
    assertNonNegativeInteger(
      totalPrimaryWindows,
      "totalPrimaryWindows",
    );
    this.#setPrimaryWindowTotal(totalPrimaryWindows);
    this.#primaryWindowsSealed = true;
    return this.#publishCurrentFraction();
  }

  /**
   * Replace the unknown repair reserve with currently justified physical
   * waves. A wave is the number of rows that would share one placement pass;
   * packing and fixed graph overhead are derived here rather than in callers.
   */
  setEstimatedRepairWaves(
    rowCounts: readonly number[],
  ): number {
    this.#assertActive();
    if (!this.#primaryWindowsSealed) {
      throw new Error(
        "Cannot resolve repair progress before primary work is sealed",
      );
    }
    let weight = 0;
    for (const rowCount of rowCounts) {
      if (!Number.isFinite(rowCount) || rowCount < 0) {
        throw new RangeError(
          "Estimated repair rows must be finite and non-negative",
        );
      }
      weight += estimatedPackedGraphWeight(
        rowCount,
        this.#batchCapacity,
      );
    }
    this.#usingDefaultRepairEstimate = false;
    this.#estimatedRepairWeight = weight;
    return this.#publishCurrentFraction();
  }

  /** Register one real physical graph exactly once. */
  beginGraph(options: {
    readonly primaryRows: number;
    readonly repairRows: number;
  }): TranscriptionGraphProgress {
    this.#assertActive();
    const { primaryRows, repairRows } = options;
    assertNonNegativeInteger(primaryRows, "primaryRows");
    assertNonNegativeInteger(repairRows, "repairRows");
    const activeRows = primaryRows + repairRows;
    if (activeRows < 1 || activeRows > this.#batchCapacity) {
      throw new RangeError(
        "Physical graph rows must fit the fixed batch capacity",
      );
    }
    if (repairRows > 0 && !this.#primaryWindowsSealed) {
      throw new Error(
        "Cannot begin repair work before primary progress is sealed",
      );
    }

    if (primaryRows > 0) {
      const nextRegisteredPrimaryRows =
        this.#registeredPrimaryRows + primaryRows;
      if (nextRegisteredPrimaryRows > this.#totalPrimaryWindows) {
        throw new RangeError(
          "Physical primary graph exceeds the planned primary work",
        );
      }
      this.#registeredPrimaryRows = nextRegisteredPrimaryRows;
      this.#futurePrimaryWeight = packedGraphWeight(
        this.#totalPrimaryWindows - this.#registeredPrimaryRows,
        this.#batchCapacity,
      );
    }
    if (repairRows > 0) {
      this.#usingDefaultRepairEstimate = false;
      this.#estimatedRepairWeight = Math.max(
        0,
        this.#estimatedRepairWeight - graphWeight(repairRows),
      );
    }

    const graphId = this.#nextGraphId;
    this.#nextGraphId += 1;
    const graph: ActiveGraph = {
      weight: graphWeight(activeRows),
      fraction: 0,
      completedWorkUnits: 0,
      totalWorkUnits: undefined,
    };
    this.#activeGraphs.set(graphId, graph);
    let complete = false;

    return {
      update: (completedWorkUnits, totalWorkUnits) => {
        if (complete) {
          throw new Error(
            "Cannot update completed transcription graph progress",
          );
        }
        assertPositiveInteger(totalWorkUnits, "totalWorkUnits");
        assertPositiveInteger(
          completedWorkUnits,
          "completedWorkUnits",
        );
        if (
          completedWorkUnits > totalWorkUnits ||
          completedWorkUnits <= graph.completedWorkUnits ||
          (graph.totalWorkUnits !== undefined &&
            totalWorkUnits !== graph.totalWorkUnits)
        ) {
          throw new RangeError(
            "Physical graph progress must be contiguous and monotonic",
          );
        }
        graph.completedWorkUnits = completedWorkUnits;
        graph.totalWorkUnits = totalWorkUnits;
        graph.fraction = completedWorkUnits / totalWorkUnits;
        return this.#publishCurrentFraction();
      },
      complete: () => {
        if (complete) return this.#publishCurrentFraction();
        complete = true;
        this.#activeGraphs.delete(graphId);
        this.#completedWeight += graph.weight;
        return this.#publishCurrentFraction();
      },
    };
  }

  finish(): number {
    this.#assertActive();
    if (!this.#primaryWindowsSealed) {
      throw new Error(
        "Cannot finish transcription progress before primary work is sealed",
      );
    }
    if (this.#activeGraphs.size !== 0) {
      throw new Error(
        "Cannot finish transcription progress with a graph in flight",
      );
    }
    if (this.#futurePrimaryWeight !== 0) {
      throw new Error(
        "Cannot finish transcription progress before all primary graphs",
      );
    }
    if (this.#estimatedRepairWeight !== 0) {
      throw new Error(
        "Cannot finish transcription progress before repair is resolved",
      );
    }
    this.#publishedFraction = 1;
    this.#finished = true;
    return 1;
  }

  #publishCurrentFraction(): number {
    if (this.#finished) return 1;
    let activeCompletedWeight = 0;
    let activeRemainingWeight = 0;
    for (const graph of this.#activeGraphs.values()) {
      activeCompletedWeight += graph.weight * graph.fraction;
      activeRemainingWeight += graph.weight * (1 - graph.fraction);
    }
    const completedWeight =
      this.#completedWeight + activeCompletedWeight;
    const totalWeight =
      completedWeight +
      activeRemainingWeight +
      this.#futurePrimaryWeight +
      this.#estimatedRepairWeight;
    const rawFraction =
      totalWeight > 0 ? completedWeight / totalWeight : 0;
    this.#publishedFraction = Math.max(
      this.#publishedFraction,
      Math.min(TRANSCRIPTION_PRE_DONE_FRACTION, rawFraction),
    );
    return this.#publishedFraction;
  }

  #assertActive(): void {
    if (this.#finished) {
      throw new Error("Transcription progress is already finished");
    }
  }

  #setPrimaryWindowTotal(totalPrimaryWindows: number): void {
    if (totalPrimaryWindows < this.#registeredPrimaryRows) {
      throw new RangeError(
        "Primary-window total cannot be below registered physical rows",
      );
    }
    this.#totalPrimaryWindows = totalPrimaryWindows;
    this.#futurePrimaryWeight = packedGraphWeight(
      totalPrimaryWindows - this.#registeredPrimaryRows,
      this.#batchCapacity,
    );
    if (this.#usingDefaultRepairEstimate) {
      this.#estimatedRepairWeight = this.#defaultRepairWeight(
        totalPrimaryWindows,
      );
    }
  }

  #defaultRepairWeight(totalPrimaryWindows: number): number {
    return totalPrimaryWindows > 1
      ? 2 *
          estimatedPackedGraphWeight(
            Math.min(
              SEAM_GAP_MAX_PROBES,
              Math.max(
                1,
                totalPrimaryWindows / this.#batchCapacity,
              ),
            ),
            this.#batchCapacity,
          )
      : 0;
  }
}

export function graphWeight(activeRows: number): number {
  if (!Number.isFinite(activeRows) || activeRows <= 0) {
    throw new RangeError("Graph rows must be finite and positive");
  }
  return activeRows + TRANSCRIPTION_GRAPH_FIXED_WORK_UNITS;
}

export function packedGraphWeight(
  rows: number,
  batchCapacity: number,
): number {
  assertNonNegativeInteger(rows, "rows");
  assertPositiveInteger(batchCapacity, "batchCapacity");
  if (rows === 0) return 0;
  const fullGraphs = Math.floor(rows / batchCapacity);
  const tailRows = rows % batchCapacity;
  return (
    fullGraphs * graphWeight(batchCapacity) +
    (tailRows === 0 ? 0 : graphWeight(tailRows))
  );
}

function estimatedPackedGraphWeight(
  rows: number,
  batchCapacity: number,
): number {
  if (!Number.isFinite(rows) || rows < 0) {
    throw new RangeError(
      "Estimated graph rows must be finite and non-negative",
    );
  }
  assertPositiveInteger(batchCapacity, "batchCapacity");
  if (rows === 0) return 0;
  if (rows < 1) return rows * graphWeight(1);
  const fullGraphs = Math.floor(rows / batchCapacity);
  const tailRows = rows - fullGraphs * batchCapacity;
  return (
    fullGraphs * graphWeight(batchCapacity) +
    (tailRows === 0 ? 0 : graphWeight(tailRows))
  );
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
}
