import {
  DEFAULT_PCM_STREAM_SAMPLES,
  PARAKEET_SAMPLE_RATE,
} from "./wav-stream";
import {
  isStreamingPcmProducerMessage,
  type StreamingPcmEndMessage,
  type StreamingPcmSegmentMessage,
} from "./streaming-decoder-protocol";

const BYTES_PER_SAMPLE = Int16Array.BYTES_PER_ELEMENT;

export interface StreamingDecoderTimings {
  readonly decoderWallMs: number;
  readonly decoderActiveMs: number;
  readonly decoderStartedAtMs: number;
  readonly decoderEndedAtMs: number;
}

export interface PcmAvailability {
  readonly availableSamples: number;
  readonly ended: boolean;
}

interface PendingAvailability {
  readonly minimumSamples: number;
  readonly resolve: (availability: PcmAvailability) => void;
  readonly reject: (reason: unknown) => void;
  readonly signal: AbortSignal | undefined;
  readonly abortListener: (() => void) | undefined;
}

/**
 * Random-access PCM16 reader over immutable OPFS segment snapshots.
 *
 * The producer closes every segment's sync handle before posting its File, so
 * reads never race a writer or depend on growing-File snapshot semantics.
 */
export class SegmentedPcm16Reader {
  readonly sampleRate = PARAKEET_SAMPLE_RATE;

  private readonly segments: StreamingPcmSegmentMessage[] = [];
  private readonly waiters = new Set<PendingAvailability>();
  private available = 0;
  private end: StreamingPcmEndMessage | undefined;
  private failure: unknown | undefined;
  private disposed = false;
  private consumeTail: Promise<void> = Promise.resolve();

  constructor(private readonly port: MessagePort) {
    port.addEventListener("message", this.handleMessage);
    port.start();
  }

  get availableSampleCount(): number {
    return this.available;
  }

  get ended(): boolean {
    return this.end !== undefined;
  }

  get sampleCount(): number {
    if (this.end === undefined) {
      throw new Error("Streaming PCM sample count is unknown before EOF");
    }
    return this.end.totalSamples;
  }

  get decoderTimings(): StreamingDecoderTimings | undefined {
    const end = this.end;
    return end === undefined
      ? undefined
      : {
          decoderWallMs: end.decoderWallMs,
          decoderActiveMs: end.decoderActiveMs,
          decoderStartedAtMs: end.decoderStartedAtMs,
          decoderEndedAtMs: end.decoderEndedAtMs,
        };
  }

  waitForSamples(
    minimumSamples: number,
    signal?: AbortSignal,
  ): Promise<PcmAvailability> {
    if (!Number.isSafeInteger(minimumSamples) || minimumSamples < 0) {
      return Promise.reject(
        new RangeError("minimumSamples must be a non-negative safe integer"),
      );
    }
    if (signal?.aborted === true) {
      return Promise.reject(abortReason(signal));
    }
    if (this.failure !== undefined) return Promise.reject(this.failure);
    if (this.available >= minimumSamples || this.end !== undefined) {
      return Promise.resolve(this.availability());
    }
    if (this.disposed) {
      return Promise.reject(new Error("Streaming PCM reader was disposed"));
    }

    return new Promise<PcmAvailability>((resolve, reject) => {
      let pending: PendingAvailability;
      const abortListener =
        signal === undefined
          ? undefined
          : () => {
              if (!this.waiters.delete(pending)) return;
              reject(abortReason(signal));
            };
      pending = {
        minimumSamples,
        resolve,
        reject,
        signal,
        abortListener,
      };
      this.waiters.add(pending);
      signal?.addEventListener("abort", abortListener!, { once: true });
    });
  }

  waitForEnd(signal?: AbortSignal): Promise<PcmAvailability> {
    return this.waitForSamples(Number.MAX_SAFE_INTEGER, signal);
  }

  async readSamplesInto(
    startSample: number,
    target: Float32Array,
    targetOffset = 0,
    sampleCount = target.length - targetOffset,
  ): Promise<number> {
    validateRead(startSample, target, targetOffset, sampleCount);
    if (this.failure !== undefined) throw this.failure;
    const count = Math.min(
      sampleCount,
      Math.max(0, this.available - startSample),
    );
    if (count === 0) return 0;

    const operation = this.consumeTail.then(() =>
      this.readAvailableRange(startSample, target, targetOffset, count),
    );
    this.consumeTail = operation.then(
      () => undefined,
      () => undefined,
    );
    await operation;
    return count;
  }

  dispose(reason: unknown = new Error("Streaming PCM reader was disposed")): void {
    if (this.disposed) return;
    this.disposed = true;
    this.port.removeEventListener("message", this.handleMessage);
    this.port.close();
    this.rejectWaiters(reason);
    this.segments.length = 0;
  }

  private readonly handleMessage = (event: MessageEvent<unknown>): void => {
    if (this.disposed || this.failure !== undefined || this.end !== undefined) {
      return;
    }
    const message = event.data;
    if (!isStreamingPcmProducerMessage(message)) {
      this.fail(new Error("Streaming decoder sent an invalid PCM message"));
      return;
    }
    if (message.type === "pcm-failed") {
      this.fail(new Error(message.message));
      return;
    }
    if (message.type === "pcm-segment") {
      if (
        message.segmentIndex !== this.segments.length ||
        message.startSample !== this.available
      ) {
        this.fail(new Error("Streaming decoder sent non-contiguous PCM segments"));
        return;
      }
      this.segments.push(message);
      this.available += message.sampleCount;
      this.resolveWaiters();
      return;
    }

    if (
      message.totalSamples !== this.available ||
      message.outputByteLength !== this.available * BYTES_PER_SAMPLE
    ) {
      this.fail(new Error("Streaming decoder EOF does not match PCM segments"));
      return;
    }
    this.end = message;
    this.resolveWaiters();
  };

  private async readAvailableRange(
    startSample: number,
    target: Float32Array,
    targetOffset: number,
    sampleCount: number,
  ): Promise<void> {
    let sourceSample = startSample;
    let destination = targetOffset;
    let remaining = sampleCount;
    let segmentIndex = findSegment(this.segments, sourceSample);

    while (remaining > 0) {
      const segment = this.segments[segmentIndex];
      if (segment === undefined) {
        throw new Error("Streaming PCM segment index is incomplete");
      }
      const localStart = sourceSample - segment.startSample;
      const availableInSegment = segment.sampleCount - localStart;
      const fromSegment = Math.min(remaining, availableInSegment);
      let copied = 0;
      while (copied < fromSegment) {
        const chunkSamples = Math.min(
          DEFAULT_PCM_STREAM_SAMPLES,
          fromSegment - copied,
        );
        const byteStart = (localStart + copied) * BYTES_PER_SAMPLE;
        const bytes = new Uint8Array(
          await segment.file
            .slice(byteStart, byteStart + chunkSamples * BYTES_PER_SAMPLE)
            .arrayBuffer(),
        );
        if (bytes.byteLength !== chunkSamples * BYTES_PER_SAMPLE) {
          throw new Error("Immutable PCM segment produced a short read");
        }
        const view = new DataView(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength,
        );
        for (let index = 0; index < chunkSamples; index += 1) {
          target[destination + copied + index] =
            view.getInt16(index * BYTES_PER_SAMPLE, true) / 32_768;
        }
        copied += chunkSamples;
      }
      sourceSample += fromSegment;
      destination += fromSegment;
      remaining -= fromSegment;
      segmentIndex += 1;
    }
  }

  private availability(): PcmAvailability {
    return {
      availableSamples: this.available,
      ended: this.end !== undefined,
    };
  }

  private resolveWaiters(): void {
    const availability = this.availability();
    for (const pending of [...this.waiters]) {
      if (
        availability.availableSamples < pending.minimumSamples &&
        !availability.ended
      ) {
        continue;
      }
      this.waiters.delete(pending);
      if (pending.abortListener !== undefined) {
        pending.signal?.removeEventListener("abort", pending.abortListener);
      }
      pending.resolve(availability);
    }
  }

  private fail(error: unknown): void {
    this.failure = error;
    this.rejectWaiters(error);
  }

  private rejectWaiters(error: unknown): void {
    for (const pending of this.waiters) {
      if (pending.abortListener !== undefined) {
        pending.signal?.removeEventListener("abort", pending.abortListener);
      }
      pending.reject(error);
    }
    this.waiters.clear();
  }
}

function findSegment(
  segments: readonly StreamingPcmSegmentMessage[],
  sample: number,
): number {
  let low = 0;
  let high = segments.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const segment = segments[middle]!;
    if (sample < segment.startSample) {
      high = middle;
    } else if (sample >= segment.startSample + segment.sampleCount) {
      low = middle + 1;
    } else {
      return middle;
    }
  }
  throw new RangeError("PCM sample is outside the available segments");
}

function validateRead(
  startSample: number,
  target: Float32Array,
  targetOffset: number,
  sampleCount: number,
): void {
  if (!Number.isSafeInteger(startSample) || startSample < 0) {
    throw new RangeError("startSample must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(targetOffset) || targetOffset < 0) {
    throw new RangeError("targetOffset must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
  if (targetOffset + sampleCount > target.length) {
    throw new RangeError("target range exceeds the destination array");
  }
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}
