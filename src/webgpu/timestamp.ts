export interface GpuTimestampSample {
  readonly label: string;
  readonly startNanoseconds: bigint;
  readonly endNanoseconds: bigint;
  readonly durationNanoseconds: number;
  readonly durationMilliseconds: number;
}

/**
 * Return the elapsed GPU time from the beginning of `firstLabel` through the
 * end of `lastLabel`. A timestamp-query feature bit only says that queries can
 * be encoded; Metal/Dawn can still return unavailable (zero) samples. Treat
 * those, reversed samples, and non-finite Number conversions as unavailable so
 * callers can use a wall-clock fallback instead of reporting zero work.
 */
export function gpuTimestampSpanMilliseconds(
  samples: readonly GpuTimestampSample[],
  firstLabel: string,
  lastLabel: string,
): number | undefined {
  const first = samples.find((sample) => sample.label === firstLabel);
  const last = samples.find((sample) => sample.label === lastLabel);
  if (
    first === undefined ||
    last === undefined ||
    last.endNanoseconds <= first.startNanoseconds
  ) {
    return undefined;
  }
  const durationMilliseconds =
    Number(last.endNanoseconds - first.startNanoseconds) / 1_000_000;
  return Number.isFinite(durationMilliseconds) && durationMilliseconds > 0
    ? durationMilliseconds
    : undefined;
}

type RecorderState = "idle" | "recording" | "resolved" | "reading" | "destroyed";

/**
 * Optional pass-level GPU timing. Constructing this class never requires
 * timestamp-query; on devices without the enabled feature it becomes a no-op
 * and callers can use wall-clock timing.
 */
export class GpuTimestampRecorder {
  readonly supported: boolean;

  private readonly maxSamples: number;
  private readonly querySet?: GPUQuerySet;
  private readonly resolveBuffer?: GPUBuffer;
  private readonly readbackBuffer?: GPUBuffer;
  private readonly byteCapacity: number;
  private labels: string[] = [];
  private resolvedSampleCount = 0;
  private state: RecorderState = "idle";

  constructor(
    device: GPUDevice,
    maxSamples: number,
    /**
     * Hard kill switch for runtimes where merely encoding timestamp queries
     * is unsafe. When false, feature probing and all query/buffer allocation
     * are skipped even if the device exposes timestamp-query.
     */
    enabled = true,
  ) {
    if (!Number.isInteger(maxSamples) || maxSamples <= 0) {
      throw new RangeError("maxSamples must be a positive integer");
    }
    this.maxSamples = maxSamples;
    this.supported = enabled && device.features.has("timestamp-query");
    this.byteCapacity = maxSamples * 2 * BigUint64Array.BYTES_PER_ELEMENT;

    if (!this.supported) return;

    this.querySet = device.createQuerySet({
      label: "parakeet-wgsl-timestamps",
      type: "timestamp",
      count: maxSamples * 2,
    });
    this.resolveBuffer = device.createBuffer({
      label: "parakeet-wgsl-timestamp-resolve",
      size: this.byteCapacity,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readbackBuffer = device.createBuffer({
      label: "parakeet-wgsl-timestamp-readback",
      size: this.byteCapacity,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  reset(): void {
    this.assertNotDestroyed();
    if (this.state === "reading") {
      throw new Error("Cannot reset GPU timestamps while a readback is pending");
    }
    this.labels = [];
    this.resolvedSampleCount = 0;
    this.state = "recording";
  }

  /**
   * Return timestamp writes to spread into a compute-pass descriptor.
   *
   * @example
   * const writes = timestamps.allocate("encoder.layer.0");
   * const pass = encoder.beginComputePass(writes ? { timestampWrites: writes } : {});
   */
  allocate(label: string): GPUComputePassTimestampWrites | undefined {
    this.assertNotDestroyed();
    if (!this.supported) return undefined;
    if (this.state === "idle") this.reset();
    if (this.state !== "recording") {
      throw new Error("GPU timestamps must be allocated before resolve()");
    }
    if (this.labels.length >= this.maxSamples) {
      throw new RangeError(`GPU timestamp capacity ${this.maxSamples} exceeded`);
    }

    const sampleIndex = this.labels.length;
    this.labels.push(label);
    return {
      querySet: this.querySet!,
      beginningOfPassWriteIndex: sampleIndex * 2,
      endOfPassWriteIndex: sampleIndex * 2 + 1,
    };
  }

  /**
   * Encode query resolution before submitting the command buffer.
   */
  resolve(encoder: GPUCommandEncoder): void {
    this.assertNotDestroyed();
    if (!this.supported) return;
    if (this.state !== "recording") {
      throw new Error("reset() and allocate() must precede GPU timestamp resolution");
    }

    this.resolvedSampleCount = this.labels.length;
    if (this.resolvedSampleCount > 0) {
      const queryCount = this.resolvedSampleCount * 2;
      const byteLength = queryCount * BigUint64Array.BYTES_PER_ELEMENT;
      encoder.resolveQuerySet(this.querySet!, 0, queryCount, this.resolveBuffer!, 0);
      encoder.copyBufferToBuffer(
        this.resolveBuffer!,
        0,
        this.readbackBuffer!,
        0,
        byteLength,
      );
    }
    this.state = "resolved";
  }

  /**
   * Read timestamps after the command buffer containing resolve() is submitted.
   * WebGPU timestamp values are nanoseconds.
   */
  async read(): Promise<GpuTimestampSample[]> {
    this.assertNotDestroyed();
    if (!this.supported) return [];
    if (this.state !== "resolved") {
      throw new Error("resolve() must be submitted before reading GPU timestamps");
    }
    if (this.resolvedSampleCount === 0) {
      this.state = "idle";
      return [];
    }

    this.state = "reading";
    const byteLength =
      this.resolvedSampleCount * 2 * BigUint64Array.BYTES_PER_ELEMENT;
    try {
      await this.readbackBuffer!.mapAsync(GPUMapMode.READ, 0, byteLength);
      const mapped = this.readbackBuffer!.getMappedRange(0, byteLength);
      const timestamps = new BigUint64Array(mapped.slice(0));
      const samples = this.labels.slice(0, this.resolvedSampleCount).map(
        (label, index): GpuTimestampSample => {
          const startNanoseconds = timestamps[index * 2]!;
          const endNanoseconds = timestamps[index * 2 + 1]!;
          const durationNanoseconds = Number(endNanoseconds - startNanoseconds);
          return {
            label,
            startNanoseconds,
            endNanoseconds,
            durationNanoseconds,
            durationMilliseconds: durationNanoseconds / 1_000_000,
          };
        },
      );
      return samples;
    } finally {
      if (this.readbackBuffer!.mapState === "mapped") {
        this.readbackBuffer!.unmap();
      }
      this.state = "idle";
    }
  }

  destroy(): void {
    if (this.state === "destroyed") return;
    if (this.readbackBuffer?.mapState === "mapped") {
      this.readbackBuffer.unmap();
    }
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.readbackBuffer?.destroy();
    this.labels = [];
    this.resolvedSampleCount = 0;
    this.state = "destroyed";
  }

  private assertNotDestroyed(): void {
    if (this.state === "destroyed") {
      throw new Error("GPU timestamp recorder has been destroyed");
    }
  }
}
