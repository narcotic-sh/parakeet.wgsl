/// <reference types="@webgpu/types" />

const STORAGE_ALIGNMENT = 256;

export interface ArenaSlice {
  readonly label: string;
  /**
   * Physical arena-buffer index. The established FP16 arena omits this field
   * and therefore remains buffer zero; the FP32 plan uses explicit indices to
   * keep every allocation below stock WebGPU binding limits.
   */
  readonly bufferIndex?: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface ActivationArenaBufferPlan {
  readonly label: string;
  readonly byteLength: number;
}

/**
 * A fixed set of GPU activation allocations. The FP16 graph continues to use
 * exactly one buffer. The FP32 graph partitions the same lifetime-aliased
 * plan across a small fixed set of buffers so no individual allocation needs
 * an unusually large WebGPU storage-buffer limit.
 */
export class GpuActivationArena {
  readonly buffers: readonly GPUBuffer[];
  readonly bufferByteLengths: readonly number[];
  readonly byteLength: number;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    byteLengthOrBuffers: number | readonly ActivationArenaBufferPlan[],
    label = "parakeet-activation-arena",
  ) {
    const plans =
      typeof byteLengthOrBuffers === "number"
        ? [{ label, byteLength: byteLengthOrBuffers }]
        : [...byteLengthOrBuffers];
    if (plans.length === 0) {
      throw new RangeError("Activation arena must contain at least one buffer");
    }
    for (const plan of plans) {
      requireArenaBufferLength(device, plan.byteLength);
    }
    const buffers: GPUBuffer[] = [];
    try {
      for (const plan of plans) {
        buffers.push(
          device.createBuffer({
            label: plan.label,
            size: plan.byteLength,
            usage:
              GPUBufferUsage.STORAGE |
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST,
          }),
        );
      }
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      throw error;
    }
    this.buffers = Object.freeze(buffers);
    this.bufferByteLengths = Object.freeze(
      plans.map((plan) => plan.byteLength),
    );
    this.byteLength = this.bufferByteLengths.reduce(
      (total, value) => total + value,
      0,
    );
  }

  get bufferCount(): number {
    return this.buffers.length;
  }

  get largestBufferByteLength(): number {
    return Math.max(...this.bufferByteLengths);
  }

  slice(
    label: string,
    byteOffset: number,
    byteLength: number,
    bufferIndex = 0,
  ): ArenaSlice {
    this.assertAlive();
    const bufferByteLength = this.bufferByteLengths[bufferIndex];
    if (
      bufferByteLength === undefined ||
      !Number.isSafeInteger(byteOffset) ||
      !Number.isSafeInteger(byteLength) ||
      byteOffset < 0 ||
      byteLength <= 0 ||
      byteOffset % STORAGE_ALIGNMENT !== 0 ||
      byteLength % 4 !== 0 ||
      byteOffset + byteLength > bufferByteLength
    ) {
      throw new RangeError(`Invalid activation slice ${label}`);
    }
    return bufferIndex === 0
      ? { label, byteOffset, byteLength }
      : { label, bufferIndex, byteOffset, byteLength };
  }

  binding(slice: ArenaSlice): GPUBufferBinding {
    const buffer = this.bufferFor(slice);
    return {
      buffer,
      offset: slice.byteOffset,
      size: slice.byteLength,
    };
  }

  bufferFor(slice: ArenaSlice): GPUBuffer {
    this.assertAlive();
    const bufferIndex = slice.bufferIndex ?? 0;
    const buffer = this.buffers[bufferIndex];
    const bufferByteLength = this.bufferByteLengths[bufferIndex];
    if (
      buffer === undefined ||
      bufferByteLength === undefined ||
      slice.byteOffset < 0 ||
      slice.byteLength <= 0 ||
      slice.byteOffset + slice.byteLength > bufferByteLength
    ) {
      throw new RangeError(`Invalid activation slice ${slice.label}`);
    }
    return buffer;
  }

  bufferByteLengthFor(slice: ArenaSlice): number {
    this.bufferFor(slice);
    return this.bufferByteLengths[slice.bufferIndex ?? 0]!;
  }

  upload(
    device: GPUDevice,
    destination: ArenaSlice,
    data: ArrayBuffer | ArrayBufferView<ArrayBuffer>,
  ): void {
    const buffer = this.bufferFor(destination);
    if (data.byteLength > destination.byteLength || data.byteLength % 4 !== 0) {
      throw new RangeError(`Upload does not fit ${destination.label}`);
    }
    device.queue.writeBuffer(buffer, destination.byteOffset, data);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.buffers) buffer.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Activation arena was destroyed");
  }
}

function requireArenaBufferLength(
  device: GPUDevice,
  byteLength: number,
): void {
  if (
    !Number.isSafeInteger(byteLength) ||
    byteLength <= 0 ||
    byteLength % STORAGE_ALIGNMENT !== 0
  ) {
    throw new RangeError(
      "Activation arena buffers must be positive 256-byte multiples",
    );
  }
  if (
    byteLength > device.limits.maxBufferSize ||
    byteLength > device.limits.maxStorageBufferBindingSize
  ) {
    throw new Error("Activation arena exceeds the WebGPU device limits");
  }
}

export class LinearArenaPlanner {
  private cursor = 0;

  allocate(label: string, byteLength: number): ArenaSlice {
    const alignedLength = align(byteLength, 4);
    this.cursor = align(this.cursor, STORAGE_ALIGNMENT);
    const allocation = {
      label,
      byteOffset: this.cursor,
      byteLength: alignedLength,
    };
    this.cursor += alignedLength;
    return allocation;
  }

  get byteLength(): number {
    return align(this.cursor, STORAGE_ALIGNMENT);
  }
}

export function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
