/// <reference types="@webgpu/types" />

export interface UniformParameterPoolPlan {
  readonly parameterBytes: number;
  readonly slotCapacity: number;
  readonly slotStride: number;
  readonly byteLength: number;
}

export function planUniformParameterPool(
  parameterBytes: number,
  slotCapacity: number,
  minUniformBufferOffsetAlignment: number,
): UniformParameterPoolPlan {
  for (const [name, value] of [
    ["parameterBytes", parameterBytes],
    ["slotCapacity", slotCapacity],
    [
      "minUniformBufferOffsetAlignment",
      minUniformBufferOffsetAlignment,
    ],
  ] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive integer`);
    }
  }
  if (parameterBytes % 4 !== 0) {
    throw new RangeError(
      "Uniform parameter bytes must be word aligned",
    );
  }
  const slotStride =
    Math.ceil(
      parameterBytes / minUniformBufferOffsetAlignment,
    ) * minUniformBufferOffsetAlignment;
  const byteLength = slotStride * slotCapacity;
  if (!Number.isSafeInteger(byteLength)) {
    throw new RangeError("Uniform parameter pool is too large");
  }
  return {
    parameterBytes,
    slotCapacity,
    slotStride,
    byteLength,
  };
}

export class UniformParameterPool {
  readonly byteLength: number;
  readonly slotStride: number;

  private readonly buffer: GPUBuffer;
  private readonly bindings = new Map<string, GPUBufferBinding>();
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly parameterBytes: number,
    readonly slotCapacity: number,
    label: string,
  ) {
    const plan = planUniformParameterPool(
      parameterBytes,
      slotCapacity,
      device.limits.minUniformBufferOffsetAlignment,
    );
    this.byteLength = plan.byteLength;
    this.slotStride = plan.slotStride;
    this.buffer = device.createBuffer({
      label,
      size: plan.byteLength,
      usage:
        GPUBufferUsage.UNIFORM |
        GPUBufferUsage.COPY_DST,
    });
  }

  get usedSlots(): number {
    return this.bindings.size;
  }

  bindingFor(
    label: string,
    data: ArrayBuffer | ArrayBufferView<ArrayBuffer>,
  ): GPUBufferBinding {
    if (this.destroyed) {
      throw new Error("Uniform parameter pool was destroyed");
    }
    const bytes = parameterBytes(data);
    if (bytes.byteLength !== this.parameterBytes) {
      throw new RangeError(
        `${label} parameters must be exactly ` +
          `${this.parameterBytes} bytes`,
      );
    }
    const key = byteKey(bytes);
    const existing = this.bindings.get(key);
    if (existing !== undefined) return existing;
    if (this.bindings.size >= this.slotCapacity) {
      throw new Error(
        `${label} exceeds uniform parameter pool capacity ` +
          `${this.slotCapacity}`,
      );
    }
    const offset = this.bindings.size * this.slotStride;
    this.device.queue.writeBuffer(
      this.buffer,
      offset,
      data,
    );
    const binding = {
      buffer: this.buffer,
      offset,
      size: this.parameterBytes,
    };
    this.bindings.set(key, binding);
    return binding;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindings.clear();
    this.buffer.destroy();
  }
}

function parameterBytes(
  data: ArrayBuffer | ArrayBufferView<ArrayBuffer>,
): Uint8Array<ArrayBuffer> {
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    );
  }
  return new Uint8Array(data);
}

function byteKey(bytes: Uint8Array<ArrayBuffer>): string {
  let key = `${bytes.byteLength}:`;
  for (const value of bytes) {
    key += value.toString(16).padStart(2, "0");
  }
  return key;
}
