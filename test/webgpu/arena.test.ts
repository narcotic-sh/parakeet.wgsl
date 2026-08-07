import { beforeAll, describe, expect, it, vi } from "vitest";

import {
  GpuActivationArena,
  type ActivationArenaBufferPlan,
} from "../../src/webgpu/arena";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 },
  });
});

describe("partitioned activation arena", () => {
  it("resolves slices in a single-buffer arena", () => {
    const buffer = fakeBuffer();
    const device = fakeDevice([buffer]);
    const arena = new GpuActivationArena(device, 1_024, "single");
    const slice = arena.slice("values", 256, 512);

    expect(arena.byteLength).toBe(1_024);
    expect(arena.bufferCount).toBe(1);
    expect(arena.largestBufferByteLength).toBe(1_024);
    expect(arena.bufferFor(slice)).toBe(buffer);
    expect(slice).toEqual({
      label: "values",
      byteOffset: 256,
      byteLength: 512,
    });
    expect(arena.binding(slice)).toEqual({
      buffer,
      offset: 256,
      size: 512,
    });
  });

  it("routes explicit slices to bounded physical buffers", () => {
    const buffers = [fakeBuffer(), fakeBuffer(), fakeBuffer()];
    const plans: readonly ActivationArenaBufferPlan[] = [
      { label: "slot-zero", byteLength: 1_024 },
      { label: "slot-one", byteLength: 2_048 },
      { label: "temporary", byteLength: 4_096 },
    ];
    const device = fakeDevice(buffers);
    const arena = new GpuActivationArena(device, plans);
    const first = arena.slice("first", 256, 512);
    const third = arena.slice("third", 1_024, 2_048, 2);

    expect(arena.byteLength).toBe(7_168);
    expect(arena.bufferCount).toBe(3);
    expect(arena.largestBufferByteLength).toBe(4_096);
    expect(arena.bufferFor(first)).toBe(buffers[0]);
    expect(arena.bufferFor(third)).toBe(buffers[2]);
    expect(arena.bufferByteLengthFor(third)).toBe(4_096);
    expect(arena.binding(third)).toEqual({
      buffer: buffers[2],
      offset: 1_024,
      size: 2_048,
    });
    expect(() =>
      arena.slice("overflow", 3_840, 512, 2),
    ).toThrow(/Invalid activation slice/);

    arena.destroy();
    arena.destroy();
    for (const buffer of buffers) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});

function fakeBuffer(): GPUBuffer & { readonly destroy: ReturnType<typeof vi.fn> } {
  return {
    destroy: vi.fn(),
  } as unknown as GPUBuffer & {
    readonly destroy: ReturnType<typeof vi.fn>;
  };
}

function fakeDevice(buffers: readonly GPUBuffer[]): GPUDevice {
  let index = 0;
  return {
    limits: {
      maxBufferSize: 8_192,
      maxStorageBufferBindingSize: 8_192,
    },
    createBuffer: vi.fn(() => buffers[index++]!),
    queue: { writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}
