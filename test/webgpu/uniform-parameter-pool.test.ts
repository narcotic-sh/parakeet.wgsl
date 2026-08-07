import { beforeAll, describe, expect, it, vi } from "vitest";

import {
  planUniformParameterPool,
  UniformParameterPool,
} from "../../src/webgpu/uniform-parameter-pool";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { COPY_DST: 1, UNIFORM: 2 },
  });
});

describe("persistent uniform parameter pool", () => {
  it("plans exact aligned bounded storage", () => {
    expect(planUniformParameterPool(16, 40, 256)).toEqual({
      parameterBytes: 16,
      slotCapacity: 40,
      slotStride: 256,
      byteLength: 10_240,
    });
    expect(planUniformParameterPool(48, 3, 512)).toEqual({
      parameterBytes: 48,
      slotCapacity: 3,
      slotStride: 512,
      byteLength: 1_536,
    });
    expect(() =>
      planUniformParameterPool(15, 1, 256),
    ).toThrow(/word aligned/);
    expect(() =>
      planUniformParameterPool(16, 0, 256),
    ).toThrow(/positive integer/);
  });

  it("deduplicates exact bytes and binds unique values at aligned offsets", () => {
    const destroy = vi.fn();
    const pooledBuffer = { destroy } as unknown as GPUBuffer;
    const writeBuffer = vi.fn();
    const createBuffer = vi.fn(() => pooledBuffer);
    const device = {
      limits: { minUniformBufferOffsetAlignment: 256 },
      createBuffer,
      queue: { writeBuffer },
    } as unknown as GPUDevice;
    const pool = new UniformParameterPool(
      device,
      16,
      2,
      "test-parameters",
    );
    const firstValues = new Uint32Array([1, 2, 3, 4]);
    const first = pool.bindingFor("first", firstValues);
    const duplicate = pool.bindingFor(
      "duplicate",
      new Uint32Array([1, 2, 3, 4]),
    );
    const secondValues = new Uint32Array([1, 2, 3, 5]);
    const second = pool.bindingFor("second", secondValues);

    expect(createBuffer).toHaveBeenCalledWith({
      label: "test-parameters",
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    expect(first).toEqual({
      buffer: pooledBuffer,
      offset: 0,
      size: 16,
    });
    expect(duplicate).toBe(first);
    expect(second).toEqual({
      buffer: pooledBuffer,
      offset: 256,
      size: 16,
    });
    expect(pool.usedSlots).toBe(2);
    expect(writeBuffer).toHaveBeenCalledTimes(2);
    expect(writeBuffer.mock.calls[0]).toEqual([
      pooledBuffer,
      0,
      firstValues,
    ]);
    expect(writeBuffer.mock.calls[1]).toEqual([
      pooledBuffer,
      256,
      secondValues,
    ]);
    expect(() =>
      pool.bindingFor(
        "overflow",
        new Uint32Array([9, 8, 7, 6]),
      ),
    ).toThrow(/capacity 2/);

    pool.destroy();
    pool.destroy();
    expect(destroy).toHaveBeenCalledOnce();
    expect(() =>
      pool.bindingFor("destroyed", firstValues),
    ).toThrow(/destroyed/);
  });

  it("requires an exact parameter payload length", () => {
    const pool = new UniformParameterPool(
      {
        limits: { minUniformBufferOffsetAlignment: 256 },
        createBuffer: () => ({ destroy: vi.fn() }),
        queue: { writeBuffer: vi.fn() },
      } as unknown as GPUDevice,
      32,
      1,
      "test-exact-size",
    );
    expect(() =>
      pool.bindingFor("short", new Uint32Array(4)),
    ).toThrow(/exactly 32 bytes/);
    pool.destroy();
  });
});
