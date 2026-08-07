import { afterEach, describe, expect, it } from "vitest";

import { SegmentedPcm16Reader } from "../../src/audio/segmented-pcm-reader";

const readers: SegmentedPcm16Reader[] = [];
const ports: MessagePort[] = [];

afterEach(() => {
  for (const reader of readers.splice(0)) reader.dispose();
  for (const port of ports.splice(0)) port.close();
});

describe("SegmentedPcm16Reader", () => {
  it("reads one requested range exactly across immutable segment boundaries", async () => {
    const { producer, reader } = createReader();
    producer.postMessage(segment(0, 0, [0, 16_384, -32_768]));
    producer.postMessage(segment(1, 3, [8_192, -16_384]));
    producer.postMessage(end(5));
    await reader.waitForEnd();

    const target = new Float32Array(6).fill(99);
    expect(await reader.readSamplesInto(1, target, 1, 4)).toBe(4);
    expect([...target]).toEqual([99, 0.5, -1, 0.25, -0.5, 99]);
    expect(reader.sampleCount).toBe(5);
  });

  it("rejects noncontiguous immutable segments", async () => {
    const { producer, reader } = createReader();
    const waiting = reader.waitForSamples(5);
    producer.postMessage(segment(0, 1, [1, 2]));

    await expect(waiting).rejects.toThrow(/non-contiguous/);
  });

  it("rejects EOF whose length differs from received segments", async () => {
    const { producer, reader } = createReader();
    const waiting = reader.waitForEnd();
    producer.postMessage(segment(0, 0, [1, 2]));
    producer.postMessage(end(3));

    await expect(waiting).rejects.toThrow(/EOF/);
  });

  it("rejects pending availability waits on producer failure", async () => {
    const { producer, reader } = createReader();
    const waiting = reader.waitForSamples(10);
    producer.postMessage({ type: "pcm-failed", message: "decode exploded" });

    await expect(waiting).rejects.toThrow("decode exploded");
  });

  it("removes an aborted waiter without poisoning later reads", async () => {
    const { producer, reader } = createReader();
    const controller = new AbortController();
    const waiting = reader.waitForSamples(10, controller.signal);
    controller.abort();
    await expect(waiting).rejects.toMatchObject({ name: "AbortError" });

    producer.postMessage(segment(0, 0, [1, 2, 3]));
    producer.postMessage(end(3));
    await expect(reader.waitForEnd()).resolves.toEqual({
      availableSamples: 3,
      ended: true,
    });
  });

  it("rejects malformed messages and every pending waiter", async () => {
    const { producer, reader } = createReader();
    const first = reader.waitForSamples(1);
    const second = reader.waitForEnd();
    producer.postMessage({ type: "pcm-segment", segmentIndex: "zero" });

    await expect(first).rejects.toThrow(/invalid PCM message/);
    await expect(second).rejects.toThrow(/invalid PCM message/);
  });

  it("observes EOF posted immediately before the producer port closes", async () => {
    const { producer, reader } = createReader();
    const waiting = reader.waitForEnd();
    producer.postMessage(segment(0, 0, [1, 2, 3]));
    producer.postMessage(end(3));
    producer.close();

    await expect(waiting).resolves.toEqual({
      availableSamples: 3,
      ended: true,
    });
  });
});

function createReader(): {
  readonly producer: MessagePort;
  readonly reader: SegmentedPcm16Reader;
} {
  const channel = new MessageChannel();
  const reader = new SegmentedPcm16Reader(channel.port1);
  readers.push(reader);
  ports.push(channel.port2);
  return { producer: channel.port2, reader };
}

function segment(
  segmentIndex: number,
  startSample: number,
  samples: readonly number[],
): Record<string, unknown> {
  const pcm = new Int16Array(samples);
  return {
    type: "pcm-segment",
    segmentIndex,
    startSample,
    sampleCount: pcm.length,
    file: new File([pcm], `segment-${segmentIndex}.s16le`),
  };
}

function end(totalSamples: number): Record<string, unknown> {
  return {
    type: "pcm-end",
    totalSamples,
    outputByteLength: totalSamples * 2,
    decoderWallMs: 10,
    decoderActiveMs: 7,
    decoderStartedAtMs: 100,
    decoderEndedAtMs: 110,
  };
}
