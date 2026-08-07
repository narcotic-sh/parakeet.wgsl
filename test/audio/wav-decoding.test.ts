import { describe, expect, it } from "vitest";

import { requiresAudioDecoding } from "../../src/audio/wav-stream";

describe("requiresAudioDecoding", () => {
  it("uses the canonical WAV parser as the exact fast-path check", async () => {
    const canonical = pcm16Wav(new Int16Array([0]));
    const wrongRate = canonical.slice();
    const wrongRateView = new DataView(
      wrongRate.buffer,
      wrongRate.byteOffset,
      wrongRate.byteLength,
    );
    wrongRateView.setUint32(24, 48_000, true);
    wrongRateView.setUint32(28, 96_000, true);

    await expect(
      requiresAudioDecoding(new Blob([canonical])),
    ).resolves.toBe(false);
    await expect(
      requiresAudioDecoding(new Blob([wrongRate])),
    ).resolves.toBe(true);
    await expect(
      requiresAudioDecoding(new Blob(["not wav"])),
    ).resolves.toBe(true);
  });
});

function pcm16Wav(samples: Int16Array): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(44 + samples.byteLength);
  const view = new DataView(bytes.buffer);
  writeFourCc(bytes, 0, "RIFF");
  view.setUint32(4, bytes.byteLength - 8, true);
  writeFourCc(bytes, 8, "WAVE");
  writeFourCc(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16_000, true);
  view.setUint32(28, 32_000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeFourCc(bytes, 36, "data");
  view.setUint32(40, samples.byteLength, true);
  for (let index = 0; index < samples.length; index += 1) {
    view.setInt16(44 + index * 2, samples[index]!, true);
  }
  return bytes;
}

function writeFourCc(
  bytes: Uint8Array<ArrayBuffer>,
  offset: number,
  value: string,
): void {
  for (let index = 0; index < value.length; index += 1) {
    bytes[offset + index] = value.charCodeAt(index);
  }
}
