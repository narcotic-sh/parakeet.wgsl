import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";

import { describe, expect, it } from "vitest";

const EXPECTED_FFMPEG_ARTIFACTS = [
  {
    name: "parakeet-wgsl-ffmpeg-core.js",
    byteLength: 82_159,
    sha256: "ca0d9083e37ae0b016f6763db2e541b718077d70c5c99f8b09817741c25578ee",
  },
  {
    name: "parakeet-wgsl-ffmpeg-core.wasm",
    byteLength: 1_599_767,
    sha256: "8703a1a66936cf39a10ef94490e9e594dc602e4d094fa7b0325d59fee0488fa5",
  },
] as const;

const PRIVATE_BUILD_PATH_PREFIXES = [
  "/Users/",
  "/private/tmp/",
  "/private/var/",
  "/tmp/",
  "/opt/homebrew/",
] as const;

describe("release FFmpeg artifacts", () => {
  for (const artifact of EXPECTED_FFMPEG_ARTIFACTS) {
    it(`pins ${artifact.name}`, async () => {
      const url = new URL(`../src/cpp/wasm/${artifact.name}`, import.meta.url);
      const [metadata, bytes] = await Promise.all([stat(url), readFile(url)]);
      expect(metadata.size).toBe(artifact.byteLength);
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(
        artifact.sha256,
      );
    });
  }

  it("contains no private build-machine paths", async () => {
    for (const artifact of EXPECTED_FFMPEG_ARTIFACTS) {
      const bytes = await readFile(
        new URL(`../src/cpp/wasm/${artifact.name}`, import.meta.url),
      );
      for (const prefix of PRIVATE_BUILD_PATH_PREFIXES) {
        expect(
          bytes.includes(Buffer.from(prefix)),
          `${artifact.name}: ${prefix}`,
        ).toBe(false);
      }
    }
  });

  it("keeps the JavaScript factory product-specific and ESM", async () => {
    const source = await readFile(
      new URL(
        "../src/cpp/wasm/parakeet-wgsl-ffmpeg-core.js",
        import.meta.url,
      ),
      "utf8",
    );
    expect(source).toContain("createParakeetWgslFFmpegCore");
    expect(source).toContain("createIncrementalAudioDecoder");
    expect(source).toContain("estimatedSampleCount");
    expect(source).not.toContain("createSenkoWebFFmpegCore");
    expect(source).toMatch(
      /export\s+default\s+createParakeetWgslFFmpegCore\s*;/,
    );
  });
});
