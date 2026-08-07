import { describe, expect, it } from "vitest";

import {
  ModelCacheSession,
  type ModelCacheAsset,
} from "../src/model-cache";

const DIGEST_A = "a".repeat(64);
const DIGEST_B = "b".repeat(64);

describe("ModelCacheSession inventory invariants", () => {
  it("derives its corruption-recovery bound from the selected inventory", () => {
    const manifest = asset("https://models.example/fp16/manifest.json", 19);
    const session = new ModelCacheSession(manifest);

    expect(session.maximumCorruptionRetries).toBe(1);

    const packageAssets = Array.from({ length: 32 }, (_, index) =>
      asset(`https://models.example/fp16/shard-${index}.bin`, index + 1),
    );
    session.registerAssets(packageAssets);

    expect(session.maximumCorruptionRetries).toBe(33);

    session.registerAssets([manifest, packageAssets[0]!]);
    expect(session.maximumCorruptionRetries).toBe(33);
  });

  it("refuses conflicting integrity metadata for a registered URL", () => {
    const manifest = asset("https://models.example/fp32/manifest.json", 23);
    const session = new ModelCacheSession(manifest);

    expect(() =>
      session.registerAssets([
        {
          ...manifest,
          sha256: DIGEST_B,
        },
      ]),
    ).toThrow(`Conflicting integrity records for model asset ${manifest.url}`);
  });
});

function asset(url: string, byteLength: number): ModelCacheAsset {
  return { url, byteLength, sha256: DIGEST_A };
}
