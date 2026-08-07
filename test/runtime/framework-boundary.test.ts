import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";

import { describe, expect, it } from "vitest";

const ROOT = process.cwd();
const SOURCE_ROOT = join(ROOT, "src");

describe("framework-free browser boundary", () => {
  it("ships no browser runtime dependencies", () => {
    const packageJson = JSON.parse(
      readFileSync(join(ROOT, "package.json"), "utf8"),
    ) as { readonly dependencies?: Readonly<Record<string, string>> };

    expect(packageJson.dependencies ?? {}).toEqual({});
  });

  it("keeps every production TypeScript import inside this implementation", () => {
    const externalImports: string[] = [];
    const importPattern =
      /(?:\bfrom\s*|\bimport\s*\()\s*["']([^"']+)["']/g;

    for (const path of typescriptFiles(SOURCE_ROOT)) {
      const source = readFileSync(path, "utf8");
      for (const match of source.matchAll(importPattern)) {
        const specifier = match[1]!;
        if (!specifier.startsWith(".")) {
          externalImports.push(`${relative(ROOT, path)}: ${specifier}`);
        }
      }
    }

    expect(externalImports).toEqual([]);
  });
});

function* typescriptFiles(directory: string): Generator<string> {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      yield* typescriptFiles(path);
    } else if (entry.isFile() && entry.name.endsWith(".ts")) {
      yield path;
    }
  }
}
